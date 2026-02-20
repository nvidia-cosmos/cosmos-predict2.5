# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import os
import pickle
import tempfile
import time
from typing import Any, Callable, Optional

import cv2
import einops
import mediapy as media
import numpy as np
import torch

from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_predict2._src.transfer2.auxiliary.sam2.sam2_model import VideoSegmentationModel

_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]

NUM_MAX_FRAMES = 5000

DUMMY_PROMPT = "The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution."

DEFAULT_NEG_T5_PROMPT_EMBEDDING_PATH = "s3://bucket/projects/edify_video/v4/video_neg_prompt_embeddings_v0.pt"


def download_from_s3_with_cache(
    s3_path: str,
    cache_fp: Optional[str] = None,
    cache_dir: Optional[str] = None,
    rank_sync: bool = True,
    backend_args: Optional[dict] = None,
    backend_key: Optional[str] = None,
) -> str:
    """download data from S3 with optional caching.

    This function first attempts to load the data from a local cache file. If
    the cache file doesn't exist, it downloads the data from S3 to the cache
    location. Caching is performed in a rank-aware manner
    using `distributed.barrier()` to ensure only one download occurs across
    distributed workers (if `rank_sync` is True).

    Args:
        s3_path (str): The S3 path of the data to load.
        cache_fp (str, optional): The path to the local cache file. If None,
            a filename will be generated based on `s3_path` within `cache_dir`.
        cache_dir (str, optional): The directory to store the cache file. If
            None, the environment variable `IMAGINAIRE_CACHE_DIR` (defaulting
            to "/tmp") will be used.
        rank_sync (bool, optional): Whether to synchronize download across
            distributed workers using `distributed.barrier()`. Defaults to True.
        backend_args (dict, optional): The backend arguments passed to easy_io to construct the backend.
        backend_key (str, optional): The backend key passed to easy_io to registry the backend or retrieve the backend if it is already registered.

    Returns:
        cache_fp (str): The path to the local cache file.

    Raises:
        FileNotFoundError: If the data cannot be found in S3 or the cache.
    """
    cache_dir = os.environ.get("TORCH_HOME") if cache_dir is None else cache_dir
    cache_dir = (
        os.environ.get("IMAGINAIRE_CACHE_DIR", os.path.expanduser("~/.cache/imaginaire"))
        if cache_dir is None
        else cache_dir
    )
    cache_dir = os.path.expanduser(cache_dir)
    if cache_fp is None:
        cache_fp = os.path.join(cache_dir, s3_path.replace("s3://", ""))
    if not cache_fp.startswith("/"):
        cache_fp = os.path.join(cache_dir, cache_fp)

    if distributed.get_rank() == 0:
        if os.path.exists(cache_fp):
            # check the size of cache_fp
            if os.path.getsize(cache_fp) < 1:
                os.remove(cache_fp)
                log.warning(f"Removed empty cache file {cache_fp}.")

    if rank_sync:
        if not os.path.exists(cache_fp):
            log.critical(f"Local cache {cache_fp} Not exist! Downloading {s3_path} to {cache_fp}.")
            log.info(f"backend_args: {backend_args}")
            log.info(f"backend_key: {backend_key}")

            easy_io.copyfile_to_local(
                s3_path, cache_fp, dst_type="file", backend_args=backend_args, backend_key=backend_key
            )
            log.info(f"Downloaded {s3_path} to {cache_fp}.")
        else:
            log.info(f"Local cache {cache_fp} already exist! {s3_path} -> {cache_fp}.")

        distributed.barrier()
    else:
        if not os.path.exists(cache_fp):
            easy_io.copyfile_to_local(
                s3_path, cache_fp, dst_type="file", backend_args=backend_args, backend_key=backend_key
            )

            log.info(f"Downloaded {s3_path} to {cache_fp}.")
    return cache_fp


def load_from_s3_with_cache(
    s3_path: str,
    cache_fp: Optional[str] = None,
    cache_dir: Optional[str] = None,
    rank_sync: bool = True,
    backend_args: Optional[dict] = None,
    backend_key: Optional[str] = None,
    easy_io_kwargs: Optional[dict] = None,
) -> Any:
    """Loads data from S3 with optional caching.

    This function first attempts to load the data from a local cache file. If
    the cache file doesn't exist, it downloads the data from S3 to the cache
    location and then loads it. Caching is performed in a rank-aware manner
    using `distributed.barrier()` to ensure only one download occurs across
    distributed workers (if `rank_sync` is True).

    Args:
        s3_path (str): The S3 path of the data to load.
        cache_fp (str, optional): The path to the local cache file. If None,
            a filename will be generated based on `s3_path` within `cache_dir`.
        cache_dir (str, optional): The directory to store the cache file. If
            None, the environment variable `IMAGINAIRE_CACHE_DIR` (defaulting
            to "/tmp") will be used.
        rank_sync (bool, optional): Whether to synchronize download across
            distributed workers using `distributed.barrier()`. Defaults to True.
        backend_args (dict, optional): The backend arguments passed to easy_io to construct the backend.
        backend_key (str, optional): The backend key passed to easy_io to registry the backend or retrieve the backend if it is already registered.

    Returns:
        Any: The loaded data from the S3 path or cache file.

    Raises:
        FileNotFoundError: If the data cannot be found in S3 or the cache.
    """
    cache_fp = download_from_s3_with_cache(s3_path, cache_fp, cache_dir, rank_sync, backend_args, backend_key)

    if easy_io_kwargs is None:
        easy_io_kwargs = {}
    return easy_io.load(cache_fp, **easy_io_kwargs)


def resize_video(video_np: np.ndarray, h: int, w: int, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize video frames to the specified height and width."""
    video_np = video_np[0].transpose((1, 2, 3, 0))  # Convert to T x H x W x C
    t = video_np.shape[0]
    resized_video = np.zeros((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        resized_video[i] = cv2.resize(video_np[i], (w, h), interpolation=interpolation)
    return resized_video.transpose((3, 0, 1, 2))[None]  # Convert back to B x C x T x H x W


def detect_aspect_ratio(img_size: tuple[int, int]) -> str:
    """Function for detecting the closest aspect ratio."""

    _aspect_ratios = np.array([(16 / 9), (4 / 3), 1, (3 / 4), (9 / 16)])
    _aspect_ratio_keys = ["16,9", "4,3", "1,1", "3,4", "9,16"]
    w, h = img_size
    current_ratio = w / h
    closest_aspect_ratio = np.argmin((_aspect_ratios - current_ratio) ** 2)
    return _aspect_ratio_keys[closest_aspect_ratio]


def read_video_or_image_into_frames_BCTHW(
    input_path: str,
    input_path_format: str = None,
    H: int = None,
    W: int = None,
    s3_credential_path: str = "credentials/pbss_dir.secret",
    normalize: bool = True,
    max_frames: int = -1,
    also_return_fps: bool = False,
) -> torch.Tensor:
    """Read video or image from file and convert it to tensor. The frames will be normalized to [-1, 1].
    Args:
        input_path (str): path to the input video or image, end with .mp4 or .png or .jpg
        H (int): height to resize the video
        W (int): width to resize the video
    Returns:
        torch.Tensor: video tensor in shape (1, C, T, H, W), range [-1, 1]
    """
    log.info(f"Reading video from {input_path}")
    backend_args = (
        {"backend": "s3", "s3_credential_path": s3_credential_path, "path_mapping": None}
        if input_path.startswith("s3://")
        else None
    )
    loaded_data = easy_io.load(input_path, file_format=input_path_format, backend_args=backend_args)
    if input_path.endswith(".png") or input_path.endswith(".jpg") or input_path.endswith(".jpeg"):
        frames = np.array(loaded_data)  # HWC, [0,255]
        if frames.shape[-1] > 3:  # RGBA, set the transparent to white
            # Separate the RGB and Alpha channels
            rgb_channels = frames[..., :3]
            alpha_channel = frames[..., 3] / 255.0  # Normalize alpha channel to [0, 1]

            # Create a white background
            white_bg = np.ones_like(rgb_channels) * 255  # White background in RGB

            # Blend the RGB channels with the white background based on the alpha channel
            frames = (rgb_channels * alpha_channel[..., None] + white_bg * (1 - alpha_channel[..., None])).astype(
                np.uint8
            )
        frames = [frames]
        fps = 1
    else:
        frames, meta_data = loaded_data
        fps = int(meta_data.get("fps"))
        if max_frames != -1:
            frames = frames[:max_frames]
    if H is not None and W is not None:
        frames = media.resize_video(frames, (H, W))  # resize using Lanczos filter, leads to a better quality.
    input_tensor = np.stack(frames, axis=0)
    input_tensor = einops.rearrange(input_tensor, "t h w c -> t c h w")
    if normalize:
        input_tensor = input_tensor / 128.0 - 1.0
        input_tensor = torch.from_numpy(input_tensor).bfloat16()  # TCHW
        log.info(f"Raw data shape: {input_tensor.shape}")
    input_tensor = einops.rearrange(input_tensor, "(b t) c h w -> b c t h w", b=1)
    if normalize:
        input_tensor = input_tensor.to("cuda")
    log.info(f"Loaded input tensor with shape {input_tensor.shape} value {input_tensor.min()}, {input_tensor.max()}")
    if also_return_fps:
        return input_tensor, fps
    return input_tensor


def _resize_to_target_resolution(
    video_tensor: torch.Tensor | np.ndarray,
    resolution: str = "720",
    interpolation: int = cv2.INTER_AREA,
) -> torch.Tensor:
    """
    Resize video tensor to target resolution based on aspect ratio.

    Args:
        video_tensor: Input video (C, T, H, W) as torch.Tensor or numpy array
        resolution: Target resolution (e.g., "720")
        interpolation: OpenCV interpolation method

    Returns:
        Resized video tensor (C, T, H, W)
    """
    if isinstance(video_tensor, torch.Tensor):
        video_np = video_tensor.numpy()
        was_torch = True
    else:
        video_np = video_tensor
        was_torch = False

    aspect_ratio = detect_aspect_ratio((video_np.shape[-1], video_np.shape[-2]))
    w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]

    resized = resize_video(video_np[None], h, w, interpolation=interpolation)[0]

    if was_torch:
        return torch.from_numpy(resized)
    return resized


def read_and_resize_input(
    input_video_path: str,
    num_total_frames: int = NUM_MAX_FRAMES,
    interpolation: int = cv2.INTER_AREA,
    resolution: str = "720",
    s3_credential_path: str | None = None,
) -> tuple[torch.Tensor, int, str, tuple[int, int]]:
    input_video, fps = read_video_or_image_into_frames_BCTHW(
        input_video_path,
        normalize=False,  # s.t. output range is [0, 255]
        max_frames=num_total_frames,
        also_return_fps=True,
        s3_credential_path=s3_credential_path,
    )  # BCTHW
    original_hw = (input_video.shape[-2], input_video.shape[-1])
    aspect_ratio = detect_aspect_ratio((input_video.shape[-1], input_video.shape[-2]))
    w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
    input_video = resize_video(input_video, h, w, interpolation=interpolation)  # BCTHW, range [0, 255]
    input_video = torch.from_numpy(input_video[0])  # CTHW, range [0, 255]
    return input_video, fps, aspect_ratio, original_hw


def read_and_process_image_context(
    img_context_path: str | None,
    resolution: tuple[int, int],
    resize: bool = True,
    s3_credential_path: str | None = None,
    context_frame_idx: int | None = None,
) -> torch.Tensor | None:
    """
    Reads an image context file, processes it for model input as image conditioning.

    The image is loaded, converted to a tensor, resized to match the target resolution,
    and normalized to the [-1, 1] range expected by the model.

    Args:
        img_context_path (str): Path to the input image context file.
        resolution (tuple[int, int]): Target resolution (W, H) for resizing.
        resize (bool, optional): Whether to resize the image to the target resolution. Defaults to True.
        s3_credential_path (str): Path to the S3 credential file.

    Returns:
        torch.Tensor: Processed image context tensor of shape (1, C, H, W) normalized to [-1, 1].

    Raises:
        ValueError: If the image extension is not one of the supported types or file doesn't exist.
    """
    if img_context_path is None:
        log.info("No image context provided.")
        return None
    else:
        log.info(f"Processing image context from: {img_context_path}")

    ext = os.path.splitext(img_context_path)[1].lower()
    if ext not in _IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS:
        raise ValueError(f"Invalid image context extension: {ext}. Supported: {_IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS}")

    t_idx = context_frame_idx if context_frame_idx is not None else 0
    log.info(f"Using context frame index: {t_idx}")
    img = read_video_or_image_into_frames_BCTHW(
        img_context_path,
        H=resolution[1],
        W=resolution[0],
        normalize=True,  # s.t. output range is [-1, 1]
        s3_credential_path=s3_credential_path,
    )[:, :, t_idx]  # BCHW

    return img


def read_and_process_video(
    video_path: str,
    resolution: str = "720",
    s3_credential_path: str | None = None,
    max_frames: int | None = None,
) -> tuple[torch.Tensor, int, str, tuple[int, int]]:
    """
    Reads an input video, resize it if needed
    Args:
        video_path (str): Path to the input video file.
        resolution (str): Target resolution (e.g., "720", "480")
        s3_credential_path (str): Path to the S3 credential file.
    Returns:
        torch.Tensor: Processed video tensor of shape (C, T, H, W).
        int: Frames per second of the original input video.
        str: Aspect ratio of the original input video.
        tuple[int, int]: Original height and width of the input video.
    Raises:
        ValueError: If the video extension is not one of the supported types.
    """
    ext = os.path.splitext(video_path)[1]
    if ext not in _VIDEO_EXTENSIONS + _IMAGE_EXTENSIONS:
        raise ValueError(f"Invalid video extension: {ext}")

    num_total_frames = NUM_MAX_FRAMES if max_frames is None else max_frames
    input_frames, fps, aspect_ratio, (H, W) = read_and_resize_input(
        video_path,
        num_total_frames=num_total_frames,
        interpolation=cv2.INTER_AREA,
        resolution=resolution,
        s3_credential_path=s3_credential_path,
    )
    return input_frames, fps, aspect_ratio, (H, W)


def normalized_float_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a normalized float image tensor to a uint8 tensor.
    """
    return (tensor * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)


def uint8_to_normalized_float(tensor: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert a uint8 image tensor to a normalized float tensor.
    """
    return (tensor / 127.5 - 1.0).to(dtype)


def get_prompt_from_path(prompt_path: str | None, prompt_str: str | None = None) -> str:
    # First check if the prompt_path exists as-is (with extension already included)
    neg_prompt = None
    if not os.path.exists(prompt_path):
        if os.path.exists(prompt_path + ".txt"):
            prompt_path = prompt_path + ".txt"
        elif os.path.exists(prompt_path + ".pkl"):
            prompt_path = prompt_path + ".pkl"
        elif os.path.exists(prompt_path + ".json"):
            prompt_path = prompt_path + ".json"

    if os.path.exists(prompt_path):
        file_ext = os.path.splitext(prompt_path)[1].lower()
        if file_ext == ".txt":
            with open(prompt_path, "r") as f:
                prompt = f.read().strip()
        elif file_ext == ".pkl":
            with open(prompt_path, "rb") as file:
                prompt_dict = pickle.load(file)
                if "negative_prompt" in prompt_dict:
                    neg_prompt = prompt_dict["negative_prompt"]
                if "prompt" in prompt_dict:
                    prompt = prompt_dict["prompt"]
                else:
                    prompt = prompt_dict[(list(prompt_dict.keys()))[0]]
                if isinstance(prompt, dict):  # for chunk-wise prompt
                    prompt = prompt[(list(prompt.keys()))[0]]
        elif file_ext == ".json":
            with open(prompt_path, "r") as file:
                prompt = json.load(file)
                if isinstance(prompt, dict):
                    if "negative_prompt" in prompt:
                        neg_prompt = prompt["negative_prompt"]
                    try:
                        prompt = prompt["prompt"]
                    except KeyError:
                        video_name = os.path.basename(prompt_path).replace(".json", ".mp4")
                        prompt = prompt[video_name]
        else:
            # Assume it's a text file if no recognized extension
            with open(prompt_path, "r") as f:
                prompt = f.read().strip()

    elif prompt_str is not None:
        prompt = prompt_str
    else:
        log.info(f"Warning: No prompt file found for {prompt_path}, using dummy prompt")
        prompt = DUMMY_PROMPT
    return prompt, neg_prompt


def get_t5_from_prompt(prompt, positive_prompt="", text_encoder_class="T5", cache_dir=None):
    log.info(f"Text encoder class: {text_encoder_class}")
    if isinstance(prompt, str):
        if positive_prompt:
            prompt = f"{prompt} {positive_prompt}"
        t5_embed = (
            get_text_embedding(prompt, text_encoder_class=text_encoder_class, cache_dir=cache_dir)
            .to(dtype=torch.bfloat16)
            .cuda()
        )
    elif isinstance(prompt, torch.Tensor):  # precomputed t5 embeddings (for entire video)
        t5_embed = prompt.unsqueeze(0).to(dtype=torch.bfloat16).cuda()
    elif isinstance(prompt, list):  # one prompt per chunk
        return [
            get_t5_from_prompt(p, positive_prompt, text_encoder_class=text_encoder_class, cache_dir=cache_dir)
            for p in prompt
        ]
    elif isinstance(prompt, dict):  # precomputed t5 embeddings (per chunk)
        # dict format:
        # {
        #     frame index: prompt or precomputed t5
        # }
        prompt = list(prompt.values())
        return get_t5_from_prompt(prompt, positive_prompt, text_encoder_class=text_encoder_class, cache_dir=cache_dir)
    else:
        raise ValueError("prompt format not recognized.")
    return t5_embed


def get_negative_prompt_embedding(
    negative_prompt=None, text_encoder_class="T5", cache_dir=None, s3_credential_path=None, imaginaire_model=None
):
    """
    Get the negative prompt embedding for the given text_encoder_class.
    Args:
        negative_prompt (str): The negative prompt to compute the embedding for.
        text_encoder_class (str): The text encoder class to use.
        cache_dir (str): The cache directory to store the pre-computed embeddings.
        s3_credential_path (str): The path to the S3 credential file.
        imaginaire_model (ImaginaireModel): Only needed if text_encoder_class is reason1_7B family.
             Will use the text_encoder in it to compute the embedding online.
    Returns:
        neg_t5_embeddings (torch.Tensor): The negative prompt embedding.
    """
    if text_encoder_class == "T5":
        if negative_prompt is not None:
            log.info(f"Computing negative prompt embedding, type: {text_encoder_class}")
            neg_t5_embeddings = get_t5_from_prompt(
                negative_prompt, text_encoder_class=text_encoder_class, cache_dir=cache_dir
            )
        else:
            neg_t5_embeddings = load_from_s3_with_cache(
                DEFAULT_NEG_T5_PROMPT_EMBEDDING_PATH,
                easy_io_kwargs={"map_location": torch.device(torch.cuda.current_device())},
                backend_args={
                    "backend": "s3",
                    "path_mapping": None,
                    "s3_credential_path": s3_credential_path,
                },
            )
        # For T5, the dim1 should be 512
        neg_emb = neg_t5_embeddings.to(dtype=torch.bfloat16).cuda()
        if neg_emb.shape[0] > 512:  # Truncate if too large
            neg_emb = neg_emb[:512]
        elif neg_emb.shape[0] < 512:  # Pad if too small
            neg_emb = torch.nn.functional.pad(neg_emb, (0, 0, 0, 512 - neg_emb.shape[0]))
        neg_t5_embeddings = neg_emb.unsqueeze(0)

    elif text_encoder_class.startswith("reason1"):
        if negative_prompt is not None:
            log.info(f"Computing negative prompt embedding, type: {text_encoder_class}")
            neg_t5_embeddings = imaginaire_model.text_encoder.compute_text_embeddings_online(
                {"ai_caption": [negative_prompt], "images": None},
                input_caption_key="ai_caption",
            )
        else:
            raise NotImplementedError(
                f"{text_encoder_class} default negative embedding is not available. Please provide a negative prompt."
            )
        neg_t5_embeddings = neg_t5_embeddings.to(dtype=torch.bfloat16).cuda()  # already has batch dim

    else:  # load pre-computed default negative prompt
        raise NotImplementedError(f"Text encoder class {text_encoder_class} is not supported.")

    return neg_t5_embeddings


def _compute_depth_maps(video_np: np.ndarray) -> torch.Tensor | None:
    """
    Compute depth maps from video frames using Depth Anything models.
    Matches video_annotation.py normalization strategy.

    Args:
        video_np: Video array with shape (T, H, W, C) and dtype uint8

    Returns:
        Depth tensor (1, T, H, W) in range [0, 255] with per-video normalization,
        or None if computation fails
    """
    try:
        from cosmos_predict2._src.transfer2.auxiliary.depth_anything.video_depth_model import VideoDepthAnythingModel

        log.info(f"Computing depth for video with shape {video_np.shape}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info("Using VideoDepthAnything")
        model = VideoDepthAnythingModel(device=device)
        model.setup()
        depth_maps = model.generate(video_np)

        # Normalize to [0, 255]
        depth_tensor = torch.from_numpy(depth_maps.astype(np.float32))
        d_min, d_max = depth_tensor.min(), depth_tensor.max()
        depth_normalized = (depth_tensor - d_min) / (d_max - d_min + 1e-8) * 255.0
        depth_normalized = depth_normalized.unsqueeze(0)  # (1, T, H, W)

        log.info(color_message(f"✓ Depth computed: {depth_normalized.shape}", "bright_green"))
        return depth_normalized

    except Exception as e:
        log.error(color_message(f"Failed to compute depth: {e}", "bright_red"))
        import traceback

        log.error(traceback.format_exc())
        return None


def generate_control_weight_mask_from_prompt(
    video_path: str,
    prompt: str,
    output_folder: str,
    modality: str,
) -> str | None:
    """Generate binary control weight mask from text prompt using SAM2.
    In multi-GPU: only rank 0 generates, others wait and reuse."""
    os.makedirs(output_folder, exist_ok=True)
    mask_name = os.path.splitext(os.path.basename(video_path))[0]
    output_mask_path = os.path.join(output_folder, f"{mask_name}_{modality}_mask.mp4")

    try:
        import torch.distributed as dist

        is_distributed = dist.is_initialized()
    except (ImportError, AttributeError):
        is_distributed = False

    if is_distributed and dist.get_rank() == 0:
        log.info(f"Generating mask from prompt: '{prompt}' for {modality}")

    if not is_distributed or dist.get_rank() == 0:
        segment = VideoSegmentationModel()
        try:
            segment(
                input_video=video_path,
                prompt=prompt,
                output_video=output_mask_path,
                weight_scaler=1.0,
                binarize_video=True,
            )
        except (IndexError, ValueError):
            log.warning(f"No mask generated for prompt '{prompt}'")
            if is_distributed:
                dist.barrier()
            return None

    if is_distributed:
        dist.barrier()
        if not os.path.exists(output_mask_path):
            return None

    return output_mask_path


def read_and_process_control_input(
    video_path: str | None,
    input_control_paths: dict[str, str] | None,
    hint_key: list[str],
    resolution: str = "720",
    seg_control_prompt: str | None = None,
    s3_credential_path: str | None = None,
):
    """
    Load or compute control inputs for video transfer.

    For each modality in hint_key:
    - If pre-computed file exists: load and resize to target resolution
    - If missing: compute on-the-fly (depth via Video Depth Anything, seg via SAM2)
    - edge/vis: skip here, will be computed by augmentor

    Args:
        video_path: Path to the input video file
        input_control_paths: Dictionary mapping modality to file path
        hint_key: List of control modalities to process (e.g., ['depth', 'edge'])
        resolution: Target resolution for processing (e.g., '720', '1080')
        seg_control_prompt: Text prompt for SAM2 segmentation
        s3_credential_path: Path to S3 credentials file

    Returns:
        Tuple of (control_input_dict, mask_video_dict) where mask_video_dict contains
        autogenerated masks
    """
    control_input_dict = {}
    mask_video_dict = {}

    # Configuration for each modality
    modality_config = {
        "edge": {
            "interpolation": cv2.INTER_LINEAR,
            "fallback_msg": "No edge control input file found, will compute online..",
        },
        "vis": {
            "interpolation": cv2.INTER_AREA,
            "fallback_msg": "No vis (blur) control input file found, will compute online..",
        },
        "depth": {
            "interpolation": cv2.INTER_LINEAR,
            "fallback_msg": "No depth control input file found, computing now using Video Depth Anything..",
        },
        "seg": {
            "interpolation": cv2.INTER_NEAREST,
            "fallback_msg": "No segmentation control input file found, computing now using SAM2..",
        },
        "inpaint": {
            "interpolation": cv2.INTER_LINEAR,
            "fallback_msg": None,
        },
        "hdmap_bbox": {
            "interpolation": None,
            "fallback_msg": None,
        },
    }

    for modality in hint_key:
        if modality not in modality_config:
            log.warning(f"Unknown control modality: {modality}, skipping")
            continue

        config = modality_config[modality]
        control_path = input_control_paths.get(modality, None)
        control_key = f"control_input_{modality}"

        if control_path and os.path.exists(control_path):
            # Load pre-computed control input
            control_attr, fps, _, _ = read_and_resize_input(
                control_path,
                resolution=resolution,
                interpolation=config["interpolation"],
                s3_credential_path=s3_credential_path,
            )
            control_input_dict[control_key] = control_attr
        elif config["fallback_msg"]:
            log.info(color_message(config["fallback_msg"], "yellow"))
            # For depth/seg: computed here using third party models
            # For edge/vis: skip (computed by augmentor)
            if modality == "seg":
                log.info(f"Computing seg masks on the fly with prompt {seg_control_prompt=}.")
                segment = VideoSegmentationModel()
                with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_output_video:
                    segment(input_video=video_path, prompt=seg_control_prompt, output_video=temp_output_video.name)
                    control_attr, fps, _, _ = read_and_resize_input(
                        temp_output_video.name,
                        resolution=resolution,
                        interpolation=config["interpolation"],
                        s3_credential_path=s3_credential_path,
                    )
                    control_input_dict["control_input_seg"] = control_attr
            elif modality == "depth":
                # Load video at original resolution, compute depth, then resize
                video_frames, _ = read_video_or_image_into_frames_BCTHW(
                    video_path,
                    H=None,
                    W=None,
                    normalize=False,
                    max_frames=-1,
                    also_return_fps=True,
                    s3_credential_path=s3_credential_path,
                )
                # Convert to (T, H, W, C) format for depth models
                if isinstance(video_frames, torch.Tensor):
                    video_np = einops.rearrange(video_frames[0].cpu().numpy(), "c t h w -> t h w c")
                else:
                    video_np = einops.rearrange(video_frames[0], "c t h w -> t h w c")
                video_np = np.clip(video_np, 0, 255).astype(np.uint8)

                depth_computed = _compute_depth_maps(video_np)
                if depth_computed is not None:
                    depth_rgb = depth_computed.expand(3, -1, -1, -1)  # (3, T, H, W)
                    control_input_dict[control_key] = _resize_to_target_resolution(
                        depth_rgb,
                        resolution=resolution,
                        interpolation=config["interpolation"],
                    )
                else:
                    control_input_dict[control_key] = None

        control_mask_path = input_control_paths.get(f"{modality}_mask")
        mask_prompt = input_control_paths.get(f"{modality}_mask_prompt")

        if control_mask_path is not None and mask_prompt is not None:
            log.warning(f"{modality}: Both mask path and mask prompt provided. Using mask path.")

        if control_mask_path is None and mask_prompt is not None:
            control_mask_path = generate_control_weight_mask_from_prompt(
                video_path=video_path, prompt=mask_prompt, output_folder=tempfile.gettempdir(), modality=modality
            )
            if control_mask_path is None:
                log.warning(f"{modality}: No mask generated from prompt '{mask_prompt}', continuing without mask.")

        if control_mask_path:
            control_mask_attr, fps, _, _ = read_and_resize_input(
                control_mask_path,
                resolution=resolution,
                interpolation=cv2.INTER_LINEAR,
                s3_credential_path=s3_credential_path,
            )
            control_input_dict[f"{control_key}_mask"] = (control_mask_attr[:1] > 127.5).to(torch.bool)
            if mask_prompt is not None:
                mask_video_dict[modality] = control_mask_attr.float() / 255.0

    return control_input_dict, mask_video_dict


def reshape_output_video_to_input_resolution(
    full_video: torch.Tensor,
    hint_key: list[str],
    show_control_condition: bool,
    show_input: bool,
    input_resl_HW: tuple[int, int],
) -> torch.Tensor:
    """
    Reshape the output video to the input resolution. Handles the videos that are composed by concatenating horizontally
    several modalities (e.g. output, control_input, input_video).

    Args:
        full_video: The output video tensor of shape (N, C, T, H, W)
        hint_key: The list of control modalities
        show_control_condition: Whether to show the control condition
        show_input: Whether to show the input video
        input_resl_HW: The original input video resolution (H_original, W_original)
    """
    N, C, T, H, W = full_video.shape

    # Calculate how many videos are concatenated horizontally (e.g. output, control_input, input_video)
    num_videos = 1  # Always have generated video
    if show_control_condition:
        num_videos += len(hint_key)  # Add number of control inputs
    if show_input:
        num_videos += 1  # Add input video
    # Calculate width of each individual video
    single_video_width = W // num_videos

    # Split and resize each video separately
    resized_videos = []
    for i in range(num_videos):
        start_w = i * single_video_width
        end_w = start_w + single_video_width
        video_part = full_video[:, :, :, :, start_w:end_w]

        # Resize this video part to input resolution using resize_video function
        h_out, w_out = [d - (d % 2) for d in input_resl_HW]  # Ensure even dimensions for ffmpeg
        video_part = uint8_to_normalized_float(
            torch.from_numpy(
                resize_video(
                    normalized_float_to_uint8(video_part).cpu().numpy(),
                    h_out,
                    w_out,
                    interpolation=cv2.INTER_LANCZOS4,
                )
            ).to(device=full_video.device),
            dtype=full_video.dtype,
        )
        resized_videos.append(video_part)

    # Concatenate all resized videos back horizontally
    full_video = torch.cat(resized_videos, dim=-1)
    return full_video


def parse_control_input_file_paths(
    input_control_folder_edge=None,
    input_control_folder_vis=None,
    input_control_folder_depth=None,
    input_control_folder_seg=None,
    input_control_folder_edge_mask=None,
    input_control_folder_vis_mask=None,
    input_control_folder_depth_mask=None,
    input_control_folder_seg_mask=None,
    input_control_folder_inpaint_mask=None,
    video_file=None,
) -> dict[str, str]:
    """
    Parse control input file paths

    Args:
        input_control_folder_*: Folder paths for each control modality
        video_file: Input video filename (e.g., "video1.mp4")
    Returns:
        Tuple of control file paths. Expect the same base name as the input video file, but in
        respective control modality folders.
    """
    base_name = os.path.splitext(video_file)[0]
    extension = os.path.splitext(video_file)[1]
    control_modalities = {
        "edge": input_control_folder_edge,
        "vis": input_control_folder_vis,
        "depth": input_control_folder_depth,
        "seg": input_control_folder_seg,
        "edge_mask": input_control_folder_edge_mask,
        "vis_mask": input_control_folder_vis_mask,
        "depth_mask": input_control_folder_depth_mask,
        "seg_mask": input_control_folder_seg_mask,
        "inpaint_mask": input_control_folder_inpaint_mask,
    }

    control_filename = f"{base_name}{extension}"
    control_video_paths = {}
    for modality, folder in control_modalities.items():
        if folder is not None:
            control_video_paths[modality] = os.path.join(folder, control_filename)
        else:
            control_video_paths[modality] = None

    return parse_control_input_single_file_paths(
        input_control_video_path_edge=control_video_paths["edge"],
        input_control_video_path_vis=control_video_paths["vis"],
        input_control_video_path_depth=control_video_paths["depth"],
        input_control_video_path_seg=control_video_paths["seg"],
        input_control_video_path_edge_mask=control_video_paths["edge_mask"],
        input_control_video_path_vis_mask=control_video_paths["vis_mask"],
        input_control_video_path_depth_mask=control_video_paths["depth_mask"],
        input_control_video_path_seg_mask=control_video_paths["seg_mask"],
        input_control_video_path_inpaint_mask=control_video_paths["inpaint_mask"],
    )


def parse_control_input_single_file_paths(
    input_control_video_path_edge=None,
    input_control_video_path_vis=None,
    input_control_video_path_depth=None,
    input_control_video_path_seg=None,
    input_control_video_path_edge_mask=None,
    input_control_video_path_vis_mask=None,
    input_control_video_path_depth_mask=None,
    input_control_video_path_seg_mask=None,
    input_control_video_path_inpaint_mask=None,
) -> dict[str, str]:
    """
    Parse control input single file paths

    Args:
        input_control_video_path_*: Direct file paths for each control modality
    Returns:
        Dict of control file paths. Uses the direct paths provided.
    """
    control_modalities = {
        "edge": input_control_video_path_edge,
        "vis": input_control_video_path_vis,
        "depth": input_control_video_path_depth,
        "seg": input_control_video_path_seg,
        "edge_mask": input_control_video_path_edge_mask,
        "vis_mask": input_control_video_path_vis_mask,
        "depth_mask": input_control_video_path_depth_mask,
        "seg_mask": input_control_video_path_seg_mask,
        "inpaint_mask": input_control_video_path_inpaint_mask,
    }

    # Generate control file paths
    control_paths = {}
    for modality, single_path in control_modalities.items():
        if single_path is not None:
            if not os.path.exists(single_path):
                log.info(
                    color_message(
                        f"Required control input file not found: {single_path}. Will compute online from input video.",
                        "yellow",
                    )
                )
                control_paths[modality] = None
            else:
                control_paths[modality] = single_path
        else:
            # No path provided for this modality
            control_paths[modality] = None

    return control_paths


def validate_image_context_path(image_context_path: str) -> None:
    # Prepare reference image info
    if image_context_path:
        if not os.path.exists(image_context_path):
            raise ValueError(f"Image context file not found: {image_context_path}")
        ref_image_path = image_context_path
        ref_image_name = os.path.splitext(os.path.basename(image_context_path))[0]
        log.info(f"Using reference image: {ref_image_name}")
    else:
        ref_image_path = None
        ref_image_name = None
        log.info(
            color_message("No reference image provided. Generating from text prompt and control videos.", "yellow")
        )
    return ref_image_path, ref_image_name


def get_unique_seed(
    video_path: str, save_root: str, experiment: str, ckpt_iter: str, num_conditional_frames: int
) -> int:
    seed = int(time.time())
    seed += int(hashlib.sha256(video_path.encode()).hexdigest(), 16) % 1000000
    seed += int(hashlib.sha256(save_root.encode()).hexdigest(), 16) % 1000000
    seed += int(hashlib.sha256(experiment.encode()).hexdigest(), 16) % 1000000
    seed += int(hashlib.sha256(ckpt_iter.encode()).hexdigest(), 16) % 1000000
    seed += num_conditional_frames
    return seed


def color_message(message: str, color: str = "white") -> str:
    """Log a message with color formatting.

    Args:
        message: The message to log
        color: Color name (red, green, yellow, blue, magenta, cyan, white)
        rank0_only: Whether to log only on rank 0
    """
    colors = {
        "red": "\033[31m",
        "bright_red": "\033[91m",
        "green": "\033[32m",
        "bright_green": "\033[92m",
        "yellow": "\033[33m",
        "bright_yellow": "\033[93m",
        "blue": "\033[34m",
        "bright_blue": "\033[94m",
        "magenta": "\033[35m",
        "bright_magenta": "\033[95m",
        "cyan": "\033[36m",
        "bright_cyan": "\033[96m",
        "white": "\033[37m",
        "bright_white": "\033[97m",
        "grey": "\033[90m",
        "gray": "\033[90m",
    }

    color_code = colors.get(color.lower(), "")
    reset_code = "\033[0m" if color_code else ""
    colored_message = f"{color_code}{message}{reset_code}"
    return colored_message


def compile_tokenizer_if_enabled(pipeline: Any, compilation_mode: str) -> None:
    """
    Optionally compiles the tokenizer's encode and decode methods using torch.compile.

    Args:
        pipeline: The inference pipeline object containing the tokenizer. This can be either
            TransferControl2WorldPipeline or MultiviewControl2WorldPipeline.
        compilation_mode: String describing the compilation type. Must be one of
            "none", "moderate", or "aggressive". "moderate" compiles only the encode method,
            "aggressive" compiles both encode and decode methods, and "none" disables compilation.
    """
    compile_tokenizer = compilation_mode != "none"

    if not compile_tokenizer or compilation_mode not in ["moderate", "aggressive", "aggressive_cuda_graphs", "none"]:
        log.info("Tokenizer compilation disabled")
        return

    if not hasattr(torch, "compile"):
        log.warning("torch.compile not available (requires PyTorch 2.0+), skipping tokenizer compilation")
        return

    if isinstance(pipeline.model.tokenizer.encode, torch.jit.ScriptModule) and isinstance(
        pipeline.model.tokenizer.decode, torch.jit.ScriptModule
    ):
        log.warning("Tokenizer is already JIT compiled, skipping torch.compile")
        return

    # Configure Dynamo settings
    try:
        # PyTorch >= 2.7
        torch._dynamo.config.recompile_limit = 32
    except AttributeError:
        try:
            torch._dynamo.config.cache_size_limit = 32
        except AttributeError:
            log.warning("Torch Dynamo configuration not available")

    def compile_method(method: Callable, method_name: str, **kwargs: Any) -> Callable:
        """Helper function to compile a method if not already compiled."""
        if hasattr(method, "_orig_mod"):
            log.info(f"Tokenizer {method_name} method already compiled")
            return method
        else:
            log.info(f"Compiling tokenizer {method_name} method")
            return torch.compile(method, dynamic=False, **kwargs)

    torch_compile_mode = "default" if compilation_mode != "aggressive_cuda_graphs" else "reduce-overhead"
    if compilation_mode != "none":
        pipeline.model.tokenizer.encode = compile_method(
            pipeline.model.tokenizer.encode, "encode", mode=torch_compile_mode
        )
        log.info("Tokenizer compilation active. Expect some overhead on the first use.")
    if compilation_mode in ["aggressive", "aggressive_cuda_graphs"]:
        pipeline.model.tokenizer.decode = compile_method(
            pipeline.model.tokenizer.decode, "decode", mode=torch_compile_mode
        )
