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

import fnmatch
import math
import os
from glob import glob

import mediapy as media
import numpy as np
import torch
from loguru import logger
from mediapy import _VideoArray
from PIL import Image
from torchvision.transforms import functional as F

from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.datasets.utils import IMAGE_RES_SIZE_INFO

_CREDENTIAL, _BACKEND = "credentials/pdx_cosmos_base.secret", "s3"
_DTYPE, _DEVICE = torch.bfloat16, "cuda"
_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)

_PROMPT_EXTENSIONS = [".txt"]
_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
_VIDEO_EXTENSIONS = [".mp4"]

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def get_sample_batch(
    resolution: str = "1024",
    aspect_ratio: str = "16,9",
    batch_size: int = 1,
):
    w, h = IMAGE_RES_SIZE_INFO[resolution][aspect_ratio]
    data_batch = {
        "dataset_name": "image_data",
        "images": torch.randn(batch_size, 3, h, w).cuda(),
        "t5_text_embeddings": torch.randn(batch_size, 512, 1024).cuda(),
        "fps": torch.randint(16, 32, (batch_size,)).cuda(),
        "padding_mask": torch.zeros(batch_size, 1, h, w).cuda(),
    }

    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
            data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

    return data_batch


def resize_input(video: torch.Tensor, resolution: tuple[int, int]):
    r"""
    Resizes and crops the input video tensor while preserving aspect ratio.

    The video is first resized so that the smaller dimension matches the target resolution,
    preserving the aspect ratio. Then, it's center-cropped to the target resolution.

    Args:
        video (torch.Tensor): Input video tensor of shape (T, C, H, W).
        resolution (list[int]): Target resolution [H, W].

    Returns:
        torch.Tensor: Resized and cropped video tensor of shape (T, C, target_H, target_W).
    """

    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = resolution

    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    video_resized = F.resize(video, list(resizing_shape))
    video_cropped = F.center_crop(video_resized, list(resolution))
    return video_cropped


def read_and_process_image(img_path: str, resolution: tuple[int, int], num_video_frames: int, resize: bool = True):
    """
    Reads an image, converts it to a video tensor, and processes it for model input.

    The image is loaded, converted to a tensor, and replicated to match the
    `num_video_frames`. It's then optionally resized and permuted to the
    standard video format (B, C, T, H, W).

    Args:
        img_path (str): Path to the input image file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        num_video_frames (int):  Number of frames needed by the model (should equal model.tokenizer.get_pixel_num_frames(model.config.state_t)).
        resize (bool, optional): Whether to resize the image to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W).

    Raises:
        ValueError: If the image extension is not one of the supported types.
    """
    ext = os.path.splitext(img_path)[1]
    if ext not in _IMAGE_EXTENSIONS:
        raise ValueError(f"Invalid image extension: {ext}")

    # Read the image
    img = Image.open(img_path)

    # Convert to tensor
    img = F.to_tensor(img)
    # Create a video tensor by repeating the first frame
    vid_input = img.unsqueeze(0)  # Add temporal dimension T=1

    # Repeat the first frame to match the desired number of video frames
    # Note: The actual content for frames > 0 will be generated by the model.
    vid_input = torch.cat([vid_input, torch.zeros_like(vid_input).repeat(num_video_frames - 1, 1, 1, 1)], dim=0)
    vid_input = (vid_input * 255.0).to(torch.uint8)  # Convert to uint8 range if needed (might depend on model)
    if resize:
        # Resize and crop to the target resolution
        vid_input = resize_input(vid_input, resolution)

    # Convert to {B, C, T, H, W} format expected by the model
    vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Add batch dim B=1 and permute
    return vid_input


def read_and_process_video(
    video_path: str,
    resolution: tuple[int, int],
    num_video_frames: int,
    num_latent_conditional_frames: int = 2,
    resize: bool = True,
):
    """
    Reads a video, processes it for model input.

    The video is loaded using easy_io, and uses the last 4x(num_latent_conditional_frames - 1) + 1 from the video.
    If the video is shorter than num_video_frames, it pads with the last frame repeated.
    The first num_latent_conditional_frames are marked as conditioning frames.

    Args:
        video_path (str): Path to the input video file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        num_video_frames (int): Number of frames needed by the model (should equal model.tokenizer.get_pixel_num_frames(model.config.state_t)).
        num_latent_conditional_frames (int): Number of latent conditional frames from the input video (1 or 2).
        resize (bool, optional): Whether to resize the video to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W) where T equals num_video_frames.

    Raises:
        ValueError: If the video extension is not supported or other validation errors.

    Note:
        Uses the last 4x(num_latent_conditional_frames - 1) + 1 frames from the video. If video is shorter, pads with last frame repeated.
    """
    ext = os.path.splitext(video_path)[1]
    if ext.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Invalid video extension: {ext}")

    # Load video using easy_io
    try:
        video_frames, video_metadata = easy_io.load(video_path)  # Returns (T, H, W, C) numpy array
        logger.info(f"Loaded video with shape {video_frames.shape}, metadata: {video_metadata}")
    except Exception as e:
        raise ValueError(f"Failed to load video {video_path}: {e}")

    # Convert numpy array to tensor and rearrange dimensions
    video_tensor = torch.from_numpy(video_frames).float() / 255.0  # Convert to [0, 1] range
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

    available_frames = video_tensor.shape[1]

    # Calculate how many frames to extract from input video
    frames_to_extract = 4 * (num_latent_conditional_frames - 1) + 1
    logger.info(f"Will extract {frames_to_extract} frames from input video and pad to {num_video_frames}")

    # Validate num_latent_conditional_frames
    if num_latent_conditional_frames not in [1, 2]:
        raise ValueError(f"num_latent_conditional_frames must be 1 or 2, but got {num_latent_conditional_frames}")

    # Create output tensor with exact num_video_frames
    C, _, H, W = video_tensor.shape
    full_video = torch.zeros(C, num_video_frames, H, W)

    if available_frames < frames_to_extract:
        raise ValueError(
            f"Video has only {available_frames} frames but needs at least {frames_to_extract} frames for num_latent_conditional_frames={num_latent_conditional_frames}"
        )

    # Extract the last frames_to_extract from input video
    start_idx = available_frames - frames_to_extract
    extracted_frames = video_tensor[:, start_idx:, :, :]
    full_video[:, :frames_to_extract, :, :] = extracted_frames
    logger.info(f"Extracted last {frames_to_extract} frames from video (frames {start_idx} to {available_frames - 1})")

    # Pad remaining frames with the last extracted frame
    if frames_to_extract < num_video_frames:
        last_frame = extracted_frames[:, -1:, :, :]  # (C, 1, H, W)
        padding_frames = num_video_frames - frames_to_extract
        last_frame_repeated = last_frame.repeat(1, padding_frames, 1, 1)  # (C, padding_frames, H, W)
        full_video[:, frames_to_extract:, :, :] = last_frame_repeated
        logger.info(f"Padded {padding_frames} frames with last extracted frame")

    # Convert to the format expected by the rest of the pipeline
    full_video = full_video.permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)
    full_video = (full_video * 255.0).to(torch.uint8)  # Convert to uint8 range

    if resize:
        # Resize and crop to the target resolution
        full_video = resize_input(full_video, resolution)

    # Convert to {B, C, T, H, W} format expected by the model
    full_video = full_video.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Add batch dim B=1 and permute
    return full_video


def set_s3_backend(backend: str = _BACKEND, credentials: str = _CREDENTIAL) -> None:
    """Set the backend with the proper credentials."""
    credentials = credentials or _CREDENTIAL
    easy_io.set_s3_backend(
        backend_args={
            "backend": backend,
            "s3_credential_path": credentials,
        }
    )


def get_filepaths(input_pattern: str) -> list[str]:
    """Returns a list of filepaths from a pattern, supporting wildcards."""
    if input_pattern.startswith("s3://"):
        return _get_s3_filepaths(input_pattern)
    else:
        filepaths = glob(str(input_pattern))
        return sorted(list(set(filepaths)))


def _get_s3_filepaths(s3_pattern: str) -> list[str]:
    """Get S3 filepaths matching a pattern with wildcards."""
    # Parse the pattern to find the base directory and pattern
    pattern_parts = s3_pattern.replace("s3://", "").split("/")

    # Find the first part with wildcards
    base_parts = []
    pattern_start_index = -1

    for i, part in enumerate(pattern_parts):
        if "*" in part or "?" in part or "[" in part:
            pattern_start_index = i
            break
        base_parts.append(part)

    if pattern_start_index == -1:
        # No wildcards, just check if the file exists
        if easy_io.exists(s3_pattern):
            return [s3_pattern]
        else:
            return []

    # Build the base directory path
    base_dir = "s3://" + "/".join(base_parts) if base_parts else "s3://"

    # Build the pattern for matching (everything after the base directory)
    pattern_suffix = "/".join(pattern_parts[pattern_start_index:])

    # Use recursive listing to get all files under the base directory
    filepaths = []
    try:
        for relative_path in easy_io.list_dir_or_file(
            base_dir,
            list_dir=False,  # Only list files, not directories
            list_file=True,
            recursive=True,  # This is the key - recursive listing
        ):
            # Check if this relative path matches our pattern
            if fnmatch.fnmatch(relative_path, pattern_suffix):
                full_path = f"{base_dir.rstrip('/')}/{relative_path}"
                filepaths.append(full_path)
    except Exception:
        # If listing fails, return empty list
        pass

    return sorted(list(set(filepaths)))


def read_video(filepath: str) -> np.ndarray:
    """Reads a video from a filepath in S3 or local.

    Args:
        filepath: The filepath to the video. (local or S3)
    Returns:
        The video as a numpy array, layout TxHxWxC, range [0..255], uint8 dtype.
    """
    if filepath.startswith("s3://"):
        video_data, metadata = easy_io.load(filepath)
        video = _VideoArray(video_data, metadata)
    else:
        video = media.read_video(filepath)
    # convert the grey scale image to RGB
    # since our tokenizers always assume 3-channel RGB image
    if video.ndim == 3:
        video = np.stack([video] * 3, axis=-1)
    # convert RGBA to RGB
    if video.shape[-1] == 4:
        video = video[..., :3]
    return video


def _pad_to_even(video: np.ndarray) -> np.ndarray:
    """Pads video frames to even height and width if necessary.

    Args:
        video: A numpy array of shape (T, H, W, C) in range [0..255], uint8 dtype.
    Returns:
        A numpy array of shape (T, H, W, C) in range [0..255], uint8 dtype.
    """
    H, W = video.shape[-3:-1]
    pad_h = H % 2
    pad_w = W % 2
    if pad_h == 0 and pad_w == 0:
        return video
    pad = ((0, 0), (0, pad_h), (0, pad_w), (0, 0))
    return np.pad(video, pad_width=pad, mode="edge")


def write_video(filepath: str, video: np.ndarray, fps: int = 24, lossless: bool = True) -> None:
    """Writes a video to a filepath in S3 or local.

    Args:
        filepath: A string filepath to save the video. For S3, the filepath should start with s3://.
        video: A numpy array of shape (T, H, W, C) in range [0..255], uint8 dtype.
        fps: The frames per second of the video.
        lossless: Whether to use lossless compression.
    """
    video = _pad_to_even(video)
    if lossless:
        ffmpeg_params = [
            "-c:v",
            "libx264",  # Use H.264 codec
            "-preset",
            "veryslow",  # Slowest preset = best compression
            "-qp",
            "0",  # Quantization parameter 0 = lossless
            "-crf",
            "0",  # Constant Rate Factor 0 = lossless
        ]
    else:
        ffmpeg_params = [
            "-c:v",
            "libx264",
            "-preset",
            "veryslow",
            "-crf",
            "23",  # Reasonable quality–compression tradeoff
        ]
    easy_io.dump(video, filepath, fps=fps, quality=None, ffmpeg_params=ffmpeg_params)


def write_image(filepath: str, image: np.ndarray, quality: int = 85) -> None:
    """Writes an image to a filepath in S3 or local.

    Args:
        filepath: A string filepath to save the image. For S3, the filepath should start with s3://.
        image: A numpy array of shape (H, W, C) in range [0..255], uint8 dtype.
        quality: The quality of the image, on a scale from 0 (worst) to 95 (best), default=85.
                 https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg
    """
    pil_image = Image.fromarray(image)
    easy_io.dump(pil_image, filepath, quality=quality)


def numpy2tensor(
    input_image: np.ndarray, dtype: torch.dtype = _DTYPE, device: str = _DEVICE, range_min: int = -1
) -> torch.Tensor:
    """Converts image(dtype=np.uint8) to `dtype` in range [0..255].

    Args:
        input_image: A batch of images in range [0..255], BxHxWx3 layout.
    Returns:
        A torch.Tensor of layout Bx3xHxW in range [-1..1], dtype.
    """
    ndim = input_image.ndim
    indices = list(range(1, ndim))[-1:] + list(range(1, ndim))[:-1]
    image = input_image.transpose((0,) + tuple(indices)) / _UINT8_MAX_F
    if range_min == -1:
        image = 2.0 * image - 1.0
    return torch.from_numpy(image).to(dtype).to(device)


def tensor2numpy(input_tensor: torch.Tensor, range_min: int = -1) -> np.ndarray:
    """Converts tensor in [-1,1] to image(dtype=np.uint8) in range [0..255].

    Args:
        input_tensor: Input image tensor of Bx3xHxW layout, range [-1..1].
    Returns:
        A numpy image of layout BxHxWx3, range [0..255], uint8 dtype.
    """
    if range_min == -1:
        input_tensor = (input_tensor.float() + 1.0) / 2.0
    ndim = input_tensor.ndim
    output_image = input_tensor.clamp(0, 1).cpu().numpy()
    output_image = output_image.transpose((0,) + tuple(range(2, ndim)) + (1,))
    return (output_image * _UINT8_MAX_F + 0.5).astype(np.uint8)
