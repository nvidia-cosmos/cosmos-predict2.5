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

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from PIL import Image

# Re-use the original helper utilities so that we only implement the deltas.
from qwen_vl_utils.vision_process import extract_vision_info, fetch_image, smart_nframes, smart_resize
from qwen_vl_utils.vision_process import process_vision_info as process_vision_info_original

from cosmos_predict2._src.imaginaire.utils import log

# Timestamp augmentation utilities (video)
from cosmos_predict2._src.reason1.datasets.augmentors.timestamp import overlay_text  # noqa: E501
from cosmos_predict2._src.reason1.utils.video_preprocess import tensor_to_pil_images

# -----------------------------------------------------------------------------
# Helper conversions between token length <--> number of raw pixels.
# -----------------------------------------------------------------------------

_DEFAULT_PATCH_SIZE = 14
_TEMPORAL_PATCH_SIZE = 2  # Qwen merges 2 consecutive frames into one token.

# Minimum spatial resolution (per side) used in Qwen video processing
_MIN_HEIGHT_WIDTH = 56  # pixels

Image.MAX_IMAGE_PIXELS = 933120000
_VIDEO_EXTENSIONS = "mp4 avi webm mov".split()
VIDEO_DECODER_OPTIONS = {}


def _token_to_pixels(
    token_length: int,
    patch_size: int = _DEFAULT_PATCH_SIZE,
    temporal_patch_size: int = _TEMPORAL_PATCH_SIZE,
) -> int:
    """Convert (vision) token length to the corresponding number of pixels.

    Qwen performs a 2×2 spatial patch merge before feeding the patches into the
    vision transformer. Consequently, one *merged patch* covers
    ``(patch_size*2) × (patch_size*2)`` pixels. When dealing with video, two
    consecutive frames are grouped together as well (``temporal_patch_size``).

    The formula therefore is::

        pixels = num_tokens · (2·patch_size)^2 · temporal_patch_size

    The same helper is used in ``_video_decoder_qwen_func`` so that we exactly
    replicate its behaviour.
    """

    merged_patch_size = patch_size * 2
    return token_length * (merged_patch_size**2) * temporal_patch_size


def _fetch_video(ele: dict, min_num_vision_tokens: int, total_pixels: int) -> Tuple[List[Image.Image], float]:
    # Manual video processing based on _video_decoder_qwen_func to handle max_frames constraint
    video_reader = decord.VideoReader(ele["video"], num_threads=0)
    total_frames, video_fps = len(video_reader), video_reader.get_avg_fps()

    min_pixels = _token_to_pixels(min_num_vision_tokens)
    max_frames = total_pixels // (_MIN_HEIGHT_WIDTH**2) // _TEMPORAL_PATCH_SIZE

    # Sample frames based on target fps (default 2.0)
    target_fps = ele.get("fps", 2.0)
    nframes = smart_nframes({"fps": target_fps}, total_frames=total_frames, video_fps=video_fps)
    nframes = min(nframes, max_frames)  # This is the key constraint missing in fetch_video!

    idx = np.linspace(0, total_frames - 1, nframes).round().astype(np.int32).tolist()
    video_frames = video_reader.get_batch(idx).asnumpy()  # Keep as numpy (T, H, W, C)
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    # Recompute max_pixels based on number of sampled frames
    nframes, height, width, channels = video_frames.shape
    max_pixels_per_frame = total_pixels // nframes
    resized_height, resized_width = smart_resize(
        height,
        width,
        min_pixels=min_pixels,
        max_pixels=max_pixels_per_frame,
    )

    # Convert numpy frames to PIL images and resize
    pil_images = []
    for frame_idx in range(nframes):
        # Get single frame (H, W, C) as numpy array
        frame_np = video_frames[frame_idx].astype(np.uint8)
        # Convert to PIL image
        pil_image = Image.fromarray(frame_np)
        # Resize using PIL
        pil_image = pil_image.resize((resized_width, resized_height), Image.BICUBIC)
        pil_images.append(pil_image)

    # Clean up video reader
    video_reader.seek(0)
    del video_reader

    return pil_images, sample_fps


# -----------------------------------------------------------------------------
# Public API – extended process_vision_info
# -----------------------------------------------------------------------------


def process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    min_num_vision_tokens: int = 16,
    max_num_vision_tokens: Optional[int] = None,
    timestamp_video: bool = False,
    return_video_kwargs: bool = False,
):
    """Extended variant of ``qwen_vl_utils.vision_process.process_vision_info``.

    Parameters
    ----------
    conversations : list | list[list]
        Either a *single* conversation (list of message dicts) or a *batch* of
        conversations (list of conversation lists).
    min_num_vision_tokens : int, default = 16
        Minimum number of vision tokens to allocate for each video.
    max_num_vision_tokens : int | None, default = ``None``
        When provided, overrides any ``max_pixels`` or ``total_pixels`` values
        passed in the individual *vision elements* so that the overall vision
        budget is limited to ``max_num_vision_tokens``. The conversion from
        *token* budget to *pixel* budget follows the formula used in
        ``_video_decoder_qwen_func``.
    timestamp_video : bool, default = ``False``
        If *True*, a timestamp (in seconds) is rendered on top of a black bar
        appended at the bottom of each video frame. The logic mirrors
        ``augmentors.timestamp.overlay_text``.
    return_video_kwargs : bool, default = ``False``
        Same semantics as in the original helper – when enabled, the returned
        tuple contains an *additional* ``dict`` with auxiliary video kwargs
        currently consisting of the sampled FPS per video.

    Returns
    -------
    image_inputs : list[Image.Image] | None
    video_inputs : list[torch.Tensor | list[Image.Image]] | None
        When max_num_vision_tokens is specified, videos are returned as lists
        of PIL images. Otherwise, follows the original qwen behavior.
    extra_kwargs : dict | None
        Only present when ``return_video_kwargs`` is *True*.
    """

    if max_num_vision_tokens is None:
        # ------------------------------------------------------------------
        # 1) original implementation
        # ------------------------------------------------------------------
        # Always ask the original util to also return `fps` so that we can
        # optionally add timestamp overlays even when the caller has
        # `return_video_kwargs=False`.
        image_inputs, video_inputs, video_kwargs = process_vision_info_original(conversations, return_video_kwargs=True)
    else:
        # ------------------------------------------------------------------
        # 2) custom implementation
        # ------------------------------------------------------------------
        vision_infos = extract_vision_info(conversations)

        # Count how many images & videos we have – required to distribute the
        # pixel budget fairly.
        num_images = sum(1 for ele in vision_infos if ("image" in ele or "image_url" in ele))
        num_videos = sum(1 for ele in vision_infos if "video" in ele)

        # Ensure only one modality (image or video) is present when using token budget
        assert not (num_images > 0 and num_videos > 0), (
            f"Cannot mix images ({num_images}) and videos ({num_videos}) when using "
            f"max_num_vision_tokens. Use separate calls for each modality."
        )

        # Derive per-media pixel budgets.
        if num_images > 0:
            img_total_pixels = _token_to_pixels(max_num_vision_tokens, temporal_patch_size=1)
            img_pixels_per_image = img_total_pixels // num_images
        else:
            img_pixels_per_image = None

        if num_videos > 0:
            vid_total_pixels = _token_to_pixels(max_num_vision_tokens, temporal_patch_size=_TEMPORAL_PATCH_SIZE)
            vid_pixels_per_video = vid_total_pixels // num_videos
        else:
            vid_pixels_per_video = None

        image_inputs: List[Image.Image] = []
        video_inputs: List[Union[torch.Tensor, List[Image.Image]]] = []
        video_sample_fps: List[float] = []

        for ele_orig in vision_infos:
            ele = copy.deepcopy(ele_orig)

            if "image" in ele or "image_url" in ele:
                # Override max_pixels for images only when we computed a budget.
                if img_pixels_per_image is not None:
                    ele.pop("max_pixels", None)
                    ele["max_pixels"] = img_pixels_per_image

                try:
                    img = fetch_image(ele)
                    image_inputs.append(img)
                except Exception as e:  # noqa: BLE001
                    log.warning("Failed to process image element %s – %s", ele, e)

            elif "video" in ele:
                try:
                    video_frames, sample_fps = _fetch_video(ele, min_num_vision_tokens, vid_pixels_per_video)
                    video_inputs.append(video_frames)
                    video_sample_fps.append(sample_fps)
                except Exception as e:  # noqa: BLE001
                    log.warning("Failed to process video element %s – %s", ele, e)
            else:
                log.warning("Unrecognised vision element – skipping: %s", ele)

        video_kwargs: Optional[Dict[str, Any]] = {"fps": video_sample_fps}

    # Convert empty lists to *None* for compatibility with the original API.
    if image_inputs is None or len(image_inputs) == 0:
        image_inputs = None  # type: ignore
    if video_inputs is None or len(video_inputs) == 0:
        video_inputs = None  # type: ignore

    # ------------------------------------------------------------------
    # 3) Timestamp overlay (optional)
    # ------------------------------------------------------------------

    if timestamp_video and video_inputs is not None:
        # Resolve FPS list – fall back to default 2.0 if unknown.
        fps_list = video_kwargs["fps"]

        for idx, (vid_obj, fps) in enumerate(zip(video_inputs, fps_list)):
            try:
                # Convert tensor to PIL images if needed
                if isinstance(vid_obj, torch.Tensor):
                    vid_obj = tensor_to_pil_images(vid_obj)

                # Apply timestamp overlay (works with list of PIL images)
                video_inputs[idx], _ = overlay_text(vid_obj, fps)
            except Exception as e:  # noqa: BLE001
                log.warning("Timestamp overlay failed for video %d – %s", idx, e)

    if return_video_kwargs:
        return image_inputs, video_inputs, video_kwargs

    return image_inputs, video_inputs, None
