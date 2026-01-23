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

from typing import Optional

import cosmos_predict2._src.imaginaire.datasets.webdataset.augmentors.image.normalize as normalize
import cosmos_predict2._src.imaginaire.datasets.webdataset.augmentors.image.padding as padding
import cosmos_predict2._src.imaginaire.datasets.webdataset.augmentors.image.resize as resize
import cosmos_predict2._src.predict2.datasets.augmentors.append_fps_frames_for_image as append_fps_frames_for_image
import cosmos_predict2._src.predict2.datasets.augmentors.caption_filter as caption_filter
import cosmos_predict2._src.predict2.datasets.augmentors.merge_datadict as merge_datadict
import cosmos_predict2._src.predict2.datasets.augmentors.text_transforms_for_image as text_transforms_for_image
import cosmos_predict2._src.predict2.datasets.augmentors.text_transforms_for_video as text_transforms_for_video
import cosmos_predict2._src.predict2.datasets.augmentors.video_parsing as video_parsing
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO

AUGMENTOR_OPTIONS = {}

CAMERA_MOVEMENT_PHRASES = [
    # Panning
    "camera pan",
    "camera pans",
    "camera slowly pan",
    "camera slowly pans",
    "camera quickly pans",
    "camera fast pans",
    "panning shot",
    "panning camera",
    "slow pan",
    "quick pan",
    "fast pan",
    "pan across",
    "pan around",
    "pan shot",
    "panoramic shot",
    # Tracking / Dolly
    "camera moves",
    "camera slowly moves",
    "camera quickly moves",
    "moving camera",
    "tracking shot",
    "tracking camera",
    "dolly shot",
    "dolly in",
    "dolly out",
    "camera follows",
    "camera tracks",
    "tracking movement",
    # Sweeps / Rotations
    "sweeping camera",
    "camera sweep",
    "rotating camera",
    "camera rotation",
    "camera rotates",
    "camera circles around",
    # Tilts
    "camera tilt",
    "camera tilts",
    "camera slowly tilts",
    "tilting camera",
    "tilt up",
    "tilt down",
    # Zooms
    "camera zoom",
    "camera zooms",
    "zooming camera",
    "zoom in",
    "zoom out",
    # Handheld / Shake
    "handheld camera",
    "handheld shot",
    "shaky camera",
    "camera shake",
    "shaky shot",
    "handheld movement",
]


def augmentor_register(key):
    log.info(f"registering {key}...")

    def decorator(func):
        AUGMENTOR_OPTIONS[key] = func
        return func

    return decorator


def get_video_text_transform(
    caption_type: str,
    embedding_type: Optional[str] = "t5_xxl",
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    num_video_frames: int = -1,
):
    del num_video_frames

    # common args that are shared across all caption types
    base_args = {
        "captions_key": "metas",
        "embeddings_key": embedding_type,
        "t5_tokens": {"num": 512},
        "is_mask_all_ones": True,
    }
    caption_probs = {
        "long": long_caption_ratio,
        "medium": medium_caption_ratio,
        "short": short_caption_ratio,
        "user": user_caption_ratio,
    }

    if caption_type == "vila_caption":
        args = {
            **base_args,
            "caption_windows_key": "windows",
            "caption_type": "vila_caption",
            "embedding_caption_type": "vila_caption",
        }
    elif caption_type == "t2w_qwen2p5_7b":
        log.info(
            f"caption_type: {caption_type}, long_caption_ratio: {long_caption_ratio}, medium_caption_ratio: {medium_caption_ratio}, short_caption_ratio: {short_caption_ratio}, user_caption_ratio: {user_caption_ratio}"
        )
        args = {
            **base_args,
            "caption_windows_key": "t2w_windows",
            "caption_type": "qwen2p5_7b_caption",
            "embedding_caption_type": caption_type,
            "caption_probs": caption_probs,
        }
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        args = {
            **base_args,
            "caption_windows_key": "i2w_windows_later_frames",
            "caption_type": "qwen2p5_7b_caption",
            "embedding_caption_type": caption_type,
            "caption_probs": caption_probs,
        }
    elif caption_type == "t2w_qwen3_vl_30b_a3b":
        log.info(
            f"caption_type: {caption_type}, long_caption_ratio: {long_caption_ratio}, medium_caption_ratio: {medium_caption_ratio}, short_caption_ratio: {short_caption_ratio}, user_caption_ratio: {user_caption_ratio}"
        )
        args = {
            **base_args,
            "caption_windows_key": "t2w_windows",
            "caption_type": "qwen3_vl_30b_a3b_caption",
            "embedding_caption_type": caption_type,
            "caption_probs": caption_probs,
        }
    elif caption_type == "i2w_qwen3_vl_30b_a3b_later_frames":
        args = {
            **base_args,
            "caption_windows_key": "i2w_windows_later_frames",
            "caption_type": "qwen3_vl_30b_a3b_caption",
            "embedding_caption_type": caption_type,
            "caption_probs": caption_probs,
        }
    elif caption_type == "t2w_qwen3_vl_235b":
        log.info(
            f"caption_type: {caption_type}, long_caption_ratio: {long_caption_ratio}, medium_caption_ratio: {medium_caption_ratio}, short_caption_ratio: {short_caption_ratio}, user_caption_ratio: {user_caption_ratio}"
        )
        args = {
            **base_args,
            "caption_windows_key": "t2w_windows",
            "caption_type": "qwen3_vl_235b_caption",
            "embedding_caption_type": caption_type,
            "caption_probs": caption_probs,
        }
    elif caption_type == "i2w_qwen3_vl_235b_later_frames":
        args = {
            **base_args,
            "caption_windows_key": "i2w_windows_later_frames",
            "caption_type": "qwen3_vl_235b_caption",
            "embedding_caption_type": caption_type,
            "caption_probs": caption_probs,
        }
    elif caption_type == "t2w_qwen3_vl_235b_fp8":
        log.info(
            f"caption_type: {caption_type}, long_caption_ratio: {long_caption_ratio}, medium_caption_ratio: {medium_caption_ratio}, short_caption_ratio: {short_caption_ratio}, user_caption_ratio: {user_caption_ratio}"
        )
        args = {
            **base_args,
            "caption_windows_key": "t2w_windows",
            "caption_type": "qwen3_vl_235b_fp8_caption",
            "embedding_caption_type": caption_type,
            "caption_probs": caption_probs,
        }
    elif caption_type == "i2w_qwen3_vl_235b_fp8_later_frames":
        args = {
            **base_args,
            "caption_windows_key": "i2w_windows_later_frames",
            "caption_type": "qwen3_vl_235b_fp8_caption",
            "embedding_caption_type": caption_type,
            "caption_probs": caption_probs,
        }
    else:
        raise ValueError(f"Unsupported caption type ({caption_type}) for video data")

    video_text_transform = L(text_transforms_for_video.TextTransformForVideo)(
        input_keys=[],
        args=args,
    )
    return video_text_transform


@augmentor_register("video_basic_augmentor_v1")
def get_video_augmentor_v1(
    resolution: str,
    caption_type: str = "vila_caption",
    embedding_type: str = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
):
    """Video augmentor V1. It relies on a separate video decoder to decode videos of required number of frames.
    Augmentors here will resize the video, add reflection padding, and extract captions and embeddings.

    Supported caption_type include vila_caption.
    Supported embedding_type include t5_xxl.
    """
    assert caption_type == "vila_caption", f"Unsupported caption type ({caption_type}) for video data"
    assert embedding_type == "t5_xxl", f"Unsupported embeddings type ({embedding_type}) for video data"
    video_text_transform = get_video_text_transform(
        caption_type=caption_type,
        embedding_type=embedding_type,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
    )

    return {
        "merge_datadict": L(merge_datadict.DataDictMerger)(
            input_keys=["video"],
            output_keys=[
                "video",
                "fps",
                "num_frames",
                "chunk_index",
                "frame_start",
                "frame_end",
                "n_orig_video_frames",
            ],
        ),
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "text_transform": video_text_transform,
    }


@augmentor_register("video_basic_augmentor_v2")
def get_video_augmentor_v2(
    resolution: str,
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: Optional[str] = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    num_video_frames: int = -1,
    use_native_fps: bool = True,
    use_original_fps: bool = False,
    use_random_consecutive_frames: bool = False,
    use_random_interleaved_frames: bool = False,
    prefer_crop_over_pad: bool = False,
):
    """
    num_video_frames: -1 means use all frames, otherwise use the number of frames specified.

    Video augmentor V2. It works with a naive video decoder ("video_naive_bytes") that does nothing.
    Augmentors here include:
    - a basic video decoder that fetches frames within a window and delegates further subsampling or duplication to the modeling code to produce videos with the required number of frames.
    - resize the video
    - add reflection padding
    - extract captions and embeddings.

    When use_random_consecutive_frames is True, the augmentor will sample random consecutive frames, preserving the original fps.
    When use_random_interleaved_frames is True, the augmentor will sample random interleaved frames, making fractional fps interpolation possible (e.g. 24->30fps).

    Supported caption_type include t2w_qwen2p5_7b and i2w_qwen2p5_7b_later_frames.
    Supported embedding_type include t5_xxl and umt5_xxl.
    """
    video_text_transform = get_video_text_transform(
        caption_type=caption_type,
        embedding_type=embedding_type,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
    )
    if caption_type.startswith("t2w_qwen"):
        key_for_caption = "t2w_windows"
    elif caption_type.startswith("i2w_qwen"):
        key_for_caption = "i2w_windows_later_frames"
    else:
        f"Unsupported caption type ({caption_type}) for video data"
    if embedding_type is not None:
        assert embedding_type in (
            "t5_xxl",
            "umt5_xxl",
        ), f"Unsupported embeddings type ({embedding_type}) for video data"

    resize_op = (
        {
            "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
                input_keys=["video"],
                args={"size": VIDEO_RES_SIZE_INFO[resolution]},
            )  # Original version, prefer padding over cropping
        }
        if not prefer_crop_over_pad
        else {
            "resize_smallest_side_aspect_ratio_preserving": L(resize.ResizeSmallestSideAspectPreserving)(
                input_keys=["video"],
                args={"size": VIDEO_RES_SIZE_INFO[resolution]},
            ),  # New version, prefer cropping over padding
        }
    )
    return {
        "video_parsing": L(video_parsing.VideoParsing)(
            input_keys=["metas", "video"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 4,
                "num_video_frames": num_video_frames,
                "use_native_fps": use_native_fps,
                "use_original_fps": use_original_fps,
                # use_random_consecutive_frames:
                #   If True, samples random consecutive frames within the window, preserving the original fps between frames.
                #   This enables consecutive clips from the source, without evenly-spaced dropping/duplication.
                "use_random_consecutive_frames": use_random_consecutive_frames,
                # use_random_interleaved_frames:
                #   If True, enables random interleaved frame subsampling (e.g., for fractional fps upsampling/downsampling such as 24->30 FPS).
                #   Produces non-consecutive, randomly-traced clip segments by mixing different strides, for more varied temporal sampling.
                "use_random_interleaved_frames": use_random_interleaved_frames,
            },
        ),
        "merge_datadict": L(merge_datadict.DataDictMerger)(
            input_keys=["video"],
            output_keys=[
                "video",
                "fps",
                "num_frames",
                "chunk_index",
                "frame_start",
                "frame_end",
                "n_orig_video_frames",
            ],
        ),
        **resize_op,
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "text_transform": video_text_transform,
    }


@augmentor_register("noframedrop_nocameramove_video_augmentor_v1")
def get_noframedrop_nocameramove_video_augmentor_v1(
    resolution: str,
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: Optional[str] = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    num_video_frames: int = -1,
    use_native_fps: bool = True,
    use_original_fps: bool = False,
    use_random_consecutive_frames: bool = False,
):
    """
    This augmentor is v2 + the following:
    - no frame drop by ensure num_multipler is always 1
    - no camera move (indiciated by the camera related bad words in the caption)
    """
    video_text_transform = get_video_text_transform(
        caption_type=caption_type,
        embedding_type=embedding_type,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
    )
    if caption_type.startswith("t2w_qwen"):
        key_for_caption = "t2w_windows"
    elif caption_type.startswith("i2w_qwen"):
        key_for_caption = "i2w_windows_later_frames"
    else:
        f"Unsupported caption type ({caption_type}) for video data"
    if embedding_type is not None:
        assert embedding_type in (
            "t5_xxl",
            "umt5_xxl",
        ), f"Unsupported embeddings type ({embedding_type}) for video data"

    contain_keyword = False  # ensure no camera move
    augmentations = {
        "video_parsing": L(video_parsing.VideoParsing)(
            input_keys=["metas", "video"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 4,
                "num_video_frames": num_video_frames,
                "use_native_fps": use_native_fps,
                "use_original_fps": use_original_fps,
                "use_random_consecutive_frames": use_random_consecutive_frames,
                # Both use_original_fps=True and "allowed_num_multiplers": [1] prevent frame dropping.
                # Key differences:
                # - use_original_fps=True: Hard-codes num_multiplier=1 and ignores allowed_num_multiplers setting.
                #   Won't skip entire videos, but may discard head/tail frames, potentially causing
                #   video-caption misalignment.
                # - "allowed_num_multiplers": [1]: Uses the multiplier system but restricts it to 1x only. May skip videos, causing slower dataloader
                "allowed_num_multiplers": [1],
            },
        ),
        "merge_datadict": L(merge_datadict.DataDictMerger)(
            input_keys=["video"],
            output_keys=[
                "video",
                "fps",
                "num_frames",
                "chunk_index",
                "frame_start",
                "frame_end",
                "n_orig_video_frames",
            ],
        ),
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "text_transform": video_text_transform,
        "caption_filter": L(caption_filter.CaptionFilter)(
            input_keys=["ai_caption"],  # Works with ai_caption from TextTransformForVideo
            args={
                "keywords": CAMERA_MOVEMENT_PHRASES,
                "contain_keyword": contain_keyword,
                "log_filtered": False,  # Enable logging to see what gets filtered
                "filter_stats": True,
                # For 4k and physics AI datasets, even if this has camera movement, it is still good
                "dont_apply_on_webdataset_names": [
                    "4k_",
                    "a2d2_",
                    "agibot_",
                    "alpamayo_",
                    "bridgev2p1_",
                    "droid_",
                    "gr00t_",
                    "nexar",
                    "onex",
                    "openx",
                    "physical-ai-special",
                    "physics-cosmos-db",
                    "wisa",
                    "robomind",
                    "smartspace_",
                ],
            },
        ),
    }
    mode_str = "contain" if contain_keyword else "exclude"
    log.info(
        f"[video] noframedrop_nocameramove_video_augmentor_v1: Added caption filter in '{mode_str}' mode "
        f"with {len(CAMERA_MOVEMENT_PHRASES)} camera movement phrases"
    )
    return augmentations


@augmentor_register("nocameramove_video_augmentor_v1")
def get_nocameramove_video_augmentor_v1(
    resolution: str,
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: Optional[str] = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    num_video_frames: int = -1,
    use_native_fps: bool = True,
    use_original_fps: bool = False,
    use_random_consecutive_frames: bool = False,
):
    """
    This augmentor is based on noframedrop_nocameramove_video_augmentor_v1 but:
    - allows limited frame drop by setting allowed_num_multiplers to [1,2]
    - no camera move (indicated by the camera related bad words in the caption)
    """
    # Get the base augmentations from the no-frame-drop version
    augmentations = get_noframedrop_nocameramove_video_augmentor_v1(
        resolution=resolution,
        caption_type=caption_type,
        embedding_type=embedding_type,
        min_fps=min_fps,
        max_fps=max_fps,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
        num_video_frames=num_video_frames,
        use_native_fps=use_native_fps,
        use_original_fps=use_original_fps,
        use_random_consecutive_frames=use_random_consecutive_frames,
    )

    # Modify only the allowed_num_multiplers parameter
    augmentations["video_parsing"].args["allowed_num_multiplers"] = [1, 2]

    log.info(
        "[video] nocameramove_video_augmentor_v1: Modified allowed_num_multiplers to [1, 2] "
        "for limited frame dropping capability"
    )
    return augmentations


@augmentor_register("image_basic_augmentor")
def get_image_augmentor(
    resolution: str,
    caption_type: str = "ai_v3p1",
    embedding_type: str = "t5_xxl",
):
    augmentation = {
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["images"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["images"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "normalize": L(normalize.Normalize)(
            input_keys=["images"],
            args={"mean": 0.5, "std": 0.5},
        ),
        "text_transform": L(text_transforms_for_image.TextTransformForImage)(
            input_keys=[],
            args={
                "caption_type": caption_type,
                "embedding_type": embedding_type,
                "weight_captions_gt": 0.05,
                "caption_probs": {"ground_truth": 0.05, "vfc_fidelity": 0.95},
                "t5_tokens": {"num": 512, "dim": 1024},
                "is_mask_all_ones": True,
            },
        ),
        "append_fps_frames": L(append_fps_frames_for_image.AppendFPSFramesForImage)(),
    }

    return augmentation


@augmentor_register("image_basic_augmentor_without_embeddings")
def get_image_augmentor_without_embeddings(
    resolution: str,
    caption_type: str = "ai_v3p1",
    embedding_type: Optional[str] = None,
):
    augmentation = {
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["images"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["images"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "normalize": L(normalize.Normalize)(
            input_keys=["images"],
            args={"mean": 0.5, "std": 0.5},
        ),
        "text_transform": L(text_transforms_for_image.TextTransformForImageWithoutEmbeddings)(
            input_keys=[],
            args={
                "caption_type": caption_type,
            },
        ),
        "append_fps_frames": L(append_fps_frames_for_image.AppendFPSFramesForImage)(),
    }

    return augmentation
