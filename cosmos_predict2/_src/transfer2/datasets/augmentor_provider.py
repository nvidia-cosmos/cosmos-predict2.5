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
import cosmos_predict2._src.predict2.datasets.augmentors.text_transforms_for_image as text_transforms_for_image
import cosmos_predict2._src.predict2.datasets.augmentors.text_transforms_for_video as text_transforms_for_video
import cosmos_predict2._src.transfer2.datasets.augmentors.control_input as control_input
import cosmos_predict2._src.transfer2.datasets.augmentors.merge_datadict as merge_datadict
import cosmos_predict2._src.transfer2.datasets.augmentors.video_parsing as video_parsing
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO

AUGMENTOR_OPTIONS = {}


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
    if caption_type == "vila_caption":
        video_text_transform = L(text_transforms_for_video.TextTransformForVideo)(
            input_keys=[],
            args={
                "captions_key": "metas",
                "embeddings_key": embedding_type,
                "caption_windows_key": "windows",
                "caption_type": "vila_caption",
                "embedding_caption_type": "vila_caption",
                "t5_tokens": {"num": 512},
                "is_mask_all_ones": True,
                "caption_probs": {
                    "long": 1,
                    "medium": 0,
                    "short": 0,
                    "user": 0,
                },
            },
        )
    elif caption_type == "t2w_qwen2p5_7b":
        log.info(
            f"caption_type: {caption_type}, long_caption_ratio: {long_caption_ratio}, medium_caption_ratio: {medium_caption_ratio}, short_caption_ratio: {short_caption_ratio}, user_caption_ratio: {user_caption_ratio}"
        )
        video_text_transform = L(text_transforms_for_video.TextTransformForVideo)(
            input_keys=[],
            args={
                "captions_key": "metas",
                "embeddings_key": embedding_type,
                "caption_windows_key": "t2w_windows",
                "caption_type": "qwen2p5_7b_caption",
                "embedding_caption_type": "t2w_qwen2p5_7b",
                "t5_tokens": {"num": 512},
                "is_mask_all_ones": True,
                "caption_probs": {
                    "long": long_caption_ratio,
                    "medium": medium_caption_ratio,
                    "short": short_caption_ratio,
                    "user": user_caption_ratio,
                },
            },
        )
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        video_text_transform = L(text_transforms_for_video.TextTransformForVideo)(
            input_keys=[],
            args={
                "captions_key": "metas",
                "embeddings_key": embedding_type,
                "caption_windows_key": "i2w_windows_later_frames",
                "caption_type": "qwen2p5_7b_caption",
                "embedding_caption_type": "i2w_qwen2p5_7b_later_frames",
                "t5_tokens": {"num": 512},
                "is_mask_all_ones": True,
                "caption_probs": {
                    "long": long_caption_ratio,
                    "medium": medium_caption_ratio,
                    "short": short_caption_ratio,
                    "user": user_caption_ratio,
                },
            },
        )
    else:
        raise ValueError(f"Unsupported caption type ({caption_type}) for video data")

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
    use_native_fps: bool = False,
):
    """
    num_video_frames: -1 means use all frames, otherwise use the number of frames specified.

    Video augmentor V2. It works with a naive video decoder ("video_naive_bytes") that does nothing.
    Augmentors here include:
    - a basic video decoder that fetches frames within a window and delegates further subsampling or duplication to the modeling code to produce videos with the required number of frames.
    - resize the video
    - add reflection padding
    - extract captions and embeddings.

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
    if caption_type == "t2w_qwen2p5_7b":
        key_for_caption = "t2w_windows"
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        key_for_caption = "i2w_windows_later_frames"
    else:
        f"Unsupported caption type ({caption_type}) for video data"
    if embedding_type is not None:
        assert embedding_type in (
            "t5_xxl",
            "umt5_xxl",
        ), f"Unsupported embeddings type ({embedding_type}) for video data"

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
    }


@augmentor_register("image_basic_augmentor")
def get_image_augmentor(
    resolution: str,
    caption_type: str = "ai_v3p1",
    embedding_type: Optional[str] = "t5_xxl",
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


@augmentor_register("video_basic_augmentor_v1_with_control")
def get_video_augmentor_v1_with_control(
    resolution: str,
    caption_type: str = "vila_caption",
    embedding_type: str = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    num_video_frames: int = -1,
    use_native_fps: bool = False,
    control_input_type: str = "edge_vis_depth_seg",
    use_control_mask_prob: float = 0.0,
    num_control_inputs_prob: list[float] = [1.0, 0.0, 0.0, 0.0],
    edge_t_lower: Optional[int] = None,
    edge_t_upper: Optional[int] = None,
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
        "add_control_input": L(control_input.AddControlInputComb)(
            input_keys=["video"],
            use_random=True,
            control_input_type=control_input_type,
            use_control_mask_prob=use_control_mask_prob,
            num_control_inputs_prob=num_control_inputs_prob,
            edge_t_lower=edge_t_lower,
            edge_t_upper=edge_t_upper,
        ),
    }


@augmentor_register("video_basic_augmentor_v2_with_control")
def get_video_augmentor_v2_with_control(
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
    use_native_fps: bool = False,
    control_input_type: str = "edge_vis_depth_seg",
    use_control_mask_prob: float = 0.0,
    num_control_inputs_prob: list[float] = [1.0, 0.0, 0.0, 0.0],
    edge_t_lower: Optional[int] = None,
    edge_t_upper: Optional[int] = None,
    **kwargs,
):
    """Video augmentor V2. It works with a naive video decoder ("video_naive_bytes") that does nothing.
    Augmentors here include:
    - a basic video decoder that fetches frames within a window and delegates further subsampling or duplication to the modeling code to produce videos with the required number of frames.
    - resize the video
    - add reflection padding
    - extract captions and embeddings.
    - add control input. If the control input is contained in the dataset, will direclty load it. Otherwise, will parse the input_hint_key name and load the corresponding processor to extract
    the control input from the input video (e.g. applying Canny edge detector / blur). Supports loading multiple control input hint keys.


    Supported caption_type include t2w_qwen2p5_7b and i2w_qwen2p5_7b_later_frames.
    Supported embedding_type include t5_xxl and umt5_xxl.
    """
    video_text_transform = get_video_text_transform(caption_type=caption_type, embedding_type=embedding_type)
    if caption_type == "t2w_qwen2p5_7b":
        key_for_caption = "t2w_windows"
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        key_for_caption = "i2w_windows_later_frames"
    else:
        f"Unsupported caption type ({caption_type}) for video data"
    if (
        embedding_type is not None
    ):  # If None, means we use CosmosReason1 embedding which is online-computed, no need to load
        assert embedding_type in (
            "t5_xxl",
            "umt5_xxl",
        ), f"Unsupported embeddings type ({embedding_type}) for video data"

    augmentor_dict = {
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
                "resolution": resolution,
            },
        ),
    }
    if "depth" in control_input_type:
        augmentor_dict["depth_parsing"] = L(video_parsing.VideoParsing)(
            input_keys=["metas", "depth_pervideo_video_depth_anything"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 4,
                "num_video_frames": num_video_frames,
                "use_native_fps": use_native_fps,
                "resolution": resolution,
            },
        )
    if "segcolor" in control_input_type:
        augmentor_dict["seg_parsing"] = L(video_parsing.VideoParsing)(
            input_keys=["metas", "segmentation_sam2_color_video_v2"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 4,
                "num_video_frames": num_video_frames,
                "use_native_fps": use_native_fps,
                "resolution": resolution,
            },
        )
    input_keys = ["video"]
    output_keys = ["video"]
    attr_keys = [
        "video",
        "fps",
        "num_frames",
        "chunk_index",
        "frame_start",
        "frame_end",
        "frame_indices",
        "n_orig_video_frames",
    ]
    if "depth" in control_input_type:
        input_keys.append("depth_pervideo_video_depth_anything")
        output_keys.append("depth")
    if "segcolor" in control_input_type:
        input_keys.append("segmentation_sam2_color_video_v2")
        output_keys.append("segmentation")
    elif "seg" in control_input_type:
        input_keys.append("segmentation_sam2")
        output_keys.append("segmentation")
    augmentor_dict.update(
        {
            "merge_datadict": L(merge_datadict.DataDictMerger)(
                input_keys=input_keys,
                output_keys=output_keys + attr_keys,
            ),
            "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
                input_keys=output_keys,
                args={"size": VIDEO_RES_SIZE_INFO[resolution]},
            ),
            "reflection_padding": L(padding.ReflectionPadding)(
                input_keys=output_keys,
                args={"size": VIDEO_RES_SIZE_INFO[resolution]},
            ),
            "text_transform": video_text_transform,
            "add_control_input": L(control_input.AddControlInputComb)(
                input_keys=["video"],
                use_random=True,
                control_input_type=control_input_type,
                use_control_mask_prob=use_control_mask_prob,
                num_control_inputs_prob=num_control_inputs_prob,
                edge_t_lower=edge_t_lower,
                edge_t_upper=edge_t_upper,
            ),
        }
    )
    return augmentor_dict


@augmentor_register("image_basic_augmentor_with_control")
def get_image_augmentor_with_control(
    resolution: str,
    caption_type: str = "ai_v3p1",
    embedding_type: Optional[str] = "t5_xxl",
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
        "append_fps_frames": L(append_fps_frames_for_image.AppendFPSFramesForImage)(),
        "add_control_input": L(control_input.AddControlInputComb)(
            input_keys=["images"],
            args={
                "use_random": True,
                "preset_strength": "medium",
            },
        ),
    }
    return augmentation


@augmentor_register("video_basic_augmentor_v2_with_control_and_image_context")
def get_video_augmentor_v2_with_control_and_image_context(
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
    use_native_fps: bool = False,
    control_input_type: str = "edge_vis_depth_seg",
    use_control_mask_prob: float = 0.0,
    num_control_inputs_prob: list[float] = [1.0, 0.0, 0.0, 0.0],
    **kwargs,
):
    """Video augmentor V2. It works with a naive video decoder ("video_naive_bytes") that does nothing.
    Augmentors here include:
    - a basic video decoder that fetches frames within a window and delegates further subsampling or duplication to the modeling code to produce videos with the required number of frames.
    - resize the video
    - add reflection padding
    - extract captions and embeddings.
    - add control input. If the control input is contained in the dataset, will direclty load it. Otherwise, will parse the input_hint_key name and load the corresponding processor to extract
    the control input from the input video (e.g. applying Canny edge detector / blur). Supports loading multiple control input hint keys.


    Supported caption_type include t2w_qwen2p5_7b and i2w_qwen2p5_7b_later_frames.
    Supported embedding_type include t5_xxl and umt5_xxl.
    """
    video_text_transform = get_video_text_transform(caption_type=caption_type, embedding_type=embedding_type)
    if caption_type == "t2w_qwen2p5_7b":
        key_for_caption = "t2w_windows"
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        key_for_caption = "i2w_windows_later_frames"
    else:
        f"Unsupported caption type ({caption_type}) for video data"
    if embedding_type is not None:
        assert embedding_type in (
            "t5_xxl",
            "umt5_xxl",
        ), f"Unsupported embeddings type ({embedding_type}) for video data"

    augmentor_dict = {
        "video_parsing_with_image_context": L(video_parsing.VideoParsingWithImageContext)(
            input_keys=["metas", "video"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 4,
                "num_video_frames": num_video_frames,
                "use_native_fps": use_native_fps,
                "resolution": resolution,
            },
        ),
    }
    if "depth" in control_input_type:
        augmentor_dict["depth_parsing"] = L(video_parsing.VideoParsing)(
            input_keys=["metas", "depth_pervideo_video_depth_anything"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 4,
                "num_video_frames": num_video_frames,
                "use_native_fps": use_native_fps,
                "resolution": resolution,
            },
        )
    if "segcolor" in control_input_type:
        augmentor_dict["seg_parsing"] = L(video_parsing.VideoParsing)(
            input_keys=["metas", "segmentation_sam2_color_video_v2"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 4,
                "num_video_frames": num_video_frames,
                "use_native_fps": use_native_fps,
                "resolution": resolution,
            },
        )
    input_keys = ["video"]
    output_keys = ["video"]
    attr_keys = [
        "fps",
        "num_frames",
        "chunk_index",
        "frame_start",
        "frame_end",
        "frame_indices",
        "n_orig_video_frames",
        "image_context",
    ]
    if "depth" in control_input_type:
        input_keys.append("depth_pervideo_video_depth_anything")
        output_keys.append("depth")
    if "segcolor" in control_input_type:
        input_keys.append("segmentation_sam2_color_video_v2")
        output_keys.append("segmentation")
    elif "seg" in control_input_type:
        input_keys.append("segmentation_sam2")
        output_keys.append("segmentation")
    augmentor_dict.update(
        {
            "merge_datadict": L(merge_datadict.DataDictMerger)(
                input_keys=input_keys,
                output_keys=output_keys + attr_keys,
            ),
            "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
                input_keys=output_keys,
                args={"size": VIDEO_RES_SIZE_INFO[resolution]},
            ),
            "reflection_padding": L(padding.ReflectionPadding)(
                input_keys=output_keys,
                args={"size": VIDEO_RES_SIZE_INFO[resolution]},
            ),
            "text_transform": video_text_transform,
            "add_control_input": L(control_input.AddControlInputComb)(
                input_keys=["video"],
                use_random=True,
                control_input_type=control_input_type,
                use_control_mask_prob=use_control_mask_prob,
                num_control_inputs_prob=num_control_inputs_prob,
            ),
            "resize_largest_side_aspect_ratio_preserving_image_context": L(resize.ResizeLargestSideAspectPreserving)(
                input_keys=["image_context"],
                args={"size": IMAGE_RES_SIZE_INFO[resolution]},
            ),
            "reflection_padding_image_context": L(padding.ReflectionPadding)(
                input_keys=["image_context"],
                args={"size": IMAGE_RES_SIZE_INFO[resolution]},
            ),
            "normalize_image_context": L(normalize.Normalize)(
                input_keys=["image_context"],
                args={"mean": 0.5, "std": 0.5},
            ),
        }
    )
    return augmentor_dict


@augmentor_register("video_basic_augmentor_with_control_input")
def get_video_augmentor_with_control_input(
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
    use_native_fps: bool = False,
    **kwargs,
):
    """Video augmentor V2. It works with a naive video decoder ("video_naive_bytes") that does nothing.
    Augmentors here include:
    - a basic video decoder that fetches frames within a window and delegates further subsampling or duplication to the modeling code to produce videos with the required number of frames.
    - resize the video
    - add reflection padding
    - extract captions and embeddings.
    - add control input. If the control input is contained in the dataset, will direclty load it. Otherwise, will parse the input_hint_key name and load the corresponding processor to extract
    the control input from the input video (e.g. applying Canny edge detector / blur). Supports loading multiple control input hint keys.


    Supported caption_type include t2w_qwen2p5_7b and i2w_qwen2p5_7b_later_frames.
    Supported embedding_type include t5_xxl and umt5_xxl.
    """
    video_text_transform = get_video_text_transform(caption_type=caption_type, embedding_type=embedding_type)
    if caption_type == "t2w_qwen2p5_7b":
        key_for_caption = "t2w_windows"
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        key_for_caption = "i2w_windows_later_frames"
    else:
        f"Unsupported caption type ({caption_type}) for video data"
    if embedding_type is not None:
        assert embedding_type in (
            "t5_xxl",
            "umt5_xxl",
        ), f"Unsupported embeddings type ({embedding_type}) for video data"

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
            },
        ),
        "cond_parsing": L(video_parsing.VideoParsing)(
            input_keys=["metas", "hdmap_bbox"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 4,
                "num_video_frames": num_video_frames,
                "use_native_fps": use_native_fps,
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
        "merge_cond": L(merge_datadict.DataDictRewriter)(
            input_keys=["video"],
            output_keys=[
                "hdmap_bbox",
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
        "resize_largest_side_aspect_ratio_preserving_cond": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["hdmap_bbox"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding_cond": L(padding.ReflectionPadding)(
            input_keys=["hdmap_bbox"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "text_transform": video_text_transform,
        "add_control_input": L(control_input.AddControlInputHdmapBbox)(
            input_keys=["hdmap_bbox"], output_keys=["control_input_hdmap_bbox"], use_random=False
        ),
    }


@augmentor_register("hdmap_augmentor_for_local_datasets")
def get_hdmap_augmentor_for_local_datasets(
    resolution: str,
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: str = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    num_video_frames: int = -1,
    use_native_fps: bool = False,
    control_input_type: str = "edge_vis_depth_seg",
    use_control_mask_prob: float = 0.0,
    num_control_inputs_prob: list[float] = [1.0, 0.0, 0.0, 0.0],
    **kwargs,
):
    """Video augmentor for local datasets (not s3 webdatasets) for specifically control input related augmentation.
    Augmentors here include:
    - resize the video
    - add reflection padding
    - add control input. If the control input is contained in the dataset, will directly load it. Otherwise, will parse the input_hint_key name and load the corresponding processor to extract
    the control input from the input video (e.g. applying Canny edge detector / blur). Supports loading multiple control input hint keys.
    """
    return {
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "resize_largest_side_aspect_ratio_preserving_cond": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["hdmap_bbox"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding_cond": L(padding.ReflectionPadding)(
            input_keys=["hdmap_bbox"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "add_control_input": L(control_input.AddControlInputHdmapBbox)(
            input_keys=["hdmap_bbox"], output_keys=["control_input_hdmap_bbox"], use_random=False
        ),
    }
