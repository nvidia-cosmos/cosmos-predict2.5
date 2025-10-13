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

import cosmos_predict2._src.imaginaire.datasets.webdataset.augmentors.image.padding as padding
import cosmos_predict2._src.imaginaire.datasets.webdataset.augmentors.image.resize as resize
import cosmos_predict2._src.predict2.datasets.augmentors.merge_datadict as merge_datadict
import cosmos_predict2._src.predict2_multiview.datasets.augmentors.av_multiview_adapter as av_multiview_adapter
import cosmos_predict2._src.predict2_multiview.datasets.augmentors.multiview_video_parsing as video_parsing
import cosmos_predict2._src.predict2_multiview.datasets.augmentors.text_transforms_for_multiview_video as text_transforms_for_multiview_video
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.augmentor_provider import augmentor_register
from cosmos_predict2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_predict2._src.predict2_multiview.configs.vid2vid.defaults.driving import (
    DrivingVideoDataloaderConfig,
    MADSDrivingVideoDataloaderConfig,
)
from cosmos_predict2._src.predict2_multiview.datasets.augmentors import (
    merge_datadict_multiview_with_control,
    singleview_video_parsing,
)


def get_multiview_video_text_transform(
    caption_type: str,
    embedding_type: str | None,
    driving_dataloader_config: DrivingVideoDataloaderConfig,
):
    video_text_transform = L(text_transforms_for_multiview_video.TextTransformForMultiviewVideo)(
        input_keys=[],
        args={
            "driving_dataloader_config": driving_dataloader_config,
            "embedding_type": embedding_type,
            "t5_tokens": {"num": 512},
            "is_mask_all_ones": True,
        },
    )
    return video_text_transform


@augmentor_register("video_basic_augmentor_v2_multiview")
def get_video_augmentor_v2_multiview(
    resolution: str,
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: str = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    driving_dataloader_config: Optional[DrivingVideoDataloaderConfig] = None,
    minimum_start_index: int = 3,
    align_last_view_frames_and_clip_from_front: bool = False,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    use_control_mask_prob: float = 0.0,
    num_control_inputs_prob: list[float] = [1.0, 0.0, 0.0, 0.0],
):
    """Video augmentor V2. It works with a naive video decoder ("video_naive_bytes") that does nothing.
    Augmentors here include:
    - a basic video decoder that fetches frames within a window and delegates further subsampling or duplication to the modeling code to produce videos with the required number of frames.
    - resize the video
    - add reflection padding
    - extract captions and embeddings.

    Supported caption_type include t2w_qwen2p5_7b and i2w_qwen2p5_7b_later_frames.
    Supported embedding_type include t5_xxl and umt5_xxl.
    """
    video_text_transform = get_multiview_video_text_transform(
        driving_dataloader_config=driving_dataloader_config,
        caption_type=caption_type,
        embedding_type=embedding_type,
    )
    if caption_type == "t2w_qwen2p5_7b":
        key_for_caption = "t2w_windows"
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        key_for_caption = "i2w_windows_later_frames"
    else:
        f"Unsupported caption type ({caption_type}) for video data"
    assert embedding_type in (
        "t5_xxl",
        "umt5_xxl",
        None,
    ), f"Unsupported embeddings type ({embedding_type}) for video data"

    augmentor_config = {
        "video_parsing": L(video_parsing.MultiViewVideoParsing)(
            input_keys=[],
            args={
                "driving_dataloader_config": driving_dataloader_config,
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "minimum_start_index": minimum_start_index,
                "align_last_view_frames_and_clip_from_front": align_last_view_frames_and_clip_from_front,
                "video_decode_num_threads": 8,
            },
        ),
        "merge_datadict": L(merge_datadict.DataDictMerger)(
            input_keys=["video"],
            output_keys=[
                "video",
                "fps",
                "view_t5_embeddings",
                "view_captions",
                "camera_keys_selection",
                "view_indices_selection",
                "n_orig_video_frames_per_view",
                "aspect_ratio",
                "num_video_frames_per_view",
                "control_weight",
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
    hint_keys = (
        driving_dataloader_config.hint_keys.split("_") if hasattr(driving_dataloader_config, "hint_keys") else []
    )
    for hint_key in hint_keys:
        if hint_key == "":
            continue
        if hint_key == "edge2x":
            from projects.cosmos.transfer2_multiview.datasets.augmentors.control_input import (
                AddControlInputEdgeDownUp2X,
            )

            augmentor_config["add_control_input_edge_down_up_2x"] = L(AddControlInputEdgeDownUp2X)(
                input_keys=["video"],
                use_random=False,
                use_control_mask_prob=use_control_mask_prob,
                num_control_inputs_prob=num_control_inputs_prob,
            )
        else:
            raise ValueError(f"Unsupported hint key: {hint_key}")
    return augmentor_config


def get_video_augmentor_v2_multiview_local(
    resolution: str,
):
    """Video augmentor V2. It works with a naive video decoder ("video_naive_bytes") that does nothing.
    Augmentors here include:
    - a basic video decoder that fetches frames within a window and delegates further subsampling or duplication to the modeling code to produce videos with the required number of frames.
    - resize the video
    - add reflection padding
    """

    augmentor_config = {
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
    }
    return augmentor_config


def get_video_augmentor_v2_multiview_no_text_emb(
    resolution: str,
    caption_type: str = "t2w_qwen2p5_7b",
    min_fps: int = 10,
    max_fps: int = 60,
    driving_dataloader_config: Optional[MADSDrivingVideoDataloaderConfig] = None,
    num_video_frames: int = -1,
    use_native_fps: bool = False,
    use_original_fps: bool = False,
):
    """Multi-video augmentor that processes video_0 through video_6, t5_xxl_0 through t5_xxl_6,
    and hdmap_bbox_0 through hdmap_bbox_6, then concatenates them together.

    This augmentor handles:
    - Multiple video inputs (video_0 to video_6)
    - Multiple t5_xxl embeddings (t5_xxl_0 to t5_xxl_6)
    - Multiple hdmap_bbox inputs (hdmap_bbox_0 to hdmap_bbox_6)
    - Concatenation of all processed inputs
    """
    n_views = driving_dataloader_config.n_views

    # Use custom multi-video text transform that handles multiple metas and t5_xxl embeddings
    if caption_type == "t2w_qwen2p5_7b":
        key_for_caption = "t2w_windows"
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        key_for_caption = "i2w_windows_later_frames"
    else:
        raise ValueError(f"Unsupported caption type ({caption_type}) for video data")

    # Create the augmentor configuration
    augmentor_config = {}

    # Add video parsing for each video_i
    for i in range(n_views):  # video_0 to video_6
        augmentor_config[f"video_parsing_{i}"] = L(singleview_video_parsing.SingleViewVideoParsing)(
            input_keys=["metas", f"video_{i}"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 8,
                "num_video_frames": num_video_frames,
                "use_native_fps": use_native_fps,
                "use_original_fps": use_original_fps,
                "driving_dataloader_config": driving_dataloader_config,
            },
        )

    # Concatenate video
    video_keys = [f"video_{i}" for i in range(n_views)]

    augmentor_config["concat_datadict"] = L(merge_datadict_multiview_with_control.DataDictConcatenator)(
        input_keys=video_keys,
        output_keys=[
            "video",  # Concatenated videos
        ],
        args={
            "concat_dim": 1,  # Concatenate along temporal dimension
        },
    )

    # Resize and padding for concatenated videos
    augmentor_config["resize_largest_side_aspect_ratio_preserving"] = L(resize.ResizeLargestSideAspectPreserving)(
        input_keys=["video"],
        args={"size": VIDEO_RES_SIZE_INFO[resolution]},
    )

    augmentor_config["reflection_padding"] = L(padding.ReflectionPadding)(
        input_keys=["video"],
        args={"size": VIDEO_RES_SIZE_INFO[resolution]},
    )

    augmentor_config["repackage_multiview"] = L(av_multiview_adapter.AVMultiviewAdapter)(
        input_keys=[],
        args={
            "driving_dataloader_config": driving_dataloader_config,
            "embedding_type": None,
        },
    )

    return augmentor_config
