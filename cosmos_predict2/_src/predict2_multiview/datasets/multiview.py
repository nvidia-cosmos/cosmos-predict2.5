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

"""
Webloaders of datasets and augmentations for visual-text multiview dataset for AV
"""

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False
import io
import random
from typing import Any, Final, Literal, Optional

import attrs
import torch
import webdataset as wds
from einops import rearrange
from torchvision.transforms import InterpolationMode, Resize

import cosmos_predict2._src.predict2.datasets.distributor.parallel_sync_multi_aspect_ratio as parallel_sync_multi_aspect_ratio
from cosmos_predict2._src.imaginaire.datasets.decoders.json_loader import json_decoder
from cosmos_predict2._src.imaginaire.datasets.decoders.video_decoder import video_naive_bytes
from cosmos_predict2._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos_predict2._src.imaginaire.datasets.webdataset.config.schema import DatasetConfig
from cosmos_predict2._src.imaginaire.datasets.webdataset.distributors import ShardlistBasic
from cosmos_predict2._src.imaginaire.datasets.webdataset.webdataset_ext import Dataset
from cosmos_predict2._src.predict2.datasets.cached_replay_dataloader import get_cached_replay_dataloader
from cosmos_predict2._src.predict2_multiview.datasets.wdinfo_utils import DEFAULT_CATALOG, get_video_dataset_info

DEFAULT_CAMERAS: Final = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
)

DEFAULT_CAMERA_VIEW_MAPPING: Final = dict(zip(DEFAULT_CAMERAS, range(len(DEFAULT_CAMERAS))))

DEFAULT_CAPTION_PREFIXES: Final = {
    "camera_front_wide_120fov": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "camera_cross_right_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
    "camera_rear_right_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
    "camera_rear_tele_30fov": "The video is captured from a camera mounted on a car. The camera is facing backwards.",
    "camera_rear_left_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
    "camera_cross_left_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
    "camera_front_tele_30fov": "The video is captured from a telephoto camera mounted on a car. The camera is facing forward.",
}

DEFAULT_CAPTION_KEY_MAPPING: Final = dict(
    zip(DEFAULT_CAMERAS, [f"metas_{k}_10s_chunks_qwen2p5_vl_32b" for k in DEFAULT_CAMERAS])
)
DEFAULT_VIDEO_KEY_MAPPING: Final = dict(zip(DEFAULT_CAMERAS, [f"video_{k}" for k in DEFAULT_CAMERAS]))


class UnpackMetas(Augmentor):
    """Unpack metas from single meta dicts list into per-camera meta dicts."""

    def __init__(
        self, position_to_camera_mapping: dict[int, str], input_key: str = "metas", output_prefix: str = "metas_"
    ) -> None:
        super().__init__([], {})
        self.position_to_camera_mapping = position_to_camera_mapping
        self.input_key = input_key
        self.output_prefix = output_prefix

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        metas = data.pop(self.input_key)
        for i, meta in enumerate(metas):
            camera_name = self.position_to_camera_mapping[i]
            data[f"{self.output_prefix}{camera_name}"] = meta
        return data


class ExtractFramesAndCaptions(Augmentor):
    """Extract frames from a videos."""

    def __init__(
        self,
        camera_order: list[str],
        num_frames: int,
        resolution_hw: tuple[int, int],
        fps_downsample_factor: int,
        caption_probability: dict[str, float],
        camera_view_mapping: dict[str, int],
        camera_caption_key_mapping: dict[str, str],
        camera_video_key_mapping: dict[str, str],
        camera_control_key_mapping: Optional[dict[str, str]] = None,
        add_view_prefix_to_caption: bool = False,
        camera_prefix_mapping: Optional[dict[str, str]] = None,
        single_caption_camera_name: Optional[str] = None,
    ) -> None:
        """Extracts frames and captions from video/metadata dicts.

        Args:
            camera_order: Order of cameras to extract
            num_frames: Number of frames to extract
            resolution_hw: Resolution of the extracted frames
            fps_downsample_factor: FPS downsample factor
            caption_probability: Probability of each caption type in t2w window
            camera_view_mapping: Mapping of camera keys to view indices
            camera_caption_key_mapping: Mapping of camera keys to caption keys
            camera_video_key_mapping: Mapping of camera keys to video keys
            camera_control_key_mapping: Mapping of camera keys to control keys
            add_view_prefix_to_caption: Whether to add caption prefix for all views
            camera_prefix_mapping: Mapping of camera keys to prefixes
            single_caption_camera_name: Name of the camera key to use for single caption conditioning.
                If `add_view_prefix_to_caption` is True, will still provide prefixes for other views.

        Returns:
            data: Dictionary with resized tensors of frames and captions
        """
        super().__init__([], {})
        self.camera_order = camera_order
        self.num_frames = num_frames
        self.resolution_hw = resolution_hw
        self.fps_downsample_factor = fps_downsample_factor
        self.caption_probability = caption_probability
        self.camera_view_mapping = camera_view_mapping
        self.camera_caption_key_mapping = camera_caption_key_mapping
        self.camera_video_key_mapping = camera_video_key_mapping
        self.camera_control_key_mapping = camera_control_key_mapping
        self.add_view_prefix_to_caption = add_view_prefix_to_caption
        self.camera_prefix_mapping = camera_prefix_mapping
        self.single_caption_camera_name = single_caption_camera_name

        if self.add_view_prefix_to_caption and self.camera_prefix_mapping is None:
            raise ValueError("camera_prefix_mapping is required when add_view_prefix_to_caption is True")

        if set(self.camera_caption_key_mapping.keys()) != set(self.camera_video_key_mapping.keys()):
            raise ValueError(
                f"Mismatching keys {set(self.camera_caption_key_mapping.keys())} != {set(self.camera_video_key_mapping.keys())}"
            )
        if self.camera_control_key_mapping is not None:
            if set(self.camera_control_key_mapping.keys()) != set(self.camera_caption_key_mapping.keys()):
                raise ValueError(
                    f"Mismatching keys {set(self.camera_control_key_mapping.keys())} != {set(self.camera_caption_key_mapping.keys())}"
                )
        for camera_name in self.camera_caption_key_mapping.keys():
            if camera_name not in self.camera_view_mapping:
                raise ValueError(f"Camera name {camera_name} not found in camera view mapping")
        if self.single_caption_camera_name and self.single_caption_camera_name not in self.camera_order:
            raise ValueError(
                f"Single caption camera name {self.single_caption_camera_name} must appear in selected cameras"
            )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract frames from a video."""

        chunk_index, extracted_frame_ids, video_fps = None, None, None
        captions, multiview_frames, multiview_control, view_indices, view_indices_selection, camera_keys_selection = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for camera_name in self.camera_order:
            video_key = self.camera_video_key_mapping[camera_name]
            if self.single_caption_camera_name:
                meta_key = self.camera_caption_key_mapping[self.single_caption_camera_name]
            else:
                meta_key = self.camera_caption_key_mapping[camera_name]

            t2w_windows = data[meta_key]["t2w_windows"]
            if chunk_index is None:
                chunk_index = random.choice(list(range(len(t2w_windows))))
            window = t2w_windows[chunk_index]

            # extract caption
            choices = list(self.caption_probability.keys())
            weights = list(self.caption_probability.values())
            caption_style = random.choices(choices, weights=weights)[0]
            caption = ""
            if not (
                self.single_caption_camera_name
                and camera_name != self.single_caption_camera_name
                and not self.add_view_prefix_to_caption
            ):
                caption = window[caption_style]

            assert isinstance(caption, str), f"Caption is not a string: {caption}"
            if self.add_view_prefix_to_caption:
                caption = f"{self.camera_prefix_mapping[camera_name]} {caption}"
            captions.append(caption)

            # extract frames
            frame_start = window["start_frame"]
            frame_end = frame_start + self.num_frames * self.fps_downsample_factor
            frame_indices = list(range(frame_start, frame_end, self.fps_downsample_factor))
            frames, original_fps = self.extract_frames(data[video_key], frame_indices, self.resolution_hw)
            assert len(frames) == self.num_frames, f"Expected {self.num_frames} frames, got {len(frames)}"
            multiview_frames.append(frames)

            # check consistency between videos
            if extracted_frame_ids is None:
                extracted_frame_ids = frame_indices
            else:
                if frame_indices != extracted_frame_ids:
                    raise ValueError("Extracted frame IDs do not match")

            if video_fps is None:
                video_fps = original_fps
            else:
                if video_fps != original_fps:
                    raise ValueError("Video FPS does not match")

            # extract control frames if available
            if self.camera_control_key_mapping is not None:
                control_key = self.camera_control_key_mapping[camera_name]
                control_frames, control_fps = self.extract_frames(data[control_key], frame_indices, self.resolution_hw)
                assert len(control_frames) == self.num_frames, (
                    f"Expected {self.num_frames} frames, got {len(control_frames)}"
                )
                assert control_fps == original_fps, f"Control FPS {control_fps} does not match video FPS {original_fps}"
                multiview_control.append(control_frames)

            view_indices.extend([self.camera_view_mapping[camera_name]] * self.num_frames)
            view_indices_selection.append(self.camera_view_mapping[camera_name])
            camera_keys_selection.append(camera_name)

        front_cam_view_idx_sample_position = (
            torch.tensor(self.camera_order.index(self.single_caption_camera_name), dtype=torch.int64)
            if self.single_caption_camera_name
            else None
        )
        if self.single_caption_camera_name and not self.add_view_prefix_to_caption:
            captions = [captions[front_cam_view_idx_sample_position]]

        if video_fps % self.fps_downsample_factor != 0:
            raise ValueError("Original FPS is not divisible by FPS downsample factor")
        fps = video_fps / self.fps_downsample_factor

        sample = {
            "__key__": data["__key__"],
            "__url__": data["__url__"],
            "video": rearrange(torch.cat(multiview_frames, dim=0), "t c h w -> c t h w"),
            "ai_caption": captions,
            "view_indices": torch.tensor(view_indices, dtype=torch.int64),
            "fps": torch.tensor(fps, dtype=torch.float64),
            "chunk_index": torch.tensor(chunk_index, dtype=torch.int64),
            "frame_indices": torch.tensor(extracted_frame_ids, dtype=torch.int64),
            "num_video_frames_per_view": torch.tensor(len(extracted_frame_ids), dtype=torch.int64),
            "view_indices_selection": torch.tensor(view_indices_selection, dtype=torch.int64),
            "camera_keys_selection": camera_keys_selection,
            "sample_n_views": torch.tensor(len(camera_keys_selection), dtype=torch.int64),
            "padding_mask": torch.zeros((1, *self.resolution_hw), dtype=torch.float32),
            "ref_cam_view_idx_sample_position": torch.tensor(-1, dtype=torch.int64),
            "front_cam_view_idx_sample_position": front_cam_view_idx_sample_position,
        }
        if self.camera_control_key_mapping is not None:
            sample["control_input_hdmap_bbox"] = rearrange(torch.cat(multiview_control, dim=0), "t c h w -> c t h w")
        return sample

    @staticmethod
    def extract_frames(
        video: bytes, frame_indices: list[int], resolution_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, float]:
        """Extract frames from a video given start and end frame range."""

        from decord import VideoReader

        video_reader = VideoReader(io.BytesIO(video))
        fps = video_reader.get_avg_fps()
        frames = video_reader.get_batch(frame_indices).asnumpy()
        frames = rearrange(torch.from_numpy(frames), "t h w c -> t c h w")
        return Resize(resolution_hw, interpolation=InterpolationMode.BILINEAR, antialias=True)(frames), fps


def get_multiview_dataset(
    *,
    dataset_name: str,
    is_train: bool,
    object_store: Literal["gcs", "s3"],
    dataset_keys: list[str],
    augmentations: dict[str, Augmentor],
    dataset_catalog: dict[str, dict[str, list[str]]],
) -> Dataset:
    """Get video-text dataset with optional custom augmentation factory.

    Args:
        is_train: Whether this is for training
        dataset_name: Name of dataset to use for loading wdinfo files
        object_store: Object store to use ("gcs" or "s3")
        dataset_keys: List of keys to use for loading dataset
        augmentations: Augmentations map to apply to dataset
        dataset_catalog: Dataset catalog to use for loading dataset
    """

    dataset_info = get_video_dataset_info(
        dataset_name,
        object_store=object_store,
        dataset_keys=dataset_keys,
        dataset_catalog=dataset_catalog,
    )

    if (
        USE_MEGATRON
        and parallel_state.is_initialized()
        and (
            parallel_state.get_context_parallel_world_size() > 1
            or parallel_state.get_tensor_model_parallel_world_size() > 1
        )
    ):
        distributor_fn = parallel_sync_multi_aspect_ratio.ShardlistMultiAspectRatioParallelSync
    else:
        distributor_fn = ShardlistBasic

    video_data_config = DatasetConfig(
        keys=[],  # keys are defined per dataset
        buffer_size=1,
        streaming_download=True,
        dataset_info=dataset_info,
        distributor=distributor_fn(
            shuffle=is_train,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=False,
            is_infinite_loader=is_train,
        ),
        decoders=[
            video_naive_bytes(),
            json_decoder,
        ],
        augmentation=augmentations,
        remove_extension_from_keys=True,
    )

    return Dataset(
        config=video_data_config,
        handler=wds.warn_and_continue,
        decoder_handler=wds.warn_and_continue,
        detshuffle=False,
    )


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    merged = dict()
    is_tensor = dict()
    for row in batch:
        for key, value in row.items():
            if key not in merged:
                merged[key] = []
            if isinstance(value, torch.Tensor):
                is_tensor[key] = True
            merged[key].append(value)
    for key, value in merged.items():
        if is_tensor.get(key, False):
            merged[key] = torch.stack(value, dim=0)
    return merged


@attrs.define(slots=False)
class AugmentationConfig:
    """Configuration for video augmentation."""

    resolution_hw: tuple[int, int] = (1080, 1920)
    fps_downsample_factor: int = 1
    num_video_frames: int = 93
    caption_probability: dict[str, float] = {
        "qwen2p5_7b_caption": 0.7,
        "qwen2p5_7b_caption_medium": 0.2,
        "qwen2p5_7b_caption_short": 0.1,
    }
    camera_keys: tuple[str, ...] = DEFAULT_CAMERAS
    camera_view_mapping: dict[str, int] = DEFAULT_CAMERA_VIEW_MAPPING
    camera_caption_key_mapping: dict[str, str] = DEFAULT_CAPTION_KEY_MAPPING
    camera_video_key_mapping: dict[str, str] = DEFAULT_VIDEO_KEY_MAPPING
    camera_control_key_mapping: Optional[dict[str, str]] = None
    position_to_camera_mapping: Optional[dict[int, str]] = None
    add_view_prefix_to_caption: bool = False
    camera_prefix_mapping: Optional[dict[str, str]] = DEFAULT_CAPTION_PREFIXES
    single_caption_camera_name: Optional[str] = None


def make_augmentations(augmentation_config: AugmentationConfig) -> tuple[dict[str, Augmentor], list[str]]:
    """Make augmentations for multiview video dataset."""

    augmentations = dict()
    if augmentation_config.position_to_camera_mapping is not None:
        augmentations["unpack_metas"] = UnpackMetas(
            position_to_camera_mapping=augmentation_config.position_to_camera_mapping
        )

    augmentations["extract_frames_and_captions"] = ExtractFramesAndCaptions(
        camera_order=augmentation_config.camera_keys,
        num_frames=augmentation_config.num_video_frames,
        resolution_hw=augmentation_config.resolution_hw,
        fps_downsample_factor=augmentation_config.fps_downsample_factor,
        caption_probability=augmentation_config.caption_probability,
        camera_view_mapping=augmentation_config.camera_view_mapping,
        camera_caption_key_mapping=augmentation_config.camera_caption_key_mapping,
        camera_video_key_mapping=augmentation_config.camera_video_key_mapping,
        camera_control_key_mapping=augmentation_config.camera_control_key_mapping,
        add_view_prefix_to_caption=augmentation_config.add_view_prefix_to_caption,
        camera_prefix_mapping=augmentation_config.camera_prefix_mapping,
        single_caption_camera_name=augmentation_config.single_caption_camera_name,
    )

    # define dataset keys to load
    dataset_keys = list(augmentation_config.camera_video_key_mapping.values())
    if augmentation_config.position_to_camera_mapping is not None:
        dataset_keys.append("metas")
    else:
        dataset_keys.extend(augmentation_config.camera_caption_key_mapping.values())
    if augmentation_config.camera_control_key_mapping is not None:
        dataset_keys.extend(augmentation_config.camera_control_key_mapping.values())

    return augmentations, dataset_keys


def get_multiview_video_loader(
    *,
    dataset_name: str,
    is_train: bool,
    object_store: Literal["gcs", "s3"] = "s3",
    augmentation_config: AugmentationConfig = AugmentationConfig(),
    batch_size: int = 1,
    num_workers: int = 4,
    prefetch_factor: int | None = 1,
):
    """Get video loader for alpamayo multiview dataset"""

    # make augmentations
    augmentations, dataset_keys = make_augmentations(augmentation_config)

    # get dataloader
    return get_cached_replay_dataloader(
        dataset=get_multiview_dataset(
            is_train=is_train,
            object_store=object_store,
            dataset_name=dataset_name,
            dataset_keys=dataset_keys,
            dataset_catalog=DEFAULT_CATALOG,
            augmentations=augmentations,
        ),
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=None,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=False,
        collate_fn=collate_fn,
        cache_replay_name="video_dataloader",
    )
