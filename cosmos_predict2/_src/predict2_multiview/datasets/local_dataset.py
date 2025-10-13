# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterable, TypedDict

import torch
from torch.utils.data import Dataset

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2_multiview.configs.vid2vid.defaults.driving import MADSDrivingVideoDataloaderConfig
from cosmos_predict2._src.predict2_multiview.datasets.augmentor_provider import (
    get_video_augmentor_v2_multiview_no_text_emb,
)
from cosmos_predict2.multiview_config import VIEW_INDEX_DICT


class MultiviewInput(TypedDict):
    __key__: str
    ai_caption: str
    aspect_ratio: str
    camera_keys_selection: list[str]
    chunk_index: torch.Tensor
    fps: float
    frame_end: torch.Tensor
    frame_indices: list[torch.Tensor]
    frame_start: torch.Tensor
    front_cam_view_idx_sample_position: torch.Tensor
    image_size: torch.Tensor
    n_orig_video_frames: torch.Tensor
    n_orig_video_frames_per_view: list[torch.Tensor]
    num_conditional_frames: int
    num_video_frames_per_view: torch.Tensor
    num_frames: torch.Tensor
    padding_mask: torch.Tensor
    ref_cam_view_idx_sample_position: torch.Tensor
    sample_n_views: torch.Tensor
    video: torch.Tensor
    view_indices: torch.Tensor
    view_indices_selection: list[torch.Tensor]


@dataclass
class LocalMultiviewAugmentorConfig:
    resolution: str
    driving_dataloader_config: MADSDrivingVideoDataloaderConfig
    caption_type: str = "t2w_qwen2p5_7b"
    min_fps_thres: int = 10
    max_fps_thres: int = 60
    num_video_frames: int = 121
    use_native_fps: bool = False
    use_original_fps: bool = False


def wrap_augmentor_func_as_generator(func: Callable, data: Iterable):
    """Wrap a regular augmentor function as a generator."""
    for data_dict in data:
        data_dict_out = func(data_dict)
        if data_dict_out is None:
            # Skip "unhealthy" samples
            continue
        yield data_dict_out


class LocalMultiviewDataset(Dataset):
    """
    Simple dataset that reads files as bytes and applies augmentors.
    Much simpler than WebDataset since we don't need TAR extraction,
    distributed training coordination, or URL streaming.
    """

    def __init__(
        self,
        video_file_dict: Dict[str, str],
        augmentor_fn: Callable,
    ):
        super().__init__()
        self.video_file_dict = video_file_dict
        self.augmentor_fn = augmentor_fn

        # Set total_images for compatibility with webdataset
        self.total_images = len(self.video_file_dict)

        log.info(f"Video files in MultiviewDataset: {self.video_file_dict}")

    def __len__(self):
        """For local dataset, we only have one sample."""
        return 1

    def __getitem__(self, idx):
        """Get a single video sample by index."""
        try:
            data_dict = {
                "__key__": "local_dataset",
                "aspect_ratio": "16,9",
            }
            # Read file as bytes
            for view_key, filepath in self.video_file_dict.items():
                log.info(f"Reading input video file: {filepath}")
                with open(filepath, "rb") as f:
                    file_bytes = f.read()
                index = VIEW_INDEX_DICT[view_key]
                data_dict[f"video_{index}"] = file_bytes

            # Apply augmentors
            if self.augmentor_fn:
                # augmentor_fn is a generator, consume and return first sample
                for sample in self.augmentor_fn([data_dict]):
                    return sample
            else:
                return MultiviewInput(**data_dict)

        except Exception as e:
            log.error(f"Failed to load files from {self.video_file_dict}: {e}")
            # Re-raise to let DataLoader handle it
            raise


class LocalMultiviewDatasetBuilder:
    """
    Local video dataset class that mirrors the structure of webdataset.Dataset.
    Processes local MP4 files instead of TAR archives from web.
    """

    def __init__(
        self,
        video_file_dict: Dict[str, str],
    ):
        """
        Initialize the dataset, similar to webdataset.Dataset.__init__.

        Args:
            config: LocalVideoConfig instance
            handler: Error handler for exceptions
            decoder_handler: Error handler during decoding (for compatibility)
        """

        self.video_file_dict = video_file_dict

    @staticmethod
    def augmentor_fn(data, augmentations):
        """
        Apply augmentors in sequence, handling both generator and regular functions.
        This matches the webdataset augmentor_fn behavior exactly.
        """
        # Build augmentor chain
        for aug_fn in augmentations:
            # Use generator function as augmentor
            # (recommended, allows skipping or replicating samples inside the augmentor)
            if getattr(aug_fn, "is_generator", False):
                data = aug_fn(data)
            else:  # Use regular function as augmentor (backward compatibility)
                data = wrap_augmentor_func_as_generator(aug_fn, data)
        yield from data

    def build_data_augmentor(self, augmentor_cfg: Dict[str, Any]) -> Callable:
        """
        Build data augmentors from config, similar to webdataset.

        Args:
            augmentor_cfg: Augmentor configuration

        Returns:
            Partial function that applies all augmentors
        """
        augmentations = []
        if augmentor_cfg:
            from cosmos_predict2._src.imaginaire.lazy_config import instantiate

            for aug in augmentor_cfg:
                augmentations.append(instantiate(augmentor_cfg[aug]))

        # This is the function that calls each augmentor in sequence.
        return partial(LocalMultiviewDatasetBuilder.augmentor_fn, augmentations=augmentations)

    def build_dataset(self, multiview_augmentor_config: LocalMultiviewAugmentorConfig) -> LocalMultiviewDataset:
        """
        Build the dataset iterator, similar to webdataset.build_dataset.

        Returns:
            SimpleFileIterator that yields video samples
        """
        assert len(self.video_file_dict) > 0, "Did not find any video files."

        # Building augmentors (similar to webdataset)
        augmentor_cfg = get_video_augmentor_v2_multiview_no_text_emb(
            resolution=multiview_augmentor_config.resolution,
            driving_dataloader_config=multiview_augmentor_config.driving_dataloader_config,
            caption_type=multiview_augmentor_config.caption_type,
            min_fps=multiview_augmentor_config.min_fps_thres,
            max_fps=multiview_augmentor_config.max_fps_thres,
            num_video_frames=multiview_augmentor_config.num_video_frames,
            use_native_fps=multiview_augmentor_config.use_native_fps,
            use_original_fps=multiview_augmentor_config.use_original_fps,
        )

        augmentor_fn = self.build_data_augmentor(augmentor_cfg)

        # Create the iterator
        dataset = LocalMultiviewDataset(
            video_file_dict=self.video_file_dict,
            augmentor_fn=augmentor_fn,
        )

        return dataset
