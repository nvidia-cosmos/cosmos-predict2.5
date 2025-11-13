# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Local file-based datasets.
"""

import glob
import os
import random
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from cosmos_predict2._src.predict2_multiview.datasets.multiview import (
    AugmentationConfig,
    make_augmentations,
)


class WaymoLocalDataset(Dataset):
    def __init__(
        self,
        video_file_dirs: list[str],
        augmentation_config: AugmentationConfig,
        shuffle: bool = True,
        gc_every_n: int = 100,
    ) -> None:
        self.video_file_dirs = video_file_dirs
        self.augmentation_config = augmentation_config
        self.shuffle = shuffle
        self.gc_every_n = gc_every_n

        self.augmentations, self.dataset_keys = make_augmentations(augmentation_config)
        self.sample_dirs = self.build_sample_dirs(self.video_file_dirs)

        if self.shuffle:
            random.shuffle(self.sample_dirs)

    def build_sample_dirs(self, video_file_dirs: list[str]) -> list[str]:
        sample_dirs = []
        for video_file_dir in video_file_dirs:
            for sample_dir in glob.glob(os.path.join(video_file_dir, "**")):
                if os.path.isdir(sample_dir):
                    sample_dirs.append(sample_dir)
        return sample_dirs

    def load_data(self, sample_dir: str) -> dict:
        sample_id = sample_dir.split("/")[-1]
        data_dict = dict()
        for filename in glob.glob(os.path.join(sample_dir, "*.mp4")):
            with open(filename, "rb") as f:
                view = filename.split("/")[-1].split(".")[0]
                data_dict[f"video_{view}"] = f.read()
        return data_dict

    def load_caption(self, sample_dir: str) -> dict:
        caption_path = os.path.join(sample_dir, "caption.jsonl")
        with open(caption_path, "r") as f:
            caption_df = pd.read_json(f, lines=True, orient="records")

        caption_dict = dict()
        for view_name, view_df in caption_df.groupby("view"):
            caption_styles = dict()
            for row in view_df.itertuples():
                caption = row.caption
                tag = None if pd.isna(row.tag) else row.tag
                caption_styles[tag or "long"] = caption
            caption_dict[f"caption_{view_name}"] = {
                "t2w_windows": [
                    {"start_frame": 0, "end_frame": self.augmentation_config.num_video_frames, **caption_styles}
                ]
            }
        return caption_dict

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> dict:
        sample_dir = self.sample_dirs[index]
        data_dict = {
            "__key__": str(index),
            "__url__": str(sample_dir[index]),
        }
        data_dict.update(self.load_data(sample_dir))
        data_dict.update(self.load_caption(sample_dir))
        for k, aug in self.augmentations.items():
            data_dict = aug(data_dict)
        return data_dict


class LocalMultiViewDataset(Dataset):
    """Dataset wrapper for local multiview sample."""

    def __init__(
        self,
        video_file_dicts: list[dict[str, Path | None]],
        prompts: list[str],
        augmentation_config: AugmentationConfig,
        camera_key_adapter: dict[str, str] | None = None,
        control_file_dicts: list[dict[str, Path | None]] | None = None,
    ) -> None:
        self.video_file_dicts = video_file_dicts
        self.prompts = prompts
        self.augmentation_config = augmentation_config
        self.camera_key_adapter = camera_key_adapter
        self.control_file_dicts = control_file_dicts

        if self.control_file_dicts is not None and len(self.video_file_dicts) != len(self.control_file_dicts):
            raise ValueError("Number of video file dicts and control file dicts must be the same!")

        if len(self.prompts) != len(self.video_file_dicts):
            raise ValueError("Number of prompts and video file dicts must be the same!")

        if self.augmentation_config.single_caption_camera_name is None:
            raise ValueError(
                "`single_caption_camera_name` must be set since only single prompt is provided by dataset!"
            )

        self.augmentations, self.dataset_keys = make_augmentations(augmentation_config)

    def __len__(self) -> int:
        return len(self.video_file_dicts)

    def __getitem__(self, index: int) -> dict:
        data_dict = {
            "__key__": str(index),
            "__url__": "local_dataset",
        }

        for view_key, filepath in self.video_file_dicts[index].items():
            if filepath is None:
                raise ValueError(f"view_key {view_key} has null filepath!")
            default_key = self.camera_key_adapter[view_key] if self.camera_key_adapter else view_key
            video_key = self.augmentation_config.camera_video_key_mapping[default_key]
            with open(filepath, "rb") as f:
                data_dict[video_key] = f.read()

        if self.control_file_dicts is not None:
            for view_key, filepath in self.control_file_dicts[index].items():
                if filepath is None:
                    raise ValueError(f"view_key {view_key} has null filepath!")
                default_key = self.camera_key_adapter[view_key] if self.camera_key_adapter else view_key
                control_key = self.augmentation_config.camera_control_key_mapping[default_key]
                with open(filepath, "rb") as f:
                    data_dict[control_key] = f.read()

        caption_styles = dict(
            zip(
                self.augmentation_config.caption_probability.keys(),
                [self.prompts[index] for _ in range(len(self.augmentation_config.caption_probability))],
            )
        )

        caption_key = self.augmentation_config.camera_caption_key_mapping[
            self.augmentation_config.single_caption_camera_name
        ]
        data_dict[caption_key] = {
            "t2w_windows": [
                {"start_frame": 0, "end_frame": self.augmentation_config.num_video_frames, **caption_styles}
            ]
        }

        for k, aug in self.augmentations.items():
            data_dict = aug(data_dict)
        return data_dict
