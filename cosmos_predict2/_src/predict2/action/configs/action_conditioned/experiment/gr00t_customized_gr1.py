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

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import LeRobotDataset
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
    construct_modality_config_and_transforms,
)

# [local] gr00t_gr1 dataset path
base_path_gr00t_gr1_local = "/project/cosmos/weichengt/datasets/gr1_unified/gr1_unified.RU0226RemoveStaticFreq20"
train_annotation_path_gr00t_gr1_local = os.path.join(base_path_gr00t_gr1_local, "annotation/train")
val_annotation_path_gr00t_gr1_local = os.path.join(base_path_gr00t_gr1_local, "annotation/train")

# Construct modality configs and transforms
modality_configs, train_transform, test_transform = construct_modality_config_and_transforms(
    num_frames=13, embodiment="gr1", downscaled_res=False
)


gr00t_customiezed_gr1_train_dataset = L(LeRobotDataset)(
    num_frames=13,
    time_division_factor=4,
    time_division_remainder=1,
    max_pixels=1920 * 1080,
    data_file_keys=("video",),
    image_file_extension=("jpg", "jpeg", "png", "webp"),
    video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
    repeat=1,
    args=None,
    dataset_path=base_path_gr00t_gr1_local,
    data_split="train",
    embodiment="gr1",
    downscaled_res=False,
)

gr00t_customiezed_gr1_val_dataset = L(LeRobotDataset)(
    num_frames=13,
    time_division_factor=4,
    time_division_remainder=1,
    max_pixels=1920 * 1080,
    data_file_keys=("video",),
    image_file_extension=("jpg", "jpeg", "png", "webp"),
    video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
    repeat=1,
    args=None,
    dataset_path=base_path_gr00t_gr1_local,
    data_split="test",
    embodiment="gr1",
    downscaled_res=False,
)


# Dataloader helper function
def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


# Dataloader definitions
gr00t_customiezed_gr1_train_dataloader = L(DataLoader)(
    dataset=gr00t_customiezed_gr1_train_dataset,
    sampler=L(get_sampler)(dataset=gr00t_customiezed_gr1_train_dataset),
    batch_size=1,
    drop_last=True,
)

gr00t_customiezed_gr1_val_dataloader = L(DataLoader)(
    dataset=gr00t_customiezed_gr1_val_dataset,
    sampler=L(get_sampler)(dataset=gr00t_customiezed_gr1_val_dataset),
    batch_size=1,
    drop_last=True,
)


# Registration function
def register_gr00t_customized_gr1_data():
    cs = ConfigStore.instance()

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="gr00t_customiezed_gr1_train",
        node=gr00t_customiezed_gr1_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="gr00t_customiezed_gr1_val",
        node=gr00t_customiezed_gr1_val_dataloader,
    )
