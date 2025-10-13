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

import omegaconf
from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.configs.common.mock_data import (
    MOCK_DATA_IMAGE_ONLY_CONFIG,
    MOCK_DATA_INTERLEAVE_CONFIG,
    MOCK_DATA_VIDEO_ONLY_CONFIG,
)
from cosmos_predict2._src.predict2.datasets.cached_replay_dataloader import get_cached_replay_dataloader
from cosmos_predict2._src.predict2.datasets.dataset_provider import get_image_dataset, get_video_dataset
from cosmos_predict2._src.predict2.datasets.joint_dataloader import IterativeJointDataLoader


def get_image_dataloader(dataset_name: str, object_store: str) -> omegaconf.dictconfig.DictConfig:
    return L(get_cached_replay_dataloader)(
        dataset=L(get_image_dataset)(
            dataset_name=dataset_name,
            resolution="720",
            is_train=True,
            object_store=object_store,
        ),
        num_workers=8,
        prefetch_factor=4,
        batch_size=2,
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
        cache_replay_name="image_dataloader",
    )


def get_video_dataloader(dataset_name: str, object_store: str) -> omegaconf.dictconfig.DictConfig:
    return L(get_cached_replay_dataloader)(
        dataset=L(get_video_dataset)(
            dataset_name=dataset_name,
            video_decoder_name="chunked_video_decoder",
            resolution="720",
            is_train=True,
            object_store=object_store,
            chunk_size=256,
        ),
        batch_size=1,
        num_workers=8,
        prefetch_factor=2,
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
        cache_replay_name="video_dataloader",
    )


def get_joint_image_video_dataloader(
    image_dataset_name: str,
    video_dataset_name: str,
    object_store="s3",
) -> omegaconf.dictconfig.DictConfig:
    image_dataloader = get_image_dataloader(dataset_name=image_dataset_name, object_store=object_store)
    video_dataloader = get_video_dataloader(dataset_name=video_dataset_name, object_store=object_store)
    c = L(IterativeJointDataLoader)(
        dataloaders={
            "image_data": {
                "dataloader": image_dataloader,
                "ratio": 1,
            },
            "video_data": {
                "dataloader": video_dataloader,
                "ratio": 1,
            },
        }
    )
    return c


def get_joint_image_two_video_dataloader(
    image_dataset_name: str,
    video_dataset_name: str,
    object_store="s3",
) -> omegaconf.dictconfig.DictConfig:
    """
    This dataloader is used for training with multi-resolution multi-fps video.
    """
    image_dataloader = get_image_dataloader(dataset_name=image_dataset_name, object_store=object_store)
    video_dataloader = get_video_dataloader(dataset_name=video_dataset_name, object_store=object_store)
    # Why do we name it video_data video_data_1 instead of video_data_1 video_data_2?
    # In our exp config, if we inherit the exp which has video_data (joint_image_video), the video_data dict will
    # stay in the config, and we end up with having
    # video_data: not used, stay here due to inheritance, can confuse users when checking config.yaml. This is an known issue of our config system.
    # video_data_1: used
    # video_data_2: used
    # Therefore, we keep the same video_data entry here, and add video_data_1
    c = L(IterativeJointDataLoader)(
        dataloaders={
            "image_data": {
                "dataloader": image_dataloader,
                "ratio": 2,
            },
            "video_data": {
                "dataloader": video_dataloader,
                "ratio": 1,
            },
            "video_data_1": {
                "dataloader": video_dataloader,
                "ratio": 1,
            },
        }
    )
    return c


def register_training_and_val_data():
    cs = ConfigStore()
    cs.store(group="data_train", package="dataloader_train", name="mock", node=MOCK_DATA_INTERLEAVE_CONFIG)
    cs.store(group="data_train", package="dataloader_train", name="mock_image", node=MOCK_DATA_IMAGE_ONLY_CONFIG)
    cs.store(group="data_train", package="dataloader_train", name="mock_video", node=MOCK_DATA_VIDEO_ONLY_CONFIG)
    cs.store(group="data_val", package="dataloader_val", name="mock", node=MOCK_DATA_INTERLEAVE_CONFIG)
