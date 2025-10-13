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

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_predict2._src.predict2_multiview.datasets.local_dataset_train import (
    LocalDrivingVideoDataloaderConfig,
    LocalMultiviewPredictDataset,
)

NUM_VIDEO_FRAMES_LOADED_PER_VIEW = 29
NUM_VIDEO_FRAMES_PER_VIEW = 29
RESOLUTION = "720p"
SINGLE_CAPTION_ONLY = True


H_W = {
    "720p": (720, 1280),
    "480p": (480, 832),
    "480": (432, 768),
    "720": (704, 1280),
}


def register_waymo_dataloader():
    H, W = H_W[RESOLUTION]

    camera_to_view_id = {
        "pinhole_front": 0,
        "pinhole_front_left": 5,
        "pinhole_front_right": 1,
        "pinhole_side_left": 4,
        "pinhole_side_right": 2,
    }

    camera_to_caption_prefix = {
        "pinhole_front": "The video is captured from a camera mounted on a car. The camera is facing forward.",
        "pinhole_front_left": "The video is captured from a camera mounted on a car. The camera is facing to the front left.",
        "pinhole_front_right": "The video is captured from a camera mounted on a car. The camera is facing to the front right.",
        "pinhole_side_left": "The video is captured from a camera mounted on a car. The camera is facing to the side left.",
        "pinhole_side_right": "The video is captured from a camera mounted on a car. The camera is facing to the side right.",
    }

    waymo_driving_dataloader_config = LocalDrivingVideoDataloaderConfig(
        sample_n_views=5,
        num_video_frames_per_view=NUM_VIDEO_FRAMES_PER_VIEW,
        num_video_frames_loaded_per_view=NUM_VIDEO_FRAMES_LOADED_PER_VIEW,
        sample_noncontiguous_views=False,
        ref_cam_view_idx=-1,
        overfit_firstn=-1,
        camera_to_view_id=camera_to_view_id,
        camera_to_caption_prefix=camera_to_caption_prefix,
        front_cam_key="pinhole_front",
        no_view_prefix=SINGLE_CAPTION_ONLY,
        single_caption_only=SINGLE_CAPTION_ONLY,
        H=H,
        W=W,
        resolution=RESOLUTION,
    )

    cs = ConfigStore.instance()

    waymo_dataset = L(LocalMultiviewPredictDataset)(
        video_file_dirs=["datasets/multiview/waymo/input"],
        driving_dataloader_config=waymo_driving_dataloader_config,
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name=f"waymo",
        node=L(get_generic_dataloader)(
            dataset=waymo_dataset,
            sampler=L(get_sampler)(dataset=waymo_dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name=f"waymo",
        node=L(get_generic_dataloader)(
            dataset=waymo_dataset,
            sampler=L(get_sampler)(dataset=waymo_dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )
