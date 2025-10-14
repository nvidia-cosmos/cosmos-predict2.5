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
from dataclasses import dataclass
from typing import Optional


@dataclass
class MADSDrivingVideoDataloaderConfig:
    num_video_frames_loaded_per_view: int
    num_video_frames_per_view: int
    n_views: int
    single_caption_only: bool
    front_cam_key: str
    camera_to_view_id: dict
    video_id_to_camera_key: dict
    use_random_view_caption: bool = False
    minimum_start_index: int = 3
    override_original_fps: Optional[float] = None
    H: int = 704
    W: int = 1280

    # sample_n_views: int
    # sample_noncontiguous_views: bool
    # ref_cam_view_idx: int
    # overfit_firstn: int
    # camera_to_view_id: dict
    # view_id_to_caption_id: dict
    # camera_to_caption_prefix: dict
    # front_tele_and_front_cam_keys: tuple[str, str]
    # concat_viewt5: bool
    # no_view_prefix: bool


front_cam_key = "camera_front_wide_120fov"

#### 4 views ####
camera_to_view_id_4views = {
    "camera_front_wide_120fov": 0,
    "camera_cross_left_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_rear_tele_30fov": 3,
}

video_id_to_camera_key_4views = {
    "0": "camera_front_wide_120fov",
    "1": "camera_cross_left_120fov",
    "2": "camera_cross_right_120fov",
    "5": "camera_rear_tele_30fov",
}


#### 7 views ####
camera_to_view_id_7views = {
    "camera_front_wide_120fov": 0,
    "camera_cross_left_120fov": 5,
    "camera_cross_right_120fov": 1,
    "camera_rear_left_70fov": 4,
    "camera_rear_right_70fov": 2,
    "camera_rear_tele_30fov": 3,
    "camera_front_tele_30fov": 6,
}
video_id_to_camera_key_7views = {
    "0": "camera_front_wide_120fov",
    "1": "camera_cross_left_120fov",
    "2": "camera_cross_right_120fov",
    "3": "camera_rear_left_70fov",
    "4": "camera_rear_right_70fov",
    "5": "camera_rear_tele_30fov",
    "6": "camera_front_tele_30fov",
}

camera_to_view_id_config = {
    4: camera_to_view_id_4views,
    7: camera_to_view_id_7views,
}

video_id_to_camera_key_config = {
    4: video_id_to_camera_key_4views,
    7: video_id_to_camera_key_7views,
}

MADS_DRIVING_DATALOADER_CONFIG_res720 = MADSDrivingVideoDataloaderConfig(
    num_video_frames_loaded_per_view=85,
    num_video_frames_per_view=29,
    n_views=7,
    minimum_start_index=3,
    single_caption_only=True,
    H=704,
    W=1280,
    front_cam_key=front_cam_key,
    camera_to_view_id=camera_to_view_id_7views,
    video_id_to_camera_key=video_id_to_camera_key_7views,
)

MADS_DRIVING_DATALOADER_CONFIG_res720p = copy.deepcopy(MADS_DRIVING_DATALOADER_CONFIG_res720)
MADS_DRIVING_DATALOADER_CONFIG_res720p.H = 720

MADS_DRIVING_DATALOADER_CONFIG_res480p = copy.deepcopy(MADS_DRIVING_DATALOADER_CONFIG_res720)
MADS_DRIVING_DATALOADER_CONFIG_res480p.H = 480
MADS_DRIVING_DATALOADER_CONFIG_res480p.W = 832


MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION = {
    "720": MADS_DRIVING_DATALOADER_CONFIG_res720,
    "720p": MADS_DRIVING_DATALOADER_CONFIG_res720p,
    "480p": MADS_DRIVING_DATALOADER_CONFIG_res480p,
}


@dataclass
class DrivingVideoDataloaderConfig:
    sample_n_views: int
    num_video_frames_per_view: int
    num_video_frames_loaded_per_view: int
    sample_noncontiguous_views: bool
    ref_cam_view_idx: int
    overfit_firstn: int
    camera_to_view_id: dict
    view_id_to_caption_id: dict
    camera_to_caption_prefix: dict
    front_tele_and_front_cam_keys: tuple[str, str]
    concat_viewt5: bool
    no_view_prefix: bool
    single_caption_only: bool
    hint_keys: str  # ie "vis_edge2x_seg"
    H: int
    W: int
    download_t5_tar: bool
    t5_store_prefix: str
