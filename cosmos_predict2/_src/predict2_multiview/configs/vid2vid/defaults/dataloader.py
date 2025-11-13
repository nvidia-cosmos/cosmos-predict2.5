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
from cosmos_predict2._src.predict2_multiview.datasets.multiview import (
    DEFAULT_CAMERAS,
    AugmentationConfig,
    get_multiview_video_loader,
)

DEFAULT_CAMERA_VIEW_CONFIGS = {
    "7views": DEFAULT_CAMERAS,
    "4views": [
        "camera_front_wide_120fov",
        "camera_cross_right_70fov",
        "camera_rear_tele_30fov",
        "camera_cross_left_70fov",
    ],
}


def register_multiview_dataloader() -> None:
    """Register multiview video dataloader configurations."""

    datasets = ["alpamayo_dec2024"]
    object_stores = ["gcs", "s3"]
    resolutions = [
        ("480p", (480, 832)),
        ("720p", (720, 1280)),
        ("1080p", (1080, 1920)),
    ]
    fps = [
        ("10fps", 3),
        ("15fps", 2),
        ("30fps", 1),
    ]
    num_video_frames = [
        ("29frames", 29),
        ("61frames", 61),
        ("93frames", 93),
    ]

    cs = ConfigStore.instance()
    for dataset in datasets:
        for object_store in object_stores:
            for resolution_str, resolution_hw in resolutions:
                for fps_str, downsample_factor in fps:
                    for num_video_frames_str, num_frames in num_video_frames:
                        for views_str, camera_keys in DEFAULT_CAMERA_VIEW_CONFIGS.items():
                            name = f"video_{dataset}_{object_store}_{resolution_str}_{fps_str}_{num_video_frames_str}_{views_str}"
                            cs.store(
                                group="data_train",
                                package="dataloader_train",
                                name=name,
                                node=L(get_multiview_video_loader)(
                                    dataset_name=dataset,
                                    is_train=True,
                                    object_store=object_store,
                                    augmentation_config=L(AugmentationConfig)(
                                        resolution_hw=resolution_hw,
                                        fps_downsample_factor=downsample_factor,
                                        num_video_frames=num_frames,
                                        camera_keys=camera_keys,
                                    ),
                                ),
                            )
