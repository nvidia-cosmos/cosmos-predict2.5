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

import json
import os
from dataclasses import dataclass

asset_dir = os.environ.get("ASSET_DIR", "assets/")
default_request_v2w = json.dumps(
    {
        "inference_type": "image2world",
        "samples": {
            "input0": {
                "input_path": os.path.join(asset_dir, "base/image2world/bus_terminal.jpg"),
                "prompt": "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene.",
            }
        },
    },
    indent=2,
)

default_request_mv = json.dumps(
    {
        "prompt_path": os.path.join(asset_dir, "multiview/prompt.txt"),
        "num_input_frames": 1,
        "front_wide": {"video_path": os.path.join(asset_dir, "multiview/urban_freeway_front_wide_120fov.mp4")},
        "cross_left": {"video_path": os.path.join(asset_dir, "multiview/urban_freeway_cross_left_120fov.mp4")},
        "cross_right": {"video_path": os.path.join(asset_dir, "multiview/urban_freeway_cross_right_120fov.mp4")},
        "rear_left": {"video_path": os.path.join(asset_dir, "multiview/urban_freeway_rear_left_70fov.mp4")},
        "rear_right": {"video_path": os.path.join(asset_dir, "multiview/urban_freeway_rear_right_70fov.mp4")},
        "rear": {"video_path": os.path.join(asset_dir, "multiview/urban_freeway_rear_30fov.mp4")},
        "front_tele": {"video_path": os.path.join(asset_dir, "multiview/urban_freeway_front_tele_30fov.mp4")},
    },
    indent=2,
)


@dataclass
class ModelConfig:
    header = {
        "video2world": "Cosmos-Predict2.5 Video2World",
        "multiview": "Cosmos-Predict2.5 Multiview",
    }

    help_text = {
        "video2world": "",
        "multiview": "",
    }

    default_request = {
        "video2world": default_request_v2w,
        "multiview": default_request_mv,
    }
