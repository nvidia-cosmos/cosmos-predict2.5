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

from cosmos_gradio.deployment_env import DeploymentEnv
from cosmos_gradio.model_ipc.model_server import ModelServer
from loguru import logger as log

from cosmos_predict2.config import InferenceArguments
from cosmos_predict2.multiview_config import MultiviewInferenceArguments

"""
compare to
COSMOS_INTERNAL=0 PYTHONPATH=. torchrun --nproc_per_node=8 examples/inference.py assets/base/image2world.json outputs/image2world
COSMOS_INTERNAL=0 PYTHONPATH=. torchrun --nproc_per_node=8 examples/multiview.py assets/multiview/multiview.json
"""

prompt = "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."
asset_dir = os.getenv("ASSET_DIR", "assets/")
asset_0 = os.path.join(asset_dir, "base/bus_terminal.jpg")
sample_dict = {
    "inference_type": "image2world",
    "samples": {"input0": {"input_path": asset_0, "prompt": prompt}},
}

sample_dict2 = {
    "prompt_path": os.path.join(asset_dir, "multiview/prompt.txt"),
    "num_input_frames": 1,
    "front_wide": {"video_path": os.path.join(asset_dir, "multiview/urban_freeway_front_wide_120fov.mp4")},
    "cross_left": {"video_path": os.path.join(asset_dir, "multiview/urban_freeway_cross_left_120fov.mp4")},
    "cross_right": {"video_path": os.path.join(asset_dir, "multiview/urban_freeway_cross_right_120fov.mp4")},
    "rear_left": {"video_path": os.path.join(asset_dir, "multiview/urban_freeway_rear_left_70fov.mp4")},
    "rear_right": {"video_path": os.path.join(asset_dir, "multiview/urban_freeway_rear_right_70fov.mp4")},
    "rear": {"video_path": os.path.join(asset_dir, "multiview/urban_freeway_rear_30fov.mp4")},
    "front_tele": {"video_path": os.path.join(asset_dir, "multiview/urban_freeway_front_tele_30fov.mp4")},
}

global_env = DeploymentEnv()


def test_video2world():
    from cosmos_predict2.gradio.video2world_worker import Video2World_Worker

    model_params = InferenceArguments(**sample_dict)
    pipeline = Video2World_Worker(num_gpus=1)

    model_params = model_params.model_dump()
    model_params["output_dir"] = "outputs/predict2/v2w/"
    pipeline.infer(model_params)


def test_multiview():
    with ModelServer(
        num_gpus=8,
        factory_module="cosmos_predict2.gradio.gradio_bootstrapper",
        factory_function="create_multiview",
    ) as pipeline:
        model_params = MultiviewInferenceArguments(**sample_dict2)
        model_params = model_params.model_dump(mode="json")
        model_params["output_dir"] = "outputs/predict2/mv/"
        pipeline.infer(model_params)


if __name__ == "__main__":
    log.info(f"test_worker current dir={os.getcwd()}")
    log.info(global_env)
    if global_env.model_name == "video2world":
        test_video2world()
    elif global_env.model_name == "multiview":
        test_multiview()
    else:
        test_video2world()
        # test_multiview()
