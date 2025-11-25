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

from typing import Literal, Protocol

import pydantic
import torch

from cosmos_predict2.config import (
    DEFAULT_NEGATIVE_PROMPT,
    CommonInferenceArguments,
    CommonSetupArguments,
    Guidance,
    ModelKey,
    ModelVariant,
    ResolvedDirectoryPath,
    get_model_literal,
    get_overrides_cls,
)

DEFAULT_MODEL_KEY = ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_AGIBOT)


class CameraLoadFn(Protocol):
    def __call__(
        self,
        text: str,
        video: torch.Tensor,
        path: str,
        base_path: str,
        latent_frames: int,
    ) -> list[dict]: ...


class RobotMultiviewSetupArguments(CommonSetupArguments):
    """Setup arguments for robot multiview inference."""

    config_file: str = "cosmos_predict2/_src/predict2/camera/configs/multiview_camera/config.py"

    base_path: ResolvedDirectoryPath
    """
    Directory where camera intrinsic, extrinsic and input images are located.

    Here is what the directory structure should look like (input_name defines the sample name):
    $ tree {base_path}
    ├── cameras
    │   ├── {input_name}_extrinsic_hand_0.txt
    │   ├── {input_name}_extrinsic_hand_1.txt
    │   ├── {input_name}_extrinsic_head.txt
    │   ├── {input_name}_intrinsic_hand_0.txt
    │   ├── {input_name}_intrinsic_hand_1.txt
    │   └── {input_name}_intrinsic_head.txt
    └── input_images
        ├── {input_name}_hand_0.png
        ├── {input_name}_hand_1.png
        └── {input_name}_head.png
    """

    num_input_frames: pydantic.PositiveInt = 1
    """Number of input frames to condition on"""
    num_output_frames: pydantic.PositiveInt = 93
    """Number of output frames to generate"""
    num_input_video: pydantic.PositiveInt = 1
    """Number of input videos present"""
    num_output_video: pydantic.PositiveInt = 3
    """Number of output videos to generate"""
    input_video_res: Literal["480p", "720p"] = "720p"
    """Input video resolution (model configuration)"""
    camera_load_create_fn: str = "cosmos_predict2.robot_multiview.load_agibot_camera_fn"
    """How to load the camera intrinsic and extrinsic data"""
    dataloader_num_workers: pydantic.NonNegativeInt = 0
    """Number of workers to use in dataloader (hint: only set >0 if multiple input videos provided)"""
    resolution: str = "none"
    """Resolution of the video (H,W). Be default it will use model trained resolution. 9:16"""

    # Override defaults
    # pyrefly: ignore  # invalid-annotation
    model: get_model_literal([ModelVariant.ROBOT_MULTIVIEW_AGIBOT]) = DEFAULT_MODEL_KEY.name


class RobotMultiviewInferenceArguments(CommonInferenceArguments):
    """Inference arguments for robot multiview inference."""

    input_name: str | None = None
    """The name of the input"""

    # pyrefly: ignore  # bad-override
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    """Negative prompt."""
    seed: int = 1
    """Seed value"""
    guidance: Guidance = 7
    """Guidance value"""


RobotMultiviewInferenceOverrides = get_overrides_cls(RobotMultiviewInferenceArguments, exclude=["name"])
