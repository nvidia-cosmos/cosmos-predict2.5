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
    ResolvedFilePath,
    get_model_literal,
    get_overrides_cls,
)

DEFAULT_MODEL_KEY = ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW)


class CameraLoadFn(Protocol):
    def __call__(
        self,
        text: str,
        video: torch.Tensor,
        path: str,
        base_path: str,
        latent_frames: int,
    ) -> list[dict]: ...


class CameraConditionedSetupArguments(CommonSetupArguments):
    """Setup arguments for camera-conditioned inference."""

    config_file: str = "cosmos_predict2/_src/predict2/configs/camera_conditioned/config.py"

    base_path: ResolvedDirectoryPath
    """Directory where camera intrinsic and extrinsic are located"""

    num_input_frames: pydantic.PositiveInt = 24
    """Number of input frames to condition on"""
    num_output_frames: pydantic.PositiveInt = 93
    """Number of output frames to generate"""
    num_input_video: pydantic.PositiveInt = 1
    """Number of input videos present"""
    num_output_video: pydantic.PositiveInt = 2
    """Number of output videos to generate"""
    input_video_res: Literal["480p", "720p"] = "720p"
    """Input video resolution (model configuration)"""
    camera_load_create_fn: str = "cosmos_predict2.camera_conditioned.load_basic_camera_fn"
    """How to load the camera intrinsic and extrinsic data"""
    dataloader_num_workers: pydantic.NonNegativeInt = 0
    """Number of workers to use in dataloader (hint: only set >0 if multiple input videos provided)"""
    resolution: str = "none"
    """Resolution of the video (H,W). Be default it will use model trained resolution. 9:16"""

    # Override defaults
    model: get_model_literal(ModelVariant.ROBOT_MULTIVIEW) = DEFAULT_MODEL_KEY.name


class CameraConditionedInferenceArguments(CommonInferenceArguments):
    """Inference arguments for camera-conditioned inference."""

    input_path: ResolvedFilePath = None
    """Optional path to an input video (relative to input_root). If None then all videos in `<input_root>/videos` will be processed."""

    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    """Negative prompt."""
    guidance: Guidance = 7
    """Guidance value"""


CameraConditionedInferenceOverrides = get_overrides_cls(CameraConditionedInferenceArguments, exclude=["name"])
