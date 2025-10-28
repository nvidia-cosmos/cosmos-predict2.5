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

import enum
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal

import pydantic
import tyro

from cosmos_predict2._src.imaginaire.flags import SMOKE
from cosmos_predict2.config import (
    MODEL_CHECKPOINTS,
    CommonInferenceArguments,
    CommonSetupArguments,
    ModelKey,
    ModelVariant,
    ResolvedFilePath,
    get_model_literal,
    get_overrides_cls,
)

DEFAULT_MODEL_KEY = ModelKey(variant=ModelVariant.AUTO_MULTIVIEW)
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_MODEL_KEY]

VIEW_INDEX_DICT = {
    "front_wide": 0,
    "cross_right": 1,
    "rear_right": 2,
    "rear": 3,
    "rear_left": 4,
    "cross_left": 5,
    "front_tele": 6,
}

StackMode = Literal["time", "height"]


class MultiviewSetupArguments(CommonSetupArguments):
    """Arguments for multiview setup."""

    # Override defaults
    # pyrefly: ignore  # invalid-annotation
    model: get_model_literal([ModelVariant.AUTO_MULTIVIEW]) = DEFAULT_MODEL_KEY.name
    use_config_dataloader: bool = False
    """Ignore input root and use dataloader in config"""


class MultiviewInferenceType(str, enum.Enum):
    """Multiview model inference type."""

    TEXT2WORLD = "text2world"
    IMAGE2WORLD = "image2world"
    VIDEO2WORLD = "video2world"

    def __str__(self) -> str:
        return self.value


class ViewConfig(pydantic.BaseModel):
    """Configuration for a single view."""

    model_config = pydantic.ConfigDict(extra="forbid")

    video_path: ResolvedFilePath | None = None
    """Path to the input video for this view."""


class MultiviewInferenceArguments(CommonInferenceArguments):
    """Arguments for multiview inference."""

    # Required parameters
    inference_type: tyro.conf.EnumChoicesFromValues[MultiviewInferenceType]
    """Inference type."""

    n_views: int = pydantic.Field(default=1 if SMOKE else 7, description="Number of views to generate")
    """Number of views to generate."""
    control_weight: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 1.0
    """Control weight for generation."""
    stack_mode: StackMode = "time"
    """Stacking mode for frames."""

    fps: pydantic.PositiveInt = 30
    """Frames per second for output video."""
    num_steps: pydantic.PositiveInt = 1 if SMOKE else 35
    """Number of generation steps."""

    # Override defaults
    # pyrefly: ignore  # bad-override
    prompt: str
    # pyrefly: ignore  # bad-override
    negative_prompt: None = pydantic.Field(None, exclude=True)

    @cached_property
    # pyrefly: ignore  # bad-return
    def num_input_frames(self) -> int:
        """Get number of input frames."""
        if self.inference_type == MultiviewInferenceType.TEXT2WORLD:
            return 0
        elif self.inference_type == MultiviewInferenceType.IMAGE2WORLD:
            return 1
        elif self.inference_type == MultiviewInferenceType.VIDEO2WORLD:
            return 2


class MultiviewInferenceArgumentsWithInputPaths(MultiviewInferenceArguments):
    """Arguments for multiview inference."""

    front_wide: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Front wide view configuration."""
    rear: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Rear view configuration."""
    rear_left: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Rear left view configuration."""
    rear_right: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Rear right view configuration."""
    cross_left: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Cross left view configuration."""
    cross_right: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Cross right view configuration."""
    front_tele: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Front tele view configuration."""

    @cached_property
    # pyrefly: ignore  # bad-return
    def num_input_frames(self) -> int:
        """Get number of input frames."""
        if self.inference_type == MultiviewInferenceType.TEXT2WORLD:
            return 0
        elif self.inference_type == MultiviewInferenceType.IMAGE2WORLD:
            return 1
        elif self.inference_type == MultiviewInferenceType.VIDEO2WORLD:
            return 2

    @property
    def input_paths(self) -> dict[str, Path | None]:
        """Get input paths for all views."""
        input_paths = {
            "front_wide": self.front_wide.video_path,
            "rear": self.rear.video_path,
            "rear_left": self.rear_left.video_path,
            "rear_right": self.rear_right.video_path,
            "cross_left": self.cross_left.video_path,
            "cross_right": self.cross_right.video_path,
            "front_tele": self.front_tele.video_path,
        }
        return input_paths


MultiviewInferenceOverrides = get_overrides_cls(
    MultiviewInferenceArgumentsWithInputPaths,
    exclude=[
        "name",
        "front_wide",
        "rear",
        "rear_left",
        "rear_right",
        "cross_left",
        "cross_right",
        "front_tele",
    ],
)
