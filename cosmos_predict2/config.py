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
import json
import os
import sys
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, Optional, TypeVar

import pydantic
import tyro
import yaml
from typing_extensions import Self, assert_never

from cosmos_predict2._src.imaginaire.flags import SMOKE
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_by_uuid


@cache
def is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def path_to_str(v: Path | None) -> str | None:
    """Convert optional path to optional string."""
    if v is None:
        return None
    return str(v)


def load_callable(name: str):
    from importlib import import_module

    idx = name.rfind(".")
    assert idx > 0, "expected <module_name>.<identifier>"
    module_name = name[0:idx]
    fn_name = name[idx + 1 :]

    module = import_module(module_name)
    fn = getattr(module, fn_name)
    return fn


_PydanticModelT = TypeVar("_PydanticModelT", bound=pydantic.BaseModel)


def get_overrides_cls(cls: type[_PydanticModelT], *, exclude: list[str] | None = None) -> type[pydantic.BaseModel]:
    """Get overrides class for a given pydantic model."""
    # pyrefly: ignore  # no-matching-overload
    names = set(cls.model_fields.keys())
    if exclude is not None:
        invalid = set(exclude) - names
        if invalid:
            raise ValueError(f"Invalid exclude: {invalid}")
        names -= set(exclude)
    # pyrefly: ignore  # no-matching-overload
    fields = {name: (Optional[cls.model_fields[name].rebuild_annotation()], None) for name in names}
    # pyrefly: ignore  # no-matching-overload, bad-argument-type, bad-argument-count
    return pydantic.create_model(f"{cls.__name__}Overrides", **fields)


def _get_root_exception(exception: Exception) -> Exception:
    if exception.__cause__ is not None:
        # pyrefly: ignore  # bad-argument-type
        return _get_root_exception(exception.__cause__)
    if exception.__context__ is not None:
        # pyrefly: ignore  # bad-argument-type
        return _get_root_exception(exception.__context__)
    return exception


def handle_tyro_exception(exception: Exception) -> NoReturn:
    root_exception = _get_root_exception(exception)
    if isinstance(root_exception, pydantic.ValidationError):
        if is_rank0():
            print(root_exception, file=sys.stderr)
        sys.exit(1)
    raise exception


def _resolve_path(v: Path) -> Path:
    """Resolve path to absolute."""
    return v.expanduser().resolve()


ResolvedFilePath = Annotated[pydantic.FilePath, pydantic.AfterValidator(_resolve_path)]
ResolvedDirectoryPath = Annotated[pydantic.DirectoryPath, pydantic.AfterValidator(_resolve_path)]


def _validate_checkpoint_uuid(v: str) -> str:
    """Validate checkpoint UUID."""
    get_checkpoint_by_uuid(v)
    return v


CheckpointUuid = Annotated[str, pydantic.AfterValidator(_validate_checkpoint_uuid)]


def _validate_checkpoint_path(v: str) -> str:
    """Validate checkpoint path or URI."""
    if SMOKE:
        return v
    if v.startswith("s3://"):
        return v
    if not os.path.exists(v):
        raise ValueError(f"Checkpoint path '{v}' does not exist.")
    return v


CheckpointPath = Annotated[str, pydantic.AfterValidator(_validate_checkpoint_path)]


class ModelSize(str, enum.Enum):
    _2B = "2B"
    _14B = "14B"

    def __str__(self) -> str:
        return self.value


class ModelVariant(str, enum.Enum):
    BASE = "base"
    AUTO_MULTIVIEW = "auto/multiview"
    ROBOT_ACTION_COND = "robot/action-cond"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, kw_only=True)
class ModelKey:
    post_trained: bool = True
    size: ModelSize = ModelSize._2B
    variant: ModelVariant = ModelVariant.BASE

    @cached_property
    def name(self) -> str:
        parts = [str(self.size)]
        if self.variant == ModelVariant.BASE:
            parts.append("post-trained" if self.post_trained else "pre-trained")
        else:
            parts.append(str(self.variant))
        return "/".join(parts)

    def __str__(self) -> str:
        return self.name


MODEL_CHECKPOINTS = {
    ModelKey(post_trained=False): get_checkpoint_by_uuid("d20b7120-df3e-4911-919d-db6e08bad31c"),
    ModelKey(): get_checkpoint_by_uuid("81edfebe-bd6a-4039-8c1d-737df1a790bf"),
    ModelKey(post_trained=False, size=ModelSize._14B): get_checkpoint_by_uuid("54937b8c-29de-4f04-862c-e67b04ec41e8"),
    ModelKey(variant=ModelVariant.AUTO_MULTIVIEW): get_checkpoint_by_uuid("6b9d7548-33bb-4517-b5e8-60caf47edba7"),
    ModelKey(variant=ModelVariant.ROBOT_ACTION_COND): get_checkpoint_by_uuid("38c6c645-7d41-4560-8eeb-6f4ddc0e6574"),
}
"""Mapping from model key to checkpoint."""

MODEL_KEYS = {k.name: k for k in MODEL_CHECKPOINTS.keys()}
"""Mapping from model name to model key."""


# pyrefly: ignore  # invalid-annotation
def get_model_literal(variants: list[ModelVariant] | None = None) -> Literal:
    """Get model literal for a given variant."""
    model_names: list[str] = []
    for k in MODEL_CHECKPOINTS.keys():
        if variants is not None and k.variant not in variants:
            continue
        model_names.append(k.name)
    # pyrefly: ignore  # bad-return, invalid-literal
    return Literal[tuple(model_names)]


DEFAULT_MODEL_KEY = ModelKey()
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_MODEL_KEY]
DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
VIDEO_EXTENSIONS = [".mp4"]


class CommonSetupArguments(pydantic.BaseModel):
    """Common arguments for model setup."""

    model_config = pydantic.ConfigDict(extra="forbid")

    # Required parameters
    output_dir: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """Output directory."""

    # Optional parameters
    # pyrefly: ignore  # invalid-annotation
    model: get_model_literal() = DEFAULT_MODEL_KEY.name
    """Model name."""
    checkpoint_path: CheckpointPath | None = None
    """Path to the checkpoint."""
    experiment: str | None = None
    """Experiment name."""
    config_file: str = "cosmos_predict2/_src/predict2/configs/video2world/config.py"
    """Configuration file for the model."""
    context_parallel_size: pydantic.PositiveInt | None = None
    """Context parallel size. Default to all nodes."""
    offload_diffusion_model: bool = False
    """Offload diffusion model to CPU to save GPU memory. Default to False."""
    offload_tokenizer: bool = False
    """Offload tokenizer to CPU to save GPU memory. Default to False."""
    offload_text_encoder: bool = False
    """Offload text encoder to CPU to save GPU memory. Default to False."""
    disable_guardrails: bool = False
    """Disable guardrails if this is set to True."""
    offload_guardrail_models: bool = True
    """Offload guardrail models to CPU to save GPU memory."""
    keep_going: bool = True
    """Keep going if an error occurs."""
    profile: bool = False
    """Run profiler and save report to output directory."""

    @cached_property
    def enable_guardrails(self) -> bool:
        return not self.disable_guardrails

    @cached_property
    def model_key(self) -> ModelKey:
        return MODEL_KEYS[self.model]

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_model(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        model_name: str | None = data.get("model")
        if model_name is None:
            raise ValueError("model is required")
        model_key = MODEL_KEYS[model_name]
        checkpoint = MODEL_CHECKPOINTS[model_key]
        if data.get("checkpoint_path") is None:
            data["checkpoint_path"] = checkpoint.path
        if data.get("experiment") is None:
            data["experiment"] = checkpoint.experiment
        if data.get("context_parallel_size") is None:
            data["context_parallel_size"] = int(os.environ.get("WORLD_SIZE", "1"))
        return data


Guidance = Annotated[int, pydantic.Field(ge=0, le=7)]


class CommonInferenceArguments(pydantic.BaseModel):
    """Common inference arguments."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    # Required parameters
    name: str
    """Name of the sample."""
    prompt_path: ResolvedFilePath | None = pydantic.Field(None, init_var=True)
    """Path to file containing the prompt."""
    prompt: str | None = None
    """Text prompt for generation."""

    # Optional parameters
    negative_prompt: str | None = None
    """Negative prompt."""

    # Advanced parameters
    seed: int = 0
    """Seed value."""
    guidance: Guidance = 3
    """Guidance value."""

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_prompt(cls, data: Any) -> Any:
        """
        Sets the 'prompt' field using the content of 'prompt_path' if it's provided.
        """
        if not isinstance(data, dict):
            return data
        prompt: str | None = data.get("prompt")
        if prompt is not None:
            return data
        prompt_path: str | None = data.get("prompt_path")
        if prompt_path is not None:
            # pyrefly: ignore  # annotation-mismatch
            prompt_path: Path = ResolvedFilePath(prompt_path)
            data["prompt"] = prompt_path.read_text().strip()
            return data
        return data

    @classmethod
    def _from_file(cls, path: Path, override_data: dict[str, Any]) -> list[Self]:
        """Load arguments from a json/jsonl/yaml file.

        Returns a list of arguments.
        """
        # Load data from file
        if path.suffix in [".json"]:
            data_list = [json.loads(path.read_text())]
        elif path.suffix in [".jsonl"]:
            data_list = [json.loads(line) for line in path.read_text().splitlines() if line]
        elif path.suffix in [".yaml", ".yml"]:
            data_list = [yaml.safe_load(path.read_text())]
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        # Validate data
        # Input paths are relative to the file path
        cwd = os.getcwd()
        os.chdir(path.parent)
        objs: list[Self] = []
        for i, data in enumerate(data_list):
            try:
                objs.append(cls.model_validate(data | override_data))
            except pydantic.ValidationError as e:
                if is_rank0():
                    print(f"Error validating parameters from '{path}' at line {i}\n{e}", file=sys.stderr)
                sys.exit(1)
        os.chdir(cwd)

        return objs

    @classmethod
    def from_files(cls, paths: list[Path], overrides: pydantic.BaseModel | None = None) -> list[Self]:
        """Load arguments from a list of json/jsonl/yaml files.

        Returns a list of arguments.
        """
        if not paths:
            if is_rank0():
                print("Error: No inference parameter files", file=sys.stderr)
            sys.exit(1)

        if overrides is None:
            override_data = {}
        else:
            override_data = overrides.model_dump(exclude_none=True)

        # Load arguments from files
        objs: list[Self] = []
        for path in paths:
            objs.extend(cls._from_file(path, override_data))
        if not objs:
            if is_rank0():
                print("Error: No inference samples", file=sys.stderr)
            sys.exit(1)

        # Check if names are unique
        names: set[str] = set()
        for obj in objs:
            if obj.name in names:
                print(f"Error: Inference samplename {obj.name} is not unique", file=sys.stderr)
                sys.exit(1)
            names.add(obj.name)

        return objs


class SetupArguments(CommonSetupArguments):
    """Base model setup arguments."""

    # Override defaults
    # pyrefly: ignore  # invalid-annotation
    model: get_model_literal([ModelVariant.BASE]) = DEFAULT_MODEL_KEY.name


class InferenceType(str, enum.Enum):
    """Base model inference type."""

    TEXT2WORLD = "text2world"
    IMAGE2WORLD = "image2world"
    VIDEO2WORLD = "video2world"

    def __str__(self) -> str:
        return self.value


INPUT_EXTENSIONS: dict[InferenceType, list[str] | None] = {
    InferenceType.TEXT2WORLD: None,
    InferenceType.IMAGE2WORLD: IMAGE_EXTENSIONS + VIDEO_EXTENSIONS,
    InferenceType.VIDEO2WORLD: IMAGE_EXTENSIONS + VIDEO_EXTENSIONS,
}


class InferenceArguments(CommonInferenceArguments):
    """Base model inference arguments."""

    # Required parameters
    inference_type: tyro.conf.EnumChoicesFromValues[InferenceType]
    """Inference type."""
    input_path: ResolvedFilePath | None = None
    """Path to the input image/video."""

    # Advanced parameters
    resolution: str = "none"
    """Resolution of the video (H,W). Be default it will use model trained resolution. 9:16"""
    num_output_frames: pydantic.PositiveInt = 77
    """Number of video frames to generate"""
    num_steps: pydantic.PositiveInt = 1 if SMOKE else 35
    """Number of generation steps."""

    # Override defaults
    # pyrefly: ignore  # bad-override
    prompt: str
    # pyrefly: ignore  # bad-override
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    seed: int = 1
    guidance: Guidance = 7

    @pydantic.model_validator(mode="after")
    def validate_input_path(self) -> Self:
        supported_extensions = INPUT_EXTENSIONS[self.inference_type]
        if supported_extensions is not None:
            if self.input_path is None:
                raise ValueError(f"input_path is required for inference type {self.inference_type}")
            if self.input_path.suffix not in supported_extensions:
                raise ValueError(
                    f"input_path has unsupported file extension '{self.input_path.suffix}' for inference type {self.inference_type}. Supported extensions: {supported_extensions}"
                )
        return self

    @cached_property
    def num_input_frames(self) -> int:
        match self.inference_type:
            case InferenceType.TEXT2WORLD:
                return 0
            case InferenceType.IMAGE2WORLD:
                return 1
            case InferenceType.VIDEO2WORLD:
                return 2
            case _:
                assert_never(self.inference_type)


InferenceOverrides = get_overrides_cls(InferenceArguments, exclude=["name"])
