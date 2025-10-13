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

"""Base model inference script."""

from pathlib import Path
from typing import Annotated

import pydantic
import tyro

from cosmos_predict2.config import (
    InferenceArguments,
    InferenceOverrides,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)
from cosmos_predict2.init import cleanup_environment, init_environment, init_output_dir


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path to the inference parameter files."""
    setup: SetupArguments
    """Setup arguments."""
    overrides: InferenceOverrides
    """Inference parameter overrides."""


def main(
    args: Args,
):
    inference_samples = InferenceArguments.from_files(args.input_files, overrides=args.overrides)
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_predict2.inference import Inference

    inference = Inference(args.setup)
    inference.generate(inference_samples, output_dir=args.setup.output_dir)


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(Args, description=__doc__, console_outputs=is_rank0(), config=(tyro.conf.OmitArgPrefixes,))
    except Exception as e:
        handle_tyro_exception(e)
    main(args)

    cleanup_environment()
