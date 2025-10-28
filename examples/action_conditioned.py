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

"""Action-conditioned Video2World inference script."""

from pathlib import Path
from typing import Annotated

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_predict2.action_conditioned_config import (
    ActionConditionedInferenceArguments,
    ActionConditionedInferenceOverrides,
    ActionConditionedSetupArguments,
)
from cosmos_predict2.config import (
    handle_tyro_exception,
    is_rank0,
)


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_file: Annotated[Path, tyro.conf.arg(aliases=("-i",))]
    """Path to the inference parameter files."""
    setup: ActionConditionedSetupArguments
    """Setup arguments."""
    overrides: ActionConditionedInferenceOverrides
    """Inference parameter overrides."""


def main(args: Args) -> None:
    inference_args = ActionConditionedInferenceArguments.from_files([args.input_file], overrides=args.overrides)[0]
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_predict2.action_conditioned import inference

    inference(args.setup, inference_args)


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(Args, description=__doc__, console_outputs=is_rank0(), config=(tyro.conf.OmitArgPrefixes,))
    except Exception as e:
        handle_tyro_exception(e)
    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
