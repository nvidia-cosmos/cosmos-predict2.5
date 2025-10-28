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

"""Auto multiview model inference script."""

from pathlib import Path
from typing import Annotated, Optional

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir
from pydantic import model_validator

from cosmos_predict2.config import handle_tyro_exception, is_rank0
from cosmos_predict2.multiview_config import (
    MultiviewInferenceArguments,
    MultiviewInferenceArgumentsWithInputPaths,
    MultiviewInferenceOverrides,
    MultiviewSetupArguments,
)


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_files: Annotated[Optional[list[Path]], tyro.conf.arg(aliases=("-i",))] = None
    """Path to the inference parameter files."""
    setup: MultiviewSetupArguments
    """Setup arguments."""
    overrides: MultiviewInferenceOverrides
    """Inference parameter overrides."""

    @model_validator(mode="after")
    def check_xor(self) -> "Args":
        if (self.input_files is not None) != self.setup.use_config_dataloader:
            return self
        raise ValueError(
            "Either `input_files` should be provided or `use_config_dataloader` should be set, but not both."
        )


def main(
    args: Args,
):
    if args.setup.use_config_dataloader:
        overrides = args.overrides.model_dump(exclude_none=True)
        if "name" not in overrides:
            overrides["name"] = args.setup.experiment
        inference_samples = MultiviewInferenceArguments.model_validate(overrides)
    else:
        assert args.input_files is not None
        inference_samples = MultiviewInferenceArgumentsWithInputPaths.from_files(
            args.input_files, overrides=args.overrides
        )
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_predict2.multiview import MultiviewInference

    multiview_inference = MultiviewInference(args.setup)
    multiview_inference.generate(inference_samples, output_dir=args.setup.output_dir)


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(Args, description=__doc__, console_outputs=is_rank0(), config=(tyro.conf.OmitArgPrefixes,))
    except Exception as e:
        handle_tyro_exception(e)
    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
