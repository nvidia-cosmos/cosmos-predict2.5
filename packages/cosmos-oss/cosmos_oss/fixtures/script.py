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

"""Inference/training script test fixtures."""

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest


@dataclass(kw_only=True, frozen=True)
class ScriptConfig:
    script: str
    """Script name."""
    gpus: int = 1
    """Minimum number of GPUs."""
    levels: list[int] = field(default_factory=lambda: [0, 1, 2])
    """Test levels."""
    marks: list[pytest.MarkDecorator, pytest.Mark] = field(default_factory=list)
    """Additional pytest marks."""

    @property
    def name(self) -> str:
        return self.script.removesuffix(".sh")


@dataclass(kw_only=True, frozen=True)
class ScriptRunner:
    request: pytest.FixtureRequest
    tmp_path_factory: pytest.TempPathFactory
    tmp_path: Path

    @property
    def output_dir(self) -> Path:
        test_name = self.request.node.name

        if "[" in test_name and "]" in test_name:
            base_part, param_part = test_name.split("[", 1)
            param_part = param_part.rstrip("]").replace("/", "_").replace("-", "_")
            sanitized_name = f"{base_part}_{param_part}"
        else:
            sanitized_name = test_name.replace("/", "_").replace("-", "_")

        return (Path("outputs/pytest") / sanitized_name).resolve()

    def get_env(
        self,
        *,
        torchrun_args: list[str] | None = None,
        inference_args: list[str] | None = None,
        train_args: list[str] | None = None,
    ) -> dict:
        num_gpus = os.environ["NUM_GPUS"]
        master_port = os.environ["MASTER_PORT"]
        env = {
            "INPUT_DIR": str(self.request.config.rootpath),
        }
        env |= dict(os.environ)
        env |= {
            "COSMOS_INTERNAL": "0",
            "COSMOS_SMOKE": "0",
            "COSMOS_VERBOSE": "0",
            "OUTPUT_DIR": f"{self.output_dir}",
            "TMP_DIR": f"{self.tmp_path}/tmp",
            "IMAGINAIRE_OUTPUT_ROOT": f"{self.tmp_path}/imaginaire4-output",
            "TORCHRUN_ARGS": " ".join(
                [
                    f"--nproc_per_node={num_gpus}",
                    f"--master_port={master_port}",
                    *(torchrun_args or []),
                ]
            ),
            "INFERENCE_ARGS": " ".join(
                [
                    *(inference_args or []),
                ]
            ),
            "TRAIN_ARGS": " ".join(
                [
                    "job.wandb_mode=disabled",
                    "~trainer.callbacks.wandb",
                    "~trainer.callbacks.wandb_10x",
                    # Only run logging/validation/checkpoint at the end
                    "trainer.logging_iter=1000",
                    "trainer.validation_iter=1000",
                    "checkpoint.save_iter=1000",
                    *(train_args or []),
                ]
            ),
        }
        return env

    @property
    def env_level_0(self) -> dict:
        return self.get_env(
            inference_args=[
                "--disable-guardrails",
            ],
            train_args=[
                "trainer.max_iter=2",
            ],
        ) | {
            "COSMOS_SMOKE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }

    @property
    def env_level_1(self) -> dict:
        return self.get_env(
            inference_args=[
                "--disable-guardrails",
            ],
            train_args=[
                "trainer.max_iter=5",
            ],
        )

    @property
    def env_level_2(self) -> dict:
        return self.get_env(
            train_args=[
                "trainer.max_iter=20",
            ],
        )

    def run(self, script: str, env: dict):
        subprocess.check_call(["bash", "-euxo", "pipefail", script], cwd=self.request.config.rootpath, env=env)


@pytest.fixture
def script_runner(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, tmp_path: Path
) -> ScriptRunner:
    return ScriptRunner(request=request, tmp_path_factory=tmp_path_factory, tmp_path=tmp_path)


def extract_bash_commands(md_file: Path) -> list[str]:
    content = md_file.read_text()
    pattern = r"```(bash|shell)([^\n]*)\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    scripts = []
    for lang, attrs, block_content in matches:
        if "exclude=true" in attrs.lower():
            continue

        lines = []
        for line in block_content.strip().split("\n"):
            if line.strip() and not line.strip().startswith("#"):
                line = line.split("#")[0].rstrip()
                # Replace --nproc_per_node with dynamic NUM_GPUS value
                line = re.sub(r"--nproc_per_node=\d+", "--nproc_per_node=$NUM_GPUS", line)
                line = re.sub(r"--master_port=\d+", "--master_port=$MASTER_PORT", line)
                if line:
                    lines.append(line)

        if lines:
            script = "\n".join(lines)
            scripts.append(script)

    return scripts
