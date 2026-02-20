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
import platform
from pathlib import Path

import pytest
from cosmos_oss.fixtures.script import ScriptConfig, ScriptRunner
from cosmos_oss.fixtures.script import script_runner as script_runner

MAX_GPUS = int(os.environ.get("MAX_GPUS", "8"))

_CURRENT_DIR = Path(__file__).parent.absolute()
_SCRIPT_DIR = _CURRENT_DIR / "docs_test"

SCRIPT_CONFIGS = [
    ScriptConfig(
        script="base.sh",
    ),
    ScriptConfig(
        script="base_model_offload.sh",
    ),
    ScriptConfig(
        script="base_distilled.sh",
    ),
    ScriptConfig(
        script="diffusers_text2image.sh",
        marks=[pytest.mark.manual] if platform.machine() == "aarch64" else [],
    ),
    ScriptConfig(
        script="multiview.sh",
        gpus=MAX_GPUS,
    ),
    ScriptConfig(
        script="robot_multiview_agibot.sh",
        gpus=MAX_GPUS,
    ),
    ScriptConfig(
        script="action_conditioned.sh",
    ),
    ScriptConfig(
        script="post-training_video2world_cosmos_nemo_assets.sh",
        gpus=MAX_GPUS,
    ),
    ScriptConfig(
        script="post-training_video2world_cosmos_groot.sh",
        gpus=MAX_GPUS,
    ),
    ScriptConfig(
        script="distilled_action.sh",
    ),
]


@pytest.mark.level(0)
@pytest.mark.gpus(1)
@pytest.mark.parametrize(
    "cfg", [pytest.param(cfg, id=cfg.name, marks=cfg.marks) for cfg in SCRIPT_CONFIGS if 0 in cfg.levels]
)
def test_level_0(cfg: ScriptConfig, script_runner: ScriptRunner):
    script_runner.run(f"{_SCRIPT_DIR}/{cfg.script}", script_runner.env_level_0)


@pytest.mark.level(1)
@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(cfg, id=cfg.name, marks=[pytest.mark.gpus(cfg.gpus), *cfg.marks])
        for cfg in SCRIPT_CONFIGS
        if 1 in cfg.levels
    ],
)
def test_level_1(cfg: ScriptConfig, script_runner: ScriptRunner):
    script_runner.run(f"{_SCRIPT_DIR}/{cfg.script}", script_runner.env_level_1)


@pytest.mark.level(2)
@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(cfg, id=cfg.name, marks=[pytest.mark.gpus(MAX_GPUS), *cfg.marks])
        for cfg in SCRIPT_CONFIGS
        if 2 in cfg.levels
    ],
)
def test_level_2(cfg: ScriptConfig, script_runner: ScriptRunner):
    script_runner.run(f"{_SCRIPT_DIR}/{cfg.script}", script_runner.env_level_2)
