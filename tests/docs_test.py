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
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from pytest_regressions.data_regression import DataRegressionFixture


_CURRENT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CURRENT_DIR.parent


def _get_env(tmp_path: Path):
    return (
        {
            "INPUT_DIR": _ROOT_DIR,
            "COSMOS_VERBOSE": "0",
        }
        | dict(os.environ)
        | {
            "COSMOS_INTERNAL": "0",
            "COSMOS_SMOKE": "0",
            "OUTPUT_DIR": f"{tmp_path}/output",
            "TMP_DIR": f"{tmp_path}/tmp",
            "IMAGINAIRE_OUTPUT_ROOT": f"{tmp_path}/imaginaire4-output",
        }
    )


_SANITIZE_KEYS = ["_target_", "cache_augment_fn", "type", "load_path"]


def _sanitize_config(config: dict):
    """Remove unstable config entries (lambda functions, local paths)."""
    for key, value in list(config.items()):
        if key in _SANITIZE_KEYS:
            del config[key]
        elif isinstance(value, dict):
            _sanitize_config(value)


@pytest.mark.gpus(1)
@pytest.mark.parametrize(
    "test_script",
    [
        pytest.param("base.sh", id="base"),
        pytest.param("multiview.sh", id="multiview"),
        # pytest.param("action_conditioned.sh", id="action_conditioned"),
        pytest.param(
            "post-training_video2world_cosmos_nemo_assets.sh", id="post_training_video2world_cosmos_nemo_assets"
        ),
    ],
)
def test_smoke(test_script: str, tmp_path: Path, data_regression: "DataRegressionFixture"):
    cmd = [
        f"{_CURRENT_DIR}/docs_test/{test_script}",
    ]
    env = _get_env(tmp_path) | {"COSMOS_SMOKE": "1"}
    output_dir = Path(env["OUTPUT_DIR"])
    subprocess.check_call(cmd, cwd=_ROOT_DIR, env=env)

    config = yaml.safe_load((output_dir / "config.yaml").read_text())
    _sanitize_config(config)
    data_regression.check(config)


@pytest.mark.gpus(8)
@pytest.mark.parametrize(
    "test_script",
    [
        pytest.param("base.sh", id="base", marks=pytest.mark.level(1)),
        pytest.param("multiview.sh", id="multiview", marks=pytest.mark.level(1)),
        # pytest.param("action_conditioned.sh", id="action_conditioned", marks=pytest.mark.level(1)),
        pytest.param(
            "post-training_video2world_cosmos_nemo_assets.sh",
            id="post_training_video2world_cosmos_nemo_assets",
            marks=pytest.mark.level(2),
        ),
    ],
)
def test_full(test_script: str, tmp_path: Path):
    cmd = [
        f"{_CURRENT_DIR}/docs_test/{test_script}",
    ]
    subprocess.check_call(
        cmd,
        cwd=_ROOT_DIR,
        env=_get_env(tmp_path),
    )
