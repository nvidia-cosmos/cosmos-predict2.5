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
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest
from typing_extensions import Self

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "tests/data"


@pytest.fixture(scope="module")
def original_datadir(request: pytest.FixtureRequest) -> Path:
    relative_path = request.path.with_suffix("").relative_to(ROOT_DIR)
    return DATA_DIR / relative_path


@cache
def _get_available_gpus() -> int:
    try:
        return len(subprocess.check_output(["nvidia-smi", "--list-gpus"], text=True).splitlines())
    except Exception as e:
        print(f"WARNING: Failed to get available GPUs: {e}")
        return 0


@dataclass(frozen=True)
class _Args:
    worker_id: str
    worker_index: int

    enable_manual: bool
    num_gpus: int | None
    levels: list[int] | None

    @classmethod
    def from_config(cls, config: pytest.Config) -> Self:
        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
        if worker_id == "master":
            worker_index = 0
        else:
            worker_index = int(worker_id.removeprefix("gw"))

        return cls(
            worker_id=worker_id,
            worker_index=worker_index,
            enable_manual=config.option.manual,
            num_gpus=config.option.num_gpus,
            levels=config.option.levels,
        )


_ARGS: _Args = None  # type: ignore


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--manual", action="store_true", default=False, help="Run manual tests")
    parser.addoption("--num-gpus", default=None, type=int, help="Run tests with the specified number of GPUs")
    parser.addoption(
        "--levels", default=None, type=int, choices=[0, 1, 2], nargs="*", help="Run tests with the specified levels"
    )


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    num_gpus: int | None = config.option.num_gpus
    if num_gpus is None:
        return 1
    if num_gpus == 0:
        # CPU
        return None

    available_gpus = _get_available_gpus()
    if available_gpus < num_gpus:
        raise ValueError(f"Not enough GPUs available. Required: {num_gpus}, Available: {available_gpus}")
    return available_gpus // num_gpus


def pytest_configure(config: pytest.Config):
    global _ARGS
    _ARGS = _Args.from_config(config)

    if _ARGS.worker_index == "master":
        return

    if _ARGS.worker_index > 1:
        if _ARGS.num_gpus is None:
            raise NotImplementedError(f"Running parallel tests requires --num-gpus to be set.")

    # Check if there are enough GPUs available.
    if _ARGS.num_gpus is not None and _ARGS.num_gpus > 0:
        required_gpus = _ARGS.num_gpus * _ARGS.worker_index
        available_gpus = _get_available_gpus()
        if available_gpus < required_gpus:
            raise ValueError(f"Not enough GPUs available. Required: {required_gpus}, Available: {available_gpus}")


def _parse_level_marker(mark: pytest.Mark) -> int:
    if len(mark.args) != 1:
        raise ValueError(f"Invalid arguments: {mark.args}")
    if mark.kwargs:
        raise ValueError(f"Invalid keyword arguments: {mark.kwargs}")
    level = int(mark.args[0])
    if level not in [0, 1, 2]:
        raise ValueError(f"Invalid level: {level}")
    return level


def _parse_gpus_marker(mark: pytest.Mark) -> int:
    if len(mark.args) != 1:
        raise ValueError(f"Invalid arguments: {mark.args}")
    if mark.kwargs:
        raise ValueError(f"Invalid keyword arguments: {mark.kwargs}")
    required_gpus = int(mark.args[0])
    if required_gpus not in [0, 1, 8]:
        raise ValueError(f"Invalid number of GPUs: {required_gpus}")
    return required_gpus


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    for item in items:
        manual_mark = item.get_closest_marker(name="manual")
        level_mark = item.get_closest_marker(name="level")
        gpus_mark = item.get_closest_marker(name="gpus")
        try:
            level = _parse_level_marker(level_mark) if level_mark else 0
            gpus = _parse_gpus_marker(gpus_mark) if gpus_mark else 0
        except ValueError as e:
            pytest.fail(f"Invalid marker on test {item.name}: {e}")
            assert False, "unreachable"

        # Check if the test should be skipped
        if not _ARGS.enable_manual and manual_mark is not None:
            item.add_marker(pytest.mark.skip(reason="test requires --manual"))
        if _ARGS.levels is not None and level not in _ARGS.levels:
            item.add_marker(pytest.mark.skip(reason=f"test requires --levels={level}"))
        if _ARGS.num_gpus is not None and gpus != _ARGS.num_gpus:
            item.add_marker(pytest.mark.skip(reason=f"test requires --num-gpus={gpus}"))
        available_gpus = _get_available_gpus()
        if gpus > available_gpus:
            item.add_marker(
                pytest.mark.skip(reason=f"test requires {gpus} GPUs, but only {available_gpus} are available")
            )


def pytest_runtest_setup(item: pytest.Item):
    gpus_mark = item.get_closest_marker(name="gpus")
    try:
        gpus = _parse_gpus_marker(gpus_mark) if gpus_mark else 0
    except ValueError as e:
        pytest.fail(f"Invalid marker on test {item.name}: {e}")
        assert False, "unreachable"

    # Limit the number of GPUs used by the test
    if gpus > 0:
        device_start = _ARGS.worker_index * gpus
        device_end = device_start + gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(device_start, device_end)))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NUM_GPUS"] = str(gpus)
