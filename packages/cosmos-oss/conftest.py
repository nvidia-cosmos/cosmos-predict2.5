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

MAX_GPUS = int(os.environ.get("MAX_GPUS", "8"))
_ALLOWED_GPUS_BY_LEVEL = {
    0: [0, 1],
    1: [0, 1, MAX_GPUS],
    2: [0, 1, MAX_GPUS],
}


@pytest.fixture(scope="module")
def original_datadir(request: pytest.FixtureRequest) -> Path:
    root_dir = request.config.rootpath
    relative_path = request.path.with_suffix("").relative_to(root_dir)
    return root_dir / "tests/data" / relative_path


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
    levels: set[int] | None

    @classmethod
    def from_config(cls, config: pytest.Config) -> Self:
        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
        if worker_id == "master":
            worker_index = 0
        else:
            worker_index = int(worker_id.removeprefix("gw"))

        if config.option.levels:
            levels = set(map(int, config.option.levels.split(",")))
            if levels.difference([0, 1, 2]):
                raise ValueError(f"Invalid levels: {levels}")
        else:
            levels = None

        return cls(
            worker_id=worker_id,
            worker_index=worker_index,
            enable_manual=config.option.manual,
            num_gpus=config.option.num_gpus,
            levels=levels,
        )


_ARGS: _Args = None  # type: ignore


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--manual", action="store_true", default=False, help="Run manual tests")
    parser.addoption("--num-gpus", default=None, type=int, help="Run tests with the specified number of GPUs")
    parser.addoption("--levels", default=None, help="Run tests with the specified levels (comma-separated list)")


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

    if _ARGS.worker_id == "master":
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


def _get_marker(item: pytest.Item, name: str) -> pytest.Mark | None:
    markers = list(item.iter_markers(name=name))
    if not markers:
        return None
    if len(markers) != 1:
        raise ValueError(f"Multiple markers found for {name}: {markers}")
    return markers[0]


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
    if required_gpus not in [0, 1, MAX_GPUS]:
        raise ValueError(f"Invalid number of GPUs: {required_gpus}")
    return required_gpus


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    for item in items:
        manual_mark = _get_marker(item, "manual")
        level_mark = _get_marker(item, "level")
        gpus_mark = _get_marker(item, "gpus")
        try:
            level = _parse_level_marker(level_mark) if level_mark else 0
            gpus = _parse_gpus_marker(gpus_mark) if gpus_mark else 0
        except ValueError as e:
            pytest.fail(f"Invalid marker on test {item.name}: {e}")
            assert False, "unreachable"

        allowed_gpus = _ALLOWED_GPUS_BY_LEVEL[level]
        if gpus not in allowed_gpus:
            pytest.fail(f"Level {level} tests must have {allowed_gpus} GPUs, but {item.name} has {gpus} GPUs")

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

    # Exclude skipped tests
    selected_items = []
    deselected_items = []
    for item in items:
        if item.get_closest_marker("skip"):
            deselected_items.append(item)
            continue
        selected_items.append(item)
    items[:] = selected_items
    config.hook.pytest_deselected(items=deselected_items)


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

    # Set master port to a unique port for each worker.
    os.environ["MASTER_PORT"] = str(12341 + _ARGS.worker_index)
