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

import pytest


def pytest_addoption(parser):
    parser.addoption("--L0", action="store_true", default=False, help="L0 tests")
    parser.addoption("--L1", action="store_true", default=False, help="L1 tests")
    parser.addoption("--L2", action="store_true", default=False, help="L2 tests")
    parser.addoption("--all", action="store_true", default=False, help="Run all tests")
    parser.addoption("--CPU", action="store_true", default=False, help="Run CPU tests only")
    parser.addoption("--GPU", action="store_true", default=False, help="Run GPU tests only")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--all"):
        return
    for item in items:
        # L0/L1/L2 test rules
        if "L0" in item.keywords:
            continue
        elif "L1" in item.keywords:
            if not config.getoption("--L1"):
                # If test is marked L1 but --L1 is not passed, skip it
                item.add_marker(pytest.mark.skip(reason=f"SKIPPING L1 test : {item.name}, run with --L1"))
        elif "L2" in item.keywords:
            if not config.getoption("--L2"):
                # If test is marked L2 but --L2 is not passed, skip it
                item.add_marker(pytest.mark.skip(reason=f"SKIPPING L2 test : {item.name}, run with --L2"))
        else:
            # If test is not marked L0/L1/L2, skip it
            skip_unknown = pytest.mark.skip(reason=f"SKIPPING UNMARKED test. Please mark {item.name} as L0/L1/L2")
            item.add_marker(skip_unknown)

        # CPU/GPU test rules
        if "CPU" not in item.keywords and "GPU" not in item.keywords:
            item.add_marker(pytest.mark.GPU)
        if config.getoption("--CPU") and "GPU" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="SKIPPING GPU test in CPU mode..."))
        elif config.getoption("--GPU") and "CPU" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="SKIPPING CPU test in GPU mode..."))
        else:
            continue

        # Flaky test rules
        if "flaky" in item.keywords and not item.config.getoption("--run-flaky"):
            item.add_marker(pytest.mark.skip(reason=f"SKIPPING FLAKY test : {item.name}."))
