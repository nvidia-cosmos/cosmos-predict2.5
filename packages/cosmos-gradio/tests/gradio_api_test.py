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

# Run tests (L2 tests require --L2 flag):
#   pytest tests/gradio_api_test.py -v -s --L2
#   pytest tests/gradio_api_test.py::test_request -v -s --L2
#   python tests/gradio_api_test.py

import pytest

from tests.bootstrap_test_server import default_request
from tests.gradio_test_harness import GradioTestHarness

env_vars = {
    "NUM_GPUS": "2",
    "WORKER_TIMEOUT": "10",  # for testing timeout we need to set a shorter timeout
}


@pytest.fixture(scope="module")
def harness():
    """
    Module-scoped fixture that starts the server once and reuses it for all tests.
    The server is started before the first test and shut down after the last test.
    """
    with GradioTestHarness(server_module="tests.bootstrap_test_server", env_vars=env_vars) as h:
        yield h


@pytest.mark.L2
def test_default_request(harness: GradioTestHarness):
    ret = harness.send_sample_api_default_request()
    assert ret


@pytest.mark.L2
def test_request(harness: GradioTestHarness):
    ret = harness.send_sample_ui(default_request)
    assert ret
    ret = harness.send_sample_api(default_request)
    assert ret


@pytest.mark.L2
def test_exception(harness: GradioTestHarness):
    ret = harness.send_sample_api({"prompt": "test_exception"})
    assert not ret, "API test should fail"


@pytest.mark.L2
def test_timeout(harness: GradioTestHarness):
    ret = harness.send_sample_api({"prompt": "timeout"})
    assert not ret, "API test should fail"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
