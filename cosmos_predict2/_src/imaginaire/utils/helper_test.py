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

"""Adapted from:

https://github.com/PyTorchLightning/pytorch-lightning/blob/master/tests/helpers/runif.py
"""

import importlib.metadata
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest
import torch
from loguru import logger
from packaging.version import Version
from pytest import MarkDecorator

from cosmos_predict2._src.imaginaire.utils.device import get_gpu_architecture


class RunIf:
    """RunIf wrapper for conditional skipping of tests.

    Fully compatible with `@pytest.mark`.

    Example:

    ```python
        @RunIf(min_torch="1.8")
        @pytest.mark.parametrize("arg1", [1.0, 2.0])
        def test_wrapper(arg1):
            assert arg1 > 0
    ```
    """

    def __new__(
        cls,
        min_gpus: int = 0,
        min_torch: Optional[str] = None,
        max_torch: Optional[str] = None,
        min_python: Optional[str] = None,
        supported_arch: Optional[List[str]] = None,
        requires_file: Optional[Union[str, List[str]]] = None,
        requires_package: Optional[Union[str, List[str]]] = None,
        **kwargs: Dict[Any, Any],
    ) -> MarkDecorator:
        """Creates a new `@RunIf` `MarkDecorator` decorator.

        :param min_gpus: Min number of GPUs required to run test.
        :param min_torch: Minimum pytorch version to run test.
        :param max_torch: Maximum pytorch version to run test.
        :param min_python: Minimum python version required to run test.
        :param requires_file: File or list of files required to run test.
        :param requires_package: Package name or list of package names required to be installed to run test.
        :param kwargs: Native `pytest.mark.skipif` keyword arguments.
        """
        conditions = []
        reasons = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        if min_torch:
            torch_version = importlib.metadata.version("torch")
            conditions.append(Version(torch_version) < Version(min_torch))
            reasons.append(f"torch>={min_torch}")

        if max_torch:
            torch_version = importlib.metadata.version("torch")
            conditions.append(Version(torch_version) >= Version(max_torch))
            reasons.append(f"torch<{max_torch}")

        if min_python:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if supported_arch:
            if isinstance(supported_arch, str):
                supported_arch = [supported_arch]
            gpu_arch = get_gpu_architecture()
            conditions.extend([gpu_arch not in supported_arch])
            reasons.append(f"supported_arch arch={','.join(supported_arch)}")

        if requires_file:
            if isinstance(requires_file, str):
                requires_file = [requires_file]
            conditions.extend([not Path(file).exists() for file in requires_file])
            reasons.append(f"requires file={','.join(requires_file)}")

        if requires_package:
            if isinstance(requires_package, str):
                requires_package = [requires_package]
            for package in requires_package:
                try:
                    __import__(package)
                except ImportError:
                    conditions.extend([True])
                    reasons.append(f"Package {package} is not installed.")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )


def run_command(
    cmd: str, max_retry_counter: int = 3, is_raise: bool = True, capture_output: bool = True
) -> subprocess.CompletedProcess:
    """Runs a shell command with the ability to retry upon failure.

    Parameters:
    - cmd (str): The shell command to run.
    - max_retry_counter (int): Maximum number of retries if the command fails.
    - is_raise (bool): Whether to raise an exception and exit the program if the command fails after all retries.
    - capture_output (bool): Whether to capture the output of the command.

    Returns:
    - subprocess.CompletedProcess: The result of the command execution.
    """

    retry_counter = 0
    while retry_counter < max_retry_counter:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, check=False)

            # Check if the command was successful (returncode = 0)
            if result.returncode == 0:
                return result

            retry_counter += 1
            logger.debug(
                f"Retry {retry_counter}/{max_retry_counter}: Command '{cmd}' failed with error"
                f" code {result.returncode}. Error message: {result.stderr.strip()}"
            )

        except Exception as e:  # pylint: disable=broad-except
            retry_counter += 1
            logger.debug(f"Retry {retry_counter}/{max_retry_counter}: Command '{cmd}' raised an exception: {e}")

    # If reached here, all retries have failed
    error_message = (
        f"Command '{cmd}' failed after {max_retry_counter} retries. Error code: {result.returncode}. "
        f"Error message: {result.stderr.strip()}"
    )
    if is_raise:
        raise RuntimeError(error_message)

    logger.critical(error_message)
    return result
