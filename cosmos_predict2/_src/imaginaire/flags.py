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

"""Feature flags."""

import os
from dataclasses import dataclass


def _parse_bool(value: str) -> bool:
    """Parse string to a boolean."""
    return value.lower() in ["true", "1", "yes", "y"]


def _get_bool(name: str, default: bool) -> bool:
    """Get a boolean flag from the environment."""
    value = os.environ.get(name, "")
    if not value:
        return default
    return _parse_bool(value)


TRAINING = _get_bool("COSMOS_TRAINING", True)
"""Whether to enable training features."""

INTERNAL = _get_bool("COSMOS_INTERNAL", False)
"""Whether to enable internal (nvidia-only) features."""

SMOKE = _get_bool("COSMOS_SMOKE", False)
"""Whether to enable smoke test.

Disables expensive operations such as checkpoint loading.
"""

VERBOSE = _get_bool("COSMOS_VERBOSE", INTERNAL)
"""Whether to enable verbose output."""

EXPERIMENTAL_CHECKPOINTS = _get_bool("COSMOS_EXPERIMENTAL_CHECKPOINTS", INTERNAL)
"""Whether to enable experimental checkpoints."""


@dataclass
class Flags:
    internal: bool = INTERNAL
    training: bool = TRAINING
    smoke: bool = SMOKE
    verbose: bool = VERBOSE
    experimental_checkpoints: bool = EXPERIMENTAL_CHECKPOINTS


FLAGS = Flags()
"""Convenience object for accessing flags."""
