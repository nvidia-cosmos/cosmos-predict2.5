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

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

NATTEN Backend
"""

import torch

from cosmos_predict2._src.imaginaire.attention.utils import safe_log as log

# 0.21.5.dev1 patches some varlen issues
# 0.21.5.dev2 adds torch compile support
# 0.21.5.dev3 fixes a few compat issues for older torch versions
# 0.21.5.dev6 gqa/mqa support
# 0.21.5.dev9 fixes attention merging
NATTEN_MIN_VERSION = "0.21.5.dev9"


def _parse_natten_version(version_str: str) -> tuple[list[int], int | None] | None:
    """
    Parse a NATTEN version string into components.

    Parameters:
        version_str (str): Version string (e.g., "0.21.5" or "0.21.5.dev12").

    Returns:
        tuple: ([major, minor, patch], dev_number) where dev_number is None for release versions.
        None: If parsing fails.
    """
    version_split = version_str.split(".")
    if len(version_split) < 3 or len(version_split) > 4:
        return None

    try:
        version = [int(x) for x in version_split[:3]]
        dev = None
        if len(version_split) >= 4:
            # If there's a 4th component, it must be a dev version
            if not version_split[3].startswith("dev"):
                return None  # Invalid: non-dev suffix
            dev = int(version_split[3].replace("dev", ""))
        return (version, dev)
    except ValueError:
        return None


def _compare_natten_versions(version1_str: str, version2_str: str) -> int | None:
    """
    Compare two NATTEN version strings.

    Parameters:
        version1_str (str): First version string (e.g., "0.21.5.dev12").
        version2_str (str): Second version string (e.g., "0.21.5.dev9").

    Returns:
        int: 1 if version1 > version2, 0 if equal, -1 if version1 < version2.
        None: If either version cannot be parsed.
    """
    parsed1 = _parse_natten_version(version1_str)
    parsed2 = _parse_natten_version(version2_str)

    if parsed1 is None or parsed2 is None:
        return None

    version1, dev1 = parsed1
    version2, dev2 = parsed2

    # Compare base versions
    if version1 > version2:
        return 1
    if version1 < version2:
        return -1

    # Same base version, check dev versions
    if dev2 is None:
        # version2 is release version
        if dev1 is None:
            return 0  # Both release
        else:
            return -1  # version1 is dev, version2 is release (release > dev)

    if dev1 is None:
        # version1 is release, version2 is dev
        return 1  # Release > dev

    # Both are dev versions, compare dev numbers
    if dev1 > dev2:
        return 1
    elif dev1 < dev2:
        return -1
    else:
        return 0


def natten_version_satisfies(min_version_str: str) -> bool:
    """
    Check if the installed NATTEN version satisfies a specific minimum version requirement.

    This allows checking for feature-specific version requirements without raising the
    global minimum version for all NATTEN features.

    Parameters:
        min_version_str (str): Minimum version string (e.g., "0.21.5" or "0.21.5.dev12").

    Returns:
        bool: True if NATTEN is installed and meets the minimum version requirement.

    Raises:
        ValueError: If min_version_str is not a valid version string.

    Example:
        >>> # Check if NATTEN >= 0.21.5.dev12
        >>> if natten_version_satisfies("0.21.5.dev12"):
        >>>     # Use varlen features
        >>>     pass
    """
    try:
        import natten
    except (ImportError, Exception):
        return False

    # Compare installed version with minimum requirement
    comparison = _compare_natten_versions(natten.__version__, min_version_str)

    if comparison is None:
        raise ValueError(
            f"Invalid minimum version string: {min_version_str}. "
            f"Expected format: 'major.minor.patch' or 'major.minor.patch.devN'"
        )

    return comparison >= 0  # installed >= minimum


def natten_supported() -> bool:
    """
    Returns whether NATTEN is supported in this environment.
    Requirements are:
        * Presence of CUDA Runtime (via PyTorch)
        * Presence of NATTEN, meeting minimum version requirements

    This check guards imports / dependencies on the NATTEN package.
    """
    if not torch.cuda.is_available():
        log.debug("NATTEN Attention is not supported because PyTorch did not detect CUDA runtime.")
        return False

    try:
        import natten
    except ImportError:
        log.debug("NATTEN Attention is not supported because the Python package was not found.")
        return False
    except Exception as e:
        log.debug(f"NATTEN Attention is not supported because importing the Python package failed: {e}")
        return False

    # Use the new version comparison API
    comparison = _compare_natten_versions(natten.__version__, NATTEN_MIN_VERSION)

    if comparison is None:
        log.debug(f"Unable to parse NATTEN version {natten.__version__}.")
        return False

    if comparison >= 0:
        return True

    log.debug(
        f"NATTEN Attention is not supported due to insufficient NATTEN version "
        f"{natten.__version__}, expected at least {NATTEN_MIN_VERSION}."
    )
    return False


NATTEN_SUPPORTED = natten_supported()

if NATTEN_SUPPORTED:
    from cosmos_predict2._src.imaginaire.attention.natten.functions import (
        natten_attention,
        natten_multi_dim_attention,
        natten_multi_dim_attention_varlen,
    )

else:
    from cosmos_predict2._src.imaginaire.attention.natten.stubs import (
        natten_attention,
        natten_multi_dim_attention,
        natten_multi_dim_attention_varlen,
    )

__all__ = ["natten_attention", "natten_multi_dim_attention", "natten_multi_dim_attention_varlen", "NATTEN_SUPPORTED"]
