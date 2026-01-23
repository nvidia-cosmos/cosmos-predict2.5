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

NATTEN_MIN_RELEASE_VERSION = [0, 21, 5]
# 0.21.5.dev1 patches some varlen issues
# 0.21.5.dev2 adds torch compile support
# 0.21.5.dev3 fixes a few compat issues for older torch versions
# 0.21.5.dev6 gqa/mqa support
# 0.21.5.dev9 fixes attention merging
NATTEN_MIN_DEV_VERSION = ([0, 21, 5], 9)


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

    natten_version_split = natten.__version__.split(".")
    if len(natten_version_split) < 3 or len(natten_version_split) > 4:
        log.debug(f"Unable to parse NATTEN version {natten.__version__}.")
        return False

    try:
        natten_version = [int(x) for x in natten_version_split[:3]]
        natten_version_dev = None
        if len(natten_version_split) >= 4 and natten_version_split[3].startswith("dev"):
            natten_version_dev = int(natten_version_split[3].replace("dev", ""))

    except ValueError:
        log.debug(f"Unable to parse NATTEN version as an int list: {natten.__version__}.")
        return False

    if (natten_version_dev is None and natten_version >= NATTEN_MIN_RELEASE_VERSION) or (
        natten_version_dev is not None
        and natten_version >= NATTEN_MIN_DEV_VERSION[0]
        and natten_version_dev >= NATTEN_MIN_DEV_VERSION[1]
    ):
        return True

    log.debug(
        "NATTEN Attention is not supported due to insufficient NATTEN version "
        f"{natten.__version__=}, expected at least {NATTEN_MIN_RELEASE_VERSION=}, "
        f"or {NATTEN_MIN_DEV_VERSION=}."
    )
    return False


NATTEN_SUPPORTED = natten_supported()

if NATTEN_SUPPORTED:
    from cosmos_predict2._src.imaginaire.attention.natten.functions import natten_attention, natten_multi_dim_attention

else:
    from cosmos_predict2._src.imaginaire.attention.natten.stubs import natten_attention, natten_multi_dim_attention

__all__ = ["natten_attention", "natten_multi_dim_attention", "NATTEN_SUPPORTED"]
