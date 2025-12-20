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

Flash Attention v3 (flash3) Backend
"""

import torch

from cosmos_predict2._src.imaginaire.attention.utils import safe_log as log

FLASH_ATTENTION_V3_MIN_VERSION = [3, 0, 0, 0]
FLASH_ATTENTION_V3_MAX_VERSION = [3, 0, 0, 1]


def flash3_supported() -> bool:
    """
    Returns whether Flash Attention is supported in this environment.
    Requirements are:
        * Presence of CUDA Runtime (via PyTorch)
        * Presence of Flash Attention, meeting minimum version requirements

    This check guards imports / dependencies on the Flash Attention package.
    """
    if not torch.cuda.is_available():
        log.debug("Flash Attention v3 is not supported because PyTorch did not detect CUDA runtime.")
        return False

    try:
        import flash_attn_3

    except ImportError:
        log.debug("Flash Attention v3 is not supported because the Python package was not found.")
        return False
    except Exception as e:
        log.debug(f"Flash Attention v3 is not supported because importing the Python package failed: {e}")
        return False

    flash3_version_str = None
    if not hasattr(flash_attn_3, "__version__"):
        from importlib.metadata import version

        flash3_version_str = version("flash_attn_3")
    else:
        flash3_version_str = flash_attn_3.__version__

    flash3_version_split = flash3_version_str.replace("b", ".").split(".")
    if len(flash3_version_split) != 4:
        log.debug(f"Unable to parse Flash Attention v3 version {flash3_version_str}.")
        return False

    try:
        flash3_version = [int(x) for x in flash3_version_split]

    except ValueError:
        log.debug(f"Unable to parse Flash Attention v3 version as an int list: {flash3_version_str}.")
        return False

    if flash3_version > FLASH_ATTENTION_V3_MAX_VERSION or flash3_version < FLASH_ATTENTION_V3_MIN_VERSION:
        log.debug(
            "Flash Attention v3 build is not supported; this backend only supports versions "
            f"{FLASH_ATTENTION_V3_MIN_VERSION} through {FLASH_ATTENTION_V3_MAX_VERSION}, got "
            f"{flash3_version}."
        )
        return False

    return True


FLASH3_SUPPORTED = flash3_supported()


if FLASH3_SUPPORTED:
    from cosmos_predict2._src.imaginaire.attention.flash3.functions import flash3_attention

else:
    from cosmos_predict2._src.imaginaire.attention.flash3.stubs import flash3_attention

__all__ = ["flash3_attention", "FLASH3_SUPPORTED"]
