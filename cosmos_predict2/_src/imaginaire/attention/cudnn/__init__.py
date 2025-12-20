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

cuDNN Backend
"""

import torch

from cosmos_predict2._src.imaginaire.attention.utils import safe_log as log

# (ahassani) [11-20-2025] Banning cuDNN until reliability issues are resolved.
# Versions checked: 91300, 91400, 91500
# (ahassani) [12-01-2025]
# 91500 ran on both GB200 and H100 SXM.
CUDNN_DISALLOWED = True

CUDNN_MIN_BACKEND_VERSION = 91300
CUDNN_MIN_FRONTEND_VERSION = [1, 14, 0]


def cudnn_supported() -> bool:
    """
    Returns whether cuDNN Attention is supported in this environment.
    Requirements are:
        * Presence of CUDA Runtime (via PyTorch)
        * Presence of cuDNN and its Python frontend, meeting minimum version requirements

    This check guards imports / dependencies on the cuDNN package.
    """
    if not torch.cuda.is_available():
        log.debug("cuDNN Attention is not supported because PyTorch did not detect CUDA runtime.")
        return False

    try:
        import cudnn

    except ImportError:
        log.debug("cuDNN Attention is not supported because the frontend Python package was not found.")
        return False
    except Exception as e:
        log.debug(f"cuDNN Attention is not supported because importing the frontend Python package failed: {e}")
        return False

    if cudnn.backend_version() < CUDNN_MIN_BACKEND_VERSION:
        log.debug(
            "cuDNN Attention is not supported due to insufficient cuDNN backend version "
            f"{cudnn.backend_version()=}, expected at least {CUDNN_MIN_BACKEND_VERSION=}."
        )
        return False

    cudnn_frontend_version_split = cudnn.__version__.split(".")
    if len(cudnn_frontend_version_split) != 3:
        log.debug(f"Unable to parse cuDNN frontend version {cudnn.__version__}.")
        return False

    try:
        cudnn_frontend_version = [int(x) for x in cudnn_frontend_version_split]
    except ValueError:
        log.debug(f"Unable to parse cuDNN frontend version as an int list: {cudnn.__version__}.")
        return False

    if cudnn_frontend_version < CUDNN_MIN_FRONTEND_VERSION:
        log.debug(
            "cuDNN Attention is not supported due to insufficient cuDNN frontend version "
            f"{cudnn_frontend_version=}, expected at least {CUDNN_MIN_FRONTEND_VERSION=}."
        )
        return False

    return True


CUDNN_SUPPORTED = cudnn_supported()


if CUDNN_SUPPORTED:
    from cosmos_predict2._src.imaginaire.attention.cudnn.functions import cudnn_attention

else:
    from cosmos_predict2._src.imaginaire.attention.cudnn.stubs import cudnn_attention

__all__ = ["cudnn_attention", "CUDNN_SUPPORTED"]
