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

# From Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

# Description:
# Single point of entry for all generic attention ops (self and cross attention), that tries to
# deliver the best performance possible given any use case (GPU and environment).
#
# SageAttention 2 (https://github.com/thu-ml/sageattention) is the first-priority backend for
# Ada Lovelace (SM89), Hopper (SM90), and Blackwell RTX (SM120, SM121) when installed and the
# corresponding compiled kernels are available (SM89_ENABLED / SM90_ENABLED in sageattention.core).
#
# On Hopper GPUs (i.e. H100, H20, H200), if SageAttention is unavailable, Flash Attention 3 is
# the next best choice. When that is also unavailable, the fallback is cuDNN attention via SDPA.
#
# For all other use cases, we will just use PyTorch's SDPA, but we need to specify backends and
# priorities.
# Flash Attention 2, which is one of the backends, is the best choice for Ampere GPUs (both RTX and
# datacenter-class).
#
# For anything pre-Ampere, the only choice is "memory-efficient" (xformers) FMHA.
#
# For Ada and Blackwell RTX, it is unclear at the moment, so we defer to Flash Attention 2, and
# fallbacks are cuDNN and xformers.
#
# For Blackwell datacenter-class (B200, GB200), cuDNN is the best choice.
#
#
# Dispatching to the desired backends/paths are done by checking the compute capability (really SM
# number, which is just compute capability * 10) of the GPU device the input tensors are on.
#
# Here's a breakdown of relevant compute capabilities:
#
# | GPU / category | Arch  |
# |================|=======|
# | A100           | SM80  |
# | A40            | SM80  |
# | Ampere RTX     | SM86  |
# |----------------|-------|
# | Ada Lovelace   | SM89  |
# |----------------|-------|
# | H20            | SM90  |
# | H100           | SM90  |
# | H200           | SM90  |
# |----------------|-------|
# | B200           | SM100 |
# | Blackwell RTX  | SM120 |
# | Blackwell RTX  | SM121 |
# |----------------|-------|
#

import os
from functools import partial

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cosmos_predict2._src.imaginaire.utils import log

try:
    from flash_attn_3.flash_attn_interface import flash_attn_func

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

_SAGE_ATTENTION_FLAG_SET = bool(int(os.environ.get("COSMOS_ENABLE_SAGE_ATTN", "0")))
try:
    if not _SAGE_ATTENTION_FLAG_SET:
        raise Exception("Sage Attention forbidden because COSMOS_ENABLE_SAGE_ATTN is not set to 1")
    from sageattention import sageattn
    from sageattention.core import SM89_ENABLED as _SAGE_SM89_ENABLED
    from sageattention.core import SM90_ENABLED as _SAGE_SM90_ENABLED

    # SM89_ENABLED covers Ada Lovelace (SM89) and Blackwell RTX (SM120, SM121).
    # SM90_ENABLED covers Hopper (SM90).
    _SAGE_ATTN_2_AVAILABLE = _SAGE_SM89_ENABLED or _SAGE_SM90_ENABLED
except (ImportError, ModuleNotFoundError, Exception) as e:
    sageattn = None
    _SAGE_SM89_ENABLED = False
    _SAGE_SM90_ENABLED = False
    _SAGE_ATTN_2_AVAILABLE = False
    if _SAGE_ATTENTION_FLAG_SET:
        log.warning(f"Error during SageAttention import: {e}")


def get_device_cc(device) -> int:
    """
    Returns the compute capability of a given torch device if it's a CUDA device, otherwise returns 0.

    Args:
        device: torch device.

    Returns:
        device_cc (int): compute capability in the SmXXX format (i.e. 90 for Hopper).
    """
    if torch.cuda.is_available() and torch.version.cuda and device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return major * 10 + minor
    return 0


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    deterministic=False,
    dtype=torch.bfloat16,
):
    supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    is_half = dtype in [torch.bfloat16, torch.float16]
    compute_cap = get_device_cc(q.device)

    if dtype not in supported_dtypes:
        raise NotImplementedError(f"{dtype=} is not supported.")

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    if q_scale is not None:
        q = q * q_scale

    # SageAttention 2 is the first-priority backend for SM89 (Ada Lovelace), SM90 (Hopper), and
    # SM120/SM121 (Blackwell RTX) when the corresponding compiled kernels are available.
    # SM89_ENABLED covers SM89, SM120, SM121; SM90_ENABLED covers SM90.
    if compute_cap == 90 and _SAGE_SM90_ENABLED and is_half:
        return sageattn(q, k, v, tensor_layout="NHD", is_causal=causal, sm_scale=softmax_scale)
    elif compute_cap in [89, 120, 121] and _SAGE_SM89_ENABLED and is_half:
        return sageattn(q, k, v, tensor_layout="NHD", is_causal=causal, sm_scale=softmax_scale)
    elif compute_cap == 90 and FLASH_ATTN_3_AVAILABLE and is_half:
        # If Flash Attention 3 is installed, and the user's running on a Hopper GPU (compute capability
        # 9.0, or SM90), use Flash Attention 3.
        return flash_attn_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0]
    else:
        # If Blackwell or Hopper (SM100 or SM90), cuDNN has native FMHA kernels. The Hopper one is
        # not always as fast as Flash Attention 3, but when Flash Attention is unavailable, it's
        # still a far better choice than Flash Attention 2 (Ampere).
        if compute_cap in [90, 100] and is_half:
            SDPA_BACKENDS = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]
            BEST_SDPA_BACKEND = SDPBackend.CUDNN_ATTENTION
        elif is_half:
            SDPA_BACKENDS = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]
            BEST_SDPA_BACKEND = SDPBackend.FLASH_ATTENTION if compute_cap >= 80 else SDPBackend.EFFICIENT_ATTENTION
        else:
            assert dtype == torch.float32, f"Unrecognized {dtype=}."
            SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION]
            BEST_SDPA_BACKEND = SDPBackend.EFFICIENT_ATTENTION

        if deterministic:
            raise NotImplementedError(
                "Deterministic mode in attention is only supported when Flash Attention 3 is available."
            )

        # Torch 2.6 and later allows priorities for backends, but for older versions
        # we can only run with a specific backend. As long as we pick ones we're certain
        # will work on that device, it should be fine.
        try:
            sdpa_kernel(backends=SDPA_BACKENDS, set_priority_order=True)
            sdpa_kernel_ = partial(sdpa_kernel, set_priority_order=True)
        except TypeError:
            sdpa_kernel_ = sdpa_kernel
            SDPA_BACKENDS = [BEST_SDPA_BACKEND]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        with sdpa_kernel_(backends=SDPA_BACKENDS):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=causal,
                dropout_p=dropout_p,
                scale=softmax_scale,
            )

        out = out.transpose(1, 2).contiguous()
        return out
