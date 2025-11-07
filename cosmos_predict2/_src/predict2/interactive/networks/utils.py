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

from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange


def apply_adaln(
    x_b_t_h_w_d: torch.Tensor,
    norm_layer: torch.nn.Module,
    scale_b_t_1_1_d: torch.Tensor,
    shift_b_t_1_1_d: torch.Tensor,
) -> torch.Tensor:
    """Apply AdaLN: norm(x) * (1 + scale) + shift.

    Assumes tensors are broadcast-compatible with layout (b, t, h, w, d).
    """
    return norm_layer(x_b_t_h_w_d) * (1 + scale_b_t_1_1_d) + shift_b_t_1_1_d


def cross_attention_block(
    x_b_t_h_w_d: torch.Tensor,
    norm_layer: torch.nn.Module,
    scale_b_t_1_1_d: torch.Tensor,
    shift_b_t_1_1_d: torch.Tensor,
    gate_b_t_1_1_d: Optional[torch.Tensor],
    cross_attn_module,
    crossattn_emb: torch.Tensor,
    rope_emb_l_1_1_d: Optional[torch.Tensor],
    # KV-cache / suffix streaming controls
    crossattn_cache=None,
) -> torch.Tensor:
    """Run cross-attention with AdaLN pre-norm and gated residual.

    If write_length > 0, only processes the suffix frames and writes the residual back
    in-place to x. Otherwise, processes the full sequence and returns x + gate * result.
    """

    _, t, h, w, _ = x_b_t_h_w_d.shape
    normalized = apply_adaln(x_b_t_h_w_d, norm_layer, scale_b_t_1_1_d, shift_b_t_1_1_d)
    flat = rearrange(normalized, "b t h w d -> b (t h w) d")
    result = rearrange(
        cross_attn_module(flat, crossattn_emb, rope_emb=rope_emb_l_1_1_d),
        "b (t h w) d -> b t h w d",
        t=t,
        h=h,
        w=w,
    )
    if gate_b_t_1_1_d is not None:
        return x_b_t_h_w_d + gate_b_t_1_1_d * result
    return x_b_t_h_w_d + result


def self_attention_block_dense(
    x_b_t_h_w_d: torch.Tensor,
    norm_layer: torch.nn.Module,
    scale_b_t_1_1_d: torch.Tensor,
    shift_b_t_1_1_d: torch.Tensor,
    gate_b_t_1_1_d: Optional[torch.Tensor],
    self_attn_module,
    rope_emb_l_1_1_d: Optional[torch.Tensor],
    extra_add: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dense self-attention block with AdaLN and gated residual.

    extra_add, if provided, is added to the normalized tensor before attention (e.g., camera embeddings).
    """
    _, t, h, w, _ = x_b_t_h_w_d.shape
    normalized = apply_adaln(x_b_t_h_w_d, norm_layer, scale_b_t_1_1_d, shift_b_t_1_1_d)
    if extra_add is not None:
        normalized = normalized + extra_add
    flat = rearrange(normalized, "b t h w d -> b (t h w) d")
    result = rearrange(
        self_attn_module(flat, None, rope_emb=rope_emb_l_1_1_d),
        "b (t h w) d -> b t h w d",
        t=t,
        h=h,
        w=w,
    )
    if gate_b_t_1_1_d is not None:
        return x_b_t_h_w_d + gate_b_t_1_1_d * result
    return x_b_t_h_w_d + result


def self_attention_block_kvcache(
    x_b_t_h_w_d: torch.Tensor,
    norm_layer: torch.nn.Module,
    scale_b_t_1_1_d: torch.Tensor,
    shift_b_t_1_1_d: torch.Tensor,
    gate_b_t_1_1_d: Optional[torch.Tensor],
    self_attn_module,
    rope_emb_l_1_1_d: Optional[torch.Tensor],
    kv_cache,
    current_start: int,
    freeze_kv: bool,
    write_length: Optional[int],
    extra_add: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Self-attention with KV-cache, AdaLN, and gated residual.

    - If write_length > 0: processes only suffix frames; writes residual in-place on the slice.
    - Else: processes full sequence and returns x + gate * result.
    - extra_add is added to normalized tensor (or slice) before attention.
    - When rope_presliced_for_suffix=True, the provided rope tensor is assumed already sliced for suffix.
    """
    _, T, H, W, _ = x_b_t_h_w_d.shape

    normalized = apply_adaln(x_b_t_h_w_d, norm_layer, scale_b_t_1_1_d, shift_b_t_1_1_d)
    if extra_add is not None:
        normalized = normalized + extra_add
    flat = rearrange(normalized, "b t h w d -> b (t h w) d")
    result = rearrange(
        self_attn_module(
            flat,
            None,
            rope_emb=rope_emb_l_1_1_d,
            kv_cache=kv_cache,
            current_start=current_start,
            freeze_kv=freeze_kv,
            write_length=write_length,
        ),
        "b (t h w) d -> b t h w d",
        t=T,
        h=H,
        w=W,
    )
    if gate_b_t_1_1_d is not None:
        return x_b_t_h_w_d + gate_b_t_1_1_d * result
    return x_b_t_h_w_d + result


def mlp_block(
    x_b_t_h_w_d: torch.Tensor,
    norm_layer: torch.nn.Module,
    scale_b_t_1_1_d: torch.Tensor,
    shift_b_t_1_1_d: torch.Tensor,
    gate_b_t_1_1_d: Optional[torch.Tensor],
    mlp_module,
) -> torch.Tensor:
    """MLP block with AdaLN and gated residual."""

    normalized = apply_adaln(x_b_t_h_w_d, norm_layer, scale_b_t_1_1_d, shift_b_t_1_1_d)
    result = mlp_module(normalized)
    if gate_b_t_1_1_d is not None:
        return x_b_t_h_w_d + gate_b_t_1_1_d * result
    return x_b_t_h_w_d + result
