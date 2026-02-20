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

from typing import Any, Optional

import torch
from einops import rearrange

from cosmos_predict2._src.imaginaire.attention import spatio_temporal_attention
from cosmos_predict2._src.predict2.utils.kv_cache import AttentionOpWithKVCache


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


def make_network_temporal_causal(net: Any, h_tokens: int, w_tokens: int) -> None:
    """Install temporal-only causal masking using I4 spatio-temporal attention.

    Replaces each block's `self_attn.attn_op` with a closure that reshapes QKV to
    [B, T, H, W, heads, head_dim] and invokes `spatio_temporal_attention` (causal on T, full spatial).

    Expected network structure:
    - `net.blocks`: iterable of blocks with a `.self_attn` module
    - `.self_attn.attn_op`: callable attention op to be replaced

    Args:
        net: The network instance to modify (modified in place).
        h_tokens: Number of tokens along height per frame (H).
        w_tokens: Number of tokens along width per frame (W).
    """

    def _i4_temporal_causal_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **_: dict) -> torch.Tensor:
        q6 = rearrange(q, "b (t h w) hh d -> b t h w hh d", h=h_tokens, w=w_tokens)
        k6 = rearrange(k, "b (t h w) hh d -> b t h w hh d", h=h_tokens, w=w_tokens)
        v6 = rearrange(v, "b (t h w) hh d -> b t h w hh d", h=h_tokens, w=w_tokens)
        assert q6.shape == k6.shape == v6.shape
        out6 = spatio_temporal_attention(q6, k6, v6)
        out4 = rearrange(out6, "b t h w hh d -> b (t h w) (hh d)")
        return out4

    for block in net.blocks:
        block.self_attn.attn_op = _i4_temporal_causal_attn


def make_network_kv_cache(net: Any, max_cache_size: Optional[int] = None) -> None:
    """Wrap attention with KV cache support and initialize list-based caches.

    Each block's self-attention op is wrapped by `AttentionOpWithKVCache`, which adds
    lightweight K/V caching for streaming or chunked inference. Initializes
    list-based caches and sets a rolling capacity measured in number of chunks
    (entries), not tokens.

    Expected network structure:
    - `net.blocks`: iterable of blocks with a `.self_attn.attn_op` callable
    - `net.model_channels`: hidden/model dimension
    - `net.num_heads`: number of attention heads

    Args:
        net: The network instance to modify (modified in place).
        max_cache_size: Maximum number of cached chunks to retain (rolling window).

    Raises:
        AttributeError: If required attributes (e.g., `blocks`, `model_channels`, `num_heads`) are missing.
    """

    for block in net.blocks:
        attn_op: Any = block.self_attn.attn_op
        if not isinstance(attn_op, AttentionOpWithKVCache):
            # Not wrapped yet, create new wrapper
            attn_op = AttentionOpWithKVCache(block.self_attn.attn_op, max_cache_size=max_cache_size)
            block.self_attn.attn_op = attn_op
        # Reset the KV cache (list-based); pass max_entries as the capacity
        attn_op.reset_kv_cache(max_cache_size=max_cache_size)
