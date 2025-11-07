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

import math
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import transformer_engine as te
from einops import rearrange, repeat
from megatron.core import parallel_state
from packaging.version import Version
from torch import nn
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed.fsdp import fully_shard
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention
from torchvision import transforms

if Version(te.__version__) >= Version("2.8.0"):
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
else:
    from transformer_engine.pytorch.attention import apply_rotary_pos_emb

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.interactive.networks.blockmask import build_blockwise_causal_mask_flex
from cosmos_predict2._src.predict2.interactive.networks.ulysses import DistributedAttention
from cosmos_predict2._src.predict2.interactive.networks.utils import (
    cross_attention_block,
    mlp_block,
    self_attention_block_dense,
)
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
    CheckpointMode,
    FinalLayer,
    GPT2FeedForward,
    LearnablePosEmbAxis,
    PatchEmbed,
    SACConfig,
    TimestepEmbedding,
    Timesteps,
    VideoRopePosition3DEmb,
    ptd_checkpoint_wrapper,
)
from cosmos_predict2._src.predict2.networks.model_weights_stats import WeightTrainingStat


def torch_attention_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    flatten_heads: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention with optional mask.

    Inputs are shaped [B, S, H, D]. If rearrange=True, flattens heads to return [B, S, H*D].
    Otherwise returns [B, S, H, D].
    """
    q_B_H_S_D = rearrange(q_B_S_H_D, "b s h d -> b h s d")
    k_B_H_S_D = rearrange(k_B_S_H_D, "b s h d -> b h s d")
    v_B_H_S_D = rearrange(v_B_S_H_D, "b s h d -> b h s d")
    result_B_H_S_D = torch.nn.functional.scaled_dot_product_attention(
        q_B_H_S_D, k_B_H_S_D, v_B_H_S_D, attn_mask=attn_mask
    )
    if flatten_heads:
        return rearrange(result_B_H_S_D, "b h s d -> b s (h d)")
    else:
        return rearrange(result_B_H_S_D, "b h s d -> b s h d")


def torch_flex_attention_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    flatten_heads: bool = False,
) -> torch.Tensor:
    # Rearrange to [B, H, S, D]
    q_B_H_Sq_D = rearrange(q_B_S_H_D, "b s h d -> b h s d")
    k_B_H_Sk_D = rearrange(k_B_S_H_D, "b s h d -> b h s d")
    v_B_H_Sk_D = rearrange(v_B_S_H_D, "b s h d -> b h s d")

    S_q = q_B_H_Sq_D.shape[2]
    S_kv = k_B_H_Sk_D.shape[2]
    # Right-pad to multiples of 128 for optimal FlexAttention kernels
    pad_q = ((S_q + 127) // 128) * 128 - S_q
    pad_kv = ((S_kv + 127) // 128) * 128 - S_kv

    if pad_q > 0:
        q_pad_tensor = torch.zeros(
            (q_B_H_Sq_D.shape[0], q_B_H_Sq_D.shape[1], pad_q, q_B_H_Sq_D.shape[3]),
            device=q_B_H_Sq_D.device,
            dtype=q_B_H_Sq_D.dtype,
        )
        q_cat = torch.cat([q_B_H_Sq_D, q_pad_tensor], dim=2)
    else:
        q_cat = q_B_H_Sq_D

    if pad_kv > 0:
        kv_pad_tensor = torch.zeros(
            (k_B_H_Sk_D.shape[0], k_B_H_Sk_D.shape[1], pad_kv, k_B_H_Sk_D.shape[3]),
            device=k_B_H_Sk_D.device,
            dtype=k_B_H_Sk_D.dtype,
        )
        k_cat = torch.cat([k_B_H_Sk_D, kv_pad_tensor], dim=2)
        v_cat = torch.cat([v_B_H_Sk_D, kv_pad_tensor], dim=2)
    else:
        k_cat, v_cat = k_B_H_Sk_D, v_B_H_Sk_D

    block_mask = None
    if attn_mask is not None and isinstance(attn_mask, BlockMask):
        block_mask = attn_mask
    else:
        # When padding is introduced without an explicit mask, build a validity mask
        if pad_q > 0 or pad_kv > 0:

            def allow_valid(b, h, q_idx, kv_idx):
                return (q_idx < S_q) & (kv_idx < S_kv)

            block_mask = create_block_mask(
                allow_valid,
                B=None,
                H=None,
                Q_LEN=q_cat.shape[2],
                KV_LEN=k_cat.shape[2],
                _compile=False,
                device=q_cat.device,
            )

    if block_mask is not None:
        out_B_H_Sqp_D = torch_flex_attention(query=q_cat, key=k_cat, value=v_cat, block_mask=block_mask)
    else:
        out_B_H_Sqp_D = torch_flex_attention(query=q_cat, key=k_cat, value=v_cat)

    out_B_H_Sq_D = out_B_H_Sqp_D[:, :, :S_q] if pad_q > 0 else out_B_H_Sqp_D
    if flatten_heads:
        return rearrange(out_B_H_Sq_D, "b h s d -> b s (h d)")
    else:
        return rearrange(out_B_H_Sq_D, "b h s d -> b s h d")


class CausalAttention(nn.Module):
    """
    Attention module with optional causal mask support. Backends:
    - "torch": local SDPA
    - "ulysses": sequence-parallel DistributedAttention wrapper
    - "torch-flex": PyTorch FlexAttention (single device)
    - "ulysses-flex": FlexAttention wrapped with Ulysses for CP
    """

    def __init__(
        self,
        query_dim: int,
        context_dim=None,
        n_heads=8,
        head_dim=64,
        dropout=0.0,
        qkv_format: str = "bshd",
        backend: str = "ulysses",
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__()
        log.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{n_heads} heads with a dimension of {head_dim}."
        )
        self.is_selfattn = context_dim is None

        assert backend in [
            "torch",
            "ulysses",
            "transformer_engine",
            "torch-flex",
            "ulysses-flex",
        ], f"Invalid backend: {backend}"
        self.backend = backend

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        self._mask = None

        if self.backend == "torch":
            self.attn_op = partial(torch_attention_op, flatten_heads=True)
        elif self.backend == "ulysses":
            if parallel_state.is_initialized():
                if parallel_state.get_context_parallel_world_size() > 1:
                    process_group = parallel_state.get_context_parallel_group()
                    self.attn_op = DistributedAttention(
                        partial(torch_attention_op, flatten_heads=False),
                        sequence_process_group=process_group,
                    )
                else:
                    self.attn_op = DistributedAttention(
                        partial(torch_attention_op, flatten_heads=False),
                        sequence_process_group=None,
                    )
            else:
                self.attn_op = partial(torch_attention_op, flatten_heads=True)
        elif self.backend == "transformer_engine":
            from transformer_engine.pytorch.attention import DotProductAttention

            # Use TE op; enable arbitrary mask for self-attention (temporal-only causal)
            attn_mask_type = "arbitrary" if self.is_selfattn else "no_mask"
            self.attn_op = DotProductAttention(
                self.n_heads,
                self.head_dim,
                num_gqa_groups=self.n_heads,
                attention_dropout=0,
                qkv_format=self.qkv_format,
                attn_mask_type=attn_mask_type,
            )
        elif self.backend == "torch-flex":
            # Use PyTorch FlexAttention. Expects optional BlockMask installed in self.mask during training.
            # Return [B, S, H*D]
            self.attn_op = partial(torch_flex_attention_op, flatten_heads=True)
        elif self.backend == "ulysses-flex":
            # FlexAttention op wrapped with Ulysses sequence parallelism
            if parallel_state.is_initialized() and parallel_state.get_context_parallel_world_size() > 1:
                process_group = parallel_state.get_context_parallel_group()
                self.attn_op = DistributedAttention(
                    partial(torch_flex_attention_op, flatten_heads=False),
                    sequence_process_group=process_group,
                )
            else:
                # Single device fallback
                self.attn_op = partial(torch_flex_attention_op, flatten_heads=True)
        else:
            raise NotImplementedError(f"Backend {self.backend} is not supported for causal attention")

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim
        self.init_weights()

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)

        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(self, x, context=None, rope_emb=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        def apply_norm_and_rotary_pos_emb(q, k, v, rope_emb):
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = self.v_norm(v)
            if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
                if self.use_wan_fp32_strategy:
                    # Temporarily cast q, k to fp32 for rotary pos emb, then cast back to original dtype
                    q_dtype, k_dtype = q.dtype, k.dtype
                    q_fp32 = q.to(torch.float32)
                    k_fp32 = k.to(torch.float32)
                    q_fp32 = apply_rotary_pos_emb(q_fp32, rope_emb, tensor_format=self.qkv_format, fused=True)
                    k_fp32 = apply_rotary_pos_emb(k_fp32, rope_emb, tensor_format=self.qkv_format, fused=True)
                    q = q_fp32.to(q_dtype)
                    k = k_fp32.to(k_dtype)
                else:
                    q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
                    k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
            return q, k, v

        q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)

        return q, k, v

    def compute_attention(self, q, k, v):
        result = self.attn_op(q, k, v, attn_mask=self.mask)  # [B, S, H, D]
        return self.output_dropout(self.output_proj(result))

    def forward(self, x, context: Optional[torch.Tensor] = None, rope_emb: Optional[torch.Tensor] = None):
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v)

    def set_context_parallel_group(self, process_group, ranks, stream):
        # Configure primary op
        if hasattr(self.attn_op, "set_context_parallel_group"):
            self.attn_op.set_context_parallel_group(process_group, ranks, stream)


class CausalI2VCrossAttention(CausalAttention):
    def __init__(self, *args, img_latent_dim: int = 1024, **kwargs):
        self.k_img, self.v_img, self.k_img_norm = None, None, None
        super().__init__(*args, **kwargs)
        inner_dim = self.head_dim * self.n_heads
        self.k_img = nn.Linear(img_latent_dim, inner_dim, bias=False)
        self.v_img = nn.Linear(img_latent_dim, inner_dim, bias=False)
        self.k_img_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

        self.init_img_weights()

    def init_weights(self) -> None:
        super().init_weights()
        self.init_img_weights()

    def init_img_weights(self) -> None:
        if self.k_img and self.v_img and self.k_img_norm:
            torch.nn.init.trunc_normal_(self.k_img.weight, std=1.0 / math.sqrt(self._inner_dim))
            torch.nn.init.trunc_normal_(self.v_img.weight, std=1.0 / math.sqrt(self._inner_dim))
            self.k_img_norm.reset_parameters()

    def compute_qkv(
        self, x: torch.Tensor, context: torch.Tensor, rope_emb: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_context, img_context = context
        q, k, v = super().compute_qkv(x, text_context, rope_emb)
        k_img = self.k_img(img_context)
        v_img = self.v_img(img_context)
        k_img, v_img = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (k_img, v_img),
        )

        return q, k, v, self.k_img_norm(k_img), v_img

    def compute_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_img: torch.Tensor, v_img: torch.Tensor
    ) -> torch.Tensor:
        result = self.attn_op(q, k, v)  # [B, S, H, D]
        result_img = self.attn_op(q, k_img, v_img)
        return self.output_dropout(self.output_proj(result + result_img))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v, k_img, v_img = self.compute_qkv(x, context, rope_emb)
        return self.compute_attention(q, k, v, k_img, v_img)


class CausalBlock(nn.Module):
    """
    Transformer block: self-attn (with optional RoPE), cross-attn, and MLP with AdaLN modulation.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        backend: str = "ulysses",
        image_context_dim: Optional[int] = None,
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = CausalAttention(
            x_dim,
            None,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
            backend=backend,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        if image_context_dim is None:
            self.cross_attn = CausalAttention(x_dim, context_dim, num_heads, x_dim // num_heads, qkv_format="bshd")
        else:
            self.cross_attn = CausalI2VCrossAttention(
                x_dim, context_dim, num_heads, x_dim // num_heads, img_latent_dim=image_context_dim, qkv_format="bshd"
            )

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        self.cp_size = None
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

    def set_context_parallel_group(self, process_group, ranks, stream):
        self.cp_size = None if ranks is None else len(ranks)
        self.self_attn.set_context_parallel_group(
            process_group=process_group,
            ranks=ranks,
            stream=stream,
        )

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        with torch.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                    self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                    self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
            else:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                    emb_B_T_D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
                )
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        # Shared AdaLN helper

        # Self-attention dense helper
        x_B_T_H_W_D = self_attention_block_dense(
            x_b_t_h_w_d=x_B_T_H_W_D,
            norm_layer=self.layer_norm_self_attn,
            scale_b_t_1_1_d=scale_self_attn_B_T_1_1_D,
            shift_b_t_1_1_d=shift_self_attn_B_T_1_1_D,
            gate_b_t_1_1_d=gate_self_attn_B_T_1_1_D,
            self_attn_module=self.self_attn,
            rope_emb_l_1_1_d=rope_emb_L_1_1_D,
        )

        x_B_T_H_W_D = cross_attention_block(
            x_b_t_h_w_d=x_B_T_H_W_D,
            norm_layer=self.layer_norm_cross_attn,
            scale_b_t_1_1_d=scale_cross_attn_B_T_1_1_D,
            shift_b_t_1_1_d=shift_cross_attn_B_T_1_1_D,
            gate_b_t_1_1_d=gate_cross_attn_B_T_1_1_D,
            cross_attn_module=self.cross_attn,
            crossattn_emb=crossattn_emb,
            rope_emb_l_1_1_d=rope_emb_L_1_1_D,
        )

        x_B_T_H_W_D = mlp_block(
            x_b_t_h_w_d=x_B_T_H_W_D,
            norm_layer=self.layer_norm_mlp,
            scale_b_t_1_1_d=scale_mlp_B_T_1_1_D,
            shift_b_t_1_1_d=shift_mlp_B_T_1_1_D,
            gate_b_t_1_1_d=gate_mlp_B_T_1_1_D,
            mlp_module=self.mlp,
        )
        return x_B_T_H_W_D


class CausalDIT(WeightTrainingStat):
    """
    Minimal v4 DiT with temporal causal masking (interactive variant).
    Based on predict2 MiniTrainDIT, with mask logic from interactive/causal_net.
    """

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        # attention settings
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        atten_backend: str = "ulysses",
        # cross attention settings
        crossattn_emb_channels: int = 1024,
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        extra_image_context_dim: Optional[int] = None,
        # positional embedding settings
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        sac_config: SACConfig = SACConfig(),
        # numerics
        use_wan_fp32_strategy: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.atten_backend = atten_backend
        # positional embedding settings
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.extra_image_context_dim = extra_image_context_dim
        self.build_patch_embed()
        self.build_pos_embed()
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )
        self.use_crossattn_projection = use_crossattn_projection
        self.crossattn_proj_in_channels = crossattn_proj_in_channels
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        self.blocks = nn.ModuleList(
            [
                CausalBlock(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    backend=atten_backend,
                    image_context_dim=None if extra_image_context_dim is None else model_channels,
                    use_wan_fp32_strategy=use_wan_fp32_strategy,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
            use_wan_fp32_strategy=self.use_wan_fp32_strategy,
        )

        self.t_embedding_norm = te.pytorch.RMSNorm(model_channels, eps=1e-6)
        if extra_image_context_dim is not None:
            self.img_context_proj = nn.Sequential(
                nn.Linear(extra_image_context_dim, model_channels, bias=True),
                nn.GELU(),
            )

        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )

        self.init_weights()
        self.enable_selective_checkpoint(sac_config, self.blocks)

        self._is_context_parallel_enabled = False

    def init_weights(self):
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.reset_parameters()

        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

        if self.extra_image_context_dim is not None:
            self.img_context_proj[0].reset_parameters()

    def build_patch_embed(self):
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
        )

    def build_pos_embed(self):
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        log.debug(f"Building positional embedding with {self.pos_emb_cls} class, impl {cls_type}")
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )
        self.pos_embedder = cls_type(
            **kwargs,
        )

        if self.extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            self.extra_pos_embedder = LearnablePosEmbAxis(
                **kwargs,
            )

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)

        return x_B_T_H_W_D, None, extra_pos_emb

    def unpatchify(self, x_B_T_H_W_M):
        x_B_C_Tt_Hp_Wp = rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        return x_B_C_Tt_Hp_Wp

    @lru_cache(maxsize=128)
    def get_self_attn_module(self):
        return_block = []
        for _, module in self.named_modules():
            if isinstance(module, CausalAttention):
                if module.is_selfattn:
                    return_block.append(module)
        return return_block

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x_B_C_T_H_W: (B, C, T, H, W) tensor of spatial-temporal inputs
            timesteps_B_T: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
        """
        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, (
                "extra_image_context_dim must be set if img_context_emb is provided"
            )
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        with torch.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # logging hooks
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        # Install temporal causal mask for self-attention (always causal for *_causal variants)
        if data_type == DataType.VIDEO:
            B, T, H, W, D = x_B_T_H_W_D.shape
            # If attention backend is flex-based, install a Flex BlockMask; else use dense boolean mask
            if self.atten_backend in ["torch-flex", "ulysses-flex"]:
                frame_seqlen = H * W
                if parallel_state.is_initialized():
                    seq_world_size = parallel_state.get_context_parallel_world_size()
                else:
                    seq_world_size = 1
                num_frames_for_mask = T * seq_world_size
                # Build BlockMask on CPU to avoid large temporary GPU allocations inside FlexAttention mask conversion
                block_mask = build_blockwise_causal_mask_flex(
                    device="cpu",
                    num_frames=num_frames_for_mask,
                    frame_seqlen=frame_seqlen,
                    compile_mask=True,
                )
                for module in self.get_self_attn_module():
                    module.mask = block_mask.to(x_B_C_T_H_W.device)
            else:
                if parallel_state.is_initialized():
                    seq_world_size = parallel_state.get_context_parallel_world_size()
                else:
                    seq_world_size = 1
                causal_mask = (
                    torch.tril(torch.ones(T * seq_world_size, T * seq_world_size), diagonal=0)
                    .bool()
                    .to(x_B_C_T_H_W.device)
                )
                causal_mask = repeat(causal_mask, "h w -> (h n) (w m)", n=H * W, m=H * W)
                for module in self.get_self_attn_module():
                    module.mask = causal_mask
        else:
            for module in self.get_self_attn_module():
                module.mask = None

        intermediate_features_outputs = []
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                x_reshaped_for_disc = rearrange(x_B_T_H_W_D, "b tp hp wp d -> b (tp hp wp) d")
                intermediate_features_outputs.append(x_reshaped_for_disc)

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        if intermediate_feature_ids:
            if len(intermediate_features_outputs) != len(intermediate_feature_ids):
                log.warning(
                    f"Collected {len(intermediate_features_outputs)} intermediate features, "
                    f"but expected {len(intermediate_feature_ids)}. "
                    f"Requested IDs: {intermediate_feature_ids}"
                )
            return x_B_C_Tt_Hp_Wp, intermediate_features_outputs

        return x_B_C_Tt_Hp_Wp

    def enable_selective_checkpoint(self, sac_config: SACConfig, blocks: nn.ModuleList):
        # FlexAttention HOP is not compatible with PyTorch's checkpoint dispatch mode.
        # Skip selective checkpointing for flex-based backends to avoid runtime errors.
        if self.atten_backend in ["torch-flex", "ulysses-flex"]:
            log.info(
                f"Skip selective checkpoint for backend '{self.atten_backend}' due to FlexAttention/HOP checkpoint incompatibility."
            )
            return self
        if sac_config.mode == CheckpointMode.NONE:
            return self

        log.info(
            f"Enable selective checkpoint with {sac_config.mode}, for every {sac_config.every_n_blocks} blocks. Total blocks: {len(blocks)}"
        )
        _context_fn = sac_config.get_context_fn()
        for block_id, block in blocks.named_children():
            if int(block_id) % sac_config.every_n_blocks == 0:
                log.info(f"Enable selective checkpoint for block {block_id}")
                block = ptd_checkpoint_wrapper(
                    block,
                    context_fn=_context_fn,
                    preserve_rng_state=False,
                )
                blocks.register_module(block_id, block)
        self.register_module(
            "final_layer",
            ptd_checkpoint_wrapper(
                self.final_layer,
                context_fn=_context_fn,
                preserve_rng_state=False,
            ),
        )

        return self

    def fully_shard(self, mesh):
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward)

        fully_shard(self.final_layer, mesh=mesh, reshard_after_forward=True)
        if self.extra_per_block_abs_pos_emb:
            fully_shard(self.extra_pos_embedder, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=False)
        if self.extra_image_context_dim is not None:
            fully_shard(self.img_context_proj, mesh=mesh, reshard_after_forward=False)

    def disable_context_parallel(self):
        self.pos_embedder.disable_context_parallel()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.disable_context_parallel()

        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = False

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None):
        self.pos_embedder.enable_context_parallel(process_group=process_group)
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.enable_context_parallel(process_group=process_group)

        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=process_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = True

    @property
    def is_context_parallel_enabled(self):
        return self._is_context_parallel_enabled


class CausalDITwithConditionalMask(CausalDIT):
    def __init__(self, *args, timestep_scale: float = 1.0, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # Add 1 for the condition mask
        self.timestep_scale = timestep_scale
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
        return super().forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T * self.timestep_scale,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            intermediate_feature_ids=intermediate_feature_ids,
            img_context_emb=img_context_emb,
        )


@dataclass
class KVContextConfig:
    run_with_kv: bool = False
    store_kv: bool = False
    start_idx: int = 0
    recompute_cross_attn_kv: bool = False


class AttenOpWithKV(nn.Module):
    """A thin wrapper that adds K/V caching to an existing attention op.

    This wrapper expects the wrapped op to accept (q, k, v, attn_mask=None)
    and return attention outputs with heads already flattened on the last dim.
    """

    def __init__(self, attn_op: nn.Module | Any):
        super().__init__()
        self.attn_op = attn_op
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        self._kv_context_cfg: Optional[KVContextConfig] = None

        # for rolling kv cache
        self.start_pointer: int = 0
        self.cache_size: int = 0

    def reset_kv_cache(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.k_cache = torch.zeros(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        self.v_cache = torch.zeros(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)

        # for rolling kv cache
        self.start_pointer = 0
        self.cache_size = seq_len

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_context_cfg: Optional[KVContextConfig] = None,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if kv_context_cfg is None:
            kv_context_cfg = self._kv_context_cfg or KVContextConfig()

        # Optionally concatenate cached prefix before current slice
        if kv_context_cfg.run_with_kv and kv_context_cfg.start_idx > 0:
            assert self.k_cache is not None and self.v_cache is not None, (
                "KV cache is not initialized. Call reset_kv_cache() first."
            )
            history_k = self.k_cache[:, : kv_context_cfg.start_idx - self.start_pointer]
            history_v = self.v_cache[:, : kv_context_cfg.start_idx - self.start_pointer]
            k_out = torch.cat([history_k, k], dim=1)
            v_out = torch.cat([history_v, v], dim=1)
        else:
            k_out = k
            v_out = v

        # Optionally store current K/V into cache at the given start index
        if kv_context_cfg.store_kv:
            assert self.k_cache is not None and self.v_cache is not None, (
                "KV cache is not initialized. Call reset_kv_cache() first."
            )
            k_seq_len = int(k.shape[1])
            start = int(kv_context_cfg.start_idx)
            end = start + k_seq_len
            if end > self.start_pointer + self.cache_size:
                # rolling kv cache
                old_start = end - self.cache_size
                self.k_cache = torch.cat(
                    [self.k_cache[:, old_start - self.start_pointer : start - self.start_pointer], k.detach()], dim=1
                )
                self.v_cache = torch.cat(
                    [self.v_cache[:, old_start - self.start_pointer : start - self.start_pointer], v.detach()], dim=1
                )
                self.start_pointer = old_start
            else:
                self.k_cache[:, start - self.start_pointer : end - self.start_pointer] = k.detach()
                self.v_cache[:, start - self.start_pointer : end - self.start_pointer] = v.detach()

        # Delegate to wrapped attention op
        try:
            return self.attn_op(q, k_out, v_out, attn_mask=attn_mask, **kwargs)
        except TypeError:
            # Fallback for ops that don't accept attn_mask as kwarg
            return self.attn_op(q, k_out, v_out)

    def set_kv_context(self, cfg: KVContextConfig | None) -> None:
        self._kv_context_cfg = cfg


class VideoSeqPos:
    """Flattened 3D grid positions for a video clip.

    Stores flattened t/h/w indices of length L = T*H*W to enable constructing
    RoPE frequencies aligned with global positions across sequential chunks.
    """

    def __init__(self, T: int, H: int, W: int, pos_h=None, pos_w=None, pos_t=None) -> None:
        self.T = T
        self.H = H
        self.W = W

        if pos_h is not None and pos_w is not None and pos_t is not None:
            self.pos_h = pos_h.to(dtype=torch.long)
            self.pos_w = pos_w.to(dtype=torch.long)
            self.pos_t = pos_t.to(dtype=torch.long)
            return

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        t = torch.arange(self.T, device=device, dtype=torch.long)
        h = torch.arange(self.H, device=device, dtype=torch.long)
        w = torch.arange(self.W, device=device, dtype=torch.long)
        pos_t, pos_h, pos_w = torch.meshgrid(t, h, w, indexing="ij")
        self.pos_t = pos_t.reshape(-1)
        self.pos_h = pos_h.reshape(-1)
        self.pos_w = pos_w.reshape(-1)

    def size(self) -> int:
        return int(self.pos_h.numel())


class CausalDITKVCache(CausalDIT):
    """Causal DiT with lightweight KV cache via attention-op wrapping.

    This variant follows the KV wrapping strategy in wan2pt1_seq.py:
    - make_it_kv_cache: install KV-enabled attention ops and allocate caches
    - forward_seq: process a sequence chunk (typically one frame) with proper RoPE slicing
    """

    def make_it_kv_cache(
        self,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        del cp_group  # CP-aware KV wrapping can be added later if needed
        head_dim = self.model_channels // self.num_heads

        for block in self.blocks:
            attn_op = block.self_attn.attn_op
            if hasattr(attn_op, "local_attn"):
                # Ulysses distributed attention wrapper
                if isinstance(attn_op.local_attn, AttenOpWithKV):
                    # Already wrapped, just reset the cache
                    attn_op.local_attn.reset_kv_cache(batch_size, seq_len, self.num_heads, head_dim, dtype, device)
                else:
                    # Not wrapped yet, create new wrapper
                    kv_op = AttenOpWithKV(attn_op.local_attn)
                    kv_op.reset_kv_cache(batch_size, seq_len, self.num_heads, head_dim, dtype, device)
                    attn_op.local_attn = kv_op
            else:
                # Direct op (torch/TE/flex). Wrap at the top-level.
                if isinstance(attn_op, AttenOpWithKV):
                    # Already wrapped, just reset the cache
                    attn_op.reset_kv_cache(batch_size, seq_len, self.num_heads, head_dim, dtype, device)
                else:
                    # Not wrapped yet, create new wrapper
                    kv_op = AttenOpWithKV(attn_op)
                    kv_op.reset_kv_cache(batch_size, seq_len, self.num_heads, head_dim, dtype, device)
                    block.self_attn.attn_op = kv_op

    def make_it_temporal_causal(
        self,
        num_frames: int,
        frame_seqlen: int,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """Install temporal-only causal behavior by replacing local attention ops (Flex).

        Follows wan2pt1_seq: compile a BlockMask once and inject an attention op that
        always uses this mask inside FlexAttention. No dense torch.tril fallback.
        """
        block_mask = build_blockwise_causal_mask_flex(
            device="cpu",
            num_frames=num_frames,
            frame_seqlen=frame_seqlen,
            compile_mask=True,
        ).to(device)

        def _flex_masked_attn(q, k, v, *_, **__):
            return torch_flex_attention_op(q, k, v, attn_mask=block_mask, flatten_heads=False)

        for block in self.blocks:
            module = block.self_attn
            attn_op = getattr(module, "attn_op", None)
            backend = getattr(module, "backend", None)
            if backend in ["torch-flex", "ulysses-flex"]:
                # Install Flex BlockMask; also swap local op for Ulysses Flex to ensure mask usage
                module.mask = block_mask
                if hasattr(attn_op, "local_attn"):
                    attn_op.local_attn = _flex_masked_attn
            elif backend in ["torch", "ulysses", "transformer_engine"]:
                # Dense boolean causal mask (S x S) for non-Flex backends
                causal = torch.tril(torch.ones(num_frames, num_frames, device=device), diagonal=0).bool()
                dense_mask = repeat(causal, "t1 t2 -> (t1 n) (t2 m)", n=frame_seqlen, m=frame_seqlen)
                module.mask = dense_mask

        self._temporal_causal_enabled = True

    def forward_seq(
        self,
        x_B_L_D: torch.Tensor,
        video_pos: VideoSeqPos,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        *,
        kv_context_cfg: Optional[KVContextConfig] = None,
        img_context_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward a sequence chunk with KV caches and correct RoPE alignment.

        Args:
            x_B_L_D: [B, L, D] sequence features for the current chunk (typically one frame, L=T*H*W with T=1)
            video_pos: VideoSeqPos describing global positions for this chunk
            timesteps_B_T: [B, 1] timesteps
            crossattn_emb: [B, N, D_ctx] cross-attention embeddings
            kv_context_cfg: KV context (start index, run/store flags)
            img_context_emb: Optional image context for i2v variants

        Returns:
            [B, L, O] token outputs after the model head (O = patch_prod * out_channels)
        """
        B, L, D = x_B_L_D.shape
        assert L == video_pos.T * video_pos.H * video_pos.W, (
            f"Token length mismatch: {L} != {video_pos.T}*{video_pos.H}*{video_pos.W}"
        )

        # Prepare context inputs
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)
        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, (
                "extra_image_context_dim must be set when img_context_emb is provided"
            )
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        # Time embeddings (affine mods)
        with torch.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # RoPE frequencies aligned to this chunk's global positions, reusing VideoRopePosition3DEmb
        # Determine full extents from absolute positions
        T_full = int(video_pos.pos_t.max().item()) + 1
        H_full = int(video_pos.pos_h.max().item()) + 1
        W_full = int(video_pos.pos_w.max().item()) + 1
        # Generate full-grid rope using the existing pos_embedder (predict2 VideoRopePosition3DEmb)
        rope_full = self.pos_embedder.generate_embeddings((1, T_full, H_full, W_full, self.model_channels))
        # Gather rope at absolute indices for this chunk
        linear_idx = (
            video_pos.pos_t.to(dtype=torch.long) * (H_full * W_full)
            + video_pos.pos_h.to(dtype=torch.long) * W_full
            + video_pos.pos_w.to(dtype=torch.long)
        )
        rope_L_1_1_D = rope_full.index_select(0, linear_idx.to(device=rope_full.device))

        # Reshape sequence to 5D for block API
        x_B_T_H_W_D = rearrange(
            x_B_L_D,
            "b (t h w) d -> b t h w d",
            t=video_pos.T,
            h=video_pos.H,
            w=video_pos.W,
        )

        # Install/update KV context on all wrapped self-attention ops
        for block in self.blocks:
            attn_op = block.self_attn.attn_op
            if hasattr(attn_op, "local_attn") and isinstance(attn_op.local_attn, AttenOpWithKV):
                attn_op.local_attn.set_kv_context(kv_context_cfg)
            elif isinstance(attn_op, AttenOpWithKV):
                attn_op.set_kv_context(kv_context_cfg)

        # Clear masks only when temporal-causal mode is not explicitly enabled
        if not getattr(self, "_temporal_causal_enabled", False):
            for module in self.get_self_attn_module():
                module.mask = None

        # Run blocks with KV-aware self-attention (installed via make_it_kv_cache)
        for block in self.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=None,
            )

        # Final head then flatten back to [B, L, O]
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_L_O = rearrange(x_B_T_H_W_O, "b t h w o -> b (t h w) o")
        return x_B_L_O
