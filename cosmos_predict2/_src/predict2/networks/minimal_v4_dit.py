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

import collections
import math
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.amp as amp
import transformer_engine as te
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper

try:
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
except ImportError:
    CheckpointPolicy = None

from torchvision import transforms
from transformer_engine.pytorch.attention import apply_rotary_pos_emb

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.context_parallel import split_inputs_cp
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.modules.neighborhood_attn import NeighborhoodAttention
from cosmos_predict2._src.predict2.networks.a2a_cp import MinimalA2AAttnOp, NattenA2AAttnOp
from cosmos_predict2._src.predict2.networks.model_weights_stats import WeightTrainingStat
from cosmos_predict2._src.predict2.networks.selective_activation_checkpoint import SACConfig as _SACConfig


# selective activation checkpoint; only apply to the minimal v4 model. if there are change in the networks, some policy will not work as we expect.
def predict2_2B_720_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            op_count_key = f"{mode}_mm_count"
            # from cosmos_predict2._src.imaginaire.utils import log
            # log.info(f"op_count_key: {op_count_key}, op_count[op_count_key]: {op_count[op_count_key]}, {args[0].shape}, {args[1].shape}")
            # there are totally 6 + 4 + 4 + 2 = 16 block
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 16
            if op_count[op_count_key] > 8:  # recompute self attn first 3 linear layers
                return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def predict2_2B_720_context_fn_aggressive():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        # The default policy is to recompute everything. This is the most memory-efficient
        # starting point. We then selectively choose what to save.
        default_policy = CheckpointPolicy.PREFER_RECOMPUTE

        # Save the output of Flash Attention. This is the most computationally
        # expensive part of a transformer block. Saving its output provides a
        # good balance between memory savings and computational overhead.
        if "flash_attn" in str(func):
            return CheckpointPolicy.MUST_SAVE

        # All other operations (e.g., torch.ops.aten.mm.default, layer norms, additions)
        # will fall through to the default policy and be recomputed.
        return default_policy

    return create_selective_checkpoint_contexts(policy_fn)


def predict2_2B_720_context_fn_aggressive_v2():
    """
    The most memory-aggressive checkpointing policy. Recomputes ALL operations.
    """

    def policy_fn(ctx, func, *args, **kwargs):
        # The policy is to always recompute everything.
        # This saves the maximum amount of memory but incurs the highest
        # computational cost during the backward pass.
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def predict2_14B_720_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            op_count_key = f"{mode}_mm_count"
            # from cosmos_predict2._src.imaginaire.utils import log
            # log.info(f"op_count_key: {op_count_key}, op_count[op_count_key]: {op_count[op_count_key]}, {args[0].shape}, {args[1].shape}")
            # there are totally 6 + 4 + 4 + 2 = 16 block
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 16
            if op_count[op_count_key] > 8:  # recompute self attn first 1 linear layers
                return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def predict2_14B_720_context_fn_aggressive():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            op_count_key = f"{mode}_mm_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 16
            if op_count[op_count_key] > 12:  # recompute self attn first 1 linear layers
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def linear_selfattn_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


class CheckpointMode(str, Enum):
    NONE = "none"
    MM_ONLY = "mm_only"
    BLOCK_WISE = "block_wise"
    LINEAR_SELFATTN = "linear_selfattn"
    PREDICT2_2B_720 = "predict2_2b_720"
    PREDICT2_14B_720 = "predict2_14b_720"
    PREDICT2_2B_720_AGGRESSIVE = "predict2_2b_720_aggressive"
    PREDICT2_2B_720_AGGRESSIVE_V2 = "predict2_2b_720_aggressive_v2"
    PREDICT2_14B_720_AGGRESSIVE = "predict2_14b_720_aggressive"

    def __str__(self) -> str:
        return self.value


@dataclass
class SACConfig(_SACConfig):
    def get_context_fn(self):
        if self.mode == CheckpointMode.LINEAR_SELFATTN:
            return linear_selfattn_context_fn
        elif self.mode == CheckpointMode.PREDICT2_2B_720:
            return predict2_2B_720_context_fn
        elif self.mode == CheckpointMode.PREDICT2_2B_720_AGGRESSIVE:
            return predict2_2B_720_context_fn_aggressive
        elif self.mode == CheckpointMode.PREDICT2_2B_720_AGGRESSIVE_V2:
            return predict2_2B_720_context_fn_aggressive_v2
        elif self.mode == CheckpointMode.PREDICT2_14B_720:
            return predict2_14B_720_context_fn
        elif self.mode == CheckpointMode.PREDICT2_14B_720_AGGRESSIVE:
            return predict2_14B_720_context_fn_aggressive
        else:
            # Reuse parent class implementation for other modes
            return super().get_context_fn()


VideoSize = namedtuple("VideoSize", ["T", "H", "W"])


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ---------------------- Feed Forward Network -----------------------
class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)

        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)

        x = self.activation(x)
        x = self.layer2(x)
        return x


def torch_attention_op(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D):
    """Computes multi-head attention using PyTorch's native implementation.

    This function provides a PyTorch backend alternative to Transformer Engine's attention operation.
    It rearranges the input tensors to match PyTorch's expected format, computes scaled dot-product
    attention, and rearranges the output back to the original format.

    The input tensor names use the following dimension conventions:

    - B: batch size
    - S: sequence length
    - H: number of attention heads
    - D: head dimension

    Args:
        q_B_S_H_D: Query tensor with shape (batch, seq_len, n_heads, head_dim)
        k_B_S_H_D: Key tensor with shape (batch, seq_len, n_heads, head_dim)
        v_B_S_H_D: Value tensor with shape (batch, seq_len, n_heads, head_dim)

    Returns:
        Attention output tensor with shape (batch, seq_len, n_heads * head_dim)
    """
    in_q_shape = q_B_S_H_D.shape
    in_k_shape = k_B_S_H_D.shape
    q_B_H_S_D = rearrange(q_B_S_H_D, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
    k_B_H_S_D = rearrange(k_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    v_B_H_S_D = rearrange(v_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    result_B_S_HD = rearrange(
        torch.nn.functional.scaled_dot_product_attention(q_B_H_S_D, k_B_H_S_D, v_B_H_S_D), "b h ... l -> b ... (h l)"
    )

    return result_B_S_HD


class Attention(nn.Module):
    """
    A flexible attention module supporting both self-attention and cross-attention mechanisms.

    This module implements a multi-head attention layer that can operate in either self-attention
    or cross-attention mode. The mode is determined by whether a context dimension is provided.
    The implementation uses scaled dot-product attention and supports optional bias terms and
    dropout regularization.

    Args:
        query_dim (int): The dimensionality of the query vectors.
        context_dim (int, optional): The dimensionality of the context (key/value) vectors.
            If None, the module operates in self-attention mode using query_dim. Default: None
        n_heads (int, optional): Number of attention heads for multi-head attention. Default: 8
        head_dim (int, optional): The dimension of each attention head. Default: 64
        dropout (float, optional): Dropout probability applied to the output. Default: 0.0
        qkv_format (str, optional): Format specification for QKV tensors. Default: "bshd"
        backend (str, optional): Backend to use for the attention operation. Default: "transformer_engine"

    Examples:
        >>> # Self-attention with 512 dimensions and 8 heads
        >>> self_attn = Attention(query_dim=512)
        >>> x = torch.randn(32, 16, 512)  # (batch_size, seq_len, dim)
        >>> out = self_attn(x)  # (32, 16, 512)

        >>> # Cross-attention
        >>> cross_attn = Attention(query_dim=512, context_dim=256)
        >>> query = torch.randn(32, 16, 512)
        >>> context = torch.randn(32, 8, 256)
        >>> out = cross_attn(query, context)  # (32, 16, 512)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim=None,
        n_heads=8,
        head_dim=64,
        dropout=0.0,
        qkv_format: str = "bshd",
        backend: str = "transformer_engine",
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__()
        log.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{n_heads} heads with a dimension of {head_dim}."
        )
        self.is_selfattn = context_dim is None  # self attention

        assert backend in ["transformer_engine", "torch", "minimal_a2a"], f"Invalid backend: {backend}"
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

        if self.backend == "transformer_engine":
            from transformer_engine.pytorch.attention import DotProductAttention

            self.attn_op = DotProductAttention(
                self.n_heads,
                self.head_dim,
                num_gqa_groups=self.n_heads,
                attention_dropout=0,
                qkv_format=qkv_format,
                attn_mask_type="no_mask",
            )
        elif self.backend == "minimal_a2a":
            self.attn_op = MinimalA2AAttnOp()
        elif self.backend == "torch":
            self.attn_op = torch_attention_op

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim

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
                if self.use_wan_fp32_strategy:  # wan will force q and k to fp32 before rotary pos emb
                    q = q.to(torch.float32)
                    k = k.to(torch.float32)
                q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
                k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
            return q, k, v

        q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)

        return q, k, v

    def compute_attention(self, q, k, v, video_size: Optional[VideoSize] = None):
        additional_args = {}
        if isinstance(self.attn_op, (NattenA2AAttnOp, NeighborhoodAttention)):
            additional_args["video_size"] = video_size

        result = self.attn_op(q, k, v, **additional_args)  # [B, S, H, D]
        return self.output_dropout(self.output_proj(result))

    def forward(
        self,
        x,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
        video_size: Optional[VideoSize] = None,
    ):
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
            rope_emb (Optional[Tensor]): RoPE embedding tensor, or no RoPE embeddings (i.e. in cross attention)
            video_size(VideoSize): Shape [T, H, W]
        """
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v, video_size=video_size)

    def set_context_parallel_group(self, process_group, ranks, stream):
        # self.attn_op.set_context_parallel_group(process_group, ranks, stream, cp_comm_type="a2a")
        self.attn_op.set_context_parallel_group(process_group, ranks, stream)


class I2VCrossAttention(Attention):
    def __init__(self, *args, img_latent_dim: int = 1024, **kwargs):
        super().__init__(*args, **kwargs)
        inner_dim = self.head_dim * self.n_heads
        self.k_img = nn.Linear(img_latent_dim, inner_dim, bias=False)
        self.v_img = nn.Linear(img_latent_dim, inner_dim, bias=False)
        self.k_img_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

    def init_weights(self) -> None:
        super().init_weights()
        torch.nn.init.trunc_normal_(self.k_img.weight, std=1.0 / math.sqrt(self._inner_dim))
        torch.nn.init.trunc_normal_(self.v_img.weight, std=1.0 / math.sqrt(self._inner_dim))
        self.k_img_norm.reset_parameters()

    def compute_qkv(
        self, x, context, rope_emb=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_context, img_context = context
        q, k, v = super().compute_qkv(x, text_context, rope_emb)
        k_img = self.k_img(img_context)
        v_img = self.v_img(img_context)
        # Rearrange k_img, v_img
        k_img, v_img = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (k_img, v_img),
        )

        return q, k, v, self.k_img_norm(k_img), v_img

    def compute_attention(self, q, k, v, k_img, v_img):
        result = self.attn_op(q, k, v)  # [B, S, H, D]
        result_img = self.attn_op(q, k_img, v_img)
        return self.output_dropout(self.output_proj(result + result_img))

    def forward(
        self,
        x,
        context=None,
        rope_emb=None,
    ):
        q, k, v, k_img, v_img = self.compute_qkv(x, context, rope_emb)
        return self.compute_attention(q, k, v, k_img, v_img)


class VideoPositionEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self._cp_group = None

    def enable_context_parallel(self, process_group: ProcessGroup):
        self._cp_group = process_group

    def disable_context_parallel(self):
        self._cp_group = None

    @property
    def seq_dim(self):
        return 1

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps=Optional[torch.Tensor]) -> torch.Tensor:
        """
        With CP, the function assume that the input tensor is already split.
        It delegates the embedding generation to generate_embeddings function.
        """
        B_T_H_W_C = x_B_T_H_W_C.shape
        if self._cp_group is not None:
            cp_ranks = get_process_group_ranks(self._cp_group)
            cp_size = len(cp_ranks)
            B, T, H, W, C = B_T_H_W_C
            B_T_H_W_C = (B, T * cp_size, H, W, C)
        embeddings = self.generate_embeddings(B_T_H_W_C, fps=fps)

        return self._split_for_context_parallel(embeddings)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]):
        raise NotImplementedError

    def _split_for_context_parallel(self, embeddings):
        if self._cp_group is not None:
            embeddings = split_inputs_cp(x=embeddings, seq_dim=self.seq_dim, cp_group=self._cp_group)
        return embeddings


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"

        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().to(self.dim_spatial_range.device)
        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.dim_spatial_range.device) / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.dim_spatial_range.device) / dim_t
        )

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        """
        Generate embeddings for the given input size.

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time, Height, Width, Channels).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor.

        Returns:
            Not specified in the original code snippet.
        """
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.float())
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.float())
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.float())

        B, T, H, W, _ = B_T_H_W_C
        assert H <= self.max_h and W <= self.max_w, (
            f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        )
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        if self.enable_fps_modulation:
            uniform_fps = (fps is None) or (fps.min() == fps.max())
            assert uniform_fps or B == 1 or T == 1, (
                "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"
            )

            # apply sequence scaling in temporal dimension
            if fps is None:  # image case
                assert T == 1, "T should be 1 for image batch."
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self):
        return 0


class LearnablePosEmbAxis(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        **kwargs,
    ):
        """
        Args:
            interpolation (str): we curretly only support "crop", ideally when we need extrapolation capacity, we should adjust frequency or other more advanced methods. they are not implemented yet.
        """
        del kwargs  # unused
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"
        self.model_channels = model_channels

        self.pos_emb_h = nn.Parameter(torch.zeros(len_h, model_channels))
        self.pos_emb_w = nn.Parameter(torch.zeros(len_w, model_channels))
        self.pos_emb_t = nn.Parameter(torch.zeros(len_t, model_channels))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.model_channels)
        torch.nn.init.trunc_normal_(self.pos_emb_h, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_w, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_t, std=std, a=-3 * std, b=3 * std)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, _ = B_T_H_W_C
        if self.interpolation == "crop":
            emb_h_H = self.pos_emb_h[:H]
            emb_w_W = self.pos_emb_w[:W]
            emb_t_T = self.pos_emb_t[:T]
            emb = (
                repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W)
                + repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W)
                + repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H)
            )
            assert list(emb.shape)[:4] == [B, T, H, W], f"bad shape: {list(emb.shape)[:4]} != {B, T, H, W}"
        else:
            raise ValueError(f"Unknown interpolation method {self.interpolation}")

        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(1e-6, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return emb / norm.to(emb.dtype)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T):
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        # wan need emb to be in fp32
        in_dype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return rearrange(emb.to(dtype=in_dype), "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        log.debug(
            f"Using AdaLN LoRA Flag:  {use_adaln_lora}. We enable bias if no AdaLN LoRA for backward compatibility."
        )
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            emb_B_T_D = emb
            adaln_lora_B_T_3D = None

        return emb_B_T_D, adaln_lora_B_T_3D


class FourierFeatures(nn.Module):
    """
    Implements a layer that generates Fourier features from input tensors, based on randomly sampled
    frequencies and phases. This can help in learning high-frequency functions in low-dimensional problems.

    [B] -> [B, D]

    Parameters:
        num_channels (int): The number of Fourier features to generate.
        bandwidth (float, optional): The scaling factor for the frequency of the Fourier features. Defaults to 1.
        normalize (bool, optional): If set to True, the outputs are scaled by sqrt(2), usually to normalize
                                    the variance of the features. Defaults to False.

    Example:
        >>> layer = FourierFeatures(num_channels=256, bandwidth=0.5, normalize=True)
        >>> x = torch.randn(10, 256)  # Example input tensor
        >>> output = layer(x)
        >>> print(output.shape)  # Expected shape: (10, 256)
    """

    def __init__(self, num_channels, bandwidth=1, normalize=False):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
        self.gain = np.sqrt(2) if normalize else 1
        self.bandwidth = bandwidth
        self.num_channels = num_channels

        self.reset_parameters()

    def reset_parameters(self) -> None:
        generator = torch.Generator()
        generator.manual_seed(0)
        self.freqs = (
            2 * np.pi * self.bandwidth * torch.randn(self.num_channels, generator=generator).to(self.freqs.device)
        )
        self.phases = 2 * np.pi * torch.rand(self.num_channels, generator=generator).to(self.freqs.device)

    def forward(self, x, gain: float = 1.0):
        """
        Apply the Fourier feature transformation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            gain (float, optional): An additional gain factor applied during the forward pass. Defaults to 1.

        Returns:
            torch.Tensor: The transformed tensor, with Fourier features applied.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
        x = x.cos().mul(self.gain * gain).to(in_dtype)
        return x


class PatchEmbed(nn.Module):
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the . This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches
    and embedding each patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int): The size of each spatial patch.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    """

    def __init__(
        self,
        spatial_patch_size,
        temporal_patch_size,
        in_channels=3,
        out_channels=768,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False
            ),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0, (
            f"H,W {(H, W)} should be divisible by spatial_patch_size {self.spatial_patch_size}"
        )
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(
        self,
        hidden_size,
        spatial_patch_size,
        temporal_patch_size,
        out_channels,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)

        self.layer_norm.reset_parameters()

    def forward(
        self,
        # x_BT_HW_D,
        x_B_T_H_W_D,
        emb_B_T_D,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_wan_fp32_strategy:
            assert emb_B_T_D.dtype == torch.float32
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                assert adaln_lora_B_T_3D is not None
                shift_B_T_D, scale_B_T_D = (
                    self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
                ).chunk(2, dim=-1)
            else:
                shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

            shift_B_T_1_1_D, scale_B_T_1_1_D = (
                rearrange(shift_B_T_D, "b t d -> b t 1 1 d"),
                rearrange(scale_B_T_D, "b t d -> b t 1 1 d"),
            )

            def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
                return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

            x_B_T_H_W_D = _fn(x_B_T_H_W_D, self.layer_norm, scale_B_T_1_1_D, shift_B_T_1_1_D)
            x_B_T_H_W_O = self.linear(
                x_B_T_H_W_D
            )  # O = spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels
        return x_B_T_H_W_O


class Block(nn.Module):
    """
    A transformer block that combines self-attention, cross-attention and MLP layers with AdaLN modulation.
    Each component (self-attention, cross-attention, MLP) has its own layer normalization and AdaLN modulation.

    Parameters:
        x_dim (int): Dimension of input features
        context_dim (int): Dimension of context features for cross-attention
        num_heads (int): Number of attention heads
        mlp_ratio (float): Multiplier for MLP hidden dimension. Default: 4.0
        use_adaln_lora (bool): Whether to use AdaLN-LoRA modulation. Default: False
        adaln_lora_dim (int): Hidden dimension for AdaLN-LoRA layers. Default: 256
        use_wan_fp32_strategy (bool): Whether to use Wan's FP32 strategy. Default: False
        If True, in Attention layer, if do self-attention, q and k will be forced to fp32 before rotary pos emb
        also, in modulation computation, force entire computation in fp32

    The block applies the following sequence:
    1. Self-attention with AdaLN modulation
    2. Cross-attention with AdaLN modulation
    3. MLP with AdaLN modulation

    Each component uses skip connections and layer normalization.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        backend: str = "transformer_engine",
        image_context_dim: Optional[int] = None,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
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
            self.cross_attn = Attention(
                x_dim, context_dim, num_heads, x_dim // num_heads, qkv_format="bshd", backend=backend
            )
        else:
            self.cross_attn = I2VCrossAttention(
                x_dim,
                context_dim,
                num_heads,
                x_dim // num_heads,
                img_latent_dim=image_context_dim,
                qkv_format="bshd",
                backend=backend,
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

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
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

        # Reshape tensors from (B, T, D) to (B, T, 1, 1, D) for broadcasting
        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )

        video_size = VideoSize(T=T, H=H, W=W)

        # (ahassani): Hack to correct `video_size` when CP is enabled.
        # I really don't like this, but there doesn't seem to be any central
        # piece of code that's responsible for handling CP/TP that also defines the
        # layout of shardings. Other parts of the code (i.e. RoPE) seem to make this
        # assumption that CP sharding is always done along T.
        if self.cp_size is not None and self.cp_size > 1:
            video_size = VideoSize(T=T * self.cp_size, H=H, W=W)

        result_B_T_H_W_D = rearrange(
            self.self_attn(
                # normalized_x_B_T_HW_D,
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
                video_size=video_size,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

        def _x_fn(
            _x_B_T_H_W_D,
            layer_norm_cross_attn,
            _scale_cross_attn_B_T_1_1_D,
            _shift_cross_attn_B_T_1_1_D,
            _gate_cross_attn_B_T_1_1_D,
        ):
            _normalized_x_B_T_H_W_D = _fn(
                _x_B_T_H_W_D, layer_norm_cross_attn, _scale_cross_attn_B_T_1_1_D, _shift_cross_attn_B_T_1_1_D
            )
            _result_B_T_H_W_D = rearrange(
                self.cross_attn(
                    rearrange(_normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                    crossattn_emb,
                    rope_emb=rope_emb_L_1_1_D,
                ),
                "b (t h w) d -> b t h w d",
                t=T,
                h=H,
                w=W,
            )
            # _x_B_T_H_W_D = _x_B_T_H_W_D + _gate_cross_attn_B_T_1_1_D * _result_B_T_H_W_D
            return _result_B_T_H_W_D

        result_B_T_H_W_D = _x_fn(
            x_B_T_H_W_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_1_1_D,
            shift_cross_attn_B_T_1_1_D,
            gate_cross_attn_B_T_1_1_D,
        )
        x_B_T_H_W_D = result_B_T_H_W_D * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_mlp,
            scale_mlp_B_T_1_1_D,
            shift_mlp_B_T_1_1_D,
        )
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D
        return x_B_T_H_W_D


class MiniTrainDIT(WeightTrainingStat):
    """
    A clean impl of DIT that can load and  reproduce the training results of the original DIT model in edify_video/v4~(cosmos 1)
    A general implementation of adaln-modulated VIT-like~(DiT) transformer for video processing.

    Args:
        max_img_h (int): Maximum height of the input images.
        max_img_w (int): Maximum width of the input images.
        max_frames (int): Maximum number of frames in the video sequence.
        in_channels (int): Number of input channels (e.g., RGB channels for color images).
        out_channels (int): Number of output channels.
        patch_spatial (int): Spatial resolution of patches for input processing.
        patch_temporal (int): Temporal resolution of patches for input processing.
        concat_padding_mask (bool): If True, includes a mask channel in the input to handle padding.
        model_channels (int): Base number of channels used throughout the model.
        num_blocks (int): Number of transformer blocks.
        num_heads (int): Number of heads in the multi-head attention layers.
        mlp_ratio (float): Expansion ratio for MLP blocks.
        crossattn_emb_channels (int): Number of embedding channels for cross-attention.
        extra_image_context_dim (int): Number of embedding channels for extra image context.
        pos_emb_cls (str): Type of positional embeddings.
        pos_emb_learnable (bool): Whether positional embeddings are learnable.
        pos_emb_interpolation (str): Method for interpolating positional embeddings.
        min_fps (int): Minimum frames per second.
        max_fps (int): Maximum frames per second.
        use_adaln_lora (bool): Whether to use AdaLN-LoRA.
        adaln_lora_dim (int): Dimension for AdaLN-LoRA.
        rope_h_extrapolation_ratio (float): Height extrapolation ratio for RoPE.
        rope_w_extrapolation_ratio (float): Width extrapolation ratio for RoPE.
        rope_t_extrapolation_ratio (float): Temporal extrapolation ratio for RoPE.
        extra_per_block_abs_pos_emb (bool): Whether to use extra per-block absolute positional embeddings.
        extra_h_extrapolation_ratio (float): Height extrapolation ratio for extra embeddings.
        extra_w_extrapolation_ratio (float): Width extrapolation ratio for extra embeddings.
        extra_t_extrapolation_ratio (float): Temporal extrapolation ratio for extra embeddings.
        n_dense_blocks (`int`, *optional*, defaults to -1):
            Number of blocks that will remain dense (not replaced with sparse attention)
            If -1, no blocks are replaced with sparse attention
            If 0, all blocks use sparse attention
            Otherwise, n_dense_blocks blocks will remain dense, distributed evenly across the network
        natten_parameters (`dict`, *optional*, defaults to None):
            NATTEN (Sparse attention) parameter list.
            The list length must be the same as the number of layers, with each list element
            indicating NATTEN parameters for that layer. If None, NATTEN will not be used in that
            layer and it would remain a full dense self attention. If not None, it must be a
            dictionary/mapping with at least the following key:
              - window_size: `tuple` of size 3 indicating neighborhood attention window size.
                window size of -1 along any dimension means self attention.
            Other optional parameters and their keys:
              - stride: `tuple` of size 3 indicating neighborhood attention stride value.
                stride = 1 is standard neighborhood attention, stride = window size means
                blocked/window self attention (WSA) along that dimension. Any other values are
                strided neighborhood attention. Refer to the GNA paper for more information.

              - dilation: `tuple` of size 3 indicating neighborhood attention dilation value.
                dilation = 1 is standard neighborhood attention. Refer to the DiNAT paper for more
                information.

              - is_causal: `tuple` of 3 booleans indicating whether causal masking is enabled for
                any of the T, H, W dimensions.
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
        atten_backend: str = "transformer_engine",
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
        n_dense_blocks: int = -1,
        natten_parameters: Union[dict, list] = None,
        # if True, will closely match wan's strategy to use fp32 in certain layers/operations
        use_wan_fp32_strategy: bool = False,
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
                Block(
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
                nn.Linear(
                    extra_image_context_dim, model_channels, bias=True
                ),  # help distinguish between image and video context
                nn.GELU(),
            )

        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )

        self.init_weights()
        self.enable_selective_checkpoint(sac_config, self.blocks)

        # Replace self-attention with sparse attention if specified
        if n_dense_blocks != -1:
            self = replace_selfattn_op_with_sparse_attn_op(self, n_dense_blocks, natten_parameters=natten_parameters)

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
        """
        Prepares an embedded sequence tensor by applying positional embeddings and handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                    If None, a default value (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the positional embedding class
                (`self.pos_emb_cls`) includes 'rope'. Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are generated using the
            `self.pos_embedder` with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering fps.
        """
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
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]

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
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
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

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        B, T, H, W, D = x_B_T_H_W_D.shape
        # x_B_THW_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")

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

        # x_B_T_H_W_D = rearrange(x_B_THW_D, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        # O = out_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size
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
        # pos_embedder
        self.pos_embedder.disable_context_parallel()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.disable_context_parallel()

        # attention
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = False

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None):
        # pos_embedder
        self.pos_embedder.enable_context_parallel(process_group=process_group)
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.enable_context_parallel(process_group=process_group)

        # attention
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


def replace_selfattn_op_with_sparse_attn_op(
    model: MiniTrainDIT, n_dense_blocks: int = 0, natten_parameters: Union[dict, list] = None
) -> MiniTrainDIT:
    """
    Replace the self-attention operator with a sparse self-attention operator.

    Args:
        model: MiniTrainDIT instance
        n_dense_blocks: Number of blocks that will remain dense (not replaced with NeighborhoodAttention)
            If 0, all blocks use NeighborhoodAttention.
            If -1, return model directly without any modifications.
            Otherwise, n_dense_blocks blocks will remain dense, distributed evenly across the network.

    Returns:
        Modified instance
    """
    # Special case: return model directly without modifications
    if n_dense_blocks == -1:
        return model

    num_blocks = len(model.blocks)

    if natten_parameters is None:
        raise ValueError("Please specify natten_parameters when n_dense_blocks > -1.")

    if isinstance(natten_parameters, Sequence) and len(natten_parameters) != num_blocks:
        raise ValueError(
            "List of NATTEN parameters must be the same length as the number of blocks, "
            f"got {len(natten_parameters)=} != {num_blocks=}."
        )

    if isinstance(natten_parameters, Sequence) and n_dense_blocks > 0:
        log.warning(f"NATTEN parameters was a list; ignoring {n_dense_blocks=}.")

    if isinstance(natten_parameters, Sequence):
        natten_parameters_list = natten_parameters
    else:
        if n_dense_blocks >= num_blocks:
            raise ValueError(f"n_dense_blocks ({n_dense_blocks}) must be less than the number of blocks ({num_blocks})")

        # Determine which blocks should remain dense
        dense_indices = set()

        if n_dense_blocks > 0:
            # General rule: distribute n_dense_blocks blocks evenly across the network
            if n_dense_blocks == 1:
                # Special case: just the middle block
                dense_indices.add(num_blocks // 2)
            else:
                # For multiple blocks, distribute them evenly from start to end
                indices = np.linspace(0, num_blocks - 1, n_dense_blocks, dtype=int)
                dense_indices.update(indices.tolist())

        natten_parameters_list = [None if i in dense_indices else natten_parameters for i in range(num_blocks)]

    # Replace self-attention with NeighborhoodAttention for non-dense blocks
    for i, block in enumerate(model.blocks):
        natten_params = natten_parameters_list[i]
        if natten_params is not None:
            natten_parameters_layer = {k: v for k, v in natten_params.items()}
            natten_parameters_layer["layer_id"] = i
            if block.self_attn.backend == "minimal_a2a":
                sparse_attn_op = NattenA2AAttnOp(natten_parameters=natten_parameters_layer)
            else:
                raise NotImplementedError(
                    f"Using sparsity with attention backend {block.self_attn.backend} is not supported."
                )

            block.self_attn.register_module("attn_op", sparse_attn_op)

    return model
