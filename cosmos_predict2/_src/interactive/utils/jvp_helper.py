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
from typing import Any, Callable, Tuple, Union

import torch
import torch.distributed as dist
from einops import rearrange
from torch import Tensor, nn

from cosmos_predict2._src.predict2.networks.a2a_cp import async_a2a_communicate
from cosmos_predict2._src.predict2.utils.flash_attention_jvp_triton import _attention

TensorWithT = Tuple[torch.Tensor, torch.Tensor]


class JVP(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        withT = kwargs.pop("withT", False)
        if withT:
            return self._forward_jvp(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward_jvp(self, *args, **kwargs):
        raise NotImplementedError

    def _forward(self, *args, **kwargs):
        raise NotImplementedError


def torch_attention_op_withT(q_B_S_H_D_withT: TensorWithT, k_B_S_H_D_withT: TensorWithT, v_B_S_H_D_withT: TensorWithT):
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
    q_B_S_H_D, t_q_B_S_H_D = q_B_S_H_D_withT
    k_B_S_H_D, t_k_B_S_H_D = k_B_S_H_D_withT
    v_B_S_H_D, t_v_B_S_H_D = v_B_S_H_D_withT
    in_q_shape = q_B_S_H_D.shape
    in_k_shape = k_B_S_H_D.shape
    q_B_H_S_D = rearrange(q_B_S_H_D, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
    t_q_B_H_S_D = rearrange(t_q_B_S_H_D, "b ... h k -> b h ... k").view(
        in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1]
    )
    k_B_H_S_D = rearrange(k_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    t_k_B_H_S_D = rearrange(t_k_B_S_H_D, "b ... h v -> b h ... v").view(
        in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1]
    )
    v_B_H_S_D = rearrange(v_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    t_v_B_H_S_D = rearrange(t_v_B_S_H_D, "b ... h v -> b h ... v").view(
        in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1]
    )
    result_B_H_S_D, t_result_B_H_S_D = _attention.apply(
        q_B_H_S_D, k_B_H_S_D, v_B_H_S_D, t_q_B_H_S_D, t_k_B_H_S_D, t_v_B_H_S_D
    )
    result_B_S_H_D = rearrange(result_B_H_S_D, "b h ... l -> b ... h l")
    t_result_B_S_H_D = rearrange(t_result_B_H_S_D, "b h ... l -> b ... h l")

    return (result_B_S_H_D, t_result_B_S_H_D.detach())


class _SeqAllToAllQKVWithT(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        q: Tensor,
        t_q: Tensor,
        k: Tensor,
        t_k: Tensor,
        v: Tensor,
        t_v: Tensor,
        cp_size: int,
        cp_stream: torch.cuda.Stream,
        local_seq_2_local_head: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.cp_stream = cp_stream
        ctx.local_seq_2_local_head = local_seq_2_local_head
        q, t_q, k, t_k, v, t_v = async_a2a_communicate(
            [q, t_q, k, t_k, v, t_v], cp_size, group, cp_stream, local_seq_2_local_head
        )
        return q, t_q, k, t_k, v, t_v

    @staticmethod
    def backward(
        ctx: Any, *grad_output: Tensor
    ) -> Tuple[None, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, None, None, None]:
        q_grad, t_q_grad, k_grad, t_k_grad, v_grad, t_v_grad = _SeqAllToAllQKVWithT.apply(
            ctx.group, *grad_output, ctx.cp_size, ctx.cp_stream, not ctx.local_seq_2_local_head
        )
        return (None, q_grad, t_q_grad, k_grad, t_k_grad, v_grad, t_v_grad, None, None, None)


class _SeqAllToAllOutputWithT(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        output,
        t_output,
        cp_size: int,
        cp_stream: torch.cuda.Stream,
        local_seq_2_local_head: bool,
    ) -> Tuple[Tensor, Tensor]:
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.cp_stream = cp_stream
        ctx.local_seq_2_local_head = local_seq_2_local_head
        output, t_output = async_a2a_communicate([output, t_output], cp_size, group, cp_stream, local_seq_2_local_head)
        return output, t_output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, Tensor, None, None, None]:
        output_grad, t_output_grad = _SeqAllToAllOutputWithT.apply(
            ctx.group, *grad_output, ctx.cp_size, ctx.cp_stream, not ctx.local_seq_2_local_head
        )
        return (None, output_grad, t_output_grad, None, None, None)


class MinimalA2AAttnOpWithT(torch.nn.Module):
    def __init__(self, local_attn_T: Union[nn.Module, Callable] = torch_attention_op_withT):
        super().__init__()
        self.local_attn_T = local_attn_T
        self.pg = None
        self.stream = None

    def forward(
        self, query_withT: TensorWithT, key_withT: TensorWithT, value_withT: TensorWithT, *args: Any, **kwargs
    ) -> TensorWithT:
        del args, kwargs
        if self.pg is None:
            output_B_S_H_D, t_output_B_S_H_D = self.local_attn_T(query_withT, key_withT, value_withT)
        else:
            pg_size = dist.get_world_size(self.pg)
            if pg_size < 2:
                output_B_S_H_D, t_output_B_S_H_D = self.local_attn_T(query_withT, key_withT, value_withT)
            else:
                query_B_S_H_D, t_query_B_S_H_D = query_withT
                key_B_S_H_D, t_key_B_S_H_D = key_withT
                value_B_S_H_D, t_value_B_S_H_D = value_withT

                (
                    query_B_S_H_D,
                    t_query_B_S_H_D,
                    key_B_S_H_D,
                    t_key_B_S_H_D,
                    value_B_S_H_D,
                    t_value_B_S_H_D,
                ) = _SeqAllToAllQKVWithT.apply(
                    self.pg,
                    query_B_S_H_D,
                    t_query_B_S_H_D,
                    key_B_S_H_D,
                    t_key_B_S_H_D,
                    value_B_S_H_D,
                    t_value_B_S_H_D,
                    pg_size,
                    self.stream,
                    True,
                )
                context_B_S_H_D, t_context_B_S_H_D = self.local_attn_T(
                    tuple([query_B_S_H_D, t_query_B_S_H_D]),
                    tuple([key_B_S_H_D, t_key_B_S_H_D]),
                    tuple([value_B_S_H_D, t_value_B_S_H_D]),
                )
                output_B_S_H_D, t_output_B_S_H_D = _SeqAllToAllOutputWithT.apply(
                    self.pg, context_B_S_H_D, t_context_B_S_H_D, pg_size, self.stream, False
                )
        return (
            rearrange(output_B_S_H_D, "b ... h l -> b ... (h l)"),
            rearrange(t_output_B_S_H_D, "b ... h l -> b ... (h l)").detach(),
        )

    def set_context_parallel_group(self, process_group, ranks, stream):
        del ranks
        self.pg = process_group
        self.stream = stream


def naive_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1).to(query.dtype)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def naive_attention_op(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D):
    in_q_shape = q_B_S_H_D.shape
    in_k_shape = k_B_S_H_D.shape
    q_B_H_S_D = rearrange(q_B_S_H_D, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
    k_B_H_S_D = rearrange(k_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    v_B_H_S_D = rearrange(v_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    return rearrange(naive_scaled_dot_product_attention(q_B_H_S_D, k_B_H_S_D, v_B_H_S_D), "b h ... l -> b ... h l")


def torch_attention_op(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D):
    in_q_shape = q_B_S_H_D.shape
    in_k_shape = k_B_S_H_D.shape
    q_B_H_S_D = rearrange(q_B_S_H_D, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
    k_B_H_S_D = rearrange(k_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    v_B_H_S_D = rearrange(v_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    return rearrange(
        torch.nn.functional.scaled_dot_product_attention(q_B_H_S_D, k_B_H_S_D, v_B_H_S_D), "b h ... l -> b ... h l"
    )
