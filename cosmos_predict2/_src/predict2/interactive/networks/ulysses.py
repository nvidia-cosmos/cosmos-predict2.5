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

# This script implements all-to-all attention for CP.
# Implemented by Qinsheng Zhang
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import CheckpointPolicy


def create_policy_fn():
    r"""Creates a checkpointing policy function for memory optimization.

    This policy determines which operations should be saved during checkpointing
    and which should be recomputed. It prioritizes saving attention and matrix
    multiplication operations while preferring to recompute other operations.

    Returns:
        A function that takes (ctx, op, *args, **kwargs) and returns a CheckpointPolicy.
    """

    def policy_fn(ctx, op, *args, **kwargs):
        op_name = str(op)
        if "aten._scaled_dot_product_efficient_attention.default" in op_name or "aten.addmm.default" in op_name:
            return CheckpointPolicy.MUST_SAVE

        return CheckpointPolicy.PREFER_RECOMPUTE

    return policy_fn


def post_all2all(local_seq_2_local_head: bool, seq_world_size: int):
    r"""Creates a post-merging function after all-to-all communication.
    Args:
        local_seq_2_local_head (bool): If True, moves sequence dimension to the head dimension
        seq_world_size (int): Number of GPUs in the process group.

    Returns:
        A function that rearranges the communicated tensor to the desired format.
    """

    def post_func(input):
        if local_seq_2_local_head:
            output = rearrange(input, "w bs seq h d -> bs (w seq) h d")
        else:
            output = rearrange(input, "w bs s h d -> bs s (w h) d", w=seq_world_size)
        return output

    return post_func


def single_all_to_all(
    input: Tensor, local_seq_2_local_head: bool, group: dist.ProcessGroup, async_op: bool = False
) -> Tensor:
    r"""Performs all-to-all communication for attention.
    The idea is two-fold
    1. First, we split the tensors by head dimension.
    2. THen combine the sequence dimension from difference processes.
    This will allow us to do the attention.

    After attention is done, we have to do the inverse of 2 and 1.
    This function supports communication for both cases.

    Args:
        input (Tensor): Input tensor to be redistributed.
        local_seq_2_local_head (bool): If True, splits the head and merges the sequence dimension. Implements 1 and 2.
            Otherwise, splits the sequence and merges the head dimension. Implements inverse of 2 and 1.
        group (dist.ProcessGroup): Process group for distributed communication.
        async_op (bool): Whether to perform asynchronous communication.

    Returns:
        Tensor: Redistributed tensor in the desired format.

    Raises:
        AssertionError: If the number of heads is not divisible by the sequence
                      parallel size.
    """
    # Get number of GPUs in the process group
    seq_world_size = dist.get_world_size(group)

    if local_seq_2_local_head:
        bs, local_seq_len, num_total_head, head_dim = input.shape
        assert num_total_head % seq_world_size == 0, (
            f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
        )
        input_t = rearrange(
            input, "bs seq_len (w h) d -> w bs seq_len h d", w=seq_world_size, h=num_total_head // seq_world_size
        ).contiguous()
        post_all2all_fun = post_all2all(local_seq_2_local_head, seq_world_size)
    else:
        bs, global_seq_len, num_local_head, head_dim = input.shape
        input_t = rearrange(
            input, "bs (w s) h d -> w bs s h d", w=seq_world_size, s=global_seq_len // seq_world_size
        ).contiguous()
        post_all2all_fun = post_all2all(local_seq_2_local_head, seq_world_size)

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group, async_op=async_op)

    res = post_all2all_fun(output)
    return res


class _SeqAllToAll(torch.autograd.Function):
    r"""Custom autograd function for sequence all-to-all communication.

    This class implements the forward and backward passes for sequence all-to-all
    communication, handling both the computation and gradient computation.
    """

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, local_seq_2_local_head: bool) -> Tensor:
        r"""Forward pass for sequence all-to-all communication.

        Args:
            ctx: Context object for storing information needed in backward pass.
            group: Process group for distributed communication.
            input: Input tensor to be redistributed.
            local_seq_2_local_head: Direction of redistribution.

        Returns:
            Redistributed tensor.
        """
        ctx.group = group
        res = single_all_to_all(input, local_seq_2_local_head, group, False)
        ctx.local_seq_2_local_head = local_seq_2_local_head
        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, not ctx.local_seq_2_local_head), None)


class DistributedAttention(torch.nn.Module):
    r"""Distributed attention module with all-to-all communication.

    Args:
        local_attention (torch.nn.Module): Local attention module for computation
                                         on each GPU.
        sequence_process_group (dist.ProcessGroup): Process group for sequence
                                                 parallel communication.
    """

    def __init__(
        self,
        local_attention: torch.nn.Module,
        sequence_process_group: dist.ProcessGroup,
    ) -> None:
        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group

    def set_context_parallel_group(self, process_group, ranks, stream):
        r"""Sets the context parallel group for distributed computation."""
        del ranks, stream
        self.spg = process_group

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Forward pass for distributed attention computation.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            attn_mask: Optional attention mask.

        Returns:
            Attention output tensor.
        """
        if self.spg is None:
            output = self.local_attn(query, key, value, attn_mask=attn_mask)
        else:
            query_layer = _SeqAllToAll.apply(self.spg, query, True)
            key_layer = _SeqAllToAll.apply(self.spg, key, True)
            value_layer = _SeqAllToAll.apply(self.spg, value, True)
            context_layer = self.local_attn(query_layer, key_layer, value_layer, attn_mask=attn_mask)

            output = _SeqAllToAll.apply(self.spg, context_layer, False)
        return rearrange(output, "bs s h d -> bs s (h d)")
