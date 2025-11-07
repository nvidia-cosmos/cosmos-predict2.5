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

import torch
from einops import rearrange
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


def prepare_temporal_only_causal_blockmask(
    device: torch.device | str,
    num_frames: int,
    frame_seqlen: int,
) -> BlockMask:
    """Construct a FlexAttention BlockMask for temporal-only causal attention.

    Within a frame: allow full attention over the frame. Across frames: allow attention to all previous frames.
    """
    total_len = num_frames * frame_seqlen
    # Right-pad to 128 multiple; FlexAttention op will pad Q/K/V in the backend impl
    padded_len = ((total_len + 127) // 128) * 128

    def attention_mask(b, h, q_idx, kv_idx):
        # Map token index -> frame index via integer division (tensor ops only)
        qf = q_idx // frame_seqlen
        kf = kv_idx // frame_seqlen
        return kf <= qf

    block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=padded_len,
        KV_LEN=padded_len,
        _compile=False,
        device=device,
    )
    return block_mask


def build_blockwise_causal_mask(
    device: torch.device,
    num_frames: int,
    frame_seqlen: int,
    num_frame_per_block: int = 1,
    height: int = 1,
    width: int = 1,
) -> torch.Tensor:
    """Build a boolean attention mask of shape [S, S] where S=T*H*W.

    Rules:
      - Causal within each temporal block of num_frame_per_block frames.
    """
    num_frames_total = num_frames
    tokens_per_frame = height * width
    frame_mask = torch.zeros((num_frames_total, num_frames_total), dtype=torch.bool, device=device)
    for query_frame in range(num_frames_total):
        block_start = (query_frame // num_frame_per_block) * num_frame_per_block
        block_end = min(block_start + num_frame_per_block, num_frames_total)
        kv_frame_start = block_start
        kv_frame_end = min(query_frame, block_end - 1)
        if kv_frame_end >= kv_frame_start:
            frame_mask[query_frame, kv_frame_start : kv_frame_end + 1] = True
        frame_mask[query_frame, query_frame] = True

    mask = rearrange(frame_mask, "t1 t2 -> t1 1 t2 1").repeat(1, tokens_per_frame, 1, tokens_per_frame)
    mask = rearrange(mask, "t1 n t2 m -> (t1 n) (t2 m)")
    return mask


def build_blockwise_causal_mask_flex(
    device: torch.device,
    num_frames: int,
    frame_seqlen: int,
    compile_mask: bool = True,
) -> BlockMask:
    total_length = num_frames * frame_seqlen
    padded_length = ((total_length + 127) // 128) * 128

    # Compute a tile-aligned temporal block size K so that K * frame_seqlen is divisible by 128
    # and K covers all frames to preserve full temporal causal attention.
    gcd128 = math.gcd(frame_seqlen, 128)
    tile_factor = 128 // gcd128 if gcd128 != 0 else 1
    num_frame_per_block = max(tile_factor * ((num_frames + tile_factor - 1) // tile_factor), 1)

    def attention_mask(b, h, q_idx, kv_idx):
        # Compute all indices via tensor arithmetic to be vmap-compatible
        qf = q_idx // frame_seqlen
        kf = kv_idx // frame_seqlen
        block_start = (qf // num_frame_per_block) * num_frame_per_block
        # Clamp block_end to [0, num_frames] using tensor min to avoid Python control flow
        num_frames_t = torch.as_tensor(num_frames, device=q_idx.device, dtype=qf.dtype)
        block_end = torch.minimum(block_start + num_frame_per_block, num_frames_t)
        return (kf >= block_start) & (kf <= qf) & (kf < block_end)

    return create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=padded_length,
        KV_LEN=padded_length,
        _compile=compile_mask,
        device=device,
    )


def build_blockwise_causal_mask_i2v(
    device: torch.device,
    num_frames: int,
    frame_seqlen: int,
    num_frame_per_block: int = 1,
    height: int = 1,
    width: int = 1,
) -> torch.Tensor:
    """i2v-style blockwise mask with first frame isolated.

    Frame 0 only attends to itself. Remaining frames are grouped into blocks
    starting at frame 1 with size num_frame_per_block. Within each block,
    queries may attend to keys up to the block end subject to local window,
    but not beyond their own frame (causal within block for SDPA).
    """
    num_frames_total = num_frames
    tokens_per_frame = height * width
    frame_mask = torch.zeros((num_frames_total, num_frames_total), dtype=torch.bool, device=device)
    if num_frames_total > 0:
        frame_mask[0, 0] = True
    frame_index = 1
    while frame_index < num_frames_total:
        block_start = frame_index
        block_end = min(block_start + num_frame_per_block, num_frames_total)
        for query_frame in range(block_start, block_end):
            kv_frame_start = block_start
            kv_frame_end = query_frame
            if kv_frame_end >= kv_frame_start:
                frame_mask[query_frame, kv_frame_start : kv_frame_end + 1] = True
            frame_mask[query_frame, query_frame] = True
        frame_index = block_end

    mask = rearrange(frame_mask, "t1 t2 -> t1 1 t2 1").repeat(1, tokens_per_frame, 1, tokens_per_frame)
    mask = rearrange(mask, "t1 n t2 m -> (t1 n) (t2 m)")
    return mask


def build_blockwise_causal_mask_i2v_flex(
    device: torch.device,
    num_frames: int,
    frame_seqlen: int,
    compile_mask: bool = True,
) -> BlockMask:
    total_length = num_frames * frame_seqlen
    padded_length = ((total_length + 127) // 128) * 128

    # Compute a tile-aligned temporal block size K for frames 1..T-1
    frames_after_first = max(num_frames - 1, 0)
    gcd128 = math.gcd(frame_seqlen, 128)
    tile_factor = 128 // gcd128 if gcd128 != 0 else 1
    num_frame_per_block = max(tile_factor * ((frames_after_first + tile_factor - 1) // tile_factor), 1)

    def attention_mask(b, h, q_idx, kv_idx):
        # Tensor-only arithmetic; no Python control flow on tensors
        qf = q_idx // frame_seqlen
        kf = kv_idx // frame_seqlen

        # Case 1: queries in frame 0 can only attend to frame 0
        is_first_query = qf == 0
        allow_first = kf == 0

        # Case 2: queries in frames 1..T-1 use i2v-style blockwise causal within [block_start, min(block_end, T))
        qf_minus_1 = torch.clamp(qf - 1, min=0)
        block_start = (qf_minus_1 // num_frame_per_block) * num_frame_per_block + 1
        num_frames_t = torch.as_tensor(num_frames, device=q_idx.device, dtype=qf.dtype)
        block_end = torch.minimum(block_start + num_frame_per_block, num_frames_t)
        allow_block = (kf >= block_start) & (kf <= qf) & (kf < block_end)

        # Select per-query branch without Python if
        return torch.where(is_first_query, allow_first, allow_block)

    return create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=padded_length,
        KV_LEN=padded_length,
        _compile=compile_mask,
        device=device,
    )
