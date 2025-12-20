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

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class KVCacheConfig:
    run_with_kv: bool = False
    store_kv: bool = False
    current_idx: int = 0
    recompute_cross_attn_kv: bool = False


class AttentionOpWithKVCache(nn.Module):
    """A thin wrapper that adds K/V caching to an existing attention op.

    This wrapper expects the wrapped op to accept (q, k, v, attn_mask=None)
    and return attention outputs with heads already flattened on the last dim.

    Cache semantics:
    - Cache entries are stored as per-chunk tensors, where each chunk corresponds
      to one latent frame composed of HxW tokens (after patchify).
    - The `max_cache_size` capacity therefore refers to the number of latent
      frames (chunks), NOT the number of individual tokens.
    - When `max_cache_size` is None, the cache grows without an automatic
      rolling window; otherwise, it acts as a rolling window of at most
      `max_cache_size` frames. Upon overflow, the oldest frames are dropped.
    """

    def __init__(self, attn_op: nn.Module | Any, max_cache_size: Optional[int] = None):
        """Initialize the KV cache wrapper.

        Args:
            attn_op: The underlying attention operation (q, k, v[, attn_mask]) -> out.
            max_cache_size: Optional capacity measured in number of latent frames
                (chunks). Each chunk is a single frame worth of HxW tokens. If None,
                the cache does not enforce a rolling capacity.
        """
        super().__init__()
        self.attn_op = attn_op
        self.reset_kv_cache(max_cache_size=max_cache_size)
        self.pg: Optional[Any] = None
        self.stream: Optional[Any] = None

    def reset_kv_cache(self, max_cache_size: Optional[int] = None) -> None:
        """Reset/initialize the KV caches.

        Args:
            max_cache_size: Optional capacity measured in number of latent frames
                (chunks). Each chunk is a single frame worth of HxW tokens. If None,
                the cache does not enforce a rolling capacity.
        """
        # Initialize list-based caches and optionally set capacity in chunks
        self.start_idx = 0
        self.k_cache: list[torch.Tensor | None] = [None] * (max_cache_size or 99999)
        self.v_cache: list[torch.Tensor | None] = [None] * (max_cache_size or 99999)
        self.max_cache_size = max_cache_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_cache_cfg: KVCacheConfig,
        **kwargs,
    ) -> torch.Tensor:
        assert self.k_cache is not None and self.v_cache is not None, (
            "KV cache is not initialized. Call reset_kv_cache() first."
        )

        # Store into cache at start_idx location (list-based)
        if kv_cache_cfg.store_kv:
            index = int(kv_cache_cfg.current_idx)
            self.k_cache[index] = k.detach()
            self.v_cache[index] = v.detach()

        # Prepend cached prefix up to start_idx (list-based)
        if kv_cache_cfg.run_with_kv and kv_cache_cfg.current_idx > 0:
            history_k = self.k_cache[self.start_idx : kv_cache_cfg.current_idx]
            history_v = self.v_cache[self.start_idx : kv_cache_cfg.current_idx]
            assert not any(x is None for x in history_k)
            assert not any(x is None for x in history_v)
            k_out = torch.cat(history_k + [k], dim=1)  # type: ignore
            v_out = torch.cat(history_v + [v], dim=1)  # type: ignore
        else:
            k_out = k
            v_out = v

        # Enforce rolling capacity in number of cached chunks (frames)
        if kv_cache_cfg.run_with_kv and self.max_cache_size is not None:
            # Instead of deleting, just update start_idx for rolling window
            self.start_idx = max(0, int(kv_cache_cfg.current_idx) - self.max_cache_size)

        return self.attn_op(q, k_out, v_out, **kwargs)

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        self.attn_op.set_context_parallel_group(process_group, ranks, stream, cp_comm_type=cp_comm_type)  # type: ignore


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

    def frame(self, t_idx: int) -> "VideoSeqPos":
        """Return a `VideoSeqPos` view for a single frame at absolute index `t_idx`.

        This is useful for streaming / KV-cache inference where the model is run on
        one frame at a time but RoPE positions must reflect global video indices.
        """
        t_idx = int(t_idx)
        if t_idx < 0 or t_idx >= int(self.T):
            raise IndexError(f"t_idx out of range: {t_idx} (valid: [0, {self.T}))")
        tokens_per_frame = int(self.H) * int(self.W)
        start = t_idx * tokens_per_frame
        end = start + tokens_per_frame
        return VideoSeqPos(
            T=1,
            H=int(self.H),
            W=int(self.W),
            pos_h=self.pos_h[start:end],
            pos_w=self.pos_w[start:end],
            pos_t=self.pos_t[start:end],
        )
