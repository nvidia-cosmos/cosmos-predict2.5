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

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.graph import create_cuda_graph
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT
from cosmos_predict2._src.predict2.utils.kv_cache import KVCacheConfig, VideoSeqPos


class CausalDIT(MiniTrainDIT):
    """Causal DiT variant that enables temporal-causal masking and KV cache.

    Notes:
    - Temporal causality is installed by replacing the local attention op with a
      closure that always uses a prebuilt Flex BlockMask (Wan-style).
    - KV cache is injected by wrapping the attention op via AttentionOpWithKVCache.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def precapture_cuda_graphs(
        self,
        batch_size: int,
        max_t: int,
        token_h: int,
        token_w: int,
        N_ctx: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Pre-capture CUDA graphs for each max_t frames and each block."""
        if not self.use_cuda_graphs or max_t <= self.cuda_graphs_max_t_registered:
            return

        assert self.cuda_graphs is None, "CUDA graphs already precaptured"
        self.cuda_graphs = {}

        head_dim = self.model_channels // self.num_heads
        # Determine context embedding dim; fall back to model_channels if no projection or unknown module type
        if (
            hasattr(self, "crossattn_proj")
            and isinstance(self.crossattn_proj, torch.nn.Sequential)
            and len(self.crossattn_proj) > 0
        ):
            proj0 = self.crossattn_proj[0]
            D_ctx = proj0.out_features if isinstance(proj0, torch.nn.Linear) else self.model_channels
        else:
            D_ctx = self.model_channels

        log.info(f"[CUDA Graph Precapture] Capturing graphs for {max_t} frames")

        for t_idx in range(max_t):
            x_dummy = torch.randn(batch_size, 1, token_h, token_w, self.model_channels, device=device, dtype=dtype)
            emb_dummy = torch.randn(batch_size, 1, self.model_channels, device=device, dtype=dtype)
            rope_dummy = torch.randn(token_h * token_w, 1, 1, head_dim, device=device)
            crossattn_dummy = torch.randn(batch_size, N_ctx, D_ctx, device=device, dtype=dtype)
            adaln_dummy = torch.randn(batch_size, 1, 3 * self.model_channels, device=device, dtype=dtype)

            kv_cache_cfg = KVCacheConfig(run_with_kv=True, store_kv=True, current_idx=t_idx)

            block_kwargs = {
                "emb_B_T_D": emb_dummy,
                "crossattn_emb": crossattn_dummy,
                "rope_emb_L_1_1_D": rope_dummy,
                "adaln_lora_B_T_3D": adaln_dummy,
                "extra_per_block_pos_emb": None,
                "kv_cache_cfg": kv_cache_cfg,
            }

            # Capture CUDA graphs for all blocks at this t_idx
            t_idx_key = f"t{t_idx}"
            shapes_graphs = create_cuda_graph(
                self.cuda_graphs,
                self.blocks,
                [x_dummy],
                block_kwargs,
                extra_key=t_idx_key,
            )

        torch.cuda.synchronize()
        self.cuda_graphs_max_t_registered = max_t
        log.info(f"[CUDA Graph Precapture] Done: {len(self.cuda_graphs)} frames captured.")

    def forward_seq(
        self,
        x_B_C_T_H_W: torch.Tensor,
        video_pos: VideoSeqPos,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        *,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache_cfg: Optional[KVCacheConfig] = None,
    ) -> torch.Tensor:
        """Forward a sequence chunk with KV caches and correct RoPE alignment.

        Accepts/returns per-chunk tensors shaped [B, C, T, H, W].
        """
        # FlexAttention-only behavior assumed by upstream utilities

        # Embed current chunk
        x_B_T_H_W_D, _, _ = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )
        B, T, H, W, D = x_B_T_H_W_D.shape
        assert T * H * W == video_pos.T * video_pos.H * video_pos.W, (
            f"Token length mismatch: {T * H * W} != {video_pos.T}*{video_pos.H}*{video_pos.W}"
        )

        # Optional context projection
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)
        context_input = crossattn_emb

        # Time embeddings (affine mods)
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # Compute RoPE for absolute positions in this chunk
        T_full = int(video_pos.pos_t.max().item()) + 1
        H_full = int(video_pos.pos_h.max().item()) + 1
        W_full = int(video_pos.pos_w.max().item()) + 1
        rope_full = self.pos_embedder.generate_embeddings(torch.Size([1, T_full, H_full, W_full, self.model_channels]))
        linear_idx = (
            video_pos.pos_t.to(dtype=torch.long) * (H_full * W_full)
            + video_pos.pos_h.to(dtype=torch.long) * W_full
            + video_pos.pos_w.to(dtype=torch.long)
        )
        rope_L_1_1_D = rope_full.index_select(0, linear_idx.to(device=rope_full.device))

        # Run blocks
        for block in self.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=None,
                kv_cache_cfg=kv_cache_cfg,
            )

        # Final head and unpatchify back to [B, C, T, H, W]
        x_B_T_H_W_O = self.final_layer(
            x_B_T_H_W_D,
            t_embedding_B_T_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
        )
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        return x_B_C_Tt_Hp_Wp


class CausalDITwithConditionalMask(CausalDIT):
    def __init__(self, *args, timestep_scale: float = 1.0, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # add one channel for the condition mask
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
        intermediate_feature_ids: Optional[list[int]] = None,
        **kwargs,
    ) -> Any:
        del kwargs
        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [
                    x_B_C_T_H_W,
                    torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device),
                ],
                dim=1,
            )

        return super().forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T * self.timestep_scale,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            intermediate_feature_ids=intermediate_feature_ids,
        )

    def forward_seq(
        self,
        x_B_C_T_H_W: torch.Tensor,
        video_pos: VideoSeqPos,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        *,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        kv_cache_cfg: Optional[KVCacheConfig] = None,
    ) -> torch.Tensor:
        if condition_video_input_mask_B_C_T_H_W is None:
            raise ValueError("condition_video_input_mask_B_C_T_H_W must be provided for conditional forward_seq")

        # Concat condition mask channel to inputs (match forward behavior)
        x_cat = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)

        # Delegate to base sequence API (embed inside base) with scaled timestep
        return super().forward_seq(
            x_B_C_T_H_W=x_cat,
            video_pos=video_pos,
            timesteps_B_T=timesteps_B_T * self.timestep_scale,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            kv_cache_cfg=kv_cache_cfg,
        )
