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

from typing import Any, Callable, List, Optional, Tuple, Type, cast

import torch
import torch.nn as nn
from einops import rearrange, repeat

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.graph import create_cuda_graph
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.interactive.networks.dit_causal import (
    AttenOpWithKV,
    CausalBlock,
    CausalDIT,
    KVContextConfig,
    VideoSeqPos,
)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer: Type[nn.Module] | Callable[[], nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Instantiate activation regardless of receiving a class or a factory callable
        self.activation = act_layer() if not isinstance(act_layer, nn.Module) else act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ActionChunkCausalDIT(CausalDIT):
    """
    Action-conditioned variant of CausalDIT.

    Mirrors the action-conditioning design in ActionChunkConditionedMinimalV1LVGDiT by:
    - projecting flattened per-latent-frame action chunks with two MLPs
    - adding the resulting embeddings into (t_embedding, adaln_lora) streams
    - optional timestep scaling for rectified-flow style schedules
    """

    def __init__(self, *args, timestep_scale: float = 1.0, **kwargs) -> None:
        # Action config
        action_dim = kwargs.pop("action_dim", 10 * 8)
        # Align number of actions aggregated per latent frame with temporal compression
        self._num_action_per_latent_frame = kwargs.pop("temporal_compression_ratio", 4)
        self._hidden_dim_in_action_embedder = kwargs.pop("hidden_dim_in_action_embedder", None)

        # Optional rescaling of timesteps (e.g., for rectified flow)
        self.timestep_scale = timestep_scale

        super().__init__(*args, **kwargs)

        if self._hidden_dim_in_action_embedder is None:
            self._hidden_dim_in_action_embedder = self.model_channels * 4

        log.info(f"hidden_dim_in_action_embedder: {self._hidden_dim_in_action_embedder}")

        # MLPs for action embeddings
        self.action_embedder_B_D = Mlp(
            in_features=action_dim * self._num_action_per_latent_frame,
            hidden_features=self._hidden_dim_in_action_embedder,
            out_features=self.model_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.action_embedder_B_3D = Mlp(
            in_features=action_dim * self._num_action_per_latent_frame,
            hidden_features=self._hidden_dim_in_action_embedder,
            out_features=self.model_channels * 3,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

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
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )

        # Prepare tokenized sequence and positional embeddings
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

        # Time embeddings (with optional scaling) and action conditioning
        timesteps_B_T = timesteps_B_T * self.timestep_scale

        with torch.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)

            # Inject action embeddings
            assert action is not None, "action must be provided"
            num_actions = action.shape[1]
            # Flatten all actions, then regroup per latent frame
            action_flat = rearrange(action, "b t d -> b 1 (t d)")
            # Derive the expected latent T for action embeddings
            t_action = num_actions // int(self._num_action_per_latent_frame)
            action_grouped = rearrange(action_flat, "b 1 (t d) -> b t d", t=t_action)
            action_emb_B_D = self.action_embedder_B_D(action_grouped)
            action_emb_B_3D = self.action_embedder_B_3D(action_grouped)

            # Best-effort alignment: if the model's T differs by +1, pad a leading zero (mirror chunk impl)
            assert action_emb_B_D.shape[1] + 1 == t_embedding_B_T_D.shape[1], (
                f"action_emb_B_D.shape[1] + 1 != t_embedding_B_T_D.shape[1]: {action_emb_B_D.shape[1] + 1} != {t_embedding_B_T_D.shape[1]}"
            )
            zero_pad_action_emb_B_D = torch.zeros_like(action_emb_B_D[:, :1, :], device=action_emb_B_D.device)
            zero_pad_action_emb_B_3D = torch.zeros_like(action_emb_B_3D[:, :1, :], device=action_emb_B_3D.device)
            action_emb_B_D = torch.cat([zero_pad_action_emb_B_D, action_emb_B_D], dim=1)
            action_emb_B_3D = torch.cat([zero_pad_action_emb_B_3D, action_emb_B_3D], dim=1)

            # If dimensions still mismatch, attempt broadcast-safe trim/pad to match time length
            # if action_emb_B_D.shape[1] != t_embedding_B_T_D.shape[1]:
            #     target_T = t_embedding_B_T_D.shape[1]
            #     cur_T = action_emb_B_D.shape[1]
            #     if cur_T > target_T:
            #         action_emb_B_D = action_emb_B_D[:, :target_T]
            #         action_emb_B_3D = action_emb_B_3D[:, :target_T]
            #     else:
            #         pad_D = target_T - cur_T
            #         pad0_D = torch.zeros_like(action_emb_B_D[:, :pad_D, :])
            #         pad0_3D = torch.zeros_like(action_emb_B_3D[:, :pad_D, :])
            #         action_emb_B_D = torch.cat([pad0_D, action_emb_B_D], dim=1)
            #         action_emb_B_3D = torch.cat([pad0_3D, action_emb_B_3D], dim=1)

            t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D
            if adaln_lora_B_T_3D is not None:
                adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D

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

        # Install temporal causal mask for self-attention
        if data_type == DataType.VIDEO:
            B, T, H, W, D = x_B_T_H_W_D.shape
            if self.atten_backend in ["torch-flex", "ulysses-flex"]:
                # Defer to base behavior: build a BlockMask once and reuse
                try:
                    from megatron.core import parallel_state as _ps  # type: ignore
                except Exception:
                    _ps = None  # type: ignore
                if _ps is not None and _ps.is_initialized():
                    seq_world_size = _ps.get_context_parallel_world_size()
                else:
                    seq_world_size = 1
                frame_seqlen = H * W
                num_frames_for_mask = T * seq_world_size
                from cosmos_predict2._src.predict2.interactive.networks.blockmask import (
                    build_blockwise_causal_mask_flex,
                )

                block_mask = build_blockwise_causal_mask_flex(
                    device=torch.device("cpu"),
                    num_frames=num_frames_for_mask,
                    frame_seqlen=frame_seqlen,
                    compile_mask=True,
                )
                for module in self.get_self_attn_module():
                    module.mask = block_mask.to(x_B_C_T_H_W.device)
            else:
                try:
                    from megatron.core import parallel_state as _ps  # type: ignore
                except Exception:
                    _ps = None  # type: ignore
                if _ps is not None and _ps.is_initialized():
                    seq_world_size = _ps.get_context_parallel_world_size()
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


class ActionChunkCausalDITKVCache(ActionChunkCausalDIT):
    """Action-conditioned Causal DiT with lightweight KV cache via attention-op wrapping.

    API mirrors CausalDITKVCache but injects action embeddings into the
    (t_embedding, adaln_lora) streams. `forward_seq` is intended to run on
    one frame per call (VideoSeqPos.T == 1).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_cuda_graphs = kwargs.get("use_cuda_graphs", False)
        self.cuda_graphs = None
        self.cuda_graphs_max_t_registered = -1

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
            block = cast(CausalBlock, block)
            attn_op_any: Any = block.self_attn.attn_op
            if hasattr(attn_op_any, "local_attn"):
                # Ulysses distributed attention wrapper
                if isinstance(attn_op_any.local_attn, AttenOpWithKV):
                    # Already wrapped, just reset the cache
                    attn_op_any.local_attn.reset_kv_cache(batch_size, seq_len, self.num_heads, head_dim, dtype, device)
                else:
                    # Not wrapped yet, create new wrapper
                    kv_op = AttenOpWithKV(attn_op_any.local_attn)
                    kv_op.reset_kv_cache(batch_size, seq_len, self.num_heads, head_dim, dtype, device)
                    attn_op_any.local_attn = kv_op
            else:
                # Direct op (torch/TE/flex). Wrap at the top-level.
                if isinstance(attn_op_any, AttenOpWithKV):
                    # Already wrapped, just reset the cache
                    attn_op_any.reset_kv_cache(batch_size, seq_len, self.num_heads, head_dim, dtype, device)
                else:
                    # Not wrapped yet, create new wrapper
                    kv_op = AttenOpWithKV(attn_op_any)
                    kv_op.reset_kv_cache(batch_size, seq_len, self.num_heads, head_dim, dtype, device)
                    block.self_attn.attn_op = kv_op

    def forward_seq(
        self,
        x_B_C_T_H_W: torch.Tensor,
        video_pos: VideoSeqPos,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        *,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        kv_context_cfg: Optional[KVContextConfig] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward a single-frame sequence chunk with KV caches and action conditioning.

        Args:
            x_B_C_T_H_W: [B, C, T, H, W] sequence features for the current chunk.
            fps: [B, 1] fps tensor.
            padding_mask: [H, W] padding mask tensor.
            video_pos: VideoSeqPos describing global absolute positions for this chunk.
            timesteps_B_T: [B, 1] timesteps for this step.
            crossattn_emb: [B, N, D_ctx] cross-attention embeddings.
            kv_context_cfg: KV context (start index, run/store flags).
            img_context_emb: Optional image context for i2v variants.
            action: Optional per-frame actions shaped [B, T_act, D_act]. If provided,
                T_act should equal self._num_action_per_latent_frame for per-frame grouping.
        Returns:
            [B, L, O] token outputs after the model head (O = patch_prod * out_channels)
        """
        # Embed current frame to tokens
        x_1f_T_H_W_D, _, _ = self.prepare_embedded_sequence(x_B_C_T_H_W, fps=fps, padding_mask=padding_mask)
        B, T1, H, W, D = x_1f_T_H_W_D.shape
        assert T1 == 1, "forward_seq expects a single frame (T=1)"
        x_B_L_D = x_1f_T_H_W_D.reshape(B, H * W, D)

        # Scale timestep if requested
        timesteps_B_T = timesteps_B_T * self.timestep_scale

        assert x_B_L_D.shape[1] == video_pos.T * video_pos.H * video_pos.W, (
            f"Token length mismatch: {x_B_L_D.shape[1]} != {video_pos.T}*{video_pos.H}*{video_pos.W}"
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

        # Time embeddings
        with torch.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)

            # Inject per-frame action embeddings if provided
            if action is not None:
                # Expect action as [B, T_act, D_act] (typically T_act == num_action_per_latent_frame)
                # Flatten along time and project with MLPs, keeping a time dim of 1
                action_flat = rearrange(action, "b t d -> b 1 (t d)")
                # Ensure last dim matches expected in_features for the MLPs by zero-padding/trimming
                expected_in = self.action_embedder_B_D.fc1.in_features
                cur_in = action_flat.shape[-1]
                assert cur_in == expected_in, f"cur_in != expected_in: {cur_in} != {expected_in}"
                # if cur_in != expected_in:
                #     if cur_in > expected_in:
                #         action_flat = action_flat[..., :expected_in]
                #     else:
                #         pad = expected_in - cur_in
                #         pad_zeros = torch.zeros(
                #             action_flat.shape[0], 1, pad, device=action_flat.device, dtype=action_flat.dtype
                #         )
                #         action_flat = torch.cat([pad_zeros, action_flat], dim=-1)

                action_emb_B_1_D = self.action_embedder_B_D(action_flat)
                action_emb_B_1_3D = self.action_embedder_B_3D(action_flat)
                t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_1_D
                if adaln_lora_B_T_3D is not None:
                    adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_1_3D

            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # Build RoPE for absolute positions of this chunk
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

        # Reshape sequence back to 5D for block API
        x_B_T_H_W_D = rearrange(x_B_L_D, "b (t h w) d -> b t h w d", t=video_pos.T, h=video_pos.H, w=video_pos.W)

        # Install/update KV context on all wrapped self-attention ops
        for block in self.blocks:
            block = cast(CausalBlock, block)
            attn_op_any: Any = block.self_attn.attn_op
            if hasattr(attn_op_any, "local_attn") and isinstance(attn_op_any.local_attn, AttenOpWithKV):
                attn_op_any.local_attn.set_kv_context(kv_context_cfg)
            elif isinstance(attn_op_any, AttenOpWithKV):
                attn_op_any.set_kv_context(kv_context_cfg)

        # Clear masks when temporal-causal mode not explicitly enabled
        if not getattr(self, "_temporal_causal_enabled", False):
            for module in self.get_self_attn_module():
                module.mask = None

        block_kwargs = {
            "emb_B_T_D": t_embedding_B_T_D,
            "crossattn_emb": context_input,
            "rope_emb_L_1_1_D": rope_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": None,
        }
        if self.use_cuda_graphs:
            assert self.cuda_graphs is not None, "CUDA graphs not pre-captured, call precapture_cuda_graphs first"
            t_idx = T_full - 1
            t_idx_key = f"t{t_idx}"
            # Should not create here, just return the key, should create during precapture_cuda_graphs
            shapes_key = create_cuda_graph(
                self.cuda_graphs,
                self.blocks,
                [x_B_T_H_W_D],
                block_kwargs,
                extra_key=t_idx_key,
            )
            cg_blocks = self.cuda_graphs[shapes_key]
        else:
            cg_blocks = self.blocks

        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = cg_blocks[i](
                x_B_T_H_W_D,
                **block_kwargs,
            )

        # Final head then flatten back to [B, L, O]
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        return x_B_C_Tt_Hp_Wp.type_as(x_B_C_T_H_W)

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
            and isinstance(self.crossattn_proj, nn.Sequential)
            and len(self.crossattn_proj) > 0
        ):
            proj0 = self.crossattn_proj[0]
            D_ctx = proj0.out_features if isinstance(proj0, nn.Linear) else self.model_channels
        else:
            D_ctx = self.model_channels

        log.info(f"[CUDA Graph Precapture] Capturing graphs for {max_t} frames")

        for t_idx in range(max_t):
            x_dummy = torch.randn(batch_size, 1, token_h, token_w, self.model_channels, device=device, dtype=dtype)
            emb_dummy = torch.randn(batch_size, 1, self.model_channels, device=device, dtype=dtype)
            rope_dummy = torch.randn(token_h * token_w, 1, 1, head_dim, device=device)
            crossattn_dummy = torch.randn(batch_size, N_ctx, D_ctx, device=device, dtype=dtype)
            adaln_dummy = torch.randn(batch_size, 1, 3 * self.model_channels, device=device, dtype=dtype)

            kv_context_cfg = KVContextConfig(
                start_idx=t_idx * token_h * token_w,
                run_with_kv=True,
                store_kv=True,
            )

            for block in self.blocks:
                block = cast(CausalBlock, block)
                attn_op_any: Any = block.self_attn.attn_op
                if hasattr(attn_op_any, "local_attn") and isinstance(attn_op_any.local_attn, AttenOpWithKV):
                    attn_op_any.local_attn.set_kv_context(kv_context_cfg)
                elif isinstance(attn_op_any, AttenOpWithKV):
                    attn_op_any.set_kv_context(kv_context_cfg)

            block_kwargs = {
                "emb_B_T_D": emb_dummy,
                "crossattn_emb": crossattn_dummy,
                "rope_emb_L_1_1_D": rope_dummy,
                "adaln_lora_B_T_3D": adaln_dummy,
                "extra_per_block_pos_emb": None,
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


class ActionChunkCausalDITwithConditionalMaskKVCache(ActionChunkCausalDITKVCache):
    def __init__(self, *args, **kwargs) -> None:
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # add 1 channel for condition mask
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
        action: Optional[torch.Tensor] = None,
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
            timesteps_B_T=timesteps_B_T,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            intermediate_feature_ids=intermediate_feature_ids,
            img_context_emb=img_context_emb,
            action=action,
        )

    def forward_seq(
        self,
        x_B_C_T_H_W: torch.Tensor,
        video_pos: "VideoSeqPos",
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        *,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        kv_context_cfg: Optional["KVContextConfig"] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-frame forward using KV cache with conditional mask.

        Args:
            x_B_C_T_H_W: [B, C, 1, H, W] current frame input.
            fps: [B, 1] fps tensor.
            padding_mask: [H, W] padding mask tensor.
            condition_video_input_mask_B_C_T_H_W: [B, 1, 1, H, W] conditional mask.
            video_pos: absolute positions for this chunk.
            timesteps_B_T: [B, 1] timestep tensor.
            crossattn_emb: [B, N, D_ctx] context.
            kv_context_cfg: KV run/store settings.
            img_context_emb: optional image context.
            action: per-frame action tensor [B, T_act, D_act].
        Returns:
            [B, L, O] token outputs for this frame.
        """
        # Concat condition mask channel
        if x_B_C_T_H_W.ndim != 5:
            raise ValueError("x_B_C_T_H_W must be 5D [B,C,T,H,W]")

        x_cat = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)

        # Delegate to parent per-frame sequence API
        return super().forward_seq(
            x_B_C_T_H_W=x_cat,
            fps=fps,
            padding_mask=padding_mask,
            video_pos=video_pos,
            timesteps_B_T=timesteps_B_T,
            crossattn_emb=crossattn_emb,
            kv_context_cfg=kv_context_cfg,
            img_context_emb=img_context_emb,
            action=action,
        )
