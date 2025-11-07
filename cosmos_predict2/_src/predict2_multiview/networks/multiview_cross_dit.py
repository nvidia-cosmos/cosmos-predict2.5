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

from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed._composable.fsdp import fully_shard
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
from torchvision import transforms

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
    Attention,
    Block,
    SACConfig,
)
from cosmos_predict2._src.predict2_multiview.networks.multiview_dit import (
    MultiCameraSinCosPosEmbAxis,
    MultiCameraVideoRopePosition3DEmb,
)


# implementation of MultiViewCrossAttention changes multiview_dit.py
class MultiViewCrossAttention(Attention):
    def __init__(self, *args, state_t: int = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.qkv_format == "bshd", "MultiViewCrossAttention only supports qkv_format='bshd'"
        self.state_t = state_t

    def forward(self, x, context=None, rope_emb=None):
        assert not self.is_selfattn, "MultiViewCrossAttention does not support self-attention"
        B, L, D = x.shape

        n_cameras = context.shape[1] // 512
        x_B_L_D = rearrange(x, "B (V L) D -> (V B) L D", V=n_cameras)
        context_B_M_D = rearrange(context, "B (V M) D -> (V B) M D", V=n_cameras) if context is not None else None
        x_B_L_D = super().forward(x_B_L_D, context_B_M_D, rope_emb=rope_emb)
        x_B_L_D = rearrange(x_B_L_D, "(V B) L D -> B (V L) D", V=n_cameras)
        return x_B_L_D


# ---------------------- Selective Activation Checkpoint Policies -----------------------
def predict2_2B_crossview_720_context_fn():
    op_count = defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"

        if func == torch.ops.aten.mm.default:
            op_count_key = f"{mode}_mm_count"
            if op_count[op_count_key] >= 10:
                result = CheckpointPolicy.MUST_SAVE
            else:
                result = CheckpointPolicy.PREFER_RECOMPUTE

            # Update count for next operation
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 20
            return result

        if "flash_attn" in str(func):
            return CheckpointPolicy.MUST_SAVE

        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


class MultiViewCheckpointMode(str, Enum):
    """Checkpoint modes for MultiViewCrossDiT architecture."""

    NONE = "none"
    MM_ONLY = "mm_only"
    BLOCK_WISE = "block_wise"
    PREDICT2_2B_CROSSVIEW_720 = "predict2_2b_crossview_720"

    def __str__(self) -> str:
        return self.value


@dataclass
class MultiViewSACConfig(SACConfig):
    """Selective Activation Checkpoint Config for MultiViewCrossDiT."""

    def get_context_fn(self):
        if self.mode == MultiViewCheckpointMode.PREDICT2_2B_CROSSVIEW_720:
            return predict2_2B_crossview_720_context_fn
        else:
            return super().get_context_fn()


VideoSize = namedtuple("VideoSize", ["T", "H", "W"])


class CrossViewAttention(Attention):
    def __init__(self, *args, cross_view_attn_map: Dict[int, List[int]], **kwargs):
        super().__init__(*args, **kwargs)
        del self.attn_op
        if self.backend == "transformer_engine":
            from transformer_engine.pytorch.attention import DotProductAttention

            self.attn_op = DotProductAttention(
                self.n_heads,
                self.head_dim,
                num_gqa_groups=self.n_heads,
                attention_dropout=0,
                qkv_format=self.qkv_format,
                attn_mask_type="padding",  # important
                attention_type="cross",  # important
            )
        else:
            raise NotImplementedError(f"Backend {self.backend} not supported")
        self.cross_view_attn_map = cross_view_attn_map
        self.max_neighbors = max(len(neighbors) for neighbors in cross_view_attn_map.values())
        self.neighbor_indices = None
        self.neighbor_mask = None

    def forward(self, x, view_indices_B_V, sv_video_size: VideoSize):
        """
        x: (B, V, L, D)
        view_indices_B_V: (B, V)
        sv_video_size: VideoSize (T, H, W), where T * H * W = L
        """
        assert not self.is_selfattn, "CrossViewAttention does not support self-attention"
        B, V, L, D = x.shape
        T, H, W = sv_video_size
        assert T * H * W == L, f"T * H * W != L: {T * H * W} != {L}"

        # move time dimension to batch dimension
        x = rearrange(x, "b v (t h w) d -> (b t) v (h w) d", t=T, h=H, w=W)
        B, V, L, D = x.shape

        view_indices_B_V = view_indices_B_V.repeat_interleave(T, dim=0).long()

        # Create neighbor indices and mask on the fly, only once.
        if self.neighbor_indices is None or self.neighbor_indices.device != x.device:
            num_total_views = len(self.cross_view_attn_map)
            neighbor_indices = torch.zeros((num_total_views, self.max_neighbors), dtype=torch.long, device=x.device)
            neighbor_mask = torch.zeros((num_total_views, self.max_neighbors), dtype=torch.bool, device=x.device)
            for i in range(num_total_views):
                neighbors = self.cross_view_attn_map[i]
                for j, neighbor_idx in enumerate(neighbors):
                    neighbor_indices[i, j] = neighbor_idx
                    neighbor_mask[i, j] = True
            self.neighbor_indices = neighbor_indices
            self.neighbor_mask = neighbor_mask

        num_total_views = len(self.cross_view_attn_map)
        view_indices_to_tensor_pos = torch.full(
            (B, num_total_views), -1, dtype=torch.long, device=x.device
        )  # include out of range view index
        b_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, V).long()
        view_indices_to_tensor_pos[b_indices, view_indices_B_V] = (
            torch.arange(V, device=x.device).unsqueeze(0).expand(B, -1)
        )

        neighbor_view_indices = self.neighbor_indices[view_indices_B_V]  # may include out of range view index
        gather_tensor_pos = view_indices_to_tensor_pos[
            b_indices.unsqueeze(2), neighbor_view_indices
        ]  # [B, V, max_neighbors], out of range view index will be -1

        # Sort to move all -1 to the end, which is convenient for creating attention mask.
        gather_tensor_pos, sorted_indices = torch.sort(gather_tensor_pos, dim=-1, descending=True)

        b_indices_for_gather = torch.arange(B, device=x.device)[:, None, None]
        # Clamp to avoid index error. Masked values will be ignored in attention.
        neighbor_features = x[
            b_indices_for_gather, torch.clamp(gather_tensor_pos, min=0)
        ]  # [B, V, max_neighbors, L, C]

        # Prepare for attention
        query = self.q_proj(rearrange(x, "b v l c -> (b v) l c"))  # [B*V, L, C]
        context = rearrange(neighbor_features, "b v n l c -> (b v) (n l) c")  # [B*V, max_neighbors*L, C]
        key = self.k_proj(context)
        value = self.v_proj(context)

        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (query, key, value),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Create attention mask
        is_neighbor_present = gather_tensor_pos != -1  # [B, V, max_neighbors]
        mask_for_input_views = self.neighbor_mask[view_indices_B_V]  # [B, V, n]

        # Reorder mask_for_input_views to match the sorted gather_tensor_pos
        mask_for_input_views = torch.gather(mask_for_input_views, -1, sorted_indices)
        final_mask = is_neighbor_present & mask_for_input_views

        mask_per_view = rearrange(final_mask, "b v n -> (b v) n")  # [BV, n]
        mask_kv = mask_per_view.repeat_interleave(L, dim=1)  # [BV, n*L]

        # Reshape mask to [batch_size, 1, 1, max_seqlen_kv] as per official documentation.
        mask = rearrange(mask_kv, "bv l_kv -> bv 1 1 l_kv")  # [BV, 1, 1, n*L]
        atten_mask_kv = ~mask  # 0 means keep, 1 means mask
        atten_mask_q = torch.zeros(query.shape[0], 1, 1, query.shape[1]).to(atten_mask_kv)

        attention_output = self.attn_op(q, k, v, attention_mask=(atten_mask_q, atten_mask_kv))
        attention_output = attention_output.flatten(2)  # [B*V, L, H*D]
        output = self.output_dropout(self.output_proj(attention_output))
        output = rearrange(output, "(b v) l d -> b v l d", v=V)
        # recover time dimension from batch to seq
        output = rearrange(output, "(b t) v (h w) d -> b v (t h w) d", t=T, h=H, w=W)
        return output

    def set_context_parallel_group(self, process_group, ranks, stream):
        raise NotImplementedError("Cross View Attention doesn't need communication")


class MultiViewCrossBlock(Block):
    """
    A transformer block that takes n_cameras as input.
    Self-Attention (Single View) -> Cross-View Attention -> Cross Attention (text and image)
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        cross_view_attn_map: Dict[int, List[int]] = None,
        state_t: int = None,
        backend: str = "transformer_engine",
        image_context_dim: Optional[int] = None,
        use_wan_fp32_strategy: bool = False,
        enable_cross_view_attn: bool = False,
    ):
        super().__init__(
            x_dim,
            context_dim,
            num_heads,
            mlp_ratio,
            use_adaln_lora,
            adaln_lora_dim,
            backend,
            image_context_dim,
            use_wan_fp32_strategy,
        )
        self.state_t = state_t
        self.cross_view_attn_map = cross_view_attn_map
        self.enable_cross_view_attn = enable_cross_view_attn
        if image_context_dim is None:
            del self.cross_attn
            # cross attention to text and image condition
            self.cross_attn = MultiViewCrossAttention(
                x_dim,
                context_dim,
                num_heads,
                x_dim // num_heads,
                qkv_format="bshd",
                state_t=state_t,
                use_wan_fp32_strategy=use_wan_fp32_strategy,
            )
        else:
            raise NotImplementedError("image_context_dim is not supported for MultiViewBlock")

        if enable_cross_view_attn:
            self.cross_view_attn = CrossViewAttention(
                x_dim,
                x_dim,  # context_dim, can not set to None
                num_heads,
                x_dim // num_heads,
                qkv_format="bshd",
                use_wan_fp32_strategy=use_wan_fp32_strategy,
                cross_view_attn_map=cross_view_attn_map,
            )
            # no modulation so we set elementwise_affine=True
            self.layer_norm_cross_view_attn = nn.LayerNorm(x_dim, elementwise_affine=True, eps=1e-6)

    def reset_parameters(self):
        super().reset_parameters()
        if self.enable_cross_view_attn:
            self.layer_norm_cross_view_attn.reset_parameters()

    def init_weights(self):
        super().init_weights()
        if self.enable_cross_view_attn:
            self.cross_view_attn.init_weights()

            # Zero-initialize the output projection
            torch.nn.init.zeros_(self.cross_view_attn.output_proj.weight)
            if self.cross_view_attn.output_proj.bias is not None:
                torch.nn.init.zeros_(self.cross_view_attn.output_proj.bias)

    # most code are copied from parent. insert cross view attn between self attn and cross attn
    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        view_indices_B_T: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        view_embedding_proj_B_V_9D: Optional[torch.Tensor] = None,
        crossattn_emb: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        block_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        x_B_T_H_W_D: (B, T, H, W, D)
        view_indices_B_T: (B, T)
        """
        num_cameras = torch.unique(view_indices_B_T[0]).shape[0]

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

        if view_embedding_proj_B_V_9D is not None:
            (
                view_shift_self_attn_B_V_D,
                view_scale_self_attn_B_V_D,
                view_gate_self_attn_B_V_D,
                view_shift_cross_attn_B_V_D,
                view_scale_cross_attn_B_V_D,
                view_gate_cross_attn_B_V_D,
                view_shift_mlp_B_V_D,
                view_scale_mlp_B_V_D,
                view_gate_mlp_B_V_D,
            ) = view_embedding_proj_B_V_9D.chunk(9, dim=-1)

            repeat_t = x_B_T_H_W_D.shape[1] // num_cameras
            assert repeat_t * num_cameras == x_B_T_H_W_D.shape[1]

            def expand_and_rearrange(x_B_V_D):
                expanded_x_B_V_T_D = x_B_V_D.unsqueeze(2).expand(-1, -1, repeat_t, -1)
                return rearrange(expanded_x_B_V_T_D, "b v t d -> b (v t) 1 1 d")

            shift_self_attn_B_T_1_1_D = shift_self_attn_B_T_1_1_D + expand_and_rearrange(
                view_shift_self_attn_B_V_D
            ).type_as(x_B_T_H_W_D)
            scale_self_attn_B_T_1_1_D = scale_self_attn_B_T_1_1_D + expand_and_rearrange(
                view_scale_self_attn_B_V_D
            ).type_as(x_B_T_H_W_D)
            gate_self_attn_B_T_1_1_D = gate_self_attn_B_T_1_1_D + expand_and_rearrange(
                view_gate_self_attn_B_V_D
            ).type_as(x_B_T_H_W_D)
            shift_cross_attn_B_T_1_1_D = shift_cross_attn_B_T_1_1_D + expand_and_rearrange(
                view_shift_cross_attn_B_V_D
            ).type_as(x_B_T_H_W_D)
            scale_cross_attn_B_T_1_1_D = scale_cross_attn_B_T_1_1_D + expand_and_rearrange(
                view_scale_cross_attn_B_V_D
            ).type_as(x_B_T_H_W_D)
            gate_cross_attn_B_T_1_1_D = gate_cross_attn_B_T_1_1_D + expand_and_rearrange(
                view_gate_cross_attn_B_V_D
            ).type_as(x_B_T_H_W_D)
            shift_mlp_B_T_1_1_D = shift_mlp_B_T_1_1_D + expand_and_rearrange(view_shift_mlp_B_V_D).type_as(x_B_T_H_W_D)
            scale_mlp_B_T_1_1_D = scale_mlp_B_T_1_1_D + expand_and_rearrange(view_scale_mlp_B_V_D).type_as(x_B_T_H_W_D)
            gate_mlp_B_T_1_1_D = gate_mlp_B_T_1_1_D + expand_and_rearrange(view_gate_mlp_B_V_D).type_as(x_B_T_H_W_D)

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )

        rope_emb_L_1_1_D_sv = rearrange(
            rope_emb_L_1_1_D,
            "(v m) 1 1 d -> v m 1 1 d",
            v=num_cameras,
        )[0]

        result_B_T_H_W_D = rearrange(
            self.self_attn(
                rearrange(normalized_x_B_T_H_W_D, "b (v t) h w d -> (b v) (t h w) d", v=num_cameras),
                None,
                rope_emb=rope_emb_L_1_1_D_sv,
            ),
            "(b v) (t h w) d -> b (v t) h w d",
            v=num_cameras,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

        # insert cross view attn here. x_B_T_H_W_D
        if self.enable_cross_view_attn:
            num_cameras = torch.unique(view_indices_B_T[0]).shape[0]
            x_B_V_T_H_W_D = rearrange(x_B_T_H_W_D, "b (v t) h w d -> b v t h w d", v=num_cameras)
            sv_video_size = VideoSize(T=x_B_V_T_H_W_D.shape[2], H=x_B_V_T_H_W_D.shape[3], W=x_B_V_T_H_W_D.shape[4])
            x_B_V_L_D = rearrange(x_B_V_T_H_W_D, "b v t h w d -> b v (t h w) d")
            view_indices_B_V = rearrange(view_indices_B_T, "b (v t) -> b v t", v=num_cameras)[..., 0]
            result_cross_view_attn_B_T_H_W_D = rearrange(
                self.cross_view_attn(self.layer_norm_cross_view_attn(x_B_V_L_D), view_indices_B_V, sv_video_size),
                "b v (t h w) d -> b (v t) h w d",
                v=num_cameras,
                t=sv_video_size.T,
                h=sv_video_size.H,
                w=sv_video_size.W,
            )
            x_B_T_H_W_D = x_B_T_H_W_D + result_cross_view_attn_B_T_H_W_D

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


class MultiViewCrossDiT(MinimalV1LVGDiT):
    def __init__(
        self,
        *args,
        timestep_scale: float = 1.0,
        crossattn_emb_channels: int = 1024,
        mlp_ratio: float = 4.0,
        state_t: int,
        n_cameras_emb: int,
        view_condition_dim: int,
        concat_view_embedding: bool,
        adaln_view_embedding: bool,
        layer_mask: Optional[List[bool]] = None,
        sac_config: MultiViewSACConfig = MultiViewSACConfig(),
        enable_cross_view_attn: bool = False,
        cross_view_attn_map_str: Optional[Dict] = None,
        camera_to_view_id: Optional[Dict] = None,
        **kwargs,
    ):
        self.crossattn_emb_channels = crossattn_emb_channels
        self.mlp_ratio = mlp_ratio
        self.state_t = state_t
        self.n_cameras_emb = n_cameras_emb
        self.view_condition_dim = view_condition_dim
        self.concat_view_embedding = concat_view_embedding
        self.adaln_view_embedding = adaln_view_embedding
        self.enable_cross_view_attn = enable_cross_view_attn

        assert not (self.adaln_view_embedding and self.concat_view_embedding), (
            "adaln_view_embedding and concat_view_embedding cannot be True at the same time"
        )
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += (
            self.view_condition_dim if self.concat_view_embedding else 0
        )  # this avoids overwritting build_patch_embed which still adds padding_mask channel as appropriate
        assert layer_mask is None, "layer_mask is not supported for MultiViewDiT"
        if "n_cameras" in kwargs:
            del kwargs["n_cameras"]
        super().__init__(
            *args,
            mlp_ratio=mlp_ratio,
            timestep_scale=timestep_scale,
            crossattn_emb_channels=crossattn_emb_channels,
            sac_config=sac_config,
            **kwargs,
        )

        cross_view_attn_map = {}
        for source_view, target_views in cross_view_attn_map_str.items():
            cross_view_attn_map[int(camera_to_view_id[source_view])] = []
            for target_view in target_views:
                cross_view_attn_map[int(camera_to_view_id[source_view])].append(int(camera_to_view_id[target_view]))
        self.cross_view_attn_map = cross_view_attn_map

        del self.blocks
        self.blocks = nn.ModuleList(
            [
                MultiViewCrossBlock(
                    x_dim=self.model_channels,
                    context_dim=self.crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    backend=self.atten_backend,
                    image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                    state_t=self.state_t,
                    use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                    cross_view_attn_map=self.cross_view_attn_map,
                    enable_cross_view_attn=self.enable_cross_view_attn,
                )
                for _ in range(self.num_blocks)
            ]
        )

        if self.concat_view_embedding:
            self.view_embeddings = nn.Embedding(self.n_cameras_emb, view_condition_dim)

        if self.adaln_view_embedding:
            self.adaln_view_embedder = nn.Embedding(self.n_cameras_emb, self.model_channels)
            # cosmos use adaln in self-attn, cross-attn, mlp
            self.adaln_view_proj = nn.Linear(self.model_channels, self.model_channels * 9)

        self.init_weights()
        self.enable_selective_checkpoint(sac_config, self.blocks)

    def fully_shard(self, mesh):
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward)

        fully_shard(self.final_layer, mesh=mesh, reshard_after_forward=True)
        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                fully_shard(extra_pos_embedder, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=False)
        if self.extra_image_context_dim is not None:
            fully_shard(self.img_context_proj, mesh=mesh, reshard_after_forward=False)

        if hasattr(self, "view_embeddings"):
            fully_shard(self.view_embeddings, mesh=mesh, reshard_after_forward=False)

        if hasattr(self, "adaln_view_embedder"):
            fully_shard(self.adaln_view_embedder, mesh=mesh, reshard_after_forward=False)
        if hasattr(self, "adaln_view_proj"):
            fully_shard(self.adaln_view_proj, mesh=mesh, reshard_after_forward=False)

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None):
        # pos_embedder
        for pos_embedder in self.pos_embedder_options.values():
            pos_embedder.enable_context_parallel(process_group=process_group)
        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                extra_pos_embedder.enable_context_parallel(process_group=process_group)

        # attention
        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=process_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = True

    def disable_context_parallel(self):
        # pos_embedder
        for pos_embedder in self.pos_embedder_options.values():
            pos_embedder.disable_context_parallel()
        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                extra_pos_embedder.disable_context_parallel()

        # attention
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = False

    def init_weights(self):
        self.x_embedder.init_weights()
        for pos_embedder in self.pos_embedder_options.values():
            pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                extra_pos_embedder.init_weights()

        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

        if self.extra_image_context_dim is not None:
            self.img_context_proj[0].reset_parameters()

        if hasattr(self, "view_embeddings"):
            torch.nn.init.normal_(self.view_embeddings.weight, mean=0.0, std=0.02)

        if hasattr(self, "adaln_view_embedder"):
            torch.nn.init.normal_(self.adaln_view_embedder.weight, mean=0.0, std=0.05)

        if hasattr(self, "adaln_view_proj"):
            torch.nn.init.zeros_(self.adaln_view_proj.weight)
            torch.nn.init.zeros_(self.adaln_view_proj.bias)

    def build_pos_embed(self):
        self.pos_embedder_options = nn.ModuleDict()
        self.extra_pos_embedders_options = nn.ModuleDict()
        for n_cameras in range(1, self.n_cameras_emb + 1):
            pos_embedder, extra_pos_embedder = self.build_pos_embed_for_n_cameras(n_cameras)
            self.pos_embedder_options[f"n_cameras_{n_cameras}"] = pos_embedder
            self.extra_pos_embedders_options[f"n_cameras_{n_cameras}"] = extra_pos_embedder

    def build_pos_embed_for_n_cameras(self, n_cameras: int):
        if self.pos_emb_cls == "rope3d":
            cls_type = MultiCameraVideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")
        pos_embedder, extra_pos_embedder = None, None
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
            n_cameras=n_cameras,
        )
        pos_embedder = cls_type(
            **kwargs,
        )
        assert pos_embedder.enable_fps_modulation == self.rope_enable_fps_modulation, (
            "enable_fps_modulation must be the same"
        )

        if self.extra_per_block_abs_pos_emb:
            raise NotImplementedError("extra_per_block_abs_pos_emb is not tested for multi-view DIT")
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            extra_pos_embedder = MultiCameraSinCosPosEmbAxis(
                **kwargs,
            )
        return pos_embedder, extra_pos_embedder

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        cp_size = (
            len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
            if parallel_state.is_initialized()
            else 1
        )
        n_cameras = (x_B_C_T_H_W.shape[2] * cp_size) // self.state_t
        pos_embedder = self.pos_embedder_options[f"n_cameras_{n_cameras}"]  # they are all the same if they are rope
        if self.concat_view_embedding:
            if view_indices_B_T is None:
                view_indices = torch.arange(n_cameras).clamp(
                    max=self.n_cameras_emb - 1
                )  # View indices [0, 1, ..., V-1]
                view_indices = view_indices.to(x_B_C_T_H_W.device)
                view_embedding = self.view_embeddings(view_indices)  # Shape: [V, embedding_dim]
                view_embedding = rearrange(view_embedding, "V D -> D V")
                view_embedding = (
                    view_embedding.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)
                )  # Shape: [1, D, V, 1, 1, 1]
            else:
                view_indices_B_T = view_indices_B_T.clamp(max=self.n_cameras_emb - 1)
                view_indices_B_T = view_indices_B_T.to(x_B_C_T_H_W.device).long()
                view_embedding = self.view_embeddings(view_indices_B_T)  # B, (V T), D
                view_embedding = rearrange(view_embedding, "B (V T) D -> B D V T", V=n_cameras)
                view_embedding = view_embedding.unsqueeze(-1).unsqueeze(-1)  # Shape: [B, D, V, T, 1, 1]
            x_B_C_V_T_H_W = rearrange(x_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=n_cameras)
            view_embedding = view_embedding.expand(
                x_B_C_V_T_H_W.shape[0],
                view_embedding.shape[1],
                view_embedding.shape[2],
                x_B_C_V_T_H_W.shape[3],
                x_B_C_V_T_H_W.shape[4],
                x_B_C_V_T_H_W.shape[5],
            )
            x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)
            x_B_C_T_H_W = rearrange(x_B_C_V_T_H_W, " B C V T H W -> B C (V T) H W", V=n_cameras)

        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_embedder = self.extra_pos_embedders_options[str(n_cameras)]
            extra_pos_emb = extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb

        if "fps_aware" in self.pos_emb_cls:
            raise NotImplementedError("FPS-aware positional embedding is not supported for multi-view DIT")

        x_B_T_H_W_D = x_B_T_H_W_D + pos_embedder(x_B_T_H_W_D)

        return x_B_T_H_W_D, None, extra_pos_emb

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        view_indices_B_T: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        # Deletes elements like condition.use_video_condition that are not used in the forward pass
        del kwargs
        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )

        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )
        timesteps_B_T = timesteps_B_T * self.timestep_scale
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            view_indices_B_T=view_indices_B_T,
        )
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            # (B, 1). input timesteps are (b, 1)
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        if self.adaln_view_embedding:
            num_cameras = torch.unique(view_indices_B_T[0]).shape[0]
            with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
                view_indices_B_V_T = rearrange(view_indices_B_T, "b (v t) -> b v t", v=num_cameras)
                view_embedding_B_V = self.adaln_view_embedder(view_indices_B_V_T[..., 0])  # B, V, D
                view_embedding_proj_B_V_9D = self.adaln_view_proj(view_embedding_B_V)  # B, V, 9D
        else:
            view_embedding_proj_B_V_9D = None

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

        for block_idx, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                view_indices_B_T,
                t_embedding_B_T_D,
                view_embedding_proj_B_V_9D,
                crossattn_emb,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                block_idx=block_idx,
            )

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)

        return x_B_C_Tt_Hp_Wp
