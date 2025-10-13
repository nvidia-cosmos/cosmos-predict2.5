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

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
from einops import rearrange, repeat
from megatron.core import parallel_state
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed._composable.fsdp import fully_shard
from torchvision import transforms

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
    Attention,
    Block,
    SACConfig,
    VideoPositionEmb,
    VideoRopePosition3DEmb,
)
from cosmos_predict2._src.predict2.utils.context_parallel import split_inputs_cp


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


class MultiViewBlock(Block):
    """
    A transformer block that takes n_cameras as input. This block
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        state_t: int = None,
        backend: str = "transformer_engine",
        image_context_dim: Optional[int] = None,
        use_wan_fp32_strategy: bool = False,
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
        if image_context_dim is None:
            del self.cross_attn
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


class MultiCameraVideoRopePosition3DEmb(VideoRopePosition3DEmb):
    def __init__(self, *args, n_cameras: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_cameras = n_cameras

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        B, T, H, W, C = B_T_H_W_C
        single_camera_B_T_H_W_C = (B, T // self.n_cameras, H, W, C)
        em_T_H_W_D = []
        for _ in range(self.n_cameras):
            em_L_1_1_D = super().generate_embeddings(
                single_camera_B_T_H_W_C,
                fps=fps,
                h_ntk_factor=h_ntk_factor,
                w_ntk_factor=w_ntk_factor,
                t_ntk_factor=t_ntk_factor,
            )
            em_T_H_W_D.append(rearrange(em_L_1_1_D, "(t h w) 1 1 d -> t h w d", t=T // self.n_cameras, h=H, w=W))
        em_T_H_W_D = torch.cat(em_T_H_W_D, dim=0)
        return em_T_H_W_D.float()

    @property
    def seq_dim(self):
        return 1

    def _split_for_context_parallel(self, embeddings):
        if self._cp_group is not None:
            embeddings = rearrange(embeddings, "(V T) H W D -> V (T H W) 1 1 D", V=self.n_cameras)
            embeddings = split_inputs_cp(x=embeddings, seq_dim=self.seq_dim, cp_group=self._cp_group)
            embeddings = rearrange(embeddings, "V T 1 1 D -> (V T) 1 1 D", V=self.n_cameras)
        else:
            embeddings = rearrange(embeddings, "t h w d -> (t h w) 1 1 d")
        return embeddings


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MultiCameraSinCosPosEmbAxis(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        n_cameras: int = 4,
        **kwargs,
    ):
        """
        Args:
            interpolation (str): we curretly only support "crop", ideally when we need extrapolation capacity, we should adjust frequency or other more advanced methods. they are not implemented yet.
        """
        del kwargs  # unused
        self.n_cameras = n_cameras
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"
        self.model_channels = model_channels
        self.len_h = len_h
        self.len_w = len_w
        self.len_t = len_t
        self.h_extrapolation_ratio = h_extrapolation_ratio
        self.w_extrapolation_ratio = w_extrapolation_ratio
        self.t_extrapolation_ratio = t_extrapolation_ratio

        emb_h, emb_w, emb_t = self.compute_1d_pos_embeddings()

        self.register_buffer("pos_emb_h", emb_h, persistent=False)
        self.register_buffer("pos_emb_w", emb_w, persistent=False)
        self.register_buffer("pos_emb_t", emb_t, persistent=False)

    def compute_1d_pos_embeddings(self):
        """
        Compute 1d pos embed for each axis
        """
        dim = self.model_channels
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"

        emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, pos=np.arange(self.len_h) * 1.0 / self.h_extrapolation_ratio)
        emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, pos=np.arange(self.len_w) * 1.0 / self.w_extrapolation_ratio)
        emb_t = get_1d_sincos_pos_embed_from_grid(dim_t, pos=np.arange(self.len_t) * 1.0 / self.t_extrapolation_ratio)

        emb_h = torch.from_numpy(emb_h).float()
        emb_w = torch.from_numpy(emb_w).float()
        emb_t = torch.from_numpy(emb_t).float()

        return emb_h, emb_w, emb_t

    def reset_parameters(self):
        emb_h, emb_w, emb_t = self.compute_1d_pos_embeddings()
        self.pos_emb_h = emb_h
        self.pos_emb_w = emb_w
        self.pos_emb_t = emb_t

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, C = B_T_H_W_C

        single_camera_T = T // self.n_cameras

        if self.interpolation == "crop":
            emb_h_H = self.pos_emb_h[:H]
            emb_w_W = self.pos_emb_w[:W]
            emb_t_T = self.pos_emb_t[:single_camera_T]
            emb = torch.cat(
                [
                    torch.cat(
                        [
                            repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W),
                            repeat(emb_h_H, "h d-> b t h w d", b=B, t=single_camera_T, w=W),
                            repeat(emb_w_W, "w d-> b t h w d", b=B, t=single_camera_T, h=H),
                        ],
                        dim=-1,
                    )
                    for _ in range(self.n_cameras)
                ],
                1,
            )
            assert list(emb.shape)[:4] == [B, T, H, W], f"bad shape: {list(emb.shape)[:4]} != {B, T, H, W}"
            return emb

    @property
    def seq_dim(self):
        return 1

    def _split_for_context_parallel(self, embeddings):
        if self._cp_group is not None:
            embeddings = rearrange(embeddings, "B (V T) H W C -> (B V) T H W C", V=self.n_cameras)
            embeddings = split_inputs_cp(x=embeddings, seq_dim=self.seq_dim, cp_group=self._cp_group)
            embeddings = rearrange(embeddings, "(B V) T H W C -> B (V T) H W C", V=self.n_cameras)
        return embeddings


class MultiViewDiT(MinimalV1LVGDiT):
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
        layer_mask: Optional[List[bool]] = None,
        sac_config: SACConfig = SACConfig(),
        **kwargs,
    ):
        self.state_t = state_t
        self.n_cameras_emb = n_cameras_emb
        self.view_condition_dim = view_condition_dim
        self.concat_view_embedding = concat_view_embedding
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
        del self.blocks
        self.blocks = nn.ModuleList(
            [
                MultiViewBlock(
                    x_dim=self.model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    backend=self.atten_backend,
                    image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                    state_t=self.state_t,
                    use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                )
                for _ in range(self.num_blocks)
            ]
        )

        if self.concat_view_embedding:
            self.view_embeddings = nn.Embedding(self.n_cameras_emb, view_condition_dim)

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
        pos_embedder = self.pos_embedder_options[f"n_cameras_{n_cameras}"]
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

        for block in self.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        return x_B_C_Tt_Hp_Wp
