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
from typing import Any, List, Literal, Optional, Tuple, Union

import megatron.core.parallel_state as parallel_state
import torch
import torch.amp as amp
import torch.nn as nn
import transformer_engine as te
from einops import rearrange
from torch.distributed import ProcessGroup, get_process_group_ranks
from torchvision import transforms

from cosmos_predict2._src.imaginaire.utils.graph import create_cuda_graph
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
    Attention,
    FinalLayer,
    PatchEmbed,
    SACConfig,
    TimestepEmbedding,
    Timesteps,
    replace_selfattn_op_with_sparse_attn_op,
)
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import Block as BaseBlock
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT as BaseMiniTrainDIT


class I2VCrossAttentionFull(Attention):
    """
    A modified Attention class that adds separate query, key, value projections for reference image attention.
    """

    def __init__(self, *args, img_latent_dim: int = 1024, **kwargs):
        super().__init__(*args, **kwargs)
        inner_dim = self.head_dim * self.n_heads
        self.k_img = nn.Linear(img_latent_dim, inner_dim, bias=False)
        self.v_img = nn.Linear(img_latent_dim, inner_dim, bias=False)
        self.q_img = nn.Linear(self._query_dim, inner_dim, bias=False)  # NEW: separate query for image attention
        self.q_img_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)  # NEW: dedicated normalization for q_img
        self.k_img_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

    def init_weights(self) -> None:
        super().init_weights()
        torch.nn.init.trunc_normal_(self.k_img.weight, std=1.0 / math.sqrt(self._inner_dim))
        torch.nn.init.trunc_normal_(self.v_img.weight, std=1.0 / math.sqrt(self._inner_dim))
        torch.nn.init.trunc_normal_(
            self.q_img.weight, std=1.0 / math.sqrt(self._query_dim)
        )  # NEW: initialize q_img with correct dim
        self.q_img_norm.reset_parameters()  # NEW: initialize q_img_norm
        self.k_img_norm.reset_parameters()

    def compute_qkv(
        self, x, context, rope_emb=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_context, img_context = context
        q, k, v = super().compute_qkv(x, text_context, rope_emb)

        # Compute image-specific query, key, value - following base Attention class pattern
        q_img = self.q_img(x)  # NEW: separate query projection for image attention (same input x as regular q)
        k_img = self.k_img(img_context)
        v_img = self.v_img(img_context)

        # Rearrange q_img, k_img, v_img - same as base Attention class
        q_img, k_img, v_img = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q_img, k_img, v_img),
        )

        # Apply normalization - following base Attention class pattern
        q_img = self.q_img_norm(q_img)  # Use dedicated q_img_norm for image queries
        k_img = self.k_img_norm(k_img)  # Use dedicated k_img_norm for image keys

        # Apply rotary embeddings if needed (only for self-attention)
        # Note: q_img and k_img don't need rotary embeddings since they're for cross-attention

        return q, k, v, q_img, k_img, v_img

    def compute_attention(self, q, k, v, q_img, k_img, v_img):
        result = self.attn_op(q, k, v)  # [B, S, H, D] - text attention using shared q
        result_img = self.attn_op(q_img, k_img, v_img)  # [B, S, H, D] - image attention using separate q_img
        return self.output_dropout(self.output_proj(result + result_img))

    def forward(
        self,
        x,
        context=None,
        rope_emb=None,
    ):
        q, k, v, q_img, k_img, v_img = self.compute_qkv(x, context, rope_emb)
        return self.compute_attention(q, k, v, q_img, k_img, v_img)


# Modified BaseBlock class with share_q_in_i2v_cross_attn parameter
class Block(BaseBlock):
    """
    This is a modified version of the BaseBlock class from minimal_v4_dit.py.
    It adds a share_q_in_i2v_cross_attn parameter to the block constructor.
    If share_q_in_i2v_cross_attn is False, it uses the I2VCrossAttentionFull class instead of the I2VCrossAttention class.
    This is used to separate the query projection for image attention.

    From original BaseBlock class:
    A transformer block that combines self-attention, cross-attention and MLP layers with AdaLN modulation.
    Each component (self-attention, cross-attention, MLP) has its own layer normalization and AdaLN modulation.

    Parameters:
        x_dim (int): Dimension of input features
        context_dim (int): Dimension of context features for cross-attention
        num_heads (int): Number of attention heads
        mlp_ratio (float): Multiplier for MLP hidden dimension. Default: 4.0
        use_adaln_lora (bool): Whether to use AdaLN-LoRA modulation. Default: False
        adaln_lora_dim (int): Hidden dimension for AdaLN-LoRA layers. Default: 256
        [NEW] share_q_in_i2v_cross_attn (bool): Whether to share q in i2v cross-attention. Default: True

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
        share_q_in_i2v_cross_attn: bool = False,
        use_wan_fp32_strategy: bool = False,
    ):
        # Call parent constructor first
        super().__init__(
            x_dim=x_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            backend=backend,
            image_context_dim=image_context_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        # Override cross attention if we need separate q for image attention
        if image_context_dim is not None and not share_q_in_i2v_cross_attn:
            self.cross_attn = I2VCrossAttentionFull(
                x_dim,
                context_dim,
                num_heads,
                x_dim // num_heads,
                img_latent_dim=image_context_dim,
                qkv_format="bshd",
            )


class ControlAwareDiTBlock(Block):
    """A modified DiTBlock for the base model that accepts control branch modulations ("hints") if available.

    Args:
        has_image_input: Whether the block accepts image input
        dim: Dimension of the input features
        num_heads: Number of attention heads
        ffn_dim: Dimension of the feed-forward network
        eps: Epsilon value for normalization
        block_id: ID of the block, used to index into the hints tensor
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
        block_id: Optional[int] = None,
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__(
            x_dim,
            context_dim,
            num_heads,
            mlp_ratio,
            use_adaln_lora,
            adaln_lora_dim,
            backend,
            image_context_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )
        self.block_id = block_id

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        hints: torch.Tensor,
        control_context_scale: Union[float, torch.Tensor] = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass of the block.

        Args:
            x: Input tensor
            hints: Control signals from the control branch. Pass a Tensor filled with zeroes in order for hints to not take any effect.
            control_context_scale: Optional scaling factor for the hints
            **kwargs: Additional arguments passed to the base DiTBlock

        Returns:
            Processed tensor with hints added
        """
        x_B_T_H_W_D = super().forward(x_B_T_H_W_D, **kwargs)
        if self.block_id is not None:
            x_B_T_H_W_D += hints[self.block_id] * control_context_scale
        return x_B_T_H_W_D


class ControlEncoderDiTBlock(Block):
    """A modified DiTBlock for the control branch that *generates* per-block control modulation signals.
      compared to the base DiTBlock, it adds skip connections and zero convolutions.

    Args:
        has_image_input: Whether the block accepts image input
        dim: Dimension of the input features
        num_heads: Number of attention heads
        ffn_dim: Dimension of the feed-forward network
        eps: Epsilon value for normalization
        block_id: ID of the block, used for skip connections
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
        block_id: int = 0,
        hint_dim: Optional[int] = None,
        use_after_proj: bool = True,
        use_wan_fp32_strategy: bool = False,
        use_cuda_graphs: bool = False,
    ) -> None:
        super().__init__(
            x_dim,
            context_dim,
            num_heads,
            mlp_ratio,
            use_adaln_lora,
            adaln_lora_dim,
            backend,
            image_context_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )
        self.block_id = block_id
        self.use_after_proj = use_after_proj
        self.use_cuda_graphs = use_cuda_graphs

        # Zero convolution as in ControlNet
        if block_id == 0:
            self.before_proj = nn.Linear(hint_dim if hint_dim else self.x_dim, self.x_dim)
        if use_after_proj:
            self.after_proj = nn.Linear(self.x_dim, self.x_dim)

    def init_weights(self):
        super().init_weights()
        if self.use_after_proj:
            nn.init.zeros_(self.after_proj.weight)
            nn.init.zeros_(self.after_proj.bias)
        if self.block_id == 0:
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

    def pre_forward(self, c: torch.Tensor, x_B_T_H_W_D: torch.Tensor):
        all_c = []
        if self.block_id == 0:
            c = self.before_proj(c) + x_B_T_H_W_D
        elif self.use_after_proj:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        return c, all_c

    def post_forward(self, c: torch.Tensor, all_c: list[torch.Tensor]):
        if self.use_after_proj:
            c_skip = self.after_proj(c)
            all_c += [c_skip, c]
            c = torch.stack(all_c)
        return c

    def forward(self, c: torch.Tensor, x_B_T_H_W_D: torch.Tensor, **kwargs):
        if self.use_cuda_graphs:
            # When using CUDA Graphs, pre_forward and post_forward methods must be called by user manually outside of
            # the forward call. That is because these methods are not graphable due to changing input shapes.
            return super().forward(c, **kwargs)
        else:
            c, all_c = self.pre_forward(c, x_B_T_H_W_D)
            c = super().forward(c, **kwargs)
            return self.post_forward(c, all_c)


# Modified BaseMiniTrainDIT class by adding reference image parameter
class MiniTrainDITImageContext(BaseMiniTrainDIT):
    """
    This is a modified version of the MiniTrainDIT class from minimal_v4_dit.py.
    It adds img_context_deep_proj and share_q_in_i2v_cross_attn functionality to the base MiniTrainDIT.

    From original MiniTrainDIT class:
    Extended MiniTrainDIT class that adds img_context_deep_proj and share_q_in_i2v_cross_attn functionality
    to the base MiniTrainDIT from minimal_v4_dit.py.

    New parameters:
        img_context_deep_proj (bool): Whether to use deep MLP projection for image context.
            - False (default): Simple projection (Linear + GELU) for backward compatibility
            - True: Deep MLP projection (Linear -> GELU -> Linear -> LayerNorm) following IP-Adapter Full style
        share_q_in_i2v_cross_attn (bool): Whether to share q in i2v cross-attention. Default: True
            - True (default): Use I2VCrossAttention (shared query between text and image attention)
            - False: Use I2VCrossAttentionFull (separate query projections for text and image attention)
    """

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: tuple,
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
        extra_image_context_dim: Optional[int] = None,  # Main flag of whether user reference image
        img_context_deep_proj: bool = False,  # work when extra_image_context_dim is not None
        share_q_in_i2v_cross_attn: bool = False,  # work when extra_image_context_dim is not None
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
        gna_parameters=None,
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        # Initialize the grandparent class (whatever the parent inherits from)
        super(BaseMiniTrainDIT, self).__init__()

        # Store parameters in the same order as parent class
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
        # NEW: Our additional parameters
        self.img_context_deep_proj = img_context_deep_proj
        self.share_q_in_i2v_cross_attn = share_q_in_i2v_cross_attn
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        # Component building (same order as parent)
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

        # Create blocks with our modified Block class
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
                    share_q_in_i2v_cross_attn=share_q_in_i2v_cross_attn,
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
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        self.t_embedding_norm = te.pytorch.RMSNorm(model_channels, eps=1e-6)

        # Create image context projection with deep support
        if extra_image_context_dim is not None:
            if img_context_deep_proj:
                # Deep MLP projection
                self.img_context_proj = nn.Sequential(
                    nn.Linear(extra_image_context_dim, extra_image_context_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(extra_image_context_dim, model_channels, bias=False),
                    nn.LayerNorm(model_channels),
                )
            else:
                # Simple projection
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
            self = replace_selfattn_op_with_sparse_attn_op(self, n_dense_blocks, gna_parameters=gna_parameters)

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

        # Handle image context projection initialization
        if self.extra_image_context_dim is not None:
            if self.img_context_deep_proj:
                # Initialize deep projection with proper scaling
                for layer in self.img_context_proj:
                    if isinstance(layer, nn.Linear):
                        std = 1.0 / math.sqrt(layer.in_features)
                        torch.nn.init.trunc_normal_(layer.weight, std=std, a=-3 * std, b=3 * std)
                    elif isinstance(layer, nn.LayerNorm):
                        layer.reset_parameters()
            else:
                # Simple projection initialization (same as parent)
                self.img_context_proj[0].reset_parameters()


class MinimalV4LVGControlVaceDiT(MiniTrainDITImageContext):
    """
    Adding control branch to the base model.
    """

    def __init__(
        self,
        *args,
        crossattn_emb_channels: int = 1024,
        mlp_ratio: float = 4.0,
        vace_has_mask: bool = False,
        vace_block_every_n: int = 2,
        condition_strategy: Literal["spaced", "first_n"] = "spaced",
        num_max_modalities: int = 8,
        use_input_hint_block: bool = False,
        spatial_compression_factor: int = 8,
        num_control_branches: int = 1,
        separate_embedders: bool = False,
        use_after_proj_for_multi_branch: bool = True,
        timestep_scale: float = 1.0,  # Add timestep scaling for rectified flow
        use_cuda_graphs: bool = False,
        **kwargs,
    ):
        """
        vace_block_every_n: create one control block every n base model blocks
        vace_has_mask: if true, control branch latent is [inactive, reactive, mask] as in VACE paper. Otherwise, just the latent of the control input
        condition_strategy: How the control blocks correspond to the base model blocks. "first_n" conditions first n base model blocks.
            "spaced" conditions every vace_block_every_n base model block. E.g. vace_block_every_n=2, condition_strategy="spaced" means control block 0
            controls base block 0 and 2, control block 1 controls base block 2, etc.
        use_cuda_graphs (bool, optional): Whether to use CUDA Graphs for inference. Defaults to False.
        """

        assert "in_channels" in kwargs, "in_channels must be provided"

        kwargs["in_channels"] += 1  # Add 1 for the condition mask
        nf = kwargs["model_channels"]
        hint_nf = kwargs.pop("hint_nf", [nf, nf, nf, nf, nf, nf, nf, nf])
        self.num_control_branches = num_control_branches
        self.use_after_proj_for_multi_branch = use_after_proj_for_multi_branch
        self.timestep_scale = timestep_scale  # Store timestep scale for rectified flow
        self.use_cuda_graphs = use_cuda_graphs
        self.cuda_graphs = {}
        self.controlnet_cuda_graphs = {}
        self.cfg_parallel = False
        super().__init__(
            *args,
            crossattn_emb_channels=crossattn_emb_channels,
            mlp_ratio=mlp_ratio,
            **kwargs,
        )

        self.crossattn_emb_channels = crossattn_emb_channels
        self.mlp_ratio = mlp_ratio

        # if vace_has_mask, the control latent is 16 + 64 (for mask)
        self.vace_has_mask = vace_has_mask
        self.num_max_modalities = num_max_modalities
        in_channels = self.in_channels - 1  # subtract the condition mask
        self.vace_in_channels = (in_channels + spatial_compression_factor**2) if vace_has_mask else in_channels
        self.vace_in_channels *= num_max_modalities
        self.vace_in_channels += 1  # adding the condition mask back

        # for finding corresponding control block with base model block.
        self.condition_strategy = condition_strategy
        if self.condition_strategy == "spaced":
            # base block k uses the 2k'th element in the hint list, {0:0, 2:1, 4:2, ...}, as in VACE paper
            self.control_layers = [i for i in range(0, self.num_blocks, vace_block_every_n)]
            self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers)}
        elif self.condition_strategy == "first_n":
            # condition first n base model blocks, where n is number of control blocks
            self.control_layers = list(range(0, self.num_blocks // vace_block_every_n))
            self.control_layers_mapping = {i: i for i in range(len(self.control_layers))}
        else:
            raise ValueError(f"Invalid condition strategy: {self.condition_strategy}")
        assert 0 in self.control_layers

        # Input hint block
        self.use_input_hint_block = use_input_hint_block
        if use_input_hint_block:
            assert self.num_control_branches == 1, "input hint block is not supported for multi-branch"
            input_hint_block = []
            nonlinearity = nn.SiLU()
            for i in range(len(hint_nf) - 1):
                input_hint_block += [nn.Linear(hint_nf[i], hint_nf[i + 1]), nonlinearity]
            self.input_hint_block = nn.Sequential(*input_hint_block)

        # -------- Base model --------

        # Base model blocks. Overwrite them to enable accepting the control branch modulations ("hints").
        # Shape remains the same as the base model so we can load pretrained weights.
        self.blocks = nn.ModuleList(
            [
                ControlAwareDiTBlock(
                    x_dim=self.model_channels,
                    context_dim=self.crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    backend=self.atten_backend,
                    image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                    block_id=self.control_layers_mapping[i] if i in self.control_layers else None,
                    use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                )
                for i in range(self.num_blocks)
            ]
        )

        # -------- Control branch --------
        self.separate_embedders = separate_embedders
        if separate_embedders:
            self.t_embedder_for_control_branch = nn.Sequential(
                Timesteps(self.model_channels),
                TimestepEmbedding(self.model_channels, self.model_channels, use_adaln_lora=self.use_adaln_lora),
            )
            self.t_embedding_norm_for_control_branch = te.pytorch.RMSNorm(self.model_channels, eps=1e-6)

        self.build_patch_embed_vace()

        if self.num_control_branches > 1:
            for nc in range(self.num_control_branches):
                setattr(
                    self,
                    f"control_blocks_{nc}",
                    nn.ModuleList(
                        [
                            ControlEncoderDiTBlock(
                                x_dim=self.model_channels,
                                context_dim=self.crossattn_emb_channels,
                                num_heads=self.num_heads,
                                mlp_ratio=self.mlp_ratio,
                                use_adaln_lora=self.use_adaln_lora,
                                adaln_lora_dim=self.adaln_lora_dim,
                                backend=self.atten_backend,
                                image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                                block_id=i,
                                hint_dim=hint_nf[-1] if use_input_hint_block else None,
                                use_after_proj=not use_after_proj_for_multi_branch,
                                use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                                use_cuda_graphs=self.use_cuda_graphs,
                            )
                            for i in self.control_layers
                        ]
                    ),
                )
            if use_after_proj_for_multi_branch:
                self.after_proj = nn.ModuleList(
                    [
                        nn.Linear(self.model_channels * self.num_control_branches, self.model_channels)
                        for _ in range(len(self.control_layers))
                    ]
                )
        else:
            self.control_blocks = nn.ModuleList(
                [
                    ControlEncoderDiTBlock(
                        x_dim=self.model_channels,
                        context_dim=self.crossattn_emb_channels,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        use_adaln_lora=self.use_adaln_lora,
                        adaln_lora_dim=self.adaln_lora_dim,
                        backend=self.atten_backend,
                        image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                        block_id=i,
                        hint_dim=hint_nf[-1] if use_input_hint_block else None,
                        use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                        use_cuda_graphs=self.use_cuda_graphs,
                    )
                    for i in self.control_layers
                ]
            )

        self.init_weights()
        sac_config = kwargs.get("sac_config", SACConfig())
        self.enable_selective_checkpoint(sac_config, self.blocks)
        if self.num_control_branches > 1:
            for nc in range(self.num_control_branches):
                self.enable_selective_checkpoint(sac_config, getattr(self, f"control_blocks_{nc}"))
        else:
            self.enable_selective_checkpoint(sac_config, self.control_blocks)

    def build_patch_embed_vace(self):
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.vace_in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        if self.num_control_branches > 1:
            self.control_embedder = nn.ModuleList(
                [
                    PatchEmbed(
                        spatial_patch_size=patch_spatial,
                        temporal_patch_size=patch_temporal,
                        in_channels=in_channels,
                        out_channels=model_channels,
                    )
                    for _ in range(self.num_control_branches)
                ]
            )
        else:
            self.control_embedder = PatchEmbed(
                spatial_patch_size=patch_spatial,
                temporal_patch_size=patch_temporal,
                in_channels=in_channels,
                out_channels=model_channels,
            )
            if self.separate_embedders:
                self.x_embedder_for_control_branch = PatchEmbed(
                    spatial_patch_size=patch_spatial,
                    temporal_patch_size=patch_temporal,
                    in_channels=self.in_channels + 1 if concat_padding_mask else self.in_channels,
                    out_channels=model_channels,
                )

    def init_weights(self):
        super().init_weights()

        if hasattr(self, "input_hint_block"):
            for module in self.input_hint_block.modules():
                if hasattr(module, "weight"):
                    std = 1.0 / math.sqrt(module.weight.shape[0])
                    torch.nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)

        if self.num_control_branches > 1:
            for nc in range(self.num_control_branches):
                if hasattr(self, "control_embedder"):  # control branch initialization
                    self.control_embedder[nc].init_weights()
                    for block in getattr(self, f"control_blocks_{nc}"):
                        block.init_weights()
            if hasattr(self, "after_proj"):
                for cl in range(len(self.control_layers)):
                    nn.init.zeros_(self.after_proj[cl].weight)
                    nn.init.zeros_(self.after_proj[cl].bias)
        else:
            if hasattr(self, "control_embedder"):
                self.control_embedder.init_weights()
            if hasattr(self, "control_blocks"):
                for block in self.control_blocks:
                    block.init_weights()

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        embedder: Optional[PatchEmbed] = None,
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
        x_B_T_H_W_D = embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]

        return x_B_T_H_W_D, None, extra_pos_emb

    def disable_context_parallel(self):
        # pos_embedder
        self.pos_embedder.disable_context_parallel()
        self.cfg_parallel = False
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.disable_context_parallel()

        # attention
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )
        if self.num_control_branches > 1:
            for nc in range(self.num_control_branches):
                for block in getattr(self, f"control_blocks_{nc}"):
                    block.self_attn.set_context_parallel_group(
                        process_group=None,
                        ranks=None,
                        stream=torch.cuda.Stream(),
                    )
        else:
            for block in self.control_blocks:
                block.self_attn.set_context_parallel_group(
                    process_group=None,
                    ranks=None,
                    stream=torch.cuda.Stream(),
                )

        self._is_context_parallel_enabled = False

    def enable_context_parallel(
        self, process_group: Optional[ProcessGroup] = None, cfg_parallel: bool = False, cp_comm_type="p2p"
    ):
        # pos_embedder: shared between base and control branch
        if (
            not self.cfg_parallel
        ):  # If cfg_parallel is enabled, we don't overwrite it. disable_context_parallel should be used first instead.
            self.cfg_parallel = cfg_parallel
        if self._is_context_parallel_enabled:
            return
        self.pos_embedder.enable_context_parallel(process_group=process_group)
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.enable_context_parallel(process_group=process_group)

        # attention
        cp_ranks = get_process_group_ranks(process_group)
        if cp_comm_type == "a2a+p2p":
            cp_group = parallel_state.get_hierarchical_context_parallel_groups()
        else:
            cp_group = process_group
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(
                process_group=cp_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
                cp_comm_type=cp_comm_type,
            )
        if self.num_control_branches > 1:
            for nc in range(self.num_control_branches):
                for block in getattr(self, f"control_blocks_{nc}"):
                    block.self_attn.set_context_parallel_group(
                        process_group=cp_group,
                        ranks=cp_ranks,
                        stream=torch.cuda.Stream(),
                        cp_comm_type=cp_comm_type,
                    )
        else:
            for block in self.control_blocks:
                block.self_attn.set_context_parallel_group(
                    process_group=cp_group,
                    ranks=cp_ranks,
                    stream=torch.cuda.Stream(),
                    cp_comm_type=cp_comm_type,
                )

        self._is_context_parallel_enabled = True

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        latent_control_input: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        img_context_emb: Optional[torch.Tensor] = None,
        control_context_scale: float | torch.Tensor = 1.0,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs
        assert not (self.training and self.use_cuda_graphs), "CUDA Graphs are supported only for inference"
        # control branch forward
        # Get the original shape
        B, C, T, H, W = x_B_C_T_H_W.shape

        def _pad_control_input(control_B_C_T_H_W):
            # Pad control input channels to match the maximum number of modalities.
            if control_B_C_T_H_W.shape[1] < self.vace_in_channels - 1:
                pad_C = self.vace_in_channels - 1 - control_B_C_T_H_W.shape[1]
                # log.info(f"Input control has {c} channels, but we need {self.vace_in_channels} channels. Padding with zeros.")
                control_B_C_T_H_W = torch.cat(
                    [
                        control_B_C_T_H_W,
                        torch.zeros(
                            (B, pad_C, T, H, W), dtype=control_B_C_T_H_W.dtype, device=control_B_C_T_H_W.device
                        ),
                    ],
                    dim=1,
                )
            return control_B_C_T_H_W

        if self.num_control_branches == 1:
            control_B_C_T_H_W = latent_control_input
            control_B_C_T_H_W = _pad_control_input(control_B_C_T_H_W)
        else:
            control_B_C_T_H_W = latent_control_input.chunk(self.num_control_branches, dim=1)
            control_B_C_T_H_W = [_pad_control_input(c) for c in control_B_C_T_H_W]

        def _prepare_transformer_input(x_B_C_T_H_W, embedder):
            if data_type == DataType.VIDEO:
                x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
            else:
                x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, torch.zeros_like(x_B_C_T_H_W[:, :1])], dim=1)

            x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
                x_B_C_T_H_W,
                fps=fps,
                padding_mask=padding_mask,
                embedder=embedder,
            )
            return x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D

        # Add condition mask to both input video and control signal
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = _prepare_transformer_input(
            x_B_C_T_H_W, embedder=self.x_embedder
        )
        if self.separate_embedders:
            x_B_T_H_W_D_for_control, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = (
                _prepare_transformer_input(x_B_C_T_H_W, embedder=self.x_embedder_for_control_branch)
            )
        else:
            x_B_T_H_W_D_for_control = x_B_T_H_W_D

        if self.num_control_branches > 1:
            control_B_T_H_W_D = []
            for nc in range(self.num_control_branches):
                control_B_C_T_H_W_i = control_B_C_T_H_W[nc]
                control_B_T_H_W_D_i, rope_emb_L_1_1_D_for_control, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D_for_control = (
                    _prepare_transformer_input(
                        control_B_C_T_H_W_i,
                        embedder=self.control_embedder[nc],
                    )
                )
                if not control_B_C_T_H_W_i.any():
                    control_B_T_H_W_D_i = torch.zeros_like(control_B_T_H_W_D_i)
                control_B_T_H_W_D.append(control_B_T_H_W_D_i)
        else:
            control_B_T_H_W_D, rope_emb_L_1_1_D_for_control, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D_for_control = (
                _prepare_transformer_input(
                    control_B_C_T_H_W,
                    embedder=self.control_embedder,
                )
            )

        #  If not using T5, project context emb's channel dim to constant shape
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if self.use_input_hint_block:
            control_B_T_H_W_D = self.input_hint_block(control_B_T_H_W_D)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, (
                "extra_image_context_dim must be set if img_context_emb is provided"
            )
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        timesteps_B_T = timesteps_B_T * self.timestep_scale

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

            if self.separate_embedders:
                t_embedding_B_T_D_for_control, adaln_lora_B_T_3D_for_control = self.t_embedder_for_control_branch(
                    timesteps_B_T
                )
                t_embedding_B_T_D_for_control = self.t_embedding_norm_for_control_branch(t_embedding_B_T_D_for_control)
            else:
                t_embedding_B_T_D_for_control = t_embedding_B_T_D
                adaln_lora_B_T_3D_for_control = adaln_lora_B_T_3D

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

        # NEW CODE: control branch forward
        def _get_control_weight(control_context_scale):
            if isinstance(control_context_scale, torch.Tensor):
                if control_context_scale.ndim == 0:  # Single scalar tensor
                    control_weight_maps = [float(control_context_scale)] * self.num_control_branches
                elif control_context_scale.ndim == 1:  # List of scalar weights
                    control_weight_maps = [float(w) for w in control_context_scale]
                else:  # Spatial-temporal weight maps
                    control_weight_maps = [w for w in control_context_scale]  # Keep as tensor
            elif isinstance(control_context_scale, (float, int)):
                control_weight_maps = [control_context_scale] * self.num_control_branches
            elif isinstance(control_context_scale, list) and all(isinstance(w, float) for w in control_context_scale):
                control_weight_maps = [float(w) for w in control_context_scale]
            else:
                raise ValueError(
                    f"Invalid control_context_scale type: {type(control_context_scale)} {control_context_scale}"
                )
            return control_weight_maps

        if self.num_control_branches > 1:
            hints = []
            has_hint_nc = [c.any() for c in control_B_T_H_W_D]
            for i in range(len(self.control_layers)):
                for nc in range(self.num_control_branches):
                    if has_hint_nc[nc] or torch.is_grad_enabled():
                        block = getattr(self, f"control_blocks_{nc}")[i]
                        control_B_T_H_W_D[nc] = block(
                            c=control_B_T_H_W_D[nc],
                            x_B_T_H_W_D=x_B_T_H_W_D_for_control,
                            emb_B_T_D=t_embedding_B_T_D_for_control,
                            crossattn_emb=context_input,
                            rope_emb_L_1_1_D=rope_emb_L_1_1_D_for_control,
                            adaln_lora_B_T_3D=adaln_lora_B_T_3D_for_control,
                            extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D_for_control,
                        )
                        control_B_T_H_W_D[nc] = control_B_T_H_W_D[nc] * has_hint_nc[nc]
                if self.use_after_proj_for_multi_branch:
                    # Normalize activations based on number of active branches
                    num_active_branches = sum(has_hint_nc)
                    control_B_T_H_W_D_concat = torch.cat(control_B_T_H_W_D, dim=-1) / num_active_branches
                    hints.append(self.after_proj[i](control_B_T_H_W_D_concat))
            if not self.use_after_proj_for_multi_branch:
                weight_maps_scalar_or_B_T_H_W_D = _get_control_weight(control_context_scale)
                control_B_T_H_W_D_sum = sum([c * w for c, w in zip(control_B_T_H_W_D, weight_maps_scalar_or_B_T_H_W_D)])
                hints = torch.unbind(control_B_T_H_W_D_sum)[:-1]  # list of layerwise control modulations
                control_context_scale = 1.0  # already scaled hints by control_context_scale
        else:
            control_block_kwargs = {
                "emb_B_T_D": t_embedding_B_T_D_for_control,
                "crossattn_emb": context_input,
                "rope_emb_L_1_1_D": rope_emb_L_1_1_D_for_control,
                "adaln_lora_B_T_3D": adaln_lora_B_T_3D_for_control,
                "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D_for_control,
            }
            if self.use_cuda_graphs:
                shapes_key = create_cuda_graph(
                    self.controlnet_cuda_graphs,
                    self.control_blocks,
                    [control_B_T_H_W_D, x_B_T_H_W_D_for_control],
                    control_block_kwargs,
                )
                cg_control_blocks = self.controlnet_cuda_graphs[shapes_key]
            else:
                cg_control_blocks = self.control_blocks
            for i, block in enumerate(self.control_blocks):
                if self.use_cuda_graphs:
                    control_B_T_H_W_D, all_c = block.pre_forward(control_B_T_H_W_D, x_B_T_H_W_D)
                control_B_T_H_W_D = cg_control_blocks[i](
                    control_B_T_H_W_D,
                    x_B_T_H_W_D_for_control,
                    **control_block_kwargs,
                )
                if self.use_cuda_graphs:
                    control_B_T_H_W_D = block.post_forward(control_B_T_H_W_D, all_c)
            hints = torch.unbind(control_B_T_H_W_D)[:-1]  # list of layerwise control modulations
            control_context_scale = control_context_scale[0]

        if isinstance(control_context_scale, float):
            control_context_scale = torch.tensor(
                control_context_scale, device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype
            )
        block_kwargs = {
            "emb_B_T_D": t_embedding_B_T_D,
            "crossattn_emb": context_input,
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            "control_context_scale": control_context_scale,
        }
        if self.use_cuda_graphs:
            hints = torch.stack(hints)
            shapes_key = create_cuda_graph(
                self.cuda_graphs,
                self.blocks,
                [x_B_T_H_W_D, hints],
                block_kwargs,
            )
            cg_blocks = self.cuda_graphs[shapes_key]
        else:
            cg_blocks = self.blocks

        for block in cg_blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                hints,
                **block_kwargs,
            )

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        return x_B_C_Tt_Hp_Wp
