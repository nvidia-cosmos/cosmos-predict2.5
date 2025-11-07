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

"""
Distriminator head for DMD2-style distillation.
Takes in intermediate features from a diffusion model and outputs a logit.

This version (V1) follows architecture from https://arxiv.org/pdf/2501.08316.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper

from cosmos_predict2._src.predict2.networks.selective_activation_checkpoint import (
    CheckpointMode,
    SACConfig,
    mm_only_context_fn,
)


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


class CrossAttention(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.query_token = nn.Parameter(torch.randn(1, 1, 1, head_dim))

        self.to_q = nn.Linear(head_dim, head_dim, bias=False)
        self.to_k = nn.Linear(head_dim, head_dim, bias=False)
        self.to_v = nn.Linear(head_dim, head_dim, bias=False)

        self.pre_norm_kv = RMSNorm(head_dim)
        self.post_norm_q = RMSNorm(head_dim)
        self.post_norm_k = RMSNorm(head_dim)

        self.to_out = nn.Linear(head_dim, head_dim)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.query_token, std=0.02)  # Initialize learnable query
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (RMSNorm, nn.LayerNorm)):
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

    def forward(self, x_intermediate_features: torch.Tensor) -> torch.Tensor:
        """
        x_intermediate_features: (B, L_visual, D_model) - Features from backbone.
        """
        B, L, D = x_intermediate_features.shape

        query = self.query_token.expand(B, -1, -1, -1)  # (B, 1, 1, D_model)
        residual = query

        q_proj = self.to_q(query)  # (B, 1, 1, D_model)

        normed_features = self.pre_norm_kv(x_intermediate_features[:, None, :, :])  # (B, 1, L_visual, D_model)
        k_proj = self.to_k(normed_features)  # (B, 1, L_visual, D_model)
        v_proj = self.to_v(normed_features)  # (B, 1, L_visual, D_model)

        q_normed = self.post_norm_q(q_proj)  # (B, 1, L_visual, D_model)
        k_normed = self.post_norm_k(k_proj)  # (B, 1, L_visual, D_model)

        out = torch.nn.functional.scaled_dot_product_attention(q_normed, k_normed, v_proj)  # (B, 1, 1, D_model)

        out = self.to_out(out)  # (B, 1, 1, D_model)

        return (out + residual)[:, 0, 0, :]  # (B, D_model)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (RMSNorm, nn.LayerNorm)):
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_normed = self.norm(x)
        x_fc1 = self.fc1(x_normed)
        x_act = self.act(x_fc1)
        x_fc2 = self.fc2(x_act)
        return x_fc2 + residual


class DiscriminatorBranch(nn.Module):
    """
    A branch consisting of a Cross Attention block followed by an MLP block,
    """

    def __init__(self, model_channels: int, mlp_ratio: float):
        super().__init__()
        self.cross_attention = CrossAttention(head_dim=model_channels)
        self.mlp = MLP(dim=model_channels, mlp_ratio=mlp_ratio)

    def forward(self, x_intermediate_features: torch.Tensor) -> torch.Tensor:
        """
        x_intermediate_features: (B, L_visual, D_model) - From backbone.
        """
        x = self.cross_attention(x_intermediate_features)
        x = self.mlp(x)
        return x  # (B, D_model)


class DiscriminatorHead(nn.Module):
    """
    Discriminator branch following https://arxiv.org/pdf/2501.08316
    """

    def __init__(
        self,
        model_channels: int,
        num_branches: int,
        mlp_ratio_branch: float = 4.0,
        output_dim: int = 1,
        sac_config: SACConfig = SACConfig(),
    ):
        super().__init__()
        self.model_channels = model_channels
        self.num_branches = num_branches

        self.branches = nn.ModuleList(
            [
                DiscriminatorBranch(model_channels=model_channels, mlp_ratio=mlp_ratio_branch)
                for _ in range(num_branches)
            ]
        )

        # Each branch outputs (B, D_model).
        concatenated_dim = model_channels * num_branches

        self.final_norm = nn.LayerNorm(concatenated_dim, eps=1e-6)
        self.final_linear = nn.Linear(concatenated_dim, output_dim)

        self.init_weights()
        self.enable_selective_checkpoint(sac_config)
        self._is_context_parallel_enabled = False

    def init_weights(self):
        if hasattr(self.final_norm, "reset_parameters"):
            self.final_norm.reset_parameters()
        if hasattr(self.final_linear, "weight"):
            nn.init.xavier_uniform_(self.final_linear.weight)
            if self.final_linear.bias is not None:
                nn.init.zeros_(self.final_linear.bias)

    def forward(self, intermediate_features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        intermediate_features_list: List of visual tokens from backbone layers.
                                      Each tensor: (B, L_visual, D_model).
        """
        if len(intermediate_features_list) != len(self.branches):
            raise ValueError("Num intermediate features must match num branches.")

        branch_outputs = []
        for i, x_intermediate in enumerate(intermediate_features_list):
            branch_out = self.branches[i](x_intermediate)  # (B, D_model)
            branch_outputs.append(branch_out)

        concatenated_features = torch.cat(branch_outputs, dim=1)  # (B, num_branches * D_model)

        normed_features = self.final_norm(concatenated_features)
        logit_output = self.final_linear(normed_features)

        return logit_output  # (B, output_dim)

    def fully_shard(self, mesh):
        for branch in self.branches:
            fully_shard(branch, mesh=mesh, reshard_after_forward=True)

        fully_shard(self.final_norm, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.final_linear, mesh=mesh, reshard_after_forward=True)

    def disable_context_parallel(self):
        self._is_context_parallel_enabled = False

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None):
        self._is_context_parallel_enabled = True

    @property
    def is_context_parallel_enabled(self):
        return self._is_context_parallel_enabled

    def enable_selective_checkpoint(self, sac_config: SACConfig):
        if sac_config.mode == CheckpointMode.MM_ONLY:
            for branch_id, branch in self.branches.named_children():
                branch = ptd_checkpoint_wrapper(
                    branch,
                    context_fn=mm_only_context_fn,
                    preserve_rng_state=False,
                )
                self.branches.register_module(branch_id, branch)
            self.register_module(
                "final_norm",
                ptd_checkpoint_wrapper(
                    self.final_norm,
                    context_fn=mm_only_context_fn,
                    preserve_rng_state=False,
                ),
            )
            self.register_module(
                "final_linear",
                ptd_checkpoint_wrapper(
                    self.final_linear,
                    context_fn=mm_only_context_fn,
                    preserve_rng_state=False,
                ),
            )
        elif sac_config.mode == CheckpointMode.NONE:
            pass
        else:
            raise ValueError(f"Invalid checkpoint mode: {sac_config.mode}")

        return self
