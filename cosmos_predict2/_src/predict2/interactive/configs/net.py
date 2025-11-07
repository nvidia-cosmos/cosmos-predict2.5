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


from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.action.configs.action_conditioned.net import COSMOS_V1_2B_NET_MININET_ACTION_CHUNK
from cosmos_predict2._src.predict2.interactive.networks.dit_action_causal import (
    ActionChunkCausalDIT,
    ActionChunkCausalDITKVCache,
    ActionChunkCausalDITwithConditionalMaskKVCache,
)
from cosmos_predict2._src.predict2.interactive.networks.dit_causal import CausalDITKVCache, CausalDITwithConditionalMask
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig

BASE_NET_KWARGS = dict(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    extra_per_block_abs_pos_emb=True,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=2.0,
    sac_config=SACConfig(),
    partial_finetune=False,
)


def make_net(cls, atten_backend: str, **overrides) -> LazyDict:
    kwargs = dict(BASE_NET_KWARGS)
    kwargs["atten_backend"] = atten_backend
    kwargs.update(overrides)
    return L(cls)(**kwargs)


# Causal DiT (no camera)
CAUSAL_COSMOS_V1_7B_NET_MININET = make_net(
    CausalDITwithConditionalMask,
    atten_backend="ulysses",
    model_channels=4096,
    num_blocks=28,
    num_heads=32,
)

CAUSAL_COSMOS_V1_2B_NET_MININET = make_net(
    CausalDITwithConditionalMask,
    atten_backend="ulysses",
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    extra_per_block_abs_pos_emb=False,
    rope_t_extrapolation_ratio=1.0,
)

CAUSAL_COSMOS_V1_14B_NET_MININET = make_net(
    CausalDITwithConditionalMask,
    atten_backend="ulysses",
    model_channels=5120,
    num_blocks=36,
    num_heads=40,
    extra_per_block_abs_pos_emb=False,
    rope_t_extrapolation_ratio=1.0,
)

# Causal DiT with KV cache (no camera)
CAUSAL_KVCACHE_COSMOS_V1_2B_NET_MININET = make_net(
    CausalDITKVCache,
    atten_backend="ulysses",
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    extra_per_block_abs_pos_emb=False,
    rope_t_extrapolation_ratio=1.0,
)

# Action-conditioned Causal DiT (no camera)
ACTION_CAUSAL_COSMOS_V1_2B_NET_MININET = make_net(
    ActionChunkCausalDIT,
    atten_backend="ulysses",
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    extra_per_block_abs_pos_emb=False,
    rope_t_extrapolation_ratio=1.0,
)

# Action-conditioned Causal DiT with KV cache (no camera)
ACTION_CAUSAL_KVCACHE_COSMOS_V1_2B_NET_MININET = make_net(
    ActionChunkCausalDITKVCache,
    atten_backend="ulysses",
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    extra_per_block_abs_pos_emb=False,
    rope_t_extrapolation_ratio=1.0,
)

ACTION_CAUSAL_KVCACHE_COSMOS_V1_2B_NET_MININET = make_net(
    ActionChunkCausalDITwithConditionalMaskKVCache,
    atten_backend="ulysses",
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    extra_per_block_abs_pos_emb=False,
    rope_t_extrapolation_ratio=1.0,
)


def register_net():
    cs = ConfigStore.instance()

    for net_group in ["net", "net_fake_score", "net_teacher"]:
        cs.store(
            group=net_group,
            package=f"model.config.{net_group}",
            name="causal_cosmos_v1_2B",
            node=CAUSAL_COSMOS_V1_2B_NET_MININET,
        )
        cs.store(
            group=net_group,
            package=f"model.config.{net_group}",
            name="action_causal_kvcache_cosmos_v1_2B",
            node=ACTION_CAUSAL_KVCACHE_COSMOS_V1_2B_NET_MININET,
        )
        cs.store(
            group=net_group,
            package=f"model.config.{net_group}",
            name="causal_kvcache_cosmos_v1_2B",
            node=CAUSAL_KVCACHE_COSMOS_V1_2B_NET_MININET,
        )
        cs.store(
            group=net_group,
            package=f"model.config.{net_group}",
            name="action_causal_cosmos_v1_2B",
            node=ACTION_CAUSAL_COSMOS_V1_2B_NET_MININET,
        )
        cs.store(
            group=net_group,
            package=f"model.config.{net_group}",
            name="action_causal_kvcache_cosmos_v1_2B",
            node=ACTION_CAUSAL_KVCACHE_COSMOS_V1_2B_NET_MININET,
        )


def register_net_fake_score():
    cs = ConfigStore.instance()
    cs.store(
        group="net_fake_score",
        package="model.config.net_fake_score",
        name="cosmos_v1_2B_action_chunk_conditioned",
        node=COSMOS_V1_2B_NET_MININET_ACTION_CHUNK,
    )


def register_net_teacher():
    cs = ConfigStore.instance()
    cs.store(
        group="net_teacher",
        package="model.config.net_teacher",
        name="cosmos_v1_2B_action_chunk_conditioned",
        node=COSMOS_V1_2B_NET_MININET_ACTION_CHUNK,
    )
