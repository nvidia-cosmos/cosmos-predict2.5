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
Action-conditioned network configurations for Cosmos.
The config args are copied from cosmos_predict2/_src/predict2/configs/video2world/defaults/net.py.
Should keep in sync with the net definition in predict2.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.interactive.networks.dit_action_causal2p5 import ActionChunkCausalDITwithConditionalMask
from cosmos_predict2._src.predict2.action.networks.action_conditioned_minimal_v1_lvg_dit import (
    ActionChunkConditionedMinimalV1LVGDiT,
    ActionConditionedMinimalV1LVGDiT,
)
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig

base_cosmos_v1_2b_action_conditioned_net_args = dict(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    extra_per_block_abs_pos_emb=False,
    # Cross-attention projection
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
    crossattn_emb_channels=1024,
)

# ============ Action Conditioned 2B net ============
cosmos_v1_2b_action_conditioned_net_args = dict(
    **base_cosmos_v1_2b_action_conditioned_net_args,
    atten_backend="minimal_a2a",
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=SACConfig(),
)

COSMOS_V1_2B_ACTION_CONDITIONED_NET: LazyDict = L(ActionConditionedMinimalV1LVGDiT)(
    **cosmos_v1_2b_action_conditioned_net_args,
    use_wan_fp32_strategy=True,
)

# ============ Causal Action Conditioned 2B net ============
cosmos_v1_2b_causal_action_conditioned_net_args = dict(
    **base_cosmos_v1_2b_action_conditioned_net_args,
    rope_enable_fps_modulation=False,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=1.0,
    use_wan_fp32_strategy=True,
    sac_config=SACConfig(mode="mm_only"),
)

COSMOSV1_2B_CAUSAL_ACTION_CHUNK_CONDITIONED_NET = L(ActionChunkCausalDITwithConditionalMask)(
    **cosmos_v1_2b_causal_action_conditioned_net_args,
    atten_backend="i4",
)

# ============ Action Chunk Conditioned 2B net ============
COSMOS_V1_2B_ACTION_CHUNK_CONDITIONED_NET: LazyDict = L(ActionChunkConditionedMinimalV1LVGDiT)(
    **cosmos_v1_2b_action_conditioned_net_args,
    use_wan_fp32_strategy=True,
)

# ============ Action Conditioned 14B net ============
cosmos_v1_14b_action_conditioned_net_args = copy.deepcopy(cosmos_v1_2b_action_conditioned_net_args)
cosmos_v1_14b_action_conditioned_net_args["model_channels"] = 5120
cosmos_v1_14b_action_conditioned_net_args["num_heads"] = 40
cosmos_v1_14b_action_conditioned_net_args["num_blocks"] = 36

COSMOS_V1_14B_ACTION_CONDITIONED_NET: LazyDict = L(ActionConditionedMinimalV1LVGDiT)(
    **cosmos_v1_14b_action_conditioned_net_args,
    use_wan_fp32_strategy=True,
)

COSMOS_V1_14B_ACTION_CHUNK_CONDITIONED_NET: LazyDict = L(ActionChunkConditionedMinimalV1LVGDiT)(
    **cosmos_v1_14b_action_conditioned_net_args,
    use_wan_fp32_strategy=True,
)


def register_net_ac():
    cs = ConfigStore.instance()
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_action_conditioned_student",
        node=COSMOS_V1_2B_ACTION_CONDITIONED_NET,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_action_chunk_conditioned_student",
        node=COSMOS_V1_2B_ACTION_CHUNK_CONDITIONED_NET,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_14B_action_conditioned_student",
        node=COSMOS_V1_14B_ACTION_CONDITIONED_NET,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_14B_action_chunk_conditioned_student",
        node=COSMOS_V1_14B_ACTION_CHUNK_CONDITIONED_NET,
    )


def register_net_ac_causal():
    cs = ConfigStore.instance()

    for net_group in ["net", "net_fake_score", "net_teacher"]:
        cs.store(
            group=net_group,
            package=f"model.config.{net_group}",
            name="cosmos_v1_2B_causal_action_chunk_conditioned_student",
            node=COSMOSV1_2B_CAUSAL_ACTION_CHUNK_CONDITIONED_NET,
        )


def register_net_fake_score_ac():
    cs = ConfigStore.instance()
    cs.store(
        group="net_fake_score",
        package="model.config.net_fake_score",
        name="cosmos_v1_2B_action_conditioned_fake_score",
        node=COSMOS_V1_2B_ACTION_CONDITIONED_NET,
    )
    cs.store(
        group="net_fake_score",
        package="model.config.net_fake_score",
        name="cosmos_v1_2B_action_chunk_conditioned_fake_score",
        node=COSMOS_V1_2B_ACTION_CHUNK_CONDITIONED_NET,
    )
    cs.store(
        group="net_fake_score",
        package="model.config.net_fake_score",
        name="cosmos_v1_14B_action_conditioned_fake_score",
        node=COSMOS_V1_14B_ACTION_CONDITIONED_NET,
    )
    cs.store(
        group="net_fake_score",
        package="model.config.net_fake_score",
        name="cosmos_v1_14B_action_chunk_conditioned_fake_score",
        node=COSMOS_V1_14B_ACTION_CHUNK_CONDITIONED_NET,
    )


def register_net_teacher_ac():
    cs = ConfigStore.instance()
    cs.store(
        group="net_teacher",
        package="model.config.net_teacher",
        name="cosmos_v1_2B_action_conditioned_teacher",
        node=COSMOS_V1_2B_ACTION_CONDITIONED_NET,
    )
    cs.store(
        group="net_teacher",
        package="model.config.net_teacher",
        name="cosmos_v1_2B_action_chunk_conditioned_teacher",
        node=COSMOS_V1_2B_ACTION_CHUNK_CONDITIONED_NET,
    )
    cs.store(
        group="net_teacher",
        package="model.config.net_teacher",
        name="cosmos_v1_14B_action_conditioned_teacher",
        node=COSMOS_V1_14B_ACTION_CONDITIONED_NET,
    )
    cs.store(
        group="net_teacher",
        package="model.config.net_teacher",
        name="cosmos_v1_14B_action_chunk_conditioned_teacher",
        node=COSMOS_V1_14B_ACTION_CHUNK_CONDITIONED_NET,
    )
