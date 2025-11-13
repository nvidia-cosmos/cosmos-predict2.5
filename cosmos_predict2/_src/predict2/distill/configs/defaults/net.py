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
The config args are copied from cosmos_predict2/_src/predict2/configs/video2world/defaults/net.py.
Should keep in sync with the net definition in predict2.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig

# ============2B net============
cosmos_v1_2b_net_args = dict(
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
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=SACConfig(),
)
# only need lvg (video2world) DiTs as it supports text2world too
COSMOS_V1_2B_NET: LazyDict = L(MinimalV1LVGDiT)(
    **cosmos_v1_2b_net_args, atten_backend="minimal_a2a", use_wan_fp32_strategy=False
)


# ============14B net============
cosmos_v1_14b_net_args = copy.deepcopy(cosmos_v1_2b_net_args)
cosmos_v1_14b_net_args["model_channels"] = 5120
cosmos_v1_14b_net_args["num_heads"] = 40
cosmos_v1_14b_net_args["num_blocks"] = 36
cosmos_v1_14b_net_args["extra_per_block_abs_pos_emb"] = False
cosmos_v1_14b_net_args["rope_t_extrapolation_ratio"] = 1.0

COSMOS_V1_14B_NET: LazyDict = L(MinimalV1LVGDiT)(
    **cosmos_v1_14b_net_args, atten_backend="minimal_a2a", use_wan_fp32_strategy=False
)


# ============mini net for debug============
mini_net_args = copy.deepcopy(cosmos_v1_2b_net_args)
mini_net_args["model_channels"] = 1024
mini_net_args["num_heads"] = 8
mini_net_args["num_blocks"] = 2
mini_net_args["rope_t_extrapolation_ratio"] = 1.0

MINI_NET: LazyDict = L(MinimalV1LVGDiT)(**mini_net_args, atten_backend="minimal_a2a", use_wan_fp32_strategy=False)


def register_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="mini_net_student", node=MINI_NET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_2B_student", node=COSMOS_V1_2B_NET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_14B_student", node=COSMOS_V1_14B_NET)


def register_net_fake_score():
    cs = ConfigStore.instance()
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="mini_net_fake_score", node=MINI_NET)
    cs.store(
        group="net_fake_score",
        package="model.config.net_fake_score",
        name="cosmos_v1_2B_fake_score",
        node=COSMOS_V1_2B_NET,
    )
    cs.store(
        group="net_fake_score",
        package="model.config.net_fake_score",
        name="cosmos_v1_14B_fake_score",
        node=COSMOS_V1_14B_NET,
    )


def register_net_teacher():
    cs = ConfigStore.instance()
    cs.store(group="net_teacher", package="model.config.net_teacher", name="mini_net_teacher", node=MINI_NET)
    cs.store(
        group="net_teacher",
        package="model.config.net_teacher",
        name="cosmos_v1_2B_teacher",
        node=COSMOS_V1_2B_NET,
    )
    cs.store(
        group="net_teacher",
        package="model.config.net_teacher",
        name="cosmos_v1_14B_teacher",
        node=COSMOS_V1_14B_NET,
    )
