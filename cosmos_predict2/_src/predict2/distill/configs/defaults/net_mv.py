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

import copy

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_predict2._src.predict2_multiview.networks.multiview_dit import MultiViewDiT

COSMOS_V1_7B_MULTIVIEW_NET: LazyDict = L(MultiViewDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=4096,
    num_blocks=28,
    num_heads=32,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=True,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=2.0,
    sac_config=SACConfig(),
    n_cameras_emb=7,
    view_condition_dim=6,
    concat_view_embedding=True,
    use_wan_fp32_strategy=True,
    layer_mask=None,
)

COSMOS_V1_2B_MULTIVIEW_NET = copy.deepcopy(COSMOS_V1_7B_MULTIVIEW_NET)
COSMOS_V1_2B_MULTIVIEW_NET.model_channels = 2048
COSMOS_V1_2B_MULTIVIEW_NET.num_blocks = 28
COSMOS_V1_2B_MULTIVIEW_NET.num_heads = 16
COSMOS_V1_2B_MULTIVIEW_NET.extra_per_block_abs_pos_emb = False
COSMOS_V1_2B_MULTIVIEW_NET.rope_t_extrapolation_ratio = 1.0

COSMOS_V1_14B_MULTIVIEW_NET = copy.deepcopy(COSMOS_V1_7B_MULTIVIEW_NET)
COSMOS_V1_14B_MULTIVIEW_NET.model_channels = 5120
COSMOS_V1_14B_MULTIVIEW_NET.num_blocks = 36
COSMOS_V1_14B_MULTIVIEW_NET.num_heads = 40
COSMOS_V1_14B_MULTIVIEW_NET.extra_per_block_abs_pos_emb = False
COSMOS_V1_14B_MULTIVIEW_NET.rope_t_extrapolation_ratio = 1.0

mini_net = copy.deepcopy(COSMOS_V1_7B_MULTIVIEW_NET)
mini_net.model_channels = 1024
mini_net.num_heads = 8
mini_net.num_blocks = 2
mini_net.rope_t_extrapolation_ratio = 1.0


def register_net_mv():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="mini_net", node=mini_net)
    cs.store(
        group="net", package="model.config.net", name="cosmos_v1_2B_multiview_student", node=COSMOS_V1_2B_MULTIVIEW_NET
    )
    cs.store(
        group="net", package="model.config.net", name="cosmos_v1_7B_multiview_student", node=COSMOS_V1_7B_MULTIVIEW_NET
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_14B_multiview_student",
        node=COSMOS_V1_14B_MULTIVIEW_NET,
    )

    cs.store(
        group="net_teacher",
        package="model.config.net_teacher",
        name="cosmos_v1_2B_multiview_teacher",
        node=COSMOS_V1_2B_MULTIVIEW_NET,
    )
    cs.store(
        group="net_teacher",
        package="model.config.net_teacher",
        name="cosmos_v1_7B_multiview_teacher",
        node=COSMOS_V1_7B_MULTIVIEW_NET,
    )
    cs.store(
        group="net_teacher",
        package="model.config.net_teacher",
        name="cosmos_v1_14B_multiview_teacher",
        node=COSMOS_V1_14B_MULTIVIEW_NET,
    )

    cs.store(
        group="net_fake_score",
        package="model.config.net_fake_score",
        name="cosmos_v1_2B_multiview_fake_score",
        node=COSMOS_V1_2B_MULTIVIEW_NET,
    )
    cs.store(
        group="net_fake_score",
        package="model.config.net_fake_score",
        name="cosmos_v1_7B_multiview_fake_score",
        node=COSMOS_V1_7B_MULTIVIEW_NET,
    )
    cs.store(
        group="net_fake_score",
        package="model.config.net_fake_score",
        name="cosmos_v1_14B_multiview_fake_score",
        node=COSMOS_V1_14B_MULTIVIEW_NET,
    )
