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
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_predict2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import MinimalV4LVGControlVaceDiT

TRANSFER2_CONTROL2WORLD_NET_2B = L(MinimalV4LVGControlVaceDiT)(
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
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=SACConfig(),
)

TRANSFER2_CONTROL2WORLD_NET_14B = copy.deepcopy(TRANSFER2_CONTROL2WORLD_NET_2B)
TRANSFER2_CONTROL2WORLD_NET_14B.model_channels = 5120
TRANSFER2_CONTROL2WORLD_NET_14B.num_heads = 40
TRANSFER2_CONTROL2WORLD_NET_14B.num_blocks = 36


def register_net():
    cs = ConfigStore.instance()
    cs.store(
        group="net",
        package="model.config.net",
        name="transfer2_control2world_net_2B",
        node=TRANSFER2_CONTROL2WORLD_NET_2B,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="transfer2_control2world_net_14B",
        node=TRANSFER2_CONTROL2WORLD_NET_14B,
    )
