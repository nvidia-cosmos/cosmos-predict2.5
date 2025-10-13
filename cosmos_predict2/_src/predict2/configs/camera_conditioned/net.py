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
from cosmos_predict2._src.predict2.networks.camera_conditioned_ar_minimal_v1_lvg_dit import (
    CameraConditionedARMinimalV1LVGDiT,
)
from cosmos_predict2._src.predict2.networks.camera_conditioned_minimal_v1_lvg_dit import (
    CameraConditionedMinimalV1LVGDiT,
)
from cosmos_predict2._src.predict2.networks.minimal_v4_dit_camera_conditioned import SACConfig

# ------------------------------------------------------------
# Camera Conditioned
# ------------------------------------------------------------

CAMERA_COSMOS_V1_7B_NET_MININET: LazyDict = L(CameraConditionedMinimalV1LVGDiT)(
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
)
CAMERA_COSMOS_V1_2B_NET_MININET = copy.deepcopy(CAMERA_COSMOS_V1_7B_NET_MININET)
CAMERA_COSMOS_V1_2B_NET_MININET.model_channels = 2048
CAMERA_COSMOS_V1_2B_NET_MININET.num_heads = 16
CAMERA_COSMOS_V1_2B_NET_MININET.num_blocks = 28
CAMERA_COSMOS_V1_2B_NET_MININET.extra_per_block_abs_pos_emb = False
CAMERA_COSMOS_V1_2B_NET_MININET.rope_t_extrapolation_ratio = 1.0


CAMERA_COSMOS_AR_V1_7B_NET_MININET: LazyDict = L(CameraConditionedARMinimalV1LVGDiT)(
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
)
CAMERA_COSMOS_AR_V1_2B_NET_MININET = copy.deepcopy(CAMERA_COSMOS_AR_V1_7B_NET_MININET)
CAMERA_COSMOS_AR_V1_2B_NET_MININET.model_channels = 2048
CAMERA_COSMOS_AR_V1_2B_NET_MININET.num_heads = 16
CAMERA_COSMOS_AR_V1_2B_NET_MININET.num_blocks = 28
CAMERA_COSMOS_AR_V1_2B_NET_MININET.extra_per_block_abs_pos_emb = False
CAMERA_COSMOS_AR_V1_2B_NET_MININET.rope_t_extrapolation_ratio = 1.0


def register_net():
    cs = ConfigStore.instance()

    # ------------------------------------------------------------
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_net_camera_conditioned",
        node=CAMERA_COSMOS_V1_2B_NET_MININET,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_net_camera_conditioned_ar",
        node=CAMERA_COSMOS_AR_V1_2B_NET_MININET,
    )
