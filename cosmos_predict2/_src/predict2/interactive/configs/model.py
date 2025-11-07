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
from cosmos_predict2._src.predict2.interactive.models.action_video2world_self_forcing import (
    ActionVideo2WorldModelTrigflowSelfForcingDMD2,
    ActionVideo2WorldModelTrigflowSelfForcingDMD2Config,
)
from cosmos_predict2._src.predict2.interactive.models.action_video2world_warmup import (
    ActionConditionedSFWarmupModelRF,
    ActionConditionedSFWarmupModelRFConfig,
)

FSDP_TRAINER = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
)


def make_fsdp_model(model_cls, config_cls, **config_overrides):
    cfg = config_cls(
        fsdp_shard_size=8,
        **config_overrides,
    )
    return dict(
        **FSDP_TRAINER,
        model=L(model_cls)(
            config=cfg,
            _recursive_=False,
        ),
    )


ACTION_CONDITIONED_MODEL_FSDP_RECTIFIED_FLOW_SF_WARMUP_CONFIG = make_fsdp_model(
    ActionConditionedSFWarmupModelRF,
    ActionConditionedSFWarmupModelRFConfig,
    min_num_conditional_frames=0,
    max_num_conditional_frames=0,
)

ACTION_VIDEO2WORLD_TRIGFLOW_RF_SELF_FORCING_DMD2_FSDP_CONFIG = make_fsdp_model(
    ActionVideo2WorldModelTrigflowSelfForcingDMD2,
    ActionVideo2WorldModelTrigflowSelfForcingDMD2Config,
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="action_video2world_self_forcing_warmup_fsdp",
        node=ACTION_CONDITIONED_MODEL_FSDP_RECTIFIED_FLOW_SF_WARMUP_CONFIG,
    )
    cs.store(
        group="model",
        package="_global_",
        name="action_video2world_self_forcing_fsdp",
        node=ACTION_VIDEO2WORLD_TRIGFLOW_RF_SELF_FORCING_DMD2_FSDP_CONFIG,
    )
