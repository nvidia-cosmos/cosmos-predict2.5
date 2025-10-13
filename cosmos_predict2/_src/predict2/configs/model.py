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
from cosmos_predict2._src.predict2.models.action_conditioned_video2world_model import (
    ActionConditionedVideo2WorldConfig,
    ActionConditionedVideo2WorldModel,
)
from cosmos_predict2._src.predict2.models.action_conditioned_video2world_rectified_flow_model import (
    ActionVideo2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)

# EDM model
DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(ActionConditionedVideo2WorldModel)(
        config=ActionConditionedVideo2WorldConfig(),
        _recursive_=False,
    ),
)

FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ActionConditionedVideo2WorldModel)(
        config=ActionConditionedVideo2WorldConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

# rectified flow model
FSDP_RECTIFIED_FLOW_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ActionVideo2WorldModelRectifiedFlow)(
        config=Video2WorldModelRectifiedFlowConfig(
            fsdp_shard_size=8,
            state_t=24,
        ),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="action_conditioned_video2world_ddp", node=DDP_CONFIG)
    cs.store(group="model", package="_global_", name="action_conditioned_video2world_fsdp", node=FSDP_CONFIG)
    cs.store(
        group="model",
        package="_global_",
        name="action_conditioned_video2world_fsdp_rectified_flow",
        node=FSDP_RECTIFIED_FLOW_CONFIG,
    )
