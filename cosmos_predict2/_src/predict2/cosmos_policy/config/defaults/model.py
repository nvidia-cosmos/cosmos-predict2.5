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
Cosmos Policy model registration for Hydra ConfigStore.

This registers policy-specific model classes that extend the base predict2 models.
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.cosmos_policy.models.policy_video2world_model import (
    CosmosPolicyVideo2WorldConfig,
    CosmosPolicyVideo2WorldModel,
)
from cosmos_predict2._src.predict2.cosmos_policy.models.policy_video2world_model_rectified_flow import (
    CosmosPolicyVideo2WorldConfigRectifiedFlow,
    CosmosPolicyVideo2WorldModelRectifiedFlow,
)

# Use policy-specific models with the same config structure
POLICY_DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(CosmosPolicyVideo2WorldModel)(
        config=CosmosPolicyVideo2WorldConfig(),
        _recursive_=False,
    ),
)

POLICY_FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CosmosPolicyVideo2WorldModel)(
        config=CosmosPolicyVideo2WorldConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

# Rectified Flow model configs
POLICY_RECTIFIED_FLOW_DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
        config=CosmosPolicyVideo2WorldConfigRectifiedFlow(),
        _recursive_=False,
    ),
)

POLICY_RECTIFIED_FLOW_FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
        config=CosmosPolicyVideo2WorldConfigRectifiedFlow(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


def register_policy_model():
    """Register Cosmos Policy model configurations."""
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="policy_ddp", node=POLICY_DDP_CONFIG)
    cs.store(group="model", package="_global_", name="policy_fsdp", node=POLICY_FSDP_CONFIG)
    # Rectified Flow models
    cs.store(group="model", package="_global_", name="policy_rectified_flow_ddp", node=POLICY_RECTIFIED_FLOW_DDP_CONFIG)
    cs.store(
        group="model", package="_global_", name="policy_rectified_flow_fsdp", node=POLICY_RECTIFIED_FLOW_FSDP_CONFIG
    )
