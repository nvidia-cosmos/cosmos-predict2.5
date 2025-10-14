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
from cosmos_predict2._src.predict2.camera.models.camera_conditioned_ar_video2world_model_rectified_flow import (
    CameraConditionedARVideo2WorldModelRectifiedFlow,
    CameraConditionedARVideo2WorldRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2.camera.models.camera_conditioned_frameinit_video2world_model_rectified_flow import (
    CameraConditionedFrameinitVideo2WorldModelRectifiedFlow,
    CameraConditionedFrameinitVideo2WorldRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2.camera.models.camera_conditioned_video2world_model import (
    CameraConditionedVideo2WorldConfig,
    CameraConditionedVideo2WorldModel,
)
from cosmos_predict2._src.predict2.camera.models.camera_conditioned_video2world_model_rectified_flow import (
    CameraConditionedVideo2WorldModelRectifiedFlow,
    CameraConditionedVideo2WorldRectifiedFlowConfig,
)

CAMERA_CONDITIONED_FSDP_HIGH_SIGMA_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CameraConditionedVideo2WorldModel)(
        config=CameraConditionedVideo2WorldConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

CAMERA_CONDITIONED_FSDP_RECTIFIED_FLOW_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CameraConditionedVideo2WorldModelRectifiedFlow)(
        config=CameraConditionedVideo2WorldRectifiedFlowConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

CAMERA_CONDITIONED_FRAMEINIT_FSDP_RECTIFIED_FLOW_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CameraConditionedFrameinitVideo2WorldModelRectifiedFlow)(
        config=CameraConditionedFrameinitVideo2WorldRectifiedFlowConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

CAMERA_CONDITIONED_AR_FSDP_RECTIFIED_FLOW_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CameraConditionedARVideo2WorldModelRectifiedFlow)(
        config=CameraConditionedARVideo2WorldRectifiedFlowConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="camera_conditioned_fsdp",
        node=CAMERA_CONDITIONED_FSDP_HIGH_SIGMA_CONFIG,
    )
    cs.store(
        group="model",
        package="_global_",
        name="camera_conditioned_rectified_flow_fsdp",
        node=CAMERA_CONDITIONED_FSDP_RECTIFIED_FLOW_CONFIG,
    )
    cs.store(
        group="model",
        package="_global_",
        name="camera_conditioned_frameinit_rectified_flow_fsdp",
        node=CAMERA_CONDITIONED_FRAMEINIT_FSDP_RECTIFIED_FLOW_CONFIG,
    )
    cs.store(
        group="model",
        package="_global_",
        name="camera_conditioned_ar_rectified_flow_fsdp",
        node=CAMERA_CONDITIONED_AR_FSDP_RECTIFIED_FLOW_CONFIG,
    )
