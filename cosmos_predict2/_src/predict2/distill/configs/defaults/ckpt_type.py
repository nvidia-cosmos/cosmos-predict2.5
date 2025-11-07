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

from typing import Dict

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.checkpointer.dummy import Checkpointer as DummyCheckpointer
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.distill.checkpointer.dcp import DistributedCheckpointer
from cosmos_predict2._src.predict2.distill.checkpointer.dcp_distill import (
    DistributedCheckpointer as DistributedCheckpointer_distill,
)

DUMMY_CHECKPOINTER: Dict[str, str] = L(DummyCheckpointer)()
DISTRIBUTED_CHECKPOINTER: Dict[str, str] = L(DistributedCheckpointer)()
DISTRIBUTED_CHECKPOINTER_DISTILL: Dict[str, str] = L(DistributedCheckpointer_distill)()


def register_ckpt_type():
    cs = ConfigStore.instance()
    cs.store(group="ckpt_type", package="checkpoint.type", name="dummy", node=DUMMY_CHECKPOINTER)
    cs.store(group="ckpt_type", package="checkpoint.type", name="dcp", node=DISTRIBUTED_CHECKPOINTER)
    cs.store(group="ckpt_type", package="checkpoint.type", name="dcp_distill", node=DISTRIBUTED_CHECKPOINTER_DISTILL)
