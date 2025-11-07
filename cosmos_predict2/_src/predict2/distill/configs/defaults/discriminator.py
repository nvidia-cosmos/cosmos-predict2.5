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
Discriminator head optionally used in DMD2-style distillation.
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.distill.networks.discriminator import DiscriminatorHead

DISCRIMINATOR_HEAD: LazyDict = L(DiscriminatorHead)(
    model_channels=2048,
    num_branches=3,
)

DISCRIMINATOR_HEAD_MINI: LazyDict = L(DiscriminatorHead)(
    model_channels=1024,
    num_branches=1,
)


def register_net_discriminator_head():
    cs = ConfigStore.instance()
    cs.store(
        group="net_discriminator_head",
        package="model.config.net_discriminator_head",
        name="discriminator",
        node=DISCRIMINATOR_HEAD,
    )
    cs.store(
        group="net_discriminator_head",
        package="model.config.net_discriminator_head",
        name="discriminator_mini",
        node=DISCRIMINATOR_HEAD_MINI,
    )
