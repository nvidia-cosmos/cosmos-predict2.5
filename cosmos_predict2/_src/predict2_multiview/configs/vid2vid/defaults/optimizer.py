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

from cosmos_predict2._src.imaginaire.lazy_config import PLACEHOLDER, LazyDict
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.utils.optim_instantiate import get_base_optimizer
from cosmos_predict2._src.predict2_multiview.utils.optim_instantiate import get_multiple_optimizer

AdamWConfig = L(get_base_optimizer)(
    model=PLACEHOLDER,
    lr=1e-4,
    weight_decay=0.1,
    betas=[0.9, 0.99],
    optim_type="adamw",
    eps=1e-8,
    fused=True,
)

FusedAdamWConfig: LazyDict = L(get_base_optimizer)(
    model=PLACEHOLDER,
    lr=1e-4,
    weight_decay=0.1,
    betas=[0.9, 0.99],
    optim_type="fusedadam",
    eps=1e-8,
    master_weights=True,
    capturable=True,
)

MultipleAdamWConfig = L(get_multiple_optimizer)(
    model=PLACEHOLDER,
    lr=1e-4,
    weight_decay=1e-3,
    betas=[0.9, 0.999],
    optim_type="adamw",
    eps=1e-8,
    fused=True,
    lr_overrides={},
)


MultipleFusedAdamWConfig = L(get_multiple_optimizer)(
    model=PLACEHOLDER, lr=1e-4, weight_decay=1e-3, betas=[0.9, 0.999], optim_type="fusedadam", eps=1e-8, lr_overrides={}
)


def register_optimizer():
    cs = ConfigStore.instance()
    cs.store(group="optimizer", package="optimizer", name="fusedadamw", node=FusedAdamWConfig)
    cs.store(group="optimizer", package="optimizer", name="adamw", node=AdamWConfig)
    cs.store(group="optimizer", package="optimizer", name="multipleadamw", node=MultipleAdamWConfig)
    cs.store(group="optimizer", package="optimizer", name="multiplefusedadamw", node=MultipleFusedAdamWConfig)
