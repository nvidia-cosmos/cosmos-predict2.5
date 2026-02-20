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

from typing import Literal

import attrs

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.modules.edm_sde import EDMSDE
from cosmos_predict2._src.interactive.configs.method_configs.config_cosmos2_interactive_base import (
    Cosmos2InteractiveModelConfig,
)
from cosmos_predict2._src.predict2.utils.optim_instantiate import get_base_optimizer

IS_PREPROCESSED_KEY = "is_preprocessed"


@attrs.define(slots=False)
class DMD2Config(Cosmos2InteractiveModelConfig):
    """
    Config for DMD2 model.

    Inherits all base fields from ``Cosmos2InteractiveModelConfig`` and adds
    DMD2-specific knobs (teacher/fake-score/discriminator, optimizers, etc.).
    """

    # ---------------- Distillation / loss scheduling ----------------
    intermediate_feature_ids: list[int] | None = None
    load_teacher_weights: bool = (
        True  # Load teacher ckpt and copy weights into student/fake-score nets (train-time only)
    )
    loss_scale_GAN_discriminator: float = 1.0
    loss_scale_GAN_generator: float = 1.0
    loss_scale_fake_score: float = 1.0
    loss_scale_sid: float = 1.0
    noise_level_parameterization: Literal["trigflow"] = "trigflow"
    student_update_freq: int = 5
    sde_D: LazyDict = L(EDMSDE)(  # same as base predict2 model
        p_mean=0.0,
        p_std=1.6,
        sigma_max=80,
        sigma_min=0.0002,
    )
    teacher_guidance: float = 3.0
    warmup_steps: int = 100  # Number of student updates before alternating with critic updates

    # ---------------- Model architecture / components ----------------
    net_discriminator_head: LazyDict | None = None
    net_fake_score: LazyDict | None = None
    net_teacher: LazyDict | None = None
    teacher_load_from: LazyDict | None = None
    student_load_from: LazyDict | None = None

    # ---------------- Optimizers ----------------
    optimizer_discriminator_config: LazyDict = L(get_base_optimizer)(
        model=None,
        lr=2e-7,
        weight_decay=0.01,
        betas=[0.0, 0.999],
        optim_type="fusedadam",
        eps=1e-8,
        master_weights=True,
        capturable=True,
    )
    optimizer_fake_score_config: LazyDict = L(get_base_optimizer)(
        model=None,
        lr=2e-7,
        weight_decay=0.01,
        betas=[0.0, 0.999],
        optim_type="fusedadam",
        eps=1e-8,
        master_weights=True,
        capturable=True,
    )

    # ---------------- misc ----------------
    vis_debug: bool = False  # Flag for visualizing intermediate results during training
