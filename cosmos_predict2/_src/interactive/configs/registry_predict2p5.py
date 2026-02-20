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

from typing import Any, List

import attrs
from cosmos_oss.checkpoints_predict2 import register_checkpoints as register_checkpoints_predict2

from cosmos_predict2._src.imaginaire import config
from cosmos_predict2._src.imaginaire.utils.config_helper import import_all_modules_from_package
from cosmos_predict2._src.interactive.configs.registry_defaults.callbacks import register_callbacks
from cosmos_predict2._src.interactive.configs.registry_defaults.ckpt_type import register_ckpt_type
from cosmos_predict2._src.interactive.configs.registry_defaults.condition_postprocessor import (
    register_condition_postprocessor,
)
from cosmos_predict2._src.interactive.configs.registry_defaults.discriminator import register_net_discriminator_head
from cosmos_predict2._src.interactive.configs.registry_defaults.model import register_model
from cosmos_predict2._src.interactive.configs.registry_defaults.net import (
    register_net,
    register_net_fake_score,
    register_net_teacher,
)
from cosmos_predict2._src.interactive.configs.registry_defaults.net_ac import (
    register_net_ac,
    register_net_ac_causal,
    register_net_fake_score_ac,
    register_net_teacher_ac,
)
from cosmos_predict2._src.interactive.configs.registry_defaults.tokenizer import register_tokenizer
from cosmos_predict2._src.interactive.trainer.trainer_distillation import ImaginaireDistillationTrainer as Trainer
from cosmos_predict2._src.predict2.action.configs.action_conditioned.conditioner import (
    register_conditioner as register_action_conditioned_conditioner,
)
from cosmos_predict2._src.predict2.action.configs.action_conditioned.data import (
    register_training_and_val_data as register_action_conditioned_data,
)
from cosmos_predict2._src.predict2.configs.common.defaults.checkpoint import register_checkpoint
from cosmos_predict2._src.predict2.configs.common.defaults.ema import register_ema
from cosmos_predict2._src.predict2.configs.common.defaults.optimizer import register_optimizer
from cosmos_predict2._src.predict2.configs.common.defaults.scheduler import register_scheduler
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import register_conditioner


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry_predict2p5.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "mock"},
            {"data_val": "mock"},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"model": "ddp"},
            {"callbacks": "basic"},
            {"net": None},
            {"net_fake_score": None},
            {"net_teacher": None},
            {"net_discriminator_head": None},
            {"conditioner": "video_prediction_conditioner"},
            {"condition_postprocessor": None},
            {"ema": "power"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"checkpoint": "s3"},
            {"ckpt_type": "dcp_distill"},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    c.job.project = "cosmos_interactive"  # this decides the wandb project name
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 100
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    # Call this function to register config groups for advanced overriding. the order follows the default config groups

    register_optimizer()
    register_scheduler()
    register_model()
    register_callbacks()
    register_net()
    register_net_ac_causal()
    register_net_teacher()
    register_net_fake_score()
    register_net_ac()
    register_net_teacher_ac()
    register_net_fake_score_ac()
    register_net_discriminator_head()

    register_ema()
    register_tokenizer()
    register_checkpoint()
    register_checkpoints_predict2()
    register_ckpt_type()

    register_conditioner()
    register_action_conditioned_conditioner()  # Also register action-conditioned conditioner
    register_condition_postprocessor()
    register_action_conditioned_data()  # Also register action-conditioned data

    # experiment config are defined in the experiment folder
    import_all_modules_from_package("cosmos_predict2._src.interactive.configs.registry_experiment", reload=True)
    return c
