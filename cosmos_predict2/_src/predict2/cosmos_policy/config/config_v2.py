# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

from typing import Any, List

import attrs

from cosmos_predict2._src.imaginaire import config
from cosmos_predict2._src.imaginaire.utils.config_helper import import_all_modules_from_package

# Cosmos v2
from cosmos_predict2._src.predict2.configs.common.defaults.checkpoint import register_checkpoint
from cosmos_predict2._src.predict2.configs.common.defaults.ckpt_type import register_ckpt_type
from cosmos_predict2._src.predict2.configs.common.defaults.ema import register_ema
from cosmos_predict2._src.predict2.configs.common.defaults.optimizer import register_optimizer
from cosmos_predict2._src.predict2.configs.common.defaults.scheduler import register_scheduler
from cosmos_predict2._src.predict2.configs.common.defaults.tokenizer import register_tokenizer
from cosmos_predict2._src.predict2.configs.video2world.defaults.callbacks import register_callbacks
from cosmos_predict2._src.predict2.configs.video2world.defaults.model import register_model
from cosmos_predict2._src.predict2.configs.video2world.defaults.net import register_net

# Cosmos Policy-specific registrations
from cosmos_predict2._src.predict2.cosmos_policy.config.conditioner.video2world_conditioner import (
    register_conditioner as register_policy_conditioner,
)
from cosmos_predict2._src.predict2.cosmos_policy.config.defaults.model import register_policy_model
from cosmos_predict2._src.predict2.cosmos_policy.config.defaults.tokenizer import register_policy_tokenizer
from cosmos_predict2._src.predict2.cosmos_policy.trainer import CosmosPolicyTrainer as Trainer


@attrs.define(slots=False)
class ConfigV2(config.Config):
    # Here are the default values of config items that will be used unless alternative values are
    # explicitly specified. We copy these as is from other config.py files to prevent runtime errors
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "mock"},
            {"data_val": "mock"},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"model": "policy_fsdp"},
            {"callbacks": "basic"},  # "basic" -> `BASIC_CALLBACKS`
            {"net": None},
            {"conditioner": "video_prediction_conditioner"},
            {"ema": "power"},
            {"tokenizer": "policy_wan2pt1_tokenizer"},
            {"checkpoint": "s3"},
            {"ckpt_type": "dummy"},
            # Apparently this `experiment` attribute must be the last one
            {"experiment": None},
        ]
    )


def make_config_v2():
    # Get default config
    c = ConfigV2(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Set config attributes
    c.job.project = "cosmos_policy"
    c.job.group = "train"
    c.job.name = "${now:%Y-%m-%d}_${now:%H-%M-%S}"
    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 100
    c.trainer.run_validation = True
    c.trainer.callbacks = None
    c.checkpoint = None

    # Register v2 config groups for advanced overriding
    # Similar to cosmos_predict2/_src/predict2/configs/video2world/config.py

    # register_training_and_val_data()
    register_optimizer()
    register_scheduler()
    register_model()
    register_policy_model()  # Register policy models
    register_callbacks()
    register_net()
    register_policy_conditioner()  # Register policy conditioners
    register_ema()
    register_tokenizer()
    register_policy_tokenizer()  # Register policy tokenizer
    register_checkpoint()
    register_ckpt_type()

    # Register Cosmos v2 experiment configs that Cosmos Policy configs depend on
    import_all_modules_from_package("cosmos_predict2._src.predict2.configs.video2world.experiment", reload=True)

    # Register all experiment configs that we need
    import_all_modules_from_package("cosmos_predict2._src.predict2.cosmos_policy.config.experiment", reload=True)

    # Register Cosmos Policy experiment configs
    from cosmos_predict2._src.predict2.cosmos_policy.config.experiment.cosmos_policy_experiment_configs import (
        register_configs as rc,
    )

    rc()
    from cosmos_predict2._src.predict2.cosmos_policy.config.callbacks import register_configs as rc

    rc()

    # Register mock data configs
    # This is a way to bypass the call to register_training_and_val_data() above, which is very slow
    from hydra.core.config_store import ConfigStore

    from cosmos_predict2._src.predict2.configs.common.mock_data import MOCK_DATA_INTERLEAVE_CONFIG

    cs = ConfigStore.instance()
    cs.store(group="data_train", package="dataloader_train", name="mock", node=MOCK_DATA_INTERLEAVE_CONFIG)
    cs.store(group="data_val", package="dataloader_val", name="mock", node=MOCK_DATA_INTERLEAVE_CONFIG)

    return c
