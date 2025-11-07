# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

import copy

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.configs.common.defaults.callbacks import (
    BASIC_CALLBACKS,
    SPEED_CALLBACKS,
)
from cosmos_predict2._src.predict2.interactive.callbacks.wandb import (
    WandbCallback as InteractiveWandbCallback,
)
from cosmos_predict2._src.predict2.interactive.callbacks.wandb import (
    WarmupWandbCallback as InteractiveWarmupWandbCallback,
)

_basic_callback = copy.deepcopy(BASIC_CALLBACKS)
# Replace the default wandb logger with the interactive wandb logger
_basic_callback["wandb"] = L(InteractiveWandbCallback)()

_basic_callback_warmup = copy.deepcopy(BASIC_CALLBACKS)
# Replace the default wandb logger with the interactive warmup wandb logger
_basic_callback_warmup["wandb"] = L(InteractiveWarmupWandbCallback)()

WANDB_CALLBACK = dict(
    wandb=L(InteractiveWandbCallback)(
        logging_iter_multipler=1,
        save_logging_iter_multipler=10,
    ),
    wandb_10x=L(InteractiveWandbCallback)(
        logging_iter_multipler=10,
        save_logging_iter_multipler=1,
    ),
)

WANDB_CALLBACK_WARMUP = dict(
    wandb=L(InteractiveWarmupWandbCallback)(
        logging_iter_multipler=1,
        save_logging_iter_multipler=10,
    ),
    wandb_10x=L(InteractiveWarmupWandbCallback)(
        logging_iter_multipler=10,
        save_logging_iter_multipler=1,
    ),
)


def register_callbacks():
    cs = ConfigStore.instance()
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=_basic_callback)
    cs.store(group="callbacks", package="trainer.callbacks", name="wandb", node=WANDB_CALLBACK)
    cs.store(group="callbacks", package="trainer.callbacks", name="basic_warmup", node=_basic_callback_warmup)
    cs.store(group="callbacks", package="trainer.callbacks", name="wandb_warmup", node=WANDB_CALLBACK_WARMUP)
    cs.store(group="callbacks", package="trainer.callbacks", name="cluster_speed", node=SPEED_CALLBACKS)
