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

import copy

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.imaginaire.utils.callback import Callback
from cosmos_predict2._src.predict2.configs.common.defaults.callbacks import (
    BASIC_CALLBACKS,
    SPEED_CALLBACKS,
    WANDB_CALLBACK,
)
from cosmos_predict2._src.transfer2.callbacks.every_n_draw_sample import EveryNDrawSample
from cosmos_predict2._src.transfer2.callbacks.frame_loss_log import FrameLossLog
from cosmos_predict2._src.transfer2.callbacks.sigma_loss_analysis_per_frame import SigmaLossAnalysisPerFrame


class LoadBaseModel(Callback):
    def on_train_start(
        self,
        model: distributed.DistributedDataParallel,
        iteration: int = 0,
    ) -> None:
        model.load_base_model()


_basic_callback = copy.deepcopy(BASIC_CALLBACKS)
_basic_callback["frame_loss_log"] = L(FrameLossLog)(
    save_s3="${upload_reproducible_setup}",
    logging_iter_multipler=1,
    save_logging_iter_multipler=10,
)


VIZ_ONLINE_SAMPLING_CALLBACKS = dict(
    every_n_sample_reg=L(EveryNDrawSample)(
        every_n=5000,
        save_s3="${upload_reproducible_setup}",
        is_x0=False,
    ),
    every_n_sample_ema=L(EveryNDrawSample)(
        every_n=5000,
        is_ema=True,
        save_s3="${upload_reproducible_setup}",
        is_x0=False,
    ),
)

LOG_SIGMA_LOSS_CALLBACKS = dict(
    sigma_loss_log=L(SigmaLossAnalysisPerFrame)(
        save_s3="${upload_reproducible_setup}",
        logging_iter_multipler=2,
        logging_viz_iter_multipler=10,
    ),
)

LOAD_BASE_MODEL_CALLBACK = dict(
    load_base_model=L(LoadBaseModel)(),
)


def register_callbacks():
    cs = ConfigStore.instance()
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=_basic_callback)
    cs.store(group="callbacks", package="trainer.callbacks", name="wandb", node=WANDB_CALLBACK)
    cs.store(group="callbacks", package="trainer.callbacks", name="cluster_speed", node=SPEED_CALLBACKS)
    cs.store(
        group="callbacks", package="trainer.callbacks", name="viz_online_sampling", node=VIZ_ONLINE_SAMPLING_CALLBACKS
    )
    cs.store(
        group="callbacks",
        package="trainer.callbacks",
        name="log_sigma_loss",
        node=LOG_SIGMA_LOSS_CALLBACKS,
    )
    cs.store(
        group="callbacks",
        package="trainer.callbacks",
        name="load_base_model_callbacks",
        node=LOAD_BASE_MODEL_CALLBACK,
    )
