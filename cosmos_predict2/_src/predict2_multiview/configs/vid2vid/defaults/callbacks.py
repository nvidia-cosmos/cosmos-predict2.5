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
from cosmos_predict2._src.predict2_multiview.callbacks.log_weight import LogWeight
from cosmos_predict2._src.predict2_multiview.callbacks.sigma_loss_analysis_per_frame import SigmaLossAnalysisPerFrame

LOG_SIGMA_LOSS_CALLBACKS = dict(
    sigma_loss_log=L(SigmaLossAnalysisPerFrame)(
        save_s3="${upload_reproducible_setup}",
        logging_iter_multipler=2,
        logging_viz_iter_multipler=10,
    ),
)

LOG_WEIGHT_CALLBACKS = dict(
    log_weight=L(LogWeight)(
        every_n=100,
    ),
)


def register_callbacks():
    cs = ConfigStore.instance()
    cs.store(
        group="callbacks",
        package="trainer.callbacks",
        name="log_sigma_loss",
        node=LOG_SIGMA_LOSS_CALLBACKS,
    )
    cs.store(
        group="callbacks",
        package="trainer.callbacks",
        name="log_weight",
        node=LOG_WEIGHT_CALLBACKS,
    )
