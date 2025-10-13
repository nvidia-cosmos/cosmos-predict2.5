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

from cosmos_predict2._src.imaginaire.config import ObjectStoreConfig
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey, ModelVariant

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(variant=ModelVariant.AUTO_MULTIVIEW)]

SAMPLE_N_VIEWS = 5

predict2_multiview_post_train_waymo = dict(
    defaults=[
        DEFAULT_CHECKPOINT.experiment,
        {"override /data_train": "waymo"},
        {"override /data_val": "waymo"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="multiview",
        name="2b_waymo",
    ),
    checkpoint=dict(
        load_path=DEFAULT_CHECKPOINT.path,
        save_to_object_store=ObjectStoreConfig(enabled=False),
        load_from_object_store=ObjectStoreConfig(enabled=False),
    ),
    model_parallel=dict(context_parallel_size=8),
    trainer=dict(
        logging_iter=100,
        max_iter=2_000,
        callbacks=dict(
            every_n_sample_reg=dict(every_n=500, sample_n_views=SAMPLE_N_VIEWS, save_s3=False),
            every_n_sample_ema=dict(every_n=500, sample_n_views=SAMPLE_N_VIEWS, save_s3=False),
        ),
        straggler_detection=dict(enabled=False),
    ),
    dataloader_train=dict(
        dataloaders=dict(
            alpamayo_1cap=dict(
                ratio=0,
            ),
            alpamayo_allcaps=dict(ratio=0),
        )
    ),
    upload_reproducible_setup=False,
)

experiments = [predict2_multiview_post_train_waymo]

cs = ConfigStore.instance()

for _item in experiments:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
