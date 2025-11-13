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

# Configs for resuming from stage3 training

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path


def _build_no_s3_run(job, local_path: bool = False):
    # If local_path is True, use the local path as the load path
    if local_path:
        load_path = job["checkpoint"]["load_path"]
    else:
        model_url = f"s3://bucket/{job['checkpoint']['load_path']}/model"
        load_path = get_checkpoint_path(model_url)
    defaults = job.get("defaults", [])
    no_s3_run = dict(
        defaults=defaults + ["_self_"] if "_self_" not in defaults else defaults,
        job=dict(
            name=f"{job['job']['name']}_no_s3" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
            wandb_mode="offline",
        ),
        checkpoint=dict(
            save_to_object_store=dict(enabled=False, credentials=""),
            load_from_object_store=dict(enabled=False),
            load_path=load_path,
        ),
        trainer=dict(
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                heart_beat=dict(save_s3=False),
                iter_speed=dict(save_s3=False),
                device_monitor=dict(save_s3=False),
                every_n_sample_reg=dict(save_s3=False),
                every_n_sample_ema=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
    )
    return no_s3_run


"""
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/interactive/configs/config.py -- experiment=cosmos_predict2p5_2B_action_gr00t_gr1_warmup ~dataloader_train.dataloaders
"""

ACTION_GR00T_WARMUP = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame",
            {"override /net": "action_causal_kvcache_cosmos_v1_2B"},
            {"override /model": "action_video2world_self_forcing_warmup_fsdp"},
            {
                "override /callbacks": [
                    "basic_warmup",
                    "wandb_warmup",
                    "cluster_speed",
                ]
            },
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_warmup",
            name="action_gr00t_warmup_base",
        ),
        checkpoint=dict(
            save_iter=100,
        ),
        optimizer=dict(
            lr=3e-5,
        ),
        scheduler=dict(
            warm_up_steps=[0],
            f_min=[1.0],
            f_max=[1.0],
        ),
        trainer=dict(
            max_iter=20000,
            logging_iter=20,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=5000000000000000000,
                ),
                every_n_sample_ema=dict(
                    every_n=5000000000000000000,
                ),
            ),
        ),
        dataloader_train=dict(
            pin_memory=False,
        ),
    ),
    flags={"allow_objects": True},
)

ACTION_GR00T_WARMUP_GR1 = LazyDict(
    dict(
        defaults=[
            f"/experiment/cosmos_predict2p5_2B_action_gr00t_warmup",
            {"override /data_train": "gr00t_gr1_warmup"},
            {"override /data_val": "gr00t_gr1_warmup"},
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_warmup",
            name="gr1",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000014000",
        ),
    ),
    flags={"allow_objects": True},
)

ACTION_GR00T_WARMUP_G1 = LazyDict(
    dict(
        defaults=[
            f"/experiment/cosmos_predict2p5_2B_action_gr00t_warmup",
            {"override /data_train": "gr00t_g1_warmup"},
            {"override /data_val": "gr00t_g1_warmup"},
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_warmup",
            name="g1",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_g1_gear_wild_merged_customized_13frame_full_16nodes/checkpoints/iter_000038000",
        ),
        model=dict(
            config=dict(
                net=dict(action_dim=43),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()

cs.store(
    group="experiment",
    package="_global_",
    name=f"cosmos_predict2p5_2B_action_gr00t_warmup",
    node=ACTION_GR00T_WARMUP,
)
cs.store(
    group="experiment",
    package="_global_",
    name=f"cosmos_predict2p5_2B_action_gr00t_gr1_warmup",
    node=ACTION_GR00T_WARMUP_GR1,
)
cs.store(
    group="experiment",
    package="_global_",
    name=f"cosmos_predict2p5_2B_action_gr00t_g1_warmup",
    node=ACTION_GR00T_WARMUP_G1,
)
cs.store(
    group="experiment",
    package="_global_",
    name=f"cosmos_predict2p5_2B_action_gr00t_gr1_warmup_no_s3",
    node=_build_no_s3_run(ACTION_GR00T_WARMUP_GR1),
)
