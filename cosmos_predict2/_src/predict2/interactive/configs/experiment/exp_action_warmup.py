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

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.distill.utils.config_helper import build_no_s3_run, deep_update_config_dict


def make_experiment(
    name: str,
    data: str,
    model: str = "action_video2world_warmup_fsdp",
    net: str = "action_causal_kvcache_cosmos_v1_2B",
    conditioner: str = "video_action_conditioner",
    tokenizer: str = "wan2pt1_tokenizer",
    overrides: dict | None = None,
) -> LazyDict:
    defaults = [
        {"override /data_train": data},
        {"override /data_val": data},
        {"override /model": model},
        {"override /net": net},
        {"override /conditioner": conditioner},
        {"override /tokenizer": tokenizer},
        {"override /ckpt_type": "dcp"},
        {"override /checkpoint": "s3"},
        {"override /callbacks": ["basic_warmup", "wandb_warmup", "cluster_speed"]},
        "_self_",
    ]
    node = dict(
        defaults=defaults,
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_warmup",
            name=name,
        ),
        checkpoint=dict(
            save_iter=100,
            save_to_object_store=dict(enabled=True),
            load_from_object_store=dict(enabled=True),
            load_training_state=False,
            strict_resume=True,
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        optimizer=dict(
            lr=3e-5,
            weight_decay=0.1,
            betas=[0.9, 0.999],
        ),
        scheduler=dict(
            warm_up_steps=[0],
            f_min=[1.0],
            f_max=[1.0],
        ),
        model=dict(
            config=dict(
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=29,
                    num_action_per_chunk=12,
                    timestep_scale=0.001,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                resolution=720,
            ),
        ),
        trainer=dict(
            max_iter=20000,
            logging_iter=20,
            callbacks=dict(
                grad_clip=dict(
                    clip_norm=0.1,
                ),
                manual_gc=dict(
                    every_n=200,
                ),
                every_n_sample_reg=dict(
                    every_n=5000000000000000000,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                ),
                every_n_sample_ema=dict(
                    every_n=5000000000000000000,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=4,
            pin_memory=False,
        ),
        upload_reproducible_setup=True,
    )
    if overrides:
        deep_update_config_dict(node, overrides)
    return LazyDict(node, flags={"allow_objects": True})


####################################
# Create and register experiments #
####################################

ACTION_GR00T_WARMUP_GR1 = make_experiment(
    name="gr1",
    data="gr00t_gr1_warmup",
    overrides=dict(
        checkpoint=dict(
            load_path="cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000014000",
        ),
    ),
)

ACTION_GR00T_WARMUP_G1 = make_experiment(
    name="g1",
    data="gr00t_g1_warmup",
    overrides=dict(
        checkpoint=dict(
            load_path="cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_g1_gear_wild_merged_customized_13frame_full_16nodes/checkpoints/iter_000038000",
        ),
        model=dict(
            config=dict(
                net=dict(action_dim=43),
            ),
        ),
    ),
)

"""
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/interactive/configs/config_warmup.py -- experiment=cosmos_predict2p5_2B_action_gr00t_gr1_warmup ~dataloader_train.dataloaders
"""

cs = ConfigStore.instance()

cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_gr1_warmup",
    node=ACTION_GR00T_WARMUP_GR1,
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_g1_warmup",
    node=ACTION_GR00T_WARMUP_G1,
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_gr1_warmup_no_s3",
    node=build_no_s3_run(ACTION_GR00T_WARMUP_GR1),
)
