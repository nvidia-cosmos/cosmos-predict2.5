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
from cosmos_predict2._src.interactive.configs.registry_defaults.teacher_model_paths import (
    ACTION_CONDITIONED_TEACHER_CKPT_2B_GROOT1,
)
from cosmos_predict2._src.interactive.utils.config_helper import deep_update_config_dict


def make_experiment(
    name: str,
    data: str,
    model: str = "action_video2world_knowledge_distill_fsdp",
    net: str = "cosmos_v1_2B_causal_action_chunk_conditioned_student",
    conditioner: str = "action_conditioned_video_conditioner",
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
        {"override /ckpt_type": "dcp_distill"},
        {"override /checkpoint": "s3"},
        {"override /optimizer": "fusedadamw"},
        {"override /callbacks": ["basic", "wandb", "cluster_speed"]},
        "_self_",
    ]
    node = dict(
        defaults=defaults,
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_knowledge_distill",
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
            lr=1e-5,
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
    name="gr1_action2p5",
    data="gr00t_gr1_knowledge_distill",
    overrides=dict(
        model=dict(
            config=dict(
                teacher_load_from=ACTION_CONDITIONED_TEACHER_CKPT_2B_GROOT1,
            ),
        ),
    ),
)

"""
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/interactive/configs/registry_interactive2p5.py -- experiment=cosmos_predict2p5_2B_action_gr00t_gr1_knowledge_distill
"""

cs = ConfigStore.instance()

cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_gr1_knowledge_distill",
    node=ACTION_GR00T_WARMUP_GR1,
)
