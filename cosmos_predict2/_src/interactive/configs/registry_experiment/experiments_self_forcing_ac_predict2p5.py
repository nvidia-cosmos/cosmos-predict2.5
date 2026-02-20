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

import math

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.interactive.configs.registry_defaults.teacher_model_paths import (
    ACTION_CONDITIONED_TEACHER_CKPT_2B_GROOT1,
)
from cosmos_predict2._src.interactive.utils.config_helper import deep_update_config_dict
from cosmos_predict2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy


def make_experiment(
    name: str,
    data: str,
    model: str = "action_video2world_self_forcing_fsdp",
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
        {"override /tokenizer": tokenizer},
        {"override /net_teacher": "cosmos_v1_2B_action_chunk_conditioned_teacher"},
        {"override /net_fake_score": "cosmos_v1_2B_action_chunk_conditioned_fake_score"},
        {"override /conditioner": conditioner},
        {"override /ckpt_type": "dcp_distill"},
        {"override /optimizer": "fusedadamw"},
        {"override /callbacks": ["basic", "wandb", "cluster_speed"]},
        {"override /checkpoint": "s3"},
        "_self_",
    ]
    node = dict(
        defaults=defaults,
        job=dict(
            group="self_forcing",
            name=name,
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        optimizer=dict(
            lr=1e-7,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            # `fusedadamw` defaults to master_weights=True (FP32 master copy), which is very memory-expensive
            # for 2B-scale nets and can trigger OOM once optimizer state is first materialized.
            master_weights=False,
        ),
        scheduler=dict(
            f_max=[1.0],
            f_min=[1.0],
            warm_up_steps=[0],
            cycle_lengths=[400_000],
        ),
        model=dict(
            config=dict(
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.0,
                        use_empty_string=False,
                    ),
                ),
                grad_clip=True,
                init_student_with_teacher=True,
                intermediate_feature_ids=None,
                loss_scale_GAN_discriminator=1.0,
                loss_scale_GAN_generator=1.0,
                loss_scale_fake_score=1.0,
                loss_scale_sid=1.0,
                max_num_conditional_frames=2,
                min_num_conditional_frames=0,
                net=dict(
                    action_dim=29,
                    temporal_compression_ratio=4,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=1.0,
                    sac_config=dict(mode="mm_only"),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                net_fake_score=dict(
                    action_dim=29,
                    temporal_compression_ratio=4,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=1.0,
                    sac_config=dict(mode="mm_only"),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                net_teacher=dict(
                    action_dim=29,
                    temporal_compression_ratio=4,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=1.0,
                    sac_config=dict(mode="mm_only"),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                optimizer_discriminator_config=dict(
                    lr=1e-5,
                    weight_decay=0.01,
                    betas=(0.9, 0.999),
                    master_weights=False,
                ),
                optimizer_fake_score_config=dict(
                    lr=1e-5,
                    weight_decay=0.01,
                    betas=(0.9, 0.999),
                    # Avoid allocating FP32 master weights for the fake-score optimizer (big memory spike after first step).
                    master_weights=False,
                ),
                resolution="720",
                resize_online=True,
                scaling="rectified_flow",
                sde=dict(
                    p_mean=-0.8,
                    p_std=1.6,
                    sigma_max=80,
                    sigma_min=0.0002,
                ),
                sde_D=dict(
                    p_mean=0.0,
                    p_std=1.6,
                    sigma_max=80,
                    sigma_min=0.0002,
                ),
                selected_sampling_time=[math.pi / 2, math.atan(15), math.atan(5), math.atan(5 / 3)],
                sigma_conditional=0.0001,
                sigma_data=1.0,
                state_t=1 + 12 // 4,
                student_update_freq=5,
                warmup_steps=1,
                teacher_load_from=ACTION_CONDITIONED_TEACHER_CKPT_2B_GROOT1,
                teacher_guidance=0.0,
                text_encoder_class="reason1p1_7B",
                # Enable generating a decoded video during training so the interactive
                # W&B callback can log `train/backward_simulation_video`.
                vis_debug=True,
                vis_debug_every_n=100,
                text_encoder_config=dict(
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=False,
                ),
                timestep_shift=5,
            ),
        ),
        checkpoint=dict(
            save_iter=100,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            max_iter=5000,
            logging_iter=20,
            callbacks=dict(
                iter_speed=dict(hit_thres=200),
                grad_clip=dict(
                    clip_norm=1.0,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=1,
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

ACTION_GR00T_GR1_SELF_FORCING = make_experiment(
    name="gr1_action2p5",
    data="gr00t_gr1_knowledge_distill",
    overrides=dict(
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_action_conditioned/interactive_knowledge_distill/gr1_action2p5/checkpoints/iter_000005000",
        ),
        model=dict(
            config=dict(
                teacher_load_from=ACTION_CONDITIONED_TEACHER_CKPT_2B_GROOT1,
            ),
        ),
    ),
)

cs = ConfigStore.instance()

cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_gr1_self_forcing",
    node=ACTION_GR00T_GR1_SELF_FORCING,
)
