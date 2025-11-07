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
from cosmos_predict2._src.predict2.interactive.configs.experiment.exp_action_self_forcing_warmup import _build_no_s3_run
from cosmos_predict2._src.predict2.models.video2world_model import HighSigmaStrategy
from cosmos_predict2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy

ACTION_GR00T_SELF_FORCING = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "gr00t_gr1_warmup"},
            {"override /data_val": "gr00t_gr1_warmup"},
            {"override /net": "action_causal_kvcache_cosmos_v1_2B"},
            {"override /net_teacher": "cosmos_v1_2B_action_chunk_conditioned"},
            {"override /net_fake_score": "cosmos_v1_2B_action_chunk_conditioned"},
            {"override /conditioner": "video_action_conditioner"},
            {"override /model": "action_video2world_self_forcing_fsdp"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "wandb",
                    "cluster_speed",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="self_forcing_action",
            name="default",
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        optimizer=dict(
            lr=1e-7,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        scheduler=dict(
            f_max=[1.0],
            f_min=[1.0],
            warm_up_steps=[0],
            cycle_lengths=[400_000],
        ),
        model=dict(
            config=dict(
                adjust_video_noise=True,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.0,
                        use_empty_string=False,
                    ),
                ),
                dmd=True,
                fd_type=0,
                grad_clip=True,  # todo (qianlim): try False
                high_sigma_ratio=0.05,
                high_sigma_strategy=str(HighSigmaStrategy.NONE),
                init_student_with_teacher=True,
                intermediate_feature_ids=[11, 19, 27],
                loss_scale=0.0,  # scm loss
                loss_scale_GAN_discriminator=1.0,
                loss_scale_GAN_generator=1.0,
                loss_scale_fake_score=1.0,
                loss_scale_sid=1.0,  # dmd2 sid loss
                max_num_conditional_frames=2,
                max_simulation_steps=1,
                max_simulation_steps_fake=4,
                min_num_conditional_frames=0,
                net=dict(
                    action_dim=29,
                    temporal_compression_ratio=4,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="none",
                    ),
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
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="none",
                    ),
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
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="none",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                optimizer_discriminator_config=dict(
                    lr=1e-5,
                    weight_decay=0.01,
                    betas=(0.9, 0.999),
                ),
                optimizer_fake_score_config=dict(
                    lr=1e-5,
                    weight_decay=0.01,
                    betas=(0.9, 0.999),
                ),
                rectified_flow_loss_weight_uniform=False,
                resolution="720",
                resize_online=True,
                scaling="rectified_flow",  # correct loss weight for rectified flow
                sde=dict(
                    p_mean=-0.8,
                    p_std=1.6,
                    sigma_max=80,  # 200
                    sigma_min=0.0002,  # 0.01
                ),
                sde_D=dict(
                    p_mean=0.0,
                    p_std=1.6,
                    sigma_max=80,
                    sigma_min=0.0002,
                ),
                selected_sampling_time=[math.pi / 2, math.atan(15), math.atan(5), math.atan(5 / 3)],
                sigma_conditional=0.0001,  # Noise level used for conditional frames
                sigma_data=1.0,
                state_t=1 + 12 // 4,
                student_update_freq=5,
                tangent_warmup=1,
                teacher_load_from=dict(
                    load_path="s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000014000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                teacher_guidance=0.0,
                text_encoder_class="reason1p1_7B",
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
            max_iter=1000,
            logging_iter=50,
            callbacks=dict(
                iter_speed=dict(hit_thres=200),
                grad_clip=dict(
                    clip_norm=1.0,
                ),
                every_n_sample_reg=dict(
                    every_n=25000000000000000000,
                    is_image=False,
                    num_samples=5,
                ),
                every_n_sample_ema=dict(
                    every_n=25000000000000000000,
                    is_image=False,
                    num_samples=5,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=1,
            pin_memory=False,
        ),
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)


ACTION_GR00T_GR1_SELF_FORCING = LazyDict(
    dict(
        defaults=[
            f"/experiment/cosmos_predict2p5_2B_action_gr00t_self_forcing",
            {"override /data_train": "gr00t_gr1_warmup"},
            {"override /data_val": "gr00t_gr1_warmup"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
            name="gr1",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_action_conditioned/interactive_warmup/gr1/checkpoints/iter_000020000",
        ),
        model=dict(
            config=dict(
                teacher_load_from=dict(
                    load_path="s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000014000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

ACTION_GR00T_G1_SELF_FORCING = LazyDict(
    dict(
        defaults=[
            f"/experiment/cosmos_predict2p5_2B_action_gr00t_self_forcing",
            {"override /data_train": "gr00t_g1_warmup"},
            {"override /data_val": "gr00t_g1_warmup"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
            name="g1",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_action_conditioned/interactive_warmup/g1/checkpoints/iter_000020000",
        ),
        model=dict(
            config=dict(
                net=dict(action_dim=43),
                net_fake_score=dict(action_dim=43),
                net_teacher=dict(action_dim=43),
                teacher_load_from=dict(
                    load_path="s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_g1_gear_wild_merged_customized_13frame_full_16nodes/checkpoints/iter_000038000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()

cs.store(
    group="experiment",
    package="_global_",
    name=f"cosmos_predict2p5_2B_action_gr00t_self_forcing",
    node=ACTION_GR00T_SELF_FORCING,
)
cs.store(
    group="experiment",
    package="_global_",
    name=f"cosmos_predict2p5_2B_action_gr00t_gr1_self_forcing",
    node=ACTION_GR00T_GR1_SELF_FORCING,
)
cs.store(
    group="experiment",
    package="_global_",
    name=f"cosmos_predict2p5_2B_action_gr00t_g1_self_forcing",
    node=ACTION_GR00T_G1_SELF_FORCING,
)
cs.store(
    group="experiment",
    package="_global_",
    name=f"cosmos_predict2p5_2B_action_gr00t_gr1_self_forcing_no_s3",
    node=_build_no_s3_run(ACTION_GR00T_GR1_SELF_FORCING),
)
