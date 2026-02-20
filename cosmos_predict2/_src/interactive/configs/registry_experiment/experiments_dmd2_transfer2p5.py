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

"""
Configs for submitting large scale job using ./_submit.py from local workstation.

Recommended usage: the config here serve as a base config for large scale job, don't modify this script; instead, override specific fields in the config
by creating a new experiment in ./experiment_list.py. See examples there for how to add new experiments and how to submit a job.
"""

import math

from hydra.core.config_store import ConfigStore  # type: ignore[import]

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.interactive.configs.registry_defaults.teacher_model_paths import (
    TRANSFER2_DEPTH_TEACHER_CKPT_2B_RELEASE,
    TRANSFER2_EDGE_TEACHER_CKPT_2B_RELEASE,
    TRANSFER2_SEG_TEACHER_CKPT_2B_RELEASE,
    TRANSFER2_VIS_TEACHER_CKPT_2B_RELEASE,
)
from cosmos_predict2._src.interactive.utils.config_helper import deep_update_config_dict
from cosmos_predict2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy


def make_experiment(
    name: str,
    model: str = "fsdp_dmd2_model_trigflow",
    net: str = "cosmos_transfer2p5_net_2B_student",
    net_teacher: str = "cosmos_transfer2p5_net_2B_teacher",
    net_fake_score: str = "cosmos_transfer2p5_net_2B_fake_score",
    net_discriminator_head: str | None = None,
    conditioner: str = "video_prediction_control_conditioner",
    condition_postprocessor: str | None = "control_condition_postprocessor",
    resolution: str = "720",
    cp_size: int = 4,
    fsdp_size: int = 8,
    overrides: dict | None = None,
) -> LazyDict:
    """
    The default net architecture is consistent with the released Transfer2.5 teacher model config:
        vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow
    as defined in cosmos_predict2/_src/transfer2/configs/vid2vid_transfer/experiment/exp_large_scale.py

    Here we add the distillation-related configs to that teacher model config.
    """
    defaults = [
        {"/net_teacher@model.config.net_teacher": net_teacher},
        {"/net_fake_score@model.config.net_fake_score": net_fake_score},
        {"/net_discriminator_head@model.config.net_discriminator_head": net_discriminator_head},
        {"override /data_train": "mock"},
        {"override /conditioner": conditioner},
        {"override /condition_postprocessor": condition_postprocessor},
        {"override /ckpt_type": "dcp_distill"},
        {"override /checkpoint": "s3"},
        {"override /tokenizer": "wan2pt1_tokenizer"},
        {"override /optimizer": "fusedadamw"},
        {
            "override /callbacks": [
                "basic",
                "wandb",
                "cluster_speed",
                "viz_online_sampling_distilled",
            ]
        },
        {"override /model": model},
        {"override /net": net},
        "_self_",
    ]
    node = dict(
        defaults=defaults,
        job=dict(group="cosmos3_interactive", name=name),
        model_parallel=dict(context_parallel_size=cp_size),
        checkpoint=dict(
            save_iter=500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="",
            load_training_state=False,
            strict_resume=True,
        ),
        optimizer=dict(
            lr=1e-6,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        scheduler=dict(
            f_max=[0.99],
            f_min=[0.4],
            warm_up_steps=[100],
            cycle_lengths=[400_000],
        ),
        trainer=dict(
            max_iter=30_000,
            logging_iter=10,
            callbacks=dict(
                iter_speed=dict(hit_thres=100),
                grad_clip=dict(
                    clip_norm=1.0,
                ),
                every_n_sample_reg=dict(
                    every_n=250,
                    is_image=False,
                    num_samples_per_prompt=3,
                ),
                every_n_sample_ema=dict(
                    every_n=250,
                    is_image=False,
                    num_samples_per_prompt=3,
                ),
            ),
        ),
        model=dict(
            config=dict(
                multiply_noise_by_video_len=True,
                conditional_frames_probs={0: 0.6, 1: 0.2, 2: 0.2},
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                        use_empty_string=False,
                    ),
                ),
                condition_postprocessor=dict(
                    hint_keys=["edge"],
                ),
                fsdp_shard_size=fsdp_size,
                grad_clip=True,
                load_teacher_weights=True,
                intermediate_feature_ids=None,
                loss_scale_GAN_discriminator=1.0,
                loss_scale_GAN_generator=1.0,
                loss_scale_fake_score=1.0,
                loss_scale_sid=1.0,  # dmd2 sid loss
                max_num_conditional_frames=2,
                min_num_conditional_frames=0,
                net=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="predict2_2b_720_aggressive",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    use_wan_fp32_strategy=True,
                    vace_block_every_n=7,
                ),
                net_fake_score=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="predict2_2b_720_aggressive",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                    use_wan_fp32_strategy=True,
                    vace_block_every_n=7,
                ),
                net_teacher=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="predict2_2b_720_aggressive",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                    use_wan_fp32_strategy=True,
                    vace_block_every_n=7,
                ),
                optimizer_discriminator_config=dict(
                    lr=2e-7,
                    weight_decay=0.01,
                    betas=(0.9, 0.999),
                ),
                optimizer_fake_score_config=dict(
                    lr=2e-7,
                    weight_decay=0.01,
                    betas=(0.9, 0.999),
                ),
                rectified_flow_loss_weight_uniform=False,
                resolution=resolution,
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
                state_t=24,
                student_update_freq=5,
                warmup_steps=1,
                teacher_load_from=TRANSFER2_EDGE_TEACHER_CKPT_2B_RELEASE,
                teacher_guidance=3,
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                ),
                timestep_shift=5,
            ),
        ),
        dataloader_train=dict(
            num_workers=4,
            dataset=dict(
                resolution="${model.config.resolution}",
                augmentor_name="video_basic_augmentor_v2_with_control",
                video_decoder_name="video_naive_bytes",
                caption_type="t2w_qwen2p5_7b",
                dataset_resolution_type="gt720p",
                embedding_type=None,  # cr1 embedding is computed on the fly
                min_fps_thres=10,
                max_fps_thres=60,
                num_video_frames=93,
                use_native_fps=True,
                control_input_type="edge",
            ),
        ),
        upload_reproducible_setup=True,
    )
    if overrides:
        deep_update_config_dict(node, overrides)
    return LazyDict(node, flags={"allow_objects": True})


####################################
# Create and register experiments #
####################################
dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_edge = make_experiment(
    name="dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_edge",
    overrides=dict(
        model=dict(
            config=dict(
                teacher_load_from=TRANSFER2_EDGE_TEACHER_CKPT_2B_RELEASE,
                condition_postprocessor=dict(
                    hint_keys=["edge"],
                ),
            ),
        ),
        dataloader_train=dict(
            dataset=dict(
                control_input_type="edge",
            ),
        ),
    ),
)

dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_vis = make_experiment(
    name="dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_vis",
    overrides=dict(
        model=dict(
            config=dict(
                teacher_load_from=TRANSFER2_VIS_TEACHER_CKPT_2B_RELEASE,
                condition_postprocessor=dict(
                    hint_keys=["vis"],
                ),
            ),
        ),
        dataloader_train=dict(
            dataset=dict(
                control_input_type="vis",
            ),
        ),
    ),
)
dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_depth = make_experiment(
    name="dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_depth",
    overrides=dict(
        model=dict(
            config=dict(
                teacher_load_from=TRANSFER2_DEPTH_TEACHER_CKPT_2B_RELEASE,
                condition_postprocessor=dict(
                    hint_keys=["depth"],
                ),
            ),
        ),
        dataloader_train=dict(
            dataset=dict(
                control_input_type="depth",
            ),
        ),
    ),
)

dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_seg = make_experiment(
    name="dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_seg",
    overrides=dict(
        model=dict(
            config=dict(
                teacher_load_from=TRANSFER2_SEG_TEACHER_CKPT_2B_RELEASE,
                condition_postprocessor=dict(
                    hint_keys=["seg"],
                ),
            ),
        ),
        dataloader_train=dict(
            dataset=dict(
                control_input_type="seg",
            ),
        ),
    ),
)


cs = ConfigStore.instance()
"""
torchrun --nproc_per_node=8 --master_port=12340 -m scripts.train --config=cosmos_predict2/_src/interactive/configs/registry_transfer2p5.py -- experiment=dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_edge
"""
for _item in [
    dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_edge,
    dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_vis,
    dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_depth,
    dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_seg,
]:
    cs.store(group="experiment", package="_global_", name=f"{_item['job']['name']}", node=_item)
