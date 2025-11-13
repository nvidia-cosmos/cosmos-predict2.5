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

import functools
import math

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.datasets.cached_replay_dataloader import (
    duplicate_batches_random,
)
from cosmos_predict2._src.predict2.distill.configs.defaults.teacher_model_paths import (
    TEACHER_CKPT_720_T24_CR1PT1_PRETRAINED_RF_RELEASE,
)
from cosmos_predict2._src.predict2.models.video2world_model import HighSigmaStrategy
from cosmos_predict2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy


def deep_update(dst, src):
    """
    Standard update in hydra only goes one level deep. This function goes arbitrarily deep.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def make_experiment(
    name: str,
    dataset_name: str | None = None,
    model: str = "fsdp_video2world_dmd2_distillation_model_trigflow",
    net: str = "cosmos_v1_2B_student",
    net_teacher: str = "cosmos_v1_2B_teacher",
    net_fake_score: str = "cosmos_v1_2B_fake_score",
    net_discriminator_head: str | None = None,
    conditioner: str = "video_prediction_conditioner",
    resolution: str = "720",
    context_parallel_size: int = 4,
    overrides: dict | None = None,
):
    defaults = [
        {"/net_teacher@model.config.net_teacher": net_teacher},
        {"/net_fake_score@model.config.net_fake_score": net_fake_score},
        {"/net_discriminator_head@model.config.net_discriminator_head": net_discriminator_head},
        {
            "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250806_dedup_accumulated_and_high_quality_v3_202505_s3"
        },
        {"override /conditioner": conditioner},
        {"override /ckpt_type": "dcp_distill"},
        {"override /checkpoint": "s3"},
        {"override /tokenizer": "wan2pt1_tokenizer"},
        {"override /optimizer": "fusedadamw"},
        {
            "override /callbacks": [
                "basic",
                "wandb",
                "cluster_speed",
            ]
        },
        {"override /model": model},
        {"override /net": net},
        "_self_",
    ]
    node = dict(
        defaults=defaults,
        job=dict(group="predict2_distill", name=name),
        model_parallel=dict(context_parallel_size=context_parallel_size),
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
            max_iter=200000,
            logging_iter=50,
            callbacks=dict(
                iter_speed=dict(hit_thres=100),
                grad_clip=dict(
                    clip_norm=1.0,
                ),
                every_n_sample_reg=dict(
                    every_n=250,
                    is_image=False,
                    num_samples=5,
                ),
                every_n_sample_ema=dict(
                    every_n=250,
                    is_image=False,
                    num_samples=5,
                ),
            ),
        ),
        model=dict(
            config=dict(
                adjust_video_noise=True,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                conditioner=dict(
                    text=dict(
                        use_empty_string=False,
                    ),
                ),
                dmd=True,
                fd_type=0,
                grad_clip=True,
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
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="predict2_2b_720_aggressive",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
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
                state_t=24,
                student_update_freq=5,
                tangent_warmup=1,
                teacher_load_from=TEACHER_CKPT_720_T24_CR1PT1_PRETRAINED_RF_RELEASE,
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
            dataloaders=dict(
                image_data=dict(
                    ratio=0,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        cache_size=32,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            dataset_name=dataset_name or "cosmos_distillation_high_quality_20250917_video_whole",
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type=None,
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=93,
                            use_native_fps=True,
                            use_original_fps=False,
                            dataset_resolution_type="gt720p",
                        ),
                        use_cache=False,
                        webdataset=True,
                    ),
                    ratio=1,
                ),
            ),
        ),
        upload_reproducible_setup=True,
    )
    if overrides:
        deep_update(node, overrides)
    return LazyDict(node, flags={"allow_objects": True})


dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_T2V = make_experiment(
    name="dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_T2V",
    resolution="720",
    context_parallel_size=4,
    overrides=dict(
        trainer=dict(
            max_iter=100000,
            logging_iter=5,
        ),
        model=dict(
            config=dict(
                conditional_frames_probs={0: 1.0, 1: 0.0, 2: 0.0},
            ),
        ),
    ),
)

dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_T2V_w_discriminator = make_experiment(
    name="dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_T2V_w_discriminator",
    resolution="720",
    net_discriminator_head="discriminator_v1",
    context_parallel_size=4,
    overrides=dict(
        trainer=dict(
            max_iter=100000,
            logging_iter=5,
        ),
        model=dict(
            config=dict(
                conditional_frames_probs={0: 1.0, 1: 0.0, 2: 0.0},
            ),
        ),
    ),
)

dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V = make_experiment(
    name="dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V",
    resolution="720",
    context_parallel_size=4,
    overrides=dict(
        trainer=dict(
            max_iter=100000,
            logging_iter=5,
        ),
        model=dict(
            config=dict(
                conditional_frames_probs={0: 0.5, 1: 0.3, 2: 0.2},
            ),
        ),
    ),
)

dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V_w_discriminator = make_experiment(
    name="dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V_w_discriminator",
    resolution="720",
    net_discriminator_head="discriminator_v1",
    context_parallel_size=4,
    overrides=dict(
        trainer=dict(
            max_iter=100000,
            logging_iter=5,
        ),
        model=dict(
            config=dict(
                conditional_frames_probs={0: 0.5, 1: 0.3, 2: 0.2},
            ),
        ),
    ),
)


cs = ConfigStore.instance()
"""
torchrun --nproc_per_node=4 --master_port=12340 -m scripts.train --config=cosmos_predict2/_src/predict2/distill/configs/registry.py -- experiment=dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V
"""
for _item in [
    dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_T2V,
    dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_T2V_w_discriminator,
    dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V,
    dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V_w_discriminator,
]:
    cs.store(group="experiment", package="_global_", name=f"{_item['job']['name']}", node=_item)
