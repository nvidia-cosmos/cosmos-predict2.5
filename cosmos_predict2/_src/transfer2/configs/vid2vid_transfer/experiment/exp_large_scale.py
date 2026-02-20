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

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.datasets.cached_replay_dataloader import duplicate_batches_random
from cosmos_predict2._src.predict2.models.video2world_model import HighSigmaStrategy
from cosmos_predict2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from cosmos_predict2._src.reason1.configs.default.model_config_qwen import QwenModelConfig, QwenVisionConfig
from cosmos_predict2._src.reason1.models.vlm_qwen_omni import QwenVLBaseModel
from cosmos_predict2._src.reason1.tokenizer.processor import build_tokenizer

BASE_CKPT_480p_T20 = dict(
    load_path="bucket/cosmos_diffusion_v2/vid2vid_control_base/Stage-c_pt_4-Index-10-Size-2B-Res-480-Fps-16-Note-06_02_TnI2V_data_all_hq/checkpoints/iter_000060000",
    credentials="credentials/s3_checkpoint.secret",
)
BASE_CKPT_720_T20 = dict(
    load_path="bucket/cosmos_diffusion_v2/vid2vid_control_base/Stage-c_pt_4-Index-11-Size-2B-Res-720-Fps-16-Note-06_02_TnI2V_data_all_hq/checkpoints/iter_000050000",
    credentials="credentials/s3_checkpoint.secret",
)
BASE_CKPT_720_T24 = dict(
    load_path="bucket/cosmos_diffusion_v2/vid2vid_control_base/Stage-c_pt_4-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_TnI2V/checkpoints/iter_000040000",
    credentials="credentials/s3_checkpoint.secret",
)
BASE_CKPT_720_T24_CR1 = dict(
    load_path="bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000005000",
    credentials="credentials/s3_checkpoint.secret",
)
BASE_CKPT_720_T24_CR1PT1 = dict(
    load_path="bucket/cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000",
    credentials="credentials/s3_checkpoint.secret",
)
BASE_CKPT_720_T24_CR1PT1_RECTIFIED_FLOW = dict(
    load_path="bucket/cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only/checkpoints/iter_000037000",
    credentials="credentials/s3_checkpoint.secret",
)
BASE_CKPT_T2I = dict(
    load_path="bucket/cosmos_diffusion_v2/ablation_2B_0502_t2i/ablation_2B_0502_t2i_426_1024res_hq_tuning_flux_lora_general_real_synth_5_3_1_highlr_0p2/checkpoints/iter_000030000/",
    credentials="credentials/s3_checkpoint.secret",
)
BASE_CKPT_14B_720_T24_CR1PT1_RECTIFIED_FLOW = dict(
    load_path="bucket/cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma/checkpoints/iter_000012500",
    credentials="credentials/s3_checkpoint.secret",
)

_TRAINER_DEBUG_CONFIG = dict(
    max_iter=25,
    logging_iter=2,
    callbacks=dict(
        every_n_sample_reg=dict(
            every_n=5,
        ),
        every_n_sample_ema=dict(
            every_n=100,
        ),
        reg_model_image2video_sora_val_sampling=dict(
            every_n=13,
            is_debug=True,
            latent_video_length="${model.config.state_t}",
        ),
        ema_model_image2video_sora_val_sampling=dict(
            every_n=13,
            is_debug=True,
            latent_video_length="${model.config.state_t}",
        ),
        reg_model_image2video_vbench_val_sampling=dict(
            every_n=13,
            is_debug=True,
            latent_video_length="${model.config.state_t}",
        ),
        ema_model_image2video_vbench_val_sampling=dict(
            every_n=13,
            is_debug=True,
            latent_video_length="${model.config.state_t}",
        ),
    ),
)
_CKPT_DEBUG_CONFIG = dict(
    save_iter=100,
    load_training_state=False,
    strict_resume=False,
)


def build_debug_runs(job):
    """
    Will take a config dict and create another config with "_WO_RESUME" suffix
    """
    debug = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_DEBUG" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=_TRAINER_DEBUG_CONFIG,
        checkpoint=_CKPT_DEBUG_CONFIG,
    )

    wo_resume = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_WO_RESUME" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=_TRAINER_DEBUG_CONFIG,
        checkpoint=_CKPT_DEBUG_CONFIG,
        model=dict(
            config=dict(
                base_load_from=None,
            ),
        ),
        dataloader_train=dict(
            num_workers=0,
            prefetch_factor=None,
        ),
    )

    mock_wo_resume = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_MOCK_WO_RESUME" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=_TRAINER_DEBUG_CONFIG,
        checkpoint=_CKPT_DEBUG_CONFIG,
        model=dict(
            config=dict(
                base_load_from=None,
            ),
        ),
    )

    return [debug, wo_resume, mock_wo_resume]


"""
"base_weight_init: init the control blocks weights from pretrained base model weights
"""
vid2vid_2B_control_480p_base_weight_init = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "video_only_cosmos_transfer2_high_quality_v2_202505_s3"},
            {"override /model": "fsdp_control_vace"},
            {"override /net": "transfer2_control2world_net_2B"},
            {"override /conditioner": "video_prediction_control_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                    "log_sigma_loss",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_480p_base_weight_init",
        ),
        optimizer=dict(
            # lr=2 ** (-14),  # 2**(-14) = 6.103515625e-05 (32nodes)
            # lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            lr=2 ** (-13.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[1.0],
            f_min=[0.4],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=0,  # choose 0 (t2vid), 1 (img2vid) or 2 (vid2vid) latent frames
                max_num_conditional_frames=2,
                loss_scale=10.0,
                adjust_video_noise=True,
                scaling="rectified_flow",
                sigma_data=1.0,
                fsdp_shard_size=8,
                resolution="480p",
                state_t=20,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    vace_block_every_n=2,  # Default setting in VACE. For 28-layer base model, this means 14 control blocks.
                    condition_strategy="spaced",  # cond signal sent to every 2 base model blocks
                    vace_has_mask=False,  # just the latent of the control input, no inactive/active/mask stuff as in VACE paper (for now)
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=1.0,
                    rope_w_extrapolation_ratio=1.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="mm_only",
                    ),
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=1000,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_training_state=False,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
            callbacks=dict(
                iter_speed=dict(hit_thres=100),
                every_n_sample_reg=dict(
                    every_n=2000,
                    is_x0=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
                every_n_sample_ema=dict(
                    every_n=2000,
                    is_x0=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
                reg_model_image2video_sora_val_sampling=dict(
                    every_n=2000,
                    use_negative_prompt=True,
                    latent_video_length="${model.config.state_t}",
                ),
                ema_model_image2video_sora_val_sampling=dict(
                    every_n=2000,
                    use_negative_prompt=True,
                    latent_video_length="${model.config.state_t}",
                ),
                reg_model_image2video_vbench_val_sampling=dict(
                    every_n=2000,
                    use_negative_prompt=True,
                    latent_video_length="${model.config.state_t}",
                ),
                ema_model_image2video_vbench_val_sampling=dict(
                    every_n=2000,
                    use_negative_prompt=True,
                    latent_video_length="${model.config.state_t}",
                ),
            ),
        ),
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)

vid2vid_2B_control_480p_control_layer7 = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_480p_base_weight_init",
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_480p_control_layer7",
        ),
        model=dict(
            config=dict(
                resolution="480p",
                state_t=20,
                fsdp_shard_size=8,
                net=dict(
                    vace_block_every_n=4,
                ),
                base_load_from=BASE_CKPT_480p_T20,
            ),
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        trainer=dict(
            max_iter=100000,
            logging_iter=100,
            callbacks=dict(
                iter_speed=dict(hit_thres=300, every_n=100),
                every_n_sample_reg=dict(
                    every_n=1000,
                ),
                every_n_sample_ema=dict(
                    every_n=1000,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=1,
            use_cache=True,
            cache_size=32,
            concat_size=1,
            cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
            dataset=dict(
                resolution="${model.config.resolution}",
                augmentor_name="video_basic_augmentor_v2_with_control",
                video_decoder_name="video_naive_bytes",
                caption_type="t2w_qwen2p5_7b",
                embedding_type="t5_xxl",
                min_fps_thres=10,
                max_fps_thres=60,
                num_video_frames=93,
                use_native_fps=True,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

vid2vid_2B_control_480p_control_layer14 = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_480p_control_layer7",
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_480p_control_layer14",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=2,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

vid2vid_2B_control_720p_control_layer7 = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_480p_control_layer7",
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_720p_control_layer7",
        ),
        model=dict(
            config=dict(
                resolution="720",
                base_load_from=BASE_CKPT_720_T20,
            ),
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
        dataloader_train=dict(
            cache_size=32,
            dataset=dict(
                resolution="720",
            ),
        ),
    ),
    flags={"allow_objects": True},
)

vid2vid_2B_control_720p_control_layer14 = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_720p_control_layer7",
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_720p_control_layer14",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=2,
                ),
            ),
        ),
    ),
)

vid2vid_2B_control_720p_t24_oldsde_control_layer14 = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_720p_control_layer7",
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_720p_t24_oldsde_control_layer14",
        ),
        model=dict(
            config=dict(
                high_sigma_ratio=0.0,
                state_t=24,
                net=dict(
                    vace_block_every_n=2,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=1.0,
                ),
                base_load_from=BASE_CKPT_720_T24,
                hint_keys="edge_vis_depth_seg",
            ),
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        dataloader_train=dict(
            use_cache=False,
            num_workers=4,
        ),
        trainer=dict(
            max_iter=100000,
            logging_iter=100,
            callbacks=dict(
                iter_speed=dict(hit_thres=300, every_n=100),
                every_n_sample_reg=dict(
                    every_n=5000,
                ),
                every_n_sample_ema=dict(
                    every_n=5000,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

vid2vid_2B_control_720p_t24_control_layer14 = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_720p_t24_oldsde_control_layer14",
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_720p_t24_control_layer14",
        ),
        optimizer=dict(
            # lr=2 ** (-14),  # 2**(-14) = 6.103515625e-05 (32nodes)
            # lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[100_000],
        ),
        model=dict(
            config=dict(
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)


# Change text embedding to cosmos reason1 (cr1). Other settings remain the same
vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_720p_t24_oldsde_control_layer14",
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[100_000],
        ),
        model=dict(
            config=dict(
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                net=dict(
                    vace_block_every_n=2,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    use_crossattn_projection=True,  # project cr1 emb to 1024 dim
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                # add support for cosmos reason1 (cr1) embedding
                text_encoder_class="reason1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                    model_config=L(QwenVLBaseModel)(
                        model_config=L(QwenModelConfig)(
                            tokenizer_type="Qwen/Qwen2.5-VL-7B-Instruct",
                            name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
                            hidden_size=3584,
                            intermediate_size=18944,
                            max_window_layers=28,
                            num_attention_heads=28,
                            num_hidden_layers=28,
                            num_key_value_heads=4,
                            tie_word_embeddings=False,
                            vocab_size=152064,
                            vision_config=L(QwenVisionConfig)(out_hidden_size=3584),
                            output_hidden_states=True,
                        ),
                        tokenizer=L(build_tokenizer)(
                            tokenizer_type="Qwen/Qwen2.5-VL-7B-Instruct",
                        ),
                    ),
                ),
            ),
        ),
        dataloader_train=dict(
            dataset=dict(
                resolution="${model.config.resolution}",
                augmentor_name="video_basic_augmentor_v2_with_control",
                video_decoder_name="video_naive_bytes",
                caption_type="t2w_qwen2p5_7b",
                embedding_type=None,  # cr1 embedding is computed on the fly
                min_fps_thres=10,
                max_fps_thres=60,
                num_video_frames=93,
                use_native_fps=True,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# Change text embedding to cosmos reason1 (cr1). Using new sde and scheduler.
vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding_v2 = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding",
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding_v2",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=0.001,
        ),
        scheduler=dict(
            f_max=[0.5],
            f_min=[0.2],
            warm_up_steps=[2_000],
            cycle_lengths=[100000],
        ),
        model=dict(
            config=dict(
                adjust_video_noise=False,
                sde=dict(
                    p_mean=math.log(5.0),
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                rectified_flow_loss_weight_uniform=True,
                scaling="rectified_flow",  # correct loss weight for rectified flow
                base_load_from=BASE_CKPT_720_T24_CR1,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# Change text embedding to cosmos reason1.1 (cr1pt1).
vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding_v2",
            {"override /data_train": "video_only_cosmos_transfer2_high_quality_v1p1_20250715_s3"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                    "log_sigma_loss",
                    "load_base_model_callbacks",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=7,
                ),
                # high_sigma_ratio=0.0,  # high_sigma_ratio was introduced to resolve shot change in the base model, maybe we want to remove it
                # add support for cosmos reason1.1 (cr1pt1) embedding
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                ),
                base_load_from=BASE_CKPT_720_T24_CR1PT1,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding_v2_with_image_data = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding_v2",
            {
                "override /data_train": "image_cosmos_pretrain_20241108_two_video_cosmos_transfer2_high_quality_v1p1_20250715_s3"
            },
            # {"override /data_train": "image_cosmos_pretrain_20241108_video_cosmos_transfer2_high_quality_v1p1_20250715_s3"},
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding_v2_with_image_data",
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    ratio=0,
                ),
                video_data=dict(
                    ratio=1,
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        num_workers=4,
                        cache_size=32,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            augmentor_name="video_basic_augmentor_v2_with_control",
                            video_decoder_name="video_naive_bytes",
                            caption_type="t2w_qwen2p5_7b",
                            embedding_type=None,  # cr1 embedding is computed on the fly
                            min_fps_thres=10,
                            max_fps_thres=60,
                            num_video_frames=93,
                            use_native_fps=True,
                        ),
                    ),
                ),
                video_data_1=dict(
                    ratio=1,
                    dataloader=dict(
                        batch_size=6,
                        use_cache=False,
                        num_workers=4,
                        cache_size=32,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            augmentor_name="video_basic_augmentor_v2_with_control",
                            video_decoder_name="video_naive_bytes",
                            caption_type="t2w_qwen2p5_7b",
                            embedding_type=None,  # cr1 embedding is computed on the fly
                            num_video_frames=1,
                            use_native_fps=True,
                        ),
                    ),
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

vid2vid_2B_control_720p_t24_control_layer14_cr1pt1_embedding_v2_with_image_data = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding_v2_with_image_data",
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_720p_t24_control_layer14_cr1pt1_embedding_v2_with_image_data",
        ),
        model=dict(
            config=dict(
                # add support for cosmos reason1.1 (cr1pt1) embedding
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                ),
                base_load_from=BASE_CKPT_720_T24_CR1PT1,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    ratio=0,
                ),
                video_data=dict(
                    ratio=1,
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        num_workers=4,
                        cache_size=32,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            augmentor_name="video_basic_augmentor_v2_with_control",
                            video_decoder_name="video_naive_bytes",
                            caption_type="t2w_qwen2p5_7b",
                            embedding_type=None,  # cr1 embedding is computed on the fly
                            min_fps_thres=10,
                            max_fps_thres=60,
                            num_video_frames=93,
                            use_native_fps=True,
                        ),
                    ),
                ),
                video_data_1=dict(
                    ratio=1,
                    dataloader=dict(
                        batch_size=6,
                        use_cache=False,
                        num_workers=4,
                        cache_size=32,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            augmentor_name="video_basic_augmentor_v2_with_control",
                            video_decoder_name="video_naive_bytes",
                            caption_type="t2w_qwen2p5_7b",
                            embedding_type=None,  # cr1 embedding is computed on the fly
                            num_video_frames=1,
                            use_native_fps=True,
                        ),
                    ),
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

vid2vid_2B_control_720p_t24_control_layer14_image_context = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "video_only_cosmos_transfer2_high_quality_v2_202505_s3"},
            {"override /model": "fsdp_control_vace_image_context"},
            {"override /net": "transfer2_control2world_net_2B"},
            {"override /conditioner": "video_prediction_control_conditioner_image_context"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                    "log_sigma_loss",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_720p_t24_control_layer14_image_context",
        ),
        optimizer=dict(
            # lr=2 ** (-14),  # 2**(-14) = 6.103515625e-05 (32nodes)
            # lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[100_000],
        ),
        model=dict(
            config=dict(
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                min_num_conditional_frames=0,  # choose 0 (t2vid), 1 (img2vid) or 2 (vid2vid) latent frames
                max_num_conditional_frames=2,
                loss_scale=10.0,
                adjust_video_noise=True,
                scaling="rectified_flow",
                sigma_data=1.0,
                fsdp_shard_size=8,
                resolution="720",
                state_t=24,
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    vace_block_every_n=2,
                    condition_strategy="spaced",  # cond signal sent to every 2 base model blocks
                    vace_has_mask=False,  # just the latent of the control input, no inactive/active/mask stuff as in VACE paper (for now)
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=1.0,
                    sac_config=dict(
                        mode="mm_only",
                    ),
                    extra_image_context_dim=1152,  # CRITICAL: This enables I2VCrossAttention
                    share_q_in_i2v_cross_attn=True,  # CRITICAL: This use I2VCrossAttention instead of I2VCrossAttentionFull
                    img_context_deep_proj=False,
                ),
                base_load_from=BASE_CKPT_720_T24,
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=1000,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/vid2vid_2B_control/multicontrol_720p_t24_spaced_layer14_hqv1_20250625_64N/checkpoints/iter_000047000",
            load_training_state=False,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=100,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
            callbacks=dict(
                iter_speed=dict(hit_thres=300, every_n=100),
                every_n_sample_reg=dict(
                    every_n=1000,
                    is_x0=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
                every_n_sample_ema=dict(
                    every_n=1000,
                    is_x0=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
                reg_model_image2video_sora_val_sampling=dict(
                    every_n=2000,
                    use_negative_prompt=True,
                    latent_video_length="${model.config.state_t}",
                ),
                ema_model_image2video_sora_val_sampling=dict(
                    every_n=2000,
                    use_negative_prompt=True,
                    latent_video_length="${model.config.state_t}",
                ),
                reg_model_image2video_vbench_val_sampling=dict(
                    every_n=2000,
                    use_negative_prompt=True,
                    latent_video_length="${model.config.state_t}",
                ),
                ema_model_image2video_vbench_val_sampling=dict(
                    every_n=2000,
                    use_negative_prompt=True,
                    latent_video_length="${model.config.state_t}",
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=1,
            use_cache=False,
            num_workers=4,
            cache_size=32,
            concat_size=1,
            cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
            dataset=dict(
                resolution="720",
                augmentor_name="video_basic_augmentor_v2_with_control_and_image_context",
                video_decoder_name="video_naive_bytes",
                caption_type="t2w_qwen2p5_7b",
                embedding_type="t5_xxl",
                min_fps_thres=10,
                max_fps_thres=60,
                num_video_frames=93,
                use_native_fps=True,
            ),
        ),
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)

vid2vid_2B_control_1024p_imageOnly_control_layer14 = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_480p_base_weight_init",
            {"override /data_train": "image_only_cosmos_dlss_20250624_s3"},
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_1024p_imageOnly_control_layer14",
        ),
        model=dict(
            config=dict(
                resolution="1024",
                state_t=1,
                fsdp_shard_size=8,
                net=dict(
                    vace_block_every_n=2,
                ),
                base_load_from=BASE_CKPT_T2I,
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        dataloader_train=dict(
            batch_size=1,
            use_cache=True,
            cache_size=32,
            concat_size=1,
            cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
            dataset=dict(
                resolution="1024",
                augmentor_name="image_basic_augmentor_with_control",
            ),
        ),
    ),
    flags={"allow_objects": True},
)

vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "video_only_cosmos_transfer2_high_quality_v3p1_20250714_s3"},
            {"override /model": "fsdp_control_vace_rectified_flow"},
            {"override /net": "transfer2_control2world_net_2B"},
            {"override /conditioner": "video_prediction_control_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                    # "log_sigma_loss",  #check
                    "load_base_model_callbacks",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow",
        ),
        optimizer=dict(
            lr=8.63e-5,  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=1e-3,
            betas=[0.9, 0.999],
        ),
        scheduler=dict(
            f_max=[0.5],
            f_min=[0.2],
            warm_up_steps=[100],
            cycle_lengths=[100_000],
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                conditional_frames_probs={0: 0.4, 1: 0.4, 2: 0.2},
                fsdp_shard_size=8,
                resolution="720",
                state_t=24,
                shift=5,
                use_dynamic_shift=False,
                train_time_weight="reweighting",
                train_time_distribution="logitnormal",
                net=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    timestep_scale=0.001,
                    sac_config=dict(
                        mode="predict2_2b_720_aggressive",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                    use_wan_fp32_strategy=True,
                    vace_block_every_n=7,
                ),
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                        use_empty_string=False,  # (TODO: hanzim): check
                    ),
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                ),
                base_load_from=BASE_CKPT_720_T24_CR1PT1_RECTIFIED_FLOW,
                hint_keys="edge",
            )
        ),
        checkpoint=dict(
            save_iter=1000,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_training_state=False,
            strict_resume=False,
            load_path="cosmos_transfer2/vid2vid_2B_control/edge_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N/checkpoints/iter_000060000",
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
            callbacks=dict(
                iter_speed=dict(hit_thres=300, every_n=100),
                grad_clip=dict(
                    clip_norm=0.1,
                ),
                manual_gc=dict(
                    every_n=200,
                ),
                every_n_sample_reg=dict(
                    every_n=5000,
                    guidance=[0, 3, 7],
                ),
                every_n_sample_ema=dict(
                    every_n=5000,
                    guidance=[0, 3, 7],
                ),
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
    ),
    flags={"allow_objects": True},
)

vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow_with_image_context_with_image_data = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow",
            {
                "override /data_train": "image_cosmos_pretrain_20241108_two_video_cosmos_transfer2_high_quality_v3p1_20250714_s3"
            },
            {"override /model": "fsdp_control_vace_rectified_flow"},
            {"override /conditioner": "video_prediction_control_conditioner_image_context"},
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow_with_image_context_with_image_data",
        ),
        model=dict(
            config=dict(
                net=dict(
                    extra_image_context_dim=1152,  # CRITICAL: This enables I2VCrossAttention / I2VCrossAttentionFull
                    share_q_in_i2v_cross_attn=False,  # CRITICAL: setting to False uses I2VCrossAttentionFull (better) instead of I2VCrossAttention
                    img_context_deep_proj=False,
                ),
                use_reference_image=True,
                hint_keys="edge",
            )
        ),
        checkpoint=dict(
            load_path="",
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    ratio=0,
                ),
                video_data=dict(
                    ratio=1,
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        num_workers=4,
                        cache_size=32,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            augmentor_name="video_basic_augmentor_v2_with_control_and_image_context",
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
                ),
                video_data_1=dict(
                    ratio=1,
                    dataloader=dict(
                        batch_size=6,
                        use_cache=False,
                        num_workers=4,
                        cache_size=32,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            augmentor_name="video_basic_augmentor_v2_with_control_and_image_context",
                            video_decoder_name="video_naive_bytes",
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="gt720p",
                            embedding_type=None,  # cr1 embedding is computed on the fly
                            num_video_frames=1,
                            use_native_fps=True,
                            control_input_type="edge",
                        ),
                    ),
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)


# 14B model with cosmos reason1.1 (cr1pt1) embedding and rectified flow
vid2vid_14B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "video_only_cosmos_transfer2_high_quality_v3p1_20250714_s3"},
            {"override /model": "fsdp_control_vace_rectified_flow"},
            {"override /net": "transfer2_control2world_net_14B"},
            {"override /conditioner": "video_prediction_control_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                    "load_base_model_callbacks",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="vid2vid_14B_control",
            name="vid2vid_14B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.001,
            betas=[0.9, 0.999],
        ),
        scheduler=dict(
            f_max=[0.3],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[200_000],
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                fsdp_shard_size=32,
                resolution="720",
                state_t=24,
                shift=5,
                use_dynamic_shift=False,
                train_time_weight="reweighting",
                train_time_distribution="logitnormal",
                net=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    timestep_scale=0.001,
                    sac_config=dict(
                        mode="predict2_14b_720_aggressive",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                    use_wan_fp32_strategy=True,
                    vace_block_every_n=9,  # 36 / 4 = 9
                ),
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                        use_empty_string=False,
                    ),
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                    s3_credential_path="credentials/s3_checkpoint.secret",
                ),
                use_high_sigma_strategy=True,
                base_load_from=BASE_CKPT_14B_720_T24_CR1PT1_RECTIFIED_FLOW,
                hint_keys="edge",
            )
        ),
        checkpoint=dict(
            save_iter=1000,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_training_state=False,
            strict_resume=True,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=100,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
            callbacks=dict(
                iter_speed=dict(hit_thres=300, every_n=100),
                grad_clip=dict(
                    clip_norm=0.1,
                ),
                manual_gc=dict(
                    every_n=200,
                ),
                every_n_sample_reg=dict(
                    every_n=5000,
                    guidance=[0, 3, 7],
                ),
                every_n_sample_ema=dict(
                    every_n=5000,
                    guidance=[0, 3, 7],
                ),
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
    ),
    flags={"allow_objects": True},
)

"""
torchrun --nproc_per_node=1 --master_port=12340 -m scripts.train --dryrun --config=cosmos_predict2/_src/transfer2/configs/vid2vid_transfer/config.py -- experiment=vid2vid_2B_control_720p_control_layer14
"""


cs = ConfigStore.instance()
for _item, _item_debug, _item_wo_resume, _item_mock_wo_resume in [
    [
        vid2vid_2B_control_480p_base_weight_init,
        *build_debug_runs(vid2vid_2B_control_480p_base_weight_init),
    ],
    [
        vid2vid_2B_control_480p_control_layer7,
        *build_debug_runs(vid2vid_2B_control_480p_control_layer7),
    ],
    [
        vid2vid_2B_control_480p_control_layer14,
        *build_debug_runs(vid2vid_2B_control_480p_control_layer14),
    ],
    # 720p
    [
        vid2vid_2B_control_720p_control_layer7,  # state_t=20
        *build_debug_runs(vid2vid_2B_control_720p_control_layer7),
    ],
    [
        vid2vid_2B_control_720p_control_layer14,  # state_t=20
        *build_debug_runs(vid2vid_2B_control_720p_control_layer14),
    ],
    [
        vid2vid_2B_control_720p_t24_oldsde_control_layer14,  # state_t=24
        *build_debug_runs(vid2vid_2B_control_720p_t24_oldsde_control_layer14),
    ],
    [
        vid2vid_2B_control_720p_t24_control_layer14,  # state_t=24
        *build_debug_runs(vid2vid_2B_control_720p_t24_control_layer14),
    ],
    [
        vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding,  # state_t=24, with cosmos reason1 (cr1) embedding
        *build_debug_runs(vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding),
    ],
    [
        vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding_v2,  # state_t=24, with cosmos reason1 (cr1) embedding
        *build_debug_runs(vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding_v2),
    ],
    [
        vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding_v2_with_image_data,  # state_t=24, with cosmos reason1 (cr1) embedding
        *build_debug_runs(vid2vid_2B_control_720p_t24_control_layer14_cr1_embedding_v2_with_image_data),
    ],
    [
        vid2vid_2B_control_720p_t24_control_layer14_cr1pt1_embedding_v2_with_image_data,  # state_t=24, with cosmos reason1.1 (cr1pt1) embedding
        *build_debug_runs(vid2vid_2B_control_720p_t24_control_layer14_cr1pt1_embedding_v2_with_image_data),
    ],
    [
        vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding,  # state_t=24, with cosmos reason1pt1 (cr1pt1) embedding
        *build_debug_runs(vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding),
    ],
    # 1024p Image Only
    [
        vid2vid_2B_control_1024p_imageOnly_control_layer14,
        *build_debug_runs(vid2vid_2B_control_1024p_imageOnly_control_layer14),
    ],
    # image context
    [
        vid2vid_2B_control_720p_t24_control_layer14_image_context,
        *build_debug_runs(vid2vid_2B_control_720p_t24_control_layer14_image_context),
    ],
    # rectified flow
    [
        vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow,
        *build_debug_runs(vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow),
    ],
    # rectified flow with image context and image data
    [
        vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow_with_image_context_with_image_data,
        *build_debug_runs(
            vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow_with_image_context_with_image_data
        ),
    ],
    # 14B model with cosmos reason1.1 (cr1pt1) embedding and rectified flow
    [
        vid2vid_14B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow,
        *build_debug_runs(vid2vid_14B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow),
    ],
]:
    cs.store(group="experiment", package="_global_", name=f"{_item['job']['name']}", node=_item)
    if _item_debug is not None:
        cs.store(
            group="experiment",
            package="_global_",
            name=f"{_item['job']['name']}_debug",
            node=_item_debug,
        )
    if _item_wo_resume is not None:
        cs.store(
            group="experiment",
            package="_global_",
            name=f"{_item['job']['name']}_wo_resume",
            node=_item_wo_resume,
        )
    if _item_mock_wo_resume is not None:
        cs.store(
            group="experiment",
            package="_global_",
            name=f"{_item['job']['name']}_mock_wo_resume",
            node=_item_mock_wo_resume,
        )
