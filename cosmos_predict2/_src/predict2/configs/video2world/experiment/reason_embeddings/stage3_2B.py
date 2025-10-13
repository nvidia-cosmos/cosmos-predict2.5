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

import functools
import math

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.datasets.cached_replay_dataloader import duplicate_batches, duplicate_batches_random
from cosmos_predict2._src.predict2.models.video2world_model import HighSigmaStrategy
from cosmos_predict2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy

_TRAINER_DEBUG_CONFIG = dict(
    max_iter=1000,
    logging_iter=50,
    callbacks=dict(
        every_n_sample_reg=dict(
            every_n=10,
        ),
        every_n_sample_ema=dict(
            every_n=10,
        ),
    ),
)
_CKPT_DEBUG_CONFIG = dict(
    save_iter=10,
    load_path="",
    load_training_state=False,
    strict_resume=False,
)


def build_debug_runs(job):
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
    )

    return [wo_resume, mock_wo_resume]


"""
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --dryrun --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4
"""
I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "mock"},
            {"override /model": "fsdp"},
            {"override /net": "cosmos_v1_2B"},
            {"override /conditioner": "video_prediction_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-3-Size-2B-Res-480-Fps-16-Note-qwen_video_only_later_frames",
        ),
        optimizer=dict(
            lr=2 ** (-14),  # 2**(-14) = 6.103515625e-05
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.99],
            f_min=[0.4],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,  # choose either 1 (img2vid) or 2 (vid2vid) latent frames
                max_num_conditional_frames=2,
                loss_scale=10.0,
                adjust_video_noise=True,
                scaling="rectified_flow",
                sigma_data=1.0,
                fsdp_shard_size=8,
                resolution="480",
                state_t=24,
                resize_online=True,
                net=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=2.0,
                    rope_w_extrapolation_ratio=2.0,
                    rope_t_extrapolation_ratio=1.0,
                ),
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.3,
                    ),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs/Stage-a_pt_3-Index-1-Size-2B-Res-480-Fps-16-Note-qwen_imagecaption/checkpoints/iter_000057500",
            load_training_state=False,
            strict_resume=False,
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        trainer=dict(
            max_iter=250_000,
            logging_iter=200,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=500000000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
                every_n_sample_ema=dict(
                    every_n=500000000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=24,
                        num_workers=6,
                        use_cache=False,
                        cache_size=8,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches, n=1),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                        ),
                    ),
                    ratio=0,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=True,
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.5),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            caption_type="i2w_qwen2p5_7b_later_frames",
                            use_native_fps=True,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY['job']['name']}",
            {"override /data_train": "mock"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[200_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720_aggressive",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                text_encoder_class="reason1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22/checkpoints/iter_000026000",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            dataset_resolution_type="gt720p",
                            caption_type="qwen2p5_7b_v4",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                    ),
                    ratio=1,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type=None,
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="all",
                            num_video_frames=93,
                            use_native_fps=True,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)


I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_RESUME4: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4",
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=50,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                video_data=dict(
                    dataloader=dict(
                        use_cache=False,
                    ),
                    ratio=1,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat/checkpoints/iter_000010000",
            load_training_state=False,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_1E2: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_RESUME4['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_RESUME4["job"][
                "group"
            ],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_1e2",
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=50,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=0.01,
        ),
        scheduler=dict(
            f_max=[0.5],
            f_min=[0.2],
            warm_up_steps=[2_000],
            cycle_lengths=[100_000],
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000",
            load_training_state=False,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_1E3: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_RESUME4['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_RESUME4["job"][
                "group"
            ],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_1e3",
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=50,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=0.001,
        ),
        scheduler=dict(
            f_max=[0.5],
            f_min=[0.2],
            warm_up_steps=[2_000],
            cycle_lengths=[100_000],
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000",
            load_training_state=False,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma",
        ),
        trainer=dict(
            max_iter=100000,
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
                sde=dict(
                    p_mean=math.log(5.5),
                    p_std=1.1,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid_debug/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma/checkpoints/iter_000002500",
            load_training_state=True,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted",
        ),
        trainer=dict(
            max_iter=100000,
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
                sde=dict(
                    p_mean=math.log(5.5),
                    p_std=1.1,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                rectified_flow_loss_weight_uniform=False,
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000",
            load_training_state=False,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted_resume2",
        ),
        trainer=dict(
            max_iter=100000,
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
                rectified_flow_loss_weight_uniform=False,
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted/checkpoints/iter_000015000/",
            load_training_state=True,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

# distribution matching + loss weight, on general pretraining dataset
I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V2_GENERAL = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted_v2_general",
        ),
        model=dict(
            config=dict(
                scaling="rectified_flow",  # correct loss weight for rectified flow
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
    ),
    flags={"allow_objects": True},
)
I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V2_GENERAL_RESUME1 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V2_GENERAL['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted_v2_general_resume1",
        ),
        checkpoint=dict(
            save_iter=2_500,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted_v2_general/checkpoints/iter_000002500/",
            load_training_state=True,
            strict_resume=True,
        ),
        trainer=dict(
            straggler_detection=dict(
                enabled=False,
            ),
            logging_iter=20,
        ),
        model=dict(
            config=dict(
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.2,
                    ),
                ),
            )
        ),
    ),
    flags={"allow_objects": True},
)
I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V2_AV = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2['job']['name']}",
            {
                "override /data_train": "mock",
            },
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted_v2_av",
        ),
        model=dict(
            config=dict(
                scaling="rectified_flow",  # correct loss weight for rectified flow
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
    ),
    flags={"allow_objects": True},
)
I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V2_AV_RESUME1 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V2_AV['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted_v2_av_resume1",
        ),
        checkpoint=dict(
            save_iter=2_500,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted_v2_av/checkpoints/iter_000007500/",
            load_training_state=True,
            strict_resume=True,
        ),
        trainer=dict(
            straggler_detection=dict(
                enabled=False,
            ),
            logging_iter=20,
        ),
        model=dict(
            config=dict(
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.2,
                    ),
                ),
            )
        ),
    ),
    flags={"allow_objects": True},
)
# distribution matching + loss weight + wan fp32, on general pretraining dataset
I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V3_GENERAL_CP4 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted_v3_general_cp4",
        ),
        model=dict(
            config=dict(
                scaling="rectified_flow",  # correct loss weight for rectified flow
                # use wan fp32
                use_wan_fp32_strategy=True,
                net=dict(
                    use_wan_fp32_strategy=True,
                ),
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.2,
                    ),
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V3_AV_CP4 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2['job']['name']}",
            {
                "override /data_train": "mock",
            },
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted_v3_av_cp4",
        ),
        model=dict(
            config=dict(
                scaling="rectified_flow",  # correct loss weight for rectified flow
                # use wan fp32
                use_wan_fp32_strategy=True,
                net=dict(
                    use_wan_fp32_strategy=True,
                ),
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.2,
                    ),
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
    ),
    flags={"allow_objects": True},
)
# distribution matching + loss weight + wan fp32 + empty string embedding, on general pretraining dataset
I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V4_GENERAL_CP4 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted_v4_general_cp4",
        ),
        model=dict(
            config=dict(
                scaling="rectified_flow",  # correct loss weight for rectified flow
                # use wan fp32
                use_wan_fp32_strategy=True,
                net=dict(
                    use_wan_fp32_strategy=True,
                ),
                conditioner=dict(
                    text=dict(
                        use_empty_string=True,
                    ),
                    use_video_condition=dict(
                        dropout_rate=0.2,
                    ),
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V4_AV_CP4 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2['job']['name']}",
            {
                "override /data_train": "mock",
            },
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted_v4_av_cp4",
        ),
        model=dict(
            config=dict(
                scaling="rectified_flow",  # correct loss weight for rectified flow
                # use wan fp32
                use_wan_fp32_strategy=True,
                net=dict(
                    use_wan_fp32_strategy=True,
                ),
                conditioner=dict(
                    text=dict(
                        use_empty_string=True,
                    ),
                    use_video_condition=dict(
                        dropout_rate=0.2,
                    ),
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
    ),
    flags={"allow_objects": True},
)
I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma",
        ),
        trainer=dict(
            max_iter=100000,
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
                sde=dict(
                    p_mean=math.log(5.5),
                    p_std=1.1,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid_debug/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma/checkpoints/iter_000002500",
            load_training_state=True,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted",
        ),
        trainer=dict(
            max_iter=100000,
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
                sde=dict(
                    p_mean=math.log(5.5),
                    p_std=1.1,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                rectified_flow_loss_weight_uniform=False,
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000",
            load_training_state=False,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted_resume2",
        ),
        trainer=dict(
            max_iter=100000,
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
                rectified_flow_loss_weight_uniform=False,
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted/checkpoints/iter_000015000/",
            load_training_state=True,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

# distribution matching + loss weight + use empty string, on general pretraining dataset
T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_EMPTY_STRING = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_empty_string",
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                scaling="rectified_flow",  # correct loss weight for rectified flow
                conditioner=dict(
                    text=dict(
                        use_empty_string=True,
                    ),
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_wd_high_sigma_loss_reweighted_resume2/checkpoints/iter_000055000",
            load_training_state=False,
            strict_resume=True,
        ),
        dataloader_train=dict(
            dataloaders=dict(
                video_data=dict(
                    dataloader=dict(
                        use_cache=False,
                    ),
                    ratio=3,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# distribution matching + loss weight, on general pretraining dataset
T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_EMPTY_STRING['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted",
        ),
        model=dict(
            config=dict(
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                conditioner=dict(
                    text=dict(
                        use_empty_string=False,
                    ),
                ),
            ),
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000032500/",
        ),
    ),
    flags={"allow_objects": True},
)


################################################################################
# Configs for different resolutions and fps

# 2B, 720p, 10fps
I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_100_SIZE_2B_RES_720_FPS10_HQ_V5_from_26 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY['job']['name']}",
            {"override /data_train": "mock"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-100-Size-2B-Res-720-Fps-10-Note-HQ_V5_from_26_qwen_concat",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[200_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="720",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=16,
                resize_online=True,
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720_aggressive",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=16.0 / 24,
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                text_encoder_class="reason1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-100-Size-2B-Res-720-Fps-10-Note-HQ_V5_from_26/checkpoints/iter_000010000/",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            dataset_resolution_type="gt720p",
                            caption_type="qwen2p5_7b_v4",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                    ),
                    ratio=1,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type=None,
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="all",
                            num_video_frames=61,
                            use_native_fps=True,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# 2B, 480p, 10fps
I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_101_SIZE_2B_RES_480_FPS10_HQ_V5_from_26 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY['job']['name']}",
            {"override /data_train": "mock"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-101-Size-2B-Res-480-Fps-10-Note-HQ_V5_from_26_qwen_concat",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[200_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="480",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=16,
                resize_online=True,
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=16.0 / 24,
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                text_encoder_class="reason1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-101-Size-2B-Res-480-Fps-10-Note-HQ_V5_from_26/checkpoints/iter_000010000/",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            dataset_resolution_type="gt720p",
                            caption_type="qwen2p5_7b_v4",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                    ),
                    ratio=1,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type=None,
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="all",
                            num_video_frames=61,
                            use_native_fps=True,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# 2B, 480p, 16fps
I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_102_SIZE_2B_RES_480_FPS16_HQ_V5_from_26 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY['job']['name']}",
            {"override /data_train": "mock"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-102-Size-2B-Res-480-Fps-16-Note-HQ_V5_from_26_qwen_concat",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[200_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                resolution="480",
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                state_t=24,
                resize_online=True,
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                text_encoder_class="reason1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-102-Size-2B-Res-480-Fps-16-Note-HQ_V5_from_26/checkpoints/iter_000010000/",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            dataset_resolution_type="gt720p",
                            caption_type="qwen2p5_7b_v4",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                    ),
                    ratio=1,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type=None,
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            dataset_resolution_type="all",
                            num_video_frames=93,
                            use_native_fps=True,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

########################################################
# Post training cooldown configs

# distribution matching + loss weight, on general pretraining dataset
T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_HQ_COOLDOWN_FROM_57K = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED['job']['name']}",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16_hq_cooldown_from_57K",
        ),
        model=dict(
            config=dict(
                conditioner=dict(
                    text=dict(
                        use_empty_string=False,
                    ),
                ),
            ),
        ),
        trainer=dict(
            max_iter=50_000,
        ),
        scheduler=dict(
            f_max=[0.4],
            f_min=[0.0],
            warm_up_steps=[0],
            cycle_lengths=[50_000],
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000057500/",
            load_training_state=False,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_HQ_COOLDOWN_FROM_57K_4K_VIDEOS = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_HQ_COOLDOWN_FROM_57K['job']['name']}",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16_hq_cooldown_from_57K_4K_videos",
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000057500/",
            load_training_state=False,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)


cs = ConfigStore.instance()

for _item, _item_wo_resume, _item_mock_wo_resume in [
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_RESUME4,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_RESUME4),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_100_SIZE_2B_RES_720_FPS10_HQ_V5_from_26,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_100_SIZE_2B_RES_720_FPS10_HQ_V5_from_26),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_101_SIZE_2B_RES_480_FPS10_HQ_V5_from_26,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_101_SIZE_2B_RES_480_FPS10_HQ_V5_from_26),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_102_SIZE_2B_RES_480_FPS16_HQ_V5_from_26,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_102_SIZE_2B_RES_480_FPS16_HQ_V5_from_26),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_1E3,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_1E3),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_1E2,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_1E2),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V2_GENERAL,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V2_GENERAL
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V2_GENERAL_RESUME1,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V2_GENERAL_RESUME1
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V2_AV,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V2_AV
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V2_AV_RESUME1,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V2_AV_RESUME1
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V3_GENERAL_CP4,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V3_GENERAL_CP4
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V3_AV_CP4,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V3_AV_CP4
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V4_GENERAL_CP4,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V4_GENERAL_CP4
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V4_AV_CP4,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_V4_AV_CP4
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED
        ),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2,
        *build_debug_runs(
            I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_RESUME2
        ),
    ],
    [
        T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_EMPTY_STRING,
        *build_debug_runs(
            T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED_EMPTY_STRING
        ),
    ],
    [
        T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED,
        *build_debug_runs(
            T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_from_22_WD_HIGH_SIGMA_LOSS_REWEIGHTED
        ),
    ],
    [
        T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_HQ_COOLDOWN_FROM_57K,
        *build_debug_runs(T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_HQ_COOLDOWN_FROM_57K),
    ],
    [
        T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_HQ_COOLDOWN_FROM_57K_4K_VIDEOS,
        *build_debug_runs(
            T2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_HQ_V6_HQ_COOLDOWN_FROM_57K_4K_VIDEOS
        ),
    ],
]:
    cs.store(group="experiment", package="_global_", name=f"{_item['job']['name']}", node=_item)
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
