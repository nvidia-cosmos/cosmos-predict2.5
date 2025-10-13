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

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.datasets.cached_replay_dataloader import (
    duplicate_batches,
    duplicate_batches_random,
    get_cached_replay_dataloader,
)
from cosmos_predict2._src.predict2.datasets.dataset_provider import get_image_dataset, get_video_dataset
from cosmos_predict2._src.predict2.datasets.joint_dataloader import IterativeJointDataLoader
from cosmos_predict2._src.predict2.models.video2world_model import HighSigmaStrategy
from cosmos_predict2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy

_TRAINER_DEBUG_CONFIG = dict(
    max_iter=1000,
    logging_iter=50,
    callbacks=dict(
        every_n_sample_reg=dict(
            every_n=1000000000000,
        ),
        every_n_sample_ema=dict(
            every_n=1000000000000,
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


T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_STANDALONE = LazyDict(
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
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_standalone",
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
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                loss_scale=10.0,
                adjust_video_noise=False,
                scaling="rectified_flow",
                sigma_data=1.0,
                fsdp_shard_size=8,
                resolution="720",
                state_t=24,
                resize_online=True,
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                rectified_flow_loss_weight_uniform=False,
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
                    crossattn_emb_channels=1024,
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
                sde=dict(
                    p_mean=1.6094379124341003,  # math.log(5.0)
                    p_std=1.0,
                    sigma_max=200,
                    sigma_min=0.01,
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
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000065000/",
            load_training_state=False,
            strict_resume=True,
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        trainer=dict(
            max_iter=100000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=5000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
                every_n_sample_ema=dict(
                    every_n=5000,
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
                        batch_size=12,
                        num_workers=6,
                        use_cache=False,
                        cache_size=8,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches, n=1),
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
                    ratio=3,
                ),
            ),
        ),
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)

T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "mock"},
            {"override /model": "fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B"},
            {"override /conditioner": "video_prediction_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
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
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only",
        ),
        optimizer=dict(
            lr=3e-5,  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=1e-3,
            betas=[0.9, 0.999],
        ),
        scheduler=dict(
            f_max=[0.99],
            f_min=[0.4],
            warm_up_steps=[100],
            cycle_lengths=[400_000],
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
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
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/",
            load_training_state=False,
            strict_resume=True,
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        trainer=dict(
            max_iter=150_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
            callbacks=dict(
                grad_clip=dict(
                    clip_norm=0.1,
                ),
                manual_gc=dict(
                    every_n=200,
                ),
                every_n_sample_reg=dict(
                    every_n=1000000000000,
                ),
                every_n_sample_ema=dict(
                    every_n=1000000000000,
                ),
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=12,
                        num_workers=6,
                        use_cache=False,
                        cache_size=8,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches, n=1),
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
                    ratio=3,
                ),
            ),
        ),
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)

T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_IMPROVED = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            # {"override /data_train": None},
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_improved",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only/checkpoints/iter_000037000/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        dataloader_train=L(IterativeJointDataLoader)(
            dataloaders={
                "image_data": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_image_dataset)(
                            dataset_name="cosmos_pretrain_and_synthetic_photoreal_20250805_image_whole",
                            object_store="gcp",
                            resolution="${model.config.resolution}",
                            is_train=True,
                            caption_type="qwen2p5_7b_v4",
                            dataset_resolution_type="gt720p",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                        batch_size=12,
                        num_workers=8,
                        prefetch_factor=4,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="image_dataloader",
                        use_cache=False,
                        cache_size=8,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches, n=1),
                    ),
                    ratio=1,
                ),
                "image_data_prompt_captions": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_image_dataset)(
                            dataset_name="cosmos_synthetic_filtered_combined_20250805_image_whole",
                            object_store="gcp",
                            resolution="${model.config.resolution}",
                            is_train=True,
                            caption_type="prompts",
                            dataset_resolution_type="gt720p",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                        batch_size=12,
                        num_workers=8,
                        prefetch_factor=4,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="image_dataloader",
                        use_cache=False,
                        cache_size=8,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches, n=1),
                    ),
                    ratio=1,
                ),
                "video_data": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_video_dataset)(
                            dataset_name="cosmos_pretrainvideo_20250806_dedup_accumulated_and_high_quality_v3_202505_video_whole",
                            object_store="s3",
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="noframedrop_nocameramove_video_augmentor_v1",
                            # will use the augmentor to filter out frame drop
                            # so min and max fps can just use generic ones
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=93,
                            # does not touch on low res data that will have jittering
                            dataset_resolution_type="gt720p",
                            use_native_fps=True,
                            embedding_type=None,
                            is_train=True,
                            chunk_size=256,
                        ),
                        batch_size=1,
                        use_cache=False,
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
                        num_workers=2,
                        prefetch_factor=2,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="video_dataloader",
                    ),
                    ratio=2,
                ),
            },
        ),
    ),
    flags={"allow_objects": True},
)


"""
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/config.py -- experiment=cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_480_640_ ~dataloader_train.dataloaders
"""
AC_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only",
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            {"override /data_train": "mock"},
            {"override /data_val": "mock"},
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_480_640_",
            project="cosmos_predict2_action_conditioned",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=500,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                ),
                every_n_sample_ema=dict(
                    every_n=500,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        model=dict(
            config=dict(
                # NOTE: this should be 1 for the action conditioned model
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                # overwrite the probs to disable random num of conditional frames
                conditional_frames_probs=None,
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=2,
        ),
    ),
    flags={"allow_objects": True},
)


"""
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/config.py -- experiment=cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_pi_benchmark_21frame_gripper_10x_ ~dataloader_train.dataloaders
"""
AC_CHUNK_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B_PI_BENCHMARK_21FRAME_20HZ_GRIPPER_10X = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_480_640_",
            {"override /net": "cosmos_v1_2B_action_chunk_conditioned"},
            {"override /data_train": "mock"},
            {"override /data_val": "mock"},
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_pi_benchmark_21frame_gripper_10x_",
            project="cosmos_predict2_action_conditioned",
        ),
        optimizer=dict(
            lr=8e-5,
            weight_decay=0.1,
        ),
        model=dict(
            config=dict(
                state_t=1 + 20 // 4,
                net=dict(
                    action_dim=7,
                    temporal_compression_ratio=4,
                ),
            ),
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=500,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                ),
                every_n_sample_ema=dict(
                    every_n=500,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=2,
            sampler=dict(
                dataset=dict(gripper_rescale_factor=10, num_action_per_chunk=20),
            ),
            dataset=dict(gripper_rescale_factor=10, num_action_per_chunk=20),
        ),
    ),
    flags={"allow_objects": True},
)


"""
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/config.py -- experiment=cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320 ~dataloader_train.dataloaders
"""
AC_CHUNK_MULTI_VIEW_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B_BRIDGE_13FRAME_256X320 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_pi_benchmark_21frame_gripper_10x_",
            {"override /net": "cosmos_v1_2B_action_chunk_conditioned"},
            {"override /data_train": "mock"},
            {"override /data_val": "mock"},
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320",
            project="cosmos_predict2_action_conditioned",
        ),
        optimizer=dict(
            lr=32e-5,
            weight_decay=0.1,
        ),
        model=dict(
            config=dict(
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=7,
                    temporal_compression_ratio=4,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=8,
            sampler=dict(
                dataset=dict(
                    gripper_rescale_factor=1, num_action_per_chunk=12, fps_downsample_ratio=1, video_size=[256, 320]
                ),
            ),
            dataset=dict(
                gripper_rescale_factor=1, num_action_per_chunk=12, fps_downsample_ratio=1, video_size=[256, 320]
            ),
        ),
    ),
    flags={"allow_objects": True},
)


cs = ConfigStore.instance()

for _item, _item_wo_resume, _item_mock_wo_resume in [
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_STANDALONE,
        *build_debug_runs(T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_STANDALONE),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW,
        *build_debug_runs(T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_IMPROVED,
        *build_debug_runs(
            T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_IMPROVED
        ),
    ],
    [
        AC_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B,
        *build_debug_runs(AC_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B),
    ],
    [
        AC_CHUNK_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B_PI_BENCHMARK_21FRAME_20HZ_GRIPPER_10X,
        *build_debug_runs(AC_CHUNK_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B_PI_BENCHMARK_21FRAME_20HZ_GRIPPER_10X),
    ],
    [
        AC_CHUNK_MULTI_VIEW_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B_BRIDGE_13FRAME_256X320,
        *build_debug_runs(AC_CHUNK_MULTI_VIEW_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B_BRIDGE_13FRAME_256X320),
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
