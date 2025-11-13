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

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from cosmos_predict2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)
from cosmos_predict2._src.predict2_multiview.callbacks.log_weight import LogWeight

"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2_multiview/configs/vid2vid/config.py -- experiment=buttercup_predict2p5_14b_7views_res720p_fps30_t8_frombase2p5_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames_nofps_uniform
"""


def buttercup_predict2p5_14b_7views_res720p_fps10_t8_fromavfinetune_allcaps_29frames_nofps_uniform() -> dict:
    sample_n_views, state_t = 7, 8
    return dict(
        defaults=[
            {"override /data_train": "video_alpamayo_dec2024_gcs_720p_10fps_29frames_7views"},
            {"override /conditioner": "video_prediction_multiview_conditioner"},
            {"override /model": "fsdp_rectified_flow_multiview"},
            {"override /net": "cosmos_v1_14B_multiview"},
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
            group="cosmos2p5_mv",
            name="buttercup_predict2p5_14b_7views_res720p_fps10_t8_fromavfinetune_allcaps_29frames_nofps_uniform",
        ),
        checkpoint=dict(
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            save_iter=250,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-43-Size-14B-Res-720-Fps-16-Note-rf_av_high_sigma_uniform/checkpoints/iter_000022000/",
        ),
        model_parallel=dict(
            context_parallel_size=8,
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
                conditional_frames_probs={0: 0.6, 1: 0.2, 2: 0.2},
                condition_locations=["first_random_n"],
                fsdp_shard_size=32,
                resolution="720p",
                online_text_embeddings_as_dict=False,
                state_t=state_t,
                shift=5,
                use_dynamic_shift=False,
                train_time_weight="uniform",
                train_time_distribution="logitnormal",
                use_high_sigma_strategy=True,
                net=dict(
                    concat_view_embedding=True,
                    view_condition_dim=7,
                    n_cameras_emb=7,
                    state_t=state_t,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=float(state_t) / 24,
                    timestep_scale=0.001,
                    sac_config=dict(
                        mode="predict2_14b_720_aggressive",
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
            )
        ),
        trainer=dict(
            straggler_detection=dict(
                enabled=False,
            ),
            callbacks=dict(
                compile_tokenizer=dict(
                    enabled=False,
                ),
                iter_speed=dict(
                    hit_thres=50,
                    every_n=100,
                ),
                every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=2_000,
                    do_x0_prediction=False,
                    is_ema=True,
                    num_sampling_step=35,
                    guidance=[0, 3],
                    fps=10,
                    sample_n_views=sample_n_views,
                    run_at_start=True,
                ),
            ),
        ),
        dataloader_train=dict(
            augmentation_config=dict(
                caption_probability={
                    "qwen2p5_7b_caption": 0.7,
                    "qwen2p5_7b_caption_medium": 0.2,
                    "qwen2p5_7b_caption_short": 0.1,
                },
            )
        ),
    )


def buttercup_predict2p5_14b_crossview_7views_res720p_fps10_t8_fromavfinetune_allcaps_29frames_nofps_uniform():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_14b_7views_res720p_fps10_t8_fromavfinetune_allcaps_29frames_nofps_uniform",
            {"override /net": "cosmos_v1_14B_multiview_crossview"},
            {"override /optimizer": "multiplefusedadamw"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv2",
            name="buttercup_predict2p5_14b_crossview_7views_res720p_fps10_t8_fromavfinetune_allcaps_29frames_nofps_uniform",
        ),
        model=dict(
            config=dict(
                net=dict(
                    cross_view_attn_map_str={
                        "camera_front_wide_120fov": [
                            "camera_cross_left_120fov",
                            "camera_cross_right_120fov",
                            "camera_front_tele_30fov",
                        ],
                        "camera_cross_left_120fov": ["camera_front_wide_120fov", "camera_rear_left_70fov"],
                        "camera_cross_right_120fov": ["camera_front_wide_120fov", "camera_rear_right_70fov"],
                        "camera_rear_left_70fov": ["camera_cross_left_120fov", "camera_rear_tele_30fov"],
                        "camera_rear_right_70fov": ["camera_cross_right_120fov", "camera_rear_tele_30fov"],
                        "camera_rear_tele_30fov": ["camera_rear_left_70fov", "camera_rear_right_70fov"],
                        "camera_front_tele_30fov": ["camera_front_wide_120fov"],
                    },
                    camera_to_view_id={
                        "camera_front_wide_120fov": 0,
                        "camera_cross_left_120fov": 5,
                        "camera_cross_right_120fov": 1,
                        "camera_rear_left_70fov": 4,
                        "camera_rear_right_70fov": 2,
                        "camera_rear_tele_30fov": 3,
                        "camera_front_tele_30fov": 6,
                    },
                ),
            ),
        ),
        optimizer=dict(
            lr=3e-5,
            lr_overrides={
                r".*cross_view_attn.*": 1e-4,
            },
        ),
        trainer=dict(
            logging_iter=50,
            callbacks=dict(
                log_weight=L(LogWeight)(
                    every_n=50,
                ),
                every_n_sample_reg=dict(
                    every_n=1500,
                ),
            ),
        ),
    )


experiments = [
    buttercup_predict2p5_14b_7views_res720p_fps10_t8_fromavfinetune_allcaps_29frames_nofps_uniform(),
    buttercup_predict2p5_14b_crossview_7views_res720p_fps10_t8_fromavfinetune_allcaps_29frames_nofps_uniform(),
]

cs = ConfigStore.instance()

for _item in experiments:
    cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
