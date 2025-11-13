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

"""Experiments for vanilla attention training on 2B model.

All experiments are using datasets located on AWS S3 unless specified.
If you want to use datasets on GCS, change the name of the override /data_train from e.g.
video_alpamayo_dec2024_s3_720p_10fps_93frames_7views -> video_alpamayo_dec2024_gcs_720p_10fps_93frames_7views
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from cosmos_predict2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)


def buttercup_predict2p5_2b_7views_res720p_fps30_t8_joint_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform_dropoutt0() -> (
    dict
):
    sample_n_views = 7
    state_t = 8
    return dict(
        defaults=[
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {"override /callbacks": ["basic", "viz_online_sampling", "wandb", "cluster_speed"]},
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            {"override /data_train": "video_alpamayo_dec2024_s3_720p_30fps_29frames_7views"},
            {"override /conditioner": "video_prediction_multiview_conditioner"},
            {"override /model": "fsdp_rectified_flow_multiview"},
            {"override /net": "cosmos_v1_2B_multiview"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_7views_res720p_fps30_t8_joint_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform_dropoutt0",
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
        checkpoint=dict(
            load_from_object_store=dict(
                enabled=True,
            ),
            save_to_object_store=dict(
                enabled=True,
            ),
            save_iter=500,
            load_path="cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames_nofps-0/checkpoints/iter_000005000",
        ),
        trainer=dict(
            callbacks=dict(
                compile_tokenizer=dict(
                    enabled=False,
                ),
                iter_speed=dict(
                    hit_thres=50,
                    every_n=100,
                ),
                every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=2_000,
                    do_x0_prediction=False,
                    is_ema=False,
                    num_sampling_step=35,
                    guidance=[0, 3, 7],
                    fps=30,
                    sample_n_views=sample_n_views,
                ),
                every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=2_000,
                    do_x0_prediction=False,
                    is_ema=True,
                    num_sampling_step=35,
                    guidance=[0, 3, 7],
                    fps=30,
                    sample_n_views=sample_n_views,
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                condition_locations=["first_random_n"],
                fsdp_shard_size=8,
                resolution="720p",
                state_t=state_t,
                shift=5,
                use_dynamic_shift=False,
                train_time_weight="uniform",
                train_time_distribution="logitnormal",
                online_text_embeddings_as_dict=False,
                net=dict(
                    concat_view_embedding=True,
                    view_condition_dim=7,
                    state_t=8,
                    n_cameras_emb=7,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=float(state_t) / 24,
                    timestep_scale=0.001,
                    sac_config=dict(
                        mode="predict2_2b_720",
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
                        dropout_rate=0.0,
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
                ),
            ),
        ),
        dataloader_train=dict(
            augmentation_config=dict(
                single_caption_camera_name="camera_front_wide_120fov",
                add_view_prefix_to_caption=True,
            )
        ),
    )


# Note, use GB200 cluster for this config, as it OOMs on H100 clusters.
# For context parallelism of 24, use multiples of 24 GPUs or 6 nodes (since 4 GB200 per node).
def buttercup_predict2p5_2b_mv_7views_res720p_fps30_t24_joint_alpamayo1capviewprefix_allcapsviewprefix_93frames_nofps_uniform_dropoutt0() -> (
    dict
):
    state_t = 24
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_2b_7views_res720p_fps30_t8_joint_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform_dropoutt0",
            {"override /data_train": "video_alpamayo_dec2024_gcs_720p_30fps_93frames_7views"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_mv_7views_res720p_fps30_t24_joint_alpamayo1capviewprefix_allcapsviewprefix_93frames_nofps_uniform_dropoutt0",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames_nofps-0/checkpoints/iter_000005000",
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                state_t=state_t,
                net=dict(
                    state_t=state_t,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=state_t / 24.0,
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                ),
            ),
        ),
        trainer=dict(
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        dataloader_train=dict(
            augmentation_config=dict(
                single_caption_camera_name="camera_front_wide_120fov",
                add_view_prefix_to_caption=True,
            ),
        ),
    )


experiments = [
    buttercup_predict2p5_2b_7views_res720p_fps30_t8_joint_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform_dropoutt0(),
    buttercup_predict2p5_2b_mv_7views_res720p_fps30_t24_joint_alpamayo1capviewprefix_allcapsviewprefix_93frames_nofps_uniform_dropoutt0(),
]

cs = ConfigStore.instance()

for _item in experiments:
    cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
