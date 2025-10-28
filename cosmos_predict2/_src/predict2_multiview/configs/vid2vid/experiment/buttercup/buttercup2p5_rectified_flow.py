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
from cosmos_predict2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)

"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2_multiview/configs/vid2vid/config.py -- experiment=buttercup_predict2p5_2b_mv_7views_res720p_fps10_t8_frompred2p1multicaps8k_jointalpamayov2mads720pmulticaps job.project="debug_p2_mv" job.name="debug" checkpoint.load_path=""
"""


def buttercup_predict2p5_2b_mv_7views_res720p_fps10_t8_frompred2p1multicaps8k_jointalpamayov2mads720pmulticaps():
    sample_n_views = 7
    return dict(
        defaults=[
            "/experiment/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only",
            {"override /data_train": "video_joint_alpamayov2_mads_720p_hybrid_captions"},
            {"override /conditioner": "video_prediction_multiview_conditioner"},
            {"override /model": "fsdp_rectified_flow_multiview"},
            {"override /net": "cosmos_v1_2B_multiview"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_mv_7views_res720p_fps10_t8_frompred2p1multicaps8k_jointalpamayov2mads720pmulticaps",
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_from42p5k_jointalpamayov2mads720p_multiwindow_multicaptions-0/checkpoints/iter_000010000/",
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames_per_view=0,  # t2w
                max_num_conditional_frames_per_view=2,  # i2w or v2v
                condition_locations=["first_random_n"],
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                state_t=8,
                online_text_embeddings_as_dict=False,  # For backward compatibility with old experiments
                net=dict(
                    concat_view_embedding=True,
                    view_condition_dim=7,
                    state_t=8,
                    n_cameras_emb=7,
                    rope_enable_fps_modulation=True,  # Check this in base model
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=8.0 / 24.0,
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                ),
                resolution="720p",  # Updated from 720 to get resolution 720 x 1280 instead of 704 x 1280
            ),
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
                    fps=10,
                    sample_n_views=sample_n_views,
                ),
                every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=2_000,
                    do_x0_prediction=False,
                    is_ema=True,
                    num_sampling_step=35,
                    guidance=[0, 3, 7],
                    fps=10,
                    sample_n_views=sample_n_views,
                ),
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                alpamayo=dict(
                    ratio=10,
                    dataloader=dict(
                        dataset=dict(
                            embedding_type=None,
                        ),
                    ),
                ),
                mads=dict(
                    ratio=1,
                    dataloader=dict(
                        dataset=dict(
                            embedding_type=None,
                        ),
                    ),
                ),
                # Disable base model dataloaders
                image_data=dict(
                    ratio=0,
                ),
                video_data=dict(
                    dataloader=dict(),
                    ratio=0,
                ),
            ),
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2_multiview/configs/vid2vid/config.py -- experiment=buttercup_predict2p5_2b_mv_7views_res720p_fps15_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps57framesto29 job.project="debug_p2_mv" job.name="debug" checkpoint.load_path=""
"""


def buttercup_predict2p5_2b_mv_7views_res720p_fps15_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps57framesto29():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_2b_mv_7views_res720p_fps10_t8_frompred2p1multicaps8k_jointalpamayov2mads720pmulticaps",
            {"override /data_train": "video_joint_alpamayov2_mads_720p_57framesto29_hybrid_captions"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_mv_7views_res720p_fps15_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps57framesto29",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res720p_fps10_t8_frompred2p1multicaps8k_jointalpamayov2mads720pmulticaps-0/checkpoints/iter_000016000/",
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    fps=15,
                ),
                every_n_sample_ema=dict(
                    fps=15,
                ),
            ),
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2_multiview/configs/vid2vid/config.py -- experiment=buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames job.project="debug_p2_mv" job.name="debug" checkpoint.load_path=""
"""


def buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_2b_mv_7views_res720p_fps10_t8_frompred2p1multicaps8k_jointalpamayov2mads720pmulticaps",
            {"override /data_train": "video_joint_alpamayov2_mads_720p_29frames_hybrid_captions"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res720p_fps10_t8_frompred2p1multicaps8k_jointalpamayov2mads720pmulticaps-0/checkpoints/iter_000016000/",
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    fps=30,
                ),
                every_n_sample_ema=dict(
                    fps=30,
                ),
            ),
        ),
    )


def buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames_nofps():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames_nofps",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames-0/checkpoints/iter_000048000/",
        ),
        model=dict(
            config=dict(
                net=dict(
                    rope_enable_fps_modulation=False,
                ),
            ),
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2_multiview/configs/vid2vid/config.py -- experiment=buttercup_predict2p5_2b_mv_7views_res480p_fps30_t16_from16kfps10mv_jointalpamayov2mads480pmulticaps61frames job.project="debug_p2_mv" job.name="debug" checkpoint.load_path=""
"""


def buttercup_predict2p5_2b_mv_7views_res480p_fps30_t16_from16kfps10mv_jointalpamayov2mads480pmulticaps61frames():
    sample_n_views = 7
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_2b_mv_7views_res720p_fps10_t8_frompred2p1multicaps8k_jointalpamayov2mads720pmulticaps",
            {"override /data_train": "video_joint_alpamayov2_mads_480p_61frames_hybrid_captions"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_mv_7views_res480p_fps30_t16_from16kfps10mv_jointalpamayov2mads480pmulticaps61frames",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res720p_fps10_t8_frompred2p1multicaps8k_jointalpamayov2mads720pmulticaps-0/checkpoints/iter_000016000/",
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames_per_view=0,  # t2w
                max_num_conditional_frames_per_view=2,  # i2w or v2v
                condition_locations=["first_random_n"],
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                state_t=16,
                online_text_embeddings_as_dict=False,  # For backward compatibility with old experiments
                net=dict(
                    concat_view_embedding=True,
                    view_condition_dim=7,
                    state_t=16,
                    n_cameras_emb=7,
                    rope_enable_fps_modulation=True,
                    rope_h_extrapolation_ratio=2.0,
                    rope_w_extrapolation_ratio=2.0,
                    rope_t_extrapolation_ratio=16.0 / 24.0,
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                ),
                resolution="480p",
            ),
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
        dataloader_train=dict(
            dataloaders=dict(
                alpamayo=dict(
                    ratio=10,
                    dataloader=dict(
                        dataset=dict(
                            embedding_type=None,
                        ),
                    ),
                ),
                mads=dict(
                    ratio=1,
                    dataloader=dict(
                        dataset=dict(
                            embedding_type=None,
                        ),
                    ),
                ),
                # Disable base model dataloaders
                image_data=dict(
                    ratio=0,
                ),
                video_data=dict(
                    dataloader=dict(),
                    ratio=0,
                ),
            ),
        ),
    )


def buttercup_predict2p5_2b_mv_7views_res720p_fps15_t8_frombase2p5_jointalpamayov2mads720pmulticaps57framesto29():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_2b_mv_7views_res720p_fps15_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps57framesto29",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_mv_7views_res720p_fps15_t8_frombase2p5_jointalpamayov2mads720pmulticaps57framesto29",
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2/checkpoints/iter_000023000/",
        ),
    )


def buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_frombase2p5_jointalpamayov2mads720pmulticaps29frames():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_frombase2p5_jointalpamayov2mads720pmulticaps29frames",
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2/checkpoints/iter_000023000/",
        ),
    )


def buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_frombase2p5_alpamayov2_720p_allviewcapswithprefix_29frames():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames",
            {"override /data_train": "alpamayo_v2_7cameras_tar_sample7views_29frames_res720p_norepeat_hybrid_captions"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_frombase2p5_alpamayov2_720p_allviewcapswithprefix_29frames",
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2/checkpoints/iter_000023000/",
        ),
    )


def buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from28kfps30mv_alpamayov2_720p_allviewcapswithprefix_29frames():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames",
            {"override /data_train": "alpamayo_v2_7cameras_tar_sample7views_29frames_res720p_norepeat_hybrid_captions"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from28kfps30mv_alpamayov2_720p_allviewcapswithprefix_29frames",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames-0/checkpoints/iter_000028500/",
        ),
        model=dict(
            config=dict(
                conditional_frames_probs={0: 0.4, 1: 0.4, 2: 0.2},
            ),
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2_multiview/configs/vid2vid/config.py -- experiment=buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames job.project="debug_p2_mv" job.name="debug" checkpoint.load_path=""
"""


def buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames",
            {
                "override /data_train": "video_joint_alpamayo1capnoviewprefix_allcapsviewprefix_720p_29frames_hybrid_captions"
            },
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames-0/checkpoints/iter_000048000/",
        ),
        model=dict(
            config=dict(
                conditional_frames_probs={0: 0.4, 1: 0.4, 2: 0.2},
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                alpamayo_1cap=dict(
                    ratio=1,
                    dataloader=dict(
                        dataset=dict(
                            embedding_type=None,
                        ),
                    ),
                ),
                alpamayo_allcaps=dict(
                    ratio=1,
                    dataloader=dict(
                        dataset=dict(
                            embedding_type=None,
                        ),
                    ),
                ),
                # Disable base model dataloaders
                alpamayo=dict(
                    ratio=0,
                ),
                mads=dict(
                    ratio=0,
                ),
                image_data=dict(
                    ratio=0,
                ),
                video_data=dict(
                    dataloader=dict(),
                    ratio=0,
                ),
            ),
        ),
    )


def buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames_nofps():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames_nofps",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames-0/checkpoints/iter_000008500/",
        ),
        model=dict(
            config=dict(
                net=dict(
                    rope_enable_fps_modulation=False,
                ),
            ),
        ),
    )


def buttercup_predict2p5_2b_7views_res720p_fps30_t8_joint_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform_dropoutt0():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames",
            {
                "override /data_train": "video_joint_alpamayo1capviewprefix_allcapsviewprefix_720p_29frames_hybrid_captions"
            },
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_7views_res720p_fps30_t8_joint_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform_dropoutt0",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames_nofps-0/checkpoints/iter_000005000",
        ),
        model=dict(
            config=dict(
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                train_time_weight="uniform",
                net=dict(
                    rope_enable_fps_modulation=False,
                ),
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.0,
                        use_empty_string=False,
                    ),
                ),
            ),
        ),
    )


def buttercup_predict2p5_2b_7views_res480p_fps30_t16_from41kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_61frames_nofps():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_2b_mv_7views_res480p_fps30_t16_from16kfps10mv_jointalpamayov2mads480pmulticaps61frames",
            {
                "override /data_train": "video_joint_alpamayo1capnoviewprefix_allcapsviewprefix_480p_61frames_hybrid_captions"
            },
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p5_2b_7views_res480p_fps30_t16_from41kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_61frames_nofps",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res480p_fps30_t16_from16kfps10mv_jointalpamayov2mads480pmulticaps61frames-0/checkpoints/iter_000041000/",
        ),
        model=dict(
            config=dict(
                conditional_frames_probs={0: 0.4, 1: 0.4, 2: 0.2},
                net=dict(
                    rope_enable_fps_modulation=False,
                ),
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                alpamayo_1cap=dict(
                    ratio=1,
                    dataloader=dict(
                        dataset=dict(
                            embedding_type=None,
                        ),
                    ),
                ),
                alpamayo_allcaps=dict(
                    ratio=1,
                    dataloader=dict(
                        dataset=dict(
                            embedding_type=None,
                        ),
                    ),
                ),
                # Disable base model dataloaders
                alpamayo=dict(
                    ratio=0,
                ),
                mads=dict(
                    ratio=0,
                ),
                image_data=dict(
                    ratio=0,
                ),
                video_data=dict(
                    dataloader=dict(),
                    ratio=0,
                ),
            ),
        ),
    )


# Note, use GB200 cluster for this config, as it OOMs on H100 clusters.
# For context parallelism of 24, use multiples of 24 GPUs or 6 nodes (since 4 GB200 per node).
def buttercup_predict2p5_2b_mv_7views_res720p_fps30_t24_joint_alpamayo1capviewprefix_allcapsviewprefix_93frames_nofps_uniform_dropoutt0():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames",
            {
                "override /data_train": "video_joint_alpamayo1capviewprefix_allcapsviewprefix_720p_93frames_hybrid_captions"
            },
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
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                train_time_weight="uniform",
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.0,
                        use_empty_string=False,
                    ),
                ),
                state_t=24,
                net=dict(
                    state_t=24,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24.0,
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
            dataloaders=dict(
                alpamayo_1cap=dict(
                    ratio=1,
                    dataloader=dict(
                        dataset=dict(
                            embedding_type=None,
                        ),
                    ),
                ),
                alpamayo_allcaps=dict(
                    ratio=1,
                    dataloader=dict(
                        dataset=dict(
                            embedding_type=None,
                        ),
                    ),
                ),
                # Disable base model dataloaders
                alpamayo=dict(
                    ratio=0,
                ),
                mads=dict(
                    ratio=0,
                ),
                image_data=dict(
                    ratio=0,
                ),
                video_data=dict(
                    dataloader=dict(),
                    ratio=0,
                ),
            ),
        ),
    )


experiments = [
    buttercup_predict2p5_2b_mv_7views_res720p_fps10_t8_frompred2p1multicaps8k_jointalpamayov2mads720pmulticaps(),
    buttercup_predict2p5_2b_mv_7views_res720p_fps15_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps57framesto29(),
    buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames(),
    buttercup_predict2p5_2b_mv_7views_res480p_fps30_t16_from16kfps10mv_jointalpamayov2mads480pmulticaps61frames(),
    buttercup_predict2p5_2b_mv_7views_res720p_fps15_t8_frombase2p5_jointalpamayov2mads720pmulticaps57framesto29(),
    buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_frombase2p5_jointalpamayov2mads720pmulticaps29frames(),
    buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_frombase2p5_alpamayov2_720p_allviewcapswithprefix_29frames(),
    buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from28kfps30mv_alpamayov2_720p_allviewcapswithprefix_29frames(),
    buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames(),
    buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames_nofps(),
    buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames_nofps(),
    buttercup_predict2p5_2b_7views_res480p_fps30_t16_from41kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_61frames_nofps(),
    buttercup_predict2p5_2b_mv_7views_res720p_fps30_t24_joint_alpamayo1capviewprefix_allcapsviewprefix_93frames_nofps_uniform_dropoutt0(),
    buttercup_predict2p5_2b_7views_res720p_fps30_t8_joint_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform_dropoutt0(),
]

cs = ConfigStore.instance()

for _item in experiments:
    cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
