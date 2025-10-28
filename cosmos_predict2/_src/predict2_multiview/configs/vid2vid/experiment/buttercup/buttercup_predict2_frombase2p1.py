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
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2_multiview/configs/vid2vid/config.py -- experiment=buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_alpamayo85to29res720p job.project="debug_p2_mv" job.name="debug" checkpoint.load_path=""
"""


def buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_alpamayo85to29res720p():
    sample_n_views = 7
    return dict(
        defaults=[
            "/experiment/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted",
            {
                "override /data_train": "alpamayo_v2_7cameras_tar_sample7views_85framesto29_res720p_noviewprefix_1cap_norepeat"
            },
            {"override /conditioner": "video_prediction_multiview_conditioner"},
            {"override /model": "fsdp_multiview"},
            {"override /net": "cosmos_v1_2B_multiview"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_alpamayo85to29res720p",
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000045000",
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
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2_multiview/configs/vid2vid/config.py -- experiment=buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_mads720p job.project="debug_p2_mv" job.name="debug" checkpoint.load_path=""
"""


def buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_mads720p():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_alpamayo85to29res720p",
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_mv_20250710_720p_s3"},
            "_self_",
        ],
        job=dict(
            name="buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_mads720p",
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2_multiview/configs/vid2vid/config.py -- experiment=buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter10k_jointalpamayov2mads720p job.project="debug_p2_mv" job.name="debug" checkpoint.load_path=""
"""


def buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_jointalpamayov2mads720p():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_alpamayo85to29res720p",
            {"override /data_train": "video_joint_alpamayov2_mads_720p_hybrid_captions"},
            "_self_",
        ],
        job=dict(
            name="buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_jointalpamayov2mads720p",
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


def buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_from23klr1x_jointalpamayov2mads720p_lr2x():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_jointalpamayov2mads720p",
            "_self_",
        ],
        job=dict(
            name="buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_from23klr1x_jointalpamayov2mads720p_lr2x",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_jointalpamayov2mads720p-0/checkpoints/iter_000023000/",
        ),
        optimizer=dict(
            lr=2 ** (-13.5),
        ),
    )


def buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_from42p5k_jointalpamayov2mads720p_multiwindow_multicaptions():
    return dict(
        defaults=[
            "/experiment/buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_jointalpamayov2mads720p",
            "_self_",
        ],
        job=dict(
            name="buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_from42p5k_jointalpamayov2mads720p_multiwindow_multicaptions",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_jointalpamayov2mads720p-0/checkpoints/iter_000042500/",
        ),
    )


experiments = [
    buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_alpamayo85to29res720p(),
    buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_mads720p(),
    buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_jointalpamayov2mads720p(),
    buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_from23klr1x_jointalpamayov2mads720p_lr2x(),
    buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_from42p5k_jointalpamayov2mads720p_multiwindow_multicaptions(),
]

cs = ConfigStore.instance()

for _item in experiments:
    cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
