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

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.configs.video2world.experiment.reason_embeddings.model_2B_reason_1p1 import (
    T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16,
)
from cosmos_predict2._src.predict2.configs.video2world.experiment.reason_embeddings.stage3_2B import (
    I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY,
    I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_102_SIZE_2B_RES_480_FPS16_HQ_V5_from_26,
    build_debug_runs,
)
from cosmos_predict2._src.predict2.configs.video2world.experiment.specialized_model.SFT_2B_RF import (
    STAGE_C_PT_4_INDEX_2_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_WITH_EDM_CKPT,
)

"""
# run local debug & training
"""
MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16 = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-reason_embeddings-Index-102-Size-2B-Res-480-Fps-16-Note-HQ_V5_from_26_qwen_concat",
            {"override /data_train": "mock"},
            {"override /net": "cosmos_v1_2B_net_camera_conditioned"},
            {"override /conditioner": "camera_conditioned_video_conditioner"},
            {"override /model": "camera_conditioned_fsdp"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_video2video_2b_res_480_fps16",
        ),
        optimizer=dict(
            lr=1e-5,
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[1_000],
            cycle_lengths=[40_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720_aggressive",
                    ),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-102-Size-2B-Res-480-Fps-16-Note-HQ_V5_from_26_qwen_concat/checkpoints/iter_000122000/",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=40_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=False,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_720_FPS16 = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted",
            {"override /data_train": "mock"},
            {"override /net": "cosmos_v1_2B_net_camera_conditioned"},
            {"override /conditioner": "camera_conditioned_video_conditioner"},
            {"override /model": "camera_conditioned_fsdp"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_video2video_2b_res_720_fps16",
        ),
        optimizer=dict(
            lr=1e-5,
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.6],
            f_min=[0.3],
            warm_up_steps=[1_000],
            cycle_lengths=[40_000],
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        model=dict(
            config=dict(
                net=dict(
                    sac_config=dict(
                        mode="predict2_2b_720_aggressive",
                    ),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=40_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=False,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16 = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt",
            {"override /data_train": "local_multiview_train"},
            {"override /net": "cosmos_v1_2B_net_camera_conditioned"},
            {"override /conditioner": "camera_conditioned_video_conditioner"},
            {"override /model": "camera_conditioned_rectified_flow_fsdp"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_video2video_rectified_flow_2b_res_720_fps16",
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_GRPO-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-posttrain_data-HQ_V7_RF_MERGE_LOCAL_ag_every2_guidance0_scorekeyoverall_reward_databeta0.01_mincon0/checkpoints/iter_000000288",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=40_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=False,
                max_diff=1.5,
            ),
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16_S3_MULTICAM_SYNCAM = LazyDict(
    dict(
        defaults=[
            "/experiment/multicamera_video2video_2b_res_480_fps16",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_video2video_2b_res_480_fps16_s3_multicam_syncam",
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
    ),
    flags={"allow_objects": True},
)

MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_480_FPS16_S3_MULTICAM_SYNCAM = LazyDict(
    dict(
        defaults=[
            "/experiment/multicamera_video2video_rectified_flow_2b_res_720_fps16",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_video2video_rectified_flow_2b_res_480_fps16_s3_multicam_syncam",
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/multicamera_video2video_2b_res_720_fps16_s3_multicam_syncam/checkpoints/iter_000011000/",
        ),
    ),
    flags={"allow_objects": True},
)

MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_720_FPS16_S3_MULTICAM_SYNCAM = LazyDict(
    dict(
        defaults=[
            "/experiment/multicamera_video2video_2b_res_720_fps16",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_video2video_2b_res_720_fps16_s3_multicam_syncam",
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
    ),
    flags={"allow_objects": True},
)

MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16_S3_MULTICAM_SYNCAM = LazyDict(
    dict(
        defaults=[
            "/experiment/multicamera_video2video_rectified_flow_2b_res_720_fps16",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_multicam_syncam",
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/multicamera_video2video_2b_res_720_fps16_s3_multicam_syncam/checkpoints/iter_000011000/",
        ),
    ),
    flags={"allow_objects": True},
)

MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16_S3_AGIBOT = LazyDict(
    dict(
        defaults=[
            "/experiment/multicamera_video2video_2b_res_480_fps16",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_video2video_2b_res_480_fps16_s3_agibot",
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
    ),
    flags={"allow_objects": True},
)

MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_720_FPS16_S3_AGIBOT = LazyDict(
    dict(
        defaults=[
            "/experiment/multicamera_video2video_2b_res_720_fps16",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_video2video_2b_res_720_fps16_s3_agibot",
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
    ),
    flags={"allow_objects": True},
)

MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16_S3_AGIBOT = LazyDict(
    dict(
        defaults=[
            "/experiment/multicamera_video2video_rectified_flow_2b_res_720_fps16",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_agibot",
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/multicamera_video2video_2b_res_720_fps16_s3_agibot/checkpoints/iter_000015000/",
        ),
    ),
    flags={"allow_objects": True},
)

MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16_S3_AGIBOT_FRAMEINIT = LazyDict(
    dict(
        defaults=[
            "/experiment/multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_agibot",
            {"override /model": "camera_conditioned_frameinit_rectified_flow_fsdp"},
            {"override /conditioner": "camera_conditioned_frameinit_video_conditioner"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_agibot_frameinit",
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        checkpoint=dict(
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_agibot/checkpoints/iter_000003000",
        ),
    ),
    flags={"allow_objects": True},
)

MULTICAMERA_AR_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16_S3_MULTICAM_SYNCAM = LazyDict(
    dict(
        defaults=[
            "/experiment/multicamera_video2video_rectified_flow_2b_res_720_fps16",
            {"override /data_train": "mock"},
            {"override /conditioner": "camera_conditioned_ar_video_conditioner"},
            {"override /model": "camera_conditioned_ar_rectified_flow_fsdp"},
            {"override /net": "cosmos_v1_2B_net_camera_conditioned_ar"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_ar_video2video_rectified_flow_2b_res_720_fps16_s3_multicam_syncam",
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        checkpoint=dict(
            save_iter=100,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_multicam_syncam/checkpoints/iter_000002000/",
        ),
    ),
    flags={"allow_objects": True},
)


MULTICAMERA_AR_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_480_FPS16_S3_MULTICAM_SYNCAM_IN4OUT1 = LazyDict(
    dict(
        defaults=[
            "/experiment/multicamera_video2video_rectified_flow_2b_res_720_fps16",
            {"override /data_train": "s3_multiview_ar_train_multicam_syncam_480p_in4out1"},
            {"override /conditioner": "camera_conditioned_ar_video_conditioner"},
            {"override /model": "camera_conditioned_ar_rectified_flow_fsdp"},
            {"override /net": "cosmos_v1_2B_net_camera_conditioned_ar"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_ar_video2video_rectified_flow_2b_res_480_fps16_s3_multicam_syncam_in4out1",
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        checkpoint=dict(
            save_iter=200,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_multicam_syncam/checkpoints/iter_000002000/",
        ),
    ),
    flags={"allow_objects": True},
)


"""
# run s3 debug
"""
MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16_S3_DEBUG = LazyDict(
    dict(
        defaults=[
            "/experiment/multicamera_video2video_2b_res_480_fps16",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_video2video_2b_res_480_fps16_s3_debug",
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        checkpoint=dict(
            save_iter=1_00,
        ),
    ),
    flags={"allow_objects": True},
)

"""
# run webdataset debug
"""
MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16_WEBDATASET_DEBUG = LazyDict(
    dict(
        defaults=[
            "/experiment/multicamera_video2video_2b_res_480_fps16",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY["job"]["group"],
            name="multicamera_video2video_2b_res_480_fps16_webdataset_debug",
        ),
        dataloader_train=dict(
            batch_size=1,
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        trainer=dict(
            max_iter=1_000,  # Shorter for debug
            logging_iter=10,
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()

for _item, _item_wo_resume, _item_mock_wo_resume in [
    [
        I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY,
        *build_debug_runs(I2V_STAGE_C_PT_4_INDEX_3_SIZE_2B_RES_480_FPS16_QWEN_VIDEO_ONLY),
    ],
    [
        I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_102_SIZE_2B_RES_480_FPS16_HQ_V5_from_26,
        *build_debug_runs(I2V_REASON_EMBEDDINGS_STAGE_C_PT_4_INDEX_102_SIZE_2B_RES_480_FPS16_HQ_V5_from_26),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16,
        *build_debug_runs(T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16),
    ],
    [
        STAGE_C_PT_4_INDEX_2_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_WITH_EDM_CKPT,
        *build_debug_runs(STAGE_C_PT_4_INDEX_2_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_WITH_EDM_CKPT),
    ],
    [
        MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16,
        *build_debug_runs(MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16),
    ],
    [
        MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16_S3_MULTICAM_SYNCAM,
        *build_debug_runs(MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16_S3_MULTICAM_SYNCAM),
    ],
    [
        MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16_S3_AGIBOT,
        *build_debug_runs(MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16_S3_AGIBOT),
    ],
    [
        MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16_S3_DEBUG,
        *build_debug_runs(MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16_S3_DEBUG),
    ],
    [
        MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16_WEBDATASET_DEBUG,
        *build_debug_runs(MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_480_FPS16_WEBDATASET_DEBUG),
    ],
    [
        MULTICAMERA_AR_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16_S3_MULTICAM_SYNCAM,
        *build_debug_runs(MULTICAMERA_AR_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16_S3_MULTICAM_SYNCAM),
    ],
    [
        MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_480_FPS16_S3_MULTICAM_SYNCAM,
        *build_debug_runs(MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_480_FPS16_S3_MULTICAM_SYNCAM),
    ],
    [
        MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16_S3_MULTICAM_SYNCAM,
        *build_debug_runs(MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16_S3_MULTICAM_SYNCAM),
    ],
    [
        MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_720_FPS16,
        *build_debug_runs(MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_720_FPS16),
    ],
    [
        MULTICAMERA_AR_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_480_FPS16_S3_MULTICAM_SYNCAM_IN4OUT1,
        *build_debug_runs(MULTICAMERA_AR_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_480_FPS16_S3_MULTICAM_SYNCAM_IN4OUT1),
    ],
    [
        MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_720_FPS16_S3_MULTICAM_SYNCAM,
        *build_debug_runs(MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_720_FPS16_S3_MULTICAM_SYNCAM),
    ],
    [
        MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_720_FPS16_S3_AGIBOT,
        *build_debug_runs(MULTICAMERA_VIDEO2VIDEO_SIZE_2B_RES_720_FPS16_S3_AGIBOT),
    ],
    [
        MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16_S3_AGIBOT_FRAMEINIT,
        *build_debug_runs(MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16_S3_AGIBOT_FRAMEINIT),
    ],
    [
        MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16,
        *build_debug_runs(MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16),
    ],
    [
        MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16_S3_AGIBOT,
        *build_debug_runs(MULTICAMERA_VIDEO2VIDEO_RECTIFIED_FLOW_SIZE_2B_RES_720_FPS16_S3_AGIBOT),
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
