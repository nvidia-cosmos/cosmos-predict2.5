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

from copy import deepcopy

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.configs.video2world.experiment.reason_embeddings.model_2B_reason_1p1_rectified_flow import (
    T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW,
    T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_IMPROVED,
)

# continual training w/ base RF config
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_FACE_FOCUSED_RF_ONLY2 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_human_only_face_focused_20250607_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_face_only2",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-3-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_face_focused_formal/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
    ),
    flags={"allow_objects": True},
)
# continual training w/ base RF config + high sigma
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_FACE_FOCUSED_RF_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_human_only_face_focused_20250607_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_face_high_sigma",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-3-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_face_focused_formal/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
        ),
    ),
    flags={"allow_objects": True},
)
# continual training w/ base RF config + high sigma + start w/ 23k base + fix conditional frame timestep discrependcy (MR 5033)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_FACE_FOCUSED_RF_HIGH_SIGMA2 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_human_only_face_focused_20250607_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_face_high_sigma2",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2/checkpoints/iter_000023000",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1000,
                ),
                every_n_sample_ema=dict(
                    every_n=1000,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)
# continual training w/ base RF config + new prompt image data + no drop frame augmentor + gt720p resolution
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_FACE_FOCUSED_RF = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_IMPROVED['job']['name']}",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_face",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-3-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_face_focused_formal/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
        dataloader_train=dict(
            dataloaders=dict(
                video_data=dict(
                    dataloader=dict(
                        dataset=dict(
                            dataset_name="cosmos_posttraining_hq_v7_human_only_face_focused_20250607_video_whole",
                        )
                    )
                )
            )
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_CROWDED_RF_ONLY2 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_human_only_crowded_20250607_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_crowded_only2",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_crowded_formal/checkpoints/iter_000032000/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_CROWDED_RF_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_human_only_crowded_20250607_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_crowded_high_sigma",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_crowded_formal/checkpoints/iter_000032000/",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_CROWDED_RF_HIGH_SIGMA2 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_human_only_crowded_20250607_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_crowded_high_sigma2",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2/checkpoints/iter_000023000",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1000,
                ),
                every_n_sample_ema=dict(
                    every_n=1000,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_CROWDED_RF = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_IMPROVED['job']['name']}",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_crowded",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_crowded_formal/checkpoints/iter_000032000/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
        dataloader_train=dict(
            dataloaders=dict(
                video_data=dict(
                    dataloader=dict(
                        dataset=dict(
                            dataset_name="cosmos_posttraining_hq_v7_human_only_crowded_20250607_video_whole",
                        )
                    )
                )
            )
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_HIGH_MOTION_RF_ONLY2 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_human_only_high_motion_20250607_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_high_motion_only2",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-4-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_high_motion_formal/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_HIGH_MOTION_RF_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_human_only_high_motion_20250607_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_high_motion_high_sigma",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-4-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_high_motion_formal/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_HIGH_MOTION_RF = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_IMPROVED['job']['name']}",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_high_motion",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-4-Size-2B-Res-720-Fps-16-Note-HQ_V7_human_only_high_motion_formal/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
        dataloader_train=dict(
            dataloaders=dict(
                video_data=dict(
                    dataloader=dict(
                        dataset=dict(
                            dataset_name="cosmos_posttraining_hq_v7_human_only_high_motion_20250607_video_whole",
                        )
                    )
                )
            )
        ),
    ),
    flags={"allow_objects": True},
)

STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_ROBOTICS_RF_ONLY2 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_robotics_only_20250717_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_robotics_only2",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-101-Size-2B-Res-720-Fps-16-Note-HQ_V7_robotics_formal/checkpoints/iter_000028000/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
    ),
    flags={"allow_objects": True},
)

STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_ROBOTICS_RF_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_robotics_only_20250717_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_robotics_high_sigma",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-101-Size-2B-Res-720-Fps-16-Note-HQ_V7_robotics_formal/checkpoints/iter_000028000/",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_ROBOTICS_RF_HIGH_SIGMA2 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250702_video_cosmos_posttraining_hq_v7_robotics_only_20250717_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_robotics_high_sigma2",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2/checkpoints/iter_000023000",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1000,
                ),
                every_n_sample_ema=dict(
                    every_n=1000,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_ROBOTICS_RF = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_IMPROVED['job']['name']}",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_robotics",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-101-Size-2B-Res-720-Fps-16-Note-HQ_V7_robotics_formal/checkpoints/iter_000028000/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
        dataloader_train=dict(
            dataloaders=dict(
                video_data=dict(
                    dataloader=dict(
                        dataset=dict(
                            dataset_name="cosmos_posttraining_hq_v7_robotics_only_20250717_video_whole",
                        )
                    )
                )
            )
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_AV_RF_ONLY2 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_av_only_20250717_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_av_only2",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-301-Size-2B-Res-720-Fps-16-Note-HQ_V7_av_formal/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_AV_RF_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_av_only_20250717_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_av_high_sigma",
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-301-Size-2B-Res-720-Fps-16-Note-HQ_V7_av_formal/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_AV_RF = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_IMPROVED['job']['name']}",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_av",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-301-Size-2B-Res-720-Fps-16-Note-HQ_V7_av_formal/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
        dataloader_train=dict(
            dataloaders=dict(
                video_data=dict(
                    dataloader=dict(
                        dataset=dict(
                            dataset_name="cosmos_posttraining_hq_v7_av_only_20250717_video_whole",
                        )
                    )
                )
            )
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_PHYSICAL_AI_RF_ONLY2 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_physical_ai_20250812_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_physical_ai_only2",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-401-Size-2B-Res-720-Fps-16-Note-HQ_V7_physical_ai_formal/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_PHYSICAL_AI_RF_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_physical_ai_20250812_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_physical_ai_high_sigma",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-401-Size-2B-Res-720-Fps-16-Note-HQ_V7_physical_ai_formal/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_PHYSICAL_AI_RF = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_IMPROVED['job']['name']}",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_physical_ai",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-Index-401-Size-2B-Res-720-Fps-16-Note-HQ_V7_physical_ai_formal/checkpoints/iter_000030000/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
        dataloader_train=dict(
            dataloaders=dict(
                video_data=dict(
                    dataloader=dict(
                        dataset=dict(
                            dataset_name="cosmos_posttraining_hq_v7_physical_ai_20250812_video_whole",
                        )
                    )
                )
            )
        ),
    ),
    flags={"allow_objects": True},
)

STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_4KCOOLDOWN_RF_ONLY2 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_4K_20250812_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_4k_cooldown_only2",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_4K_cooldown_from_10K/checkpoints/iter_000047500/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
        scheduler=dict(
            f_max=[0.4],
            f_min=[0.0],
            warm_up_steps=[0],
            cycle_lengths=[50_000],
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_4KCOOLDOWN_RF_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_4K_20250812_s3"
            },
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_4k_cooldown_high_sigma",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_4K_cooldown_from_10K/checkpoints/iter_000047500/",
            load_training_state=False,
            strict_resume=True,
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
        trainer=dict(
            logging_iter=20,
        ),
        scheduler=dict(
            f_max=[0.4],
            f_min=[0.0],
            warm_up_steps=[0],
            cycle_lengths=[50_000],
        ),
    ),
    flags={"allow_objects": True},
)
STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_4KCOOLDOWN_RF = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_IMPROVED['job']['name']}",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-1-Size-2B-Res-720-Fps-16-Note-rf_4k_cooldown",
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_4K_cooldown_from_10K/checkpoints/iter_000047500/",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            logging_iter=20,
        ),
        dataloader_train=dict(
            dataloaders=dict(
                video_data=dict(
                    dataloader=dict(
                        dataset=dict(
                            dataset_name="cosmos_posttraining_hq_v7_4K_20250812_video_whole",
                        )
                    )
                )
            )
        ),
    ),
    flags={"allow_objects": True},
)

# This is not for training, but a config to load the edm ckpt (e.g. the merged predict 2.5, or edm sft ckpt etc)
# and add the needed overrides to load the edm ckpt and run rf inference
STAGE_C_PT_4_INDEX_2_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_WITH_EDM_CKPT = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW['job']['name']}",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt",
        ),
        model=dict(
            config=dict(
                conditional_frame_timestep=0.1,
                use_kerras_sigma_at_inference=True,
            )
        ),
    )
)

variants = [
    (
        "variant1",  # smaller lr
        {
            "scheduler.f_max": [0.4],
            "scheduler.f_min": [0.1],
        },
    ),
]


def apply_kv_to_config(config, key, value):
    """
    The key is dot seperated, e.g. "model.config.sde.p_mean"
    Creates and returns a new config with only the specified key modified.
    When the full key path is in the config, the value is updated.
    Otherwise, the key is added to the config.
    """
    # Create a deep copy of the config
    new_config = deepcopy(config)

    parts = key.split(".")
    current = new_config

    # Navigate to the parent of the final attribute, creating missing parts as needed
    for i, part in enumerate(parts[:-1]):
        if isinstance(current, (dict, LazyDict)):
            if part not in current:
                # Create a new LazyDict for missing intermediate keys
                current[part] = dict()
            current = current[part]
        else:
            # For object attributes
            if not hasattr(current, part):
                # Create a new LazyDict for missing intermediate attributes
                setattr(current, part, dict())
            current = getattr(current, part)

    # Set the value on the final attribute
    final_key = parts[-1]
    if isinstance(current, (dict, LazyDict)):
        current[final_key] = value
    else:
        setattr(current, final_key, value)

    return new_config


cs = ConfigStore.instance()

for _item in [
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_FACE_FOCUSED_RF,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_CROWDED_RF,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_HIGH_MOTION_RF,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_ROBOTICS_RF,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_AV_RF,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_PHYSICAL_AI_RF,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_4KCOOLDOWN_RF,
    # new set of exp that only do RF, 8/29
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_FACE_FOCUSED_RF_ONLY2,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_CROWDED_RF_ONLY2,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_HIGH_MOTION_RF_ONLY2,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_ROBOTICS_RF_ONLY2,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_AV_RF_ONLY2,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_PHYSICAL_AI_RF_ONLY2,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_4KCOOLDOWN_RF_ONLY2,
    # with high sigma
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_FACE_FOCUSED_RF_HIGH_SIGMA,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_CROWDED_RF_HIGH_SIGMA,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_HIGH_MOTION_RF_HIGH_SIGMA,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_ROBOTICS_RF_HIGH_SIGMA,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_AV_RF_HIGH_SIGMA,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_PHYSICAL_AI_RF_HIGH_SIGMA,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_4KCOOLDOWN_RF_HIGH_SIGMA,
    # new high sigma w/ 23k base + fix conditional frame timestep discrependcy (MR 5033)
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_FACE_FOCUSED_RF_HIGH_SIGMA2,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_HUMAN_ONLY_CROWDED_RF_HIGH_SIGMA2,
    STAGE_C_PT_4_INDEX_1_SIZE_2B_RES_720_FPS16_ROBOTICS_RF_HIGH_SIGMA2,
    STAGE_C_PT_4_INDEX_2_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW_WITH_EDM_CKPT,
]:
    log.info(f"Storing {_item['job']['name']}")
    cs.store(group="experiment", package="_global_", name=f"{_item['job']['name']}", node=_item)
    for variant_name, variant_config in variants:
        _item_variant = deepcopy(_item)
        # Apply all overrides from variant_config
        # import pdb; pdb.set_trace()
        for key, value in variant_config.items():
            _item_variant = apply_kv_to_config(_item_variant, key, value)

        # Update the job name to include the variant
        _item_variant["job"]["name"] = f"{_item['job']['name']}_{variant_name}"

        # Store the variant configuration
        cs.store(group="experiment", package="_global_", name=f"{_item_variant['job']['name']}", node=_item_variant)
