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

import functools

from cosmos_predict2._src.imaginaire.utils.checkpoint_db import (
    CheckpointConfig,
    CheckpointDirHf,
    CheckpointDirS3,
    CheckpointFileHf,
    CheckpointFileS3,
    register_checkpoint,
)


@functools.cache
def register_checkpoints():
    from cosmos_oss.checkpoints import register_checkpoints as _register_checkpoints

    _register_checkpoints()

    register_checkpoint(
        CheckpointConfig(
            uuid="d20b7120-df3e-4911-919d-db6e08bad31c",
            name="nvidia/Cosmos-Predict2.5-2B/base/pre-trained",
            experiment="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2/checkpoints/iter_000023000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Predict2.5-2B",
                revision="15a82a2ec231bc318692aa0456a36537c806e7d4",
                filename="base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",
            ),
        ),
    )

    checkpoint_hf = CheckpointDirHf(
        repository="nvidia/Cosmos-Experimental",
        revision="eda2f0ca1db6281c9a960908bb6bf14607a0fea0",
        subdirectory="308eb96c-c4c0-4a06-9cc1-103a43beff28",
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="308eb96c-c4c0-4a06-9cc1-103a43beff28",
            name="nvidia/Cosmos-Predict2.5-2B/base/pre-trained",
            experiment="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/model",
            ),
            hf=checkpoint_hf,
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="7bbc8d06-2bc9-448d-94ee-b48b4ab7189c",
            name="nvidia/Cosmos-Predict2.5-2B/interactive",
            experiment="cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_sf_warmup",
            s3=CheckpointFileS3(
                uri="s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000014000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Experimental",
                revision="2b5e9a99b58d5a61259ca99962c4c74127481006",
                filename="7bbc8d06-2bc9-448d-94ee-b48b4ab7189c/model_ema_bf16.pt",
            ),
        ),
    )
    register_checkpoint(
        CheckpointConfig(
            uuid="bedc35da-1a54-4144-83db-6072c29b0fd9",
            name="nvidia/Cosmos-Predict2.5-2B/interactive",
            experiment="cosmos_predict2p5_2B_action_gr00t_gr1_warmup",
            s3=CheckpointFileS3(
                uri="s3://bucket/cosmos_predict2_action_conditioned/interactive_warmup/gr1/checkpoints/iter_000020000/model"
            ),
            hf=CheckpointDirHf(
                repository="nvidia/Cosmos-Experimental",
                revision="ded876a5b2e19aef64cd9d1100c03e5b05cf2f9c",
                subdirectory="bedc35da-1a54-4144-83db-6072c29b0fd9",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="81edfebe-bd6a-4039-8c1d-737df1a790bf",
            name="nvidia/Cosmos-Predict2.5-2B/base/post-trained",
            experiment="Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointFileS3(
                uri="s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_GRPO-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-posttrain_data-HQ_V7_RF_MERGE_LOCAL_ag_every2_guidance0_scorekeyoverall_reward_databeta0.01_mincon0/checkpoints/iter_000000288/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Predict2.5-2B",
                revision="15a82a2ec231bc318692aa0456a36537c806e7d4",
                filename="base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="524af350-2e43-496c-8590-3646ae1325da",
            name="nvidia/Cosmos-Predict2.5-2B/auto/multiview",
            experiment="buttercup_predict2p5_2b_7views_res720p_fps30_t8_joint_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform_dropoutt0",
            metadata={
                "resolution": "720p",
                "fps": 30,
                "views": 7,
                "frames": 29,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_7views_res720p_fps30_t8_joint_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform_dropoutt0-0/checkpoints/iter_000012000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Predict2.5-2B",
                revision="865baf084d4c9e850eac59a021277d5a9b9e8b63",
                filename="auto/multiview/524af350-2e43-496c-8590-3646ae1325da_ema_bf16.pt",
            ),
        )
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="6b9d7548-33bb-4517-b5e8-60caf47edba7",
            name="nvidia/Cosmos-Predict2.5-2B/auto/multiview",
            experiment="buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames_nofps",
            metadata={
                "resolution": "720p",
                "fps": 30,
                "views": 7,
                "frames": 29,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames_nofps-0/checkpoints/iter_000005000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Predict2.5-2B",
                revision="15a82a2ec231bc318692aa0456a36537c806e7d4",
                filename="auto/multiview/6b9d7548-33bb-4517-b5e8-60caf47edba7_ema_bf16.pt",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="f740321e-2cd6-4370-bbfe-545f4eca2065",
            name="nvidia/Cosmos-Predict2.5-2B/robot/multiview-agibot",
            experiment="multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_agibot_frameinit",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_agibot_frameinit/checkpoints/iter_000016500",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Predict2.5-2B",
                revision="fbe72c18d152053029a19db3b211cf78671ad422",
                filename="robot/multiview-agibot/f740321e-2cd6-4370-bbfe-545f4eca2065_ema_bf16.pt",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="38c6c645-7d41-4560-8eeb-6f4ddc0e6574",
            name="nvidia/Cosmos-Predict2.5-2B/robot/action-cond",
            experiment="cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320",
            metadata={
                "resolution": "360p",
                "fps": 4,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320/checkpoints/iter_000016000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Predict2.5-2B",
                revision="main",
                filename="robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="24a3b7b8-6a3d-432d-b7d1-5d30b9229465",
            name="nvidia/Cosmos-Predict2.5-2B/transfer2.5",
            experiment="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only/checkpoints/iter_000037000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Experimental",
                revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
                filename="24a3b7b8-6a3d-432d-b7d1-5d30b9229465/model_ema_bf16.pt",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="575edf0f-d973-4c74-b52c-69929a08d0a5",
            name="nvidia/Cosmos-Predict2.5-2B/base/distilled",
            experiment="dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V",
            metadata={
                "size": "2B",
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointFileS3(
                uri="s3://bucket/cosmos_predict2_distill/predict2_distill/dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional/checkpoints/iter_000007500/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Predict2.5-2B",
                revision="e26f8a125a2235c5a00245a65207402dd0cdcb89",
                filename="base/distilled/575edf0f-d973-4c74-b52c-69929a08d0a5_ema_bf16.pt",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="54937b8c-29de-4f04-862c-e67b04ec41e8",
            name="nvidia/Cosmos-Predict2.5-14B/base/pre-trained",
            experiment="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma",
            metadata={
                "size": "14B",
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointFileS3(
                uri="s3://bucket/cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma/checkpoints/iter_000012500/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Predict2.5-14B",
                revision="03eb354f35eae0d6e0c1be3c9f94d8551e125570",
                filename="base/pre-trained/54937b8c-29de-4f04-862c-e67b04ec41e8_ema_bf16.pt",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="e21d2a49-4747-44c8-ba44-9f6f9243715f",
            name="nvidia/Cosmos-Predict2.5-14B/base/post-trained",
            experiment="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma",
            metadata={
                "size": "14B",
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointFileS3(
                uri="s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_GRPO-reason_embeddings-Index-26-Size-14B-Res-720-Fps-16-posttrain_data-HQ_V7_RF_MERGE_GENERAL_steps20_every2_lr3e-6_guidance0_scorekeyoverall_reward_databeta0.01_mincon0/checkpoints/iter_000000128/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Predict2.5-14B",
                revision="2bc4ca5ba5a20b9858a7ddb856bc82d70b030fbe",
                filename="base/post-trained/e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt",
            ),
        ),
    )
