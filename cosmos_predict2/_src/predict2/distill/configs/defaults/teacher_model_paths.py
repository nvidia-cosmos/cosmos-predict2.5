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
List of teacher model checkpoints to be distilled.
"""

# ================================================
# Predict2.5 2B released checkpoints
# Single unified model for t2i, t2v and v2v. 3 versions:
# - Only pretrained. No model merging, no RL involved.
# - Pretrained + posttrain + merge. No RL involved.
# - Pretrained + posttrain + merge + RL.
# ================================================
# Only pretrained. No model merging, no RL involved.
TEACHER_CKPT_720_T24_CR1PT1_PRETRAINED_RF_RELEASE = dict(
    load_path="s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2/checkpoints/iter_000023000/model",
    credentials="credentials/s3_checkpoint.secret",
)

# Pretrained + posttrain + merge. No RL involved.
TEACHER_CKPT_720_T24_CR1PT1_PRETRAINED_RF_POSTTRAIN_MERGE_RELEASE = dict(
    load_path="s3://bucket/cosmos_diffusion_v2/merge_models/20250822/model_soup/base_crowded_face_highmotion_robotics_cooldown4k_3.pt",
    credentials="credentials/s3_checkpoint.secret",
)

# Pretrained + posttrain + merge + RL.
TEACHER_CKPT_720_T24_CR1PT1_RL_RELEASE = dict(
    load_path="s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_GRPO-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-posttrain_data-HQ_V7_RF_MERGE_LOCAL_ag_every2_guidance0_scorekeyoverall_reward_databeta0.01_mincon0/checkpoints/iter_000000288/model",
    credentials="credentials/s3_checkpoint.secret",
)

# (For reference) The last sft-ed checkpoint trained with the EDM-wrapped Rectified Flow base model.
TEACHER_CKPT_720_T24_CR1PT1_EDM_RF = dict(
    load_path="s3://bucket/cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000045000/model",
    credentials="credentials/s3_checkpoint.secret",
)

# ================================================
# Predict2.5 14B released checkpoints
# ================================================
TEACHER_CKPT_720_T24_CR1PT1_RL_PRETRAINED_14B = dict(
    load_path="s3://bucket/cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma/checkpoints/iter_000012500/model",
    credentials="credentials/s3_checkpoint.secret",
)

TEACHER_CKPT_720_T24_CR1PT1_RL_RELEASE_14B = dict(
    load_path="s3://bucket/cosmos_diffusion_v2/pretrain_weights/Predict2.5-14B-merged.dcp/model",
    credentials="credentials/s3_checkpoint.secret",
)

# ================================================
# Multiview-Predict2.5 2B released checkpoints
# ================================================
MV_TEACHER_CKPT_T16_4FROM7VIEWS_CR1PT1_PRETRAINED_RF = dict(
    # good quality, no mosaic artifacts. state_t=16, 61 pixel frames
    load_path="s3://bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res480p_fps30_t16_from7kuniform7views_alpamayo1capviewprefix_allcapsviewprefix_61frames_nofps_uniform_textdrop0_4viewdropout-0/checkpoints/iter_000010250/model",
    credentials="credentials/s3_checkpoint.secret",
)
