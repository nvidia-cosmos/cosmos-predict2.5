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
    register_checkpoint(
        CheckpointConfig(
            uuid="7219c6c7-f878-4137-bbdb-76842ea85e70",
            name="Qwen/Qwen2.5-VL-7B-Instruct",
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_reasoning1/pretrained/Qwen_tokenizer/Qwen/Qwen2.5-VL-7B-Instruct",
            ),
            hf=CheckpointDirHf(
                repository="nvidia/Cosmos-Reason1-7B",
                revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="685afcaa-4de2-42fe-b7b9-69f7a2dee4d8",
            name="Wan2.1/vae",
            s3=CheckpointFileS3(
                uri="s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/Wan2.1_VAE.pth",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Predict2.5-2B",
                revision="6787e176dce74a101d922174a95dba29fa5f0c55",
                filename="tokenizer.pth",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="cb3e3ffa-7b08-4c34-822d-61c7aa31a14f",
            name="nvidia/Cosmos-Reason1.1-7B",
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model",
            ),
            hf=CheckpointDirHf(
                repository="nvidia/Cosmos-Reason1-7B",
                revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
            ),
        ),
    )
