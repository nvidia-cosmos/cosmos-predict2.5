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

torchrun $TORCHRUN_ARGS examples/action_conditioned.py \
    -i $INPUT_DIR/assets/action_conditioned/basic/inference_params.json \
    -o $OUTPUT_DIR \
    --context_parallel_size 1 \
    $INFERENCE_ARGS --save_root $OUTPUT_DIR

# Dry run post-training test
torchrun $TORCHRUN_ARGS scripts/train.py \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    --dryrun \
    -- \
    experiment=ac_reason_embeddings_rectified_flow_2b_256_320 \
    ~dataloader_train.dataloaders
