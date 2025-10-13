#!/bin/bash
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

set -e

# Posttraining for groot Example
echo -e "\e[95mRunning Posttraining for groot example...\e[0m"

export COSMOS_INTERNAL=0
export COSMOS_CHECKPOINTS_DIR="$(pwd)/checkpoints"
export COSMOS_REASON1_DIR="$(pwd)/checkpoints/nvidia/Cosmos-Reason1-7B"
export COSMOS_QWEN_2PT5_VL_7B_INSTRUCT_PATH="$(pwd)/checkpoints/nvidia/Cosmos-Reason1-7B"
export COSMOS_WAN2PT1_VAE_PATH="$(pwd)/checkpoints/nvidia/Cosmos-Predict2.5-2B/base/tokenizer.pth"
export COSMOS_WAN2PT1_VAE_MEAN_STD_PATH="$(pwd)/checkpoints/nvidia/Cosmos-Predict2.5-2B/base/mean_std.pt"
PYTHONPATH=. python examples/posttraining/groot/post_training_groot.py
