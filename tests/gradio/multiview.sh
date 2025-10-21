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

echo -e "\e[95mRunning predict worker...\e[0m"

# WAR
ln -s packages/cosmos-gradio/cosmos_gradio cosmos_gradio

export MODEL_NAME="multiview"
export COSMOS_INTERNAL="0"

if PYTHONPATH=. python tests/gradio/test_worker.py; then
    echo -e "\e[92mPython command succeeded\e[0m"
else
    echo -e "\e[91mPython command failed with exit code $?\e[0m"
    exit 1
fi
