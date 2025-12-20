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

# Enable coverage subprocess tracking if coverage is enabled
COVERAGE_RUN="torchrun"
if [ -n "$COVERAGE_ENABLED" ]; then
    export COVERAGE_PROCESS_START="$(pwd)/pyproject.toml"
    export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
    COVERAGE_RUN="coverage run --parallel-mode --source=cosmos_predict2 -m torch.distributed.run"
fi

$COVERAGE_RUN $TORCHRUN_ARGS examples/multiview.py \
    -i $INPUT_DIR/assets/multiview/urban_freeway.jsonl \
    -o $OUTPUT_DIR \
    $INFERENCE_ARGS
