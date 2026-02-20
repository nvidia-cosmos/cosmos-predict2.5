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

if [ "$COSMOS_SMOKE" == "1" ]; then
    SMOKE_ARGS="--num_steps 1"
else
    SMOKE_ARGS=""
fi

mkdir -p "$OUTPUT_DIR"

uv run --script scripts/diffusers_inference.py \
    --input_path "$INPUT_DIR/assets/base/bus_terminal_long.json" \
    --num_output_frames 1 \
    --output_path "$OUTPUT_DIR/diffusers_text2image.png" \
    $SMOKE_ARGS
