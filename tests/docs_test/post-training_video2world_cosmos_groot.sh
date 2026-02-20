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

# Download the dataset
export DATASET_DIR="$TMP_DIR/datasets/gr1"
mkdir -p "$DATASET_DIR"
hf download nvidia/PhysicalAI-Robotics-GR00T-GR1\
  --repo-type dataset \
  --local-dir "$DATASET_DIR/hf_gr1" \
  --include "*.mp4*" "metadata.csv"

# Organize dataset structure
mkdir -p "$DATASET_DIR/videos"
mv "$DATASET_DIR/hf_gr1/gr1"/*.mp4 "$DATASET_DIR/videos/" || true
mv "$DATASET_DIR/hf_gr1/metadata.csv" "$DATASET_DIR/" || true

# Create prompts for the dataset
python -m scripts.create_prompts_for_gr1_dataset \
    --dataset_path "$DATASET_DIR" \
    --meta_csv "$DATASET_DIR/metadata.csv"

# Train the model
torchrun $TORCHRUN_ARGS scripts/train.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- \
  experiment=predict2_video2world_training_2b_groot_gr1_480 \
  dataloader_train.dataset.dataset_dir="$DATASET_DIR" \
  dataloader_train.sampler.dataset.dataset_dir="$DATASET_DIR" \
  $TRAIN_ARGS

# Get path to the latest checkpoint
CHECKPOINTS_DIR="${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict_v2p5/video2world/2b_groot_gr1_480/checkpoints"
CHECKPOINT_ITER="$(cat "$CHECKPOINTS_DIR/latest_checkpoint.txt")"
CHECKPOINT_DIR="$CHECKPOINTS_DIR/$CHECKPOINT_ITER"

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py "$CHECKPOINT_DIR/model" "$CHECKPOINT_DIR"

# Run inference
torchrun $TORCHRUN_ARGS examples/inference.py \
  -i "$INPUT_DIR/assets/sample_gr00t_dreams_gr1/gr00t_image2world.json" \
  -o "$OUTPUT_DIR" \
  --checkpoint-path "$CHECKPOINT_DIR/model_ema_bf16.pt" \
  --experiment predict2_video2world_training_2b_groot_gr1_480 \
  $INFERENCE_ARGS
