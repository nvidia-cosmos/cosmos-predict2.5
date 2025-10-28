#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Script for Video2World with Context Parallel on 8 GPUs

set -e

# ====== Configuration ======
export COSMOS_INTERNAL=1

EXPERIMENT="Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4"
CHECKPOINT_PATH="s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/${EXPERIMENT}/checkpoints/iter_000045000"
SAVE_ROOT="results/cli_debug_from_s3"
INPUT_ROOT="/home/checkpoints/assets/base/"
NUM_LATENT_FRAMES=2
CONTEXT_PARALLEL_SIZE=8
NUM_GPUS=8

# ====== Run Script ======
echo "Running Video2World with context parallel using ${NUM_GPUS} GPUs..."

PYTHONPATH=. torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  cosmos_predict2/_src/predict2/inference/video2world.py \
  --experiment="${EXPERIMENT}" \
  --ckpt_path="${CHECKPOINT_PATH}" \
  --save_root="${SAVE_ROOT}" \
  --input_root="${INPUT_ROOT}" \
  --num_latent_conditional_frames="${NUM_LATENT_FRAMES}" \
  --context_parallel_size="${CONTEXT_PARALLEL_SIZE}"
