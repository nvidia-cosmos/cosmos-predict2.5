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


set -e

# Setup: install dependencies, activate venv, authenticate
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/nfs/common/gradio_endpoints/uv_cache}"
export UV_LINK_MODE=copy
echo "Setting up environment..."
uv sync --extra="${UV_EXTRA:-cu128}"
source .venv/bin/activate
if [ -n "$HF_TOKEN" ]; then
  hf auth login --token "$HF_TOKEN"
fi
echo "Environment setup complete."

# Download assets from HuggingFace if COSMOS_ASSET_HF_PATH is set
if [ -n "$COSMOS_ASSET_HF_PATH" ]; then
  echo "Downloading assets: $COSMOS_ASSET_HF_PATH..."
  mkdir -p "$ASSET_DIR"
  hf download nvidia/Cosmos-Assets --repo-type dataset \
    --include "${COSMOS_ASSET_HF_PATH}/*" --local-dir /tmp/cosmos_assets_download
  mv "/tmp/cosmos_assets_download/${COSMOS_ASSET_HF_PATH}" "$ASSET_DIR/"
  rm -rf /tmp/cosmos_assets_download
  echo "Asset download complete."
fi

export GRADIO_APP=${GRADIO_APP:-cosmos_predict2/gradio/gradio_bootstrapper.py}
export MODEL_NAME=${MODEL_NAME:-"video2world"}

export NUM_GPUS=${NUM_GPUS:-2}

export WORKSPACE_DIR=${WORKSPACE_DIR:-outputs/}
# if OUTPUT_DIR, UPLOADS_DIR, or LOG_FILE are not set, set them to defaults based on WORKSPACE_DIR and MODEL_NAME
export OUTPUT_DIR=${OUTPUT_DIR:-${WORKSPACE_DIR}/${MODEL_NAME}_gradio}
export UPLOADS_DIR=${UPLOADS_DIR:-${WORKSPACE_DIR}/${MODEL_NAME}_gradio}
export LOG_FILE=${LOG_FILE:-${WORKSPACE_DIR}/${MODEL_NAME}_gradio/$(date +%Y%m%d_%H%M%S).txt}

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$UPLOADS_DIR"
mkdir -p "$(dirname "$LOG_FILE")"


# Start the app and tee output to the log file
echo "Starting the app: PYTHONPATH=. python3 $GRADIO_APP 2>&1 | tee -a $LOG_FILE"
PYTHONPATH=. python3 "$GRADIO_APP" 2>&1 | tee -a "$LOG_FILE"
