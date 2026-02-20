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

"""Utilities for depth estimation models."""

import os
from pathlib import Path


def get_cache_dir() -> Path:
    """
    Get the cache directory for model weights.

    Priority:
    1. COSMOS_CACHE_DIR environment variable
    2. HuggingFace cache directory
    3. ~/.cache/cosmos_transfer2
    """
    if "COSMOS_CACHE_DIR" in os.environ:
        cache_dir = Path(os.environ["COSMOS_CACHE_DIR"]) / "depth_models"
    elif "HF_HOME" in os.environ:
        cache_dir = Path(os.environ["HF_HOME"]) / "cosmos_depth_models"
    else:
        cache_dir = Path.home() / ".cache" / "cosmos_transfer2" / "depth_models"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_model_cache_path(model_name: str) -> Path:
    """Get the cache path for a specific model."""
    cache_dir = get_cache_dir()
    # Replace slashes in model name to create valid directory structure
    model_path = cache_dir / model_name.replace("/", "--")
    model_path.mkdir(parents=True, exist_ok=True)
    return model_path
