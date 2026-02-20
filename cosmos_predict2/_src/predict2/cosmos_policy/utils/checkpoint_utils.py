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

"""
Checkpoint utilities for Cosmos Policy, including HuggingFace checkpoint loading.
"""

from huggingface_hub import hf_hub_download

from cosmos_predict2._src.imaginaire.utils import log


def resolve_checkpoint_path(checkpoint_path: str, cache_dir: str | None = None) -> str:
    """
    Resolve checkpoint path, downloading from HuggingFace if needed.

    Supports:
    - Local paths: /path/to/checkpoint.pth
    - S3 paths: s3://bucket/path/to/checkpoint.pth
    - HuggingFace paths: hf://org/repo-name/path/to/file.pth

    Args:
        checkpoint_path: Path to checkpoint file
        cache_dir: Optional cache directory for HuggingFace downloads

    Returns:
        Resolved local path to checkpoint

    Examples:
        >>> # HuggingFace download
        >>> path = resolve_checkpoint_path("hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth")

        >>> # S3 or local path (passed through unchanged)
        >>> path = resolve_checkpoint_path("s3://bucket/checkpoint.pth")
        >>> path = resolve_checkpoint_path("/local/path/checkpoint.pth")
    """
    if checkpoint_path.startswith("hf://"):
        # Parse HuggingFace path: hf://org/repo-name/path/to/file
        hf_path = checkpoint_path[len("hf://") :]

        # Split into at most 3 parts: org, repo-name, path/to/file
        parts = hf_path.split("/", 2)

        if len(parts) != 3:
            raise ValueError(
                f"Invalid HuggingFace path format: {checkpoint_path}. Expected format: hf://org/repo-name/path/to/file"
            )

        org, repo_name, filename = parts
        repo_id = f"{org}/{repo_name}"

        log.info(f"Downloading checkpoint from HuggingFace: {repo_id}/{filename}")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )
        log.success(f"Downloaded checkpoint to: {local_path}")
        return local_path

    # Return path as-is (S3, local, etc.)
    return checkpoint_path
