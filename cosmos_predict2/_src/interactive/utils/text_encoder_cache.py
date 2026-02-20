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
Utility functions for caching text encoder checkpoints locally.
"""

import os
from urllib.parse import urlparse

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io


def _basic_dcp_cache_valid(local_cache_path: str) -> bool:
    if not os.path.isdir(local_cache_path):
        return False
    all_files = os.listdir(local_cache_path)
    has_metadata = ".metadata" in all_files
    has_distcp = any(f.endswith(".distcp") for f in all_files)
    return has_metadata and has_distcp


def _is_cache_complete(
    s3_ckpt_path: str,
    local_cache_path: str,
    s3_credential_path: str,
) -> bool:
    if not _basic_dcp_cache_valid(local_cache_path):
        return False
    try:
        backend = easy_io.get_file_backend(
            backend_args={
                "backend": "s3",
                "s3_credential_path": s3_credential_path,
                "path_mapping": None,
            }
        )
        expected_files = list(backend.list_dir_or_file(dir_path=s3_ckpt_path, list_dir=False, list_file=True))
    except Exception as e:
        log.warning(f"Failed to list S3 checkpoint for cache validation: {e}")
        return True

    missing_files = [
        obj_suffix for obj_suffix in expected_files if not os.path.exists(os.path.join(local_cache_path, obj_suffix))
    ]
    if missing_files:
        sample = ", ".join(missing_files[:5])
        log.warning(f"Cache missing {len(missing_files)} files (e.g., {sample})")
        return False
    return True


def cache_text_encoder_checkpoint(
    s3_ckpt_path: str,
    cache_dir: str = "./cosmos3_interactive_cache_ckpts/text_encoder",
    s3_credential_path: str = "credentials/s3_checkpoint.secret",
) -> str:
    """
    Cache text encoder checkpoint from S3 to local disk.

    Args:
        s3_ckpt_path: S3 path to the text encoder checkpoint
        cache_dir: Local directory to cache the checkpoint
        s3_credential_path: Path to S3 credentials

    Returns:
        Local path to the cached checkpoint
    """
    if not INTERNAL:
        # For external builds, the checkpoint_db handles caching via HuggingFace
        # Just return the original path
        return s3_ckpt_path

    if not s3_ckpt_path.startswith("s3://"):
        # Already a local path
        return s3_ckpt_path

    parsed = urlparse(s3_ckpt_path)
    # parsed.path is /path/after/bucket (with leading /)
    source_prefix = parsed.path.lstrip("/")
    local_cache_path = os.path.join(cache_dir, source_prefix)

    # Check if already cached
    if os.path.exists(local_cache_path):
        if os.path.isdir(local_cache_path):
            if _is_cache_complete(s3_ckpt_path, local_cache_path, s3_credential_path):
                log.info(f"Text encoder checkpoint already cached at {local_cache_path}")
                return local_cache_path
            log.warning(f"Cache directory exists but appears incomplete: {local_cache_path}")
        else:
            log.warning(f"Cache path exists but is not a directory: {local_cache_path}")
            os.remove(local_cache_path)

    # Download and cache the checkpoint
    log.info(f"Downloading and caching text encoder checkpoint from {s3_ckpt_path}")

    try:
        # Use cosmos_predict2._src.imaginaire's object_store utility to sync the S3 directory
        # sync_s3_dir_to_local will automatically create the cache_dir/source_prefix structure
        from cosmos_predict2._src.imaginaire.utils.object_store import sync_s3_dir_to_local

        cached_dir = sync_s3_dir_to_local(
            s3_dir=s3_ckpt_path,
            s3_credential_path=s3_credential_path,
            cache_dir=cache_dir,
            rank_sync=False,  # Don't sync across ranks, each rank will check cache
            local_rank_sync=True,  # Sync within node to avoid partial cache reads
        )

        log.info(f"Successfully cached text encoder checkpoint to {cached_dir}")
        return cached_dir

    except Exception as e:
        log.error(f"Failed to cache text encoder checkpoint: {e}")
        log.warning("Falling back to loading directly from S3")
        # Clean up partial cache if it exists
        if os.path.exists(local_cache_path):
            import shutil

            shutil.rmtree(local_cache_path, ignore_errors=True)
        return s3_ckpt_path


def get_cached_text_encoder_config(config, cache_dir: str = None):
    """
    Modify the config to use cached text encoder checkpoint if available.

    Args:
        config: Model config with text_encoder_config
        cache_dir: Directory for caching (uses from model_loader if None)

    Returns:
        Modified config with cached checkpoint path
    """
    if config.text_encoder_config is None:
        return config

    if not hasattr(config.text_encoder_config, "ckpt_path"):
        return config

    original_ckpt_path = config.text_encoder_config.ckpt_path

    # Use cache_dir from the config if not provided
    if cache_dir is None:
        cache_dir = getattr(config, "text_encoder_cache_dir", "./cosmos3_interactive_cache_ckpts/text_encoder")

    # Cache the checkpoint
    cached_path = cache_text_encoder_checkpoint(
        s3_ckpt_path=original_ckpt_path,
        cache_dir=cache_dir,
        s3_credential_path=config.text_encoder_config.s3_credential_path,
    )

    # Update config with cached path
    if cached_path != original_ckpt_path:
        log.info(f"Updated text encoder checkpoint path from {original_ckpt_path} to {cached_path}")
        # Create a new config with the updated path
        from copy import deepcopy

        config = deepcopy(config)
        config.text_encoder_config.ckpt_path = cached_path

    return config
