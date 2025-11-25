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
Interactive dataset utility functions (minimal).
"""

import os

from huggingface_hub import hf_hub_download

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import log


def extract_cr1_embedding(cr1_embeddings_path: str) -> str:
    """
    Ensure CR1 embeddings are available at the specified path.

    For INTERNAL builds, verifies the file exists and raises an error if not.
    For external builds, downloads from Hugging Face if the file doesn't exist.

    Args:
        cr1_embeddings_path: Path where CR1 embeddings should be located

    Raises:
        FileNotFoundError: If embeddings are not found (INTERNAL) or download fails (external)
    """
    if INTERNAL:
        if not os.path.exists(cr1_embeddings_path):
            log.error(f"CR1 embeddings not found at {cr1_embeddings_path}. Please set --cr1_embeddings_path correctly.")
            raise FileNotFoundError(cr1_embeddings_path)
        return cr1_embeddings_path
    else:
        if os.path.exists(cr1_embeddings_path):
            log.info(f"CR1 embeddings found at {cr1_embeddings_path}. Skipping download.")
            return cr1_embeddings_path
        log.info(f"Downloading CR1 embeddings from Hugging Face to {cr1_embeddings_path}...")
        try:
            downloaded = hf_hub_download(
                "nvidia/Cosmos-Predict2.5-2B", "robot/action-cond/cr1_empty_string_text_embeddings.pt"
            )
            log.info("Successfully downloaded CR1 embeddings")
            return downloaded
        except Exception as e:
            raise FileNotFoundError(f"Failed to download CR1 embeddings: {e}")
