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
Shared utilities for precomputing Reason1 text embeddings (for rectified flow models).

This module provides functions to compute and save Reason1 7B embeddings for
language instructions, similar to how T5 embeddings are precomputed in
t5_embedding_utils.py.

The key difference from T5:
- T5 embeddings have shape (1, 512, 1024)
- Reason1 embeddings with full_concat have shape (512, hidden_size * num_layers)
  where hidden_size=3584 and num_layers=28 for Qwen2.5-VL-7B
"""

import os
import pickle
from typing import Dict, List

import torch
from tqdm import tqdm

from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder, TextEncoderConfig

# Global encoder to avoid reloading the 7B model multiple times
_reason1_encoder = None


def get_reason1_encoder(
    ckpt_path: str = "s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
    s3_credential_path: str = "credentials/s3_checkpoint.secret",
    embedding_concat_strategy: str = "full_concat",
) -> TextEncoder:
    """
    Get or create the Reason1 text encoder (singleton pattern to avoid reloading).

    Args:
        ckpt_path: Path to Reason1 checkpoint (S3 or local)
        s3_credential_path: Path to S3 credentials file
        embedding_concat_strategy: Strategy for combining hidden layer outputs
            - "full_concat": Concatenate all layers (largest embeddings)
            - "mean_pooling": Average across layers
            - "pool_every_n_layers_and_concat": Pool groups then concat

    Returns:
        TextEncoder instance
    """
    global _reason1_encoder
    if _reason1_encoder is None:
        print("Loading Reason1 7B model (this may take a minute)...")
        config = TextEncoderConfig(
            compute_online=True,
            embedding_concat_strategy=embedding_concat_strategy,
            ckpt_path=ckpt_path,
            s3_credential_path=s3_credential_path,
        )
        _reason1_encoder = TextEncoder(config, device="cuda")
        print("Reason1 model loaded successfully!")
    return _reason1_encoder


def get_reason1_text_embedding(text: str) -> torch.Tensor:
    """
    Get Reason1 embedding for a single text string.

    Args:
        text: Input text string

    Returns:
        Embedding tensor of shape (1, seq_len, embedding_dim)
    """
    encoder = get_reason1_encoder()
    data_batch = {"ai_caption": [text]}
    with torch.no_grad():
        embedding = encoder.compute_text_embeddings_online(data_batch, "ai_caption")
    return embedding


def generate_reason1_embeddings(
    unique_commands: List[str],
    batch_size: int = 8,
    ckpt_path: str = "s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
    s3_credential_path: str = "credentials/s3_checkpoint.secret",
    embedding_concat_strategy: str = "full_concat",
) -> Dict[str, torch.Tensor]:
    """
    Generate Reason1 text embeddings for a list of commands.

    This is the Reason1 equivalent of generate_t5_embeddings() in t5_embedding_utils.py.

    Args:
        unique_commands: List of unique command strings
        batch_size: Batch size for processing (default 8, adjust based on GPU memory)
        ckpt_path: Path to Reason1 checkpoint
        s3_credential_path: Path to S3 credentials
        embedding_concat_strategy: Strategy for combining hidden layers

    Returns:
        Dictionary mapping command strings to their Reason1 embeddings (bfloat16, on CPU)
    """
    encoder = get_reason1_encoder(
        ckpt_path=ckpt_path,
        s3_credential_path=s3_credential_path,
        embedding_concat_strategy=embedding_concat_strategy,
    )

    reason1_embeddings = dict()

    print(f"Computing Reason1 embeddings for {len(unique_commands)} commands...")
    print(f"Using batch_size={batch_size}, strategy={embedding_concat_strategy}")

    # Process in batches for efficiency
    num_batches = (len(unique_commands) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Computing embeddings"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(unique_commands))
        batch_commands = unique_commands[start_idx:end_idx]

        # Prepare data batch in the format expected by the encoder
        data_batch = {"ai_caption": batch_commands}

        # Compute embeddings
        with torch.no_grad():
            batch_embeddings = encoder.compute_text_embeddings_online(data_batch, "ai_caption")

        # Store each embedding (move to CPU to save GPU memory)
        for j, command in enumerate(batch_commands):
            # Keep the same format as T5 embeddings: (1, seq_len, embedding_dim)
            # T5 saves with batch dim for compatibility with inference code that calls .repeat(batch_size, 1, 1)
            embedding = batch_embeddings[j].unsqueeze(0).to(dtype=torch.bfloat16).cpu()
            reason1_embeddings[command] = embedding

        # Clear CUDA cache periodically to prevent OOM
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    return reason1_embeddings


def save_reason1_embeddings(
    embeddings: Dict[str, torch.Tensor],
    data_dir: str,
    filename: str = "reason1_embeddings.pkl",
    check_exists: bool = False,
) -> str:
    """
    Save Reason1 text embeddings to a pickle file.

    This is the Reason1 equivalent of save_embeddings() in t5_embedding_utils.py.

    Args:
        embeddings: Dictionary of embeddings to save
        data_dir: Directory where embeddings should be saved
        filename: Output filename (default: reason1_embeddings.pkl)
        check_exists: If True, prompt user for new filename if file exists

    Returns:
        Path where embeddings were saved
    """
    print("Saving Reason1 embeddings...")
    save_path = os.path.join(data_dir, filename)

    if check_exists and os.path.exists(save_path):
        print(f"File {save_path} already exists.")
        new_filename = input("Please enter a new filename (e.g., reason1_embeddings_v2.pkl): ")
        save_path = os.path.join(data_dir, new_filename)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as file:
        pickle.dump(embeddings, file)

    # Print summary info
    sample_key = list(embeddings.keys())[0]
    sample_embedding = embeddings[sample_key]
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)

    print(f"Saved {len(embeddings)} embeddings to: {save_path}")
    print(f"Sample embedding shape: {sample_embedding.shape}")
    print(f"Sample embedding dtype: {sample_embedding.dtype}")
    print(f"File size: {file_size_mb:.2f} MB")

    return save_path


def load_reason1_embeddings(embeddings_path: str) -> Dict[str, torch.Tensor]:
    """
    Load Reason1 text embeddings from a pickle file.

    Args:
        embeddings_path: Path to the pickle file

    Returns:
        Dictionary mapping command strings to embedding tensors
    """
    print(f"Loading Reason1 embeddings from {embeddings_path}...")
    with open(embeddings_path, "rb") as file:
        embeddings = pickle.load(file)

    sample_key = list(embeddings.keys())[0]
    sample_embedding = embeddings[sample_key]
    print(f"Loaded {len(embeddings)} embeddings, shape: {sample_embedding.shape}")

    return embeddings
