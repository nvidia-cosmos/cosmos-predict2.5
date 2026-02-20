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
Shared utilities for precomputing and visualizing T5 text embeddings.
"""

import os
import pickle
from typing import Dict, List

import torch
from tqdm import tqdm

from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding


def generate_t5_embeddings(unique_commands: List[str]) -> Dict[str, torch.Tensor]:
    """
    Generate T5 text embeddings for a list of commands.

    Args:
        unique_commands: List of unique command strings

    Returns:
        Dictionary mapping command strings to their T5 embeddings (bfloat16, on CPU)
    """
    t5_text_embeddings = dict()
    print("Getting text embeddings...")
    for command in tqdm(unique_commands):
        embedding = get_text_embedding(command).to(dtype=torch.bfloat16).cpu()  # (1, 512, 1024)
        t5_text_embeddings[command] = embedding
    return t5_text_embeddings


def save_embeddings(t5_text_embeddings: Dict[str, torch.Tensor], data_dir: str, check_exists: bool = False) -> str:
    """
    Save T5 text embeddings to a pickle file.

    Args:
        t5_text_embeddings: Dictionary of embeddings to save
        data_dir: Directory where embeddings should be saved
        check_exists: If True, prompt user for new filename if file exists

    Returns:
        Path where embeddings were saved
    """
    print("Saving text embeddings...")
    save_path = os.path.join(data_dir, "t5_embeddings.pkl")
    if check_exists and os.path.exists(save_path):
        print(f"File {save_path} already exists.")
        new_filename = input("Please enter a new filename for saving the embeddings (e.g., t5_embeddings_v2.pkl): ")
        save_path = os.path.join(data_dir, new_filename)

    with open(save_path, "wb") as file:
        pickle.dump(t5_text_embeddings, file)
        print(f"Saved embeddings at: {save_path}")

    return save_path
