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

"""Utils for evaluating Cosmos policies."""

import json
import os
import pickle
import queue
import secrets
import shutil
import time
import traceback
import uuid
from typing import Any, Dict, List

import numpy as np
import requests
import torch
import torchvision.transforms.functional as F
from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from huggingface_hub import snapshot_download
from PIL import Image
from torch.multiprocessing import Event, Process, Queue

from cosmos_predict2._src.predict2.cosmos_policy.constants import ACTION_DIM
from cosmos_predict2._src.predict2.cosmos_policy.datasets.dataset_utils import apply_jpeg_compression_np, resize_images
from cosmos_predict2._src.predict2.cosmos_policy.datasets.reason1_embedding_utils import get_reason1_text_embedding
from cosmos_predict2._src.predict2.cosmos_policy.utils.utils import duplicate_array
from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

# Initialize important constants
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
COSMOS_IMAGE_SIZE = 224  # Standard image size expected by Cosmos policies
COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4

# Initialize global T5 text embeddings cache (to be filled later)
t5_text_embeddings_cache = {}
t5_text_embeddings_path_global = None  # Path to T5 embeddings file
t5_text_embeddings_newly_computed = False  # Global boolean - tracks whether new embeddings were computed and T5 embeddings file should be updated (on disk)
text_embeddings_kind: str | None = None  # "t5" or "reason1" - set by caller

# Configure numpy print settings - print to 3 decimals
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def get_latent_indices_from_model_config(model):
    """
    Determine latent sequence indices based on model configuration.

    The latent sequence structure depends on the model's state representation:
    - state_t: Total number of latent frames in the sequence
    - min_num_conditional_frames: Number of frames used as conditional input

    Args:
        model: Cosmos model with config attributes

    Returns:
        tuple: (curr_state_start, curr_state_end, future_state_start, future_state_end)

    Examples:
        LIBERO (state_t=9, min_conditional_frames=4):
            Sequence: [blank, curr_proprio, curr_wrist_img, curr_third_person_img, action,
                      future_proprio, future_wrist, future_third_person_img, value]
            Returns: (1, 3, 5, 7)

        RoboCasa (state_t=11, min_conditional_frames=5):
            Sequence: [blank, curr_proprio, curr_wrist_img, curr_third_person_img_left, curr_third_person_img_right, action,
                      future_proprio, future_wrist_img, future_third_person_img_left, future_third_person_img_right, value]
            Returns: (1, 4, 6, 9)

        ALOHA (state_t=11, min_conditional_frames=5):
            Sequence: [blank, curr_proprio, curr_wrist_img1, curr_wrist_img2, curr_third_person_img, action,
                      future_proprio, future_wrist1, future_wrist2, future_third_person_img, value]
            Returns: (1, 4, 6, 9)
    """
    state_t = model.config.state_t
    min_conditional_frames = model.config.min_num_conditional_frames

    if state_t == 9 and min_conditional_frames == 4:
        # LIBERO setup
        return (1, 3, 5, 7)
    elif state_t == 11 and min_conditional_frames == 5:
        # RoboCasa/ALOHA setup
        return (1, 4, 6, 9)
    else:
        raise ValueError(
            f"Unknown model config! state_t={state_t}, min_num_conditional_frames={min_conditional_frames}."
        )


def is_hf_checkpoint_path(checkpoint_path: str) -> bool:
    """
    Check if a checkpoint path is a HuggingFace repository ID.

    Args:
        checkpoint_path (str): Path to checkpoint

    Returns:
        bool: True if it's a HF repo ID, False otherwise
    """
    if checkpoint_path is None or checkpoint_path == "":
        return False

    # Check if it matches HF repo pattern (org/model-name or org/model-name/subdir)
    # Exclude local filesystem paths
    if "/" in checkpoint_path and not checkpoint_path.startswith("/") and not checkpoint_path.startswith("./"):
        parts = checkpoint_path.split("/")
        # Support org/repo or org/repo/subdir/... patterns (2 or more parts)
        if len(parts) >= 2 and not any(part.startswith(".") for part in parts):
            return True

    return False


def download_hf_checkpoint(repo_id: str, cache_dir: str | None = None) -> str:
    """
    Download a Cosmos Policy checkpoint from HuggingFace and return the local path.

    Args:
        repo_id (str): HuggingFace repository ID, optionally with subdirectory
                      (e.g., "nvidia/Cosmos-Policy-LIBERO-Predict2-2B" or
                       "nvidia/Cosmos-Experimental/<UUID>")
        cache_dir (str, optional): Local cache directory. If None, uses HF default cache.

    Returns:
        str: Local path to the downloaded checkpoint directory
    """
    # Parse repo_id to extract subdirectory if present
    parts = repo_id.split("/")
    if len(parts) > 2:
        # Format: org/repo/subdir/...
        actual_repo_id = f"{parts[0]}/{parts[1]}"
        subdir = "/".join(parts[2:])
        allow_patterns = [f"{subdir}/*", f"{subdir}/**/*"]
    else:
        actual_repo_id = repo_id
        subdir = None
        allow_patterns = None

    print(
        f"Downloading checkpoint from HuggingFace: {actual_repo_id}" + (f" (subdirectory: {subdir})" if subdir else "")
    )

    # Use HF_HOME environment variable if cache_dir not specified
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", None)

    # Download the repository (or specific subdirectory)
    local_dir = snapshot_download(
        repo_id=actual_repo_id,
        cache_dir=cache_dir,
        resume_download=True,
        allow_patterns=allow_patterns,
    )

    print(f"Checkpoint downloaded successfully to: {local_dir}")

    # If subdirectory was specified, return path to that subdirectory
    if subdir:
        local_dir = os.path.join(local_dir, subdir)

    # Check for checkpoint structure
    model_dir = os.path.join(local_dir, "model")
    if os.path.exists(model_dir):
        return model_dir
    else:
        # Look for .pt files
        pt_files = [f for f in os.listdir(local_dir) if f.endswith(".pt")]
        if pt_files:
            return os.path.join(local_dir, pt_files[0])

    return local_dir


def download_hf_file(hf_path: str, cache_dir: str | None = None, repo_type: str | None = None) -> str:
    """
    Download a single file from a HuggingFace repository and return the local path.

    Args:
        hf_path (str): HuggingFace file path in format "repo_id/filename"
                      (e.g., "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json")
                      For dataset repositories, prefix with "datasets/":
                      (e.g., "datasets/nvidia/LIBERO-Cosmos-Policy/success_only/reason1_embeddings.pkl")
        cache_dir (str, optional): Local cache directory. If None, uses HF default cache.
        repo_type (str, optional): Repository type ("model" or "dataset"). If None, auto-detected
                                   from path prefix "datasets/".

    Returns:
        str: Local path to the downloaded file
    """
    from huggingface_hub import hf_hub_download

    # Check if path starts with "datasets/" prefix to auto-detect repo_type
    if hf_path.startswith("datasets/"):
        hf_path = hf_path[len("datasets/") :]
        if repo_type is None:
            repo_type = "dataset"

    # Parse the HF path
    parts = hf_path.split("/", 2)  # Split into at most 3 parts: org, repo, filename
    if len(parts) < 3:
        raise ValueError(f"Invalid HF file path: {hf_path}. Expected format: 'org/repo/filename'")

    repo_id = f"{parts[0]}/{parts[1]}"
    filename = parts[2]

    print(f"Downloading file from HuggingFace: {hf_path}" + (f" (repo_type={repo_type})" if repo_type else ""))

    # Use HF_HOME environment variable if cache_dir not specified
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", None)

    # Download the specific file
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        repo_type=repo_type,
    )

    print(f"File downloaded successfully to: {local_path}")

    return local_path


def resolve_path(path: str, cache_dir: str | None = None, repo_type: str | None = None) -> str:
    """
    Resolve a path - download from HuggingFace if it's an HF path, otherwise return as-is.

    Args:
        path (str): Path to resolve. Can be:
                    - Local path (e.g., "/path/to/file.pkl")
                    - HF model repo path (e.g., "nvidia/Cosmos-Policy/file.json")
                    - HF dataset repo path (e.g., "datasets/nvidia/LIBERO-Cosmos-Policy/file.pkl")
        cache_dir (str, optional): Cache directory for HF downloads
        repo_type (str, optional): Repository type ("model" or "dataset"). If None, auto-detected
                                   from path prefix "datasets/".

    Returns:
        str: Resolved local path
    """
    if path is None or path == "":
        return path

    # Check for "datasets/" prefix to detect HF dataset paths
    if path.startswith("datasets/"):
        return download_hf_file(path, cache_dir=cache_dir, repo_type=repo_type)

    # Check if it's an HF file path (org/repo/filename with 3+ parts)
    if "/" in path and not path.startswith("/") and not path.startswith("./"):
        parts = path.split("/")
        if len(parts) >= 3:
            # This looks like an HF file path
            return download_hf_file(path, cache_dir=cache_dir, repo_type=repo_type)

    # Return local path as-is
    return path


def load_dataset_stats(dataset_stats_path: str) -> dict:
    """
    Load dataset statistics from a JSON file.

    This function loads normalization statistics needed for action un-normalization
    and proprio rescaling. It handles both local paths and HuggingFace paths.

    Args:
        dataset_stats_path (str): Path to dataset statistics JSON file.
                                  Can be a local path or HF path (e.g., "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json")

    Returns:
        dict: Dataset statistics with numpy arrays for keys like "actions_min", "actions_max", "proprio_min", "proprio_max"

    Raises:
        AssertionError: If dataset_stats_path is empty or file doesn't exist
    """
    assert dataset_stats_path != "", "Must provide `dataset_stats_path` for action un-normalization!"
    dataset_stats_path = resolve_path(dataset_stats_path)
    assert os.path.exists(dataset_stats_path), f"Dataset stats do not exist at path: {dataset_stats_path}"

    with open(dataset_stats_path, "r") as f:
        dataset_stats = json.load(f)

    # Convert JSON lists back to numpy arrays
    for stat_name, stat_value in dataset_stats.items():
        dataset_stats[stat_name] = np.array(stat_value)

    return dataset_stats


def get_model(cfg):
    """
    Load and initialize the Cosmos model and configuration from checkpoint.

    Supports loading from:
    - HuggingFace repositories (e.g., "nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    - Local filesystem paths

    Args:
        cfg: Evaluation configuration object

    Returns:
        Tuple[torch.nn.Module, Config]: Tuple of (model, config)
    """
    print("Instantiating pretrained Cosmos model...")

    # Resolve checkpoint path (download from HF if needed)
    checkpoint_path = cfg.ckpt_path
    if is_hf_checkpoint_path(checkpoint_path):
        print(f"Detected HuggingFace repository: {checkpoint_path}")
        checkpoint_path = download_hf_checkpoint(checkpoint_path)

    # Load the model
    model, config = load_model_from_checkpoint(
        experiment_name=cfg.config,
        s3_checkpoint_dir=checkpoint_path,
        config_file=cfg.config_file,
        load_ema_to_reg=False,
    )
    model.eval()
    model = model.to(DEVICE)
    return model, config


def get_planning_model(cfg, device: str = "cuda"):
    """
    Initialize the "planning model" used for world model and value function predictions.

    The planning model is typically a model fine-tuned from the base Cosmos Policy checkpoint.
    Its world modeling and value function prediction capabilities are refined via training on
    policy rollouts.

    Supports loading from:
    - HuggingFace repositories (e.g., "nvidia/Cosmos-Policy-ALOHA-Planning-Model-Predict2-2B")
    - Local filesystem paths

    Args:
        cfg (Config): Configuration object with planning_model_config_name and planning_model_ckpt_path
        device (str): Device to load model on

    Returns:
        Tuple[torch.nn.Module, Config]: Tuple of (model, config)
    """
    # Resolve checkpoint path (download from HF if needed)
    checkpoint_path = cfg.planning_model_ckpt_path
    if is_hf_checkpoint_path(checkpoint_path):
        print(f"Detected HuggingFace repository for planning model: {checkpoint_path}")
        checkpoint_path = download_hf_checkpoint(checkpoint_path)

    planning_model, config = load_model_from_checkpoint(
        experiment_name=cfg.planning_model_config_name,
        s3_checkpoint_dir=checkpoint_path,
        config_file=cfg.config_file,
        load_ema_to_reg=False,
    )
    planning_model.eval()
    planning_model = planning_model.to(device)
    return planning_model, config


def init_t5_text_embeddings_cache(
    t5_text_embeddings_path: str = None,
    worker_id: int = 0,
    embeddings_kind: str = "t5",
) -> dict:
    """
    Initialize text embeddings cache (for language instructions).

    Supports both T5 embeddings (shape (1, 512, 1024)) and Reason1 embeddings (shape (1, 512, 100352)).
    Cache is a dict; key: language instruction (str), val: text embedding (torch.Tensor, shape (1, seq_len, embed_dim)).

    Args:
        t5_text_embeddings_path (str): Path to precomputed T5 text embeddings dictionary.
                                       Can be a local path or HF path (e.g., "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl")
        worker_id (int): Worker ID (if using parallel inference)
        embeddings_kind (str): Type of embeddings - either "t5" or "reason1"

    Returns:
        dict: T5 text embeddings cache
    """
    global t5_text_embeddings_path_global, t5_text_embeddings_newly_computed, text_embeddings_kind

    text_embeddings_kind = embeddings_kind

    # Resolve HF path to local path if needed
    if t5_text_embeddings_path is not None:
        t5_text_embeddings_path = resolve_path(t5_text_embeddings_path)

    # Preload from saved embeddings file if it exists
    if (
        t5_text_embeddings_path is not None
        and os.path.exists(t5_text_embeddings_path)
        and t5_text_embeddings_cache == {}
    ):
        # Use file lock to prevent reading while another process is writing
        lock_path = t5_text_embeddings_path + ".lock"
        lock = FileLock(lock_path, timeout=3000)
        try:
            with lock:
                with open(t5_text_embeddings_path, "rb") as file:
                    data = pickle.load(file)
                    # Move embeddings to the appropriate device
                    device = torch.device(f"cuda:{worker_id}" if torch.cuda.is_available() else "cpu")
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data[key] = value.to(device)
                    t5_text_embeddings_cache.update(data)

            print(f"Loaded text embeddings from {t5_text_embeddings_path} onto device {device}")

            # Store the path for later saving
            t5_text_embeddings_path_global = t5_text_embeddings_path
            t5_text_embeddings_newly_computed = False
        except FileLockTimeout:
            print(
                "Warning: Could not acquire lock for text embeddings file after 30 seconds. "
                "Another process may be writing to it. Skipping cache load - embeddings will be computed on-demand."
            )
        except Exception as e:
            print(f"Warning: Error loading T5 embeddings cache: {e}. Embeddings will be computed on-demand.")


def get_t5_embedding_from_cache(task_label: str) -> torch.Tensor:
    """
    Get text embedding of language instruction from cache.

    If the embedding is not in cache, computes it using the appropriate encoder
    (T5 or Reason1) based on `text_embeddings_kind` and saves the updated cache to disk.

    Args:
        task_label (str): Task description string

    Returns:
        torch.Tensor: Text embedding (T5 or Reason1 depending on config)
    """
    global t5_text_embeddings_newly_computed, text_embeddings_kind
    if task_label in t5_text_embeddings_cache:
        text_embedding = t5_text_embeddings_cache[task_label]
    else:
        # Compute embedding using the appropriate encoder based on text_embeddings_kind
        if text_embeddings_kind == "reason1":
            print(f"Computing Reason1 embedding for new instruction: '{task_label}'...")
            text_embedding = get_reason1_text_embedding(task_label)
            print(f"Computing Reason1 embedding for new instruction: '{task_label}'... Finished!")
        else:
            # Default to T5 embeddings
            print(f"Computing T5 embedding for new instruction: '{task_label}'...")
            text_embedding = get_text_embedding(task_label)
            print(f"Computing T5 embedding for new instruction: '{task_label}'... Finished!")
        t5_text_embeddings_cache[task_label] = text_embedding
        t5_text_embeddings_newly_computed = True
        # Save the updated cache to disk
        save_t5_text_embeddings_cache()
    return text_embedding


def save_t5_text_embeddings_cache():
    """
    Save the T5 text embeddings cache to disk with automatic backup.

    Creates a backup of the original file before overwriting if new embeddings were computed.
    Uses file locking to prevent race conditions when multiple processes try to save simultaneously.
    If the lock cannot be acquired within 30 seconds, gracefully skips saving without crashing.
    This function is useful because it can prevent needing to recompute T5 embeddings that
    get added to the initial set of embeddings during the evaluations.

    Args:
        None

    Returns:
        None
    """
    global t5_text_embeddings_newly_computed
    if t5_text_embeddings_path_global is None:
        print("Warning: No T5 embeddings path set. Cannot save new embeddings. Skipping save.")
        return
    if not t5_text_embeddings_newly_computed:
        return
    # Use file lock to prevent concurrent writes from multiple processes
    lock_path = t5_text_embeddings_path_global + ".lock"
    lock = FileLock(lock_path, timeout=10)
    try:
        with lock:
            # Create backup of original file
            if os.path.exists(t5_text_embeddings_path_global):
                backup_path = t5_text_embeddings_path_global + ".backup"
                print(f"Creating backup of T5 embeddings at {backup_path}")
                shutil.copy2(t5_text_embeddings_path_global, backup_path)
            # Save updated cache to disk (move tensors to CPU first)
            print(f"Saving updated T5 embeddings cache to {t5_text_embeddings_path_global}")
            save_data = {}
            for key, value in t5_text_embeddings_cache.items():
                if isinstance(value, torch.Tensor):
                    save_data[key] = value.cpu()
                else:
                    save_data[key] = value
            with open(t5_text_embeddings_path_global, "wb") as f:
                pickle.dump(save_data, f)
            print(f"Saved {len(save_data)} T5 embeddings to {t5_text_embeddings_path_global}")
            t5_text_embeddings_newly_computed = False
    except FileLockTimeout:
        print(
            "Warning: Could not acquire lock for saving T5 embeddings after 30 seconds. "
            "Another process may be writing to it. Skipping save - embeddings will be recomputed if needed."
        )
        # Reset the flag so we don't keep trying to save on every call
        t5_text_embeddings_newly_computed = False
    except Exception as e:
        print(f"Error saving T5 embeddings cache: {e}")
        # For other errors, still reset the flag to avoid repeated attempts
        t5_text_embeddings_newly_computed = False
        print("Skipping save - embeddings will be recomputed if needed.")


def check_images_format(images: np.ndarray) -> None:
    """
    Validate input images format.

    Args:
        images (np.ndarray): Images to check

    Returns:
        None

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(images, np.ndarray)
    has_correct_shape = len(images.shape) == 4 and images.shape[-1] == 3
    has_correct_dtype = images.dtype == np.uint8
    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect images format detected! Make sure that the input images are a "
        "numpy array with shape (T, H, W, 3) and dtype np.uint8!"
    )


def apply_image_transforms(images: np.ndarray) -> np.ndarray:
    """
    Apply test-time image transformations to match the image augmentations used at train time.

    At train time, we use random resized crops (90% area) and color jitter. At test time, we only need to
    do a 90% area center crop.

    Args:
        images (np.ndarray): Images of shape (T, H, W, C) and dtype np.uint8

    Returns:
        np.ndarray: Images with test-time transformations applied
    """
    # Get dimensions
    _, H, W, C = images.shape
    assert H == W, f"Image height and width must be equal! Got H == {H} and W == {W}"
    assert C == 3, f"Number of channels should be 3! Got C == {C}"
    # Convert numpy array to torch tensor and reshape to (T, C, H, W)
    images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
    # Apply deterministic transformations to all images
    results = []
    for img in images_tensor:
        # Apply center crop. Since we used 90% area in training, use a 90% center crop here
        crop_size = int(H * 0.9**0.5)  # Square root because (sqrt(.9) * H) * (sqrt(.9) * H) = 0.9 * H^2
        img_crop = F.center_crop(img, crop_size)
        # Resize back to original size
        img_resized = F.resize(img_crop, [H, W], antialias=True)
        results.append(img_resized)
    # Combine results
    transformed_images = torch.stack(results)
    # Convert back to numpy array with shape (T, H, W, C)
    transformed_images = transformed_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return transformed_images


def prepare_images_for_model(images: List[np.ndarray], cfg, flip_images: bool = False) -> List[Image.Image]:
    """
    Prepare images for Cosmos model input by resizing and cropping as needed.

    Args:
        images (List[np.ndarray]): List of input images as numpy arrays
        cfg: Configuration object with parameters
        flip_images (bool): Whether to flip images vertically across x-axis

    Returns:
        np.ndarray: Processed images ready for the model
    """
    images = np.stack(images, axis=0)  # (T, H, W, C)
    # Check that the images have the right format
    check_images_format(images)
    # Flip images vertically across x-axis if needed (e.g., for LIBERO and RoboCasa)
    if flip_images:
        images = np.flipud(images)
    # Apply JPEG compression (if trained on JPEG-compressed images)
    if cfg.use_jpeg_compression:
        images = apply_jpeg_compression_np(images, quality=95)
    # Resize images to match training distribution
    processed_images = resize_images(images, COSMOS_IMAGE_SIZE)
    # Apply image transformations if trained with image augmentations
    if cfg.trained_with_image_aug:
        processed_images = apply_image_transforms(processed_images)
    return processed_images


def extract_action_chunk_from_latent_sequence(
    output_latent: torch.Tensor, action_shape: tuple, action_indices: torch.Tensor
) -> torch.Tensor:
    """
    Extract the predicted action chunk from the generated latent sequence.

    Args:
        output_latent (torch.Tensor): The full latent sequence generated by the model, with shape (B, C', T', H', W')
        action_shape (tuple): Target shape of the action chunk: (chunk_size, action_dim)
        action_indices (torch.Tensor): Batch indices specifying which index of the latent sequence contains the action

    Returns:
        torch.Tensor: Batch of extracted action chunks, with shape (B, chunk_size, action_dim)
    """
    # Get the action latent frame
    batch_indices = torch.arange(output_latent.shape[0], device=output_latent.device)
    action_latent_frame = output_latent[batch_indices, :, action_indices, :, :]  # (B, C', H', W')
    # Get shape of latent frames
    batch_size, latent_channels, latent_h, latent_w = action_latent_frame.shape
    # Flatten the action latent frame into a vector (preserving batch dimension)
    flat_action_latent = action_latent_frame.reshape(batch_size, -1)
    num_latent_elements = flat_action_latent.shape[1]
    # Calculate the number of elements in the target action chunk
    assert len(action_shape) == 2, "action_shape should have 2 elements: (chunk_size, action_dim)!"
    num_action_elements = action_shape[0] * action_shape[1]
    # Check that there are enough elements in the latent to extract the action
    assert num_action_elements <= num_latent_elements, (
        f"Action shape {action_shape} requires {num_action_elements} elements, but the latent only has {num_latent_elements} elements!"
    )
    # Calculate how many full action chunks we can extract
    num_action_chunks = num_latent_elements // num_action_elements
    # Get the full action chunks in the flat latent and then reshape to separate out the chunks
    # New shape: (batch_size, num_action_chunks, num_action_elements)
    all_action_chunks = flat_action_latent[:, : num_action_chunks * num_action_elements].reshape(
        batch_size, num_action_chunks, num_action_elements
    )
    # Reshape each chunk to the target shape
    # New shape: (batch_size, num_action_chunks, chunk_size, action_dim])
    all_action_chunks = all_action_chunks.reshape(batch_size, num_action_chunks, action_shape[0], action_shape[1])
    # Take the average over all chunks, along dimension 1 (the num_action_chunks dimension)
    # New shape: (batch_size, chunk_size, action_dim)
    final_action_chunk = torch.mean(all_action_chunks, dim=1)
    return final_action_chunk


def extract_value_from_latent_sequence(output_latent: torch.Tensor, value_indices: torch.Tensor) -> torch.Tensor:
    """
    Extract the predicted value from the generated latent sequence.

    Args:
        output_latent (torch.Tensor): The full latent sequence generated by the model, with shape (B, C', T', H', W')
        value_indices (torch.Tensor): Batch indices specifying which index of the latent sequence contains the value

    Returns:
        torch.Tensor: Batch of extracted values, with shape (B, value_dim)
    """
    # Get the value latent frame
    batch_indices = torch.arange(output_latent.shape[0], device=output_latent.device)
    value_latent_frame = output_latent[batch_indices, :, value_indices, :, :]  # (B, C', H', W')
    # Flatten value latent frame into a vector (B, C' * H' * W')
    flat_value_latent = value_latent_frame.reshape(value_latent_frame.shape[0], -1)
    # Take the average across all elements in the value latent frame
    final_value = torch.mean(flat_value_latent, dim=1)
    return final_value


def unnormalize_actions(actions: np.ndarray, dataset_stats: dict, scale_multiplier: float = 1.0):
    """
    Unnormalize actions to the original dataset scale.

    This undoes the normalization used at train time.

    Args:
        actions (np.ndarray): Actions to be unnormalized
        dataset_stats (dict): Dataset statistics needed for the rescaling formula
        scale_multiplier (float): Multiplier to adjust scale from [-scale_multiplier,+scale_multiplier] back to [-1,+1]

    Returns:
        np.ndarray: Unnormalized actions
    """
    actions_min = dataset_stats["actions_min"]
    actions_max = dataset_stats["actions_max"]
    # Reshape actions from (B, chunk_size, action_dim) to (B * chunk_size, action_dim)
    original_shape = actions.shape
    actions = actions.reshape(-1, actions_min.shape[0])
    # First, undo the scale_multiplier scaling
    actions = actions / scale_multiplier
    # Then, scale back to original data scale: x_new = 0.5 * (x + 1) * (x_max - x_min) + x_min
    actions = 0.5 * (actions + 1) * (actions_max - actions_min) + actions_min
    # Reshape actions back to (B, chunk_size, action_dim)
    actions = actions.reshape(original_shape)
    return actions


def rescale_proprio(proprio, dataset_stats, non_negative_only=False, scale_multiplier=1.0):
    """
    Rescale (normalize) proprio to the range [-1,+1] or [0,+1], with optional scaling by scale_multiplier.

    Args:
        proprio (np.ndarray): Proprio to be rescaled
        dataset_stats (dict): Dataset statistics needed for rescaling formula
        non_negative_only (bool): Whether to use [0,+1] range (True) or [-1,+1] range (False)
        scale_multiplier (float): Multiplier to adjust final scale

    Returns:
        np.ndarray: Rescaled proprio
    """
    arr = proprio
    curr_min = dataset_stats["proprio_min"]
    curr_max = dataset_stats["proprio_max"]
    # First, scale to [-1,+1] or [0,+1]:
    # - For [-1,+1]: x_new = 2 * ((x - curr_min) / (curr_max - curr_min)) - 1
    # - For [0,+1]: x_new = (x - curr_min) / (curr_max - curr_min)
    if not non_negative_only:  # [-1,+1]
        rescaled_arr = 2 * ((arr - curr_min) / (curr_max - curr_min)) - 1
    else:  # [0,+1]
        rescaled_arr = (arr - curr_min) / (curr_max - curr_min)
    # Scale to [-scale_multiplier,+scale_multiplier] or [0,+scale_multiplier]
    rescaled_arr = scale_multiplier * rescaled_arr
    proprio = rescaled_arr
    return proprio


def undo_latent_injection(
    sample: torch.Tensor, orig_clean_latent_frames: torch.Tensor, INDICES_TO_REPLACE: List[int]
) -> torch.Tensor:
    """
    Undo the latent injections so that VAE decoding will reconstruct images without visual artifacts.

    This is done by replacing the latent frames with the original (pre-injection) latent frames. The
    reason why this is needed is because the latent injection happens after the VAE encodes the raw
    image sequence. If the latent frames are decoded without removing the injections, the reoncstructed
    images have heavy visual distortions. To fix this, we replace the injected latent frames with the original
    (pre-injection) latent frames. These original frames really just correspond to blank/black (all zero)
    images that served as placeholders for the latent injections.

    Args:
        sample (torch.Tensor): Generated samples with latent injections, shape (B, C', T', H', W')
        orig_clean_latent_frames (torch.Tensor): Original clean (unnoised) latent frames in the condition object
        INDICES_TO_REPLACE (List[int]): Indices to replace with the original (pre-injection) latent frames

    Returns:
        torch.Tensor: Samples with latent injections undone
    """
    for index in INDICES_TO_REPLACE:
        sample[:, :, index, :, :] = orig_clean_latent_frames[:, :, index, :, :]
    return sample


def get_future_images_from_generated_samples(
    model: torch.nn.Module,
    sample: torch.Tensor,
    cfg,
    orig_clean_latent_frames: torch.Tensor = None,
    INDICES_TO_REPLACE: List[int] = None,
    future_wrist_image_latent_idx: int = None,
    future_wrist_image2_latent_idx: int = None,
    future_image_latent_idx: int = None,
    future_image2_latent_idx: int = None,
    temporal_compression_factor: int = 4,
) -> Dict[str, Any]:
    """
    Get predicted future images from generated samples.

    Args:
        model (torch.nn.Module): The model
        sample (torch.Tensor): Generated sample
        cfg (Config): Configuration object
        orig_clean_latent_frames (torch.Tensor): Original clean (unnoised) latent frames in the condition object
        INDICES_TO_REPLACE (List[int]): Indices to replace with original (pre-injection) latent frames
        future_wrist_image_latent_idx (int): Index of future wrist image in video
        future_wrist_image2_latent_idx (int): Index of future secondary wrist image in video
        future_image_latent_idx (int): Index of future primary image in video
        future_image2_latent_idx (int): Index of future secondary image in video
        temporal_compression_factor (int): Temporal compression factor for VAE

    Returns:
        Dict[str, Any]: Dictionary containing future image predictions
    """
    if orig_clean_latent_frames is not None:
        # Undo the latent injection
        sample = undo_latent_injection(sample, orig_clean_latent_frames, INDICES_TO_REPLACE)
    # Decode the latent and unnormalize
    generated_images = ((model.decode(sample) + 1.0) * 127.5).clamp(0, 255)  # (B, C, T, H, W), range [0, 255]
    generated_images = (
        generated_images.permute(0, 2, 3, 4, 1).to(torch.uint8).cpu().numpy().astype(np.uint8)
    )  # (B, T, H, W, C), range [0, 255]
    # Compute raw image indices given latent indices
    future_wrist_image_raw_idx = (future_wrist_image_latent_idx - 1) * temporal_compression_factor + 1
    future_wrist_image2_raw_idx = (future_wrist_image2_latent_idx - 1) * temporal_compression_factor + 1
    future_image_raw_idx = (future_image_latent_idx - 1) * temporal_compression_factor + 1
    future_image2_raw_idx = (future_image2_latent_idx - 1) * temporal_compression_factor + 1
    # Get raw future image predictions
    future_image_predictions = {}
    if cfg.use_wrist_image:
        future_wrist_image = generated_images[:, future_wrist_image_raw_idx]
        future_image_predictions["future_wrist_image"] = future_wrist_image
        if cfg.num_wrist_images == 2:
            future_wrist_image2 = generated_images[:, future_wrist_image2_raw_idx]
            future_image_predictions["future_wrist_image2"] = future_wrist_image2
    if cfg.use_third_person_image:
        future_image = generated_images[:, future_image_raw_idx]
        future_image_predictions["future_image"] = future_image
        if cfg.num_third_person_images == 2:
            future_image2 = generated_images[:, future_image2_raw_idx]
            future_image_predictions["future_image2"] = future_image2
    return future_image_predictions


def aggregate_value_predictions(
    all_value_predictions: List[np.ndarray], value_ensemble_aggregation_scheme: str
) -> np.ndarray:
    """
    Aggregate value predictions from a list of value predictions.

    Args:
        value_ensemble_aggregation_scheme (str): Aggregation scheme to use for ensemble value predictions
        all_value_predictions (List[np.ndarray]): List of value predictions

    Returns:
        np.ndarray: Aggregated value predictions
    """
    # Aggregate value predictions
    if value_ensemble_aggregation_scheme == "average":
        value_prediction = torch.stack(all_value_predictions).mean(0)
    elif value_ensemble_aggregation_scheme == "lcb":
        # Get average
        mean_value = torch.stack(all_value_predictions).mean(0)
        # Get standard deviation
        std_value = torch.stack(all_value_predictions).std(0)
        # Compute value = mean_value - beta * std_value
        beta = 1.0
        value_prediction = mean_value - beta * std_value
    elif value_ensemble_aggregation_scheme == "success_vote":
        success_threshold = 0.05
        # Get percentage of value predictions that are greater than the success threshold
        value_prediction = (torch.stack(all_value_predictions) > success_threshold).float().mean(0)
    elif value_ensemble_aggregation_scheme == "majority_mean":
        # Return the mean of the majority of the value predictions
        # If the majority predict failure (lower than some success threshold) -> return 0
        # Else return the mean of the value predictions that are greater than the success threshold
        success_threshold = 0.05
        stacked_values = torch.stack(all_value_predictions)  # [num_ensemble, batch_size]

        # Filter for success predictions
        success_mask = stacked_values > success_threshold
        num_success = success_mask.float().sum(0)
        num_total = stacked_values.shape[0]
        success_ratio = num_success / num_total

        # Compute mean of success values (only where success_mask is True)
        # Set non-success values to 0 so they don't affect the sum
        masked_values = torch.where(success_mask, stacked_values, torch.zeros_like(stacked_values))
        sum_success = masked_values.sum(0)
        mean_success = sum_success / num_success.clamp(min=1)  # avoid div by zero

        # Use mean_success where majority predicts success, else 0
        value_prediction = torch.where(success_ratio >= 0.5, mean_success, torch.zeros_like(mean_success))
    else:
        raise ValueError(f"Invalid search value scheme: {value_ensemble_aggregation_scheme}")
    # print(f"All value predictions: \t{all_value_predictions}")
    # print(f"Final value: \t\t{value_prediction.item()}")
    # Clip value predictions to [0, 1]
    value_prediction = torch.clamp(value_prediction, min=0, max=1)
    return value_prediction


def average_future_image_predictions(pred_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Average a list of future image prediction dictionaries.

    This function is used when we have a future image prediction ensemble and we want to
    aggregate the predictions for visualization purposes.

    Args:
        pred_dicts (List[Dict[str, Any]]): List of future image prediction dictionaries

    Returns:
        Dict[str, Any]: Averaged future image predictions
    """
    if len(pred_dicts) == 0:
        return {}
    result = {}
    for key in pred_dicts[0].keys():
        vals = [d[key] for d in pred_dicts if d.get(key, None) is not None]
        if len(vals) == 0:
            result[key] = None
            continue
        stacked = np.stack([v.astype(np.float32) for v in vals], axis=0)
        avg = np.clip(np.round(stacked.mean(axis=0)), 0, 255).astype(np.uint8)
        result[key] = avg
    return result


def get_action(
    cfg,
    model: torch.nn.Module,
    dataset_stats: dict,
    obs: Dict[str, Any],
    task_label_or_embedding: Any,
    seed: int = 1,
    randomize_seed: bool = False,
    num_denoising_steps_action: int = 5,
    generate_future_state_and_value_in_parallel: bool = True,
    worker_id: int = 0,
    batch_size: int = 1,
) -> List[np.ndarray]:
    """
    Generate action predictions with the policy.

    Supports both predict2 models (using cached T5 embeddings) and predict2.5 models
    (computing reason1 embeddings on-the-fly from ai_caption).

    Args:
        cfg (Config): Configuration object with parameters
        model (torch.nn.Module): The policy model (predict2 or predict2.5)
        dataset_stats (dict): Dataset statistics needed for rescaling formula
        obs (Dict[str, Any]): Observation dictionary
        task_label_or_embedding (Any): Text description of the task or T5/reason1 embedding of the task.
            If string: For predict2.5 models with online embedding computation, embeddings
            are computed on-the-fly using reason1 encoder. For predict2 models, uses cached T5 embeddings.
            If np.ndarray: Uses the provided embedding directly.
        seed (int): Seed for sampling from the model
        randomize_seed (bool): Whether to randomize the seed for sampling actions in each query (still depends on base seed)
        num_denoising_steps_action (int): Number of denoising steps to use for action prediction
        generate_future_state_and_value_in_parallel (bool): Whether to generate future state and value in parallel with the actions
        worker_id (int): Worker ID (if using parallel inference)
        batch_size (int): Batch size for inference

    Returns:
        Dict[str, Any]: Dictionary containing actions and related predictions
    """
    # If applicable, randomize the seed used for sampling
    if randomize_seed:
        seed = secrets.randbits(32) % 256

    with torch.inference_mode():
        # Get T5 embedding of language instruction
        # For predict2.5 models with online reason1 embedding computation, compute embeddings on the fly
        if isinstance(task_label_or_embedding, str):
            text_embedding = get_t5_embedding_from_cache(task_label_or_embedding)
        elif isinstance(task_label_or_embedding, np.ndarray):
            text_embedding = torch.tensor(task_label_or_embedding, dtype=torch.bfloat16).cuda()

        # Collect all input images
        # Examples:
        #  - LIBERO: 1 wrist image, 1 primary (third-person) image
        #  - RoboCasa: 1 wrist image, 1 primary (third-person) image, 1 secondary (third-person) image
        #  - ALOHA: 2 wrist images, 1 primary (third-person) image
        IMAGE_IDX, IMAGE2_IDX, WRIST_IMAGE_IDX, WRIST_IMAGE2_IDX = -1, -1, -1, -1
        if cfg.suite == "libero":
            all_camera_images = [
                obs["wrist_image"],
                obs["primary_image"],
            ]
            WRIST_IMAGE_IDX = 0
            IMAGE_IDX = 1
        elif cfg.suite == "robocasa":
            all_camera_images = [
                obs["wrist_image"],
                obs["primary_image"],
                obs["secondary_image"],
            ]
            WRIST_IMAGE_IDX = 0
            IMAGE_IDX = 1
            IMAGE2_IDX = 2
        elif cfg.suite == "aloha":
            all_camera_images = [
                obs["left_wrist_image"],
                obs["right_wrist_image"],
                obs["primary_image"],
            ]
            WRIST_IMAGE_IDX = 0
            WRIST_IMAGE2_IDX = 1
            IMAGE_IDX = 2
        else:
            raise ValueError(f"Eval suite not implemented yet: {cfg.suite}")

        # Preprocess images
        # Shape: (N, H, W, C)
        all_camera_images = prepare_images_for_model(all_camera_images, cfg)

        # Process the robot proprioceptive state
        proprio = None
        if cfg.use_proprio:
            proprio = obs["proprio"]
            if cfg.normalize_proprio:
                proprio = rescale_proprio(proprio, dataset_stats, non_negative_only=False, scale_multiplier=1.0)

        # Build the raw image sequence that will be fed to the model (and the VAE tokenizer)
        image_sequence = []
        current_sequence_idx = 0  # Used to track which index in the sequence of images we are on

        # Add blank placeholder image (special placeholder for 1+T temporal VAE compression)
        primary_image = all_camera_images[IMAGE_IDX]
        blank_image = np.zeros_like(primary_image)
        image_sequence.append(np.expand_dims(np.zeros_like(blank_image), axis=0))
        current_sequence_idx += 1

        # Add blank placeholder images for robot proprioceptive state (proprio will be injected into latent later)
        if cfg.use_proprio:
            blank_image_duplicated = duplicate_array(
                blank_image.copy(), total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR
            )
            image_sequence.append(blank_image_duplicated)
            current_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add current wrist image(s)
        wrist_image, wrist_image2 = None, None
        if cfg.use_wrist_image:
            wrist_image = all_camera_images[WRIST_IMAGE_IDX]
            wrist_image_duplicated = duplicate_array(wrist_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
            image_sequence.append(wrist_image_duplicated)
            current_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1
            if cfg.num_wrist_images == 2:
                wrist_image2 = all_camera_images[WRIST_IMAGE2_IDX]
                wrist_image2_duplicated = duplicate_array(
                    wrist_image2, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR
                )
                image_sequence.append(wrist_image2_duplicated)
                current_wrist_image2_latent_idx = current_sequence_idx
                current_sequence_idx += 1
            else:
                current_wrist_image2_latent_idx = -1

        # Add current primary image and optional secondary image (these are third-person images)
        if cfg.use_third_person_image:
            primary_image_duplicated = duplicate_array(
                primary_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR
            )
            image_sequence.append(primary_image_duplicated)
            current_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1
            if cfg.num_third_person_images == 2:
                secondary_image = all_camera_images[IMAGE2_IDX]
                secondary_image_duplicated = duplicate_array(
                    secondary_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR
                )
                image_sequence.append(secondary_image_duplicated)
                current_image2_latent_idx = current_sequence_idx
                current_sequence_idx += 1

        # Add blank placeholder images for action chunk (action chunk will be injected into latent later)
        image_sequence.append(blank_image_duplicated.copy())
        action_latent_idx = current_sequence_idx
        current_sequence_idx += 1

        # Add blank placeholder images for future proprioceptive state (future proprio will be injected into latent later)
        if cfg.use_proprio:
            image_sequence.append(blank_image_duplicated.copy())
            future_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1
        # Add placeholders for the future wrist image(s) - copies of the current wrist image(s)
        if cfg.use_wrist_image:
            image_sequence.append(wrist_image_duplicated.copy())
            future_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1
            if cfg.num_wrist_images == 2:
                image_sequence.append(wrist_image2_duplicated.copy())
                future_wrist_image2_latent_idx = current_sequence_idx
                current_sequence_idx += 1
            else:
                future_wrist_image2_latent_idx = -1
        # Add placeholders for the future primary/secondary images - copies of the current primary/secondary images
        if cfg.use_third_person_image:
            image_sequence.append(primary_image_duplicated.copy())
            future_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1
            if cfg.num_third_person_images == 2:
                image_sequence.append(secondary_image_duplicated.copy())
                future_image2_latent_idx = current_sequence_idx
                current_sequence_idx += 1
            else:
                future_image2_latent_idx = -1

        # Add placeholder for the value (value will be injected into latent later)
        image_sequence.append(blank_image_duplicated.copy())
        value_latent_idx = current_sequence_idx
        current_sequence_idx += 1

        # Prepare input data batch, which is needed for sampling
        # We follow the logic in cosmos_predict2._src.predict2.inference.video2world.py > _get_data_batch_input
        raw_image_sequence = np.concatenate(image_sequence, axis=0)
        raw_image_sequence = np.expand_dims(raw_image_sequence, axis=0)  # (T, H, W, C) -> (1, T, H, W, C)
        raw_image_sequence = np.tile(raw_image_sequence, (batch_size, 1, 1, 1, 1))  # (1, T, H, W, C) -> (B, T, H, W, C)
        raw_image_sequence = np.transpose(raw_image_sequence, (0, 4, 1, 2, 3))  # (B, T, H, W, C) -> (B, C, T, H, W)
        raw_image_sequence = torch.from_numpy(raw_image_sequence).to(dtype=torch.uint8).cuda()
        if cfg.use_proprio:
            # Convert proprio to tensor so that it can be injected into latent later
            proprio_tensor = (
                torch.from_numpy(proprio).reshape(batch_size, -1).to(dtype=torch.bfloat16).cuda()
            )  # (B, proprio_dim)
        # Prepare text embeddings for the data batch
        # Handle both 2D tensors (seq_len, embed_dim) and 3D tensors (batch, seq_len, embed_dim)
        # This supports both T5 embeddings (1, 512, 1024) and Reason1 embeddings (1, 512, 100352)
        if text_embedding.dim() == 2:
            # Add batch dimension if missing (for backwards compatibility with old embeddings)
            text_embedding = text_embedding.unsqueeze(0)
        # If online embeddings were used, they already have shape (batch_size, seq_len, embedding_dim)
        # If cached embeddings were used, they have shape (1, seq_len, embedding_dim) and need to be repeated
        if text_embedding.shape[0] == batch_size:
            text_embedding_batch = text_embedding.to(dtype=torch.bfloat16).cuda()
        else:
            text_embedding_batch = text_embedding.repeat(batch_size, 1, 1).to(dtype=torch.bfloat16).cuda()

        data_batch = {
            "dataset_name": "video_data",
            "video": raw_image_sequence,  # (B, C, T, H, W)
            "t5_text_embeddings": text_embedding_batch,
            "fps": torch.tensor(
                [16] * batch_size, dtype=torch.bfloat16
            ).cuda(),  # Just match the training config (always 16 FPS)
            "padding_mask": torch.zeros(
                (batch_size, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE), dtype=torch.bfloat16
            ).cuda(),  # Padding mask (assume no padding here)
            "num_conditional_frames": model.config.min_num_conditional_frames,  # Number of latent frames used as conditioning
            "proprio": proprio_tensor if cfg.use_proprio else None,
            # Specify the indices of various elements in the latent diffusion sequence
            "current_proprio_latent_idx": (
                torch.tensor([current_proprio_latent_idx] * batch_size, dtype=torch.int64).cuda()
                if cfg.use_proprio
                else torch.tensor([-1] * batch_size, dtype=torch.int64).cuda()
            ),
            "current_wrist_image_latent_idx": (
                torch.tensor([current_wrist_image_latent_idx] * batch_size, dtype=torch.int64).cuda()
                if cfg.use_wrist_image
                else torch.tensor([-1] * batch_size, dtype=torch.int64).cuda()
            ),
            "current_wrist_image2_latent_idx": (
                torch.tensor([current_wrist_image2_latent_idx] * batch_size, dtype=torch.int64).cuda()
                if cfg.use_wrist_image
                else torch.tensor([-1] * batch_size, dtype=torch.int64).cuda()
            ),
            "current_image_latent_idx": (
                torch.tensor([current_image_latent_idx] * batch_size, dtype=torch.int64).cuda()
                if cfg.use_third_person_image
                else torch.tensor([-1] * batch_size, dtype=torch.int64).cuda()
            ),
            "current_image2_latent_idx": (
                torch.tensor([current_image2_latent_idx] * batch_size, dtype=torch.int64).cuda()
                if cfg.use_third_person_image and cfg.num_third_person_images == 2
                else torch.tensor([-1] * batch_size, dtype=torch.int64).cuda()
            ),
            "action_latent_idx": torch.tensor([action_latent_idx] * batch_size, dtype=torch.int64).cuda(),
            "future_proprio_latent_idx": (
                torch.tensor([future_proprio_latent_idx] * batch_size, dtype=torch.int64).cuda()
                if cfg.use_proprio
                else torch.tensor([-1] * batch_size, dtype=torch.int64).cuda()
            ),
            "future_wrist_image_latent_idx": (
                torch.tensor([future_wrist_image_latent_idx] * batch_size, dtype=torch.int64).cuda()
                if cfg.use_wrist_image
                else torch.tensor([-1] * batch_size, dtype=torch.int64).cuda()
            ),
            "future_wrist_image2_latent_idx": (
                torch.tensor([future_wrist_image2_latent_idx] * batch_size, dtype=torch.int64).cuda()
                if cfg.use_wrist_image and cfg.num_wrist_images == 2
                else torch.tensor([-1] * batch_size, dtype=torch.int64).cuda()
            ),
            "future_image_latent_idx": (
                torch.tensor([future_image_latent_idx] * batch_size, dtype=torch.int64).cuda()
                if cfg.use_third_person_image
                else torch.tensor([-1] * batch_size, dtype=torch.int64).cuda()
            ),
            "future_image2_latent_idx": (
                torch.tensor([future_image2_latent_idx] * batch_size, dtype=torch.int64).cuda()
                if cfg.use_third_person_image and cfg.num_third_person_images == 2
                else torch.tensor([-1] * batch_size, dtype=torch.int64).cuda()
            ),
            "value_latent_idx": torch.tensor([value_latent_idx] * batch_size, dtype=torch.int64).cuda(),
        }

        # Generate the output latent sequence - contains the predicted action chunk, future state, and value, but
        # the action chunk is what we care about here
        generated_latent_with_action, orig_clean_latent_frames = model.generate_samples_from_batch(
            data_batch,
            n_sample=batch_size,  # Generate samples
            num_steps=num_denoising_steps_action,
            seed=seed,
            is_negative_prompt=False,  # Negative prompt is for CFG
            use_variance_scale=cfg.use_variance_scale,  # Whether to vary the magnitude of the initial noise - increases diversity slightly in generations
            return_orig_clean_latent_frames=True,  # Return the original (pre-injection) latent frames - needed for future image visualizations
            shift=cfg.shift,  # Shift parameter for rectified flow scheduler
            guidance=0,  # No guidance for action prediction
        )  # (B, C'=16, T', H'=28, W'=28)

        # Extract the predicted action chunk from the generated sample
        action_indices = torch.full(
            (batch_size,), action_latent_idx, dtype=torch.int64, device=generated_latent_with_action.device
        )
        actions = (
            extract_action_chunk_from_latent_sequence(
                generated_latent_with_action, action_shape=(cfg.chunk_size, ACTION_DIM), action_indices=action_indices
            )
            .to(torch.float32)
            .cpu()
            .numpy()
        )

        # Unnormalize actions back to original dataset scale
        if cfg.unnormalize_actions:
            actions = unnormalize_actions(actions, dataset_stats)

        # If generating future state and value in parallel with the actions (instead of autoregressively),
        # extract future state and value predictions from the generated sample now
        if generate_future_state_and_value_in_parallel:
            # Get indices in the generated sample to replace with the original (pre-injection) latent frames so that VAE decoding produces correct images
            if cfg.suite == "libero":
                INDICES_TO_REPLACE = [
                    0,
                    1,
                    4,
                    5,
                ]  # 0: blank, 1: curr proprio, 2: curr wrist img, 3: curr primary img, 4: action, 5: future proprio, 6: future wrist img, 7: future primary img, 8: value
            elif cfg.suite == "robocasa":
                INDICES_TO_REPLACE = [
                    0,
                    1,
                    5,
                    6,
                ]  # 0: blank, 1: curr proprio, 2: curr wrist img, 3: curr primary img, 4: curr secondary img, 5: action, 6: future proprio, 7: future wrist img, 8: future primary img, 9: future secondary img, 10: value
            elif cfg.suite == "aloha":
                INDICES_TO_REPLACE = [
                    0,
                    1,
                    5,
                    6,
                ]  # 0: blank, 1: curr proprio, 2: curr left wrist img, 3: curr right wrist img, 4: curr primary img, 5: action, 6: future proprio, 7: future left wrist img, 8: future right wrist img, 9: future primary img, 10: value
            else:
                raise ValueError(f"Eval suite not implemented yet: {cfg.suite}")
            future_image_predictions = get_future_images_from_generated_samples(
                model,
                generated_latent_with_action.clone(),
                cfg,
                orig_clean_latent_frames,
                INDICES_TO_REPLACE,
                future_wrist_image_latent_idx if cfg.use_wrist_image else -1,
                future_wrist_image2_latent_idx if cfg.use_wrist_image and cfg.num_wrist_images == 2 else -1,
                future_image_latent_idx if cfg.use_third_person_image else -1,
                future_image2_latent_idx if cfg.use_third_person_image and cfg.num_third_person_images == 2 else -1,
                temporal_compression_factor=COSMOS_TEMPORAL_COMPRESSION_FACTOR,
            )
            # Get value predictions from the generated sample
            value_indices = torch.full((batch_size,), -1, dtype=torch.int64, device=generated_latent_with_action.device)
            value_prediction = extract_value_from_latent_sequence(generated_latent_with_action, value_indices)
            # Unnormalize value predictions from [-1, +1] to [0, 1], and clip to [0, 1]
            value_prediction = (value_prediction + 1) / 2
            value_prediction = torch.clamp(value_prediction, min=0, max=1)

        # Return full batch of samples, or just 1 sample if batch_size == 1
        if batch_size > 1:
            # Batch case
            actions_list = []
            for act in actions:
                actions_list.append([act[i] for i in range(len(act))])
            actions = actions_list
            if generate_future_state_and_value_in_parallel:
                future_image_predictions_list = []
                for i in range(batch_size):
                    future_image_predictions_i = {}
                    for k, v in future_image_predictions.items():
                        future_image_predictions_i[k] = v[i]
                    future_image_predictions_list.append(future_image_predictions_i)
                future_image_predictions = future_image_predictions_list
                value_predictions_list = []
                for i in range(batch_size):
                    value_predictions_list.append(value_prediction[i].item())
                value_prediction = value_predictions_list
        else:
            # Single sample case
            actions = actions[0]
            actions = [actions[i] for i in range(len(actions))]
            if generate_future_state_and_value_in_parallel:
                future_image_predictions = {k: v[0] for k, v in future_image_predictions.items() if v is not None}
                value_prediction = value_prediction[0].item()

        # Gather all results into a single return dict
        return_dict = dict(
            actions=actions,
            generated_latent=generated_latent_with_action,
            orig_clean_latent_frames=orig_clean_latent_frames,
            data_batch=data_batch,
            latent_indices=dict(
                action_latent_idx=action_latent_idx,
                future_proprio_latent_idx=future_proprio_latent_idx,
                future_wrist_image_latent_idx=future_wrist_image_latent_idx,
                future_wrist_image2_latent_idx=future_wrist_image2_latent_idx,
                future_image_latent_idx=future_image_latent_idx,
                future_image2_latent_idx=future_image2_latent_idx,
                value_latent_idx=value_latent_idx,
            ),
            all_camera_images=all_camera_images,
            proprio=proprio,
            text_embedding=text_embedding,
        )
        if generate_future_state_and_value_in_parallel:
            return_dict["future_image_predictions"] = future_image_predictions
            return_dict["value_prediction"] = value_prediction

    return return_dict


def get_future_state_prediction(
    cfg,
    model: torch.nn.Module,
    data_batch: Dict[str, Any],
    generated_latent_with_action: torch.Tensor,
    orig_clean_latent_frames: torch.Tensor,
    future_proprio_latent_idx: int,
    future_wrist_image_latent_idx: int,
    future_wrist_image2_latent_idx: int,
    future_image_latent_idx: int,
    future_image2_latent_idx: int,
    seed: int = 1,
    randomize_seed: bool = False,
    num_denoising_steps_future_state: int = 1,
    worker_id: int = 0,
    batch_size: int = 1,
    use_ensemble_future_state_predictions: bool = False,
    num_future_state_predictions_in_ensemble: int = 1,
    future_state_ensemble_aggregation_scheme: str = "last",
) -> List[np.ndarray]:
    """
    Generate future state predictions with the world model.

    Args:
        cfg (Config): Configuration object with parameters
        model (torch.nn.Module): The world model
        data_batch (Dict[str, Any]): Data batch for video model generation
        generated_latent_with_action (torch.Tensor): Generated action sample
        orig_clean_latent_frames (torch.Tensor): Original clean (unnoised) latent frames in the condition object
        future_proprio_latent_idx (int): Index of future proprio in video
        future_wrist_image_latent_idx (int): Index of future wrist image in video
        future_wrist_image2_latent_idx (int): Index of future secondary wrist image in video
        future_image_latent_idx (int): Index of future primary image in video
        future_image2_latent_idx (int): Index of future secondary image in video
        seed (int): Seed for sampling from the model
        randomize_seed (bool): Whether to randomize the seed for sampling actions in each query (still depends on base seed)
        num_denoising_steps_future_state (int): Number of denoising steps to use for future state prediction
        worker_id (int): Worker ID (if using parallel inference)
        batch_size (int): Batch size for inference
        use_ensemble_future_state_predictions (bool): Whether to ensemble future state predictions
        num_future_state_predictions_in_ensemble (int): Number of future state predictions to ensemble (only applicable if doing autoregressive future state prediction)
        future_state_ensemble_aggregation_scheme (str): How to return future state ("average" or "last") (only applicable if doing autoregressive future state prediction)

    Returns:
        Dict[str, Any]: Dictionary containing future state predictions
    """
    # If applicable, randomize the seed used for sampling
    if randomize_seed:
        seed = secrets.randbits(32) % 256

    with torch.inference_mode():
        # Get indices in the generated sample to replace with the original (pre-injection) latent frames so that VAE decoding produces correct images
        if cfg.suite == "libero":
            INDICES_TO_REPLACE = [
                0,
                1,
                4,
                5,
            ]  # 0: blank, 1: curr proprio, 2: curr wrist img, 3: curr primary img, 4: action, 5: future proprio, 6: future wrist img, 7: future primary img, 8: value
        elif cfg.suite == "robocasa":
            INDICES_TO_REPLACE = [
                0,
                1,
                5,
                6,
            ]  # 0: blank, 1: curr proprio, 2: curr wrist img, 3: curr primary img, 4: curr secondary img, 5: action, 6: future proprio, 7: future wrist img, 8: future primary img, 9: future secondary img, 10: value
        elif cfg.suite == "aloha":
            INDICES_TO_REPLACE = [
                0,
                1,
                5,
                6,
            ]  # 0: blank, 1: curr proprio, 2: curr left wrist img, 3: curr right wrist img, 4: curr primary img, 5: action, 6: future proprio, 7: future left wrist img, 8: future right wrist img, 9: future primary img, 10: value
        else:
            raise ValueError(f"Eval suite not implemented yet: {cfg.suite}")

        # Generate future state prediction conditioned on action
        data_batch["num_conditional_frames"] = (
            model.config.min_num_conditional_frames + 1
        )  # 1 more conditional frame for the action chunk
        future_state_samples_list = []
        if use_ensemble_future_state_predictions:
            assert batch_size == 1, "Ensemble future state predictions only supported for batch size 1!"
            # Create varying seeds and number of denoising steps to increase diversity
            if randomize_seed:
                future_state_seeds = [
                    secrets.randbits(32) % 256 for _ in range(num_future_state_predictions_in_ensemble)
                ]
            else:
                future_state_seeds = [seed + i for i in range(num_future_state_predictions_in_ensemble)]
            future_state_denoising_steps_list = (
                np.linspace(1, num_denoising_steps_future_state, num_future_state_predictions_in_ensemble)
                .astype(int)
                .tolist()
            )
            future_image_predictions_list = []
            for i in range(num_future_state_predictions_in_ensemble):
                sample_future_state_i = model.generate_samples_from_batch(
                    data_batch,
                    n_sample=batch_size,
                    num_steps=future_state_denoising_steps_list[i],
                    seed=future_state_seeds[i],
                    is_negative_prompt=False,  # Negative prompt is for CFG
                    use_variance_scale=cfg.use_variance_scale,
                    skip_vae_encoding=True,  # Reuse the latent frames from the generated sample used to get actions
                    previous_generated_latent=generated_latent_with_action,  # Use the generated action sample as the previous sample
                    shift=cfg.shift,  # Shift parameter for rectified flow scheduler
                )  # (B, C'=16, T', H'=28, W'=28)
                future_state_samples_list.append(sample_future_state_i)
                pred_i = get_future_images_from_generated_samples(
                    model,
                    sample_future_state_i.clone(),
                    cfg,
                    orig_clean_latent_frames,
                    INDICES_TO_REPLACE,
                    future_wrist_image_latent_idx if cfg.use_wrist_image else -1,
                    future_wrist_image2_latent_idx if cfg.use_wrist_image and cfg.num_wrist_images == 2 else -1,
                    future_image_latent_idx if cfg.use_third_person_image else -1,
                    (
                        future_image2_latent_idx
                        if cfg.use_third_person_image and cfg.num_third_person_images == 2
                        else -1
                    ),
                    temporal_compression_factor=COSMOS_TEMPORAL_COMPRESSION_FACTOR,
                )
                future_image_predictions_list.append(
                    {k: v.clone() if hasattr(v, "clone") else v for k, v in pred_i.items()}
                )
            # Choose how to return the ensemble of future image predictions
            if future_state_ensemble_aggregation_scheme.lower() == "average":
                # Simply return the average of the ensemble image predictions
                future_image_predictions = average_future_image_predictions(future_image_predictions_list)
            else:  # "last"
                # Just return the last prediction, which is usually the highest-quality one (since it corresponds to the largest number of denoising steps)
                future_image_predictions = future_image_predictions_list[-1]
            # For compatibility, set generated_latent_with_future_state to the last sample
            # This isn't really needed since we also return the full future_state_samples_list
            generated_latent_with_future_state = future_state_samples_list[-1]
        else:
            # Generate a single future state sample with actions fed as condition
            generated_latent_with_future_state = model.generate_samples_from_batch(
                data_batch,
                n_sample=batch_size,
                num_steps=num_denoising_steps_future_state,
                seed=seed,
                is_negative_prompt=False,  # Negative prompt is for CFG
                use_variance_scale=cfg.use_variance_scale,
                skip_vae_encoding=True,  # Reuse the latent frames from the generated sample used to get actions
                previous_generated_latent=generated_latent_with_action,  # Use the generated action sample as the previous sample
                shift=cfg.shift,  # Shift parameter for rectified flow scheduler
            )  # (B, C'=16, T', H'=28, W'=28)
            # Get future image predictions from the generated sample
            future_image_predictions = get_future_images_from_generated_samples(
                model,
                generated_latent_with_future_state.clone(),
                cfg,
                orig_clean_latent_frames,
                INDICES_TO_REPLACE,
                future_wrist_image_latent_idx if cfg.use_wrist_image else -1,
                future_wrist_image2_latent_idx if cfg.use_wrist_image and cfg.num_wrist_images == 2 else -1,
                future_image_latent_idx if cfg.use_third_person_image else -1,
                future_image2_latent_idx if cfg.use_third_person_image and cfg.num_third_person_images == 2 else -1,
                temporal_compression_factor=COSMOS_TEMPORAL_COMPRESSION_FACTOR,
            )

        # Return full batch of samples, or just 1 sample if batch_size == 1
        if batch_size > 1:
            # Batch case
            future_image_predictions_list = []
            for i in range(batch_size):
                future_image_predictions_i = {}
                for k, v in future_image_predictions.items():
                    future_image_predictions_i[k] = v[i]
                future_image_predictions_list.append(future_image_predictions_i)
            future_image_predictions = future_image_predictions_list
        else:
            # Single sample case
            future_image_predictions = {k: v[0] for k, v in future_image_predictions.items() if v is not None}

        return_dict = dict(
            future_image_predictions=future_image_predictions,
            future_state_samples_list=(
                future_state_samples_list
                if use_ensemble_future_state_predictions
                else [generated_latent_with_future_state]
            ),
        )

    return return_dict


def get_value_prediction(
    cfg,
    model: torch.nn.Module,
    data_batch: Dict[str, Any],
    future_state_samples_list,
    seed: int = 1,
    randomize_seed: bool = False,
    num_denoising_steps_value: int = 1,
    worker_id: int = 0,
    batch_size: int = 1,
    use_ensemble_value_predictions: bool = False,
    num_value_predictions_in_ensemble: int = 1,
) -> List[np.ndarray]:
    """
    Generate value predictions with the value function model.

    Args:
        cfg (Config): Configuration object with parameters
        model (torch.nn.Module): The value function model
        data_batch (Dict[str, Any]): Data batch for video model generation
        future_state_samples_list (List[torch.Tensor]): List of future state samples
        seed (int): Seed for sampling from the model
        randomize_seed (bool): Whether to randomize the seed for sampling actions in each query (still depends on base seed)
        num_denoising_steps_value (int): Number of denoising steps to use for value prediction
        worker_id (int): Worker ID (if using parallel inference)
        batch_size (int): Batch size for inference
        use_ensemble_value_predictions (bool): Whether to use ensemble value predictions
        num_value_predictions_in_ensemble (int): Number of value predictions in ensemble

    Returns:
        Dict[str, Any]: Dictionary containing value predictions
    """
    # If applicable, randomize the seed used for sampling
    if randomize_seed:
        seed = secrets.randbits(32) % 256

    with torch.inference_mode():
        # Generate samples with all latent frames except for the final value frame used as conditioning
        data_batch["num_conditional_frames"] = model.config.state_t - 1
        data_batch["mask_current_state_action_for_value_prediction"] = (
            cfg.mask_current_state_action_for_value_prediction
        )  # If True, mask out the current state and action for value prediction

        # For each future state sample we generated before (may be one sample, or more if using ensemble),
        # get one or more value predictions and add to the values list
        all_value_predictions = []
        for fs_sample in future_state_samples_list:
            if use_ensemble_value_predictions:
                assert batch_size == 1, "Ensemble value predictions only supported for batch size 1!"
                # Create varying seeds and number of denoising steps to increase diversity
                if randomize_seed:
                    value_seeds = [secrets.randbits(32) % 256 for _ in range(num_value_predictions_in_ensemble)]
                else:
                    value_seeds = [seed + i for i in range(num_value_predictions_in_ensemble)]
                value_denoising_steps_list = (
                    np.linspace(1, num_denoising_steps_value, num_value_predictions_in_ensemble).astype(int).tolist()
                )
                for i in range(num_value_predictions_in_ensemble):
                    generate_value_sample_kwargs = {
                        "n_sample": batch_size,
                        "num_steps": value_denoising_steps_list[i],
                        "seed": value_seeds[i],
                        "is_negative_prompt": False,  # Negative prompt is for CFG
                        "use_variance_scale": cfg.use_variance_scale,
                        "skip_vae_encoding": True,  # Reuse the latent frames from the future state sample
                        "previous_generated_latent": fs_sample,  # Use the future state sample as the previous sample
                        "shift": cfg.shift,  # Shift parameter for rectified flow scheduler
                    }
                    generated_latent_with_value = model.generate_samples_from_batch(
                        data_batch, **generate_value_sample_kwargs
                    )  # (B, C'=16, T', H'=28, W'=28)
                    # Get value predictions from the generated sample
                    value_indices = torch.full(
                        (batch_size,), -1, dtype=torch.int64, device=generated_latent_with_value.device
                    )
                    value_prediction = extract_value_from_latent_sequence(generated_latent_with_value, value_indices)
                    # Unnormalize value predictions from [-1, +1] to [0, 1], and clip to [0, 1]
                    value_prediction = (value_prediction + 1) / 2
                    value_prediction = torch.clamp(value_prediction, min=0, max=1)
                    # Add the value prediction to the list of all value predictions
                    all_value_predictions.append(value_prediction)
            else:
                if randomize_seed:
                    value_seed = secrets.randbits(32) % 256
                else:
                    value_seed = seed
                generate_value_sample_kwargs = {
                    "n_sample": batch_size,
                    "num_steps": num_denoising_steps_value,
                    "seed": value_seed,
                    "is_negative_prompt": False,  # Negative prompt is for CFG
                    "use_variance_scale": cfg.use_variance_scale,
                    "skip_vae_encoding": True,  # Reuse the latent frames from the future state sample
                    "previous_generated_latent": fs_sample,  # Use the future state sample as the previous sample
                    "shift": cfg.shift,  # Shift parameter for rectified flow scheduler
                }
                # Generate a single value sample with the future state sample fed as condition
                generated_latent_with_value = model.generate_samples_from_batch(
                    data_batch, **generate_value_sample_kwargs
                )  # (B, C'=16, T', H'=28, W'=28)
                # Get value predictions from the generated sample
                value_indices = torch.full(
                    (batch_size,), -1, dtype=torch.int64, device=generated_latent_with_value.device
                )
                value_prediction = extract_value_from_latent_sequence(generated_latent_with_value, value_indices)
                # Unnormalize value predictions from [-1, +1] to [0, 1], and clip to [0, 1]
                value_prediction = (value_prediction + 1) / 2
                value_prediction = torch.clamp(value_prediction, min=0, max=1)
                # Add the value prediction to the list of all value predictions
                all_value_predictions.append(value_prediction)

        # Aggregate the value predictions from the ensemble (or just return the one value prediction if not using ensemble)
        if use_ensemble_value_predictions:
            value_prediction = aggregate_value_predictions(all_value_predictions, cfg.value_ensemble_aggregation_scheme)
        else:
            value_prediction = all_value_predictions[0]

        # Return full batch of samples, or just 1 sample if batch_size == 1
        if batch_size > 1:
            # Batch case
            value_predictions_list = []
            for i in range(batch_size):
                value_predictions_list.append(value_prediction[i].item())
            value_prediction = value_predictions_list
        else:
            # Single sample case
            value_prediction = value_prediction[0].item()

        return_dict = dict(
            value_prediction=value_prediction,
        )

    return return_dict


def get_qvalue_prediction(
    cfg,
    model: torch.nn.Module,
    data_batch: Dict[str, Any],
    action_sample: torch.Tensor,
    seed: int = 1,
    randomize_seed: bool = False,
    num_denoising_steps_value: int = 1,
    worker_id: int = 0,
    batch_size: int = 1,
    use_ensemble_value_predictions: bool = False,
    num_value_predictions_in_ensemble: int = 1,
) -> List[np.ndarray]:
    """
    Generate Q-value predictions with the value function model.

    This is a variant that takes the current state and action as conditioning and generates Q(s, a).
    Note that this requires a separate checkpoint that is trained to predict values conditioned on the current state and action.

    Args:
        cfg (Config): Configuration object with parameters
        model (torch.nn.Module): The value function model
        data_batch (Dict[str, Any]): Data batch for video model generation
        action_sample (torch.Tensor): Generated action sample
        seed (int): Seed for sampling from the model
        randomize_seed (bool): Whether to randomize the seed for sampling actions in each query (still depends on base seed)
        num_denoising_steps_value (int): Number of denoising steps to use for value prediction
        worker_id (int): Worker ID (if using parallel inference)
        batch_size (int): Batch size for inference
        use_ensemble_value_predictions (bool): Whether to use ensemble value predictions
        num_value_predictions_in_ensemble (int): Number of value predictions in ensemble

    Returns:
        Dict[str, Any]: Dictionary containing Q-value predictions
    """
    # If applicable, randomize the seed used for sampling
    if randomize_seed:
        seed = secrets.randbits(32) % 256

    with torch.inference_mode():
        # Generate samples with all latent frames except for the final value frame used as conditioning
        data_batch["num_conditional_frames"] = model.config.state_t - 1
        data_batch["mask_future_state_for_qvalue_prediction"] = (
            cfg.mask_future_state_for_qvalue_prediction
        )  # If True, mask out the predicted future state for Q-value prediction

        # Get Q-value ensemble predictions
        all_value_predictions = []
        if use_ensemble_value_predictions:
            assert batch_size == 1, "Ensemble value predictions only supported for batch size 1!"
            # Create varying seeds and number of denoising steps to increase diversity
            if randomize_seed:
                value_seeds = [secrets.randbits(32) % 256 for _ in range(num_value_predictions_in_ensemble)]
            else:
                value_seeds = [seed + i for i in range(num_value_predictions_in_ensemble)]
            value_denoising_steps_list = (
                np.linspace(1, num_denoising_steps_value, num_value_predictions_in_ensemble).astype(int).tolist()
            )
            for i in range(num_value_predictions_in_ensemble):
                generated_latent_with_value = model.generate_samples_from_batch(
                    data_batch,
                    n_sample=batch_size,
                    num_steps=value_denoising_steps_list[i],
                    seed=value_seeds[i],
                    is_negative_prompt=False,  # Negative prompt is for CFG
                    use_variance_scale=cfg.use_variance_scale,
                    skip_vae_encoding=True,  # Reuse the latent frames from the action sample
                    previous_generated_latent=action_sample,  # Use action sample here instead of future state sample
                    shift=cfg.shift,  # Shift parameter for rectified flow scheduler
                )  # (B, C'=16, T', H'=28, W'=28)

                # Get value predictions from the generated samples
                value_indices = torch.full(
                    (batch_size,), -1, dtype=torch.int64, device=generated_latent_with_value.device
                )
                value_prediction = extract_value_from_latent_sequence(generated_latent_with_value, value_indices)
                # Unnormalize value predictions: Rescale from [-1, +1] to [0, 1], clip to [0, 1]
                value_prediction = (value_prediction + 1) / 2
                value_prediction = torch.clamp(value_prediction, min=0, max=1)
                # Add the value prediction to the list of all value predictions
                all_value_predictions.append(value_prediction)
        else:
            if randomize_seed:
                value_seed = secrets.randbits(32) % 256
            else:
                value_seed = seed
            generated_latent_with_value = model.generate_samples_from_batch(
                data_batch,
                n_sample=batch_size,
                num_steps=num_denoising_steps_value,
                seed=value_seed,
                is_negative_prompt=False,  # Negative prompt is for CFG
                use_variance_scale=cfg.use_variance_scale,
                skip_vae_encoding=True,  # Reuse the latent frames from the action sample
                previous_generated_latent=action_sample,  # Use action sample here instead of future state sample
                shift=cfg.shift,  # Shift parameter for rectified flow scheduler
            )  # (B, C'=16, T', H'=28, W'=28)

            # Get value predictions from the generated sample
            value_indices = torch.full((batch_size,), -1, dtype=torch.int64, device=generated_latent_with_value.device)
            value_prediction = extract_value_from_latent_sequence(generated_latent_with_value, value_indices)
            # Unnormalize value predictions from [-1, +1] to [0, 1], and clip to [0, 1]
            value_prediction = (value_prediction + 1) / 2
            value_prediction = torch.clamp(value_prediction, min=0, max=1)
            # Add the value prediction to the list of all value predictions
            all_value_predictions.append(value_prediction)

        # Aggregate the value predictions from the ensemble (or just return the one value prediction if not using ensemble)
        if use_ensemble_value_predictions:
            value_prediction = aggregate_value_predictions(all_value_predictions, cfg.value_ensemble_aggregation_scheme)
        else:
            value_prediction = all_value_predictions[0]

        # Return full batch of samples, or just 1 sample if batch_size == 1
        if batch_size > 1:
            # Batch case
            value_predictions_list = []
            for i in range(batch_size):
                value_predictions_list.append(value_prediction[i].item())
            value_prediction = value_predictions_list
        else:
            # Single sample case
            value_prediction = value_prediction[0].item()

        return_dict = dict(
            future_image_predictions=None,  # No future state predictions if doing model-free planning with Q-value
            value_prediction=value_prediction,
        )

    return return_dict


class WorkerPoolManager:
    """
    Manages a pool of persistent parallel workers across multiple GPUs (for best-of-N search).
    """

    def __init__(self, cfg, dataset_stats, available_gpus):
        self.cfg = cfg
        self.dataset_stats = dataset_stats
        self.available_gpus = available_gpus
        self.workers = {}
        self.task_queues = {}
        self.result_queue = Queue()
        self.error_queue = Queue()
        self.shutdown_events = {}
        self.initialized = False

    def start_workers(self):
        """Start persistent workers on all available GPUs."""
        if self.initialized:
            return
        print(f"Starting persistent workers on GPUs: {self.available_gpus}")
        # Start workers
        for gpu_id in self.available_gpus:
            task_queue = Queue()
            shutdown_event = Event()
            worker = Process(
                target=persistent_parallel_worker,
                args=(
                    gpu_id,
                    self.cfg,
                    self.dataset_stats,
                    task_queue,
                    self.result_queue,
                    self.error_queue,
                    shutdown_event,
                ),
            )
            worker.start()
            self.workers[gpu_id] = worker
            self.task_queues[gpu_id] = task_queue
            self.shutdown_events[gpu_id] = shutdown_event
        # Wait for all workers to initialize
        initialized_workers = set()
        start_time = time.time()
        timeout = 600  # 10 minutes timeout for initialization
        while len(initialized_workers) < len(self.available_gpus):
            if time.time() - start_time > timeout:
                raise RuntimeError("Worker initialization timeout")
            # Check for initialization completion
            try:
                while not self.result_queue.empty():
                    result = self.result_queue.get_nowait()
                    if result.get("type") == "init_complete":
                        initialized_workers.add(result["gpu_id"])
                        print(f"Worker on GPU {result['gpu_id']} initialized successfully")
            except Exception:
                pass
            # Check for initialization errors
            try:
                while not self.error_queue.empty():
                    error = self.error_queue.get_nowait()
                    if error.get("type") == "init_error":
                        print(f"Worker initialization failed on GPU {error['gpu_id']}: {error['error']}")
                        if "traceback" in error:
                            print(f"GPU {error['gpu_id']} traceback:\n{error['traceback']}")
                        raise RuntimeError(f"Worker initialization failed on GPU {error['gpu_id']}")
            except RuntimeError:
                raise
            except Exception:
                pass
            time.sleep(0.1)
        self.initialized = True
        print(f"All {len(self.available_gpus)} workers initialized successfully")

    def submit_tasks(
        self,
        observation_dict,
        task_description,
    ):
        """Submit tasks to workers and return results."""
        if not self.initialized:
            raise RuntimeError("Workers not initialized. Call start_workers() first.")
        # Submit tasks to workers
        task_ids = []
        for query_idx in range(self.cfg.num_queries_best_of_n):
            task_id = str(uuid.uuid4())
            gpu_id = self.available_gpus[query_idx % len(self.available_gpus)]
            task = {
                "task_id": task_id,
                "observation_dict": observation_dict,
                "task_description": task_description,
            }
            self.task_queues[gpu_id].put(task)
            task_ids.append(task_id)
        # Collect results
        results = {}
        errors = []
        start_time = time.time()
        timeout = self.cfg.parallel_timeout
        while len(results) < self.cfg.num_queries_best_of_n and (time.time() - start_time) < timeout:
            # Check for results
            try:
                while not self.result_queue.empty():
                    result = self.result_queue.get_nowait()
                    if result.get("type") == "result" and result.get("task_id") in task_ids:
                        results[result["task_id"]] = result
            except Exception:
                pass
            # Check for errors
            try:
                while not self.error_queue.empty():
                    error = self.error_queue.get_nowait()
                    if error.get("type") == "task_error" and error.get("task_id") in task_ids:
                        errors.append(error)
            except Exception:
                pass
            time.sleep(0.01)

        # Convert results to list in original order
        query_results = []
        for task_id in task_ids:
            if task_id in results:
                result = results[task_id]
                query_results.append((result["return_dict"]))
        if errors:
            print(f"Parallel inference errors: {len(errors)} tasks failed")
            for error in errors:
                print(f"Task {error['task_id']} on GPU {error['gpu_id']}: {error['error']}")
        return query_results

    def shutdown(self):
        """Shutdown all workers."""
        if not self.initialized:
            return
        print("Shutting down worker pool...")
        # Signal shutdown
        for shutdown_event in self.shutdown_events.values():
            shutdown_event.set()
        # Send None to all task queues to wake up workers
        for task_queue in self.task_queues.values():
            try:
                task_queue.put(None)
            except Exception:
                pass
        # Wait for workers to finish
        for gpu_id, worker in self.workers.items():
            try:
                worker.join(timeout=10)
                if worker.is_alive():
                    print(f"Force terminating worker on GPU {gpu_id}")
                    worker.terminate()
                    worker.join(timeout=5)
                    if worker.is_alive():
                        worker.kill()
            except Exception as e:
                print(f"Error shutting down worker on GPU {gpu_id}: {e}")
        self.initialized = False
        print("Worker pool shutdown complete")


def persistent_parallel_worker(gpu_id, cfg, dataset_stats, task_queue, result_queue, error_queue, shutdown_event):
    """
    Persistent parallel worker function that loads the model once and handles queries in a while loop.
    Used in best-of-N planning with N GPUs working in parallel.
    Runs on one specific GPU and processes tasks from a task queue.
    """
    model = None
    planning_model = None

    try:
        # Initialize CUDA context for this process
        torch.cuda.init()

        # Set CUDA device
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        # Verify GPU is available
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA not available on GPU {gpu_id}")

        # Initialize model on this worker (this happens only once)
        print(f"Loading model on GPU {gpu_id}...")
        model, config = get_model(cfg)
        model = model.to(device)
        print(f"Model loaded successfully on GPU {gpu_id}")

        # Initialize planning model on this worker (this happens only once)
        if cfg.planning_model_ckpt_path != "":
            print(f"Loading planning model on GPU {gpu_id}...")
            planning_model, planning_model_config = get_planning_model(cfg)
            planning_model = planning_model.to(device)
            print(f"Planning model loaded successfully on GPU {gpu_id}")
        else:
            planning_model, planning_model_config = None, None

        # Initialize T5 text embeddings cache on this worker (this happens only once)
        init_t5_text_embeddings_cache(
            cfg.t5_text_embeddings_path, worker_id=gpu_id, embeddings_kind=getattr(cfg, "text_embeddings_kind", "t5")
        )

        # Signal that initialization is complete
        result_queue.put({"gpu_id": gpu_id, "type": "init_complete"})

        # Process tasks until shutdown
        while not shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                task = task_queue.get(timeout=0.01)
                if task is None:
                    break

                # Extract task data and other relevant information
                task_id = task["task_id"]
                observation = task["observation_dict"]
                task_description = task["task_description"]

                # Prepare containers for storing results
                actions_by_depth = []  # Action chunks across all depths of the search
                future_image_predictions_by_depth = []  # Future image predictions across all depths of the search
                value_predictions_by_depth = []  # Value predictions across all depths of the search
                return_dict = {}

                # Query model to get action
                start_time = time.time()
                action_return_dict = get_action(
                    cfg,
                    model,
                    dataset_stats,
                    observation,
                    task_description,
                    seed=cfg.seed + gpu_id,
                    randomize_seed=cfg.randomize_seed,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=not (
                        cfg.ar_future_prediction or cfg.ar_value_prediction or cfg.ar_qvalue_prediction
                    ),
                    worker_id=gpu_id,
                )
                if gpu_id == 0:
                    query_time = time.time() - start_time
                    print(f"-- Query {gpu_id}: Action query time = {query_time:.3f} sec")
                return_dict["actions"] = action_return_dict["actions"]
                actions_by_depth.append(return_dict["actions"])

                if cfg.ar_future_prediction:
                    # Autoregressively query model to get future state prediction
                    start_time = time.time()
                    future_state_return_dict = get_future_state_prediction(
                        cfg,
                        model=planning_model if planning_model is not None else model,
                        data_batch=action_return_dict["data_batch"],
                        generated_latent_with_action=action_return_dict["generated_latent"],
                        orig_clean_latent_frames=action_return_dict["orig_clean_latent_frames"],
                        future_proprio_latent_idx=action_return_dict["latent_indices"]["future_proprio_latent_idx"],
                        future_wrist_image_latent_idx=action_return_dict["latent_indices"][
                            "future_wrist_image_latent_idx"
                        ],
                        future_wrist_image2_latent_idx=action_return_dict["latent_indices"][
                            "future_wrist_image2_latent_idx"
                        ],
                        future_image_latent_idx=action_return_dict["latent_indices"]["future_image_latent_idx"],
                        future_image2_latent_idx=action_return_dict["latent_indices"]["future_image2_latent_idx"],
                        seed=cfg.seed + gpu_id,
                        randomize_seed=cfg.randomize_seed,
                        num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                        use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                        num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                        future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                    )
                    if gpu_id == 0:
                        query_time = time.time() - start_time
                        print(f"-- Query {gpu_id}: Future state prediction query time = {query_time:.3f} sec")
                    return_dict["future_image_predictions"] = future_state_return_dict["future_image_predictions"]
                    future_image_predictions_by_depth.append(return_dict["future_image_predictions"])

                else:
                    if not cfg.ar_qvalue_prediction:
                        return_dict["future_image_predictions"] = action_return_dict["future_image_predictions"]

                if cfg.ar_value_prediction:
                    # Autoregressively query model to get value prediction
                    start_time = time.time()
                    value_return_dict = get_value_prediction(
                        cfg,
                        model=planning_model if planning_model is not None else model,
                        data_batch=action_return_dict["data_batch"],
                        future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                        seed=cfg.seed + gpu_id,
                        randomize_seed=cfg.randomize_seed,
                        num_denoising_steps_value=cfg.num_denoising_steps_value,
                        use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                        num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                    )
                    if gpu_id == 0:
                        query_time = time.time() - start_time
                        print(f"-- Query {gpu_id}: Value prediction query time = {query_time:.3f} sec")
                    return_dict["value_prediction"] = value_return_dict["value_prediction"]
                    value_predictions_by_depth.append(return_dict["value_prediction"])
                    print(f"Query {gpu_id}: Value prediction: {return_dict['value_prediction']:.4f}")
                elif cfg.ar_qvalue_prediction:
                    # Autoregressively query model to get Q-value prediction
                    start_time = time.time()
                    value_return_dict = get_qvalue_prediction(
                        cfg,
                        model=planning_model if planning_model is not None else model,
                        data_batch=action_return_dict["data_batch"],
                        action_sample=action_return_dict["generated_latent"],
                        seed=cfg.seed + gpu_id,
                        randomize_seed=cfg.randomize_seed,
                        num_denoising_steps_value=cfg.num_denoising_steps_value,
                        use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                        num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                    )
                    if gpu_id == 0:
                        query_time = time.time() - start_time
                        print(f"-- Query {gpu_id}: Value prediction query time = {query_time:.3f} sec")
                    return_dict["value_prediction"] = value_return_dict["value_prediction"]
                    value_predictions_by_depth.append(return_dict["value_prediction"])
                    print(f"Query {gpu_id}: Value prediction: {return_dict['value_prediction']:.4f}")
                else:
                    return_dict["value_prediction"] = action_return_dict["value_prediction"]

                # If using search depth > 1, iteratively query model to get next action, future state, and value predictions
                if cfg.search_depth > 1:
                    assert not cfg.ar_qvalue_prediction, "Search depth > 1 not supported for Q(s, a) value prediction!"
                    # Get latent indices from model config
                    (
                        curr_state_start,
                        curr_state_end,
                        future_state_start,
                        future_state_end,
                    ) = get_latent_indices_from_model_config(model)
                    for depth in range(2, cfg.search_depth + 1):
                        for future_state_latent in future_state_return_dict["future_state_samples_list"]:
                            next_generated_latent_with_future_state = future_state_latent.clone()
                            # Rearrange latent frames such that predicted future state replaces current state in the sequence
                            rearranged_next_latent_with_future_state = next_generated_latent_with_future_state.clone()
                            rearranged_next_latent_with_future_state[:, :, curr_state_start : curr_state_end + 1] = (
                                next_generated_latent_with_future_state[:, :, future_state_start : future_state_end + 1]
                            )
                            ################################
                            # Predict next action
                            ################################
                            data_batch = action_return_dict["data_batch"]
                            data_batch["num_conditional_frames"] = (
                                model.config.min_num_conditional_frames
                            )  # Reset to the original value
                            data_batch["mask_current_state_action_for_value_prediction"] = (
                                False  # Don't use input masking for action prediction
                            )
                            if cfg.randomize_seed:
                                seed = secrets.randbits(32) % 256
                            else:
                                seed = cfg.seed + gpu_id
                            batch_size = 1
                            next_generated_latent_with_action, next_orig_clean_latent_frames = (
                                model.generate_samples_from_batch(
                                    data_batch,
                                    n_sample=batch_size,
                                    num_steps=cfg.num_denoising_steps_action,
                                    seed=seed,
                                    is_negative_prompt=False,
                                    use_variance_scale=cfg.use_variance_scale,
                                    skip_vae_encoding=True,
                                    previous_generated_latent=rearranged_next_latent_with_future_state,  # Use future state sample since parts of the value sample might be masked out
                                    return_orig_clean_latent_frames=True,
                                    shift=cfg.shift,  # Shift parameter for rectified flow scheduler
                                )
                            )  # (B, C'=16, T', H'=28, W'=28)
                            # Extract the action chunk prediction from the generated samples
                            action_latent_idx = action_return_dict["latent_indices"]["action_latent_idx"]
                            action_indices = torch.full(
                                (batch_size,),
                                action_latent_idx,
                                dtype=torch.int64,
                                device=next_generated_latent_with_action.device,
                            )
                            next_actions = (
                                extract_action_chunk_from_latent_sequence(
                                    next_generated_latent_with_action,
                                    (cfg.chunk_size, ACTION_DIM),
                                    action_indices=action_indices,
                                )
                                .to(torch.float32)
                                .cpu()
                                .numpy()
                            )
                            # Unnormalize actions
                            if cfg.unnormalize_actions:
                                next_actions = unnormalize_actions(next_actions, dataset_stats)
                            # Squeeze and convert to list
                            next_actions = next_actions[0]
                            next_actions = [next_actions[i] for i in range(len(next_actions))]
                            actions_by_depth.append(next_actions)
                            ################################
                            # Predict next future state
                            ################################
                            future_state_return_dict = get_future_state_prediction(
                                cfg,
                                model=planning_model if planning_model is not None else model,
                                data_batch=action_return_dict["data_batch"],
                                generated_latent_with_action=next_generated_latent_with_action,
                                orig_clean_latent_frames=next_orig_clean_latent_frames,
                                future_proprio_latent_idx=action_return_dict["latent_indices"][
                                    "future_proprio_latent_idx"
                                ],
                                future_wrist_image_latent_idx=action_return_dict["latent_indices"][
                                    "future_wrist_image_latent_idx"
                                ],
                                future_wrist_image2_latent_idx=action_return_dict["latent_indices"][
                                    "future_wrist_image2_latent_idx"
                                ],
                                future_image_latent_idx=action_return_dict["latent_indices"]["future_image_latent_idx"],
                                future_image2_latent_idx=action_return_dict["latent_indices"][
                                    "future_image2_latent_idx"
                                ],
                                seed=cfg.seed + gpu_id,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                                use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                                num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                                future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                            )
                            # Track per-depth prediction
                            future_image_predictions_by_depth.append(
                                future_state_return_dict["future_image_predictions"]
                            )
                            ################################
                            # Predict next value
                            ################################
                            value_return_dict = get_value_prediction(
                                cfg,
                                model=planning_model if planning_model is not None else model,
                                data_batch=action_return_dict["data_batch"],
                                future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                                seed=cfg.seed + gpu_id,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_value=cfg.num_denoising_steps_value,
                                use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                                num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                            )
                            return_dict["value_prediction"] = value_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])
                            print(f"Query {gpu_id + 1}: Value prediction: {return_dict['value_prediction']:.4f}")

                # Add all results to the return dict
                return_dict["future_image_predictions_by_depth"] = future_image_predictions_by_depth
                return_dict["value_predictions_by_depth"] = value_predictions_by_depth
                return_dict["actions_by_depth"] = actions_by_depth

                # Convert tensors to CPU before putting in queue to avoid CUDA context issues
                if isinstance(return_dict["actions"], torch.Tensor):
                    return_dict["action"] = return_dict["action"].cpu()

                # Handle future image predictions
                if "future_image_predictions" in return_dict:
                    for key, value in return_dict["future_image_predictions"].items():
                        if isinstance(value, torch.Tensor):
                            return_dict["future_image_predictions"][key] = value.cpu()

                # Put results in queue
                result_queue.put(
                    {
                        "gpu_id": gpu_id,
                        "task_id": task_id,
                        "type": "result",
                        "return_dict": return_dict,
                    }
                )

            except Exception as e:
                # Only put error in queue if it's not an empty queue error from task_queue.get() (which is normal)
                if not isinstance(e, queue.Empty):
                    error_queue.put(
                        {
                            "gpu_id": gpu_id,
                            "task_id": task.get("task_id", "unknown") if "task" in locals() else "unknown",
                            "type": "task_error",
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )

    except Exception as e:
        error_queue.put({"gpu_id": gpu_id, "type": "init_error", "error": str(e), "traceback": traceback.format_exc()})


def query_model_parallel(
    cfg,
    observation,
    task_description,
    worker_pool,
    timeout=60,
):
    """
    Query the model in parallel across multiple GPUs using a persistent worker pool.

    This function is useful when doing best-of-N search with N GPUs working in parallel.

    Args:
        cfg (Config): Configuration object
        observation (Dict[str, Any]): Input observation dict with keys like "primary_image", "wrist_image", "proprio", etc.
        task_description (str): Task description string
        worker_pool (WorkerPoolManager): WorkerPoolManager instance
        timeout (int): Timeout for the operation (uses cfg.parallel_timeout from worker_pool)

    Returns:
        List[Dict[str, Any]]: List of return_dict dictionaries
    """
    if not worker_pool.initialized:
        raise RuntimeError(
            f"Parallel inference requires initialized worker pool, but got worker_pool.initialized={worker_pool.initialized}"
        )
    # Build observation dict
    observation_dict = {
        "primary_image": observation["primary_image"],
        "proprio": observation["proprio"],
    }
    # Add optional keys if present (e.g., wrist images for LIBERO or ALOHA, secondary_image for RoboCasa)
    if "wrist_image" in observation:
        observation_dict["wrist_image"] = observation["wrist_image"]
    if "left_wrist_image" in observation:
        observation_dict["left_wrist_image"] = observation["left_wrist_image"]
    if "right_wrist_image" in observation:
        observation_dict["right_wrist_image"] = observation["right_wrist_image"]
    if "secondary_image" in observation:
        observation_dict["secondary_image"] = observation["secondary_image"]

    try:
        # Submit tasks to worker pool
        query_results = worker_pool.submit_tasks(
            observation_dict,
            task_description,
        )

        # If we got some results, return them
        if query_results:
            return query_results

    except Exception as e:
        print(f"Parallel inference failed: {e}")
        raise e


def get_action_from_server(
    observation: Dict[str, Any], server_endpoint: str = "http://0.0.0.0:8777/act"
) -> Dict[str, Any]:
    """
    Get action from remote policy inference server.

    Args:
        observation (Dict[str, Any]): Observation data to send to server
        server_endpoint (str): URL of the inference server

    Returns:
        Dict[str, Any]: Action response from server
    """
    response = requests.post(
        server_endpoint,
        json=observation,
    )
    return response.json()
