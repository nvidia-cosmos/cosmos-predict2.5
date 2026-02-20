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

import io
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from tqdm import tqdm


def get_hdf5_files(data_dir: str, is_train: bool | None = None) -> list:
    """
    Recursively get a list of all HDF5 files in the specified directory and its subdirectories,
    including those reached via symbolic links.

    Args:
        data_dir (str): Path to the directory to search
        is_train (bool | None): If None, returns all HDF5 files.
                                If True, returns only files in 'train' subdirectories.
                                If False, returns only files in 'val' subdirectories.

    Returns:
        list: List of paths to HDF5 files
    """
    hdf5_files = []

    # Check if directory exists
    assert os.path.exists(data_dir), f"Error: Directory '{data_dir}' does not exist."

    # Walk through all directories and subdirectories, following symlinks
    for root, dirs, files in os.walk(data_dir, followlinks=True):
        # Get all files with .h5, .hdf5, or .he5 extensions
        for file in files:
            if file.lower().endswith((".h5", ".hdf5", ".he5")):
                filepath = os.path.join(root, file)

                # Filter by train/val if requested
                if is_train is not None:
                    # Check if the file is in a 'train' or 'val' subdirectory
                    path_parts = os.path.normpath(os.path.relpath(filepath, data_dir)).split(os.sep)

                    # Add file to list based on parent directory
                    if is_train and "train" in path_parts:
                        hdf5_files.append(filepath)
                    elif not is_train and "val" in path_parts:
                        hdf5_files.append(filepath)
                else:
                    # If is_train is None, add all files
                    hdf5_files.append(filepath)

    return hdf5_files


def apply_jpeg_compression_np(image_np: np.ndarray, quality: int = 95) -> np.ndarray:
    """Apply JPEG compression/decompression to a NumPy image or batch of images.

    Accepts either a single image with shape (H, W, C) **or** a batch of images
    with shape (B, H, W, C). All inputs must be uint8 RGB.

    Args:
        image_np (np.ndarray): Input image(s) as uint8 array(s).
        quality (int): JPEG quality factor (1–95).

    Returns:
        np.ndarray: JPEG-compressed (and re-decoded) image(s) with the same shape
        as the input.
    """

    def _compress_single(img: np.ndarray) -> np.ndarray:
        """JPEG-compress a single image (H, W, C)."""
        assert img.dtype == np.uint8, f"Expected uint8 image but got {img.dtype}"
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        return np.array(compressed_img)

    # Determine if image_np is a single image or a batch
    if image_np.ndim == 3:
        # Single image
        return _compress_single(image_np)
    elif image_np.ndim == 4:
        # Batch of images: iterate and stack
        compressed = [_compress_single(img) for img in image_np]
        return np.stack(compressed, axis=0)
    else:
        raise ValueError(f"Expected image_np with shape (H, W, C) or (B, H, W, C) but got shape {image_np.shape}")


def decode_single_jpeg_frame(jpeg_bytes: np.ndarray) -> np.ndarray:
    """Decode a single JPEG frame from bytes to numpy array."""
    img = Image.open(io.BytesIO(jpeg_bytes.tobytes()))
    return np.array(img).astype(np.uint8)


def decode_jpeg_bytes_dataset(jpeg_ds) -> np.ndarray:
    """Decode a variable-length JPEG byte dataset (T,) → (T, H, W, 3) uint8."""
    frames = []
    for jpeg_arr in jpeg_ds:
        img = Image.open(io.BytesIO(jpeg_arr.tobytes()))
        frames.append(np.array(img))
    return np.stack(frames, axis=0).astype(np.uint8)


def calculate_dataset_statistics(data):
    """
    Calculate statistics over all actions and proprio in the dataset.

    Args:
        data (dict): Dataset dictionary

    Returns:
        dict: Dataset statistics dictionary
    """
    # First, collect all actions and proprio from all episodes into lists
    all_actions = []
    all_proprio = []
    print("Collecting all actions and proprio...")
    for episode_idx, episode_data in tqdm(data.items()):
        actions = episode_data["actions"]  # Shape: (T, D)
        proprio = episode_data["proprio"]  # Shape: (T, D)
        all_actions.append(actions)
        all_proprio.append(proprio)

    # Concatenate all actions and proprio along the timestep dimension
    # This will give numpy arrays of shape (total_timesteps, D)
    all_actions_array = np.concatenate(all_actions, axis=0)
    all_proprio_array = np.concatenate(all_proprio, axis=0)

    # Calculate statistics along the timestep dimension (axis=0)
    # Each statistic will have shape (D,)
    print("Computing dataset statistics...")
    actions_min = np.min(all_actions_array, axis=0)
    actions_max = np.max(all_actions_array, axis=0)
    actions_mean = np.mean(all_actions_array, axis=0)
    actions_std = np.std(all_actions_array, axis=0)
    actions_median = np.median(all_actions_array, axis=0)

    proprio_min = np.min(all_proprio_array, axis=0)
    proprio_max = np.max(all_proprio_array, axis=0)
    proprio_mean = np.mean(all_proprio_array, axis=0)
    proprio_std = np.std(all_proprio_array, axis=0)
    proprio_median = np.median(all_proprio_array, axis=0)

    # Package all statistics into a dictionary
    stats = {
        "actions_min": actions_min,
        "actions_max": actions_max,
        "actions_mean": actions_mean,
        "actions_std": actions_std,
        "actions_median": actions_median,
        "proprio_min": proprio_min,
        "proprio_max": proprio_max,
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
        "proprio_median": proprio_median,
    }

    return stats


def rescale_data(data, dataset_stats, data_key, non_negative_only=False, scale_multiplier=1.0):
    """
    Rescale some dataset element to the range [-1,+1] or [0,+1].

    If `non_negative_only` is True, then the target range will be [0,+1]. Else, it will be [-1,+1].

    The `scale_multiplier` can be used to change the final range. For example, if `scale_multiplier==2.0`, then
    we use [-2,+2] instead of [-1,+1] (or [0,+2] instead of [0,+1] if `non_negative_only==True`).

    Args:
        data (dict): Dataset dictionary
        dataset_stats (dict): Dataset statistics (pre-normalization)
        data_key (str): Key to the item that should be normalized (e.g., "actions", "proprio")
        scale_multiplier (float): Multiplier to adjust scale from [-1,+1] to [-scale_multiplier,+scale_multiplier]

    Returns:
        dict: Rescaled dataset
    """
    rescaled_data = {}

    for episode_idx, episode_data in data.items():
        arr = episode_data[data_key]
        curr_min = dataset_stats[f"{data_key}_min"]
        curr_max = dataset_stats[f"{data_key}_max"]

        # First, scale to [-1,+1] or [0,+1]:
        # - For [-1,+1]: x_new = 2 * ((x - curr_min) / (curr_max - curr_min)) - 1
        # - For [0,+1]: x_new = (x - curr_min) / (curr_max - curr_min)
        if not non_negative_only:  # [-1,+1]
            rescaled_arr = 2 * ((arr - curr_min) / (curr_max - curr_min)) - 1
        else:  # [0,+1]
            rescaled_arr = (arr - curr_min) / (curr_max - curr_min)

        # Scale to [-scale_multiplier,+scale_multiplier] or [0,+scale_multiplier]
        rescaled_arr = scale_multiplier * rescaled_arr

        # Create a copy of the episode data with rescaling
        rescaled_episode = episode_data.copy()
        rescaled_episode[data_key] = rescaled_arr

        # Store the rescaled episode data
        rescaled_data[episode_idx] = rescaled_episode

    return rescaled_data


def rescale_episode_data(episode_data, dataset_stats, data_key, non_negative_only=False, scale_multiplier=1.0):
    """
    Rescale a single episode's data to the range [-1,+1] or [0,+1].

    Args:
        episode_data (dict): Single episode data dictionary
        dataset_stats (dict): Dataset statistics (pre-normalization)
        data_key (str): Key to the item that should be normalized (e.g., "actions", "proprio")
        non_negative_only (bool): If True, scale to [0,+1], else [-1,+1]
        scale_multiplier (float): Multiplier to adjust scale

    Returns:
        np.ndarray: Rescaled array
    """
    arr = episode_data[data_key]
    curr_min = dataset_stats[f"{data_key}_min"]
    curr_max = dataset_stats[f"{data_key}_max"]

    # First, scale to [-1,+1] or [0,+1]:
    if not non_negative_only:  # [-1,+1]
        rescaled_arr = 2 * ((arr - curr_min) / (curr_max - curr_min)) - 1
    else:  # [0,+1]
        rescaled_arr = (arr - curr_min) / (curr_max - curr_min)

    # Scale to [-scale_multiplier,+scale_multiplier] or [0,+scale_multiplier]
    rescaled_arr = scale_multiplier * rescaled_arr

    return rescaled_arr


def resize_images(images: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resizes multiple images to some target size.

    Assumes that the resulting images will be square.

    Args:
        images (np.ndarray): Input images with shape (T, H, W, C)
        target_size (int): Target image size (square)

    Returns:
        np.ndarray: Resized images with shape (T, target_size, target_size, C)
    """
    assert len(images.shape) == 4, f"Expected 4 dimensions in images but got: {len(images.shape)}"

    # If the images are already the target size, return them as is
    if images.shape[-3:] == (target_size, target_size, 3):
        return images.copy()

    # Get the number of images
    num_images = images.shape[0]

    # Create an empty array for the resized images
    # We assume the channel dimension C remains the same
    C = images.shape[3]
    resized_images = np.empty((num_images, target_size, target_size, C), dtype=images.dtype)

    # Resize each image
    for i in range(num_images):
        resized_images[i] = np.array(Image.fromarray(images[i]).resize((target_size, target_size)))

    return resized_images


def apply_image_aug(images: torch.Tensor, stronger: bool = False) -> torch.Tensor:
    """
    Apply image augmentations to a batch of images represented as a torch.Tensor of shape (C, T, H, W).

    Args:
        images: A torch.Tensor of shape (C, T, H, W) and dtype torch.uint8 representing a set of images.
        stronger (bool): Whether to apply stronger augmentations

    Returns:
        A torch.Tensor of the same shape and dtype with augmentations applied.
    """
    # Get dimensions
    _, _, H, W = images.shape
    assert H == W, "Image height and width must be equal"
    assert images.dtype == torch.uint8, f"Expected images dtype == torch.uint8 but got: {images.dtype}"

    # Convert to (T, C, H, W) format for compatibility with torchvision transforms
    images = images.permute(1, 0, 2, 3)

    # Detect consecutive duplicate images to avoid redundant augmentations
    # Build a list of (start_idx, end_idx, is_duplicate_group) tuples
    unique_groups = []
    num_images = len(images)
    i = 0
    while i < num_images:
        # Check if this image is the same as the next one
        group_start = i
        while i + 1 < num_images and torch.equal(images[i], images[i + 1]):
            i += 1
        group_end = i + 1  # end is exclusive
        group_size = group_end - group_start
        unique_groups.append((group_start, group_end, group_size))
        i += 1

    # Define augmentations with the same parameters for all images
    # 1. Random resized crop
    i, j, h, w = T.RandomResizedCrop.get_params(
        img=torch.zeros(H, W),  # Dummy tensor for getting params
        scale=(0.9, 0.9),  # 90% area
        ratio=(1.0, 1.0),  # Always maintain square aspect ratio
    )

    # 2. Random rotation (only for stronger augmentations)
    if stronger:
        angle = torch.FloatTensor(1).uniform_(-5, 5).item()
    else:
        angle = 0.0  # No rotation in default aug pipeline

    # 3. Color jitter – use wider ranges when `stronger` is True
    if stronger:
        brightness_factor = torch.FloatTensor(1).uniform_(0.7, 1.3).item()  # ±0.3
        contrast_factor = torch.FloatTensor(1).uniform_(0.6, 1.4).item()  # ±0.4
        saturation_factor = torch.FloatTensor(1).uniform_(0.5, 1.5).item()  # ±0.5
    else:
        brightness_factor = torch.FloatTensor(1).uniform_(0.8, 1.2).item()  # ±0.2
        contrast_factor = torch.FloatTensor(1).uniform_(0.8, 1.2).item()  # ±0.2
        saturation_factor = torch.FloatTensor(1).uniform_(0.8, 1.2).item()  # ±0.2
    hue_factor = torch.FloatTensor(1).uniform_(-0.05, 0.05).item()  # 0.05 hue

    # Apply the same transformations to unique images only
    results = []

    for group_idx, (group_start, group_end, group_size) in enumerate(unique_groups):
        # Only augment the first image in each group
        img = images[group_start]

        # 1. Apply random crop and resize back
        img = F.resized_crop(img, i, j, h, w, size=[H, W], antialias=True)

        # 2. Apply random rotation (skip if angle == 0)
        if stronger:
            img = F.rotate(img, angle, expand=False)

        # 3. Apply color jitter
        img = F.adjust_brightness(img, brightness_factor)
        img = F.adjust_contrast(img, contrast_factor)
        img = F.adjust_saturation(img, saturation_factor)
        img = F.adjust_hue(img, hue_factor)

        # Replicate the augmented image for all duplicates in this group
        for _ in range(group_size):
            results.append(img)

    # Combine results and revert to original shape (C, T, H, W)
    augmented_images = torch.stack(results)
    augmented_images = augmented_images.permute(1, 0, 2, 3)

    return augmented_images


def preprocess_image(
    images: np.ndarray,
    final_image_size: int,
    normalize_images: bool = False,
    use_image_aug: bool = True,
    stronger_image_aug: bool = False,
) -> torch.Tensor:
    """
    Preprocesses images for training.

    Resizes to final_image_size, permutes from (T, H, W, C) to (C, T, H, W),
    converts to torch.Tensor, optionally applies image augmentations, and optionally normalizes (no need to
    normalize if, e.g., the dataloader logic will normalize later).

    Args:
        images (np.ndarray): Images to be preprocessed
        final_image_size (int): Target size for resized images (square)
        normalize_images (bool): Whether the images should be normalized in the end
        stronger_image_aug (bool): Whether to apply stronger image augmentations

    Returns:
        torch.Tensor: Preprocessed images
    """
    assert isinstance(images, np.ndarray), f"Images are not of type `np.ndarray`! Got type: {type(images)}"
    assert images.dtype == np.uint8, f"Images do not have dtype `np.uint8`! Got dtype: {images.dtype}"
    assert len(images.shape) == 4 and images.shape[-1] == 3, (
        f"Unexpected images shape! Expected (T, H, W, 3) but got: {images.shape}"
    )

    images = resize_images(images, final_image_size)

    images = np.transpose(images, (3, 0, 1, 2))  # (T, H, W, C) -> (C, T, H, W)
    images = torch.from_numpy(images)
    if use_image_aug:
        images = apply_image_aug(images, stronger=stronger_image_aug)
    if normalize_images:
        # Normalize images and return as dtype torch.float32
        images = images.to(torch.float32)
        images = images / 255.0
        norm_func = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        images = norm_func(images)
    else:
        # Keep images as dtype torch.uint8
        images = images.to(torch.uint8)
    return images
