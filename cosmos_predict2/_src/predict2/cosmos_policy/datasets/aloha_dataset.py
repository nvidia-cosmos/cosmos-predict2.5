# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""
ALOHA robot tasks dataloader.

Run this command to print a few samples from the ALOHA dataset:
    python -m cosmos_predict2._src.predict2.cosmos_policy.datasets.aloha_dataset
"""

import os
import pickle
import random
import time

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_predict2._src.predict2.cosmos_policy.datasets.dataset_common import (
    build_demo_step_index_mapping,
    build_rollout_step_index_mapping,
    calculate_epoch_structure,
    compute_monte_carlo_returns,
    determine_sample_type,
    get_action_chunk_with_padding,
    load_or_compute_dataset_statistics,
    load_or_compute_post_normalization_statistics,
)
from cosmos_predict2._src.predict2.cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    decode_single_jpeg_frame,
    get_hdf5_files,
    preprocess_image,
    rescale_data,
    rescale_episode_data,
    resize_images,
)

# Set floating point precision to 3 decimal places and disable line wrapping
np.set_printoptions(precision=3, linewidth=np.inf)


def load_video_as_images(video_path, resize_size: int = None):
    """
    Loads an MP4 video into a numpy array of images (T, H, W, C) in RGB uint8.

    Args:
        video_path (str): Absolute path to the MP4 file

    Returns:
        np.ndarray: Array of frames (uint8, RGB)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    frames = np.array(frames, dtype=np.uint8)

    if resize_size is not None:
        frames = resize_images(frames, resize_size)

    return frames


def get_video_num_frames(video_path):
    """Return number of frames in an MP4 video using OpenCV metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count <= 0:
        # Fallback to counting by reading (slow path, unlikely)
        return load_video_as_images(video_path).shape[0]
    return frame_count


def get_history_indices(curr_step_index: int, num_history_indices: int, spacing_factor: int) -> tuple:
    """
    Computes the step indices corresponding to the history, given the current step index.

    If any indices would go out of bounds (i.e., be less than 0), we simply return 0 for those indices.

    Args:
        curr_step_index (int): Current step index
        num_history_indices (int): Number of steps in the history
        spacing_factor (int): Spacing factor; returns 1 step in each spacing_factor steps

    Returns:
        tuple: History step indices
    """
    # Create array [num_history_indices, num_history_indices-1, ..., 1]
    steps_back = np.arange(num_history_indices, 0, -1)

    # Calculate indices by multiplying steps_back by spacing_factor and subtracting from current step
    indices = curr_step_index - (steps_back * spacing_factor)

    # Clip negative values to 0
    indices = np.maximum(indices, 0)

    # Convert to tuple and return
    return tuple(indices.tolist())


class ALOHADataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        is_train: bool = True,
        chunk_size: int = 25,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images=False,
        normalize_actions=True,
        normalize_proprio=True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = False,
        debug: bool = False,
        debug2: bool = False,
        use_proprio: bool = False,
        num_history_indices: int = 8,
        history_spacing_factor: int = 12,
        num_duplicates_per_image: int = 8,
        return_value_function_returns: bool = False,
        gamma: float = 0.998,
        lazy_video_decompression: bool = False,
        rollout_data_dir: str = "",
        demonstration_sampling_prob: float = 0.5,
        success_rollout_sampling_prob: float = 0.5,
        treat_demos_as_success_rollouts: bool = False,
        treat_success_rollouts_as_demos: bool = False,
        use_jpeg_for_rollouts: bool = False,
        load_all_rollouts_into_ram: bool = False,
        use_third_person_images: bool = True,
        use_wrist_images: bool = True,
    ):
        """
        Initialize ALOHA dataset for training.

        Args:
            data_dir (str): Path to directory containing preprocessed ALOHA HDF5 files
            is_train (bool): If True, loads train set; else loads val set
            chunk_size (int): Action chunk size
            final_image_size (int): Target size for resized images (square)
            t5_text_embeddings_path (str): Path to precomputed T5 text embeddings dictionary (key: instruction, val: embedding)
            num_images_per_sample (int): Number of images to return per sample
            normalize_images (bool): Whether to normalize the images and return as torch.float32
            normalize_actions (bool): Whether to normalize the actions
            normalize_proprio (bool): Whether to normalize the proprioceptive state
            use_image_aug (bool): Whether to apply image augmentations
            use_stronger_image_aug (bool): Whether to apply stronger image augmentations
            debug (bool): If True, loads only the first episode and returns only the first sample in that episode
            debug2 (bool): If True, loads all episodes but returns only one specific sample in the whole dataset
            use_proprio (bool): If True, adds proprio to image observations
            num_history_indices (int): Number of frames to include in history
            history_spacing_factor (int): Spacing amount between frames in history
            num_duplicates_per_image (int): Temporal compression factor for the image tokenizer
            return_value_function_returns (bool): If True, returns value function returns for rollout episodes
            gamma (float): Discount factor for value function returns
            lazy_video_decompression (bool): Whether to lazily decompress videos
            rollout_data_dir (str): Path to directory containing rollout data (if provided, will load rollout data in addition to base dataset)
            demonstration_sampling_prob (float): Probability of sampling from demonstration data instead of rollout data
            success_rollout_sampling_prob (float): Probability of sampling from success rollout data instead of failure rollout data
            treat_demos_as_success_rollouts (bool): If True, copy demonstration episodes into rollout data as successful rollouts
            treat_success_rollouts_as_demos (bool): If True, copy successful rollout episodes into demonstration dataset (self.data)
            use_third_person_images (bool): This is a null arg that is always True. We need it here to match the signature of the LIBERODataset class.
            use_wrist_images (bool): This is a null arg that is always True. We need it here to match the signature of the LIBERODataset class.
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.debug = debug
        self.debug2 = debug2
        self.use_proprio = use_proprio
        self.num_history_indices = num_history_indices
        self.history_spacing_factor = history_spacing_factor
        self.num_duplicates_per_image = num_duplicates_per_image
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.lazy_video_decompression = lazy_video_decompression
        self.rollout_data_dir = rollout_data_dir
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.success_rollout_sampling_prob = success_rollout_sampling_prob
        self.treat_demos_as_success_rollouts = treat_demos_as_success_rollouts
        self.treat_success_rollouts_as_demos = treat_success_rollouts_as_demos
        self.use_jpeg_for_rollouts = use_jpeg_for_rollouts
        self.load_all_rollouts_into_ram = load_all_rollouts_into_ram
        self._jpeg_rollout_hint_emitted = False

        # Get all HDF5 files in data directory
        hdf5_files = get_hdf5_files(data_dir, is_train=is_train)

        # In debug mode, only load the first demo
        if os.environ.get("DEBUGGING", "False").lower() == "true":
            hdf5_files = hdf5_files[:1]

        # Load all episodes into RAM
        # Save dataset in this structure:
        # self.data = {
        #   episode index: {
        #      images=images,
        #      left_wrist_images=left_wrist_images,
        #      right_wrist_images=right_wrist_images,
        #      proprio=proprio,
        #      actions=actions,
        #      command=command,
        #      num_steps=num_steps,
        #   }
        # }
        self.data = {}
        self.num_episodes = 0
        self.num_steps = 0
        self.unique_commands = set()
        for file in tqdm(hdf5_files):
            with h5py.File(file, "r") as f:
                # Determine storage format: raw RGB frames vs MP4 video paths
                obs_group = f["observations"]
                has_raw_images = "images" in obs_group and all(
                    cam_key in obs_group["images"] for cam_key in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
                )
                has_video_paths = "video_paths" in obs_group and all(
                    cam_key in obs_group["video_paths"] for cam_key in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
                )

                # Auto-detect whether to use MP4 videos
                use_mp4 = has_video_paths and not has_raw_images

                # Load actions and proprio (non-image data)
                actions = f["action"][:]  # (episode_len, action_dim=14), float32
                proprio = f["observations/qpos"][:]  # (episode_len, proprio_dim=14), float32

                if not use_mp4:
                    # Load raw images from HDF5
                    images = obs_group["images"]["cam_high"][:]  # uint8
                    left_wrist_images = obs_group["images"]["cam_left_wrist"][:]
                    right_wrist_images = obs_group["images"]["cam_right_wrist"][:]
                    episode_num_steps = len(images)
                else:
                    # Load MP4 videos
                    def _read_path(ds):
                        val = ds[()]
                        if isinstance(val, bytes):
                            return val.decode("utf-8")
                        return str(val)

                    video_filenames = {
                        "cam_high": _read_path(obs_group["video_paths"]["cam_high"]),
                        "cam_left_wrist": _read_path(obs_group["video_paths"]["cam_left_wrist"]),
                        "cam_right_wrist": _read_path(obs_group["video_paths"]["cam_right_wrist"]),
                    }
                    file_dir = os.path.dirname(file)
                    video_paths = {k: os.path.join(file_dir, v) for k, v in video_filenames.items()}

                    if self.lazy_video_decompression:
                        # Lazy path: store paths and frame count only
                        images = None
                        left_wrist_images = None
                        right_wrist_images = None
                        episode_num_steps = get_video_num_frames(video_paths["cam_high"])  # assume aligned
                    else:
                        # Immediate decompression: load all frames now
                        images = load_video_as_images(
                            video_paths["cam_high"], resize_size=self.final_image_size
                        )  # uint8 RGB
                        left_wrist_images = load_video_as_images(
                            video_paths["cam_left_wrist"], resize_size=self.final_image_size
                        )  # uint8 RGB
                        right_wrist_images = load_video_as_images(
                            video_paths["cam_right_wrist"], resize_size=self.final_image_size
                        )  # uint8 RGB
                        episode_num_steps = len(images)
                # Compute language instruction
                # NOTE: We just hardcode based on the file path for now. Ideally, the demo files would
                #       contain the task description as a string that we extract.
                raw_file_string = file.split("/")[-3]
                if "fold_shirt" in raw_file_string:
                    raw_file_string = "fold_shirt"
                elif "candies_in_bowl" in raw_file_string:
                    raw_file_string = "put_candies_in_bowl"
                elif "candy_in_bag" in raw_file_string:
                    raw_file_string = "put_candy_in_bag"
                elif "flatten_shirt" in raw_file_string:
                    raw_file_string = "flatten_shirt"
                elif "brown_chicken_wing_on_plate" in raw_file_string:
                    raw_file_string = "put_brown_chicken_wing_on_plate"
                elif "purple_eggplant_on_plate" in raw_file_string:
                    raw_file_string = "put_purple_eggplant_on_plate"
                else:
                    raise ValueError(f"Unknown command: {raw_file_string}")
                command = raw_file_string.replace("_", " ")
                self.unique_commands.add(command)
                num_steps = episode_num_steps
                # Add value function returns if applicable
                if self.return_value_function_returns:
                    returns = compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=self.gamma)
                # Add entry to dataset dict
                episode_entry = dict(
                    file_path=file,
                    proprio=proprio,
                    actions=actions,
                    command=command,
                    num_steps=num_steps,
                    returns=returns.copy() if self.return_value_function_returns else None,
                    success=True,
                )

                if use_mp4:
                    if self.lazy_video_decompression:
                        episode_entry["video_paths"] = video_paths
                        episode_entry["is_lazy_video"] = True
                    else:
                        episode_entry["images"] = images
                        episode_entry["left_wrist_images"] = left_wrist_images
                        episode_entry["right_wrist_images"] = right_wrist_images
                        episode_entry["is_lazy_video"] = False
                else:
                    episode_entry["images"] = images
                    episode_entry["left_wrist_images"] = left_wrist_images
                    episode_entry["right_wrist_images"] = right_wrist_images
                    episode_entry["is_lazy_video"] = False

                self.data[self.num_episodes] = episode_entry
                # Update number of episodes
                self.num_episodes += 1
                # Update number of steps
                self.num_steps += num_steps

        # Build mapping from global step index to episode step (demo data)
        self._build_step_index_mapping()

        self.chunk_size = chunk_size

        # If applicable, load precomputed T5 text embeddings
        if t5_text_embeddings_path != "":
            with open(t5_text_embeddings_path, "rb") as file:
                self.t5_text_embeddings = pickle.load(file)

        # Calculate dataset statistics if the stats file doesn't exist
        self.dataset_stats = load_or_compute_dataset_statistics(
            data_dir=self.data_dir,
            data=self.data,
            calculate_dataset_statistics_func=calculate_dataset_statistics,
        )

        # Normalize actions and/or proprio
        if self.normalize_actions or self.normalize_proprio:
            if self.normalize_actions:
                self.data = rescale_data(self.data, self.dataset_stats, "actions")
            if self.normalize_proprio:
                self.data = rescale_data(self.data, self.dataset_stats, "proprio")

            # Calculate post-normalization action statistics
            self.dataset_stats_post_norm = load_or_compute_post_normalization_statistics(
                data_dir=self.data_dir,
                data=self.data,
                calculate_dataset_statistics_func=calculate_dataset_statistics,
            )

        # ====================================================================
        # If applicable, load rollout dataset metadata (lazy loading)
        # Mirrors LIBERODataset design but for ALOHA data format (raw images or MP4 paths)
        # ====================================================================
        self.rollout_episode_metadata = {}  # For lazy loading: episode_idx -> metadata dict
        self.rollout_data = {}  # In-memory rollout data storage; used if treat_demos_as_success_rollouts=True or load_all_rollouts_into_ram=True
        self.rollout_num_episodes = 0
        self.rollout_num_steps = 0

        # If treating demonstrations as success rollouts, add them to rollout data in-memory
        if self.treat_demos_as_success_rollouts:
            for _, episode_data in self.data.items():
                ep_copy = dict(
                    file_path=episode_data["file_path"],
                    images=episode_data.get("images"),
                    left_wrist_images=episode_data.get("left_wrist_images"),
                    right_wrist_images=episode_data.get("right_wrist_images"),
                    proprio=episode_data["proprio"],
                    actions=episode_data["actions"],
                    command=episode_data["command"],
                    num_steps=episode_data["num_steps"],
                    is_lazy_video=episode_data.get("is_lazy_video", False),
                    success=True,
                )
                if self.return_value_function_returns:
                    # Returns already computed for demos; just copy
                    ep_copy["returns"] = episode_data.get("returns")
                if episode_data.get("is_lazy_video", False):
                    ep_copy["video_paths"] = episode_data["video_paths"]
                self.rollout_data[self.rollout_num_episodes] = ep_copy
                self.rollout_num_steps += episode_data["num_steps"]
                self.rollout_num_episodes += 1

        if isinstance(self.rollout_data_dir, str) and len(self.rollout_data_dir) > 0:
            assert os.path.exists(self.rollout_data_dir), (
                f"Error: Rollout data directory '{self.rollout_data_dir}' does not exist."
            )
            rollout_hdf5_files = []
            for root, dirs, files in os.walk(self.rollout_data_dir, followlinks=True):
                for file in files:
                    if file.lower().endswith((".h5", ".hdf5", ".he5")):
                        rollout_hdf5_files.append(os.path.join(root, file))

            # In debug mode, only load the first few rollout files
            if os.environ.get("DEBUGGING", "False").lower() == "true":
                rollout_hdf5_files = rollout_hdf5_files[:10]

            for file in tqdm(rollout_hdf5_files, desc="Loading ALOHA rollout metadata"):
                with h5py.File(file, "r") as f:
                    # Detect formats: top-level JPEG datasets, raw HDF5 images, or MP4 via observations/video_paths
                    obs_group = f["observations"] if "observations" in f else None
                    has_top_jpeg = any(
                        k in f
                        for k in (
                            "primary_images_jpeg",
                            "wrist_images_jpeg",
                            "wrist_left_images_jpeg",
                            "wrist_right_images_jpeg",
                        )
                    )
                    has_raw_images = (
                        obs_group is not None
                        and "images" in obs_group
                        and all(
                            cam_key in obs_group["images"]
                            for cam_key in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
                        )
                    )
                    has_video_paths = (
                        obs_group is not None
                        and "video_paths" in obs_group
                        and all(
                            cam_key in obs_group["video_paths"]
                            for cam_key in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
                        )
                    )
                    use_jpeg = self.use_jpeg_for_rollouts and has_top_jpeg
                    use_mp4 = (not use_jpeg) and has_video_paths and not has_raw_images

                    # Hint: JPEG present but flag not enabled
                    if has_top_jpeg and not self.use_jpeg_for_rollouts and not self._jpeg_rollout_hint_emitted:
                        print(
                            "WARNING: Detected JPEG-compressed rollout images in HDF5 (e.g., 'primary_images_jpeg'), "
                            "but use_jpeg_for_rollouts=False. Set use_jpeg_for_rollouts=True to load these rollouts."
                        )
                        self._jpeg_rollout_hint_emitted = True

                    # Determine number of steps
                    if use_jpeg:
                        if "primary_images_jpeg" in f:
                            num_steps = len(f["primary_images_jpeg"])  # prefer primary length
                        elif "wrist_images_jpeg" in f:
                            num_steps = len(f["wrist_images_jpeg"])  # single wrist
                        elif "wrist_left_images_jpeg" in f:
                            num_steps = len(f["wrist_left_images_jpeg"])  # left-specific
                        elif "wrist_right_images_jpeg" in f:
                            num_steps = len(f["wrist_right_images_jpeg"])  # right-specific
                        else:
                            raise KeyError(f"No JPEG image datasets found in rollout file: {file}")
                    elif use_mp4:
                        # Read video file paths (relative) and compute frame count using cam_high
                        def _read_path(ds):
                            val = ds[()]
                            if isinstance(val, bytes):
                                return val.decode("utf-8")
                            return str(val)

                        video_filenames = {
                            "cam_high": _read_path(obs_group["video_paths"]["cam_high"]),
                            "cam_left_wrist": _read_path(obs_group["video_paths"]["cam_left_wrist"]),
                            "cam_right_wrist": _read_path(obs_group["video_paths"]["cam_right_wrist"]),
                        }
                        file_dir = os.path.dirname(file)
                        video_paths = {k: os.path.join(file_dir, v) for k, v in video_filenames.items()}
                        num_steps = (
                            get_video_num_frames(video_paths["cam_high"])
                            if self.lazy_video_decompression
                            else load_video_as_images(video_paths["cam_high"]).shape[0]
                        )
                    else:
                        num_steps = len(obs_group["images"]["cam_high"])  # raw images

                    # If there happens to be a mismatch between the number of steps computed from the frames and the number of steps in the action array, skip this episode entirely
                    if num_steps != f["action"].shape[0]:
                        print(
                            f"WARNING: For file {file}:\n\tMismatch between number of video frames and action steps: {num_steps} frames != {f['action'].shape[0]} action steps.\n\tSkipping loading this episode."
                        )
                        continue

                    # Command / task description
                    command = f.attrs.get("task_description", "")
                    if command == "":
                        # Fallback: derive from folder name like demos
                        raw_file_string = os.path.basename(os.path.dirname(os.path.dirname(file)))
                        if "fold_shirt" in raw_file_string:
                            raw_file_string = "fold_shirt"
                        elif "candies_in_bowl" in raw_file_string:
                            raw_file_string = "put_candies_in_bowl"
                        elif "candy_in_bag" in raw_file_string:
                            raw_file_string = "put_candy_in_bag"
                        elif "flatten_shirt" in raw_file_string:
                            raw_file_string = "flatten_shirt"
                        elif "brown_chicken_wing_on_plate" in raw_file_string:
                            raw_file_string = "put_brown_chicken_wing_on_plate"
                        elif "purple_eggplant_on_plate" in raw_file_string:
                            raw_file_string = "put_purple_eggplant_on_plate"
                        else:
                            raise ValueError(f"Unknown command: {raw_file_string}")
                        command = raw_file_string.replace("_", " ")
                    self.unique_commands.add(command)

                    success = bool(f.attrs.get("success"))
                    # Use continuous success score if available; fallback to binary success
                    try:
                        success_score = float(f.attrs.get("success_score"))
                    except Exception:
                        success_score = 1.0 if success else 0.0

                    # Store metadata for lazy loading
                    metadata_entry = dict(
                        file_path=file,
                        command=command,
                        num_steps=int(num_steps),
                        success=success,
                        success_score=float(success_score),
                        use_mp4=bool(use_mp4),  # Flag to indicate MP4 compression
                        use_jpeg=bool(use_jpeg),  # Flag to indicate JPEG-compressed frames stored in HDF5
                    )
                    if self.return_value_function_returns:
                        returns = compute_monte_carlo_returns(
                            int(num_steps), terminal_reward=float(success_score), gamma=self.gamma
                        )
                        metadata_entry["returns"] = returns

                    self.rollout_episode_metadata[self.rollout_num_episodes] = metadata_entry
                    self.rollout_num_episodes += 1
                    self.rollout_num_steps += int(num_steps)

            # Optionally eager-load all rollout episodes into RAM
            if self.load_all_rollouts_into_ram and len(self.rollout_episode_metadata) > 0:
                for ep_idx, ep_meta in tqdm(
                    self.rollout_episode_metadata.items(), desc="Preloading rollout episodes into RAM"
                ):
                    ep_entry = self._load_rollout_episode_data(ep_meta)
                    # Store by index for fast lookup in __getitem__
                    self.rollout_data[ep_idx] = ep_entry

            # Optionally copy successful rollout episodes into demonstration dataset
            if self.treat_success_rollouts_as_demos and len(self.rollout_episode_metadata) > 0:
                for ep_idx, ep_meta in self.rollout_episode_metadata.items():
                    if not bool(ep_meta.get("success")):
                        continue

                    # Prefer in-memory rollout data if available; otherwise lazy load once
                    if ep_idx in self.rollout_data:
                        episode_data = self.rollout_data[ep_idx]
                    else:
                        episode_data = self._load_rollout_episode_data(ep_meta)

                    # Insert into demonstration dataset (preserve lazy JPEG/MP4 flags for memory efficiency)
                    ep_copy = dict(
                        file_path=ep_meta.get("file_path"),
                        images=episode_data.get("images"),
                        left_wrist_images=episode_data.get("left_wrist_images"),
                        right_wrist_images=episode_data.get("right_wrist_images"),
                        proprio=episode_data["proprio"],
                        actions=episode_data["actions"],
                        command=episode_data["command"],
                        num_steps=int(episode_data["num_steps"]),
                        is_lazy_video=episode_data.get("is_lazy_video", False),
                        success=True,
                    )
                    if episode_data.get("is_lazy_video", False):
                        ep_copy["video_paths"] = episode_data.get("video_paths")
                    if episode_data.get("is_lazy_jpeg", False):
                        ep_copy["is_lazy_jpeg"] = True
                        ep_copy["jpeg_file_path"] = episode_data.get("jpeg_file_path")
                        ep_copy["jpeg_primary_key"] = episode_data.get("jpeg_primary_key")
                        ep_copy["jpeg_left_key"] = episode_data.get("jpeg_left_key")
                        ep_copy["jpeg_right_key"] = episode_data.get("jpeg_right_key")
                    if "returns" in episode_data:
                        ep_copy["returns"] = episode_data["returns"]

                    self.data[self.num_episodes] = ep_copy
                    self.unique_commands.add(ep_copy["command"])
                    self.num_steps += int(ep_copy["num_steps"])
                    self.num_episodes += 1

                # Rebuild step index mapping to include newly added demos
                self._build_step_index_mapping()

        # Build mapping from global rollout step → (episode, rel_idx) with success/failure split
        if self.rollout_data != {} or self.rollout_episode_metadata != {}:
            self._build_rollout_step_index_mapping()

        # Calculate epoch structure and counts for mixed sampling
        self._calculate_epoch_structure()

    def _build_step_index_mapping(self):
        """Build a mapping from global step index to (episode index, relative index within episode)."""
        result = build_demo_step_index_mapping(self.data)
        self._step_to_episode_map = result["_step_to_episode_map"]
        self._total_steps = result["_total_steps"]

    def _build_rollout_step_index_mapping(self):
        """Build mapping for rollout dataset with separate tracking for successful/failure episodes."""
        result = build_rollout_step_index_mapping(self.rollout_data, self.rollout_episode_metadata)
        self._rollout_success_step_to_episode_map = result["_rollout_success_step_to_episode_map"]
        self._rollout_failure_step_to_episode_map = result["_rollout_failure_step_to_episode_map"]
        self._rollout_success_total_steps = result["_rollout_success_total_steps"]
        self._rollout_failure_total_steps = result["_rollout_failure_total_steps"]
        self._rollout_total_steps = result["_rollout_total_steps"]

    def _calculate_epoch_structure(self):
        """Calculate epoch layout with proper scaling: demos, success rollouts, failure rollouts."""
        # Defaults when no rollout data present
        if not hasattr(self, "_rollout_success_total_steps"):
            self._rollout_success_total_steps = 0
        if not hasattr(self, "_rollout_failure_total_steps"):
            self._rollout_failure_total_steps = 0
        if not hasattr(self, "_rollout_total_steps"):
            self._rollout_total_steps = self._rollout_success_total_steps + self._rollout_failure_total_steps

        result = calculate_epoch_structure(
            num_steps=self.num_steps,
            rollout_success_total_steps=self._rollout_success_total_steps,
            rollout_failure_total_steps=self._rollout_failure_total_steps,
            demonstration_sampling_prob=self.demonstration_sampling_prob,
            success_rollout_sampling_prob=self.success_rollout_sampling_prob,
        )
        self.adjusted_demo_count = result["adjusted_demo_count"]
        self.adjusted_success_rollout_count = result["adjusted_success_rollout_count"]
        self.adjusted_failure_rollout_count = result["adjusted_failure_rollout_count"]
        self.epoch_length = result["epoch_length"]

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        # In debug mode, let the number of samples be 1
        if self.debug:
            return 1
        # In debug2 mode, let the number of samples be 1
        if self.debug2:
            return 1

        # Use mixed epoch length if rollout data is present
        if hasattr(self, "epoch_length") and self.epoch_length > 0:
            return self.epoch_length
        return self.num_steps

    def __getitem__(self, idx):
        """
        Fetches images and action chunk sample by index.
        Returns action chunk rather than just single-step action.
        If the action chunk retrieval would go out of bounds, the last action is repeated however
        many times needed to fill up the chunk.

        Args:
            idx: Integer index to retrieve sample

        Returns:
            dict: Data sample: {
                video=images,
                actions=action chunk,
                t5_text_embeddings=text embedding,
                t5_text_mask=text embedding mask,
                fps=frames per second,
                padding_mask=padding mask,
                num_frames=number of frames per sequence,
                image_size=image size,
            }
        """
        t0 = time.time()
        t_prev = t0
        # In debug mode, always return the sample at index 0
        if self.debug:
            idx = 0
        # In debug2 mode, always return the sample at some index
        if self.debug2:
            idx = 96200

        # Determine which dataset to sample from based on index ranges
        # Layout of indices: [demos] [success rollouts] [failure rollouts]
        sample_type = determine_sample_type(idx, self.adjusted_demo_count, self.adjusted_success_rollout_count)

        if sample_type == "demo":
            global_step_idx = idx % self.num_steps
            # Using global step index, get episode index and relative step index within that episode
            episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
            episode_metadata = None
            episode_data = self.data[episode_idx]
            global_rollout_idx = -1
        elif sample_type == "success_rollout":
            success_idx = idx - self.adjusted_demo_count
            global_rollout_idx = success_idx % max(1, getattr(self, "_rollout_success_total_steps", 1))
            episode_idx, relative_step_idx = self._rollout_success_step_to_episode_map[global_rollout_idx]
            # Check if episode is in memory (from demos treated as success rollouts) or needs lazy loading
            if episode_idx in self.rollout_data:
                episode_metadata = None
                episode_data = self.rollout_data[episode_idx]
            else:
                episode_metadata = self.rollout_episode_metadata[episode_idx]
                episode_data = self._load_rollout_episode_data(episode_metadata)
        else:
            sample_type = "failure_rollout"
            failure_idx = idx - self.adjusted_demo_count - self.adjusted_success_rollout_count
            global_rollout_idx = failure_idx % max(1, getattr(self, "_rollout_failure_total_steps", 1))
            episode_idx, relative_step_idx = self._rollout_failure_step_to_episode_map[global_rollout_idx]
            # Check if episode is in memory (from demos treated as success rollouts) or needs lazy loading
            if episode_idx in self.rollout_data:
                episode_metadata = None
                episode_data = self.rollout_data[episode_idx]
            else:
                episode_metadata = self.rollout_episode_metadata[episode_idx]
                episode_data = self._load_rollout_episode_data(episode_metadata)

        t_prev = time.time()

        # If returning value function samples, randomly choose whether this sample is for
        # world model training or value function training (rollouts only)
        is_world_model_sample = False
        is_value_function_sample = False
        if sample_type != "demo":
            if self.return_value_function_returns:
                p_world_model = 0.5
                if np.random.rand() < p_world_model:
                    is_world_model_sample = True
                    is_value_function_sample = False
                else:
                    is_world_model_sample = False
                    is_value_function_sample = True
            else:
                is_world_model_sample = True
                is_value_function_sample = False

        # Lazy-load videos for this episode if needed (demos or rollouts)
        if episode_data.get("is_lazy_video", False) and (
            ("images" not in episode_data) or (episode_data["images"] is None)
        ):
            video_paths = episode_data["video_paths"]
            images = load_video_as_images(video_paths["cam_high"], resize_size=self.final_image_size)  # uint8
            left_wrist_images = load_video_as_images(
                video_paths["cam_left_wrist"], resize_size=self.final_image_size
            )  # uint8
            right_wrist_images = load_video_as_images(
                video_paths["cam_right_wrist"], resize_size=self.final_image_size
            )  # uint8
            episode_data["images"] = images
            episode_data["left_wrist_images"] = left_wrist_images
            episode_data["right_wrist_images"] = right_wrist_images
            episode_data["is_lazy_video"] = False

        t_prev = time.time()

        # Calculate future frame index if needed (used by lazy JPEG and later logic)
        future_frame_idx = relative_step_idx + self.chunk_size
        max_possible_idx = episode_data["num_steps"] - 1
        if future_frame_idx > max_possible_idx:
            future_frame_idx = max_possible_idx

        # If JPEG-in-HDF5 lazy mode, fetch only required frames for current and future steps
        primary_current = None
        left_current = None
        right_current = None
        primary_future = None
        left_future = None
        right_future = None
        if episode_data.get("is_lazy_jpeg", False):
            jpeg_file = episode_data["jpeg_file_path"]
            primary_key = episode_data.get("jpeg_primary_key")
            left_key = episode_data.get("jpeg_left_key")
            right_key = episode_data.get("jpeg_right_key")

            def _decode_one(ds, idx):
                arr = ds[idx]
                return decode_single_jpeg_frame(arr)

            with h5py.File(jpeg_file, "r") as f_j:
                # Current frames
                if primary_key and primary_key in f_j:
                    primary_current = _decode_one(f_j[primary_key], relative_step_idx)
                if left_key and left_key in f_j:
                    left_current = _decode_one(f_j[left_key], relative_step_idx)
                if right_key and right_key in f_j:
                    right_current = _decode_one(f_j[right_key], relative_step_idx)

                # Future frames
                if primary_key and primary_key in f_j:
                    primary_future = _decode_one(f_j[primary_key], future_frame_idx)
                if left_key and left_key in f_j:
                    left_future = _decode_one(f_j[left_key], future_frame_idx)
                if right_key and right_key in f_j:
                    right_future = _decode_one(f_j[right_key], future_frame_idx)

            # Ensure frames are the expected size
            def _ensure_size(img: np.ndarray) -> np.ndarray:
                if img.shape[0] != self.final_image_size or img.shape[1] != self.final_image_size:
                    return resize_images(np.expand_dims(img, axis=0), self.final_image_size).squeeze(0)
                return img

            primary_current = _ensure_size(primary_current)
            left_current = _ensure_size(left_current)
            right_current = _ensure_size(right_current)
            primary_future = _ensure_size(primary_future)
            left_future = _ensure_size(left_future)
            right_future = _ensure_size(right_future)

        # Build a list of unique frames (no per-frame duplication) and per-frame repeat counts
        # We'll preprocess the unique frames once (same aug params across the whole sequence),
        # then expand by repeat counts to produce the final sequence.
        frames = []  # list of np.ndarray frames with shape (H, W, C)
        repeats = []  # list of ints with how many times to repeat each frame in time dimension
        cum_frames = 0  # cumulative final-frame count for expansion
        segment_idx = 0  # logical segment index used for *_latent_idx
        # Pre-initialize indices that will be filled as we append frames
        action_latent_idx = -1
        value_latent_idx = -1
        current_proprio_latent_idx = -1
        current_wrist_image_latent_idx = -1
        current_wrist_image2_latent_idx = -1
        current_image_latent_idx = -1
        future_proprio_latent_idx = -1
        future_wrist_image_latent_idx = -1
        future_wrist_image2_latent_idx = -1
        future_image_latent_idx = -1

        # future_frame_idx already computed above

        # Add blank first input image (needed for the tokenizer)
        ref_image_for_shape = (
            primary_current if episode_data.get("is_lazy_jpeg", False) else episode_data["images"][relative_step_idx]
        )
        blank_first_input_frame = np.zeros_like(ref_image_for_shape)
        frames.append(blank_first_input_frame)
        repeats.append(1)
        cum_frames += 1
        segment_idx += 1

        # Add current proprio
        if self.use_proprio:
            proprio = episode_data["proprio"][relative_step_idx]
            image = (
                primary_current
                if episode_data.get("is_lazy_jpeg", False)
                else episode_data["images"][relative_step_idx]
            )
            # Proprio values will be injected into latent diffusion sequence later
            # For now just add blank image
            blank_proprio_image = np.zeros_like(
                primary_current
                if episode_data.get("is_lazy_jpeg", False)
                else episode_data["images"][relative_step_idx]
            )
            current_proprio_latent_idx = segment_idx
            frames.append(blank_proprio_image)
            repeats.append(self.num_duplicates_per_image)
            cum_frames += self.num_duplicates_per_image
            segment_idx += 1

        # Add current left wrist image
        left_wrist_image = (
            left_current
            if episode_data.get("is_lazy_jpeg", False)
            else episode_data["left_wrist_images"][relative_step_idx]
        )
        current_wrist_image_latent_idx = segment_idx
        frames.append(left_wrist_image)
        repeats.append(self.num_duplicates_per_image)
        cum_frames += self.num_duplicates_per_image
        segment_idx += 1

        # Add current right wrist image
        right_wrist_image = (
            right_current
            if episode_data.get("is_lazy_jpeg", False)
            else episode_data["right_wrist_images"][relative_step_idx]
        )
        current_wrist_image2_latent_idx = segment_idx
        frames.append(right_wrist_image)
        repeats.append(self.num_duplicates_per_image)
        cum_frames += self.num_duplicates_per_image
        segment_idx += 1

        # Add current primary image
        primary_image = (
            primary_current if episode_data.get("is_lazy_jpeg", False) else episode_data["images"][relative_step_idx]
        )
        current_image_latent_idx = segment_idx
        frames.append(primary_image)
        repeats.append(self.num_duplicates_per_image)
        cum_frames += self.num_duplicates_per_image
        segment_idx += 1

        # Add blank image for action chunk
        blank_action_image = np.zeros_like(
            primary_current if episode_data.get("is_lazy_jpeg", False) else episode_data["images"][relative_step_idx]
        )
        action_latent_idx = segment_idx
        frames.append(blank_action_image)
        repeats.append(self.num_duplicates_per_image)
        cum_frames += self.num_duplicates_per_image
        segment_idx += 1

        # Add future proprio
        if self.use_proprio:
            future_proprio = episode_data["proprio"][future_frame_idx]
            # Proprio values will be injected into latent diffusion sequence later
            # For now just add blank image
            blank_proprio_image = np.zeros_like(
                primary_current
                if episode_data.get("is_lazy_jpeg", False)
                else episode_data["images"][relative_step_idx]
            )
            future_proprio_latent_idx = segment_idx
            frames.append(blank_proprio_image)
            repeats.append(self.num_duplicates_per_image)
            cum_frames += self.num_duplicates_per_image
            segment_idx += 1
        else:
            future_proprio_latent_idx = -1

        # Add future left wrist image
        future_left_wrist_image = (
            left_future
            if episode_data.get("is_lazy_jpeg", False)
            else episode_data["left_wrist_images"][future_frame_idx]
        )
        future_wrist_image_latent_idx = segment_idx
        frames.append(future_left_wrist_image)
        repeats.append(self.num_duplicates_per_image)
        cum_frames += self.num_duplicates_per_image
        segment_idx += 1

        # Add future right wrist image
        future_right_wrist_image = (
            right_future
            if episode_data.get("is_lazy_jpeg", False)
            else episode_data["right_wrist_images"][future_frame_idx]
        )
        future_wrist_image2_latent_idx = segment_idx
        frames.append(future_right_wrist_image)
        repeats.append(self.num_duplicates_per_image)
        cum_frames += self.num_duplicates_per_image
        segment_idx += 1

        # Add future primary image
        future_image = (
            primary_future if episode_data.get("is_lazy_jpeg", False) else episode_data["images"][future_frame_idx]
        )
        future_image_latent_idx = segment_idx
        frames.append(future_image)
        repeats.append(self.num_duplicates_per_image)
        cum_frames += self.num_duplicates_per_image
        segment_idx += 1

        # Add blank value image
        if self.return_value_function_returns:
            value_image = np.zeros_like(
                primary_current
                if episode_data.get("is_lazy_jpeg", False)
                else episode_data["images"][relative_step_idx]
            )
            value_latent_idx = segment_idx
            frames.append(value_image)
            repeats.append(self.num_duplicates_per_image)
            cum_frames += self.num_duplicates_per_image
            segment_idx += 1
        else:
            value_latent_idx = -1

        t_prev = time.time()

        # Sanity: segment indices must be within [0, len(frames)-1]
        num_segments = len(frames)
        for name, val in (
            ("action_latent_idx", action_latent_idx),
            ("value_latent_idx", value_latent_idx),
            ("current_proprio_latent_idx", current_proprio_latent_idx if self.use_proprio else -1),
            ("current_wrist_image_latent_idx", current_wrist_image_latent_idx),
            ("current_wrist_image2_latent_idx", current_wrist_image2_latent_idx),
            ("current_image_latent_idx", current_image_latent_idx),
            ("future_proprio_latent_idx", future_proprio_latent_idx if self.use_proprio else -1),
            ("future_wrist_image_latent_idx", future_wrist_image_latent_idx),
            ("future_wrist_image2_latent_idx", future_wrist_image2_latent_idx),
            ("future_image_latent_idx", future_image_latent_idx),
        ):
            if val != -1:
                assert 0 <= val < num_segments, f"{name}={val} out of range for num_segments={num_segments}"

        # Concatenate unique frames and preprocess once
        all_unique_images = np.stack(frames, axis=0)
        all_unique_images = preprocess_image(
            all_unique_images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )
        t_prev = time.time()
        # Expand unique preprocessed images by repeat counts along time dimension
        lengths = torch.as_tensor(repeats, dtype=torch.long, device=all_unique_images.device)
        all_images = torch.repeat_interleave(all_unique_images, lengths, dim=1)
        # Sanity: expanded length matches repeats sum
        assert all_images.shape[1] == int(lengths.sum().item()), "Expanded T does not match repeats sum"
        t_prev = time.time()

        # Calculate how many actions we can get from the current index
        action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        t_prev = time.time()

        # Get return for value function prediction
        if self.return_value_function_returns:
            if episode_metadata is not None:
                value_function_return = episode_metadata["returns"][future_frame_idx]
            else:
                value_function_return = episode_data["returns"][future_frame_idx]
        else:
            value_function_return = float("-100")  # Just a placeholder

        t_prev = time.time()

        # Calculate and return the next action chunk as well
        next_relative_step_idx = min(relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1)
        next_action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=next_relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        # Calculate next future frame index if needed
        next_future_frame_idx = next_relative_step_idx + self.chunk_size
        max_possible_idx = episode_data["num_steps"] - 1
        if next_future_frame_idx > max_possible_idx:
            next_future_frame_idx = max_possible_idx

        # Next value function return as well
        if self.return_value_function_returns:
            if episode_metadata is not None:
                next_value_function_return = episode_metadata["returns"][next_future_frame_idx]
            else:
                next_value_function_return = episode_data["returns"][next_future_frame_idx]
        else:
            next_value_function_return = float("-100")

        t_prev = time.time()

        rollout_data_mask = 0 if sample_type == "demo" else 1
        rollout_data_success_mask = 1 if sample_type == "success_rollout" else 0

        t_now = time.time()

        return {
            "video": all_images,
            "command": episode_data["command"],
            "actions": action_chunk,
            "t5_text_embeddings": torch.squeeze(self.t5_text_embeddings[episode_data["command"]]),
            "ai_caption": episode_data["command"],
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": proprio if self.use_proprio else np.zeros_like(episode_data["proprio"][relative_step_idx]),
            "future_proprio": (
                future_proprio if self.use_proprio else np.zeros_like(episode_data["proprio"][future_frame_idx])
            ),
            "__key__": idx,
            "value_function_return": value_function_return,
            "next_action_chunk": next_action_chunk,
            "next_value_function_return": next_value_function_return,
            "rollout_data_mask": rollout_data_mask,
            "rollout_data_success_mask": rollout_data_success_mask,
            "world_model_sample_mask": 1 if is_world_model_sample else 0,
            "value_function_sample_mask": 1 if is_value_function_sample else 0,
            "global_rollout_idx": global_rollout_idx,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": value_latent_idx,
            "current_proprio_latent_idx": current_proprio_latent_idx if self.use_proprio else -1,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx,
            "current_wrist_image2_latent_idx": current_wrist_image2_latent_idx,
            "current_image_latent_idx": current_image_latent_idx,
            "future_proprio_latent_idx": future_proprio_latent_idx if self.use_proprio else -1,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx,
            "future_wrist_image2_latent_idx": future_wrist_image2_latent_idx,
            "future_image_latent_idx": future_image_latent_idx,
        }

    def _load_rollout_episode_data(self, episode_metadata):
        """Load rollout episode data from HDF5 file using metadata (raw or MP4). Applies normalization if enabled."""
        file = episode_metadata["file_path"]
        with h5py.File(file, "r") as f:
            obs_group = f["observations"] if "observations" in f else None
            # Load images based on storage format (JPEG in HDF5, raw HDF5 images, or MP4 via paths)
            has_top_jpeg = any(
                k in f
                for k in (
                    "primary_images_jpeg",
                    "wrist_images_jpeg",
                    "wrist_left_images_jpeg",
                    "wrist_right_images_jpeg",
                )
            )
            has_raw_images = (
                obs_group is not None
                and "images" in obs_group
                and all(cam_key in obs_group["images"] for cam_key in ["cam_high", "cam_left_wrist", "cam_right_wrist"])
            )
            has_video_paths = (
                obs_group is not None
                and "video_paths" in obs_group
                and all(
                    cam_key in obs_group["video_paths"] for cam_key in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
                )
            )
            use_jpeg = self.use_jpeg_for_rollouts and has_top_jpeg
            use_mp4 = (not use_jpeg) and has_video_paths and not has_raw_images

            actions = f["action"][:].astype(np.float32)
            proprio = f["observations/qpos"][:].astype(np.float32)

            if use_jpeg:
                # Mark for per-frame JPEG decoding in __getitem__
                images = None
                left_wrist_images = None
                right_wrist_images = None
                is_lazy_video = False
                is_lazy_jpeg = True
                video_paths = None

                # Pick dataset keys
                primary_key = None
                if "primary_images_jpeg" in f:
                    primary_key = "primary_images_jpeg"
                elif "wrist_images_jpeg" in f:
                    primary_key = "wrist_images_jpeg"
                elif "wrist_left_images_jpeg" in f:
                    primary_key = "wrist_left_images_jpeg"
                elif "wrist_right_images_jpeg" in f:
                    primary_key = "wrist_right_images_jpeg"
                else:
                    raise KeyError("No JPEG datasets found to decode primary images")

                left_key = None
                if "wrist_left_images_jpeg" in f:
                    left_key = "wrist_left_images_jpeg"
                elif "wrist_images_jpeg" in f:
                    left_key = "wrist_images_jpeg"

                right_key = None
                if "wrist_right_images_jpeg" in f:
                    right_key = "wrist_right_images_jpeg"

            elif not use_mp4:
                images = obs_group["images"]["cam_high"][:]
                left_wrist_images = obs_group["images"]["cam_left_wrist"][:]
                right_wrist_images = obs_group["images"]["cam_right_wrist"][:]
                is_lazy_video = False
                video_paths = None
            else:

                def _read_path(ds):
                    val = ds[()]
                    if isinstance(val, bytes):
                        return val.decode("utf-8")
                    return str(val)

                video_filenames = {
                    "cam_high": _read_path(obs_group["video_paths"]["cam_high"]),
                    "cam_left_wrist": _read_path(obs_group["video_paths"]["cam_left_wrist"]),
                    "cam_right_wrist": _read_path(obs_group["video_paths"]["cam_right_wrist"]),
                }
                file_dir = os.path.dirname(file)
                video_paths = {k: os.path.join(file_dir, v) for k, v in video_filenames.items()}

                if self.lazy_video_decompression:
                    images = None
                    left_wrist_images = None
                    right_wrist_images = None
                    is_lazy_video = True
                else:
                    images = load_video_as_images(
                        video_paths["cam_high"], resize_size=self.final_image_size
                    )  # uint8 RGB
                    left_wrist_images = load_video_as_images(
                        video_paths["cam_left_wrist"], resize_size=self.final_image_size
                    )  # uint8 RGB
                    right_wrist_images = load_video_as_images(
                        video_paths["cam_right_wrist"], resize_size=self.final_image_size
                    )  # uint8 RGB
                    is_lazy_video = False

            # Apply normalization if needed
            if self.normalize_actions:
                actions = rescale_episode_data({"actions": actions}, self.dataset_stats, "actions")
            if self.normalize_proprio:
                proprio = rescale_episode_data({"proprio": proprio}, self.dataset_stats, "proprio")

            # Create episode data dictionary
            episode_entry = dict(
                images=images,
                left_wrist_images=left_wrist_images,
                right_wrist_images=right_wrist_images,
                proprio=proprio,
                actions=actions,
                command=episode_metadata["command"],
                num_steps=int(episode_metadata["num_steps"]),
                is_lazy_video=is_lazy_video,
            )
            if is_lazy_video:
                episode_entry["video_paths"] = video_paths
            if use_jpeg:
                episode_entry["is_lazy_jpeg"] = True
                episode_entry["jpeg_file_path"] = file
                episode_entry["jpeg_primary_key"] = primary_key
                episode_entry["jpeg_left_key"] = left_key
                episode_entry["jpeg_right_key"] = right_key
            # Include success flag and returns (if available) for downstream bookkeeping
            episode_entry["success"] = bool(episode_metadata.get("success"))
            if "returns" in episode_metadata:
                episode_entry["returns"] = episode_metadata["returns"]

        return episode_entry


if __name__ == "__main__":
    dataset = ALOHADataset(
        data_dir="/lustre/fsw/portfolios/dir/users/user/data/aloha/preprocessed/mixture_20250905_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80_185_demos",
        t5_text_embeddings_path="/lustre/fsw/portfolios/dir/users/user/data/aloha/preprocessed/mixture_20250905_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80_185_demos/t5_embeddings.pkl",
        chunk_size=50,
        use_image_aug=True,
        use_stronger_image_aug=True,
        use_proprio=True,
        normalize_proprio=True,
        normalize_actions=True,
        num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
        treat_demos_as_success_rollouts=False,  # Don't include demos as success rollouts because they have a fixed episode length + we want to focus on real policy rollouts
        demonstration_sampling_prob=0.1,  # Smaller demonstration sampling prob - more emphasis on rollouts
        success_rollout_sampling_prob=0.5,
        return_value_function_returns=True,
        gamma=0.998,  # Higher gamma for ALOHA because episodes can have up to 1.5-2.0K steps  # (s, a, s', v)
        rollout_data_dir="/lustre/fsw/portfolios/dir/users/user/data/aloha/rollout_data/mixture_20250921_648rollouts_505evalSuite_143candyInBag",  # JPEG images
        use_jpeg_for_rollouts=True,  # JPEG images
    )

    # Fetch a sample
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    idx = 50
    sample = dataset[idx]
    print(f"\nImages shape, dtype: {sample['video'].shape, sample['video'].dtype}")
    print(f"Actions shape, dtype: {sample['actions'].shape, sample['actions'].dtype}")
    print(f"Actions:\n{sample['actions']}")
    print(f"T5 text embeddings shape, dtype: {sample['t5_text_embeddings'].shape, sample['t5_text_embeddings'].dtype}")
    print(f"T5 text embeddings:\n{sample['t5_text_embeddings']}")
    print(f"Unique commands: {dataset.unique_commands}")

    # Fetch more samples and save sample images
    os.makedirs("./temp", exist_ok=True)
    for _ in range(50):
        global_step_index = random.randint(0, len(dataset) - 1)
        sample = dataset[global_step_index]
        images = sample["video"].permute(1, 2, 3, 0).numpy()
        for i in range(images.shape[0]):
            image_path = f"./temp/video__global_step_index_{global_step_index}__{sample['command']}__is_rollout={sample['rollout_data_mask']}__global_rollout_idx={sample['global_rollout_idx']}__is_success_rollout={sample['rollout_data_success_mask']}__value_function_return={sample['value_function_return']:.4f}__frame_idx={i}.png"
            Image.fromarray(images[i]).save(image_path)
            print(f"Saved image at path: {image_path}")
