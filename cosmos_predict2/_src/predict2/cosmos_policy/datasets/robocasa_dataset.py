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
RoboCasa simulation benchmark dataloader.

Run this command to print a few samples from the RoboCasa dataset:
    python -m cosmos_predict2._src.predict2.cosmos_policy.datasets.robocasa_dataset
"""

import os
import pickle
import random

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_predict2._src.predict2.cosmos_policy.datasets.dataset_common import (
    build_rollout_step_index_mapping,
    calculate_epoch_structure,
    compute_monte_carlo_returns,
    determine_sample_type,
    load_or_compute_dataset_statistics,
    load_or_compute_post_normalization_statistics,
)
from cosmos_predict2._src.predict2.cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    decode_jpeg_bytes_dataset,
    decode_single_jpeg_frame,
    get_hdf5_files,
    preprocess_image,
    rescale_data,
    rescale_episode_data,
)
from cosmos_predict2._src.predict2.cosmos_policy.utils.utils import duplicate_array

# Set floating point precision to 3 decimal places and disable line wrapping
np.set_printoptions(precision=3, linewidth=np.inf)


class RoboCasaDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 8,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images=False,
        normalize_actions=True,
        normalize_proprio=True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        use_wrist_images: bool = True,
        use_third_person_images: bool = True,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        rollout_data_dir: str = "",
        demonstration_sampling_prob: float = 0.5,
        success_rollout_sampling_prob: float = 0.5,
        p_world_model: float = 0.5,
        treat_success_rollouts_as_demos: bool = False,
        return_value_function_returns: bool = True,
        gamma: float = 0.99,
        lazy_load_demos: bool = False,
        skip_computing_dataset_statistics: bool = False,
    ):
        """
        Initialize RoboCasa dataset for training.

        Args:
            data_dir (str): Path to directory containing RoboCasa dataset HDF5 files
            chunk_size (int): Action chunk size
            final_image_size (int): Target size for resized images (square), defaults to 224
            t5_text_embeddings_path (str): Path to precomputed T5 text embeddings dictionary (key: instruction, val: embedding)
            num_images_per_sample (int): Number of images to return per sample
            normalize_images (bool): Whether to normalize the images and return as torch.float32
            normalize_actions (bool): Whether to normalize the actions
            normalize_proprio (bool): Whether to normalize the proprioceptive state
            use_image_aug (bool): Whether to apply image augmentations
            use_stronger_image_aug (bool): Whether to apply stronger image augmentations
            use_wrist_images (bool): If True, loads wrist-mounted camera images
            use_third_person_images (bool): If True, loads third-person images
            use_proprio (bool): If True, adds proprio to image observations
            num_duplicates_per_image (int): Number of times to duplicate each image (so that each type of image fills 1 latent frame when encoded with the tokenizer)
            rollout_data_dir (str): Path to directory containing rollout data (if provided, will load rollout data in addition to base dataset)
            demonstration_sampling_prob (float): Probability of sampling from demonstration data instead of rollout data
            success_rollout_sampling_prob (float): Probability of sampling from success rollout data instead of failure rollout data
            p_world_model (float): Probability of sampling a world model sample instead of a value function sample
            treat_success_rollouts_as_demos (bool): If True, copy successful rollout episodes into demonstration dataset (self.data)
            return_value_function_returns (bool): If True, returns value function returns for rollout episodes
            gamma (float): Discount factor for value function returns
            lazy_load_demos (bool): If True, only load demo metadata at initialization and load full data on-demand during __getitem__
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
        self.use_wrist_images = use_wrist_images
        self.use_third_person_images = use_third_person_images
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.rollout_data_dir = rollout_data_dir
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.success_rollout_sampling_prob = success_rollout_sampling_prob
        self.p_world_model = p_world_model
        self.treat_success_rollouts_as_demos = treat_success_rollouts_as_demos
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.lazy_load_demos = lazy_load_demos

        assert self.use_wrist_images or self.use_third_person_images, (
            "Must use at least one of wrist images or third-person images!"
        )

        # Get all HDF5 files in data directory
        hdf5_files = get_hdf5_files(data_dir)

        # In debug mode, only load the first demo
        if os.environ.get("DEBUGGING", "False").lower() == "true":
            hdf5_files = hdf5_files[:1]

        # Placeholder list for rollout files (may be empty)
        rollout_hdf5_files = []
        if self.rollout_data_dir:
            assert os.path.exists(self.rollout_data_dir), (
                f"Error: Rollout data directory '{self.rollout_data_dir}' does not exist."
            )
            rollout_hdf5_files = get_hdf5_files(self.rollout_data_dir)

        # Load all episodes into RAM
        # Save dataset in this structure:
        # data = {
        #   episode index: {
        #      left_primary_images=primary images,
        #      right_primary_images=primary images,
        #      wrist_images=wrist images,
        #      proprio=proprio states,
        #      actions=actions,
        #      command=language instruction,
        #      num_steps=number of steps in episode,
        #      returns=observed returns,
        #   }
        # }
        self.data = {}
        self.demo_episode_metadata = {}  # For lazy loading: episode_idx -> metadata dict
        self.rollout_episode_metadata = {}  # For lazy loading: episode_idx -> metadata dict
        self.num_episodes = 0
        self.num_steps = 0
        self.rollout_num_episodes = 0
        self.rollout_num_steps = 0
        self.unique_commands = set()
        self.action_dim = 7  # The full RoboCasa action space has 12 dims, but we only use the first 7 (the rest are for the mobile base)
        if self.demonstration_sampling_prob > 0:  # Only load demos if they are used
            for file in tqdm(hdf5_files):
                with h5py.File(file, "r") as f:
                    # Get demo keys and sort them numerically ("demo_0", "demo_1", ...)
                    demo_keys_list = list(f["data"].keys())
                    sorted_demo_keys = sorted(demo_keys_list, key=lambda x: int(x.split("_")[1]))

                    for demo_key in tqdm(sorted_demo_keys):
                        # Determine whether the dataset stores raw RGB frames or JPEG bytes
                        obs_group = f[f"data/{demo_key}/obs"]

                        # Check if JPEG format
                        is_jpeg = "robot0_agentview_left_rgb_jpeg" in obs_group

                        # Get language instruction
                        command = f[f"data/{demo_key}"].attrs["task_description"]
                        self.unique_commands.add(command)

                        # Get number of steps
                        if is_jpeg:
                            num_steps = len(obs_group["robot0_agentview_left_rgb_jpeg"])
                        else:
                            num_steps = len(obs_group["robot0_agentview_left_rgb"])

                        # Add value function returns if applicable
                        if self.return_value_function_returns:
                            returns = compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=self.gamma)
                        else:
                            returns = None

                        if self.lazy_load_demos:
                            # Store metadata for lazy loading
                            self.demo_episode_metadata[self.num_episodes] = dict(
                                file_path=file,
                                demo_key=demo_key,
                                command=command,
                                num_steps=num_steps,
                                is_jpeg=is_jpeg,
                                returns=returns.copy() if returns is not None else None,
                            )
                        else:
                            # Load full data into RAM (original behavior)
                            # Left agent-view (third-person) images
                            if "robot0_agentview_left_rgb" in obs_group:
                                left_primary_images = obs_group["robot0_agentview_left_rgb"][:]  # (T, H, W, 3) uint8
                            elif "robot0_agentview_left_rgb_jpeg" in obs_group:
                                left_primary_images = decode_jpeg_bytes_dataset(
                                    obs_group["robot0_agentview_left_rgb_jpeg"]
                                )
                            else:
                                raise KeyError(
                                    "Neither 'robot0_agentview_left_rgb' nor 'robot0_agentview_left_rgb_jpeg' found in HDF5 file."
                                )
                            # Right agent-view (third-person) images
                            if "robot0_agentview_right_rgb" in obs_group:
                                right_primary_images = obs_group["robot0_agentview_right_rgb"][:]  # (T, H, W, 3) uint8
                            elif "robot0_agentview_right_rgb_jpeg" in obs_group:
                                right_primary_images = decode_jpeg_bytes_dataset(
                                    obs_group["robot0_agentview_right_rgb_jpeg"]
                                )
                            else:
                                raise KeyError(
                                    "Neither 'robot0_agentview_right_rgb' nor 'robot0_agentview_right_rgb_jpeg' found in HDF5 file."
                                )
                            # Wrist-mounted camera images
                            if "robot0_eye_in_hand_rgb" in obs_group:
                                wrist_images = obs_group["robot0_eye_in_hand_rgb"][:]
                            elif "robot0_eye_in_hand_rgb_jpeg" in obs_group:
                                wrist_images = decode_jpeg_bytes_dataset(obs_group["robot0_eye_in_hand_rgb_jpeg"])
                            else:
                                raise KeyError(
                                    "Neither 'robot0_eye_in_hand_rgb' nor 'robot0_eye_in_hand_rgb_jpeg' found in HDF5 file."
                                )
                            # Actions
                            actions = f[f"data/{demo_key}/actions"][:, : self.action_dim].astype(
                                np.float32
                            )  # (episode_len, action_dim=self.action_dim), float32
                            # Proprio states
                            proprio = f[f"data/{demo_key}/robot_states"][:].astype(
                                np.float32
                            )  # (episode_len, proprio_dim=9), float32

                            # Add entry to dataset dict
                            self.data[self.num_episodes] = dict(
                                left_primary_images=left_primary_images,
                                right_primary_images=right_primary_images,
                                wrist_images=wrist_images,
                                proprio=proprio,
                                actions=actions,
                                command=command,
                                num_steps=num_steps,
                                returns=returns.copy() if returns is not None else None,
                            )

                        # Update number of episodes and steps
                        self.num_episodes += 1
                        self.num_steps += num_steps

        # Build mapping from global step index to episode step
        self._build_step_index_mapping()

        self.chunk_size = chunk_size

        # If applicable, load precomputed T5 text embeddings
        if t5_text_embeddings_path != "":
            with open(t5_text_embeddings_path, "rb") as file:
                self.t5_text_embeddings = pickle.load(file)

        # Calculate dataset statistics if the stats file doesn't exist and we're not skipping this step
        if not skip_computing_dataset_statistics:
            if self.lazy_load_demos and not os.path.exists(os.path.join(self.data_dir, "dataset_statistics.json")):
                raise ValueError(
                    "Dataset statistics file for this dataset does not yet exist. Please rerun with RoboCasaDataset(lazy_load_demos=False) once so that the dataset statistics are computed and saved. Then you can rerun with RoboCasaDataset(lazy_load_demos=True)."
                )
            self.dataset_stats = load_or_compute_dataset_statistics(
                data_dir=self.data_dir,
                data=self.data,
                calculate_dataset_statistics_func=calculate_dataset_statistics,
            )

        # Normalize actions and/or proprio
        if (self.normalize_actions or self.normalize_proprio) and not skip_computing_dataset_statistics:
            # Only normalize self.data if not lazy loading demos (if lazy loading, normalization happens on-demand)
            if not self.lazy_load_demos:
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
        # ====================================================================
        if len(rollout_hdf5_files) > 0:
            for file in tqdm(rollout_hdf5_files, desc="Loading rollout metadata"):
                with h5py.File(file, "r") as f:
                    # Determine storage format of images (raw vs. JPEG)
                    if "primary_images" in f:
                        is_jpeg = False
                        num_steps = len(f["primary_images"])
                    elif "primary_images_jpeg" in f:
                        is_jpeg = True
                        num_steps = len(f["primary_images_jpeg"])
                    else:
                        raise KeyError(f"No primary images found in rollout file: {file}")

                    # Get task description
                    command = f.attrs["task_description"]
                    self.unique_commands.add(command)
                    # Get success flag
                    success = bool(f.attrs.get("success", False))

                    # Store metadata for lazy loading
                    self.rollout_episode_metadata[self.rollout_num_episodes] = dict(
                        file_path=file,
                        command=command,
                        num_steps=num_steps,
                        success=success,
                        is_jpeg=is_jpeg,  # Flag to indicate JPEG compression
                    )
                    # Add value function returns if applicable
                    if self.return_value_function_returns:
                        # Get success label
                        success = bool(f.attrs.get("success"))
                        terminal_reward = 1.0 if success else 0.0
                        returns = compute_monte_carlo_returns(
                            num_steps, terminal_reward=terminal_reward, gamma=self.gamma
                        )
                        self.rollout_episode_metadata[self.rollout_num_episodes]["returns"] = returns.copy()

                    self.rollout_num_episodes += 1
                    self.rollout_num_steps += num_steps

            # If applicable, copy successful rollout episodes into demonstration dataset
            if self.treat_success_rollouts_as_demos:
                for ep_idx, ep_meta in self.rollout_episode_metadata.items():
                    if not ep_meta.get("success", False):
                        continue

                    if self.lazy_load_demos:
                        # Store metadata only for lazy loading
                        # Use precomputed returns
                        returns = ep_meta.get("returns")
                        if returns is not None:
                            returns = returns.copy()

                        # Add to demo metadata with special flag to indicate it's from rollout data
                        self.demo_episode_metadata[self.num_episodes] = dict(
                            file_path=ep_meta["file_path"],
                            demo_key=None,  # Rollout files don't have demo_key structure
                            command=ep_meta.get("command"),
                            num_steps=ep_meta.get("num_steps"),
                            is_jpeg=ep_meta.get("is_jpeg"),
                            returns=returns,
                            is_from_rollout=True,  # Flag to indicate this came from rollout data
                        )
                    else:
                        # Load full data into RAM (original behavior)
                        # Lazy load rollout episode data
                        episode_data = self._load_rollout_episode_data(ep_meta)

                        # Decode to raw uint8 arrays for demos if source was JPEG bytes
                        if episode_data.get("is_jpeg", False):
                            left_primary_images = np.stack(
                                [decode_single_jpeg_frame(b) for b in episode_data["left_primary_images"]], axis=0
                            ).astype(np.uint8)
                            right_primary_images = np.stack(
                                [decode_single_jpeg_frame(b) for b in episode_data["right_primary_images"]], axis=0
                            ).astype(np.uint8)
                            wrist_images = np.stack(
                                [decode_single_jpeg_frame(b) for b in episode_data["wrist_images"]], axis=0
                            ).astype(np.uint8)
                        else:
                            left_primary_images = episode_data["left_primary_images"]
                            right_primary_images = episode_data["right_primary_images"]
                            wrist_images = episode_data["wrist_images"]

                        actions = episode_data["actions"]
                        proprio = episode_data["proprio"]

                        # Use precomputed returns
                        returns = ep_meta.get("returns")
                        if returns is not None:
                            returns = returns.copy()

                        # Insert into demonstration dataset
                        self.data[self.num_episodes] = dict(
                            left_primary_images=left_primary_images,
                            right_primary_images=right_primary_images,
                            wrist_images=wrist_images,
                            proprio=proprio,
                            actions=actions,
                            command=ep_meta.get("command"),
                            num_steps=ep_meta.get("num_steps"),
                            returns=returns,
                        )

                    self.unique_commands.add(ep_meta.get("command"))
                    self.num_episodes += 1
                    self.num_steps += ep_meta.get("num_steps")

                # Rebuild step index mapping to include newly added demos
                self._build_step_index_mapping()

            # Build mapping from global rollout step → (episode, rel_idx)
            self._build_rollout_step_index_mapping()

        print(f"Number of unique commands found: {len(self.unique_commands)}")

        # Calculate epoch structure and counts
        self._calculate_epoch_structure()

    def _calculate_epoch_structure(self):
        """Calculate epoch layout with proper scaling: demos, success rollouts, failure rollouts."""
        # Initialize rollout step counts if not available
        if not hasattr(self, "_rollout_success_total_steps"):
            self._rollout_success_total_steps = 0
        if not hasattr(self, "_rollout_failure_total_steps"):
            self._rollout_failure_total_steps = 0
        if not hasattr(self, "_rollout_total_steps"):
            self._rollout_total_steps = self._rollout_success_total_steps + self._rollout_failure_total_steps

        demo_base_count = self.num_steps

        result = calculate_epoch_structure(
            num_steps=demo_base_count,
            rollout_success_total_steps=self._rollout_success_total_steps,
            rollout_failure_total_steps=self._rollout_failure_total_steps,
            demonstration_sampling_prob=self.demonstration_sampling_prob,
            success_rollout_sampling_prob=self.success_rollout_sampling_prob,
        )
        self.adjusted_demo_count = result["adjusted_demo_count"]
        self.adjusted_success_rollout_count = result["adjusted_success_rollout_count"]
        self.adjusted_failure_rollout_count = result["adjusted_failure_rollout_count"]
        self.epoch_length = result["epoch_length"]

    def _build_step_index_mapping(self):
        """Build a mapping from global step index to (episode index, relative index within episode)."""
        self._step_to_episode_map = {}
        self._total_steps = 0

        # Handle both lazy-loaded demos and regular demos
        if self.lazy_load_demos:
            # Use metadata to build mapping
            for episode_idx, episode_metadata in self.demo_episode_metadata.items():
                num_steps = episode_metadata["num_steps"]
                for i in range(num_steps):
                    self._step_to_episode_map[self._total_steps] = (episode_idx, i)
                    self._total_steps += 1
        else:
            # Use regular data dict
            for episode_idx, episode_data in self.data.items():
                num_steps = episode_data["num_steps"]
                for i in range(num_steps):
                    self._step_to_episode_map[self._total_steps] = (episode_idx, i)
                    self._total_steps += 1

    def _build_rollout_step_index_mapping(self):
        """Build mapping for rollout dataset with separate tracking for successful/failure episodes."""
        result = build_rollout_step_index_mapping({}, self.rollout_episode_metadata)
        self._rollout_success_step_to_episode_map = result["_rollout_success_step_to_episode_map"]
        self._rollout_failure_step_to_episode_map = result["_rollout_failure_step_to_episode_map"]
        self._rollout_success_total_steps = result["_rollout_success_total_steps"]
        self._rollout_failure_total_steps = result["_rollout_failure_total_steps"]
        self._rollout_total_steps = result["_rollout_total_steps"]

    def _load_demo_episode_data(self, episode_metadata, frame_indices=None, action_start_idx=None, action_end_idx=None):
        """
        Load demo episode data from HDF5 file using metadata.
        Optimized to only load required frames and action chunks.

        Args:
            episode_metadata (dict): Episode metadata containing file_path, demo_key, etc.
            frame_indices (set or None): Set of frame indices to load. If None, loads all frames.
            action_start_idx (int or None): Start index for action chunk. If None, loads all actions.
            action_end_idx (int or None): End index for action chunk (exclusive). If None, loads all actions.

        Returns:
            dict: Episode data dictionary with loaded arrays (only requested frames/actions)
        """
        # Check if this metadata is from rollout data (added via treat_success_rollouts_as_demos)
        if episode_metadata.get("is_from_rollout", False):
            # Use the rollout loading method
            return self._load_rollout_episode_data(episode_metadata, frame_indices, action_start_idx, action_end_idx)

        file_path = episode_metadata["file_path"]
        demo_key = episode_metadata["demo_key"]

        with h5py.File(file_path, "r") as f:
            obs_group = f[f"data/{demo_key}/obs"]

            # Determine if we should load all frames or specific ones
            load_all = frame_indices is None

            # Load images based on storage format
            if episode_metadata["is_jpeg"]:
                # For JPEG, we need to load and decode specific frames
                if load_all:
                    left_primary_images = decode_jpeg_bytes_dataset(obs_group["robot0_agentview_left_rgb_jpeg"])
                    right_primary_images = decode_jpeg_bytes_dataset(obs_group["robot0_agentview_right_rgb_jpeg"])
                    wrist_images = decode_jpeg_bytes_dataset(obs_group["robot0_eye_in_hand_rgb_jpeg"])
                else:
                    # Load only specific frames
                    frame_indices_list = sorted(list(frame_indices))
                    left_primary_images = {}
                    right_primary_images = {}
                    wrist_images = {}
                    for idx in frame_indices_list:
                        left_primary_images[idx] = decode_single_jpeg_frame(
                            obs_group["robot0_agentview_left_rgb_jpeg"][idx]
                        )
                        right_primary_images[idx] = decode_single_jpeg_frame(
                            obs_group["robot0_agentview_right_rgb_jpeg"][idx]
                        )
                        wrist_images[idx] = decode_single_jpeg_frame(obs_group["robot0_eye_in_hand_rgb_jpeg"][idx])
            else:
                if load_all:
                    left_primary_images = obs_group["robot0_agentview_left_rgb"][:]
                    right_primary_images = obs_group["robot0_agentview_right_rgb"][:]
                    wrist_images = obs_group["robot0_eye_in_hand_rgb"][:]
                else:
                    # Load only specific frames (HDF5 supports fancy indexing but dict is more flexible)
                    frame_indices_list = sorted(list(frame_indices))
                    left_primary_images = {}
                    right_primary_images = {}
                    wrist_images = {}
                    for idx in frame_indices_list:
                        left_primary_images[idx] = obs_group["robot0_agentview_left_rgb"][idx]
                        right_primary_images[idx] = obs_group["robot0_agentview_right_rgb"][idx]
                        wrist_images[idx] = obs_group["robot0_eye_in_hand_rgb"][idx]

            # Load actions - only load required chunk
            if action_start_idx is not None and action_end_idx is not None:
                # Load only the required action chunk
                actions = f[f"data/{demo_key}/actions"][action_start_idx:action_end_idx, : self.action_dim].astype(
                    np.float32
                )
            else:
                # Load all actions
                actions = f[f"data/{demo_key}/actions"][:, : self.action_dim].astype(np.float32)

            # Load proprio - only load required timesteps
            if frame_indices is not None and not load_all:
                # Load only specific timesteps
                frame_indices_list = sorted(list(frame_indices))
                proprio = {}
                for idx in frame_indices_list:
                    proprio[idx] = f[f"data/{demo_key}/robot_states"][idx].astype(np.float32)
            else:
                # Load all proprio
                proprio = f[f"data/{demo_key}/robot_states"][:].astype(np.float32)

            # Apply normalization if needed
            if self.normalize_actions:
                actions = rescale_episode_data({"actions": actions}, self.dataset_stats, "actions")
            if self.normalize_proprio:
                if isinstance(proprio, dict):
                    # Normalize each timestep separately
                    for idx in proprio:
                        proprio_array = proprio[idx].reshape(1, -1)
                        proprio[idx] = rescale_episode_data(
                            {"proprio": proprio_array}, self.dataset_stats, "proprio"
                        ).flatten()
                else:
                    proprio = rescale_episode_data({"proprio": proprio}, self.dataset_stats, "proprio")

            # Create episode data dictionary
            episode_data = dict(
                left_primary_images=left_primary_images,
                right_primary_images=right_primary_images,
                wrist_images=wrist_images,
                proprio=proprio,
                actions=actions,
                command=episode_metadata["command"],
                num_steps=episode_metadata["num_steps"],
                is_jpeg=episode_metadata["is_jpeg"],
            )

            return episode_data

    def _load_rollout_episode_data(
        self, episode_metadata, frame_indices=None, action_start_idx=None, action_end_idx=None
    ):
        """
        Load rollout episode data from HDF5 file using metadata.
        Optimized to only load required frames and action chunks.

        Args:
            episode_metadata (dict): Episode metadata containing file_path, success, etc.
            frame_indices (set or None): Set of frame indices to load. If None, loads all frames.
            action_start_idx (int or None): Start index for action chunk. If None, loads all actions.
            action_end_idx (int or None): End index for action chunk (exclusive). If None, loads all actions.

        Returns:
            dict: Episode data dictionary with loaded arrays (only requested frames/actions)
        """
        file_path = episode_metadata["file_path"]

        with h5py.File(file_path, "r") as f:
            # Determine if we should load all frames or specific ones
            load_all = frame_indices is None

            # Load images based on storage format
            if episode_metadata["is_jpeg"]:
                if load_all:
                    # Store raw JPEG bytes
                    left_primary_images = f["primary_images_jpeg"][:]
                    right_primary_images = f["secondary_images_jpeg"][:]
                    wrist_images = f["wrist_images_jpeg"][:]
                else:
                    # Load only specific frames
                    frame_indices_list = sorted(list(frame_indices))
                    left_primary_images = {}
                    right_primary_images = {}
                    wrist_images = {}
                    for idx in frame_indices_list:
                        left_primary_images[idx] = f["primary_images_jpeg"][idx]
                        right_primary_images[idx] = f["secondary_images_jpeg"][idx]
                        wrist_images[idx] = f["wrist_images_jpeg"][idx]
            else:
                if load_all:
                    left_primary_images = f["primary_images"][:]
                    right_primary_images = f["secondary_images"][:]
                    wrist_images = f["wrist_images"][:]
                else:
                    # Load only specific frames
                    frame_indices_list = sorted(list(frame_indices))
                    left_primary_images = {}
                    right_primary_images = {}
                    wrist_images = {}
                    for idx in frame_indices_list:
                        left_primary_images[idx] = f["primary_images"][idx]
                        right_primary_images[idx] = f["secondary_images"][idx]
                        wrist_images[idx] = f["wrist_images"][idx]

            # Load actions - only load required chunk
            if action_start_idx is not None and action_end_idx is not None:
                # Load only the required action chunk
                actions = f["actions"][action_start_idx:action_end_idx, : self.action_dim].astype(np.float32)
            else:
                # For actions, only keep the first self.action_dim dimensions (the rest are for the mobile base)
                actions = f["actions"][:, : self.action_dim].astype(np.float32)

            # Load proprio - only load required timesteps
            if frame_indices is not None and not load_all:
                # Load only specific timesteps
                frame_indices_list = sorted(list(frame_indices))
                proprio = {}
                for idx in frame_indices_list:
                    proprio[idx] = f["proprio"][idx].astype(np.float32)
            else:
                proprio = f["proprio"][:].astype(np.float32)

            # Apply normalization if needed
            if self.normalize_actions:
                actions = rescale_episode_data({"actions": actions}, self.dataset_stats, "actions")
            if self.normalize_proprio:
                if isinstance(proprio, dict):
                    # Normalize each timestep separately
                    for idx in proprio:
                        proprio_array = proprio[idx].reshape(1, -1)
                        proprio[idx] = rescale_episode_data(
                            {"proprio": proprio_array}, self.dataset_stats, "proprio"
                        ).flatten()
                else:
                    proprio = rescale_episode_data(
                        {"proprio": proprio},
                        self.dataset_stats,
                        "proprio",
                    )

            # Create episode data dictionary
            episode_data = dict(
                left_primary_images=left_primary_images,
                right_primary_images=right_primary_images,
                wrist_images=wrist_images,
                proprio=proprio,
                actions=actions,
                command=episode_metadata["command"],
                num_steps=episode_metadata["num_steps"],
                success=episode_metadata["success"],
                is_jpeg=episode_metadata["is_jpeg"],
            )

            return episode_data

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        # Return pre-calculated epoch length
        return self.epoch_length

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
                proprio=proprio state,
                __key__=unique sample identifier,
            }
        """

        # Determine which dataset to sample from based on index ranges
        # Layout of indices within dataset: [demos] [success rollouts] [failure rollouts]
        sample_type = determine_sample_type(idx, self.adjusted_demo_count, self.adjusted_success_rollout_count)

        rollout_data_mask = 1 if sample_type != "demo" else 0
        rollout_data_success_mask = 1 if sample_type == "success_rollout" else 0

        if sample_type == "demo":
            # Get demonstration sample
            # Cycle through demo samples
            global_step_idx = idx % self.num_steps
            # Using global step index, get episode index and relative step index within that episode
            episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
            episode_metadata = None

            # Load episode data (either from RAM or on-demand from HDF5)
            if self.lazy_load_demos:
                episode_metadata = self.demo_episode_metadata[episode_idx]
                # Calculate which frames we need to load
                future_frame_idx_temp = min(relative_step_idx + self.chunk_size, episode_metadata["num_steps"] - 1)
                frame_indices_needed = {relative_step_idx, future_frame_idx_temp}
                # Calculate which actions we need
                action_end_idx = min(relative_step_idx + self.chunk_size, episode_metadata["num_steps"])
                # Load only required data
                episode_data = self._load_demo_episode_data(
                    episode_metadata,
                    frame_indices=frame_indices_needed,
                    action_start_idx=relative_step_idx,
                    action_end_idx=action_end_idx,
                )
            else:
                episode_data = self.data[episode_idx]

            global_rollout_idx = -1  # Not applicable for demonstration data
        elif sample_type == "success_rollout":
            # Success rollout sample
            success_idx = idx - self.adjusted_demo_count  # Index within success rollouts section
            global_rollout_idx = success_idx % self._rollout_success_total_steps
            episode_idx, relative_step_idx = self._rollout_success_step_to_episode_map[global_rollout_idx]
            # Lazy load from HDF5 file
            episode_metadata = self.rollout_episode_metadata[episode_idx]
            # Calculate which frames we need to load
            future_frame_idx_temp = min(relative_step_idx + self.chunk_size, episode_metadata["num_steps"] - 1)
            frame_indices_needed = {relative_step_idx, future_frame_idx_temp}
            # Calculate which actions we need
            action_end_idx = min(relative_step_idx + self.chunk_size, episode_metadata["num_steps"])
            # Load only required data
            episode_data = self._load_rollout_episode_data(
                episode_metadata,
                frame_indices=frame_indices_needed,
                action_start_idx=relative_step_idx,
                action_end_idx=action_end_idx,
            )
        else:
            # Failure rollout sample
            failure_idx = (
                idx - self.adjusted_demo_count - self.adjusted_success_rollout_count
            )  # Index within failure rollouts section
            global_rollout_idx = failure_idx % self._rollout_failure_total_steps
            episode_idx, relative_step_idx = self._rollout_failure_step_to_episode_map[global_rollout_idx]
            # Lazy load from HDF5 file
            episode_metadata = self.rollout_episode_metadata[episode_idx]
            # Calculate which frames we need to load
            future_frame_idx_temp = min(relative_step_idx + self.chunk_size, episode_metadata["num_steps"] - 1)
            frame_indices_needed = {relative_step_idx, future_frame_idx_temp}
            # Calculate which actions we need
            action_end_idx = min(relative_step_idx + self.chunk_size, episode_metadata["num_steps"])
            # Load only required data
            episode_data = self._load_rollout_episode_data(
                episode_metadata,
                frame_indices=frame_indices_needed,
                action_start_idx=relative_step_idx,
                action_end_idx=action_end_idx,
            )

        # If returning value function samples, randomly choose whether this sample is for
        # world model training or value function training
        is_world_model_sample = False
        is_value_function_sample = False
        if sample_type != "demo":
            if self.return_value_function_returns:
                if random.random() < self.p_world_model:
                    is_world_model_sample = True
                    is_value_function_sample = False
                else:
                    is_world_model_sample = False
                    is_value_function_sample = True
            else:
                is_world_model_sample = True
                is_value_function_sample = False

        # Calculate future frame index if needed
        future_frame_idx = relative_step_idx + self.chunk_size
        max_possible_idx = episode_data["num_steps"] - 1
        if future_frame_idx > max_possible_idx:
            future_frame_idx = max_possible_idx

        # Handle JPEG decompression for rollout data if needed
        # Also handle dict vs array access for lazy-loaded data
        decompressed_left_primary_images = {}
        decompressed_right_primary_images = {}
        decompressed_wrist_images = {}
        frames_needed = {relative_step_idx, future_frame_idx}
        for frame_idx in frames_needed:
            # Check if images are stored as dict (lazy loading) or array
            if isinstance(episode_data["left_primary_images"], dict):
                # Lazy loaded data - images are already in dict form
                if episode_data.get("is_jpeg", False):
                    # JPEG data needs decompression
                    decompressed_left_primary_images[frame_idx] = decode_single_jpeg_frame(
                        episode_data["left_primary_images"][frame_idx]
                    )
                    decompressed_right_primary_images[frame_idx] = decode_single_jpeg_frame(
                        episode_data["right_primary_images"][frame_idx]
                    )
                    decompressed_wrist_images[frame_idx] = decode_single_jpeg_frame(
                        episode_data["wrist_images"][frame_idx]
                    )
                else:
                    # Already decompressed
                    decompressed_left_primary_images[frame_idx] = episode_data["left_primary_images"][frame_idx]
                    decompressed_right_primary_images[frame_idx] = episode_data["right_primary_images"][frame_idx]
                    decompressed_wrist_images[frame_idx] = episode_data["wrist_images"][frame_idx]
            else:
                # Non-lazy loaded data - images are in array form
                if sample_type != "demo" and episode_data.get("is_jpeg", False):
                    # Decompress JPEG frames for rollout data
                    decompressed_left_primary_images[frame_idx] = decode_single_jpeg_frame(
                        episode_data["left_primary_images"][frame_idx]
                    )
                    decompressed_right_primary_images[frame_idx] = decode_single_jpeg_frame(
                        episode_data["right_primary_images"][frame_idx]
                    )
                    decompressed_wrist_images[frame_idx] = decode_single_jpeg_frame(
                        episode_data["wrist_images"][frame_idx]
                    )
                else:
                    # Use images as-is (array indexing)
                    decompressed_left_primary_images[frame_idx] = episode_data["left_primary_images"][frame_idx]
                    decompressed_right_primary_images[frame_idx] = episode_data["right_primary_images"][frame_idx]
                    decompressed_wrist_images[frame_idx] = episode_data["wrist_images"][frame_idx]

        # Initialize list to store all images
        image_list = []
        current_sequence_idx = 0  # Used to track which sequence of images we are on

        # Get blank array for the first input frame (needed for the tokenizer)
        # Do not duplicate this image
        first_input_image = np.expand_dims(np.zeros_like(decompressed_left_primary_images[relative_step_idx]), axis=0)
        image_list.append(first_input_image)
        current_sequence_idx += 1

        # Add proprio state if using proprio
        if self.use_proprio:
            # Handle dict vs array access for lazy-loaded data
            if isinstance(episode_data["proprio"], dict):
                proprio = episode_data["proprio"][relative_step_idx]
            else:
                proprio = episode_data["proprio"][relative_step_idx]
            image = decompressed_left_primary_images[relative_step_idx]
            # Proprio values will be injected into latent diffusion sequence later
            # For now just add blank image
            blank_image = np.zeros_like(image)
            blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_image)
            current_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add wrist image if using wrist images
        if self.use_wrist_images:
            wrist_image = decompressed_wrist_images[relative_step_idx]
            # Duplicate wrist image
            wrist_image = duplicate_array(wrist_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(wrist_image)
            current_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add current third-person images: left and right
        if self.use_third_person_images:
            current_left_image = decompressed_left_primary_images[relative_step_idx]
            current_left_image = duplicate_array(current_left_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(current_left_image)
            current_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1
            current_right_image = decompressed_right_primary_images[relative_step_idx]
            current_right_image = duplicate_array(current_right_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(current_right_image)
            current_image2_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add blank image for action chunk
        blank_image = np.zeros_like(decompressed_left_primary_images[relative_step_idx])
        # Duplicate blank image
        blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
        image_list.append(blank_image)
        action_latent_idx = current_sequence_idx
        current_sequence_idx += 1

        # Add future proprio
        if self.use_proprio:
            # Handle dict vs array access for lazy-loaded data
            if isinstance(episode_data["proprio"], dict):
                future_proprio = episode_data["proprio"][future_frame_idx]
            else:
                future_proprio = episode_data["proprio"][future_frame_idx]
            # Not using proprio image; proprio values will be injected into latent diffusion sequence later
            # For now just add blank image
            blank_image = np.zeros_like(decompressed_left_primary_images[relative_step_idx])
            blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_image)
            future_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add future wrist image
        if self.use_wrist_images:
            future_wrist_image = decompressed_wrist_images[future_frame_idx]
            future_wrist_image = duplicate_array(future_wrist_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_wrist_image)
            future_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add future primary images: left and right
        if self.use_third_person_images:
            future_left_image = decompressed_left_primary_images[future_frame_idx]
            future_left_image = duplicate_array(future_left_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_left_image)
            future_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1
            future_right_image = decompressed_right_primary_images[future_frame_idx]
            future_right_image = duplicate_array(future_right_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_right_image)
            future_image2_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add blank value image
        if self.return_value_function_returns:
            value_image = np.zeros_like(decompressed_left_primary_images[relative_step_idx])
            value_image = duplicate_array(value_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(value_image)
            value_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Stack images and preprocess
        images = np.concatenate(image_list, axis=0)

        # Validate image data
        if np.isnan(images).any() or np.isinf(images).any():
            raise ValueError(f"Invalid image data detected (NaN or Inf) for sample {idx}")

        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )

        # Validate processed tensor
        if torch.isnan(images).any() or torch.isinf(images).any():
            raise ValueError(f"Invalid processed image tensor detected for sample {idx}")

        # Calculate how many actions we can get from the current index
        remaining_actions = episode_data["num_steps"] - relative_step_idx

        # Check if actions were loaded as a slice (lazy loading) or full array
        # For lazy loaded data, actions start from index 0 (already sliced from relative_step_idx)
        # For non-lazy loaded data, we need to index from relative_step_idx
        actions_already_sliced = (sample_type == "demo" and self.lazy_load_demos) or sample_type != "demo"

        if remaining_actions >= self.chunk_size:
            # If we have enough actions, get the full chunk
            if actions_already_sliced:
                # Actions array already starts from relative_step_idx, so use index 0
                action_chunk = episode_data["actions"][0 : self.chunk_size]
            else:
                # Actions array contains full episode, need to index from relative_step_idx
                action_chunk = episode_data["actions"][relative_step_idx : relative_step_idx + self.chunk_size]
        else:
            # If not enough actions remain, take what we can and repeat the last action
            if actions_already_sliced:
                available_actions = episode_data["actions"][:]
            else:
                available_actions = episode_data["actions"][relative_step_idx:]
            num_padding_needed = self.chunk_size - remaining_actions

            # Repeat the last action to fill the chunk
            if actions_already_sliced:
                padding = np.tile(episode_data["actions"][-1], (num_padding_needed, 1))
            else:
                padding = np.tile(episode_data["actions"][-1], (num_padding_needed, 1))

            # Concatenate available actions with padding
            action_chunk = np.concatenate([available_actions, padding], axis=0)

        # Get return for value function prediction
        if self.return_value_function_returns:
            return_timestep = future_frame_idx
            # For lazy-loaded demos, use metadata; for rollouts or non-lazy demos, check appropriately
            if sample_type == "demo" and self.lazy_load_demos:
                value_function_return = self.demo_episode_metadata[episode_idx]["returns"][return_timestep]
            elif episode_metadata is not None:
                value_function_return = episode_metadata["returns"][return_timestep]
            else:
                value_function_return = episode_data["returns"][return_timestep]
        else:
            value_function_return = float("-100")  # Just a placeholder

        sample_dict = {
            "video": images,
            "command": episode_data["command"],
            "actions": action_chunk,
            "t5_text_embeddings": torch.squeeze(self.t5_text_embeddings[episode_data["command"]]),
            "ai_caption": episode_data["command"],
            "t5_text_mask": torch.ones(512, dtype=torch.int64),  # Just copying what others have done in this codebase
            "fps": 16,  # Just set to some fixed value since we aren't generating videos anyway
            "padding_mask": torch.zeros(
                1, self.final_image_size, self.final_image_size
            ),  # Just copying what others have done in this codebase
            "image_size": self.final_image_size
            * torch.ones(
                4
            ),  # Just copying what others have done in this codebase; important because it shows up as model input
            "proprio": proprio if self.use_proprio else np.zeros_like(episode_data["proprio"][relative_step_idx]),
            "future_proprio": (
                future_proprio if self.use_proprio else np.zeros_like(episode_data["proprio"][future_frame_idx])
            ),
            "__key__": idx,  # Unique sample identifier (required for callbacks)
            "rollout_data_mask": rollout_data_mask,
            "rollout_data_success_mask": rollout_data_success_mask,
            "world_model_sample_mask": 1 if is_world_model_sample else 0,
            "value_function_sample_mask": 1 if is_value_function_sample else 0,
            "global_rollout_idx": global_rollout_idx,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": value_latent_idx if self.return_value_function_returns else -1,
            "current_proprio_latent_idx": current_proprio_latent_idx if self.use_proprio else -1,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx if self.use_wrist_images else -1,
            "current_image_latent_idx": current_image_latent_idx if self.use_third_person_images else -1,
            "current_image2_latent_idx": current_image2_latent_idx if self.use_third_person_images else -1,
            "future_proprio_latent_idx": future_proprio_latent_idx if self.use_proprio else -1,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx if self.use_wrist_images else -1,
            "future_image_latent_idx": future_image_latent_idx if self.use_third_person_images else -1,
            "future_image2_latent_idx": future_image2_latent_idx if self.use_third_person_images else -1,
            "value_function_return": value_function_return,
        }

        return sample_dict


if __name__ == "__main__":
    dataset = RoboCasaDataset(
        data_dir="/lustre/fsw/portfolios/dir/users/user/data/robocasa/robocasa_regen_v2_1199succDemos/",  # Successful demos
        t5_text_embeddings_path="/lustre/fsw/portfolios/dir/users/user/data/robocasa/robocasa_regen_v2_1199succDemos/t5_embeddings.pkl",
        chunk_size=32,
        use_image_aug=True,
        use_wrist_images=True,
        use_third_person_images=True,
        use_proprio=True,
        normalize_proprio=True,
        normalize_actions=True,
        num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
        use_stronger_image_aug=True,
        rollout_data_dir="/lustre/fsw/portfolios/dir/users/user/data/robocasa/robocasa_regen_rollout_data_v2_1291episodes/",  # All demo rollouts (successes + failures)
        demonstration_sampling_prob=0.5,
        success_rollout_sampling_prob=0.5,
        return_value_function_returns=True,
        gamma=0.99,
    )

    # Fetch a sample
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    idx = 0
    sample = dataset[idx]
    print(f"\nImages shape, dtype: {sample['video'].shape, sample['video'].dtype}")
    print(f"Actions shape, dtype: {sample['actions'].shape, sample['actions'].dtype}")
    print(f"Actions:\n{sample['actions']}")
    print(f"T5 text embeddings shape, dtype: {sample['t5_text_embeddings'].shape, sample['t5_text_embeddings'].dtype}")
    print(f"T5 text embeddings:\n{sample['t5_text_embeddings']}")

    # Fetch more samples and save sample images
    os.makedirs("./temp", exist_ok=True)
    for _ in range(50):
        global_step_index = random.randint(0, len(dataset) - 1)
        sample = dataset[global_step_index]
        images = sample["video"].permute(1, 2, 3, 0).numpy()
        for i in range(images.shape[0]):
            img_np = images[i]
            image_path = f"./temp/LAZYLOAD_global_step_index_{global_step_index}__task={sample['command'][:30]}__isRollout={sample['rollout_data_mask']}__globalRolloutIdx={sample['global_rollout_idx']}__isSuccess={sample['rollout_data_success_mask']}__value={sample['value_function_return']:.4f}__frameIdx={i}.png"
            Image.fromarray(img_np).save(image_path)
            print(f"Saved image at path: {image_path}")

    print("\n" + "=" * 80)
    print("Data samples loading test completed successfully!")
    print("=" * 80 + "\n")
