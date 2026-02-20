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
LIBERO simulation benchmark task suites dataloader.

Run this command to print a few samples from the LIBERO dataset:
    python -m cosmos_predict2._src.predict2.cosmos_policy.datasets.libero_dataset
"""

import os
import pickle
import random
from collections import defaultdict

import h5py
import imageio
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
    get_action_chunk_with_padding,
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


class LIBERODataset(Dataset):
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
        treat_success_rollouts_as_demos: bool = False,
        return_value_function_returns: bool = True,
        gamma: float = 0.99,
        lazy_load_demos: bool = False,
        skip_computing_dataset_statistics: bool = False,
    ):
        """
        Initialize LIBERO dataset for training.

        Args:
            data_dir (str): Path to directory containing LIBERO task suite HDF5 files
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
            treat_success_rollouts_as_demos (bool): If True, copy successful rollout episodes into demonstration dataset (self.data)
            return_value_function_returns (bool): If True, returns value function returns for rollout episodes
            gamma (float): Discount factor for value function returns
            lazy_load_demos (bool): If True, only load demo metadata at initialization and load full data on-demand during __getitem__
            skip_computing_dataset_statistics (bool): If True, skip computing dataset statistics (requires pre-computed stats file)
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
        #      images=primary images,
        #      wrist_images=wrist images,
        #      proprio=proprio states,
        #      actions=actions,
        #      command=language instruction,
        #      num_steps=number of steps in episode,
        #      suite=task suite name,
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

        # Global step mapping from task suite name to list[global_step_idx]
        # Populated later in `_build_step_index_mapping()`
        self._suite_to_step_indices = {}
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
                        is_jpeg = "agentview_rgb_jpeg" in obs_group

                        # Compute language instruction
                        raw_file_string = os.path.basename(file).split("/")[-1]
                        words = raw_file_string[:-10].split("_")
                        command = ""
                        for w in words:
                            if "SCENE" in w:
                                command = ""
                                continue
                            command = command + w + " "
                        command = command[:-1]
                        self.unique_commands.add(command)

                        # Get number of steps
                        if is_jpeg:
                            num_steps = len(obs_group["agentview_rgb_jpeg"])
                        elif "agentview_rgb" in obs_group:
                            num_steps = len(obs_group["agentview_rgb"])
                        else:
                            raise KeyError("Neither 'agentview_rgb' nor 'agentview_rgb_jpeg' found in HDF5 file.")

                        # Add value function returns if applicable
                        if self.return_value_function_returns:
                            returns = compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=self.gamma)
                        else:
                            returns = None

                        suite = os.path.relpath(file, self.data_dir).split(os.sep)[
                            0
                        ]  # Task suite folder name (e.g. libero_spatial_no_noops_rerendered)

                        if self.lazy_load_demos:
                            # Store metadata for lazy loading (no images/actions/proprio loaded)
                            self.demo_episode_metadata[self.num_episodes] = dict(
                                file_path=file,
                                demo_key=demo_key,
                                command=command,
                                num_steps=num_steps,
                                is_jpeg=is_jpeg,
                                suite=suite,
                                returns=returns.copy() if returns is not None else None,
                            )
                        else:
                            # Load full data into RAM (original behavior)
                            # Agent-view (third-person) images
                            if "agentview_rgb" in obs_group:
                                images = obs_group["agentview_rgb"][:]  # (T, H, W, 3) uint8
                            elif is_jpeg:
                                images = decode_jpeg_bytes_dataset(obs_group["agentview_rgb_jpeg"])
                            else:
                                raise KeyError("Neither 'agentview_rgb' nor 'agentview_rgb_jpeg' found in HDF5 file.")
                            # Wrist-mounted camera images
                            if "eye_in_hand_rgb" in obs_group:
                                wrist_images = obs_group["eye_in_hand_rgb"][:]
                            elif "eye_in_hand_rgb_jpeg" in obs_group:
                                wrist_images = decode_jpeg_bytes_dataset(obs_group["eye_in_hand_rgb_jpeg"])
                            else:
                                raise KeyError(
                                    "Neither 'eye_in_hand_rgb' nor 'eye_in_hand_rgb_jpeg' found in HDF5 file."
                                )
                            # Actions
                            actions = f[f"data/{demo_key}/actions"][:].astype(
                                np.float32
                            )  # (episode_len, action_dim=7), float32
                            # Proprio states
                            proprio = f[f"data/{demo_key}/robot_states"][:].astype(
                                np.float32
                            )  # (episode_len, proprio_dim=9), float32
                            # Add entry to dataset dict
                            self.data[self.num_episodes] = dict(
                                images=images,
                                wrist_images=wrist_images,
                                proprio=proprio,
                                actions=actions,
                                command=command,
                                num_steps=num_steps,
                                suite=suite,
                                returns=returns.copy() if self.return_value_function_returns else None,
                            )
                        # Update number of episodes
                        self.num_episodes += 1
                        # Update number of steps
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
                    "Dataset statistics file for this dataset does not yet exist. Please rerun with "
                    "LIBERODataset(lazy_load_demos=False) once so that the dataset statistics are computed "
                    "and saved. Then you can rerun with LIBERODataset(lazy_load_demos=True)."
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
            if not self.lazy_load_demos:
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
                        raise KeyError(f"No primary/wrist images found in rollout file: {file}")

                    # Get task description
                    command = f.attrs.get("task_description", "")
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

                    # Determine suite name from rollout directory (for balanced sampling bookkeeping)
                    if "suite=libero_spatial" in ep_meta["file_path"]:
                        suite_name = "libero_spatial"
                    elif "suite=libero_object" in ep_meta["file_path"]:
                        suite_name = "libero_object"
                    elif "suite=libero_goal" in ep_meta["file_path"]:
                        suite_name = "libero_goal"
                    elif "suite=libero_10" in ep_meta["file_path"]:
                        suite_name = "libero_10"
                    else:
                        raise ValueError(
                            f"Could not determine suite name from rollout file path: {ep_meta['file_path']}"
                        )

                    if self.lazy_load_demos:
                        # Store as metadata for lazy loading (mark as rollout-sourced)
                        self.demo_episode_metadata[self.num_episodes] = dict(
                            file_path=ep_meta["file_path"],
                            command=ep_meta.get("command"),
                            num_steps=ep_meta.get("num_steps"),
                            is_jpeg=ep_meta.get("is_jpeg", False),
                            suite=suite_name,
                            returns=ep_meta.get("returns", np.array([])).copy()
                            if ep_meta.get("returns") is not None
                            else None,
                            is_from_rollout=True,  # Flag so _load_demo_episode_data uses rollout format
                        )
                    else:
                        # Lazy load rollout episode data
                        episode_data = self._load_rollout_episode_data(ep_meta)

                        # Decode to raw uint8 arrays for demos if source was JPEG bytes
                        if episode_data.get("is_jpeg", False):
                            images = np.stack(
                                [decode_single_jpeg_frame(b) for b in episode_data["images"]], axis=0
                            ).astype(np.uint8)
                            wrist_images = np.stack(
                                [decode_single_jpeg_frame(b) for b in episode_data["wrist_images"]], axis=0
                            ).astype(np.uint8)
                        else:
                            images = episode_data["images"]
                            wrist_images = episode_data["wrist_images"]

                        actions = episode_data["actions"]
                        proprio = episode_data["proprio"]

                        # Use precomputed returns
                        returns = ep_meta.get("returns")
                        if returns is not None:
                            returns = returns.copy()

                        # Insert into demonstration dataset
                        self.data[self.num_episodes] = dict(
                            images=images,
                            wrist_images=wrist_images,
                            proprio=proprio,
                            actions=actions,
                            command=ep_meta.get("command"),
                            num_steps=ep_meta.get("num_steps"),
                            suite=suite_name,
                            returns=returns,
                        )

                    self.unique_commands.add(ep_meta.get("command"))
                    self.num_episodes += 1
                    self.num_steps += ep_meta.get("num_steps")

                # Rebuild step index mapping to include newly added demos
                self._build_step_index_mapping()

            # Build mapping from global rollout step → (episode, rel_idx)
            self._build_rollout_step_index_mapping()

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

        # Reset suite mapping if it already exists
        self._suite_to_step_indices = defaultdict(list)

        # Use demo_episode_metadata when lazy loading, self.data otherwise
        if self.lazy_load_demos and len(self.demo_episode_metadata) > 0:
            source = self.demo_episode_metadata
        else:
            source = self.data

        for episode_idx, episode_data in source.items():
            num_steps = episode_data["num_steps"]
            for i in range(num_steps):
                self._step_to_episode_map[self._total_steps] = (episode_idx, i)
                self._suite_to_step_indices[episode_data["suite"]].append(self._total_steps)
                self._total_steps += 1

        # Additional bookkeeping for balanced sampling
        self._suites = list(self._suite_to_step_indices.keys())
        if len(self._suites) > 0:
            self._max_suite_len = max(len(v) for v in self._suite_to_step_indices.values())

    def _build_rollout_step_index_mapping(self):
        """Build mapping for rollout dataset with separate tracking for successful/failure episodes."""
        result = build_rollout_step_index_mapping({}, self.rollout_episode_metadata)
        self._rollout_success_step_to_episode_map = result["_rollout_success_step_to_episode_map"]
        self._rollout_failure_step_to_episode_map = result["_rollout_failure_step_to_episode_map"]
        self._rollout_success_total_steps = result["_rollout_success_total_steps"]
        self._rollout_failure_total_steps = result["_rollout_failure_total_steps"]
        self._rollout_total_steps = result["_rollout_total_steps"]

    def _load_demo_episode_data(
        self,
        episode_metadata: dict,
        frame_indices: set[int] | None = None,
    ) -> dict:
        """
        Load demo episode data from HDF5 file using metadata.
        Optimized to only load required frames when frame_indices is provided.

        Handles both original demo HDF5 format and rollout HDF5 format
        (when successful rollouts are treated as demos via treat_success_rollouts_as_demos).

        Args:
            episode_metadata (dict): Episode metadata containing file_path, demo_key, etc.
            frame_indices (set[int] | None): Set of frame indices to load. If None, loads all frames.

        Returns:
            dict: Episode data dictionary with loaded arrays.
                  When frame_indices is provided, images and wrist_images are dicts
                  mapping frame index to data; otherwise they are full arrays.
        """
        # If this metadata came from a rollout file, delegate to the rollout loader
        if episode_metadata.get("is_from_rollout", False):
            return self._load_rollout_episode_data(episode_metadata, frame_indices=frame_indices)

        file_path = episode_metadata["file_path"]
        demo_key = episode_metadata["demo_key"]

        with h5py.File(file_path, "r") as f:
            obs_group = f[f"data/{demo_key}/obs"]
            load_all = frame_indices is None

            # Load images based on storage format
            if episode_metadata["is_jpeg"]:
                if load_all:
                    images = decode_jpeg_bytes_dataset(obs_group["agentview_rgb_jpeg"])
                    wrist_images = decode_jpeg_bytes_dataset(obs_group["eye_in_hand_rgb_jpeg"])
                else:
                    # Store raw JPEG bytes; decoding happens in __getitem__
                    frame_indices_list = sorted(list(frame_indices))
                    images = {}
                    wrist_images = {}
                    for idx in frame_indices_list:
                        images[idx] = obs_group["agentview_rgb_jpeg"][idx]
                        wrist_images[idx] = obs_group["eye_in_hand_rgb_jpeg"][idx]
            else:
                if load_all:
                    images = obs_group["agentview_rgb"][:]
                    wrist_images = obs_group["eye_in_hand_rgb"][:]
                else:
                    frame_indices_list = sorted(list(frame_indices))
                    images = {}
                    wrist_images = {}
                    for idx in frame_indices_list:
                        images[idx] = obs_group["agentview_rgb"][idx]
                        wrist_images[idx] = obs_group["eye_in_hand_rgb"][idx]

            # Always load all actions (small arrays)
            actions = f[f"data/{demo_key}/actions"][:].astype(np.float32)

            # Load proprio - only load required timesteps
            if frame_indices is not None and not load_all:
                frame_indices_list = sorted(list(frame_indices))
                proprio = {}
                for idx in frame_indices_list:
                    proprio[idx] = f[f"data/{demo_key}/robot_states"][idx].astype(np.float32)
            else:
                proprio = f[f"data/{demo_key}/robot_states"][:].astype(np.float32)

            # Apply normalization if needed
            if self.normalize_actions:
                actions = rescale_episode_data({"actions": actions}, self.dataset_stats, "actions")
            if self.normalize_proprio:
                if isinstance(proprio, dict):
                    for idx in proprio:
                        proprio_array = proprio[idx].reshape(1, -1)
                        proprio[idx] = rescale_episode_data(
                            {"proprio": proprio_array}, self.dataset_stats, "proprio"
                        ).flatten()
                else:
                    proprio = rescale_episode_data({"proprio": proprio}, self.dataset_stats, "proprio")

            episode_data = dict(
                images=images,
                wrist_images=wrist_images,
                proprio=proprio,
                actions=actions,
                command=episode_metadata["command"],
                num_steps=episode_metadata["num_steps"],
                suite=episode_metadata["suite"],
                is_jpeg=episode_metadata["is_jpeg"],
                returns=episode_metadata.get("returns"),
            )

            return episode_data

    def _load_rollout_episode_data(self, episode_metadata, frame_indices: set[int] | None = None):
        """
        Load rollout episode data from HDF5 file using metadata.
        Optimized to only load required frames when frame_indices is provided.

        Args:
            episode_metadata (dict): Episode metadata containing file_path, success, etc.
            frame_indices (set[int] | None): Set of frame indices to load. If None, loads all frames.

        Returns:
            dict: Episode data dictionary with loaded arrays.
                  When frame_indices is provided, images and wrist_images are dicts
                  mapping frame index to data; otherwise they are full arrays.
        """
        file_path = episode_metadata["file_path"]

        with h5py.File(file_path, "r") as f:
            load_all = frame_indices is None

            # Load images based on storage format
            if episode_metadata["is_jpeg"]:
                if load_all:
                    images = f["primary_images_jpeg"][:]
                    wrist_images = f["wrist_images_jpeg"][:]
                else:
                    # Load only specific frames
                    frame_indices_list = sorted(list(frame_indices))
                    images = {}
                    wrist_images = {}
                    for idx in frame_indices_list:
                        images[idx] = f["primary_images_jpeg"][idx]
                        wrist_images[idx] = f["wrist_images_jpeg"][idx]
            else:
                if load_all:
                    images = f["primary_images"][:]
                    wrist_images = f["wrist_images"][:]
                else:
                    # Load only specific frames
                    frame_indices_list = sorted(list(frame_indices))
                    images = {}
                    wrist_images = {}
                    for idx in frame_indices_list:
                        images[idx] = f["primary_images"][idx]
                        wrist_images[idx] = f["wrist_images"][idx]

            # Load actions and proprio (small arrays, always load fully)
            actions = f["actions"][:].astype(np.float32)
            proprio = f["proprio"][:].astype(np.float32)

            # Apply normalization if needed
            if self.normalize_actions:
                actions = rescale_episode_data({"actions": actions}, self.dataset_stats, "actions")
            if self.normalize_proprio:
                proprio = rescale_episode_data(
                    {"proprio": proprio},
                    self.dataset_stats,
                    "proprio",
                )

            # Create episode data dictionary
            episode_data = dict(
                images=images,
                wrist_images=wrist_images,
                proprio=proprio,
                actions=actions,
                command=episode_metadata["command"],
                num_steps=episode_metadata["num_steps"],
                success=episode_metadata.get("success", False),
                is_jpeg=episode_metadata["is_jpeg"],
                suite=episode_metadata.get("suite"),
                returns=episode_metadata.get("returns"),
            )

            return episode_data

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        # Return pre-calculated epoch length (which already accounts for suite balancing if enabled)
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
                # Load only required image frames; actions/proprio loaded fully (they are small)
                episode_data = self._load_demo_episode_data(
                    episode_metadata,
                    frame_indices=frame_indices_needed,
                )
            else:
                episode_data = self.data[episode_idx]

            global_rollout_idx = -1  # Not applicable for demonstration data
        elif sample_type == "success_rollout":
            # Success rollout sample
            success_idx = idx - self.adjusted_demo_count  # Index within success rollouts section
            global_rollout_idx = success_idx % self._rollout_success_total_steps
            episode_idx, relative_step_idx = self._rollout_success_step_to_episode_map[global_rollout_idx]
            # Lazy load from HDF5 file — only load the 2 frames we actually need
            episode_metadata = self.rollout_episode_metadata[episode_idx]
            future_frame_idx_temp = min(relative_step_idx + self.chunk_size, episode_metadata["num_steps"] - 1)
            frame_indices_needed = {relative_step_idx, future_frame_idx_temp}
            episode_data = self._load_rollout_episode_data(episode_metadata, frame_indices=frame_indices_needed)
        else:
            # Failure rollout sample
            failure_idx = (
                idx - self.adjusted_demo_count - self.adjusted_success_rollout_count
            )  # Index within failure rollouts section
            global_rollout_idx = failure_idx % self._rollout_failure_total_steps
            episode_idx, relative_step_idx = self._rollout_failure_step_to_episode_map[global_rollout_idx]
            # Lazy load from HDF5 file — only load the 2 frames we actually need
            episode_metadata = self.rollout_episode_metadata[episode_idx]
            future_frame_idx_temp = min(relative_step_idx + self.chunk_size, episode_metadata["num_steps"] - 1)
            frame_indices_needed = {relative_step_idx, future_frame_idx_temp}
            episode_data = self._load_rollout_episode_data(episode_metadata, frame_indices=frame_indices_needed)

        # If returning value function samples, randomly choose whether this sample is for
        # world model training or value function training
        is_world_model_sample = False
        is_value_function_sample = False
        if sample_type != "demo":
            if self.return_value_function_returns:
                p_world_model = 0.5
                if random.random() < p_world_model:
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
        decompressed_images = {}
        decompressed_wrist_images = {}
        frames_needed = {relative_step_idx, future_frame_idx}
        for frame_idx in frames_needed:
            # Check if images are stored as dict (lazy loading) or array
            if isinstance(episode_data["images"], dict):
                # Lazy loaded data - images are already in dict form
                if episode_data.get("is_jpeg", False):
                    decompressed_images[frame_idx] = decode_single_jpeg_frame(episode_data["images"][frame_idx])
                    decompressed_wrist_images[frame_idx] = decode_single_jpeg_frame(
                        episode_data["wrist_images"][frame_idx]
                    )
                else:
                    decompressed_images[frame_idx] = episode_data["images"][frame_idx]
                    decompressed_wrist_images[frame_idx] = episode_data["wrist_images"][frame_idx]
            elif sample_type != "demo" and episode_data.get("is_jpeg", False):
                # Rollout JPEG data
                decompressed_images[frame_idx] = decode_single_jpeg_frame(episode_data["images"][frame_idx])
                decompressed_wrist_images[frame_idx] = decode_single_jpeg_frame(episode_data["wrist_images"][frame_idx])
            else:
                # Eager-loaded array data
                decompressed_images[frame_idx] = episode_data["images"][frame_idx]
                decompressed_wrist_images[frame_idx] = episode_data["wrist_images"][frame_idx]

        # Initialize list to store all images
        image_list = []
        current_sequence_idx = 0  # Used to track which sequence of images we are on

        # Get blank array for the first input frame (needed for the tokenizer)
        # Do not duplicate this image
        first_input_image = np.expand_dims(np.zeros_like(decompressed_images[relative_step_idx]), axis=0)
        image_list.append(first_input_image)
        current_sequence_idx += 1

        # Add proprio state if using proprio
        if self.use_proprio:
            proprio = episode_data["proprio"][relative_step_idx]
            image = decompressed_images[relative_step_idx]
            # Proprio values will be injected into latent diffusion sequence later
            # For now just add blank image
            blank_image = np.zeros_like(decompressed_images[relative_step_idx])
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

        # Add current third-person image
        if self.use_third_person_images:
            current_image = decompressed_images[relative_step_idx]
            current_image = duplicate_array(current_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(current_image)
            current_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add blank image for action chunk
        blank_image = np.zeros_like(decompressed_images[relative_step_idx])
        # Duplicate blank image
        blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
        image_list.append(blank_image)
        action_latent_idx = current_sequence_idx
        current_sequence_idx += 1

        # Add future proprio
        if self.use_proprio:
            future_proprio = episode_data["proprio"][future_frame_idx]
            # Not using proprio image; proprio values will be injected into latent diffusion sequence later
            # For now just add blank image
            blank_image = np.zeros_like(decompressed_images[relative_step_idx])
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

        # Add future primary image
        if self.use_third_person_images:
            future_image = decompressed_images[future_frame_idx]
            future_image = duplicate_array(future_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_image)
            future_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add blank value image
        if self.return_value_function_returns:
            value_image = np.zeros_like(decompressed_images[relative_step_idx])
            value_image = duplicate_array(value_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(value_image)
            value_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Stack images and preprocess
        images = np.concatenate(image_list, axis=0)
        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )

        # Calculate how many actions we can get from the current index
        action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        # Return the next action chunk as well
        # Calculate how many actions we can get from the current index
        next_relative_step_idx = min(relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1)
        next_action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=next_relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        # Get return for value function prediction
        if self.return_value_function_returns:
            return_timestep = future_frame_idx
            if episode_metadata is not None:
                value_function_return = episode_metadata["returns"][return_timestep]
            else:
                value_function_return = episode_data["returns"][return_timestep]
        else:
            value_function_return = float("-100")  # Just a placeholder

        # Calculate next future frame index if needed
        next_future_frame_idx = next_relative_step_idx + self.chunk_size
        max_possible_idx = episode_data["num_steps"] - 1
        if next_future_frame_idx > max_possible_idx:
            next_future_frame_idx = max_possible_idx

        # Return the next value function return as well
        if self.return_value_function_returns:
            return_timestep = next_future_frame_idx
            if episode_metadata is not None:
                next_value_function_return = episode_metadata["returns"][return_timestep]
            else:
                next_value_function_return = episode_data["returns"][return_timestep]
        else:
            next_value_function_return = float("-100")  # Just a placeholder

        sample_dict = {
            "video": images,
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
            "future_proprio_latent_idx": future_proprio_latent_idx if self.use_proprio else -1,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx if self.use_wrist_images else -1,
            "future_image_latent_idx": future_image_latent_idx if self.use_third_person_images else -1,
            "value_function_return": value_function_return,
            "next_action_chunk": next_action_chunk,
            "next_value_function_return": next_value_function_return,
        }

        return sample_dict


def create_augmentation_visualization(
    data_dir: str,
    t5_text_embeddings_path: str,
    fixed_idx: int = 100,
    num_augmentations: int = 50,
    output_dir: str = "./temp",
):
    """
    Create MP4 videos visualizing the distribution of augmentations for a fixed data point.

    Args:
        data_dir (str): Path to the dataset directory
        t5_text_embeddings_path (str): Path to T5 embeddings file
        fixed_idx (int): Index of the data point to apply augmentations to
        num_augmentations (int): Number of different augmentations to sample
        output_dir (str): Directory to save the visualization videos
    """
    print(f"\nCreating augmentation visualization with {num_augmentations} samples...")

    # Create a dataset instance with augmentations enabled
    aug_dataset = LIBERODataset(
        data_dir=data_dir,
        chunk_size=16,
        t5_text_embeddings_path=t5_text_embeddings_path,
        normalize_images=False,
        normalize_actions=True,
        use_image_aug=True,  # Enable augmentations
        use_wrist_images=True,
        use_proprio=True,
        normalize_proprio=True,
        num_duplicates_per_image=1,
        use_stronger_image_aug=True,
    )

    # Collect different augmentations of the same data point
    augmented_samples = []

    print(f"Generating {num_augmentations} augmentations for data point {fixed_idx}...")
    for aug_idx in tqdm(range(num_augmentations)):
        sample = aug_dataset[fixed_idx]
        augmented_samples.append(sample)

    # Extract images from all augmented samples and organize them
    # Each sample has shape (C, T, H, W) where T is number of frames
    all_augmented_videos = []
    for sample in augmented_samples:
        video = sample["video"].permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
        all_augmented_videos.append(video)

    # Stack all augmented videos: (num_augmentations, T, H, W, C)
    all_augmented_videos = np.stack(all_augmented_videos, axis=0)

    # Get dimensions
    num_augs, num_frames, height, width, channels = all_augmented_videos.shape
    print(f"Augmented video array shape: {all_augmented_videos.shape}")

    # Create video frames for each frame type
    frame_names = [
        "blank_input",
        "proprio",
        "wrist",
        "current_view",
        "future_proprio",
        "future_wrist",
        "future_view",
        "blank_action",
    ]
    if not aug_dataset.use_proprio:
        frame_names.remove("proprio")
    if not aug_dataset.use_wrist_images:
        frame_names.remove("wrist")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create MP4 videos for each frame type
    for frame_idx in range(min(num_frames, len(frame_names))):
        frame_name = frame_names[frame_idx] if frame_idx < len(frame_names) else f"frame_{frame_idx}"

        # Skip blank frames for visualization
        if "blank" in frame_name:
            continue

        print(f"Creating video for {frame_name}...")

        # Extract frames for this frame type across all augmentations
        frames_for_video = []
        for aug_idx in range(num_augs):
            frame = all_augmented_videos[aug_idx, frame_idx]  # (H, W, C)
            frames_for_video.append(frame)

        # Save as MP4 video
        video_path = os.path.join(output_dir, f"augmentation_visualization_{frame_name}.mp4")

        # Convert to uint8 if needed
        frames_array = np.stack(frames_for_video, axis=0)  # (num_augs, H, W, C)
        if frames_array.dtype != np.uint8:
            frames_array = frames_array.astype(np.uint8)

        # Save video with slower frame rate to better see the augmentations
        imageio.mimsave(video_path, frames_array, fps=5, macro_block_size=None)
        print(f"Saved augmentation visualization video: {video_path}")

    # Also create a combined video showing all frame types side by side
    print("Creating combined video with all frame types...")

    # Only use non-blank frames
    valid_frame_indices = []
    valid_frame_names = []
    for frame_idx in range(min(num_frames, len(frame_names))):
        frame_name = frame_names[frame_idx] if frame_idx < len(frame_names) else f"frame_{frame_idx}"
        if "blank" not in frame_name:
            valid_frame_indices.append(frame_idx)
            valid_frame_names.append(frame_name)

    if len(valid_frame_indices) > 0:
        combined_frames = []
        for aug_idx in range(num_augs):
            # Extract valid frames for this augmentation
            frames_to_combine = []
            for frame_idx in valid_frame_indices:
                frame = all_augmented_videos[aug_idx, frame_idx]  # (H, W, C)
                frames_to_combine.append(frame)

            # Concatenate frames horizontally
            combined_frame = np.concatenate(frames_to_combine, axis=1)  # (H, W*num_frames, C)
            combined_frames.append(combined_frame)

        # Save combined video
        combined_frames_array = np.stack(combined_frames, axis=0)  # (num_augs, H, W*num_frames, C)
        if combined_frames_array.dtype != np.uint8:
            combined_frames_array = combined_frames_array.astype(np.uint8)

        combined_video_path = os.path.join(output_dir, "augmentation_visualization_combined.mp4")
        imageio.mimsave(combined_video_path, combined_frames_array, fps=5, macro_block_size=None)
        print(f"Saved combined augmentation visualization video: {combined_video_path}")
        print(f"Combined video shows frames in order: {' | '.join(valid_frame_names)}")

    print("Augmentation visualization complete!")


if __name__ == "__main__":
    dataset = LIBERODataset(
        data_dir="/lustre/fsw/portfolios/dir/users/user/data/libero_regen",  # Successful demos
        t5_text_embeddings_path="/lustre/fsw/portfolios/dir/users/user/data/libero_regen/t5_embeddings.pkl",
        chunk_size=16,
        use_image_aug=True,
        use_wrist_images=True,
        use_proprio=True,
        normalize_proprio=True,
        normalize_actions=True,
        num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
        use_stronger_image_aug=True,
        rollout_data_dir="/lustre/fsw/portfolios/dir/users/user/data/libero_regen_rollout_data/",  # All demo rollouts (successes + failures)
        demonstration_sampling_prob=0.5,
        success_rollout_sampling_prob=0.5,
        return_value_function_returns=True,
        gamma=0.99,
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
            img_np = images[i]
            image_path = f"./temp/video__global_step_index_{global_step_index}__is_rollout={sample['rollout_data_mask']}__global_rollout_idx={sample['global_rollout_idx']}__is_success={sample['rollout_data_success_mask']}__value_function_return={sample['value_function_return']:.4f}__frame_idx={i}.png"
            Image.fromarray(img_np).save(image_path)
            print(f"Saved image at path: {image_path}")
