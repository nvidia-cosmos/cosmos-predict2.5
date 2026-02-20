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
Common helper functions for robot dataset classes.
"""

import json
import os

import numpy as np


def compute_monte_carlo_returns(num_steps: int, terminal_reward: float, gamma: float) -> np.ndarray:
    """
    Compute Monte Carlo returns with sparse reward at the end of an episode.
    Returns are rescaled from [0, terminal_reward] to [-1, 1].

    Args:
        num_steps (int): Number of steps in the episode
        terminal_reward (float): Reward at the terminal state (usually 0 or 1)
        gamma (float): Discount factor

    Returns:
        np.ndarray: Returns array of shape (num_steps,) rescaled to [-1, 1]
    """
    T = num_steps
    rewards = np.zeros(T, dtype=np.float32)
    rewards[-1] = terminal_reward

    # Compute Monte-Carlo returns
    returns = np.zeros_like(rewards)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G

    # Rescale returns from [0, terminal_reward] to [-1, 1]
    if terminal_reward > 0:
        returns = 2 * returns / terminal_reward - 1
    else:
        returns = 2 * returns - 1

    return returns


def get_action_chunk_with_padding(
    actions: np.ndarray,
    relative_step_idx: int,
    chunk_size: int,
    num_steps: int,
) -> np.ndarray:
    """
    Get action chunk starting from relative_step_idx with padding if needed.

    If there aren't enough remaining actions to fill the chunk, the last action
    is repeated to fill the chunk.

    Args:
        actions (np.ndarray): Full actions array of shape (num_steps, action_dim)
        relative_step_idx (int): Starting index for the action chunk
        chunk_size (int): Size of the action chunk
        num_steps (int): Total number of steps in the episode

    Returns:
        np.ndarray: Action chunk of shape (chunk_size, action_dim)
    """
    remaining_actions = num_steps - relative_step_idx

    if remaining_actions >= chunk_size:
        # If we have enough actions, get the full chunk
        action_chunk = actions[relative_step_idx : relative_step_idx + chunk_size]
    else:
        # If not enough actions remain, take what we can and repeat the last action
        available_actions = actions[relative_step_idx:]
        num_padding_needed = chunk_size - remaining_actions

        # Repeat the last action to fill the chunk
        padding = np.tile(actions[-1], (num_padding_needed, 1))

        # Concatenate available actions with padding
        action_chunk = np.concatenate([available_actions, padding], axis=0)

    return action_chunk


def build_rollout_step_index_mapping(rollout_data: dict, rollout_episode_metadata: dict):
    """
    Build mapping for rollout dataset with separate tracking for successful/failure episodes.

    This function creates three mappings:
    - _rollout_success_step_to_episode_map: Maps global success step index to (episode_idx, relative_idx)
    - _rollout_failure_step_to_episode_map: Maps global failure step index to (episode_idx, relative_idx)
    - Counters for total steps in each category

    Args:
        rollout_data (dict): In-memory rollout data (episode_idx -> episode_data)
        rollout_episode_metadata (dict): Lazy-loaded rollout metadata (episode_idx -> metadata)

    Returns:
        dict: Dictionary containing:
            - '_rollout_success_step_to_episode_map': dict mapping success step indices
            - '_rollout_failure_step_to_episode_map': dict mapping failure step indices
            - '_rollout_success_total_steps': int
            - '_rollout_failure_total_steps': int
            - '_rollout_total_steps': int
    """
    rollout_total_steps = 0

    # Separate mappings for successful and failure episodes
    rollout_success_step_to_episode_map = {}
    rollout_failure_step_to_episode_map = {}
    rollout_success_total_steps = 0
    rollout_failure_total_steps = 0

    # Handle in-memory rollout data (from demos treated as success rollouts or eager-loaded rollouts)
    for episode_idx, episode_data in rollout_data.items():
        num_steps = episode_data["num_steps"]
        is_success = episode_data["success"]

        for i in range(num_steps):
            # Separate mappings based on success/failure
            if is_success:
                rollout_success_step_to_episode_map[rollout_success_total_steps] = (episode_idx, i)
                rollout_success_total_steps += 1
            else:
                rollout_failure_step_to_episode_map[rollout_failure_total_steps] = (episode_idx, i)
                rollout_failure_total_steps += 1
            rollout_total_steps += 1

    # Handle lazy-loaded rollout data (from metadata)
    for episode_idx, episode_metadata in rollout_episode_metadata.items():
        num_steps = int(episode_metadata["num_steps"])
        is_success = bool(episode_metadata["success"])

        for i in range(num_steps):
            if is_success:
                rollout_success_step_to_episode_map[rollout_success_total_steps] = (episode_idx, i)
                rollout_success_total_steps += 1
            else:
                rollout_failure_step_to_episode_map[rollout_failure_total_steps] = (episode_idx, i)
                rollout_failure_total_steps += 1
            rollout_total_steps += 1

    return {
        "_rollout_success_step_to_episode_map": rollout_success_step_to_episode_map,
        "_rollout_failure_step_to_episode_map": rollout_failure_step_to_episode_map,
        "_rollout_success_total_steps": rollout_success_total_steps,
        "_rollout_failure_total_steps": rollout_failure_total_steps,
        "_rollout_total_steps": rollout_total_steps,
    }


def calculate_epoch_structure(
    num_steps: int,
    rollout_success_total_steps: int,
    rollout_failure_total_steps: int,
    demonstration_sampling_prob: float,
    success_rollout_sampling_prob: float,
):
    """
    Calculate epoch layout with proper scaling: demos, success rollouts, failure rollouts.

    This function computes adjusted counts for each data type to maintain the desired
    sampling probabilities during training.

    Args:
        num_steps (int): Number of demonstration steps
        rollout_success_total_steps (int): Number of success rollout steps
        rollout_failure_total_steps (int): Number of failure rollout steps
        demonstration_sampling_prob (float): Target probability of sampling demos vs rollouts
        success_rollout_sampling_prob (float): Target probability of sampling success vs failure rollouts

    Returns:
        dict: Dictionary containing:
            - 'adjusted_demo_count': int
            - 'adjusted_success_rollout_count': int
            - 'adjusted_failure_rollout_count': int
            - 'epoch_length': int
    """
    rollout_total_steps = rollout_success_total_steps + rollout_failure_total_steps
    has_rollout_data = rollout_total_steps > 0

    if has_rollout_data:
        # Step 1: Scale success vs failure rollouts to maintain success_rollout_sampling_prob
        if success_rollout_sampling_prob == 0 or rollout_success_total_steps == 0:
            adjusted_success_rollout_count = 0
            adjusted_failure_rollout_count = rollout_failure_total_steps
            success_rollout_sampling_prob = 0  # Override to 0
        elif success_rollout_sampling_prob == 1 or rollout_failure_total_steps == 0:
            adjusted_success_rollout_count = rollout_success_total_steps
            adjusted_failure_rollout_count = 0
            success_rollout_sampling_prob = 1  # Override to 1
        else:
            s = rollout_success_total_steps
            f = rollout_failure_total_steps
            s_new = int(f * success_rollout_sampling_prob / (1 - success_rollout_sampling_prob))
            f_new = int(s * (1 - success_rollout_sampling_prob) / success_rollout_sampling_prob)
            if s < s_new:
                adjusted_success_rollout_count = s_new
                adjusted_failure_rollout_count = f
            else:
                adjusted_success_rollout_count = s
                adjusted_failure_rollout_count = f_new

        # Step 2: Scale demos vs total rollouts to maintain demonstration_sampling_prob
        if demonstration_sampling_prob == 0 or num_steps == 0:
            adjusted_demo_count = 0
        elif demonstration_sampling_prob == 1:
            adjusted_demo_count = num_steps
            adjusted_success_rollout_count = 0
            adjusted_failure_rollout_count = 0
        else:
            d = num_steps
            r = adjusted_success_rollout_count + adjusted_failure_rollout_count
            d_new = int(r * demonstration_sampling_prob / (1 - demonstration_sampling_prob))
            r_new = int(d * (1 - demonstration_sampling_prob) / demonstration_sampling_prob)
            if d < d_new:
                adjusted_demo_count = d_new
            else:
                adjusted_demo_count = num_steps
                adjusted_success_rollout_count = int(r_new * success_rollout_sampling_prob)
                adjusted_failure_rollout_count = int(r_new * (1 - success_rollout_sampling_prob))
    else:
        adjusted_demo_count = num_steps
        adjusted_success_rollout_count = 0
        adjusted_failure_rollout_count = 0

    epoch_length = adjusted_demo_count + adjusted_success_rollout_count + adjusted_failure_rollout_count

    return {
        "adjusted_demo_count": adjusted_demo_count,
        "adjusted_success_rollout_count": adjusted_success_rollout_count,
        "adjusted_failure_rollout_count": adjusted_failure_rollout_count,
        "epoch_length": epoch_length,
    }


def load_or_compute_dataset_statistics(data_dir: str, data: dict, calculate_dataset_statistics_func):
    """
    Load dataset statistics from JSON file if it exists, otherwise compute and save them.

    Args:
        data_dir (str): Directory where statistics JSON file should be stored
        data (dict): Dataset dictionary (episode_idx -> episode_data)
        calculate_dataset_statistics_func: Function to compute statistics from data

    Returns:
        dict: Dataset statistics with numpy array values
    """
    dataset_stats_path = os.path.join(data_dir, "dataset_statistics.json")
    if os.path.exists(dataset_stats_path):
        # Load dataset statistics
        with open(dataset_stats_path, "r") as f:
            json_stats = json.load(f)
        print(f"Loaded dataset statistics from: {dataset_stats_path}")
    else:
        # Compute dataset statistics
        dataset_stats = calculate_dataset_statistics_func(data)
        # Convert numpy arrays to lists for JSON serialization
        json_stats = {}
        for stat_name, stat_value in dataset_stats.items():
            json_stats[stat_name] = stat_value.tolist()
        # Write to JSON file
        with open(dataset_stats_path, "w") as f:
            json.dump(json_stats, f, indent=4)
        print(f"Dataset statistics saved to: {dataset_stats_path}")

    # Convert JSON lists back to numpy arrays
    dataset_stats = {}
    for stat_name, stat_value in json_stats.items():
        dataset_stats[stat_name] = np.array(stat_value)

    return dataset_stats


def load_or_compute_post_normalization_statistics(data_dir: str, data: dict, calculate_dataset_statistics_func):
    """
    Load post-normalization dataset statistics from JSON file if it exists, otherwise compute and save them.

    Args:
        data_dir (str): Directory where statistics JSON file should be stored
        data (dict): Dataset dictionary (episode_idx -> episode_data) with normalized data
        calculate_dataset_statistics_func: Function to compute statistics from data

    Returns:
        dict: Post-normalization dataset statistics with numpy array values
    """
    dataset_stats_post_norm_path = os.path.join(data_dir, "dataset_statistics_post_norm.json")
    if os.path.exists(dataset_stats_post_norm_path):
        # Load dataset statistics
        with open(dataset_stats_post_norm_path, "r") as f:
            json_stats = json.load(f)
        print(f"Loaded post-normalization dataset statistics from: {dataset_stats_post_norm_path}")
    else:
        # Compute dataset statistics
        dataset_stats_post_norm = calculate_dataset_statistics_func(data)
        # Convert numpy arrays to lists for JSON serialization
        json_stats = {}
        for stat_name, stat_value in dataset_stats_post_norm.items():
            json_stats[stat_name] = stat_value.tolist()
        # Write to JSON file
        with open(dataset_stats_post_norm_path, "w") as f:
            json.dump(json_stats, f, indent=4)
        print(f"Post-normalization dataset statistics saved to: {dataset_stats_post_norm_path}")

    # Convert JSON lists back to numpy arrays
    dataset_stats_post_norm = {}
    for stat_name, stat_value in json_stats.items():
        dataset_stats_post_norm[stat_name] = np.array(stat_value)

    return dataset_stats_post_norm


def build_demo_step_index_mapping(data: dict):
    """
    Build a mapping from global step index to (episode index, relative index within episode).

    Args:
        data (dict): Dataset dictionary (episode_idx -> episode_data)

    Returns:
        dict: Dictionary containing:
            - '_step_to_episode_map': dict mapping global step index to (episode_idx, relative_idx)
            - '_total_steps': int
    """
    step_to_episode_map = {}
    total_steps = 0
    for episode_idx, episode_data in data.items():
        num_steps = episode_data["num_steps"]
        for i in range(num_steps):
            step_to_episode_map[total_steps] = (episode_idx, i)
            total_steps += 1

    return {
        "_step_to_episode_map": step_to_episode_map,
        "_total_steps": total_steps,
    }


def determine_sample_type(idx: int, adjusted_demo_count: int, adjusted_success_rollout_count: int) -> str:
    """
    Determine which dataset to sample from based on index ranges.

    Layout of indices: [demos] [success rollouts] [failure rollouts]

    Args:
        idx (int): Sample index
        adjusted_demo_count (int): Number of demo samples in epoch
        adjusted_success_rollout_count (int): Number of success rollout samples in epoch

    Returns:
        str: One of "demo", "success_rollout", or "failure_rollout"
    """
    if idx < adjusted_demo_count:
        return "demo"
    elif idx < adjusted_demo_count + adjusted_success_rollout_count:
        return "success_rollout"
    else:
        return "failure_rollout"
