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
run_aloha_eval.py

Evaluates a model in a real-world ALOHA environment.

Setup:
    pip install -r experiments/robot/aloha/requirements_aloha.txt

Usage examples:

    # Base Cosmos Policy
    uv run -m experiments.robot.aloha.run_aloha_eval \
        --policy_name cosmos--a_s_v--mixture_20250905_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80_185_demos--chkpt50000--10stepsA \
        --input_image_size 224 \
        --trained_with_image_aug True \
        --num_open_loop_steps 50 \
        --policy_server_ip 10.12.186.130 \
        --future_img True \
        --max_steps 1100 \
        --run_id_note "" \
        --local_log_dir experiments/robot/aloha/logs/ \
        --return_all_query_results True \
        --data_collection True

    # Cosmos Policy + model-based planning w/ V(s') value function (main planning config)
    # Best-of-8, 3 future state ensemble, 5 value ensemble, majority mean, seed switching, 10/5/5 steps a/s'/v
    # Policy: (a, s', V(s, a, s')) policy - 50K checkpoint
    # WMVF model: V(s') - 648 rollouts - 18K checkpoint
    uv run -m experiments.robot.aloha.run_aloha_eval \
        --policy_name cosmos--a_s_v--mixture_20250905_185_demos--chkpt50000 \
        --input_image_size 224 \
        --trained_with_image_aug True \
        --num_open_loop_steps 50 \
        --policy_server_ip 10.12.186.130 \
        --future_img True \
        --max_steps 1100 \
        --run_id_note "Vsprime--bo8--3futureEnsemb--5valEnsemb--majorityMean--seedSwitch--10stepA_5stepSprimeV" \
        --local_log_dir experiments/robot/aloha/logs/ \
        --return_all_query_results True \
        --data_collection True

    # 2025-11-08: Cosmos Policy
    # Best-of-8, Search depth 3, no future state ensemble, no value ensemble, use_last_value search depth value aggregation, 10/5/5 steps a/s'/v
    # Policy: (a, s', V(s, a, s')) policy - 50K checkpoint
    # WMVF model: V(s') - 648 rollouts - 18K checkpoint
    uv run -m experiments.robot.aloha.run_aloha_eval \
        --policy_name cosmos--a_s_v--mixture_20250905_185_demos--chkpt50000 \
        --input_image_size 224 \
        --trained_with_image_aug True \
        --num_open_loop_steps 50 \
        --policy_server_ip 10.12.186.130 \
        --future_img True \
        --max_steps 1100 \
        --run_id_note "Vsprime--bo8--depth3--useLastDepthAgg--10stepA_5stepSprimeV" \
        --local_log_dir experiments/robot/aloha/logs/ \
        --return_all_query_results True \
        --data_collection True

"""

import json
import logging
import os
import pickle
import select
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import draccus
import h5py
import json_numpy
import numpy as np
import requests
import tqdm
from PIL import Image

from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.aloha.aloha_utils import (
    DATE_TIME,
    get_aloha_env,
    get_aloha_image,
    get_aloha_wrist_images,
    get_next_task_label,
    save_rollout_video,
    save_rollout_video_with_all_types_future_image_predictions,
    save_rollout_video_with_future_image_predictions,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

json_numpy.patch()  # Needed for JSON serialization

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})  # Print numpy arrays to 3 decimal places


@dataclass
class PolicyEvalConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    policy_name: str = "cosmos"                     # Model family

    input_image_size: int = 224                      # Image size expected by policy
    trained_with_image_aug: bool = True              # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 50                    # Number of actions to execute open-loop before requerying policy

    policy_server_ip: Union[str, Path] = ""          # Remote policy server IP address (set to 127.0.0.1 if on same machine)

    future_img: bool = False                         # Future image mode: Save model's future image prediction(s)
    return_all_query_results: bool = False           # Whether to return all query results (e.g. if doing best-of-N sampling for planning)

    use_history: bool = False                        # Whether to use observation history
    num_history_indices: int = 8                     # Number of frames to include in history
    history_spacing_factor: int = 12                 # Spacing amount between frames in history

    #################################################################################################################
    # ALOHA environment-specific parameters
    #################################################################################################################
    num_rollouts_planned: int = 50                   # Number of test rollouts
    max_steps: int = 2000                            # Max number of steps per rollout

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    data_collection: bool = True                     # Data collection mode: Collect data and record success/fail
    user_name: str = "USER"                          # User name (for data collection mode)

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def setup_logging(cfg: PolicyEvalConfig):
    """Set up logging to file."""
    # Create run ID
    if cfg.data_collection:
        run_id = f"DATA_COLLECTION--{DATE_TIME}--{cfg.policy_name}-"
    else:
        run_id = f"EVAL--{DATE_TIME}--{cfg.policy_name}-"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging, updating the local log directory to one that is specific to this eval session
    cfg.local_log_dir = os.path.join(cfg.local_log_dir, run_id)
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    print(message)
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def get_server_endpoint(cfg: PolicyEvalConfig):
    """Get the server endpoint for remote inference."""
    return f"http://{cfg.policy_server_ip}:8777/act"


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_aloha_image(obs)
    left_wrist_img, right_wrist_img = get_aloha_wrist_images(obs)

    # Resize images for policy
    img = resize_images(np.expand_dims(img, axis=0), resize_size).squeeze(0)
    left_wrist_img = resize_images(np.expand_dims(left_wrist_img, axis=0), resize_size).squeeze(0)
    right_wrist_img = resize_images(np.expand_dims(right_wrist_img, axis=0), resize_size).squeeze(0)

    # Prepare observations dict
    observation = {
        "primary_image": img,
        "left_wrist_image": left_wrist_img,
        "right_wrist_image": right_wrist_img,
        "proprio": obs.observation["qpos"],
    }

    return observation, img, left_wrist_img, right_wrist_img


def user_requested_terminate() -> bool:
    """Non-blocking check if user typed 't' or 'T' then Enter on stdin."""
    try:
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        if rlist:
            line = sys.stdin.readline()
            if line and line.strip().lower().startswith("t"):
                print("\nReceived 't' from user; terminating current episode...")
                return True
    except Exception:
        # If stdin isn't a TTY or select fails, ignore and continue
        return False
    return False


def get_observation_history_for_policy(cfg: PolicyEvalConfig, full_observation_history: deque, resize_size: int):
    """
    Given a full observation history, returns subsampled observation history for policy.
    """
    assert len(full_observation_history) == full_observation_history.maxlen, (
        "Expected full_observation_history to be totally filled!"
    )

    # Get numpy arrays for individual histories for easy splicing
    full_proprio_history = []
    full_left_wrist_image_history = []
    full_right_wrist_image_history = []
    full_primary_wrist_image_history = []
    for i in range(len(full_observation_history)):
        full_proprio_history.append(full_observation_history[i]["proprio"])
        full_left_wrist_image_history.append(full_observation_history[i]["left_wrist_image"])
        full_right_wrist_image_history.append(full_observation_history[i]["right_wrist_image"])
        full_primary_wrist_image_history.append(full_observation_history[i]["primary_image"])
    full_proprio_history = np.stack(full_proprio_history)
    full_left_wrist_image_history = np.stack(full_left_wrist_image_history)
    full_right_wrist_image_history = np.stack(full_right_wrist_image_history)
    full_primary_wrist_image_history = np.stack(full_primary_wrist_image_history)

    # Compute the step indices for the observation histories
    full_history_last_idx = len(full_observation_history) - 1
    history_indices = get_history_indices(
        curr_step_index=full_history_last_idx,
        num_history_indices=cfg.num_history_indices,
        spacing_factor=cfg.history_spacing_factor,
    )

    # Get individual observation histories
    proprio_history = full_proprio_history[np.array(history_indices)]
    left_wrist_images_history = full_left_wrist_image_history[np.array(history_indices)]
    right_wrist_images_history = full_right_wrist_image_history[np.array(history_indices)]
    primary_images_history = full_primary_wrist_image_history[np.array(history_indices)]

    # Resize images to final size expected by policy
    import time

    start_time = time.time()
    left_wrist_images_history = resize_images(left_wrist_images_history, resize_size)
    right_wrist_images_history = resize_images(right_wrist_images_history, resize_size)
    primary_images_history = resize_images(primary_images_history, resize_size)
    print(f"Image resizing time (sec): {time.time() - start_time:.3f}")

    # Compose the final observation history
    # Order of images: left wrist image, right wrist image, primary image
    EXPECTED_HISTORY_LEN = cfg.num_history_indices * 4
    proprio_history = [x for x in proprio_history]
    images_history = []
    for hist in [left_wrist_images_history, right_wrist_images_history, primary_images_history]:
        for img in hist:
            images_history.append(img)

    assert len(proprio_history) + len(images_history) == EXPECTED_HISTORY_LEN, (
        "Incorrect number of history elements detected!"
    )

    return proprio_history, images_history


def get_action_from_server(
    observation: Dict[str, Any],
    server_endpoint: str = "http://0.0.0.0:8777/act",
    return_all_query_results: bool = False,
) -> Dict[str, Any]:
    """
    Get action from remote inference server.

    Args:
        observation: Observation data to send to server
        server_endpoint: URL of the inference server

    Returns:
        Dict[str, Any]: Action response from server
    """
    # Indicate whether the server should return all query results (e.g. for best-of-N sampling)
    observation["return_all_query_results"] = return_all_query_results

    response = requests.post(
        server_endpoint,
        json=observation,
    ).json()
    if isinstance(response, list):  # Server returned actions only
        actions = response
        future_image_predictions = None
        value_predictions = None
        all_actions = None
        all_future_image_predictions = None
        all_value_predictions = None
        all_actions_by_depth = None
        all_future_image_predictions_by_depth = None
        all_value_predictions_by_depth = None
    elif isinstance(response, dict):  # Server returned dictionary containing actions, future image prediction
        actions = response["actions"]
        future_image_predictions = response["future_image_predictions"]
        value_predictions = response["value_predictions"]
        all_actions = response["all_actions"] if "all_actions" in response else None
        all_future_image_predictions = (
            response["all_future_image_predictions"] if "all_future_image_predictions" in response else None
        )
        all_value_predictions = response["all_value_predictions"] if "all_value_predictions" in response else None
        all_actions_by_depth = response["all_actions_by_depth"] if "all_actions_by_depth" in response else None
        all_future_image_predictions_by_depth = (
            response["all_future_image_predictions_by_depth"]
            if "all_future_image_predictions_by_depth" in response
            else None
        )
        all_value_predictions_by_depth = (
            response["all_value_predictions_by_depth"] if "all_value_predictions_by_depth" in response else None
        )
    else:
        raise ValueError(f"Got unexpected server response of type {type(response)}")

    return (
        actions,
        future_image_predictions,
        value_predictions,
        all_actions,
        all_future_image_predictions,
        all_value_predictions,
        all_actions_by_depth,
        all_future_image_predictions_by_depth,
        all_value_predictions_by_depth,
    )


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

    # Stop if the images are already at the target size
    if images.shape[1] == images.shape[2] == target_size:
        print("No need to resize!")
        return images

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


def create_video_from_images(images: np.ndarray, video_path: str, fps: int = 25):
    """
    Creates an MP4 video from a sequence of images (RGB uint8).

    Args:
        images: numpy array of shape (num_frames, height, width, channels)
        video_path: output MP4 path
        fps: frames per second
    """
    if images is None or len(images) == 0:
        raise ValueError("No images provided for video creation")

    height, width, channels = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for img in images:
        # Convert RGB to BGR for OpenCV
        if channels == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        out.write(img_bgr)

    out.release()
    print(f"Video saved: {video_path}")


def save_rollout_episode_hdf5(
    episode_idx: int,
    rollout_dir: str,
    cam_high_images: np.ndarray,
    cam_left_wrist_images: np.ndarray,
    cam_right_wrist_images: np.ndarray,
    actions: np.ndarray,
    qpos: np.ndarray,
    task_description: str,
    success_flag: bool,
    success_score: float,
    user_description: str,
    policy_name: str,
    user_name: str,
    fps: int = 25,
    average_model_query_time: float = None,
    args_json: str = None,
):
    """
    Save rollout episode in the same format as preprocess_split_aloha_data.py:
      - MP4 videos for images, referenced via observations/video_paths in HDF5
      - HDF5 with observations/{qpos,qvel,effort}, action, relative_action
      - Root attributes include: sim, task_description, success, user_description, policy_name, user_name, date_time
    """
    os.makedirs(rollout_dir, exist_ok=True)

    # Ensure arrays
    cam_high_images = np.asarray(cam_high_images)
    cam_left_wrist_images = np.asarray(cam_left_wrist_images)
    cam_right_wrist_images = np.asarray(cam_right_wrist_images)
    actions = np.asarray(actions, dtype=np.float32)
    qpos = np.asarray(qpos, dtype=np.float32)

    episode_len = int(actions.shape[0])

    # Create MP4 videos
    vid_cam_high = os.path.join(rollout_dir, f"episode_{episode_idx}_cam_high.mp4")
    vid_cam_left = os.path.join(rollout_dir, f"episode_{episode_idx}_cam_left_wrist.mp4")
    vid_cam_right = os.path.join(rollout_dir, f"episode_{episode_idx}_cam_right_wrist.mp4")
    create_video_from_images(cam_high_images, vid_cam_high, fps=fps)
    create_video_from_images(cam_left_wrist_images, vid_cam_left, fps=fps)
    create_video_from_images(cam_right_wrist_images, vid_cam_right, fps=fps)

    # Relative paths for portability
    video_filenames = {
        "cam_high": os.path.basename(vid_cam_high),
        "cam_left_wrist": os.path.basename(vid_cam_left),
        "cam_right_wrist": os.path.basename(vid_cam_right),
    }

    # Build relative actions
    relative_actions = np.zeros_like(actions, dtype=np.float32)
    if episode_len >= 2:
        relative_actions[:-1] = actions[1:] - actions[:-1]
        relative_actions[-1] = relative_actions[-2]

    # qvel/effort unknown from real robot interface here; save zeros for compatibility
    qvel = np.zeros_like(qpos, dtype=np.float32)
    effort = np.zeros_like(qpos, dtype=np.float32)

    # Save HDF5
    h5_path = os.path.join(rollout_dir, f"episode_{episode_idx}.hdf5")
    with h5py.File(h5_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["task_description"] = task_description
        root.attrs["success"] = bool(success_flag)
        if user_description is not None:
            root.attrs["user_description"] = user_description
        if policy_name is not None:
            root.attrs["policy_name"] = policy_name
        if user_name is not None:
            root.attrs["user_name"] = user_name
        root.attrs["date_time"] = DATE_TIME
        if average_model_query_time is not None:
            root.attrs["average_model_query_time_sec"] = float(average_model_query_time)
        if success_score is not None:
            root.attrs["success_score"] = float(success_score)

        obs = root.create_group("observations")
        _ = obs.create_dataset("qpos", (episode_len, qpos.shape[1]))
        _ = obs.create_dataset("qvel", (episode_len, qvel.shape[1]))
        _ = obs.create_dataset("effort", (episode_len, effort.shape[1]))
        root["/observations/qpos"][...] = qpos
        root["/observations/qvel"][...] = qvel
        root["/observations/effort"][...] = effort

        video_paths_group = obs.create_group("video_paths")
        for cam_name, video_filename in video_filenames.items():
            video_paths_group.create_dataset(cam_name, data=video_filename.encode("utf-8"))

        _ = root.create_dataset("action", (episode_len, actions.shape[1]))
        root["/action"][...] = actions

        _ = root.create_dataset("relative_action", (episode_len, actions.shape[1]))
        root["/relative_action"][...] = relative_actions

        # Save full set of args (config) as JSON for reproducibility
        if args_json is not None:
            try:
                dt = h5py.string_dtype(encoding="utf-8")
                root.create_dataset("args_json", data=args_json, dtype=dt)
            except Exception:
                pass

    print(f"Saved rollout episode HDF5: {h5_path}")
    return h5_path


def run_episode(
    cfg: PolicyEvalConfig,
    env,
    task_description: str,
    server_endpoint: str,
    episode_idx: int,
    log_file=None,
):
    """Run a single episode in the ALOHA environment."""
    # Define control frequency
    STEP_DURATION_IN_SEC = 1.0 / 25.0

    # Reset environment
    obs = env.reset()

    # Initialize action queue
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Initialize full observation history
    if cfg.use_history:
        FULL_HISTORY_LEN = cfg.num_history_indices * cfg.history_spacing_factor + 1
        full_observation_history = deque(maxlen=FULL_HISTORY_LEN)

    # Setup
    t = 0
    primary_images_list = []
    left_wrist_images_list = []
    right_wrist_images_list = []
    resized_primary_images_list = []
    resized_left_wrist_images_list = []
    resized_right_wrist_images_list = []
    future_image_predictions_list = []
    executed_actions_list = []
    executed_qpos_list = []

    # If in data collection mode, set up data buffers
    if cfg.data_collection:
        history_proprio_list = []
        history_images_list = []
        actions_list = []
        all_actions_list = []
        all_future_image_predictions_list = []
        all_value_predictions_list = []
        all_actions_by_depth_list = []
        all_future_image_predictions_by_depth_list = []
        all_value_predictions_by_depth_list = []
        all_timesteps_list = []

    print("Prepare the scene, and then press Enter to begin...")
    print("(Tip: type 't' then Enter at any time during the episode to terminate it early.)")
    input()

    # Reset environment again to fetch first timestep observation
    obs = env.reset()

    # Fetch initial robot state (but sleep first so that robot stops moving)
    time.sleep(3)

    episode_start_time = time.time()
    total_model_query_time = 0.0
    model_query_count = 0

    should_end_episode_early = False

    try:
        while t < cfg.max_steps:
            # Get step start time (used to compute how much to sleep between steps)
            step_start_time = time.time()

            # Non-blocking termination check
            if user_requested_terminate():
                break

            # Get observation
            obs = env.get_observation(t=t)

            # Add to observation history
            if cfg.use_history:
                if t == 0:
                    # In the first step, fill full observation history with duplicates of the first step observation
                    for i in range(full_observation_history.maxlen):
                        full_observation_history.append(
                            dict(
                                proprio=obs.observation["qpos"],
                                left_wrist_image=obs.observation["images"]["cam_left_wrist"],
                                right_wrist_image=obs.observation["images"]["cam_right_wrist"],
                                primary_image=obs.observation["images"]["cam_high"],
                            )
                        )
                else:
                    # In all other steps, just add one observation to the full history
                    full_observation_history.append(
                        dict(
                            proprio=obs.observation["qpos"],
                            left_wrist_image=obs.observation["images"]["cam_left_wrist"],
                            right_wrist_image=obs.observation["images"]["cam_right_wrist"],
                            primary_image=obs.observation["images"]["cam_high"],
                        )
                    )

            # Save raw images for replay videos and data collection
            primary_images_list.append(obs.observation["images"]["cam_high"])
            left_wrist_images_list.append(obs.observation["images"]["cam_left_wrist"])
            right_wrist_images_list.append(obs.observation["images"]["cam_right_wrist"])

            # If action queue is empty, requery model
            should_execute_action = True
            if len(action_queue) == 0:
                print(f"\nt == {t}. Action queue empty. Time to requery model...")

                if should_end_episode_early:
                    break

                # Prepare observation
                observation, img_resized, left_wrist_resized, right_wrist_resized = prepare_observation(
                    obs, resize_size=cfg.input_image_size
                )
                observation["task_description"] = task_description

                # Prepare observation history that will be sent to policy
                if cfg.use_history:
                    proprio_history, images_history = get_observation_history_for_policy(
                        cfg, full_observation_history, resize_size=cfg.input_image_size
                    )
                    observation["proprio_history"] = proprio_history
                    observation["images_history"] = images_history

                # Save processed images for additional replay videos
                resized_primary_images_list.append(img_resized)
                resized_left_wrist_images_list.append(left_wrist_resized)
                resized_right_wrist_images_list.append(right_wrist_resized)

                # Query model to get action
                print("Requerying model...")
                model_query_start_time = time.time()
                (
                    actions,
                    future_image_predictions,
                    value_predictions,
                    all_actions,
                    all_future_image_predictions,
                    all_value_predictions,
                    all_actions_by_depth,
                    all_future_image_predictions_by_depth,
                    all_value_predictions_by_depth,
                ) = get_action_from_server(observation, server_endpoint, cfg.return_all_query_results)
                actions = actions[: cfg.num_open_loop_steps]
                total_model_query_time += time.time() - model_query_start_time
                model_query_count += 1

                if should_execute_action:
                    action_queue.extend(actions)
                    future_image_predictions_list.append(future_image_predictions)
                    if cfg.data_collection:
                        if cfg.use_history:
                            history_proprio_list.append(observation["proprio_history"])
                            history_images_list.append(observation["images_history"])
                        actions_list.append(actions)
                        all_actions_list.append(all_actions)
                        all_future_image_predictions_list.append(all_future_image_predictions)
                        all_value_predictions_list.append(all_value_predictions)
                        all_actions_by_depth_list.append(all_actions_by_depth)
                        all_future_image_predictions_by_depth_list.append(all_future_image_predictions_by_depth)
                        all_value_predictions_by_depth_list.append(all_value_predictions_by_depth)
                        all_timesteps_list.append(t)

            if not should_execute_action:
                continue

            # Get action from queue
            action = action_queue.popleft()
            # Execute action in environment
            # For safety reasons, pause if the difference between current and target joints exceeds a certain threshold
            MAX_JOINT_DIFFERENCE = 0.15
            START_SAFETY_CHECK_TIME = time.time()
            should_terminate = False
            while True:
                if user_requested_terminate():
                    should_execute_action = False
                    break
                left_arm_difference = (action - env.get_qpos())[:6]  # Left arm joints, excluding gripper
                right_arm_difference = (action - env.get_qpos())[7:13]  # Right arm joints, excluding gripper
                if np.any(left_arm_difference > MAX_JOINT_DIFFERENCE) or np.any(
                    right_arm_difference > MAX_JOINT_DIFFERENCE
                ):
                    # Print a warning every second
                    if time.time() - START_SAFETY_CHECK_TIME > 1.0:
                        user_input_safety = input(
                            f"\nDangerous action detected! Pausing evaluation...\n\tMAX_JOINT_DIFFERENCE = {MAX_JOINT_DIFFERENCE}\n\tleft_arm_difference: {left_arm_difference}\n\tright_arm_difference: {right_arm_difference}\n\taction: {action}\n\tstate: {env.get_qpos()}\nPress 'c' to continue, or 't' to terminate: "
                        )
                        if user_input_safety == "c":
                            print("Proceeding...")
                            break
                        elif user_input_safety == "t":
                            print("Terminating due to safety...")
                            should_terminate = True
                            break
                        START_SAFETY_CHECK_TIME = time.time()
                else:
                    break
            # Record pre-action qpos and action to logs
            executed_qpos_list.append(env.get_qpos().copy())
            executed_actions_list.append(action.copy())

            if should_terminate:
                break
            obs = env.step(action.tolist())
            t += 1

            # Sleep until next timestep
            step_elapsed_time = time.time() - step_start_time
            if step_elapsed_time < STEP_DURATION_IN_SEC:
                time_to_sleep = STEP_DURATION_IN_SEC - step_elapsed_time
                print(f"Sleeping {time_to_sleep} sec...")
                time.sleep(time_to_sleep)

    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\nCaught KeyboardInterrupt: Terminating episode early.")
        else:
            print(f"\nCaught exception: {e}")

    episode_end_time = time.time()

    # Save temporary replay video in case the user wasn't paying attention to what happened
    save_rollout_video(
        primary_images_list,
        -1,
        success="Null",
        task_description=task_description,
        policy_name="Null",
        rollout_dir="./temp/",
        log_file=None,
        notes="TEMP_REPLAY",
    )

    # Offer redo option before asking success question
    while True:
        redo_input = input(
            "\nRedo this episode without saving/logging anything? Enter 'r' to redo, or press Enter to continue: "
        )
        if redo_input.lower() == "r":
            confirm = input(
                "You chose to redo this episode. If this is NOT correct, press 'b' to go back. Otherwise, press Enter to confirm: "
            )
            if confirm == "b":
                continue
            # Signal caller to redo; do not log or save anything
            return (
                None,
                primary_images_list,
                left_wrist_images_list,
                right_wrist_images_list,
                resized_primary_images_list,
                resized_left_wrist_images_list,
                resized_right_wrist_images_list,
                future_image_predictions_list,
                True,
            )
        break

    # Get success feedback from user
    print("-----------------------------------------------------\n\nEpisode complete!")
    while True:
        user_input1 = input(
            "\nSuccess? Enter 'y' for yes (score=1.0), 'p' for partial (you will enter a score), or 'n' for no (score=0.0): "
        )
        if user_input1.lower() in ["y", "p", "n"]:
            user_input2 = input(
                f"You entered '{user_input1}'. If this is NOT correct, press 'b' to go back and reselect ('y', 'p', 'n'). Otherwise, press Enter to continue: "
            )
            if user_input2 == "b":
                continue  # Go back and try again
            else:
                break  # Correct user input confirmed; move on
        else:
            continue  # Go back and try again
    success_score = None
    if user_input1 == "y":
        success = "True"
        success_score = 1.0
    elif user_input1 == "p":
        success = "Partial"
        # Ask user to enter a custom floating point score and confirm
        while True:
            score_str = input("Enter a numeric success score (e.g., 0.7): ")
            try:
                parsed_score = float(score_str)
            except Exception:
                print("Invalid number. Please try again.")
                continue
            confirm = input(
                f"You entered a score of {parsed_score}. If this is NOT correct, press 'b' to reenter. Otherwise, press Enter to continue: "
            )
            if confirm == "b":
                continue
            success_score = parsed_score
            break
    else:
        success = "False"
        success_score = 0.0

    # Get user description of policy rollout
    while True:
        user_input1 = input("Please describe in 1-3 sentences what the policy did in this episode: ")
        user_input2 = input(
            f"You entered the description below:\n\n{user_input1}\n\nIf this is NOT correct, press 'b' to go back and reenter the description. Otherwise, press Enter to continue: "
        )
        if user_input2 == "b":
            continue  # Go back and try again
        else:
            break  # Correct user input confirmed; move on
    user_description = user_input1

    # Calculate episode statistics
    episode_stats = {
        "success": success,
        "success_score": success_score,
        "user_description": user_description,
        "total_steps": t,
        "average_model_query_time": total_model_query_time / model_query_count,
        "episode_duration": episode_end_time - episode_start_time,
    }

    # In data collection mode, save rollout episode as HDF5 + MP4 videos
    if cfg.data_collection:
        # Enforce strict 1:1 mapping between observations and actions (hard fail on mismatch)
        num_exec_steps = len(executed_actions_list)
        assert num_exec_steps > 0, "No executed actions recorded; refusing to save empty rollout."
        assert len(primary_images_list) == num_exec_steps, (
            f"Mismatch: cam_high images ({len(primary_images_list)}) != actions ({num_exec_steps})"
        )
        assert len(left_wrist_images_list) == num_exec_steps, (
            f"Mismatch: left wrist images ({len(left_wrist_images_list)}) != actions ({num_exec_steps})"
        )
        assert len(right_wrist_images_list) == num_exec_steps, (
            f"Mismatch: right wrist images ({len(right_wrist_images_list)}) != actions ({num_exec_steps})"
        )
        assert len(executed_qpos_list) == num_exec_steps, (
            f"Mismatch: qpos count ({len(executed_qpos_list)}) != actions ({num_exec_steps})"
        )

        cam_high_np = np.stack(primary_images_list, axis=0)
        cam_left_np = np.stack(left_wrist_images_list, axis=0)
        cam_right_np = np.stack(right_wrist_images_list, axis=0)
        actions_np = np.stack(executed_actions_list, axis=0).astype(np.float32)
        qpos_np = np.stack(executed_qpos_list, axis=0).astype(np.float32)

        # Save main HDF5 to disk
        print("Saving rollout episode (HDF5 + MP4)...")
        _ = save_rollout_episode_hdf5(
            episode_idx=episode_idx,
            rollout_dir=cfg.local_log_dir,
            cam_high_images=cam_high_np,
            cam_left_wrist_images=cam_left_np,
            cam_right_wrist_images=cam_right_np,
            actions=actions_np,
            qpos=qpos_np,
            task_description=task_description,
            success_flag=True if episode_stats["success"] == "True" else False,
            success_score=episode_stats["success_score"],
            user_description=user_description,
            policy_name=cfg.policy_name,
            user_name=cfg.user_name,
            fps=25,
            average_model_query_time=episode_stats["average_model_query_time"],
            args_json=json.dumps(asdict(cfg), default=str, indent=2),
        )

        # Also save a pickle file of all query results if needed (e.g. for post-hoc analysis of best-of-N sampling)
        if cfg.return_all_query_results:
            assert (
                len(all_actions_list)
                == len(all_future_image_predictions_list)
                == len(all_value_predictions_list)
                == len(all_timesteps_list)
            ), (
                f"Error: Lengths of data lists do not match!\n\tlen(all_actions_list) == {len(all_actions_list)}\n\tlen(all_future_image_predictions_list) == {len(all_future_image_predictions_list)}\n\tlen(all_value_predictions_list) == {len(all_value_predictions_list)}\n\tlen(all_timesteps_list) == {len(all_timesteps_list)}"
            )
            assert (
                len(all_actions_by_depth_list)
                == len(all_future_image_predictions_by_depth_list)
                == len(all_value_predictions_by_depth_list)
                == len(all_timesteps_list)
            ), (
                f"Error: Lengths of data lists do not match!\n\tlen(all_actions_by_depth_list) == {len(all_actions_by_depth_list)}\n\tlen(all_future_image_predictions_by_depth_list) == {len(all_future_image_predictions_by_depth_list)}\n\tlen(all_value_predictions_by_depth_list) == {len(all_value_predictions_by_depth_list)}\n\tlen(all_timesteps_list) == {len(all_timesteps_list)}"
            )
            all_query_results = []
            for i in range(len(all_actions_list)):
                # Remove the "future_proprio" image from the future image predictions dict if it exists
                for future_image_predictions in all_future_image_predictions_list[i]:
                    if "future_proprio" in future_image_predictions:
                        future_image_predictions.pop("future_proprio")

                # Gather all query results
                all_query_results.append(
                    dict(
                        timestep=all_timesteps_list[i],
                        all_actions=all_actions_list[i],
                        all_future_image_predictions=all_future_image_predictions_list[i],
                        all_value_predictions=all_value_predictions_list[i],
                        all_actions_by_depth=all_actions_by_depth_list[i],
                        all_future_image_predictions_by_depth=all_future_image_predictions_by_depth_list[i],
                        all_value_predictions_by_depth=all_value_predictions_by_depth_list[i],
                    )
                )
            pickle_path = os.path.join(cfg.local_log_dir, f"episode_{episode_idx}_all_query_results.pkl")
            with open(pickle_path, "wb") as file:
                pickle.dump(all_query_results, file)
            print(f"Saved all query results for rollout episode: {pickle_path}")

    return (
        episode_stats,
        primary_images_list,
        left_wrist_images_list,
        right_wrist_images_list,
        resized_primary_images_list,
        resized_left_wrist_images_list,
        resized_right_wrist_images_list,
        future_image_predictions_list,
        False,
    )


def save_episode_videos(
    primary_images_list,
    left_wrist_images_list,
    right_wrist_images_list,
    resized_primary_images_list,
    resized_left_wrist_images_list,
    resized_right_wrist_images_list,
    episode_idx,
    success,
    task_description,
    policy_name,
    local_log_dir,
    log_file=None,
):
    """Save videos of the episode from different camera angles."""
    # Save raw replay videos
    save_rollout_video(
        primary_images_list,
        episode_idx,
        success=success,
        task_description=task_description,
        policy_name=policy_name,
        rollout_dir=local_log_dir,
        log_file=log_file,
        notes="raw",
    )
    save_rollout_video(
        left_wrist_images_list,
        episode_idx,
        success=success,
        task_description=task_description,
        policy_name=policy_name,
        rollout_dir=local_log_dir,
        log_file=log_file,
        notes="left_wrist_raw",
    )
    save_rollout_video(
        right_wrist_images_list,
        episode_idx,
        success=success,
        task_description=task_description,
        policy_name=policy_name,
        rollout_dir=local_log_dir,
        log_file=log_file,
        notes="right_wrist_raw",
    )

    # Save processed replay videos
    save_rollout_video(
        resized_primary_images_list,
        episode_idx,
        success=success,
        task_description=task_description,
        policy_name=policy_name,
        rollout_dir=local_log_dir,
        log_file=log_file,
        notes="resized",
    )
    save_rollout_video(
        resized_left_wrist_images_list,
        episode_idx,
        success=success,
        task_description=task_description,
        policy_name=policy_name,
        rollout_dir=local_log_dir,
        log_file=log_file,
        notes="left_wrist_resized",
    )
    save_rollout_video(
        resized_right_wrist_images_list,
        episode_idx,
        success=success,
        task_description=task_description,
        policy_name=policy_name,
        rollout_dir=local_log_dir,
        log_file=log_file,
        notes="right_wrist_resized",
    )


@draccus.wrap()
def eval_aloha(cfg: PolicyEvalConfig) -> None:
    """Main function to evaluate a trained policy in a real-world ALOHA environment."""
    # Check that a new policy is loaded onto the remote server
    user_input = input("\nREMINDER -- Have you asked Moo Jin to load the new policy? Enter 'y' or 'n': ")
    if user_input != "y":
        print("Please ask Moo Jin to load the new policy. Afterwards, rerun this script. Quitting...")
        exit(0)

    # Check if user wants to return all query results when doing best-of-N sampling for planning
    if not cfg.return_all_query_results:
        user_input = ""
        while user_input.lower() not in ["y", "n"]:
            user_input = input(
                "\nWould you like the server to return all query results for best-of-N sampling? Enter 'y' or 'n': "
            )
            if user_input == "y":
                cfg.return_all_query_results = True
            elif user_input == "n":
                cfg.return_all_query_results = False

    # In data collection mode, ask for user's name
    if cfg.data_collection:
        user_input = input(
            f"\nCurrent user name is {cfg.user_name}. Press Enter to continue or enter a new username to overwrite: "
        )
        if user_input != "":
            cfg.user_name = user_input
        print(f"Continuing as user {cfg.user_name}...\n")
        time.sleep(1)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Get ALOHA environment
    env = get_aloha_env()

    # Get server endpoint for remote inference
    server_endpoint = get_server_endpoint(cfg)

    # Initialize task description
    task_description = ""

    # Start evaluation
    num_rollouts_completed, total_successes = 0, 0

    for episode_idx in tqdm.tqdm(range(cfg.num_rollouts_planned)):
        # Get task description from user
        task_description = get_next_task_label(task_description)
        print(f"\nTask: {task_description}")

        print(f"Starting episode {num_rollouts_completed + 1}...")

        # Run episode
        (
            episode_stats,
            primary_images_list,
            left_wrist_images_list,
            right_wrist_images_list,
            resized_primary_images_list,
            resized_left_wrist_images_list,
            resized_right_wrist_images_list,
            future_image_predictions_list,
            redo_requested,
        ) = run_episode(cfg, env, task_description, server_endpoint, episode_idx, log_file)

        # If redo requested, rerun the same episode index without logging/saving/updating counters
        if redo_requested:
            print("Redo requested. Re-running episode without logging...")
            continue

        # Update counters
        num_rollouts_completed += 1
        # Accumulate numeric success score
        total_successes += float(episode_stats.get("success_score", 0.0))

        # Reset environment now so that the user can set the next scene while rollout videos are being saved
        env.reset()

        # Save videos
        print("Saving rollout videos; please wait...")
        save_episode_videos(
            primary_images_list,
            left_wrist_images_list,
            right_wrist_images_list,
            resized_primary_images_list,
            resized_left_wrist_images_list,
            resized_right_wrist_images_list,
            num_rollouts_completed,
            episode_stats["success"],
            task_description,
            cfg.policy_name,
            cfg.local_log_dir,
            log_file,
        )

        # Save replay video with future image predictions included
        if cfg.future_img:
            if future_image_predictions_list[0]["future_wrist_image"] is not None:
                future_wrist_image_predictions = [x["future_wrist_image"] for x in future_image_predictions_list]
                future_wrist_image2_predictions = [x["future_wrist_image2"] for x in future_image_predictions_list]
                future_primary_image_predictions = [x["future_image"] for x in future_image_predictions_list]
                save_rollout_video_with_all_types_future_image_predictions(
                    primary_images_list,
                    left_wrist_images_list,
                    right_wrist_images_list,
                    future_wrist_image_predictions,
                    future_wrist_image2_predictions,
                    future_primary_image_predictions,
                    num_rollouts_completed,
                    success=episode_stats["success"],
                    task_description=task_description,
                    chunk_size=max(cfg.num_open_loop_steps, 25),  # ALOHA @ 25 Hz control
                    num_open_loop_steps=cfg.num_open_loop_steps,
                    trained_with_image_aug=cfg.trained_with_image_aug,
                    policy_name=cfg.policy_name,
                    rollout_dir=cfg.local_log_dir,
                    log_file=log_file,
                )
            else:
                future_primary_image_predictions = [x["future_image"] for x in future_image_predictions_list]
                save_rollout_video_with_future_image_predictions(
                    primary_images_list,
                    left_wrist_images_list,
                    right_wrist_images_list,
                    future_primary_image_predictions,
                    num_rollouts_completed,
                    success=episode_stats["success"],
                    task_description=task_description,
                    chunk_size=max(cfg.num_open_loop_steps, 25),  # ALOHA @ 25 Hz control
                    num_open_loop_steps=cfg.num_open_loop_steps,
                    trained_with_image_aug=cfg.trained_with_image_aug,
                    policy_name=cfg.policy_name,
                    rollout_dir=cfg.local_log_dir,
                    log_file=log_file,
                )

        # Log results
        log_message(f"Success: {episode_stats['success']}", log_file)
        log_message(f"Success score: {episode_stats['success_score']}", log_file)
        log_message(f"User description: {episode_stats['user_description']}", log_file)
        log_message(f"# episodes completed so far: {num_rollouts_completed}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / num_rollouts_completed * 100:.1f}%)", log_file)
        log_message(f"Average model query time: {episode_stats['average_model_query_time']:.2f} sec", log_file)
        log_message(f"Total episode elapsed time: {episode_stats['episode_duration']:.2f} sec", log_file)

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(num_rollouts_completed) if num_rollouts_completed > 0 else 0

    # Log final results
    log_message("\nFinal results:", log_file)
    log_message(f"Total episodes: {num_rollouts_completed}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_aloha()
