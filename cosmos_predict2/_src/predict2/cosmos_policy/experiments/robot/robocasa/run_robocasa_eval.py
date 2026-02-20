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
run_robocasa_eval.py

Evaluates a trained policy in a RoboCasa simulation benchmark task suite.

Parallel inference:
    (Only applicable when doing best-of-N search with N GPUs)
    To enable parallel inference across 8 GPUs, use:
        --use_parallel_inference True
        --num_queries_best_of_n 8
        --available_gpus "0,1,2,3,4,5,6,7"

Usage examples:
    # *** Main checkpoint: 67.1% avg success rate ***
    #   Replace `task_suite_name` with one of {libero_spatial, libero_object, libero_goal, libero_10}
    #   Replace `seed` with one of {195, 196, 197}
    #   Replace `run_id_note` with a unique identifier for the run
    uv run -m cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
        --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
        --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
        --config_file cosmos_predict2/_src/predict2/cosmos_policy/config/config.py \
        --use_wrist_image True \
        --num_wrist_images 1 \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 32 \
        --num_open_loop_steps 16 \
        --task_name TurnOffMicrowave \
        --num_trials_per_task 50 \
        --run_id_note chkpt45000--5stepAct--seed195--deterministic \
        --local_log_dir cosmos_policy/experiments/robot/robocasa/logs/ \
        --seed 195 \
        --randomize_seed False \
        --deterministic True \
        --use_variance_scale False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --data_collection False

"""

import ast
import multiprocessing as mp
import os
import pickle
import secrets
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import draccus
import h5py
import numpy as np
import robosuite
import torch
import wandb
from robocasa.utils.dataset_registry import MULTI_STAGE_TASK_DATASETS, SINGLE_STAGE_TASK_DATASETS

from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
    ACTION_DIM,
    WorkerPoolManager,
    extract_action_chunk_from_latent_sequence,
    get_action,
    get_future_state_prediction,
    get_model,
    get_planning_model,
    get_qvalue_prediction,
    get_value_prediction,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    query_model_parallel,
    unnormalize_actions,
)
from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.robocasa.robocasa_utils import (
    save_rollout_video,
    save_rollout_video_with_future_image_predictions,
)
from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.robot_utils import (
    DATE_TIME,
    log_message,
    setup_logging,
)
from cosmos_predict2._src.predict2.cosmos_policy.utils.utils import jpeg_encode_image, set_seed_everywhere

# Cosmos Policy latent sequence indices
# 0: blank, 1: curr proprio, 2: curr wrist img, 3: curr primary img, 4: curr secondary img, 5: action, 6: future proprio, 7: future wrist img, 8: future primary img, 9: future secondary img, 10: value
CURR_STATE_START_LATENT_IDX, CURR_STATE_END_LATENT_IDX = 1, 4
FUTURE_STATE_START_LATENT_IDX, FUTURE_STATE_END_LATENT_IDX = 6, 9

# Path to fixed controller configs
CONTROLLER_CONFIGS_PATH: str = (
    "cosmos_predict2/_src/predict2/cosmos_policy/experiments/robot/robocasa/robocasa_controller_configs.pkl"
)

# Define max steps for each RoboCasa task (based on horizons in dataset_registry)
TASK_MAX_STEPS = {
    # Pick and place tasks
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    # Door tasks
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    # Drawer tasks
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    # Stove tasks
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    # Sink tasks
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    # Coffee tasks
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    # Microwave tasks
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}


@dataclass
class PolicyEvalConfig:
    # fmt: off
    suite: str = "robocasa"                                              # Evaluation suite name

    #################################################################################################################
    # Cosmos Policy-specific parameters
    #################################################################################################################
    model_family: str = "cosmos"                                         # Model family
    config: str = ""                                                     # Inference config name
    ckpt_path: str = ""                                                  # Pretrained checkpoint path
    planning_model_config_name: str = ""                                 # Planning model config name
    planning_model_ckpt_path: str = ""                                   # Planning model checkpoint path
    config_file: str = "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py"  # Cosmos default config file path

    use_third_person_image: bool = True                                  # Whether to include third-person ("primary") image in input
    num_third_person_images: int = 2                                     # Number of third-person images to include in input (RoboCasa: 1 for left, 1 for right)
    use_wrist_image: bool = True                                         # Whether to include wrist image in input
    num_wrist_images: int = 1                                            # Number of wrist images to include in input (RoboCasa: 1 wrist image)
    use_proprio: bool = True                                             # Whether to include proprio state in input
    flip_images: bool = True                                             # Whether to flip images vertically across x-axis (RoboCasa: True because environment returns images upside down)
    use_variance_scale: bool = False                                     # Whether to scale variance used to sample sigma max for denoising for increased diversity in generations
    use_jpeg_compression: bool = True                                    # Whether to use JPEG compression on images before querying policy
    ar_future_prediction: bool = False                                   # Whether to predict future state autoregressively
    ar_value_prediction: bool = False                                    # Whether to predict future state value autoregressively
    ar_qvalue_prediction: bool = False                                   # Whether to predict Q-value autoregressively
    num_denoising_steps_action: int = 5                                  # Number of denoising steps to take for action prediction
    num_denoising_steps_future_state: int = 1                            # Number of denoising steps to take for future state prediction (only applicable if ar_future_prediction is True; otherwise equal to num_denoising_steps_action)
    num_denoising_steps_value: int = 1                                   # Number of denoising steps to take for value prediction (only applicable if ar_value_prediction is True; otherwise equal to num_denoising_steps_action)
    shift: int = 5                                                # Shift parameter for rectified flow scheduler timestep distribution
    unnormalize_actions: bool = True                                     # Unnormalize actions if trained with normalized actions
    normalize_proprio: bool = True                                       # Normalize proprio input if trained with normalized proprio
    dataset_stats_path: str = ""                                         # Path to dataset statistics file for action unnormalization and proprio normalization
    t5_text_embeddings_path: str = ""                                    # Path to precomputed T5 text embeddings dictionary (key: instruction, val: embedding)
    text_embeddings_kind: str = "t5"                                     # Type of text embeddings: "t5" or "reason1"
    trained_with_image_aug: bool = True                                  # Whether the model was trained with image augmentations (needed for test-time image transformations)
    chunk_size: int = 32                                                 # Number of actions to predict in chunk
    num_open_loop_steps: int = 16                                        # Number of actions in predicted chunk to execute open-loop before requerying policy

    deterministic: bool = True                                           # Whether to run in deterministic mode
    deterministic_reset: bool = False                                    # Whether to run in deterministic reset mode (sets global random seed right before env reset)
    deterministic_reset_seed: int = None                                 # (Only applicable if deterministic_reset==True) The seed to set before deterministic reset; if not provided, defaults to the base seed

    #################################################################################################################
    # Planning model and best-of-N search parameters
    #################################################################################################################
    use_ensemble_future_state_predictions: bool = False                  # Whether to use ensemble of future state predictions
    num_future_state_predictions_in_ensemble: int = 3                    # Number of future state predictions in ensemble
    future_state_ensemble_aggregation_scheme: str = "average"            # How to aggregate future state predictions in an ensemble of future state predictions (options: "average", "last")
    use_ensemble_value_predictions: bool = False                         # Whether to use ensemble of value predictions
    num_value_predictions_in_ensemble: int = 5                           # Number of value predictions in ensemble
    value_ensemble_aggregation_scheme: str = "average"                   # How to aggregate values in an ensemble of value predictions (options: "average", "lcb", "success_vote", "majority_mean")
    search_depth: int = 1                                                # Number of levels to search through in the best-of-N search tree
    search_depth_value_aggregation_scheme: str = "use_last_value"        # How to aggregate value predictions across search depth (options: use_last_value, average)
    mask_current_state_action_for_value_prediction: bool = False         # Whether to use input masking to mask out certain inputs (current state and action) during value prediction
    mask_future_state_for_qvalue_prediction: bool = False                # Whether to use input masking to mask out certain inputs (future state) during Q(s, a) value prediction

    num_queries_best_of_n: int = 1                                       # Number of queries to make to the model (this is the N in best-of-N search)
    use_parallel_inference: bool = False                                 # Whether to use parallel inference across multiple GPUs
    available_gpus: str = "0,1,2,3,4,5,6,7"                              # Comma-separated list of GPU IDs available for use for parallel inference (defaults to all 8 GPUs on a node)
    parallel_timeout: int = 15                                           # Timeout in seconds for each parallel query

    #################################################################################################################
    # RoboCasa-specific parameters
    #################################################################################################################
    task_name: str = "PnPCounterToCab"                                   # Task name (must be in SINGLE_STAGE_TASK_DATASETS or MULTI_STAGE_TASK_DATASETS)
    num_trials_per_task: int = 50                                        # Number of rollouts per task
    env_img_res: int = 224                                               # Resolution for rendering environment images
    robots: str = "PandaMobile"                                          # Robot type for RoboCasa (PandaMobile is alias for PandaOmron)
    controllers: str = "OSC_POSE"                                        # Controller type (OSC_POSE = Operational Space Control with 6-DOF end-effector pose)
    obj_instance_split: str = "B"                                        # Object instance split - "B" = held-out test objects
    layout_and_style_ids: str = "((1,1),(2,2),(4,4),(6,9),(7,10))"       # Layout and style IDs - 5 test scenes
    randomize_cameras: bool = False                                      # Whether to randomize camera positions

    #################################################################################################################
    # Utils
    #################################################################################################################
    local_log_dir: str = "./experiments/logs"                            # Local directory for eval logs
    run_id_note: Optional[str] = None                                    # Extra note to add to end of run ID for logging

    use_wandb: bool = False                                              # Whether to also log results in Weights & Biases
    wandb_entity: str = "YOUR_ENTITY"                                    # Name of WandB entity
    wandb_project: str = "YOUR_PROJECT"                                  # Name of WandB project

    seed: int = 195                                                      # Random seed (for reproducibility)
    randomize_seed: bool = False                                         # Whether to randomize the seed for sampling

    #################################################################################################################
    # Data collection parameters
    #################################################################################################################
    data_collection: bool = False                                        # If True, save policy rollouts for later offline use
    jpeg_compress: bool = True                                           # If True, apply JPEG compression to images before saving

    #################################################################################################################
    # Renderer retry parameters
    #################################################################################################################
    max_renderer_retries: int = 3                                        # Max number of retries if renderer produces black frames
    black_frame_threshold: float = 10.0                                   # Mean pixel intensity threshold below which a frame is considered black
    black_frame_ratio_threshold: float = 0.1                             # Fraction of frames that must be black to trigger a retry

    # fmt: on


def validate_config(cfg: PolicyEvalConfig) -> None:
    """Validate that the evaluation configuration is valid."""
    # Check that the task name is valid
    all_tasks = {**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}
    if cfg.task_name not in all_tasks:
        raise ValueError(
            f"Task name '{cfg.task_name}' not found in RoboCasa suite. Available tasks: {list(all_tasks.keys())}"
        )

    # Check that num_third_person_images is 2 (1 for left, 1 for right)
    assert cfg.num_third_person_images == 2, (
        f"Expecting `num_third_person_images` to be 2 (1 for left agentview, 1 for right agentview), "
        f"but got `num_third_person_images={cfg.num_third_person_images}`"
    )

    # Check that the dataset stats path is provided if action unnormalization or proprio normalization is enabled
    if (cfg.unnormalize_actions or cfg.normalize_proprio) and cfg.dataset_stats_path == "":
        raise ValueError(
            "Must provide `dataset_stats_path` when `unnormalize_actions=True` or `normalize_proprio=True`"
        )

    # Check parallel inference configuration
    if cfg.use_parallel_inference and cfg.num_queries_best_of_n <= 1:
        raise ValueError("Parallel inference is enabled but `num_queries_best_of_n <= 1`!")


def _detach_image_from_env_buffer(img: np.ndarray, *, flip_images: bool) -> np.ndarray:
    if flip_images:
        img = np.flipud(img)
    # Ensure the returned image is independent and has a standard positive-stride, contiguous layout.
    return np.ascontiguousarray(img).copy()


def is_image_mostly_black(img: np.ndarray, threshold: float = 5.0) -> bool:
    """Check if an image is mostly black (renderer failure indicator).

    Args:
        img: Image as numpy array with shape (H, W, C) or (H, W).
        threshold: Mean pixel intensity below which the image is considered black.

    Returns:
        True if the image is mostly black (mean intensity below threshold).
    """
    if img is None:
        return True
    return float(np.mean(img)) < threshold


def has_renderer_failed(
    replay_primary_images: list[np.ndarray],
    replay_secondary_images: list[np.ndarray],
    replay_wrist_images: list[np.ndarray],
    black_frame_threshold: float = 5.0,
    black_frame_ratio_threshold: float = 0.1,
) -> bool:
    """Check if the renderer has failed by detecting black frames in replay images.

    The renderer is considered to have failed if a significant fraction of frames
    across any of the camera views are mostly black.

    Args:
        replay_primary_images: List of primary camera images.
        replay_secondary_images: List of secondary camera images.
        replay_wrist_images: List of wrist camera images.
        black_frame_threshold: Mean pixel intensity threshold below which a frame is considered black.
        black_frame_ratio_threshold: Fraction of frames that must be black to consider renderer failed.

    Returns:
        True if renderer appears to have failed (too many black frames detected).
    """
    if not replay_primary_images and not replay_secondary_images and not replay_wrist_images:
        return False

    total_frames = 0
    black_frames = 0

    for images in [replay_primary_images, replay_secondary_images, replay_wrist_images]:
        for img in images:
            if img is not None:
                total_frames += 1
                if is_image_mostly_black(img, threshold=black_frame_threshold):
                    black_frames += 1

    if total_frames == 0:
        return False

    black_ratio = black_frames / total_frames
    return black_ratio >= black_frame_ratio_threshold


def prepare_observation(obs: dict, flip_images: bool = False) -> dict[str, np.ndarray]:
    """Prepare observations from environment for policy input.

    Returns:
        dict: Observation dictionary with keys:
            - "primary_image": Left third-person image (primary camera)
            - "secondary_image": Right third-person image
            - "wrist_image": Eye-in-hand wrist camera image
            - "proprio": Proprioceptive state (eef pose + gripper state)
    """
    # Extract images based on available cameras
    primary_img = None
    secondary_img = None
    wrist_img = None
    # RoboCasa has multiple camera views: left third-person, right third-person, wrist camera
    # We call the left third-person image the primary image and the right third-person image the secondary image
    if "robot0_agentview_left_image" in obs:
        img = obs["robot0_agentview_left_image"]
        primary_img = _detach_image_from_env_buffer(img, flip_images=flip_images)
    if "robot0_agentview_right_image" in obs:
        img = obs["robot0_agentview_right_image"]
        secondary_img = _detach_image_from_env_buffer(img, flip_images=flip_images)
    if "robot0_eye_in_hand_image" in obs:
        img = obs["robot0_eye_in_hand_image"]
        wrist_img = _detach_image_from_env_buffer(img, flip_images=flip_images)
    # Extract proprioceptive state
    proprio = np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]))
    # Prepare observations dict
    observation = {
        "primary_image": primary_img,
        "secondary_image": secondary_img,
        "wrist_image": wrist_img,
        "proprio": proprio,
    }
    return observation


def create_robocasa_env(cfg: PolicyEvalConfig, seed=None, episode_idx=None):
    """Create a RoboCasa environment.

    Args:
        cfg: Configuration object
        seed: Random seed for environment
        episode_idx: Episode index for deterministic scene selection (if None, uses all scenes)
    """
    # Parse layout_and_style_ids
    if cfg.layout_and_style_ids:
        all_layout_style_ids = ast.literal_eval(cfg.layout_and_style_ids)
        # Deterministically select one scene based on episode index
        # Episodes 0-9 use scene 0, episodes 10-19 use scene 1, etc.
        if episode_idx is not None:
            scene_index = (episode_idx // 10) % len(all_layout_style_ids)
            layout_and_style_ids = (all_layout_style_ids[scene_index],)
        else:
            # If no episode index provided, use all scenes (random selection by env)
            layout_and_style_ids = all_layout_style_ids
    else:
        layout_and_style_ids = None

    with open(CONTROLLER_CONFIGS_PATH, "rb") as pickle_file:
        controller_configs = pickle.load(pickle_file)
    # Create environment
    # We use the same args used in the official RoboCasa evals (robocasa/utils/eval_utils.py)
    env_kwargs = dict(
        env_name=cfg.task_name,
        robots=cfg.robots,
        controller_configs=controller_configs,
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_widths=cfg.env_img_res,
        camera_heights=cfg.env_img_res,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=cfg.obj_instance_split,
        generative_textures=None,
        randomize_cameras=cfg.randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )
    env = robosuite.make(**env_kwargs)
    return env, env_kwargs


def run_episode(
    cfg: PolicyEvalConfig,
    env,
    task_description: str,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    episode_idx: int,
    log_file=None,
):
    """Run a single evaluation episode."""
    # Wait for objects to stabilize
    NUM_STEPS_WAIT = 10
    for _ in range(NUM_STEPS_WAIT):
        dummy_action = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(dummy_action)
    # Get max steps for this task
    max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)
    # Important variables
    success = False
    episode_length = 0
    action_queue = deque()
    # Containers for episode replay images for saving videos
    replay_primary_images = []  # Left third-person camera
    replay_secondary_images = []  # Right third-person camera
    replay_wrist_images = []  # Wrist camera
    # Containers for episode data collection
    if cfg.data_collection:
        collected_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "success": [],
        }
        primary_images_list = []
        secondary_images_list = []
        wrist_images_list = []
        proprio_list = []
        actions_list = []
    # Best-of-N search variables
    future_image_predictions_list = []
    # Main episode loop
    for t in range(max_steps):
        observation = prepare_observation(obs, cfg.flip_images)
        # Store replay images for video saving
        replay_primary_images.append(observation["primary_image"])
        replay_secondary_images.append(observation["secondary_image"])
        replay_wrist_images.append(observation["wrist_image"])
        # Collect data if enabled
        if cfg.data_collection:
            primary_images_list.append(observation["primary_image"])
            secondary_images_list.append(observation["secondary_image"])
            wrist_images_list.append(observation["wrist_image"])
            proprio_list.append(observation["proprio"])
        # Query policy for new action chunk
        if len(action_queue) == 0:
            # Use parallel inference if enabled
            if cfg.use_parallel_inference and worker_pool and worker_pool.initialized:
                # Query model in parallel
                start_time = time.time()
                query_results = query_model_parallel(
                    cfg,
                    observation,
                    task_description,
                    worker_pool,
                    timeout=cfg.parallel_timeout,
                )
                total_query_time = time.time() - start_time

                log_message(
                    f"Parallel queries completed: {len(query_results)} results in {total_query_time:.3f}s", log_file
                )

            else:
                # Serial execution
                query_results = []
                for query_idx in range(cfg.num_queries_best_of_n):
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
                        seed=cfg.seed + query_idx,
                        randomize_seed=cfg.randomize_seed,
                        num_denoising_steps_action=cfg.num_denoising_steps_action,
                        generate_future_state_and_value_in_parallel=not (
                            cfg.ar_future_prediction or cfg.ar_value_prediction or cfg.ar_qvalue_prediction
                        ),
                    )
                    query_time = time.time() - start_time
                    log_message(
                        f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Action query time = {query_time:.3f} sec",
                        log_file,
                    )
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
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                            use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                            num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                            future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                        )
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Future state prediction query time = {query_time:.3f} sec",
                            log_file,
                        )
                        return_dict["future_image_predictions"] = future_state_return_dict["future_image_predictions"]
                        future_image_predictions_by_depth.append(return_dict["future_image_predictions"])

                    else:
                        return_dict["future_image_predictions"] = action_return_dict["future_image_predictions"]

                    if cfg.ar_value_prediction:
                        # Autoregressively query model to get value prediction
                        start_time = time.time()
                        value_return_dict = get_value_prediction(
                            cfg,
                            model=planning_model if planning_model is not None else model,
                            data_batch=action_return_dict["data_batch"],
                            future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_value=cfg.num_denoising_steps_value,
                            use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                            num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                        )
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Value prediction query time = {query_time:.3f} sec",
                            log_file,
                        )
                        return_dict["value_prediction"] = value_return_dict["value_prediction"]
                        value_predictions_by_depth.append(return_dict["value_prediction"])
                        log_message(
                            f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Value prediction: {return_dict['value_prediction']:.4f}",
                            log_file,
                        )
                    elif cfg.ar_qvalue_prediction:
                        # Autoregressively query model to get Q-value prediction
                        start_time = time.time()
                        value_return_dict = get_qvalue_prediction(
                            cfg,
                            model=planning_model if planning_model is not None else model,
                            data_batch=action_return_dict["data_batch"],
                            action_sample=action_return_dict["generated_latent"],
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_value=cfg.num_denoising_steps_value,
                            use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                            num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                        )
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Value prediction query time = {query_time:.3f} sec",
                            log_file,
                        )
                        return_dict["value_prediction"] = value_return_dict["value_prediction"]
                        value_predictions_by_depth.append(return_dict["value_prediction"])
                        log_message(
                            f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Value prediction: {return_dict['value_prediction']:.4f}",
                            log_file,
                        )
                    else:
                        return_dict["value_prediction"] = action_return_dict["value_prediction"]
                        value_predictions_by_depth.append(return_dict["value_prediction"])

                    if cfg.search_depth > 1:
                        assert not cfg.ar_qvalue_prediction, "Search depth > 1 not supported for Q-value prediction!"
                        for depth in range(2, cfg.search_depth + 1):
                            for future_state_latent in future_state_return_dict["future_state_samples_list"]:
                                next_generated_latent_with_future_state = future_state_latent.clone()
                                # Rearrange latent frames such that predicted future state replaces current state in the sequence
                                rearranged_next_latent_with_future_state = (
                                    next_generated_latent_with_future_state.clone()
                                )
                                rearranged_next_latent_with_future_state[
                                    :, :, CURR_STATE_START_LATENT_IDX : CURR_STATE_END_LATENT_IDX + 1
                                ] = next_generated_latent_with_future_state[
                                    :, :, FUTURE_STATE_START_LATENT_IDX : FUTURE_STATE_END_LATENT_IDX + 1
                                ]
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
                                    seed = cfg.seed + query_idx
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
                                        previous_generated_latent=rearranged_next_latent_with_future_state,  # Use future state sample since parts of value sample might be masked out
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
                                    future_image_latent_idx=action_return_dict["latent_indices"][
                                        "future_image_latent_idx"
                                    ],
                                    future_image2_latent_idx=action_return_dict["latent_indices"][
                                        "future_image2_latent_idx"
                                    ],
                                    seed=cfg.seed + query_idx,
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
                                    seed=cfg.seed + query_idx,
                                    randomize_seed=cfg.randomize_seed,
                                    num_denoising_steps_value=cfg.num_denoising_steps_value,
                                    use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                                    num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                                )
                                return_dict["value_prediction"] = value_return_dict["value_prediction"]
                                value_predictions_by_depth.append(return_dict["value_prediction"])
                                log_message(
                                    f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Value prediction: {return_dict['value_prediction']:.4f}",
                                    log_file,
                                )
                    # Add results to the return dict
                    return_dict["future_image_predictions_by_depth"] = future_image_predictions_by_depth
                    return_dict["value_predictions_by_depth"] = value_predictions_by_depth
                    return_dict["actions_by_depth"] = actions_by_depth
                    query_results.append(return_dict)

            # Replace value of each query with aggregate value over all value predictions
            # This is only applicable when search depth > 1 because otherwise the aggregation is handled in get_value_prediction()
            # For search depth == 1, the return dict contains the aggregated value prediction already
            if cfg.search_depth > 1:
                for query_idx, return_dict in enumerate(query_results):
                    if cfg.search_depth_value_aggregation_scheme == "average":
                        return_dict["value_prediction"] = np.mean(return_dict["value_predictions_by_depth"]).item()
                    elif cfg.search_depth_value_aggregation_scheme == "use_last_value":
                        return_dict["value_prediction"] = return_dict["value_predictions_by_depth"][-1]
                    else:
                        raise ValueError(
                            f"Invalid search depth value aggregation scheme: {cfg.search_depth_value_aggregation_scheme}"
                        )
            # Print all value predictions
            for query_idx, return_dict in enumerate(query_results):
                predicted_value = return_dict["value_prediction"]
                log_message(
                    f"Query {query_idx + 1}/{cfg.num_queries_best_of_n} (seed {cfg.seed + query_idx}): Predicted value = {predicted_value:.4f}",
                    log_file,
                )
            # Only keep the first num_open_loop_steps timesteps of the action chunk
            for query_idx, return_dict in enumerate(query_results):
                return_dict["actions"] = return_dict["actions"][: cfg.num_open_loop_steps]
            # Get dict: seed number -> (action chunk, future state, value)
            seed_to_return_dict = {
                cfg.seed + query_idx: (
                    return_dict["actions"],
                    return_dict["future_image_predictions"],
                    return_dict["value_prediction"],
                )
                for query_idx, return_dict in enumerate(query_results)
            }
            # Get seed with highest value
            best_seed, best_return_dict = max(seed_to_return_dict.items(), key=lambda x: x[1][2])
            best_actions = best_return_dict[0]
            best_future_predictions = best_return_dict[1]
            best_value_predictions = best_return_dict[2]
            # Use the best actions, future predictions, and value predictions found
            action_queue.extend(best_actions)
            future_image_predictions_list.append(best_future_predictions)
            log_message(f"t={t}: Selected seed {best_seed} with value = {best_value_predictions:.4f}", log_file)

        # Get next action from chunk
        action = action_queue.popleft()
        # RoboCasa: Policy was trained on 7-dim manipulation actions, but env expects 12-dim (7 + 5 mobile base)
        # Append [0, 0, 0, 0, -1] for mobile base since we're not using it
        if action.shape[-1] == 7 and env.action_dim == 12:
            mobile_base_action = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
            action = np.concatenate([action, mobile_base_action])
        # Execute action
        print(f"t: {t}, action: {action}")
        obs, reward, done, info = env.step(action)
        episode_length += 1
        # Collect action data if enabled
        if cfg.data_collection:
            actions_list.append(action)
        # Check for success
        if env._check_success():
            success = True
            log_message(f"  Success detected at timestep {t}!", log_file)
            break

    # Log episode result
    log_message(
        f"  Episode {episode_idx}: {'SUCCESS' if success else 'FAILURE'} (length: {episode_length})",
        log_file,
    )
    # Prepare collected data if enabled
    if cfg.data_collection:
        collected_data = dict(
            primary_images=np.stack(primary_images_list, axis=0),  # (T, H, W, C) - left camera
            secondary_images=np.stack(secondary_images_list, axis=0),  # (T, H, W, C) - right camera
            wrist_images=np.stack(wrist_images_list, axis=0),  # (T, H, W, C)
            proprio=np.stack(proprio_list, axis=0),  # (T, D)
            actions=np.stack(actions_list, axis=0),  # (T, action_dim)
            success=success,
        )
        # Add future image predictions
        if len(future_image_predictions_list) > 0:
            # Primary camera predictions (left third-person)
            if (
                "future_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_image"] is not None
            ):
                future_primary_images = [x["future_image"] for x in future_image_predictions_list]
                collected_data["future_primary_images"] = np.stack(future_primary_images, axis=0)
            # Secondary camera predictions (right third-person)
            if (
                "future_image2" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_image2"] is not None
            ):
                future_secondary_images = [x["future_image2"] for x in future_image_predictions_list]
                collected_data["future_secondary_images"] = np.stack(future_secondary_images, axis=0)
            # Wrist camera predictions
            if (
                "future_wrist_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_wrist_image"] is not None
            ):
                future_wrist_images = [x["future_wrist_image"] for x in future_image_predictions_list]
                collected_data["future_wrist_images"] = np.stack(future_wrist_images, axis=0)
    else:
        collected_data = None
    return (
        success,
        episode_length,
        replay_primary_images,
        replay_secondary_images,
        replay_wrist_images,
        future_image_predictions_list,
        collected_data,
    )


def run_task(
    cfg: PolicyEvalConfig,
    task_name: str,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    log_file=None,
):
    """Run evaluation for a single task."""
    log_message(f"\nEvaluating task: {task_name}", log_file)
    successes = []
    episode_lengths = []
    total_episodes = 0
    total_successes = 0
    for episode_idx in range(cfg.num_trials_per_task):
        log_message(f"Starting episode {episode_idx + 1}...", log_file)

        # Retry loop for renderer failures (black frames)
        for retry_attempt in range(cfg.max_renderer_retries):
            # Create environment with scene selection based on episode index
            # Episodes 0-9 use scene 0, 10-19 use scene 1, etc.
            if cfg.deterministic or cfg.deterministic_reset:
                # Deterministic seeding for reproducibility
                # Add retry_attempt to seed to get different randomization on retry
                seed = cfg.seed * episode_idx * 256 + retry_attempt
            else:
                seed = None
            env, env_kwargs = create_robocasa_env(cfg, seed=seed, episode_idx=episode_idx)
            # Reset environment
            # NOTE: Every reset changes the scene/task! So only reset ONCE per episode.
            if cfg.deterministic_reset:
                reset_seed = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
                set_seed_everywhere(reset_seed)
            env.reset()
            # Get task description
            # IMPORTANT: Get the task description AFTER resetting the environment. Resetting the environment changes the task!
            task_description = env.get_ep_meta()["lang"]
            log_message(f"\nTask description: {task_description}", log_file)
            # Run episode
            (
                success,
                length,
                replay_primary_images,
                replay_secondary_images,
                replay_wrist_images,
                future_image_predictions_list,
                collected_data,
            ) = run_episode(
                cfg,
                env,
                task_description,
                model,
                planning_model,
                dataset_stats,
                worker_pool,
                episode_idx,
                log_file,
            )

            # Check if renderer failed (produced black frames)
            renderer_failed = has_renderer_failed(
                replay_primary_images,
                replay_secondary_images,
                replay_wrist_images,
                black_frame_threshold=cfg.black_frame_threshold,
                black_frame_ratio_threshold=cfg.black_frame_ratio_threshold,
            )

            if renderer_failed:
                log_message(
                    f"WARNING: Renderer failure detected (black frames) on attempt {retry_attempt + 1}/{cfg.max_renderer_retries}. "
                    f"{'Retrying...' if retry_attempt < cfg.max_renderer_retries - 1 else 'Max retries reached, proceeding with last result.'}",
                    log_file,
                )
                # Close the environment before retrying
                env.close()
                if retry_attempt < cfg.max_renderer_retries - 1:
                    continue  # Retry
            else:
                # Success - no black frames detected
                if retry_attempt > 0:
                    log_message(f"Renderer succeeded on retry attempt {retry_attempt + 1}.", log_file)
                break  # Exit retry loop
        successes.append(success)
        episode_lengths.append(length)
        # Update counters
        total_episodes += 1
        if success:
            total_successes += 1
        # Save rollout video
        rollout_data_dir = os.path.join(cfg.local_log_dir, "rollout_data", f"{task_name}--{DATE_TIME}")
        os.makedirs(rollout_data_dir, exist_ok=True)
        save_rollout_video(
            replay_primary_images,
            replay_secondary_images,
            replay_wrist_images,
            episode_idx,
            success=success,
            task_description=task_description,
            rollout_data_dir=rollout_data_dir,
            log_file=log_file,
        )
        # Save rollout video with future image predictions
        if len(future_image_predictions_list) > 0:
            # Extract future predictions from the list
            future_primary_image_predictions = None
            future_secondary_image_predictions = None
            future_wrist_image_predictions = None
            if (
                "future_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_image"] is not None
            ):
                future_primary_image_predictions = [x["future_image"] for x in future_image_predictions_list]
            if (
                "future_image2" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_image2"] is not None
            ):
                future_secondary_image_predictions = [x["future_image2"] for x in future_image_predictions_list]
            if (
                "future_wrist_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_wrist_image"] is not None
            ):
                future_wrist_image_predictions = [x["future_wrist_image"] for x in future_image_predictions_list]
            # Save video with predictions if all three camera predictions are available
            if (
                future_primary_image_predictions is not None
                and future_secondary_image_predictions is not None
                and future_wrist_image_predictions is not None
            ):
                save_rollout_video_with_future_image_predictions(
                    replay_primary_images,
                    replay_secondary_images,
                    replay_wrist_images,
                    episode_idx,
                    success=success,
                    task_description=task_description,
                    rollout_data_dir=rollout_data_dir,
                    chunk_size=cfg.chunk_size,
                    num_open_loop_steps=cfg.num_open_loop_steps,
                    future_primary_image_predictions=future_primary_image_predictions,
                    future_secondary_image_predictions=future_secondary_image_predictions,
                    future_wrist_image_predictions=future_wrist_image_predictions,
                    show_diff=False,
                    log_file=log_file,
                    show_timestep=True,
                )
            else:
                log_message(
                    f"Skipping video with future predictions - not all camera predictions available "
                    f"(primary: {future_primary_image_predictions is not None}, "
                    f"secondary: {future_secondary_image_predictions is not None}, "
                    f"wrist: {future_wrist_image_predictions is not None})",
                    log_file,
                )
        # Save collected data if data_collection is enabled
        if cfg.data_collection and collected_data is not None:
            # Skip episodes that are less than 5 timesteps long (because sometimes success is detected immediately upon starting since the envs are buggy)
            if len(collected_data["actions"]) < 5:
                log_message(
                    f"Skipping saving this episode: less than 5 timesteps long (only {len(collected_data['actions'])} timesteps).",
                    log_file,
                )
                continue
            # Save episodic HDF5 data
            processed_task_description = (
                task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:35]
            )
            ep_filename = f"{DATE_TIME}--episode_data--task={processed_task_description}--ep={episode_idx}--success={success}.hdf5"
            ep_filepath = os.path.join(rollout_data_dir, ep_filename)
            with h5py.File(ep_filepath, "w") as f:
                for k, v in collected_data.items():
                    if isinstance(v, np.ndarray):
                        is_image = v.ndim == 4 and v.shape[-1] == 3 and v.dtype == np.uint8
                        if is_image and cfg.jpeg_compress:
                            jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in v]
                            if len(jpeg_list) == 1:
                                # Skip saving the array if it only has one element (causes error during create_dataset())
                                continue
                            dt = h5py.vlen_dtype(np.dtype("uint8"))
                            f.create_dataset(k + "_jpeg", data=jpeg_list, dtype=dt)
                        else:
                            f.create_dataset(k, data=v)
                    else:
                        f.attrs[k] = v
                f.attrs["task_description"] = task_description
            log_message(f"Saved episode data to: {ep_filepath}", log_file)
        # Close environment after each episode
        env.close()
        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)
    # Calculate statistics
    success_rate = np.mean(successes)
    avg_length = np.mean(episode_lengths)
    log_message(f"Task {task_name} results:", log_file)
    log_message(f"  Success rate: {success_rate:.4f} ({int(success_rate * 100)}%)", log_file)
    log_message(f"  Average episode length: {avg_length:.1f}", log_file)
    log_message(
        f"  Successes: {sum(successes)}/{len(successes)}",
        log_file,
    )
    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_name}": success_rate,
                f"avg_episode_length/{task_name}": avg_length,
                f"num_successes/{task_name}": sum(successes),
                f"num_episodes/{task_name}": len(successes),
            }
        )
    return success_rate, avg_length, successes


@draccus.wrap()
def eval_robocasa(cfg: PolicyEvalConfig) -> float:
    """Main function to evaluate a trained policy on RoboCasa tasks."""
    # Set DETERMINISTIC environment variable if on deterministic mode
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    # Set random seed
    set_seed_everywhere(cfg.seed)
    # Set multiprocessing start method if using parallel inference
    if cfg.use_parallel_inference:
        mp.set_start_method("spawn", force=True)
    # Validate evaluation configuration
    validate_config(cfg)
    # Initialize T5 text embeddings cache
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path, embeddings_kind=cfg.text_embeddings_kind)
    # Load Cosmos Policy dataset stats
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    # If using parallel inference, initialize worker pool
    worker_pool = None
    if cfg.use_parallel_inference:
        available_gpus = [int(gpu.strip()) for gpu in cfg.available_gpus.split(",")]
        available_gpus = available_gpus[: cfg.num_queries_best_of_n]  # Only need N parallel workers
        worker_pool = WorkerPoolManager(cfg, dataset_stats, available_gpus)
        worker_pool.start_workers()
        # Set model to None here because each worker will load its own copy
        model = None
        planning_model = None
    # If using serial inference, initialize model and Cosmos config
    else:
        model, cosmos_config = get_model(cfg)
        assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
            f"Mismatch found between train and test chunk sizes! Train: {cosmos_config.dataloader_train.dataset.chunk_size}, Test: {cfg.chunk_size}"
        )
        worker_pool = None
        # Initialize planning model if specified
        if cfg.planning_model_ckpt_path != "":
            planning_model, _ = get_planning_model(cfg)
        else:
            planning_model = None
    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(
        cfg=cfg,
        task_identifier=cfg.task_name,
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    log_message(f"Eval config: {cfg}", log_file)
    # Log parallel inference configuration if enabled
    if cfg.use_parallel_inference and worker_pool:
        available_gpus = [int(gpu.strip()) for gpu in cfg.available_gpus.split(",")]
        log_message(f"Parallel inference enabled on GPUs: {available_gpus}", log_file)
        log_message(f"Parallel timeout: {cfg.parallel_timeout}s", log_file)
    # Run evaluation
    log_message(f"\nStarting evaluation for task: {cfg.task_name}", log_file)
    log_message(f"Number of trials: {cfg.num_trials_per_task}", log_file)
    success_rate, avg_length, successes = run_task(
        cfg,
        cfg.task_name,
        model,
        planning_model,
        dataset_stats,
        worker_pool,
        log_file,
    )
    # Log final results
    log_message("\n" + "=" * 80, log_file)
    log_message("FINAL RESULTS", log_file)
    log_message("=" * 80, log_file)
    log_message(f"Task: {cfg.task_name}", log_file)
    log_message(f"Success rate: {success_rate:.4f} ({int(success_rate * 100)}%)", log_file)
    log_message(f"Average episode length: {avg_length:.1f}", log_file)
    log_message(f"Total episodes: {len(successes)}", log_file)
    log_message(f"Total successes: {sum(successes)}", log_file)
    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "final_success_rate": success_rate,
                "final_avg_episode_length": avg_length,
                "total_episodes": len(successes),
                "total_successes": sum(successes),
            }
        )
        wandb.save(local_log_filepath)
        wandb.finish()
    # Cleanup
    if worker_pool:
        worker_pool.shutdown()
    log_message(f"\nResults saved to: {local_log_filepath}", log_file)
    return success_rate


if __name__ == "__main__":
    eval_robocasa()
