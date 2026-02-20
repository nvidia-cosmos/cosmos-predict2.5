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
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.

Adapted from: https://github.com/user/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py

Parallel Inference:
    To enable parallel inference across multiple GPUs, use:
        --use_parallel_inference True
        --available_gpus "0,1,2,3"
        --num_queries_best_of_n 4

    This will run model queries in parallel across the specified GPUs using torch.multiprocessing, which can
    significantly speed up evaluation when using value functions that require multiple queries per action.

    Requirements:
    - Multiple GPUs must be available
    - CUDA must be properly configured
    - Sufficient GPU memory for multiple model copies

    Note: Uses torch.multiprocessing with 'spawn' start method for CUDA compatibility.

Usage examples:
    # *** Main checkpoint: 98.5% success rate ***
    #   Replace `task_suite_name` with one of {libero_spatial, libero_object, libero_goal, libero_10}
    #   Replace `seed` with one of {195, 196, 197}
    #   Replace `run_id_note` with a unique identifier for the run
    uv run -m cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.libero.run_libero_eval \
        --config cosmos_predict2_2b_480p_libero__inference_only \
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --config_file cosmos_predict2/_src/predict2/cosmos_policy/config/config.py \
        --use_wrist_image True \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 16 \
        --num_open_loop_steps 16 \
        --task_suite_name libero_10 \
        --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
        --randomize_seed False \
        --data_collection False \
        --available_gpus "0,1,2,3,4,5,6,7" \
        --seed 195 \
        --use_variance_scale False \
        --deterministic True \
        --run_id_note chkpt45000--5stepAct--seed195--deterministic \
        --ar_future_prediction False \
        --ar_value_prediction False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1
    # Same as above, but with deterministic reset (seed=195/196/197, reset seed=0)
    # Also gets 98.5% success rate
    uv run -m cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.libero.run_libero_eval \
        --config cosmos_predict2_2b_480p_libero__inference_only \
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --config_file cosmos_predict2/_src/predict2/cosmos_policy/config/config.py \
        --use_wrist_image True \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 16 \
        --num_open_loop_steps 16 \
        --task_suite_name libero_10 \
        --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
        --randomize_seed False \
        --data_collection False \
        --available_gpus "0,1,2,3,4,5,6,7" \
        --seed 195 \
        --use_variance_scale False \
        --deterministic True \
        --run_id_note chkpt45000--5stepAct--seed195--deterministicand_deterministicResetSeed0 \
        --ar_future_prediction False \
        --ar_value_prediction False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --deterministic_reset True \
        --deterministic_reset_seed 0

"""

import json
import logging
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import draccus
import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
import tqdm
import wandb
from libero.libero import benchmark

from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
    WorkerPoolManager,
    get_action,
    get_future_state_prediction,
    get_model,
    get_planning_model,
    get_qvalue_prediction,
    get_value_prediction,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    query_model_parallel,
)
from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    save_rollout_video,
    save_rollout_video_with_future_image_predictions,
)
from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    log_message,
    setup_logging,
)
from cosmos_predict2._src.predict2.cosmos_policy.utils.utils import jpeg_encode_image, set_seed_everywhere

# Cosmos Policy latent sequence indices
# 0: blank, 1: curr proprio, 2: curr wrist img, 3: curr primary img, 4: action, 5: future proprio, 6: future wrist img, 7: future primary img, 8: value
CURR_STATE_START_LATENT_IDX, CURR_STATE_END_LATENT_IDX = 1, 3
FUTURE_STATE_START_LATENT_IDX, FUTURE_STATE_END_LATENT_IDX = 5, 7


def extract_ckpt_config_name(ckpt_path: str) -> str | None:
    """
    Extract the config name from the checkpoint path.

    The config name is the directory name right before "checkpoints/iter_xxxx".
    For example:
        s3://bucket/.../cosmos_predict2p5_2b_480p_libero__cond_frame_ts-1/checkpoints/iter_000002000/
        -> cosmos_predict2p5_2b_480p_libero__cond_frame_ts-1

    Args:
        ckpt_path: The checkpoint path.

    Returns:
        The config name, or None if it cannot be extracted.
    """
    import re

    # Normalize the path (remove trailing slashes)
    ckpt_path = ckpt_path.rstrip("/")

    # Match pattern: .../config_name/checkpoints/iter_xxxx
    match = re.search(r"/([^/]+)/checkpoints/iter_\d+$", ckpt_path)
    if match:
        return match.group(1)

    return None


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


@dataclass
class PolicyEvalConfig:
    # fmt: off
    suite: str = "libero"                                                # Evaluation suite name

    #################################################################################################################
    # Cosmos Policy-specific parameters
    #################################################################################################################
    model_family: str = "cosmos"                                         # Model family
    config: str = ""                                                     # Inference config name
    ckpt_path: str = ""                                                  # Pretrained checkpoint path
    planning_model_config_name: str = ""                                 # Planning model config name
    planning_model_ckpt_path: str = ""                                   # Planning model checkpoint path
    config_file: str = "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py"  # Cosmos default config file path

    use_third_person_image: bool = True                                  # Whether to include primary (third-person) image in input
    num_third_person_images: int = 1                                     # Number of third-person images to include in input (LIBERO: 1 agentview image)
    use_wrist_image: bool = True                                         # Whether to include wrist image in input
    num_wrist_images: int = 1                                            # Number of wrist images to include in input (LIBERO: 1 wrist image)
    use_proprio: bool = True                                             # Whether to include proprio state in input
    flip_images: bool = True                                             # Whether to flip images vertically across x-axis
    use_variance_scale: bool = False                                     # Whether to scale variance used to sample sigma max for denoising for increased diversity in generations
    use_jpeg_compression: bool = True                                    # Whether to use JPEG compression on images before querying policy
    ar_future_prediction: bool = False                                   # Whether to predict future state autoregressively
    ar_value_prediction: bool = False                                    # Whether to predict future state value autoregressively
    ar_qvalue_prediction: bool = False                                   # Whether to predict Q-value autoregressively
    num_denoising_steps_action: int = 5                                  # Number of denoising steps to take for action prediction
    num_denoising_steps_future_state: int = 1                            # Number of denoising steps to take for future state prediction (only applicable if ar_future_prediction is True; otherwise equal to num_denoising_steps_action)
    num_denoising_steps_value: int = 1                                   # Number of denoising steps to take for value prediction (only applicable if ar_value_prediction is True; otherwise equal to num_denoising_steps_action)
    shift: int = 5                                              # Shift parameter for rectified flow scheduler timestep distribution
    unnormalize_actions: bool = True                                     # Unnormalize actions if trained with normalized actions
    normalize_proprio: bool = True                                       # Normalize proprio input if trained with normalized proprio
    dataset_stats_path: str = ""                                         # Path to dataset statistics file for action unnormalization and proprio normalization
    t5_text_embeddings_path: str = ""                                    # Path to precomputed T5 text embeddings dictionary (key: instruction, val: embedding)
    text_embeddings_kind: str = "t5"                                     # Type of text embeddings: "t5" or "reason1"
    trained_with_image_aug: bool = True                                  # Whether the model was trained with image augmentations (needed for test-time image transformations)
    chunk_size: int = 16                                                 # Number of actions to predict in chunk
    num_open_loop_steps: int = 16                                        # Number of actions in predicted chunk to execute open-loop before requerying policy

    deterministic: bool = True                                           # Whether to run in deterministic mode
    deterministic_reset: bool = False                                    # Whether to run in deterministic reset mode (sets global random seed right before env reset)
    deterministic_reset_seed: int = None                                 # (Only applicable if deterministic_reset==True) The seed to set before deterministic reset; if not provided, defaults to the base seed

    #################################################################################################################
    # Planning model and best-of-N search parameters
    #################################################################################################################
    use_ensemble_future_state_predictions: bool = False                  # Whether to use ensemble of future state predictions
    num_future_state_predictions_in_ensemble: int = 3                    # Number of future state predictions in ensemble
    future_state_ensemble_aggregation_scheme: str = "average"            # How to aggregate future state predictions in an ensemble of future state predictions (options: "average", "first")
    use_ensemble_value_predictions: bool = False                         # Whether to use ensemble of value predictions
    num_value_predictions_in_ensemble: int = 5                           # Number of value predictions in ensemble
    value_ensemble_aggregation_scheme: str = "average"                   # How to aggregate values in an ensemble of value predictions (options: "average", "gamma_weighted_average", "lcb", "success_vote", "majority_mean")
    search_depth: int = 1                                                # Number of levels to search through in the best-of-N search tree
    mask_current_state_action_for_value_prediction: bool = False         # Whether to use input masking to mask out certain inputs (current state and action) during value prediction
    mask_future_state_for_qvalue_prediction: bool = False                # Whether to use input masking to mask out certain inputs (future state) during Q(s, a) value prediction

    num_queries_best_of_n: int = 1                                       # Number of queries to make to the model (this is the N in best-of-N search)
    use_parallel_inference: bool = False                                 # Whether to use parallel inference across multiple GPUs
    available_gpus: str = "0,1,2,3,4,5,6,7"                              # Comma-separated list of GPU IDs available for use for parallel inference (defaults to all 8 GPUs on a node)
    parallel_timeout: int = 15                                           # Timeout in seconds for each parallel query

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL                      # Task suite (must be one of: LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_10, LIBERO_90)
    num_trials_per_task: int = 50                                        # Number of rollouts per task
    initial_states_path: str = "DEFAULT"                                 # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                                               # Resolution for rendering environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    local_log_dir: str = "./experiments/logs"                            # Local directory for eval logs
    run_id_note: Optional[str] = None                                    # Extra note to add to end of run ID for logging

    use_wandb: bool = False                                              # Whether to also log results in Weights & Biases
    wandb_entity: str = "YOUR_ENTITY"                                    # Name of WandB entity
    wandb_project: str = "YOUR_PROJECT"                                  # Name of WandB project

    seed: int = 7                                                        # Random seed (for reproducibility)
    randomize_seed: bool = False                                         # Whether to randomize the seed for sampling

    #################################################################################################################
    # Data collection parameters
    #################################################################################################################
    data_collection: bool = False                                        # If True, save episodic data for later offline use
    jpeg_compress: bool = True                                           # If True, apply JPEG compression to images before saving

    # fmt: on


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def validate_config(cfg: PolicyEvalConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.ckpt_path is not None, "ckpt_path must not be None!"

    if "image_aug" in str(cfg.ckpt_path):
        assert cfg.trained_with_image_aug, (
            "Expecting `trained_with_image_aug==True` because model was trained with image augmentations!"
        )

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def check_unnorm_key(cfg: PolicyEvalConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in Cosmos Policy `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def load_initial_states(cfg: PolicyEvalConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size, flip_images: bool = False):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs, flip_images)
    wrist_img = get_libero_wrist_image(obs, flip_images)

    # Prepare observations dict
    observation = {
        "primary_image": img,
        "wrist_image": wrist_img,
        "proprio": np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])),
    }

    return observation  # Return processed observation


def run_episode(
    cfg: PolicyEvalConfig,
    env,
    task_description: str,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    resize_size,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    if cfg.deterministic_reset:
        reset_seed = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
        set_seed_everywhere(reset_seed)
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != cfg.chunk_size:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match cfg.chunk_size "
            f"{cfg.chunk_size}! For best performance (in terms of both speed and success rate), we "
            "recommend executing the full action chunk."
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    replay_wrist_images = [] if cfg.use_wrist_image else None
    future_image_predictions_list = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Best-of-N search variables
    base_seed = cfg.seed  # Used for seed switching (if applicable)

    # Data collection buffers
    if cfg.data_collection:
        primary_images_list = []
        wrist_images_list = []
        proprio_list = []
        actions_list = []

    # Run episode
    success = False
    try:
        NUM_STEPS_WAIT = 10
        while t < max_steps + NUM_STEPS_WAIT:
            # If the deterministic flag is set, reset the random state with the same seed in every step
            if os.environ.get("DETERMINISTIC", "").lower() == "true":
                seed = 0
                set_seed_everywhere(seed)

            # Do nothing for the first few timesteps to let objects stabilize
            if t < NUM_STEPS_WAIT:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation = prepare_observation(obs, resize_size, cfg.flip_images)
            replay_images.append(observation["primary_image"])
            if replay_wrist_images is not None:
                replay_wrist_images.append(observation["wrist_image"])

            if cfg.data_collection:
                primary_images_list.append(observation["primary_image"])
                wrist_images_list.append(observation["wrist_image"])
                proprio_list.append(observation["proprio"])

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                best_actions = None
                best_future_predictions = None

                # Query model multiple times if value functions are available
                num_queries = cfg.num_queries_best_of_n

                # Use parallel inference if enabled and multiple queries are needed
                if cfg.use_parallel_inference and num_queries > 1 and worker_pool and worker_pool.initialized:
                    # Query model in parallel
                    start_time = time.time()
                    query_results = query_model_parallel(
                        cfg, observation, task_description, worker_pool, cfg.parallel_timeout
                    )
                    total_query_time = time.time() - start_time

                    log_message(
                        f"Parallel queries completed: {len(query_results)} results in {total_query_time:.3f}s", log_file
                    )

                else:
                    # Serial execution (original behavior)
                    query_results = []
                    for query_idx in range(num_queries):
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
                            f"Query {query_idx + 1}/{num_queries}: Action query time = {query_time:.3f} sec", log_file
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
                                seed=cfg.seed + query_idx,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                                use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                                num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                                future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                            )
                            query_time = time.time() - start_time
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Future state prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["future_image_predictions"] = future_state_return_dict[
                                "future_image_predictions"
                            ]
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
                                f"Query {query_idx + 1}/{num_queries}: Value prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["value_prediction"] = value_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction: {return_dict['value_prediction']:.4f}",
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
                                f"Query {query_idx + 1}/{num_queries}: Value prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["value_prediction"] = value_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction: {return_dict['value_prediction']:.4f}",
                                log_file,
                            )
                        else:
                            return_dict["value_prediction"] = action_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])

                        return_dict["future_image_predictions_by_depth"] = future_image_predictions_by_depth
                        return_dict["value_predictions_by_depth"] = value_predictions_by_depth
                        return_dict["actions_by_depth"] = actions_by_depth
                        query_results.append(return_dict)

                # Print all value predictions
                log_message(f"t={t}: Current base seed: {base_seed}", log_file)
                for query_idx, return_dict in enumerate(query_results):
                    predicted_value = return_dict["value_prediction"]
                    log_message(
                        f"Query {query_idx + 1}/{num_queries} (seed {cfg.seed + query_idx}): Predicted value = {predicted_value:.4f}",
                        log_file,
                    )
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

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            print(f"t: {t}\t action: {action}")

            if cfg.data_collection:
                actions_list.append(action.copy())

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        error_msg = f"Episode error: {e}"
        traceback_str = traceback.format_exc()
        log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)

    # Fill data collection buffers
    if cfg.data_collection:
        collected_data = dict(
            primary_images=np.stack(primary_images_list, axis=0),  # (T, H, W, C)
            wrist_images=np.stack(wrist_images_list, axis=0),  # (T, H, W, C)
            proprio=np.stack(proprio_list, axis=0),  # (T, D)
            actions=np.stack(actions_list, axis=0),  # (T, action_dim)
            success=success,
        )
        # Add future image predictions if available
        if len(future_image_predictions_list) > 0:
            if cfg.use_third_person_image:
                future_primary_images = [
                    x["future_image"] for x in future_image_predictions_list if x["future_image"] is not None
                ]
                if len(future_primary_images) > 0:
                    collected_data["future_primary_images"] = np.stack(future_primary_images, axis=0)
            # Wrist image predictions (may be None depending on config)
            if (
                cfg.use_wrist_image
                and "future_wrist_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_wrist_image"] is not None
            ):
                future_wrist_images = [x["future_wrist_image"] for x in future_image_predictions_list]
                collected_data["future_wrist_images"] = np.stack(future_wrist_images, axis=0)
    else:
        collected_data = None

    return success, replay_images, replay_wrist_images, future_image_predictions_list, collected_data


def run_task(
    cfg: PolicyEvalConfig,
    task_suite,
    task_id: int,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    resize_size,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    ckpt_config_name: str | None = None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, replay_wrist_images, future_image_predictions_list, collected_data = run_episode(
            cfg,
            env,
            task_description,
            model,
            planning_model,
            dataset_stats,
            worker_pool,
            resize_size,
            initial_state,
            log_file,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images,
            total_episodes,
            success=success,
            task_description=task_description,
            log_file=log_file,
            ckpt_config_name=ckpt_config_name,
        )

        # Save replay video with future image predictions included
        future_primary_image_predictions = None
        if cfg.use_third_person_image:
            future_primary_image_predictions = [x["future_image"] for x in future_image_predictions_list]
        future_wrist_image_predictions = None
        if cfg.use_wrist_image:
            future_wrist_image_predictions = [x["future_wrist_image"] for x in future_image_predictions_list]
        save_rollout_video_with_future_image_predictions(
            replay_images,
            total_episodes,
            success=success,
            task_description=task_description,
            chunk_size=cfg.chunk_size,
            num_open_loop_steps=cfg.num_open_loop_steps,
            rollout_wrist_images=replay_wrist_images,
            future_primary_image_predictions=future_primary_image_predictions,
            future_wrist_image_predictions=future_wrist_image_predictions,
            log_file=log_file,
            show_diff=False,
            ckpt_config_name=ckpt_config_name,
        )

        # Save episodic data (in data collection mode)
        if cfg.data_collection and collected_data is not None:

            def _save_episode_data():
                """Save collected episode data to HDF5 file."""
                ep_filename = f"episode_data--suite={cfg.task_suite_name}--{DATE_TIME}--task={task_id}--ep={total_episodes}--success={success}--{cfg.run_id_note}.hdf5"
                rollout_data_dir = os.path.join(cfg.local_log_dir, "rollout_data")
                os.makedirs(rollout_data_dir, exist_ok=True)
                ep_filepath = os.path.join(rollout_data_dir, ep_filename)
                with h5py.File(ep_filepath, "w") as f:
                    for k, v in collected_data.items():
                        if isinstance(v, np.ndarray):
                            is_image = v.ndim == 4 and v.shape[-1] == 3 and v.dtype == np.uint8
                            if is_image and cfg.jpeg_compress:
                                jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in v]
                                dt = h5py.vlen_dtype(np.dtype("uint8"))
                                f.create_dataset(k + "_jpeg", data=jpeg_list, dtype=dt)
                            else:
                                f.create_dataset(k, data=v)
                        else:
                            f.attrs[k] = v
                    f.attrs["task_description"] = task_description

            _save_episode_data()

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{cfg.task_suite_name}/{task_description}": task_success_rate,
                f"num_episodes/{cfg.task_suite_name}/{task_description}": task_episodes,
                f"num_successes/{cfg.task_suite_name}/{task_description}": task_successes,
            },
        )

    return (
        total_episodes,
        total_successes,
    )


@draccus.wrap()
def eval_libero(cfg: PolicyEvalConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""

    # Set DETERMINISTIC environment variable if on deterministic mode (makes some model operations deterministic)
    assert not (cfg.deterministic and cfg.randomize_seed), (
        "Cannot enable both deterministic mode and randomize seed mode!"
    )
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"

    # Set multiprocessing start method if using parallel inference
    if cfg.use_parallel_inference:
        mp.set_start_method("spawn", force=True)

    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize T5 text embeddings cache
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    # Load Cosmos Policy dataset stats
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    # If using parallel inference with multiple queries, initialize worker pool
    worker_pool = None
    if cfg.use_parallel_inference and cfg.num_queries_best_of_n > 1:
        available_gpus = [int(gpu.strip()) for gpu in cfg.available_gpus.split(",")]
        available_gpus = available_gpus[: cfg.num_queries_best_of_n]  # Only need N parallel workers
        worker_pool = WorkerPoolManager(cfg, dataset_stats, available_gpus)
        model = None
        planning_model = None

    # If using serial inference (or parallel with only 1 query), initialize model and Cosmos config
    else:
        model, cosmos_config = get_model(cfg)
        assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
            f"Mismatch found between train and test chunk sizes! Train: {cosmos_config.dataloader_train.dataset.chunk_size}, Test: {cfg.chunk_size}"
        )
        worker_pool = None

        # Initialize model for world model and value function
        if cfg.planning_model_ckpt_path != "":
            planning_model, _ = get_planning_model(cfg)
        else:
            planning_model = None

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg.model_family)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(
        cfg=cfg,
        task_identifier=cfg.task_suite_name,
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    log_message(f"Eval config: {cfg}", log_file)

    # Log parallel inference configuration and start worker pool
    if cfg.use_parallel_inference and worker_pool:
        log_message(f"Parallel inference enabled on GPUs: {available_gpus}", log_file)
        log_message(f"Parallel timeout: {cfg.parallel_timeout}s", log_file)
        log_message(f"Multiprocessing start method: {mp.get_start_method()}", log_file)

        # Verify GPUs are available
        for gpu_id in available_gpus:
            if gpu_id >= torch.cuda.device_count():
                log_message(
                    f"Warning: GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs found)", log_file
                )

        # Start worker pool
        try:
            log_message("Starting worker pool...", log_file)
            worker_pool.start_workers()
            log_message("Worker pool started successfully", log_file)
        except Exception as e:
            error_msg = f"Failed to start worker pool: {e}"
            traceback_str = traceback.format_exc()
            log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)
            log_message("Disabling parallel inference for this run", log_file)
            worker_pool = None
    else:
        log_message("Using serial inference (parallel inference disabled)", log_file)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Number of tasks: {num_tasks}", log_file)

    # Extract checkpoint config name from ckpt_path
    ckpt_config_name = extract_ckpt_config_name(cfg.ckpt_path)
    if ckpt_config_name:
        log_message(f"Checkpoint config name: {ckpt_config_name}", log_file)
    else:
        log_message("Could not extract checkpoint config name from ckpt_path", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        (
            total_episodes,
            total_successes,
        ) = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            planning_model,
            dataset_stats,
            worker_pool,
            resize_size,
            total_episodes,
            total_successes,
            log_file,
            ckpt_config_name,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{cfg.task_suite_name}/total": final_success_rate,
                f"num_episodes/{cfg.task_suite_name}/total": total_episodes,
                f"num_successes/{cfg.task_suite_name}/total": total_successes,
            },
        )
        wandb.save(local_log_filepath)

    # Cleanup worker pool
    if worker_pool:
        try:
            worker_pool.shutdown()
        except Exception as e:
            error_msg = f"Error shutting down worker pool: {e}"
            traceback_str = traceback.format_exc()
            log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
