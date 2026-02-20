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
deploy.py

Starts policy server which the client can query to get robot actions.

Setup:
    pip install draccus json_numpy uvicorn fastapi

Usage examples:

    # *** Main checkpoint: Base Cosmos Policy only ***
    uv run -m cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.aloha.deploy \
        --config cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only \
        --ckpt_path nvidia/Cosmos-Policy-ALOHA-Predict2-2B \
        --config_file cosmos_predict2/_src/predict2/cosmos_policy/config/config.py \
        --use_third_person_image True \
        --use_wrist_image True \
        --num_wrist_images 2 \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-ALOHA-Predict2-2B/aloha_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-ALOHA-Predict2-2B/aloha_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 50 \
        --num_open_loop_steps 50 \
        --ar_future_prediction False \
        --ar_value_prediction False \
        --use_jpeg_compression False \
        --flip_images False \
        --num_denoising_steps_action 10 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --deterministic True \
        --seed 195

    # Cosmos Policy + model-based planning w/ V(s') value function (main planning config)
    # Policy model: Same as main checkpoint
    # Planning model: Checkpoint fine-tuned on 648 rollouts from Cosmos Policy, pi05, pi0, OpenVLA-OFT+, Diffusion Policy
    #   V(s') value function
    #   Ensembling: 3 future state predictions, 5 value predictions
    uv run -m cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.aloha.deploy \
        --config cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only \
        --ckpt_path nvidia/Cosmos-Policy-ALOHA-Predict2-2B \
        --planning_model_config_name cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only \
        --planning_model_ckpt_path nvidia/Cosmos-Policy-ALOHA-Planning-Model-Predict2-2B \
        --config_file cosmos_predict2/_src/predict2/cosmos_policy/config/config.py \
        --use_third_person_image True \
        --use_wrist_image True \
        --num_wrist_images 2 \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-ALOHA-Predict2-2B/aloha_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-ALOHA-Predict2-2B/aloha_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 50 \
        --num_open_loop_steps 50 \
        --ar_future_prediction True \
        --ar_value_prediction True \
        --ar_qvalue_prediction False \
        --use_jpeg_compression False \
        --flip_images False \
        --num_denoising_steps_action 10 \
        --num_denoising_steps_future_state 5 \
        --num_denoising_steps_value 5 \
        --deterministic True \
        --seed 195 \
        --search_depth 1 \
        --value_ensemble_aggregation_scheme majority_mean \
        --randomize_seed False \
        --num_queries_best_of_n 8 \
        --use_parallel_inference True \
        --use_ensemble_value_predictions True \
        --num_value_predictions_in_ensemble 5 \
        --use_ensemble_future_state_predictions True \
        --num_future_state_predictions_in_ensemble 3 \
        --future_state_ensemble_aggregation_scheme "average" \
        --mask_current_state_action_for_value_prediction True

    # Same as above (Cosmos Policy + model-based planning w/ V(s')) except:
    #  Search depth = 3
    #  No future state / value ensembles
    uv run -m cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.aloha.deploy \
        --config cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only \
        --ckpt_path nvidia/Cosmos-Policy-ALOHA-Predict2-2B \
        --planning_model_config_name cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only \
        --planning_model_ckpt_path nvidia/Cosmos-Policy-ALOHA-Planning-Model-Predict2-2B \
        --config_file cosmos_predict2/_src/predict2/cosmos_policy/config/config.py \
        --use_third_person_image True \
        --use_wrist_image True \
        --num_wrist_images 2 \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-ALOHA-Predict2-2B/aloha_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-ALOHA-Predict2-2B/aloha_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 50 \
        --num_open_loop_steps 50 \
        --ar_future_prediction True \
        --ar_value_prediction True \
        --ar_qvalue_prediction False \
        --use_jpeg_compression False \
        --flip_images False \
        --num_denoising_steps_action 10 \
        --num_denoising_steps_future_state 5 \
        --num_denoising_steps_value 5 \
        --deterministic True \
        --seed 195 \
        --search_depth 3 \
        --search_depth_value_aggregation_scheme use_last_value \
        --randomize_seed False \
        --num_queries_best_of_n 8 \
        --use_parallel_inference True \
        --use_ensemble_value_predictions False \
        --num_value_predictions_in_ensemble 1 \
        --value_ensemble_aggregation_scheme average \
        --use_ensemble_future_state_predictions False \
        --num_future_state_predictions_in_ensemble 1 \
        --future_state_ensemble_aggregation_scheme "average" \
        --mask_current_state_action_for_value_prediction True

"""

import json
import logging
import os
import secrets
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import draccus
import json_numpy
import numpy as np
import torch
import torch.multiprocessing as mp
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

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
from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.robot_utils import get_image_resize_size
from cosmos_predict2._src.predict2.cosmos_policy.utils.utils import set_seed_everywhere

json_numpy.patch()

# Cosmos Policy latent sequence indices
# 0: blank, 1: curr proprio, 2: curr left wrist img, 3: curr right wrist img, 4: curr primary img, 5: action, 6: future proprio, 7: future left wrist img, 8: future right wrist img, 9: future primary img, 10: value
CURR_STATE_START_LATENT_IDX, CURR_STATE_END_LATENT_IDX = 1, 4
FUTURE_STATE_START_LATENT_IDX, FUTURE_STATE_END_LATENT_IDX = 6, 9


@dataclass
class DeployConfig:
    # fmt: off
    suite: str = "aloha"                               # Evaluation suite name

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8777                                                    # Host Port

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "cosmos"                                         # Model family
    config: str = ""                                                     # Inference config name
    ckpt_path: str = ""                                                  # Pretrained checkpoint path
    planning_model_config_name: str = ""                                 # Planning model config name
    planning_model_ckpt_path: str = ""                                   # Planning model checkpoint path
    config_file: str = "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py"  # Cosmos default config file path

    use_third_person_image: bool = True                                  # Whether to include third-person ("primary") image in input
    num_third_person_images: int = 1                                     # Number of third-person images to include in input (RoboCasa: 1 for left, 1 for right)
    use_wrist_image: bool = True                                         # Whether to include wrist image in input
    num_wrist_images: int = 2                                            # Number of wrist images to include in input (RoboCasa: 1 wrist image)
    use_proprio: bool = True                                             # Whether to include proprio state in input
    flip_images: bool = False                                            # Whether to flip images vertically across x-axis
    use_variance_scale: bool = False                                     # Whether to scale variance used to sample sigma max for denoising for increased diversity in generations
    use_jpeg_compression: bool = False                                   # Whether to use JPEG compression on images before querying policy
    ar_future_prediction: bool = False                                   # Whether to predict future state autoregressively
    ar_value_prediction: bool = False                                    # Whether to predict future state value autoregressively
    ar_qvalue_prediction: bool = False                                   # Whether to predict Q-value autoregressively
    num_denoising_steps_action: int = 10                                 # Number of denoising steps to take for action prediction
    num_denoising_steps_future_state: int = 1                            # Number of denoising steps to take for future state prediction (only applicable if ar_future_prediction is True; otherwise equal to num_denoising_steps_action)
    num_denoising_steps_value: int = 1                                   # Number of denoising steps to take for value prediction (only applicable if ar_value_prediction is True; otherwise equal to num_denoising_steps_action)
    unnormalize_actions: bool = True                                     # Unnormalize actions if trained with normalized actions
    normalize_proprio: bool = True                                       # Normalize proprio input if trained with normalized proprio
    dataset_stats_path: str = ""                                         # Path to dataset statistics file for action unnormalization and proprio normalization
    t5_text_embeddings_path: str = ""                                    # Path to precomputed T5 text embeddings dictionary (key: instruction, val: embedding)
    trained_with_image_aug: bool = True                                  # Whether the model was trained with image augmentations (needed for test-time image transformations)
    chunk_size: int = 50                                                 # Number of actions to predict in chunk
    num_open_loop_steps: int = 50                                        # Number of actions in predicted chunk to execute open-loop before requerying policy

    deterministic: bool = True                                           # Whether to run in deterministic mode

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
    # Utils
    #################################################################################################################
    seed: int = 195                                                      # Random seed (for reproducibility)
    randomize_seed: bool = False                                         # Whether to randomize the seed for sampling

    # fmt: on


def validate_config(cfg) -> None:
    """Validate configuration parameters."""
    assert cfg.ckpt_path is not None, "ckpt_path must not be None!"

    # Assertion: Center crop should be enabled for models trained with image augmentations
    if "image_aug" in str(cfg.ckpt_path) and "no_image_aug" not in str(cfg.ckpt_path):
        assert cfg.trained_with_image_aug, (
            "Expecting `trained_with_image_aug==True` because model was trained with image augmentations!"
        )

    # Assertion: Center crop should be disabled for models trained without image augmentations
    if "no_image_aug" in str(cfg.ckpt_path):
        assert not cfg.trained_with_image_aug, (
            "Expecting `trained_with_image_aug==False` because model was not trained with image augmentations!"
        )


# === Server Interface ===
class PolicyServer:
    def __init__(self, cfg) -> Path:
        """
        A simple policy server; exposes `/act` to predict an action for a given observation + task description.
        """
        self.cfg = cfg

        # Validate configuration
        validate_config(cfg)

        # Set random seed
        set_seed_everywhere(cfg.seed)

        # Initialize T5 text embeddings cache
        init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

        # Load Cosmos Policy dataset stats
        self.dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

        # Initialize model and Cosmos config
        self.model, self.cosmos_config = get_model(cfg)
        assert cfg.chunk_size == self.cosmos_config.dataloader_train.dataset.chunk_size, (
            f"Mismatch found between train and test chunk sizes! Train: {self.cosmos_config.dataloader_train.dataset.chunk_size}, Test: {cfg.chunk_size}"
        )

        # Initialize model for world model and value function
        if cfg.planning_model_ckpt_path != "":
            self.planning_model, self.planning_model_config = get_planning_model(cfg)
        else:
            self.planning_model, self.planning_model_config = None, None

        # Get expected image dimensions
        self.resize_size = get_image_resize_size(self.cfg.model_family)
        self.cfg.resize_size = self.resize_size

        # Initialize worker pool for parallel inference if requested
        self.worker_pool = None
        if self.cfg.use_parallel_inference and self.cfg.num_queries_best_of_n > 1:
            try:
                available_gpus = [int(gpu.strip()) for gpu in self.cfg.available_gpus.split(",")]
                self.worker_pool = WorkerPoolManager(self.cfg, self.dataset_stats, available_gpus)
                self.worker_pool.start_workers()
            except Exception as e:
                print(f"[deploy] Failed to initialize parallel worker pool: {e}. Falling back to serial inference.")
                self.worker_pool = None

    def get_server_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            observation = payload
            task_description = observation["task_description"]

            # Convert lists to numpy arrays
            observation["primary_image"] = np.array(observation["primary_image"], dtype=np.uint8)
            observation["left_wrist_image"] = np.array(observation["left_wrist_image"], dtype=np.uint8)
            observation["right_wrist_image"] = np.array(observation["right_wrist_image"], dtype=np.uint8)
            observation["proprio"] = np.array(observation["proprio"], dtype=np.float32)

            # Record metadata
            return_all_query_results = False
            if "return_all_query_results" in observation:
                return_all_query_results = observation["return_all_query_results"]

            # Determine number of queries (best-of-N)
            num_queries = max(1, int(self.cfg.num_queries_best_of_n))

            # Execute queries: parallel or serial
            if (
                self.cfg.use_parallel_inference
                and num_queries > 1
                and self.worker_pool
                and self.worker_pool.initialized
            ):
                start_time = time.time()
                query_results = query_model_parallel(
                    self.cfg,
                    observation,
                    task_description,
                    self.worker_pool,
                )
                print(f"Parallel queries completed: {len(query_results)} results in {time.time() - start_time:.3f}s")
            else:
                query_results = []
                for query_idx in range(num_queries):
                    actions_by_depth = []  # Action chunks across all depths of the search
                    future_image_predictions_by_depth = []  # Future image predictions across all depths of the search
                    value_predictions_by_depth = []  # Value predictions across all depths of the search
                    return_dict = {}
                    # Query model to get action
                    start_time = time.time()
                    action_return_dict = get_action(
                        self.cfg,
                        self.model,
                        self.dataset_stats,
                        observation,
                        task_description,
                        seed=self.cfg.seed + query_idx,
                        randomize_seed=self.cfg.randomize_seed,
                        num_denoising_steps_action=self.cfg.num_denoising_steps_action,
                        generate_future_state_and_value_in_parallel=not (
                            self.cfg.ar_future_prediction
                            or self.cfg.ar_value_prediction
                            or self.cfg.ar_qvalue_prediction
                        ),
                    )
                    query_time = time.time() - start_time
                    print(f"Query {query_idx + 1}/{num_queries}: Action query time = {query_time:.3f} sec")
                    return_dict["actions"] = action_return_dict["actions"]
                    actions_by_depth.append(return_dict["actions"])

                    if self.cfg.ar_future_prediction:
                        # Autoregressively query model to get future state prediction
                        start_time = time.time()
                        future_state_return_dict = get_future_state_prediction(
                            self.cfg,
                            model=self.planning_model if self.planning_model is not None else self.model,
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
                            seed=self.cfg.seed + query_idx,
                            randomize_seed=self.cfg.randomize_seed,
                            num_denoising_steps_future_state=self.cfg.num_denoising_steps_future_state,
                            use_ensemble_future_state_predictions=self.cfg.use_ensemble_future_state_predictions,
                            num_future_state_predictions_in_ensemble=self.cfg.num_future_state_predictions_in_ensemble,
                            future_state_ensemble_aggregation_scheme=self.cfg.future_state_ensemble_aggregation_scheme,
                        )
                        query_time = time.time() - start_time
                        print(
                            f"Query {query_idx + 1}/{num_queries}: Future state prediction query time = {query_time:.3f} sec"
                        )
                        return_dict["future_image_predictions"] = future_state_return_dict["future_image_predictions"]
                        future_image_predictions_by_depth.append(return_dict["future_image_predictions"])

                    else:
                        if not self.cfg.ar_qvalue_prediction:
                            return_dict["future_image_predictions"] = action_return_dict["future_image_predictions"]

                    if self.cfg.ar_value_prediction:
                        # Autoregressively query model to get value prediction
                        start_time = time.time()
                        value_return_dict = get_value_prediction(
                            self.cfg,
                            model=self.planning_model if self.planning_model is not None else self.model,
                            data_batch=action_return_dict["data_batch"],
                            future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                            seed=self.cfg.seed + query_idx,
                            randomize_seed=self.cfg.randomize_seed,
                            num_denoising_steps_value=self.cfg.num_denoising_steps_value,
                            use_ensemble_value_predictions=self.cfg.use_ensemble_value_predictions,
                            num_value_predictions_in_ensemble=self.cfg.num_value_predictions_in_ensemble,
                        )
                        query_time = time.time() - start_time
                        print(
                            f"Query {query_idx + 1}/{num_queries}: Value prediction query time = {query_time:.3f} sec"
                        )
                        return_dict["value_prediction"] = value_return_dict["value_prediction"]
                        value_predictions_by_depth.append(return_dict["value_prediction"])
                        print(
                            f"Query {query_idx + 1}/{num_queries}: Value prediction: {return_dict['value_prediction']:.4f}"
                        )
                    elif self.cfg.ar_qvalue_prediction:
                        # Autoregressively query model to get Q-value prediction
                        start_time = time.time()
                        value_return_dict = get_qvalue_prediction(
                            self.cfg,
                            model=self.planning_model if self.planning_model is not None else self.model,
                            data_batch=action_return_dict["data_batch"],
                            action_sample=action_return_dict["generated_latent"],
                            seed=self.cfg.seed + query_idx,
                            randomize_seed=self.cfg.randomize_seed,
                            num_denoising_steps_value=self.cfg.num_denoising_steps_value,
                            use_ensemble_value_predictions=self.cfg.use_ensemble_value_predictions,
                            num_value_predictions_in_ensemble=self.cfg.num_value_predictions_in_ensemble,
                        )
                        query_time = time.time() - start_time
                        print(
                            f"Query {query_idx + 1}/{num_queries}: Value prediction query time = {query_time:.3f} sec"
                        )
                        return_dict["value_prediction"] = value_return_dict["value_prediction"]
                        value_predictions_by_depth.append(return_dict["value_prediction"])
                        print(
                            f"Query {query_idx + 1}/{num_queries}: Value prediction: {return_dict['value_prediction']:.4f}"
                        )
                    else:
                        return_dict["value_prediction"] = action_return_dict["value_prediction"]
                        value_predictions_by_depth.append(return_dict["value_prediction"])

                    if self.cfg.search_depth > 1:
                        assert not self.cfg.ar_qvalue_prediction, (
                            "Search depth > 1 not supported for Q(s, a) value prediction!"
                        )
                        for depth in range(2, self.cfg.search_depth + 1):
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
                                    self.model.config.min_num_conditional_frames
                                )  # Reset to the original value
                                data_batch["mask_current_state_action_for_value_prediction"] = (
                                    False  # Don't use input masking for action prediction
                                )
                                if self.cfg.randomize_seed:
                                    seed = secrets.randbits(32) % 256
                                else:
                                    seed = self.cfg.seed + query_idx
                                batch_size = 1
                                next_generated_latent_with_action = self.model.generate_samples_from_batch(
                                    data_batch,
                                    n_sample=batch_size,
                                    num_steps=self.cfg.num_denoising_steps_action,
                                    seed=seed,
                                    is_negative_prompt=False,
                                    use_variance_scale=self.cfg.use_variance_scale,
                                    skip_vae_encoding=True,
                                    previous_generated_latent=rearranged_next_latent_with_future_state,  # Use future state sample since parts of value sample might be masked out
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
                                        (self.cfg.chunk_size, ACTION_DIM),
                                        action_indices=action_indices,
                                    )
                                    .to(torch.float32)
                                    .cpu()
                                    .numpy()
                                )
                                # Unnormalize actions
                                if self.cfg.unnormalize_actions:
                                    next_actions = unnormalize_actions(next_actions, self.dataset_stats)
                                # Squeeze and convert to list
                                next_actions = next_actions[0]
                                next_actions = [next_actions[i] for i in range(len(next_actions))]
                                actions_by_depth.append(next_actions)

                                ################################
                                # Predict next future state
                                ################################
                                future_state_return_dict = get_future_state_prediction(
                                    self.cfg,
                                    model=self.planning_model if self.planning_model is not None else self.model,
                                    data_batch=action_return_dict["data_batch"],
                                    generated_latent_with_action=next_generated_latent_with_action,
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
                                    future_image_latent_idx=action_return_dict["latent_indices"][
                                        "future_image_latent_idx"
                                    ],
                                    future_image2_latent_idx=action_return_dict["latent_indices"][
                                        "future_image2_latent_idx"
                                    ],
                                    seed=self.cfg.seed + query_idx,
                                    randomize_seed=self.cfg.randomize_seed,
                                    num_denoising_steps_future_state=self.cfg.num_denoising_steps_future_state,
                                    use_ensemble_future_state_predictions=self.cfg.use_ensemble_future_state_predictions,
                                    num_future_state_predictions_in_ensemble=self.cfg.num_future_state_predictions_in_ensemble,
                                    future_state_ensemble_aggregation_scheme=self.cfg.future_state_ensemble_aggregation_scheme,
                                )
                                # Track per-depth prediction
                                future_image_predictions_by_depth.append(
                                    future_state_return_dict["future_image_predictions"]
                                )
                                ################################
                                # Predict next value
                                ################################
                                value_return_dict = get_value_prediction(
                                    self.cfg,
                                    model=self.planning_model if self.planning_model is not None else self.model,
                                    data_batch=action_return_dict["data_batch"],
                                    future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                                    seed=self.cfg.seed + query_idx,
                                    randomize_seed=self.cfg.randomize_seed,
                                    num_denoising_steps_value=self.cfg.num_denoising_steps_value,
                                    use_ensemble_value_predictions=self.cfg.use_ensemble_value_predictions,
                                    num_value_predictions_in_ensemble=self.cfg.num_value_predictions_in_ensemble,
                                )
                                return_dict["value_prediction"] = value_return_dict["value_prediction"]
                                value_predictions_by_depth.append(return_dict["value_prediction"])
                                print(
                                    f"Query {query_idx + 1}/{num_queries}: Value prediction: {return_dict['value_prediction']:.4f}",
                                )

                    return_dict["future_image_predictions_by_depth"] = future_image_predictions_by_depth
                    return_dict["value_predictions_by_depth"] = value_predictions_by_depth
                    return_dict["actions_by_depth"] = actions_by_depth
                    query_results.append(return_dict)

            # Fallback in case parallel path returned nothing
            if not query_results:
                raise RuntimeError("No query results returned from model inference")

            # Replace value of each query with aggregate value over all value predictions
            # This is only applicable when search depth > 1 because otherwise the aggregation is handled in get_value_prediction()
            # For search depth == 1, the return dict contains the aggregated value prediction already
            if self.cfg.search_depth > 1:
                for query_idx, return_dict in enumerate(query_results):
                    if self.cfg.search_depth_value_aggregation_scheme == "average":
                        return_dict["value_prediction"] = np.mean(return_dict["value_predictions_by_depth"]).item()
                    elif self.cfg.search_depth_value_aggregation_scheme == "use_last_value":
                        return_dict["value_prediction"] = return_dict["value_predictions_by_depth"][-1]
                    else:
                        raise ValueError(
                            f"Invalid search depth value aggregation scheme: {self.cfg.search_depth_value_aggregation_scheme}"
                        )

            # Print all value predictions
            for query_idx, return_dict in enumerate(query_results):
                predicted_value = return_dict["value_prediction"]
                print(
                    f"Query {query_idx + 1}/{num_queries} (seed {self.cfg.seed + query_idx}): Predicted value = {predicted_value:.4f}",
                )

            # Build mapping: seed -> (actions, future_image_predictions, value_prediction)
            seed_to_return_dict = {
                self.cfg.seed + query_idx: (
                    rd.get("actions"),
                    rd.get("future_image_predictions", None),
                    float(rd.get("value_prediction", 0.0)),
                )
                for query_idx, rd in enumerate(query_results)
            }

            # Base seed and best-of-N selection
            base_seed = self.cfg.seed
            base_seed_actions, base_seed_future_preds, base_seed_value = seed_to_return_dict[base_seed]
            best_actions = base_seed_actions
            best_future_predictions = base_seed_future_preds
            best_value_predictions = base_seed_value
            best_seed = base_seed

            highest_value_seed, highest_value_return = max(seed_to_return_dict.items(), key=lambda x: x[1][2])
            highest_value_actions = highest_value_return[0]
            highest_value_future_preds = highest_value_return[1]
            highest_value_value = highest_value_return[2]

            if highest_value_value > base_seed_value:
                best_actions = highest_value_actions
                best_future_predictions = highest_value_future_preds
                best_value_predictions = highest_value_value
                best_seed = highest_value_seed

            print(f"Selected seed {best_seed} with value = {best_value_predictions:.4f}")

            # In case client saves all data, gather all actions, future image predictions, and value predictions
            if return_all_query_results:
                all_actions = []
                all_future_image_predictions = []
                all_value_predictions = []
                all_actions_by_depth = []
                all_future_image_predictions_by_depth = []
                all_value_predictions_by_depth = []
                for return_dict in query_results:
                    all_actions.append(return_dict["actions"])
                    if "future_image_predictions" in return_dict:
                        all_future_image_predictions.append(return_dict["future_image_predictions"])
                    else:
                        all_future_image_predictions.append(None)  # None if doing model-free planning with Q-value
                    all_value_predictions.append(return_dict["value_prediction"])
                    all_actions_by_depth.append(return_dict["actions_by_depth"])
                    all_future_image_predictions_by_depth.append(return_dict["future_image_predictions_by_depth"])
                    all_value_predictions_by_depth.append(return_dict["value_predictions_by_depth"])
                response = dict(
                    actions=best_actions,
                    future_image_predictions=best_future_predictions,
                    value_prediction=best_value_predictions,
                    all_actions=all_actions,
                    all_future_image_predictions=all_future_image_predictions,
                    all_value_predictions=all_value_predictions,
                    all_actions_by_depth=all_actions_by_depth,
                    all_future_image_predictions_by_depth=all_future_image_predictions_by_depth,
                    all_value_predictions_by_depth=all_value_predictions_by_depth,
                )
            else:
                response = dict(
                    actions=best_actions,
                    future_image_predictions=best_future_predictions,
                    value_prediction=best_value_predictions,
                )

            if double_encode:
                return JSONResponse(json_numpy.dumps(response))
            else:
                return JSONResponse(response)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict, 'task_description': str}\n"
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        uvicorn.run(self.app, host=host, port=port)


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    # Set deterministic mode if specified
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
        set_seed_everywhere(cfg.seed)

    # Set multiprocessing start method if using parallel inference
    if cfg.use_parallel_inference:
        mp.set_start_method("spawn", force=True)

    server = PolicyServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
