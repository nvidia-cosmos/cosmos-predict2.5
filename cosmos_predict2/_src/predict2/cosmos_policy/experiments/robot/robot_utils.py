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

"""Utils for evaluating robot policies in various environments."""

import logging
import os
import time
from typing import Any, Optional, Tuple

import numpy as np
import torch
import wandb

# Initialize important constants
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Model image size configuration
MODEL_IMAGE_SIZES = {
    "cosmos": 224,
}


def get_image_resize_size(model_family: str) -> int:
    """
    Get image resize dimensions for a specific model (assumes square images).

    Args:
        model_family: Model family name (e.g., "cosmos")

    Returns:
        int: Image resize dimension

    Raises:
        ValueError: If model family is not supported
    """
    if model_family not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Unsupported model family: {model_family}")

    return MODEL_IMAGE_SIZES[model_family]


def setup_logging(
    cfg: Any,
    task_identifier: str,
    log_dir: str,
    run_id_note: Optional[str] = None,
    use_wandb: bool = True,
    wandb_entity: str = "nvidia-dir",
    wandb_project: str = "cosmos_policy_eval",
    extra_wandb_tags: Optional[list] = None,
) -> Tuple[Any, str, str]:
    """
    Set up logging to file and optionally to wandb.

    Args:
        cfg: Configuration object with model parameters
        task_identifier: Task/suite identifier (e.g., "libero_spatial", "PnPCounterToCab")
        log_dir: Local directory for log files
        run_id_note: Optional note to append to run ID
        use_wandb: Whether to log to Weights & Biases
        wandb_entity: WandB entity name
        wandb_project: WandB project name
        extra_wandb_tags: Optional additional tags for WandB run

    Returns:
        Tuple of (log_file, local_log_filepath, run_id)
    """
    # Create run ID
    data_collection = getattr(cfg, "data_collection", False)
    model_family = getattr(cfg, "model_family", "cosmos")

    if data_collection:
        run_id = f"DATA_COLLECTION-{task_identifier}-{DATE_TIME}"
    else:
        run_id = f"ENV_EVAL-{task_identifier}-{model_family}-{DATE_TIME}"

    if run_id_note is not None:
        run_id += f"--{run_id_note}"

    # Set up local logging
    os.makedirs(log_dir, exist_ok=True)
    local_log_filepath = os.path.join(log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if use_wandb:
        tags = [task_identifier]
        if extra_wandb_tags:
            tags.extend(extra_wandb_tags)

        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=run_id,
            tags=tags,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """
    Log a message to console and optionally to a log file.

    Args:
        message: Message to log
        log_file: Optional file handle to write to
    """
    print(message)
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()
