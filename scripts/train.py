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

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import importlib

from loguru import logger as logging

from cosmos_predict2._src.imaginaire.config import Config, pretty_print_overrides
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.predict2.utils.model_loader import create_model_from_consolidated_checkpoint_with_fsdp


@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    # Need to initialize the distributed environment before calling config.validate() because it tries to synchronize
    # a buffer across ranks. If you don't do this, then you end up allocating a bunch of buffers on rank 0, and also that
    # check doesn't actually do anything.
    distributed.init()

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)

    # Create the model and load the consolidated checkpoint if provided.
    # If the checkpoint is in DCP format, checkpoint loading will be handled by the DCP checkpointer.
    if isinstance(config.checkpoint.load_path, str) and config.checkpoint.load_path.endswith(".pt"):
        model = create_model_from_consolidated_checkpoint_with_fsdp(config)
    else:
        model = instantiate(config.model)

    # Create the dataloaders.
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_val = instantiate(config.dataloader_val)
    # Start training
    trainer.train(
        model,
        dataloader_train,
        dataloader_val,
    )


if __name__ == "__main__":
    # Usage: torchrun --nproc_per_node=1 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiments=predict2_video2world_training_2b_cosmos_nemo_assets

    # Get the config file from the input arguments.
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", help="Path to the config file", required=True)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the training for a few iterations to smoke test the config.",
    )
    args = parser.parse_args()
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    overrides = list(args.opts)
    if args.smoke:
        overrides.append("job.wandb_mode=disabled")
        overrides.append("trainer.max_iter=2")
        overrides.append("trainer.logging_iter=1")
        overrides.append("trainer.validation_iter=1")
    config = override(config, overrides)
    if args.dryrun:
        logging.info(
            "Config:\n" + config.pretty_print(use_color=True) + "\n" + pretty_print_overrides(args.opts, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        # Launch the training job.
        launch(config, args)
