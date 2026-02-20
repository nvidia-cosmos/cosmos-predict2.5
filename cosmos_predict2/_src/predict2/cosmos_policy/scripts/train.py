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
Cosmos Policy training script with manual DistributedSampler instantiation.

This script extends the base training script to manually create DistributedSampler
instead of using instantiate(), avoiding duplicate dataset creation.
"""

import argparse
import os
import traceback

from loguru import logger as logging
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.config import Config, load_config, pretty_print_overrides
from cosmos_predict2._src.imaginaire.lazy_config import LazyConfig, instantiate
from cosmos_predict2._src.imaginaire.serialization import to_yaml
from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.imaginaire.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos_predict2._src.imaginaire.utils.launch import log_reproducible_setup


@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    # Need to initialize the distributed environment before calling config.validate() because it tries to synchronize
    # a buffer across ranks. If you don't do this, then you end up allocating a bunch of buffers on rank 0, and also that
    # check doesn't actually do anything.
    with distributed_init():
        distributed.init()

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)
    # Setup the miscellaneous stuff for reproducibility.
    log_reproducible_setup(config, args)

    with model_init():
        model = instantiate(config.model)

    # Create the dataloaders.
    with data_loader_init():
        # NOTE (user): We manually instantiate the dataloader instead of using instantiate(config.dataloader_train),
        # since it is difficult to set up the DistributedSampler without creating two duplicates of the dataset.
        # We intentionally instantiate the dataloader on every process (rather than the rank 0 process only) to work with the DistributedSampler.
        dataset = instantiate(config.dataloader_train.dataset)
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=parallel_state.get_data_parallel_world_size(),
            rank=parallel_state.get_data_parallel_rank(),
            shuffle=True,
            seed=0,
        )
        dataloader_train = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=config.dataloader_train.batch_size,
            drop_last=config.dataloader_train.drop_last,
            num_workers=config.dataloader_train.num_workers,
            persistent_workers=config.dataloader_train.persistent_workers,
            pin_memory=config.dataloader_train.pin_memory,
            pin_memory_device=config.dataloader_train.pin_memory_device,
            timeout=config.dataloader_train.timeout,
        )

        dataloader_val = None
        if config.trainer.run_validation:
            # NOTE (user): Manually instantiate the val dataloader as well
            dataset_val = instantiate(config.dataloader_val.dataset)
            sampler_val = DistributedSampler(
                dataset=dataset_val,
                num_replicas=parallel_state.get_data_parallel_world_size(),
                rank=parallel_state.get_data_parallel_rank(),
                shuffle=False,  # Do not shuffle the validation set
                seed=0,
            )
            dataloader_val = DataLoader(
                dataset=dataset_val,
                sampler=sampler_val,
                batch_size=config.dataloader_val.batch_size,
                drop_last=config.dataloader_val.drop_last,
                num_workers=config.dataloader_val.num_workers,
                persistent_workers=config.dataloader_val.persistent_workers,
                pin_memory=config.dataloader_val.pin_memory,
                pin_memory_device=config.dataloader_val.pin_memory_device,
                timeout=config.dataloader_val.timeout,
            )

    # Start training
    trainer.train(
        model,
        dataloader_train,
        dataloader_val,
    )


if __name__ == "__main__":
    # Usage: torchrun --nproc_per_node=1 -m cosmos_predict2._src.predict2.cosmos_policy.scripts.train --config=cosmos_predict2/_src/predict2/cosmos_policy/config/experiment/your_config.py

    # Get the config file from the input arguments.
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", help="Path to the config file", required=False)
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
    args = parser.parse_args()

    config = load_config(args.config, args.opts, enable_one_logger=True)

    if args.dryrun:
        logging.info(
            "Config:\n" + config.pretty_print(use_color=True) + "\n" + pretty_print_overrides(args.opts, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        try:
            to_yaml(config, f"{config.job.path_local}/config.yaml")
        except Exception:
            logging.error("to_yaml failed, falling back to LazyConfig.save_yaml:")
            logging.error(f"Traceback: {traceback.format_exc()}")
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        # Launch the training job.
        launch(config, args)
