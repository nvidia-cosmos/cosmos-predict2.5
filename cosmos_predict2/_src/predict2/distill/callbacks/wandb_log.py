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

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.utils.data
import wandb

from cosmos_predict2._src.imaginaire.utils import distributed, log, misc, wandb_util
from cosmos_predict2._src.imaginaire.utils.callback import Callback
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io

if TYPE_CHECKING:
    from cosmos_predict2._src.imaginaire.model import ImaginaireModel


class WandBCallback(Callback):
    """The callback class for logging to Weights and Biases (W&B).

    By default, WandBCallback logs the following training stats to W&B every config.trainer.logging_iter:
    - iteration: The current iteration number (useful for visualizing the training progress over time).
    - train/loss: The computed overall loss in the training batch.
    - optim/lr: The current learning rate and weight decay for each optimizer group
    - timer/*: The averaged timing results of each code block recorded by trainer.training_timer.
    For validation, WandBCallback logs:
    - val/loss: The computed overall loss in the validation dataset.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.training_loss = 0  # variable to store the accumulated training loss

    @distributed.rank0_only
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        wandb_util.init_wandb(self.config, model=model)
        config = self.config
        job_local_path = config.job.path_local
        # read optional job_env saved by `log_reproducible_setup`
        if os.path.exists(f"{job_local_path}/job_env.yaml"):
            job_info = easy_io.load(f"{job_local_path}/job_env.yaml")
            if wandb.run:
                wandb.run.config.update({f"JOB_INFO/{k}": v for k, v in job_info.items()}, allow_val_change=True)

        if os.path.exists(f"{config.job.path_local}/config.yaml") and "SLURM_LOG_DIR" in os.environ:
            easy_io.copyfile(
                f"{config.job.path_local}/config.yaml",
                os.path.join(os.environ["SLURM_LOG_DIR"], "config.yaml"),
            )

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:  # Log the curent learning rate.
        if iteration % self.config.trainer.logging_iter == 0 and distributed.is_rank0():
            info = {}
            info["sample_counter"] = getattr(self.trainer, "sample_counter", iteration)

            for i, param_group in enumerate(optimizer.param_groups):
                info[f"optim/lr_{i}"] = param_group["lr"]
                info[f"optim/weight_decay_{i}"] = param_group["weight_decay"]

            wandb.log(info, step=iteration)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Log the timing results (over a number of iterations) and the training loss.
        self.training_loss += loss.detach().float()
        if iteration % self.config.trainer.logging_iter == 0:
            timer_results = self.trainer.training_timer.compute_average_results()
            avg_loss = self.training_loss / self.config.trainer.logging_iter
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            if distributed.is_rank0():
                info = {f"timer/{key}": value for key, value in timer_results.items()}
                info["train/loss"] = avg_loss.item()
                info["iteration"] = iteration
                info["sample_counter"] = getattr(self.trainer, "sample_counter", iteration)
                wandb.log(info, step=iteration)
            self.trainer.training_timer.reset()
            self.training_loss = 0

    def on_validation_start(
        self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        # Cache for collecting data/output batches.
        self._val_cache: dict[str, Any] = dict(
            data_batches=[],
            output_batches=[],
            loss=torch.tensor(0.0, device="cuda"),
            sample_size=torch.tensor(0, device="cuda"),
        )

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Collect the validation batch and aggregate the overall loss.
        # Collect the validation batch and aggregate the overall loss.
        batch_size = misc.get_data_batch_size(data_batch)
        self._val_cache["loss"] += loss * batch_size
        self._val_cache["sample_size"] += batch_size

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        # Compute the average validation loss across all devices.
        dist.all_reduce(self._val_cache["loss"], op=dist.ReduceOp.SUM)
        dist.all_reduce(self._val_cache["sample_size"], op=dist.ReduceOp.SUM)
        loss = self._val_cache["loss"].item() / self._val_cache["sample_size"]
        # Log data/stats of validation set to W&B.
        if distributed.is_rank0():
            log.info(f"Validation loss (iteration {iteration}): {loss}")
            wandb.log({"val/loss": loss}, step=iteration)

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        wandb.finish()
