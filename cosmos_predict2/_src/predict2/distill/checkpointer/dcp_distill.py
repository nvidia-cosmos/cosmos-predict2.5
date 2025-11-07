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
modify the interface, since we will have mulitple models / optimizers used in gan-like training
"""

import os
import time
from typing import Any, Dict, Tuple

import torch
import torch.distributed.checkpoint as dcp

from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import log, misc
from cosmos_predict2._src.predict2.distill.checkpointer.dcp import (
    AsyncMode,
    DefaultLoadPlanner,
    DefaultSavePlanner,
    ModelWrapper,
    OptimizerWrapper,
)
from cosmos_predict2._src.predict2.distill.checkpointer.dcp import DistributedCheckpointer as _DistributedCheckpointer


class DistributedCheckpointer(_DistributedCheckpointer):
    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LRScheduler] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_start(model)

        model_dict = model.model_dict()

        resume_keys, checkpoint_path = self.keys_to_resume_during_load()
        resume_keys = sorted(resume_keys)
        log.critical(f"Resuming ckpt {checkpoint_path} with keys: {resume_keys}")

        iteration = 0

        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            for key in resume_keys:
                load_planner = DefaultLoadPlanner(allow_partial_load=True)
                if hasattr(load_planner, "set_partial_channel_weight"):
                    log.critical(f"set_partial_channel_weight: {self.config_checkpoint.dcp_allow_mismatched_size}")
                    load_planner.set_partial_channel_weight(self.config_checkpoint.dcp_allow_mismatched_size)
                cur_key_ckpt_full_path = os.path.join(checkpoint_path, key)
                log.critical(f"Start loading checkpoint from {checkpoint_path}")
                torch.distributed.barrier()
                log.critical(f"starting {cur_key_ckpt_full_path}", rank0_only=False)
                if key == "model":
                    storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                    log.info("- Loading the model...")
                    _model_wrapper = ModelWrapper(model)
                    _state_dict = _model_wrapper.state_dict()

                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    _model_wrapper.load_state_dict(_state_dict)
                elif key == "optim":
                    for k, v in optimizer_dict.items():
                        storage_reader = self.get_storage_reader(f"{cur_key_ckpt_full_path}_{k}")
                        log.info("- Loading the optimizer...")
                        _optim_wrapper = OptimizerWrapper(model_dict[k], v)
                        _state_dict = _optim_wrapper.state_dict()
                        dcp.load(
                            _state_dict,
                            storage_reader=storage_reader,
                            planner=load_planner,
                        )
                        _optim_wrapper.load_state_dict(_state_dict)
                elif key == "scheduler":
                    for k, v in scheduler_dict.items():
                        storage_reader = self.get_storage_reader(f"{cur_key_ckpt_full_path}_{k}")
                        log.info("- Loading the scheduler...")
                        _state_dict = scheduler_dict[k].state_dict()
                        dcp.load(
                            _state_dict,
                            storage_reader=storage_reader,
                            planner=load_planner,
                        )
                        scheduler_dict[k].load_state_dict(_state_dict)
                elif key == "trainer":
                    storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                    log.info("- Loading the trainer...")
                    _state_dict = {
                        "grad_scaler": grad_scaler.state_dict(),
                        "iteration": iteration,
                    }
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    grad_scaler.load_state_dict(_state_dict["grad_scaler"])
                    iteration = _state_dict["iteration"]
                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")
            if self.callbacks is not None:
                self.callbacks.on_load_checkpoint(model, state_dict=_state_dict)
            log.critical(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")
        else:
            log.info("Training from scratch.")
        torch.cuda.empty_cache()

        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)
        return iteration

    def save_state_dict_worker(self, to_save_dict: Dict[str, Tuple[Any, str]], checkpoint_file: str) -> None:
        for k, (v, full_checkpoint_path) in to_save_dict.items():
            if k in ["optim", "scheduler"]:
                for key_net, state_dict in v.items():
                    storage_writer = self.get_storage_writer(f"{full_checkpoint_path}_{key_net}")
                    dcp.save(
                        state_dict,
                        storage_writer=storage_writer,
                        planner=DefaultSavePlanner(dedup_save_to_lowest_rank=True),
                    )
            else:
                storage_writer = self.get_storage_writer(full_checkpoint_path)
                dcp.save(
                    v,
                    storage_writer=storage_writer,
                    planner=DefaultSavePlanner(dedup_save_to_lowest_rank=True),
                )

        self._write_latest_checkpoint_file(checkpoint_file)
        log.critical(f"Saved checkpoint to {os.path.join(self.save_dirname, checkpoint_file)}", rank0_only=True)

    def save(
        self,
        model: ImaginaireModel,
        optimizer_dict: Dict[str, torch.optim.Optimizer],
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LRScheduler],
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.get_previous_checkpoint_results(wait_for=0)

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_start(model, iteration)

        model_dict = model.model_dict()
        checkpoint_file = f"iter_{iteration:09}"
        to_save_dict = {
            "model": ModelWrapper(model).state_dict(),
            "optim": {k: OptimizerWrapper(model_dict[k], v).state_dict() for k, v in optimizer_dict.items()},
            "scheduler": {k: v.state_dict() for k, v in scheduler_dict.items()},
            "trainer": {
                "grad_scaler": grad_scaler.state_dict(),
                "iteration": iteration,
            },
        }
        for k in to_save_dict.keys():
            output_dirname = os.path.join(self.save_dirname, f"iter_{iteration:09}/{k}")
            to_save_dict[k] = (to_save_dict[k], output_dirname)

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint(model, state_dict=to_save_dict)

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._async_with_pinned_memory(checkpoint_file, to_save_dict)
        else:
            start_time = time.monotonic()
            try:
                self.save_state_dict_worker(to_save_dict, checkpoint_file)
            finally:
                if self.callbacks is not None:
                    self.callbacks.on_save_checkpoint_success(
                        iteration=iteration, elapsed_time=time.monotonic() - start_time
                    )

        # This measures exposed (synchronous) checkpoint time, on_save_checkpoint_success()
        # is instead called to measure the entire duration for asynchronous checkpoint for the async case too.
        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)
