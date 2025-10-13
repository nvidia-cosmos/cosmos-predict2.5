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

import collections
import functools
import itertools
import math
from copy import deepcopy
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LambdaLR

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.reason1.configs.default.model_config import FSDP2ModelConfig
from cosmos_predict2._src.reason1.utils.fused_adam import FusedAdam


def _optimizer_cls(params: List[nn.Parameter], optimizer_kwargs: Dict[str, Any], name: str):
    if name == "Adam":
        optimizer = torch.optim.Adam(params, **optimizer_kwargs)
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(params, **optimizer_kwargs)
    elif name == "FusedAdam":
        optimizer = FusedAdam(
            params,
            lr=optimizer_kwargs["lr"],
            weight_decay=optimizer_kwargs["weight_decay"],
            betas=optimizer_kwargs["betas"],
            capturable=True,
            master_weights=True,
        )
    else:
        raise NotImplementedError(f"Optimizer {name} not added.")
    return optimizer


class OptimizersContainer(Stateful):
    """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages
    and saving/loading optimizer state_dict at checkpoint.
    """

    def __init__(
        self,
        model_parts: List[nn.Module],
        optimizer_kwargs: Dict[str, Any],
        name: str,
        lr_multiplier: list[float],
        model_part_names: list[str],
    ) -> None:
        assert len(model_parts) == len(lr_multiplier), "lr_multiplier must have the same length as model_parts"
        self.model_parts = model_parts
        self.optimizers = [[] for _ in self.model_parts]
        self.model_part_names = model_part_names
        for model_id, model in enumerate(self.model_parts):
            optimizer_kwargs_copy = deepcopy(optimizer_kwargs)
            optimizer_kwargs_copy["lr"] *= lr_multiplier[model_id]

            if optimizer_kwargs_copy["fused"]:
                # Group the parameters by device mesh to do optimizer fusion.
                parameters_by_mesh = collections.defaultdict(list)
                for p in model.parameters():
                    if p.requires_grad:
                        device_mesh = p.device_mesh if hasattr(p, "device_mesh") else "default"
                        parameters_by_mesh[device_mesh].append(p)
                for params in parameters_by_mesh.values():
                    optimizer = _optimizer_cls(params, optimizer_kwargs_copy, name)
                    self.optimizers[model_id].append(optimizer)
            else:
                for p in model.parameters():
                    if p.requires_grad:
                        optimizer = _optimizer_cls([p], optimizer_kwargs_copy, name)
                        self.optimizers[model_id].append(optimizer)

    def __iter__(self) -> torch.optim.Optimizer:
        return iter(itertools.chain(*self.optimizers))

    def step(self) -> None:
        for optimizer in itertools.chain(*self.optimizers):
            optimizer.step()

    def zero_grad(self, set_to_none: bool = False) -> None:
        for optimizer in itertools.chain(*self.optimizers):
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        sd = {}
        for model, optimizers in zip(self.model_parts, self.optimizers):
            sd.update(
                get_optimizer_state_dict(
                    model=model,
                    optimizers=optimizers,
                    options=StateDictOptions(flatten_optimizer_state_dict=True),
                )
            )
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for model, optimizers in zip(self.model_parts, self.optimizers):
            set_optimizer_state_dict(
                model=model,
                optimizers=optimizers,
                optim_state_dict=state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )


class OptimizersInBackwardContainer(OptimizersContainer):
    """Optimiers in backward to skip .step() and .zero_grad()"""

    def __init__(
        self,
        model_parts: List[nn.Module],
        optimizer_kwargs: Dict[str, Any],
        name: str,
        lr_multiplier: list[float] = [1.0, 1.0, 1.0],
        model_part_names: list[str] = [],
    ) -> None:
        self.model_parts = model_parts
        self.optimizers = [None for _ in self.model_parts]
        self.model_part_names = model_part_names
        optim_dict = {}
        for model_id, model in enumerate(self.model_parts):
            optimizer_kwargs_copy = deepcopy(optimizer_kwargs)
            optimizer_kwargs_copy["lr"] *= lr_multiplier[model_id]

            for param in model.parameters():
                optim_dict[param] = _optimizer_cls([param], optimizer_kwargs_copy, name)

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model_id, model in enumerate(self.model_parts):
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

            self.optimizers[model_id] = [optim_dict[param] for param in model.parameters()]

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass


# consider split between PP and non-PP
def build_optimizers(
    model_parts: List[nn.Module],
    job_config: FSDP2ModelConfig,
    lr_multiplier: list[float],
    model_part_names: list[str],
) -> OptimizersContainer:
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """
    assert len(model_parts) == len(lr_multiplier) == len(model_part_names), (
        "lr_multiplier and model_part_names must have the same length as model_parts"
    )
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    if optim_in_bwd and job_config.experimental.pipeline_parallel_degree > 1:
        raise NotImplementedError("Optimizers in backward is not supported with pipeline parallelism.")
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    fused = job_config.optimizer.fused
    optimizer_kwargs = {
        "lr": lr,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "fused": fused,
        "foreach": not fused,
    }

    return (
        OptimizersContainer(model_parts, optimizer_kwargs, name, lr_multiplier, model_part_names)
        if not optim_in_bwd
        else OptimizersInBackwardContainer(model_parts, optimizer_kwargs, name, lr_multiplier, model_part_names)
    )


class SchedulersContainer(Stateful):
    """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

    def __init__(self, optimizers: OptimizersContainer, lr_lambda) -> None:
        self.schedulers = []
        for optimizer in optimizers:
            self.schedulers.append(LambdaLR(optimizer, lr_lambda=lr_lambda))

    def step(self) -> None:
        for id, scheduler in enumerate(self.schedulers):
            scheduler.step()

    def state_dict(self) -> Dict[str, Any]:
        # Currently, we have one scheduler per optimizer. However, when using MultiSchedule PP or optimizer-in-backward,
        # there are multiple optimizers and schedulers, but the scheduler state_dict remains the same for all.
        # Therefore, we only save the first one and later load it for all.
        assert len(self.schedulers) > 0, "Must have at least one scheduler to save state_dict"
        return self.schedulers[0].state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Load the same state_dict for all schedulers. The key value we're concerned with in scheduler.state_dict() is `last_epoch`,
        # which is an integer that will be automatically copied. As long as `training.steps` and `training.warmup_steps` remain
        # unchanged when resuming from a checkpoint, this approach is safe. We call `.copy()` here to ensure extra safety.
        last_epoch = state_dict["last_epoch"]  # Extract last known epoch
        _step_count = state_dict["_step_count"]
        log.info(f"Resuming schedulers by stepping them to last_epoch: {last_epoch}; _step_count: {_step_count}")

        # Manually step all schedulers to match the saved state -- this is a workaround for the inherited issue in the state dict saving (only saved the first scheduler)
        # But we have different learning rate for each scheduler, so we need to step them separately instead of loading the state dict
        # The benefit of this approach is that we can resume from a checkpoint even if the learning rate is changed
        for idx, scheduler in enumerate(self.schedulers):
            for step in range(_step_count):
                scheduler.step()  # Step forward to match previous training state
            log.info(f"Scheduler {idx + 1}/{len(self.schedulers)} stepped {_step_count} times.")
            log.info(f"Updated learning rate: {scheduler.get_last_lr()}")

    def get_last_lr(self) -> List[float]:
        return [scheduler.get_last_lr() for scheduler in self.schedulers]


def linear_warmup_linear_decay(warmup_steps: int, decay_steps: int, current_step: int) -> float:
    """Computes linear warmup followed by linear decay.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    if current_step < warmup_steps:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (warmup_steps + 1))

    else:
        # linear decay
        normalized_step = decay_steps - (current_step - warmup_steps)
        curr_adjustment = 1 - (decay_steps - normalized_step) / decay_steps

    return curr_adjustment


def linear_warmup(warmup_steps: int, current_step: int) -> float:
    """Computes linear warmup only
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    if current_step < warmup_steps:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (warmup_steps + 1))
    else:
        curr_adjustment = 1

    return curr_adjustment


def linear_warmup_cosine_cooldown(
    warmup_steps: int, cooldown_steps: int, current_step: int, base_lr: float, init_lr: float, end_lr: float
) -> float:
    """This scheduler will warmup the learning rate from init_lr to base_lr for warmup_steps,
    then decay the learning rate from base_lr to end_lr for cooldown_steps. After cooldown_steps + warmup_steps,
    the learning rate will be set to end_lr.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.

    Args:
        warmup_steps (int): The number of steps to warmup the learning rate.
        cooldown_steps (int): The number of steps to decay the learning rate.
        current_step (int): The current step.
        base_lr (float): The base learning rate.
        init_lr (float): The initial learning rate before warmup.
        end_lr (float): The final learning rate after cooldown.

    Returns:
        float: The multiplicative factor to adjust the learning rate.
    """
    total_steps = warmup_steps + cooldown_steps

    # Normalize
    init_multiplier = init_lr / base_lr
    end_multiplier = end_lr / base_lr
    if current_step <= warmup_steps:
        progress = float(current_step / warmup_steps)
        return init_multiplier + (1.0 - init_multiplier) * progress
    elif current_step <= total_steps:
        progress = (current_step - warmup_steps) / cooldown_steps
        return end_multiplier + 0.5 * (1.0 - end_multiplier) * (1 + math.cos(math.pi * progress))
    else:
        return end_multiplier


def build_lr_schedulers(optimizers: OptimizersContainer, job_config: FSDP2ModelConfig) -> SchedulersContainer:
    warmup_steps = int(job_config.training.warmup_steps)
    decay_steps = float(max(1, job_config.training.steps - warmup_steps))
    if job_config.training.use_cosine_decay:
        lr_lambda = functools.partial(
            linear_warmup_cosine_cooldown,
            warmup_steps,
            decay_steps,
            base_lr=job_config.optimizer.lr,
            init_lr=job_config.optimizer.init_lr,
            end_lr=job_config.optimizer.end_lr,
        )
    elif job_config.training.use_linear_decay:
        lr_lambda = functools.partial(linear_warmup_linear_decay, warmup_steps, decay_steps)
    else:
        lr_lambda = functools.partial(linear_warmup, warmup_steps)

    return SchedulersContainer(optimizers, lr_lambda)
