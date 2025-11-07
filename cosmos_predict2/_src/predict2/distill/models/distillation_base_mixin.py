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

import collections
import math
import random
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

import attrs
import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from einops import rearrange, repeat
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor.api import DTensor
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_predict2._src.common.modules.res_sampler import Sampler
from cosmos_predict2._src.imaginaire.checkpointer.s3_filesystem import S3StorageReader
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_predict2._src.imaginaire.utils import log, misc
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast, broadcast_split_tensor
from cosmos_predict2._src.imaginaire.utils.count_params import count_params
from cosmos_predict2._src.imaginaire.utils.ema import FastEmaModelUpdater
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition

# from torch.nn.utils.clip_grad import clip_grad_norm_ as clip_grad_norm_impl_
from cosmos_predict2._src.predict2.distill.utils.torch_future import clip_grad_norm_ as clip_grad_norm_impl_
from cosmos_predict2._src.predict2.models.denoise_prediction import DenoisePrediction
from cosmos_predict2._src.predict2.utils.dtensor_helper import (
    DTensorFastEmaModelUpdater,
    broadcast_dtensor_model_states,
)
from cosmos_predict2._src.predict2.utils.optim_instantiate import get_base_optimizer


@attrs.define(slots=False)
class BaseDistillConfig:
    adaptive_weight: bool = False
    align: bool = False
    change_time_embed: bool = False
    disable_proj_grad: bool = True
    dmd: bool = True
    fd_size: float = 1e-4
    fd_type: int = 0  # finite difference type
    fsdp_shard_size: int = 4
    grad_clip: bool = False
    init_student_with_teacher: bool = True
    intermediate_feature_ids: Optional[List[int]] = None
    loss_scale_GAN_discriminator: float = 1.0
    loss_scale_GAN_generator: float = 1.0
    loss_scale_fake_score: float = 1.0
    loss_scale_sid: float = 1.0
    max_simulation_steps: int = 1
    max_simulation_steps_fake: int = 4
    max_t_prob: float = 1.0
    neg_embed_path: str = ""
    neg_prompt_str: str = "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all."
    net_discriminator_head: LazyDict = None
    # for SiD/GAN
    net_fake_score: LazyDict = None
    net_teacher: LazyDict = None
    old_warmup: bool = False
    optimizer_discriminator_config: LazyDict = L(get_base_optimizer)(
        model=None,
        lr=2e-7,
        weight_decay=0.01,
        betas=[0.0, 0.999],
        optim_type="fusedadam",
        eps=1e-8,
        master_weights=True,
        capturable=True,
    )
    optimizer_fake_score_config: LazyDict = L(get_base_optimizer)(
        model=None,
        lr=2e-7,
        weight_decay=0.01,
        betas=[0.0, 0.999],
        optim_type="fusedadam",
        eps=1e-8,
        master_weights=True,
        capturable=True,
    )
    scaling: Literal["rectified_flow", "edm"] = "rectified_flow"
    sigma_data: float = 1.0
    student_update_freq: int = 5
    tangent_warmup: int = 1000
    teacher_guidance: float = 4.0
    teacher_load_from: LazyDict = None  # contains both load_path and credentials
    timestep_shift: float = 5


class DistillationCoreMixin:
    """
    Shared distillation logic by both DMD2 and sCM, agnostic to EDM/RF/TrigFlow scaling specifics.
    """

    # ------------------------ init helpers ------------------------
    def init_distillation_common(self, config) -> None:
        # Delay Sampler() creation until after nn.Module.__init__ completes.
        # It will be initialized in set_up_model.
        self.sampler = None

        self.grad_clip = config.grad_clip
        self.tangent_warmup = config.tangent_warmup  # tangent warmup, only used for scm
        self.teacher_guidance = config.teacher_guidance
        self.change_time_embed = config.change_time_embed
        self.intermediate_feature_ids = config.intermediate_feature_ids
        self.max_t_prob = config.max_t_prob
        self.student_update_freq = config.student_update_freq
        self.loss_scale_sid = config.loss_scale_sid
        self.loss_scale_GAN_generator = config.loss_scale_GAN_generator
        self.loss_scale_fake_score = config.loss_scale_fake_score
        self.loss_scale_GAN_discriminator = config.loss_scale_GAN_discriminator
        self.fd_type = config.fd_type
        self.fd_size = config.fd_size
        self.max_simulation_steps = config.max_simulation_steps
        self.max_simulation_steps_fake = config.max_simulation_steps_fake
        self.dmd = config.dmd
        self.adaptive_weight = config.adaptive_weight
        self.disable_proj_grad = config.disable_proj_grad
        self.timestep_shift = config.timestep_shift
        self.neg_prompt_str = config.neg_prompt_str

        # Negative prompt embedding (if provided)
        if getattr(config, "neg_embed_path", ""):
            from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io

            self.neg_embed = easy_io.load(config.neg_embed_path)
        else:
            self.neg_embed = None

    # ------------------------ model building / checkpoint IO ------------------------
    def build_net(self, net_config_dict):
        init_device = "meta"
        with misc.timer("Creating PyTorch model"):
            with torch.device(init_device):
                net = lazy_instantiate(net_config_dict)

            self._param_count = count_params(net, verbose=False)

            if self.fsdp_device_mesh:
                net.fully_shard(mesh=self.fsdp_device_mesh)
                net = fully_shard(net, mesh=self.fsdp_device_mesh, reshard_after_forward=True)

            with misc.timer("meta to cuda and broadcast model states"):
                net.to_empty(device="cuda")
                # IMPORTANT: model init should not depend on current tensor shape, or it can handle DTensor shape.
                net.init_weights()

            if self.fsdp_device_mesh:
                # recall model weight init; be careful for buffers!
                broadcast_dtensor_model_states(net, self.fsdp_device_mesh)
                for name, param in net.named_parameters():
                    assert isinstance(param, DTensor), f"param should be DTensor, {name} got {type(param)}"
        return net

    @misc.timer("DistillationCoreMixin: set_up_model")
    def set_up_model(self):
        config = self.config
        with misc.timer("Creating PyTorch model and ema if enabled"):
            # Initialize sampler after Module.__init__ to avoid early module assignment.
            if self.sampler is None:
                self.sampler = Sampler()
            self.conditioner = lazy_instantiate(config.conditioner)
            assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, (
                "conditioner should not have learnable parameters"
            )

            assert config.teacher_load_from.load_path, (
                "A pretrained teacher model checkpoint is required for distillation"
            )

            self.net_teacher = self.build_net(config.net_teacher)
            if self.config.init_student_with_teacher:
                log.info("==========Loading teacher checkpoint to TEACHER net==========")
                self.load_ckpt_to_net(self.net_teacher, config.teacher_load_from.load_path)

            self.net = self.build_net(config.net)
            if self.config.init_student_with_teacher:
                log.info("==========Loading teacher net weights to STUDENT net==========")
                # strict copy teacher -> student while ignoring *_extra_state; keep target for non-matching keys
                to_load = {k: v for k, v in self.net_teacher.state_dict().items() if not k.endswith("_extra_state")}
                res = self.net.load_state_dict(to_load, strict=False)
                missing = [k for k in res.missing_keys if not k.endswith("_extra_state")]
                unexpected = [k for k in res.unexpected_keys if not k.endswith("_extra_state")]
                if missing or unexpected:
                    log.warning(f"!!!!!!!!!!!!!!!!!Missing: {missing[:10]}, Unexpected: {unexpected}")
                if not missing and not unexpected:
                    log.info("==========teacher -> student: All keys matched successfully.")

            # fake score net for approximating score func of the student generator output
            if config.net_fake_score:
                # init fake score net with the teacher score func (teacher model)
                self.net_fake_score = self.build_net(config.net_fake_score)
                if self.config.init_student_with_teacher:
                    log.info("==========Loading teacher net weights to FAKE SCORE net==========")
                    to_load = {k: v for k, v in self.net_teacher.state_dict().items() if not k.endswith("_extra_state")}
                    res = self.net_fake_score.load_state_dict(to_load, strict=False)
                    missing = [k for k in res.missing_keys if not k.endswith("_extra_state")]
                    unexpected = [k for k in res.unexpected_keys if not k.endswith("_extra_state")]
                    if missing or unexpected:
                        log.warning(f"!!!!!!!!!!!!!!!!!Missing: {missing[:10]}, Unexpected: {unexpected}")
                    if not missing and not unexpected:
                        log.info("==========teacher -> fake score: All keys matched successfully.")
                assert self.loss_scale_sid > 0 or self.loss_scale_GAN_generator > 0
            else:
                self.net_fake_score = None

            # discriminator
            if config.net_discriminator_head:
                self.net_discriminator_head = self.build_net(config.net_discriminator_head)

                # assert self.loss_scale_GAN_generator > 0
                assert config.net_fake_score
                # assert self.net_discriminator_head.model_channels == self.net_fake_score.model_channels
                assert config.intermediate_feature_ids
                assert self.net_discriminator_head.num_branches == len(config.intermediate_feature_ids)
            else:
                self.net_discriminator_head = None

            # freeze models
            if self.net.use_crossattn_projection and self.disable_proj_grad:
                log.info("Freezing the CR1 embedding projection layer in student net..")
                self.net.crossattn_proj.requires_grad_(False)

            if self.net_fake_score and self.net_fake_score.use_crossattn_projection and self.disable_proj_grad:
                log.info("Freezing the CR1 embedding projection layer in fake score net..")
                self.net_fake_score.crossattn_proj.requires_grad_(False)

            log.info("Freezing teacher net..")
            self.net_teacher.requires_grad_(False)

            self._param_count = count_params(self.net, verbose=False)

            # create ema model
            if config.ema.enabled:
                self.net_ema = self.build_net(config.net)
                self.net_ema.requires_grad_(False)

                if self.fsdp_device_mesh:
                    self.net_ema_worker = DTensorFastEmaModelUpdater()
                else:
                    self.net_ema_worker = FastEmaModelUpdater()

                s = config.ema.rate
                self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()

                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)
        torch.cuda.empty_cache()

    def get_storage_reader(self, checkpoint_path: str, credential_path: str):
        if "s3://" in checkpoint_path:
            storage_reader = S3StorageReader(
                credential_path=credential_path,
                path=checkpoint_path,
            )
        else:
            storage_reader = FileSystemReader(checkpoint_path)
        return storage_reader

    def load_ckpt_to_net(self, net, ckpt_path, prefix="net_ema"):
        """
        For loading pretrained teacher net weights
        """
        if hasattr(self.config, "teacher_load_from") and self.config.teacher_load_from is not None:
            credential_path = self.config.teacher_load_from.credentials

        storage_reader = self.get_storage_reader(ckpt_path, credential_path)
        if ckpt_path.endswith(".dcp/model"):
            prefix = "net"
        _state_dict = get_model_state_dict(net)

        metadata = storage_reader.read_metadata()
        checkpoint_keys = metadata.state_dict_metadata.keys()

        model_keys = set(_state_dict.keys())

        # Add the prefix to the model keys for comparison
        prefixed_model_keys = {f"{prefix}.{k}" for k in model_keys}

        missing_keys = prefixed_model_keys - checkpoint_keys
        if missing_keys:
            log.warning(f"Missing keys in checkpoint: {missing_keys}")

        unexpected_keys = checkpoint_keys - prefixed_model_keys
        assert prefix in ["net", "net_ema"], "prefix must be either net or net_ema"
        # if load "net_ema." keys, those starting with "net." are fine to ignore in the checkpoint
        if prefix == "net_ema":
            unexpected_keys = [k for k in unexpected_keys if "net." not in k]
        else:
            unexpected_keys = [k for k in unexpected_keys if "net_ema." not in k]
        log.warning("Ignoring _extra_state keys..")
        unexpected_keys = [k for k in unexpected_keys if "_extra_state" not in k]
        if unexpected_keys:
            log.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        if not missing_keys and not unexpected_keys:
            log.info("All keys matched successfully.")

        _new_state_dict = collections.OrderedDict()
        for k in _state_dict.keys():
            # if "_extra_state" in k:
            #     log.warning(k)
            _new_state_dict[f"{prefix}.{k}"] = _state_dict[k]
        dcp.load(_new_state_dict, storage_reader=storage_reader, planner=DefaultLoadPlanner(allow_partial_load=True))
        for k in _state_dict.keys():
            _state_dict[k] = _new_state_dict[f"{prefix}.{k}"]

        log.info(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=False)))
        del _state_dict, _new_state_dict

    # ------------------------ training hooks ------------------------
    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """
        update the net_ema
        """
        # the net and its master weights are handled by the low precision callback
        # manually update the fake score and discriminator if needed
        if not self.is_student_phase(iteration):
            if self.net_fake_score:
                optimizer = self.optimizer_dict["fake_score"]
                from cosmos_predict2._src.predict2.distill.utils.misc import update_master_weights

                update_master_weights(optimizer)
            if self.net_discriminator_head:
                optimizer = self.optimizer_dict["discriminator"]
                from cosmos_predict2._src.predict2.distill.utils.misc import update_master_weights

                update_master_weights(optimizer)
        del scheduler, optimizer

        if self.net_fake_score:
            scheduler, optimizer = self.optimizer_dict["fake_score"], self.scheduler_dict["fake_score"]
            del scheduler, optimizer
        if self.net_discriminator_head:
            scheduler, optimizer = self.optimizer_dict["discriminator"], self.scheduler_dict["discriminator"]
            del scheduler, optimizer

        if not self.is_student_phase(iteration):
            return

        if self.config.ema.enabled:
            # calculate beta for EMA update
            ema_beta = self.ema_beta(self.get_effective_iteration(iteration))
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.ema.enabled:
            self.net_ema.to(dtype=torch.float32)
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()

        self.net = self.net.to(memory_format=memory_format, **self.tensor_kwargs)
        self.net_teacher = self.net_teacher.to(memory_format=memory_format, **self.tensor_kwargs)

        if self.net_fake_score:
            self.net_fake_score = self.net_fake_score.to(memory_format=memory_format, **self.tensor_kwargs)

        if self.net_discriminator_head:
            self.net_discriminator_head = self.net_discriminator_head.to(
                memory_format=memory_format, **self.tensor_kwargs
            )

        if hasattr(self.config, "use_torch_compile") and self.config.use_torch_compile:  # compatible with old config
            if torch.__version__ < "2.3":
                log.warning(
                    "torch.compile in Pytorch version older than 2.3 doesn't work well with activation checkpointing.\n"
                    "It's very likely there will be no significant speedup from torch.compile.\n"
                    "Please use at least 24.04 Pytorch container, or imaginaire4:v7 container."
                )
            # Increasing cache size.
            torch._dynamo.config.accumulated_cache_size_limit = 256
            # dynamic=False means that a separate kernel is created for each shape.
            self.net = torch.compile(self.net, dynamic=False, disable=not self.config.use_torch_compile)
            self.net_teacher = torch.compile(self.net_teacher, dynamic=False, disable=not self.config.use_torch_compile)
            if self.net_fake_score:
                self.net_fake_score = torch.compile(
                    self.net_fake_score, dynamic=False, disable=not self.config.use_torch_compile
                )

            if self.net_discriminator_head:
                self.net_discriminator_head = torch.compile(
                    self.net_discriminator_head, dynamic=False, disable=not self.config.use_torch_compile
                )

    # ------------------------ optimizer / scheduler utils ------------------------
    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        """Creates the optimizer and scheduler for the model."""
        # instantiate the net optimizer
        net_optimizer = lazy_instantiate(optimizer_config, model=self.net)
        self.optimizer_dict = {"net": net_optimizer}

        # instantiate the net scheduler
        from cosmos_predict2._src.imaginaire.utils.optim_instantiate import get_base_scheduler

        net_scheduler = get_base_scheduler(net_optimizer, self, scheduler_config)
        self.scheduler_dict = {"net": net_scheduler}

        if self.net_fake_score:
            # instantiate the optimizer and lr scheduler for fake_score
            fake_score_optimizer = lazy_instantiate(self.config.optimizer_fake_score_config, model=self.net_fake_score)
            fake_score_scheduler = get_base_scheduler(fake_score_optimizer, self, scheduler_config)
            self.optimizer_dict["fake_score"] = fake_score_optimizer
            self.scheduler_dict["fake_score"] = fake_score_scheduler

        if self.net_discriminator_head:
            # instantiate the optimizer and lr scheduler for discriminator
            discriminator_optimizer = lazy_instantiate(
                self.config.optimizer_discriminator_config, model=self.net_discriminator_head
            )
            discriminator_scheduler = get_base_scheduler(discriminator_optimizer, self, scheduler_config)
            self.optimizer_dict["discriminator"] = discriminator_optimizer
            self.scheduler_dict["discriminator"] = discriminator_scheduler

        return net_optimizer, net_scheduler

    def is_student_phase(self, iteration: int):
        # Train critic/fake-score at the very first iteration if available
        # if iteration == 0 and self.net_fake_score is not None:
        #     return False
        return (
            self.net_fake_score is None
            or iteration < self.tangent_warmup
            or iteration % self.config.student_update_freq == 0
        )

    def get_effective_iteration(self, iteration: int):
        return (
            iteration
            if self.net_fake_score is None or iteration < self.tangent_warmup
            else self.tangent_warmup + (iteration - self.tangent_warmup) // self.config.student_update_freq
        )

    def get_effective_iteration_fake(self, iteration: int):
        return iteration - self.get_effective_iteration(iteration) - 1

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        if self.is_student_phase(iteration):
            return [self.optimizer_dict["net"]]
        else:
            if self.net_discriminator_head:
                return [self.optimizer_dict["fake_score"], self.optimizer_dict["discriminator"]]
            else:
                return [self.optimizer_dict["fake_score"]]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        if self.is_student_phase(iteration):
            return [self.scheduler_dict["net"]]
        else:
            if self.net_discriminator_head:
                return [self.scheduler_dict["fake_score"], self.scheduler_dict["discriminator"]]
            else:
                return [self.scheduler_dict["fake_score"]]

    def optimizers_zero_grad(self, iteration: int) -> None:
        for optimizer in self.get_optimizers(iteration):
            optimizer.zero_grad()

    def optimizers_schedulers_step(self, grad_scaler: torch.cuda.amp.GradScaler, iteration: int) -> None:
        for optimizer in self.get_optimizers(iteration):
            grad_scaler.step(optimizer)
            grad_scaler.update()

        for scheduler in self.get_lr_schedulers(iteration):
            scheduler.step()

    # ------------------------ training step (model-agnostic) ------------------------
    @torch.no_grad()
    def forward(self, xt, t, condition: Video2WorldCondition):
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            t (torch.Tensor): The time parameter in trigflow parameterization representing noise level.
            condition (Video2WorldCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """
        return self.denoise(xt, t, condition, net_type="student")

    def denoise_edm(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        time: torch.Tensor,
        condition: Video2WorldCondition,
        net_type: Literal["teacher", "fake_score", "student"] = "teacher",
    ) -> DenoisePrediction:
        """
        Network forward to denoise the input noised data given noise level, and condition.

        Assumes EDM-scaling parameterization.

        Compared to base class denoise function, this function supports different net types:
        - fake_score: the fake score net on student generator's outputs
        - student: the student net (few-step generator)

        Args:
            xt (torch.Tensor): The input noise data.
            time (torch.Tensor): The noise level under TrigFlow parameterization.
            condition (Video2WorldCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """
        if time.ndim == 1:
            time_B_T = rearrange(time, "b -> b 1")
        elif time.ndim == 2:
            time_B_T = time
        else:
            raise ValueError(f"time shape {time.shape} is not supported")
        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")

        if condition.is_video:
            # replace the noise level of the cond frames to be the pre-defined conditional noise level (very low)
            # the scaling coefficients computed later will inherit the setting.
            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )
            condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True).type_as(
                time_B_1_T_1_1
            )  # (B,1,T,1,1)
            t_cond = torch.atan(torch.ones_like(time_B_1_T_1_1) * (self.config.sigma_conditional / self.sigma_data))
            time_B_1_T_1_1 = t_cond * condition_video_mask_B_1_T_1_1 + time_B_1_T_1_1 * (
                1 - condition_video_mask_B_1_T_1_1
            )

        # convert noise level time to EDM-formulation coefficients
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling_from_time(time_B_1_T_1_1)

        # EDM preconditioning
        net_state_in_B_C_T_H_W = xt_B_C_T_H_W * c_in_B_1_T_1_1

        if net_type == "student" and self.change_time_embed:
            # Use c_noise(t)=t to improve numerical stability
            c_noise_B_1_T_1_1 = time_B_1_T_1_1

        net = {"student": self.net, "teacher": self.net_teacher, "fake_score": self.net_fake_score}[net_type]

        # Apply vid2vid conditioning
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(net_state_in_B_C_T_H_W) / self.config.sigma_data
            # during training we temporarily concat some variables (e.g. x0 and G_x0)
            # into a batch. They are from the same data sample, so their use_video_condition
            # is a boolean tensor with batch dim; their bool value should be the same.
            use_video_cond = condition.use_video_condition
            if isinstance(use_video_cond, torch.Tensor):
                assert bool((use_video_cond == use_video_cond[0]).all().item()), (
                    "inconsistent use_video_condition in concatenated batch"
                )
                use_video_cond = bool(use_video_cond[0].item())
            if not use_video_cond:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                net_state_in_B_C_T_H_W
            )

            # Replace the first few frames of the video with the conditional frames
            # Update the c_noise as the conditional frames are clean and have very low noise

            # x_in = mask*GT + (1-mask)*x; tangent passes only through the (1-mask) branch
            net_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + net_state_in_B_C_T_H_W * (
                1 - condition_video_mask
            )

            # # Adjust c_noise for the conditional frames
            # t_cond = torch.atan(torch.ones_like(time_B_1_T_1_1) * (self.config.sigma_conditional / self.sigma_data))
            # if net_type == "student" and self.change_time_embed:
            #     c_noise_cond_B_1_T_1_1 = t_cond
            # else:
            #     _, _, _, c_noise_cond_B_1_T_1_1 = self.scaling_from_time(trigflow_t=t_cond)
            # condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
            # c_noise_B_1_T_1_1 = c_noise_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + c_noise_B_1_T_1_1 * (
            #     1 - condition_video_mask_B_1_T_1_1
            # )

        call_kwargs = dict(
            x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        )
        if net_type == "fake_score" and getattr(self, "intermediate_feature_ids", None):
            call_kwargs["intermediate_feature_ids"] = self.intermediate_feature_ids

        # forward pass through the network
        net_out = net(**call_kwargs)

        if net_type == "fake_score" and getattr(self, "intermediate_feature_ids", None):
            net_output_B_C_T_H_W, intermediate_features_outputs = net_out
            net_output_B_C_T_H_W = net_output_B_C_T_H_W.float()
        else:
            net_output_B_C_T_H_W = net_out.float()
            intermediate_features_outputs = []

        net_output_B_C_T_H_W = net_output_B_C_T_H_W.to(dtype=xt_B_C_T_H_W.dtype)

        # EDM reconstruction of x0
        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W

        # Replace GT on conditioned frames to avoid training on pinned frames (parity with base Video2WorldModel)
        if getattr(self.config, "denoise_replace_gt_frames", False) and condition.is_video:
            # Replace condition frames to be gt frames to zero out loss on these frames
            gt_frames = condition.gt_frames.type_as(x0_pred_B_C_T_H_W)
            x0_pred_B_C_T_H_W = gt_frames * condition_video_mask.type_as(x0_pred_B_C_T_H_W) + x0_pred_B_C_T_H_W * (
                1 - condition_video_mask
            )

        if net_type == "fake_score":
            return DenoisePrediction(x0=x0_pred_B_C_T_H_W, intermediate_features=intermediate_features_outputs)
        else:  # student and teacher need F
            F_pred_B_C_T_H_W = (torch.cos(time_B_1_T_1_1) * xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / (
                torch.sin(time_B_1_T_1_1) * self.sigma_data
            )
            return DenoisePrediction(
                x0=x0_pred_B_C_T_H_W, F=F_pred_B_C_T_H_W, intermediate_features=intermediate_features_outputs
            )

    def denoise(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        time: torch.Tensor,
        condition: Video2WorldCondition,
        net_type: Literal["teacher", "fake_score", "student"] = "teacher",
    ) -> DenoisePrediction:
        return self.denoise_edm(xt_B_C_T_H_W, time, condition, net_type)

    def training_step_generator(self, *args, **kwargs):
        pass

    def training_step_critic(self, *args, **kwargs):
        pass

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the diffusion model.

        This method is model-agnostic and delegates the core losses to
        `training_step_generator` and `training_step_critic`, which should
        be implemented by the inheriting EDM/RF model classes.
        """
        self._update_train_stats(data_batch)
        # if data_batch.get("t5_text_embeddings", None) is None:
        #     log.error(str(data_batch))
        #     assert False, "Missing t5_text_embeddings in data_batch"

        # Obtain text embeddings online (for CR1)
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        # Get the input data to noise and denoise and the corresponding conditioner.
        _, x0_B_C_T_H_W, condition, uncondition = self.get_data_and_condition(data_batch)

        if self.is_student_phase(iteration):
            # update the student
            log.info(f"==================Student phase. iteration: {iteration}")
            self.net.train().requires_grad_(True)
            if self.net.use_crossattn_projection and self.disable_proj_grad:
                self.net.crossattn_proj.requires_grad_(False)
            if self.net_fake_score:
                self.net_fake_score.eval().requires_grad_(False)
            if self.net_discriminator_head:
                self.net_discriminator_head.eval().requires_grad_(False)

            output_batch, kendall_loss = self.training_step_generator(x0_B_C_T_H_W, condition, uncondition, iteration)

        else:
            log.info(f"==================Critic phase. iteration: {iteration}")
            # update the fake_score and discriminator
            self.net.eval().requires_grad_(False)
            if self.net_fake_score:
                self.net_fake_score.train().requires_grad_(True)
                if self.net_fake_score.use_crossattn_projection and self.disable_proj_grad:
                    self.net_fake_score.crossattn_proj.requires_grad_(False)
            if self.net_discriminator_head:
                self.net_discriminator_head.train().requires_grad_(True)

            output_batch, kendall_loss = self.training_step_critic(x0_B_C_T_H_W, condition, uncondition, iteration)

        kendall_loss = kendall_loss.mean()  # each loss term has been separately scaled
        return output_batch, kendall_loss

    # ------------------------ Distributed Parallel ------------------------
    def sync(self, tensor, condition):
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            tensor = broadcast(tensor, cp_group)
        return tensor

    def broadcast_split_for_model_parallelsim(self, x0_B_C_T_H_W, condition, uncondition, epsilon_B_C_T_H_W, time_B_T):
        """
        Broadcast and split the input data and condition for model parallelism.
        Currently, we only support context parallelism.

        Compared to base class, here we extend the method to new nets: teacher, fake score, and discriminator.
        """
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            x0_B_C_T_H_W = broadcast_split_tensor(x0_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            epsilon_B_C_T_H_W = broadcast_split_tensor(epsilon_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            if time_B_T is not None:
                assert time_B_T.ndim == 2, "time_B_T should be 2D tensor"
                if time_B_T.shape[-1] == 1:  # single sigma / time is shared across all frames
                    time_B_T = broadcast(time_B_T, cp_group)
                else:  # different sigma for each frame
                    time_B_T = broadcast_split_tensor(time_B_T, seq_dim=1, process_group=cp_group)
            condition = condition.broadcast(cp_group)
            uncondition = uncondition.broadcast(cp_group)
            self.net.enable_context_parallel(cp_group)
            self.net_teacher.enable_context_parallel(cp_group)
            if self.net_fake_score:
                self.net_fake_score.enable_context_parallel(cp_group)
        else:
            self.net.disable_context_parallel()
            self.net_teacher.disable_context_parallel()
            if self.net_fake_score:
                self.net_fake_score.disable_context_parallel()

        return x0_B_C_T_H_W, condition, uncondition, epsilon_B_C_T_H_W, time_B_T

    # ------------------ Data Preprocessing ------------------
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor], set_video_condition: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
        """
        Assumes a video2world model that also supports text2world mode.
        """
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # Latent state
        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_data_key]
        latent_state = self.encode(raw_state).contiguous().float()

        # Condition
        if self.neg_embed is not None:
            t5_shape = data_batch["t5_text_embeddings"].shape
            data_batch["neg_t5_text_embeddings"] = repeat(
                self.neg_embed.to(**self.tensor_kwargs), "l d -> b l d", b=data_batch["t5_text_embeddings"].shape[0]
            )
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        # Important: set video condition masks etc. to support video2world mode.
        if set_video_condition:
            # Sample a shared per-example number of conditional frames for both condition and uncondition
            B = latent_state.shape[0]
            T = latent_state.shape[2]
            if T == 1:
                shared_num_conditional_frames = torch.zeros(B, dtype=torch.int32)
            else:
                num_cf_from_batch = data_batch.get("num_conditional_frames", None)
                if num_cf_from_batch is not None:
                    if isinstance(num_cf_from_batch, torch.Tensor):
                        if num_cf_from_batch.ndim == 0:
                            shared_num_conditional_frames = torch.ones(B, dtype=torch.int32) * int(
                                num_cf_from_batch.item()
                            )
                        else:
                            shared_num_conditional_frames = num_cf_from_batch.to(dtype=torch.int32, device="cpu")
                    else:
                        shared_num_conditional_frames = torch.ones(B, dtype=torch.int32) * int(num_cf_from_batch)
                elif getattr(self.config, "conditional_frames_probs", None):
                    frames_options = list(self.config.conditional_frames_probs.keys())
                    weights = list(self.config.conditional_frames_probs.values())
                    shared_num_conditional_frames = torch.tensor(
                        random.choices(frames_options, weights=weights, k=B), dtype=torch.int32
                    )
                else:
                    shared_num_conditional_frames = torch.randint(
                        self.config.min_num_conditional_frames,
                        self.config.max_num_conditional_frames + 1,
                        size=(B,),
                        dtype=torch.int32,
                    )

            log.info(
                f"=========Using num_conditional_frames for both condition and uncondition: {shared_num_conditional_frames}"
            )

            condition = condition.set_video_condition(
                gt_frames=latent_state.to(**self.tensor_kwargs),
                random_min_num_conditional_frames=self.config.min_num_conditional_frames,
                random_max_num_conditional_frames=self.config.max_num_conditional_frames,
                num_conditional_frames=shared_num_conditional_frames,
                conditional_frames_probs=None,
            )
            uncondition = uncondition.set_video_condition(
                gt_frames=latent_state.to(**self.tensor_kwargs),
                random_min_num_conditional_frames=self.config.min_num_conditional_frames,
                random_max_num_conditional_frames=self.config.max_num_conditional_frames,
                num_conditional_frames=shared_num_conditional_frames,
                conditional_frames_probs=None,
            )

        return raw_state, latent_state, condition, uncondition

    # ------------------ Checkpointing ------------------
    def model_dict(self) -> Dict[str, Any]:
        model_dict: Dict[str, Any] = {"net": self.net}
        if self.net_fake_score:
            model_dict["fake_score"] = self.net_fake_score
        if self.net_discriminator_head:
            model_dict["discriminator"] = self.net_discriminator_head
        return model_dict

    def state_dict(self) -> Dict[str, Any]:
        net_state_dict = self.net.state_dict(prefix="net.")
        if self.config.ema.enabled:
            ema_state_dict = self.net_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)
        if self.net_fake_score:
            fake_score_state_dict = self.net_fake_score.state_dict(prefix="net_fake_score.")
            net_state_dict.update(fake_score_state_dict)
        if self.net_discriminator_head:
            discriminator_state_dict = self.net_discriminator_head.state_dict(prefix="net_discriminator_head.")
            net_state_dict.update(discriminator_state_dict)
        return net_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        _fake_score_state_dict = collections.OrderedDict()
        _discriminator_score_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v
            elif k.startswith("net_fake_score."):
                _fake_score_state_dict[k.replace("net_fake_score.", "")] = v
            elif k.startswith("net_discriminator_head."):
                _discriminator_score_state_dict[k.replace("net_discriminator_head.", "")] = v

        state_dict = _reg_state_dict

        if strict:
            reg_results: _IncompatibleKeys = self.net.load_state_dict(_reg_state_dict, strict=strict, assign=assign)

            if self.config.ema.enabled:
                ema_results: _IncompatibleKeys = self.net_ema.load_state_dict(
                    _ema_state_dict, strict=strict, assign=assign
                )
            if self.net_fake_score:
                fake_score_results: _IncompatibleKeys = self.net_fake_score.load_state_dict(
                    _fake_score_state_dict, strict=strict, assign=assign
                )
            if self.net_discriminator_head:
                discriminator_results: _IncompatibleKeys = self.net_discriminator_head.load_state_dict(
                    _discriminator_score_state_dict, strict=strict, assign=assign
                )

            return _IncompatibleKeys(
                missing_keys=reg_results.missing_keys
                + (ema_results.missing_keys if self.config.ema.enabled else [])
                + (fake_score_results.missing_keys if self.net_fake_score else [])
                + (discriminator_results.missing_keys if self.net_discriminator_head else []),
                unexpected_keys=reg_results.unexpected_keys
                + (ema_results.unexpected_keys if self.config.ema.enabled else [])
                + (fake_score_results.unexpected_keys if self.net_fake_score else [])
                + (discriminator_results.unexpected_keys if self.net_discriminator_head else []),
            )
        else:
            from cosmos_predict2._src.imaginaire.utils.checkpointer import non_strict_load_model

            log.critical("load model in non-strict mode")
            log.critical(non_strict_load_model(self.net, _reg_state_dict), rank0_only=False)
            if self.config.ema.enabled:
                log.critical("load ema model in non-strict mode")
                log.critical(non_strict_load_model(self.net_ema, _ema_state_dict), rank0_only=False)
            if self.net_fake_score:
                log.critical("load fake score model in non-strict mode")
                log.critical(non_strict_load_model(self.net_fake_score, _fake_score_state_dict), rank0_only=False)
            if self.net_discriminator_head:
                log.critical("load discriminator model in non-strict mode")
                log.critical(
                    non_strict_load_model(self.net_discriminator_head, _discriminator_score_state_dict),
                    rank0_only=False,
                )

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ):
        if not self.grad_clip:
            max_norm = 1e10
        if self.net_fake_score:
            for param in self.net_fake_score.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
            clip_grad_norm_impl_(
                self.net_fake_score.parameters(),
                max_norm=max_norm,
                norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite,
                foreach=foreach,
            )
        if self.net_discriminator_head:
            for param in self.net_discriminator_head.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
            clip_grad_norm_impl_(
                self.net_discriminator_head.parameters(),
                max_norm=max_norm,
                norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite,
                foreach=foreach,
            )
        return clip_grad_norm_impl_(
            self.net.parameters(),
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        ).cpu()


class TrigFlowMixin:
    """
    Shared logic for the TrigFlow parameterization.
    """

    def convert_sigma_to_time(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.arctan(sigma / self.sigma_data)

    def convert_time_to_sigma(self, time: torch.Tensor, is_video_batch: bool, eps: float = 1e-6) -> torch.Tensor:
        t64 = time.to(torch.float64).clamp(-math.pi / 2 + eps, math.pi / 2 - eps)
        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma = (self.sigma_data / float(multiplier)) * torch.tan(t64)
        return sigma.to(dtype=time.dtype, device=time.device)

    def draw_training_time_and_epsilon(self, x0_size: int, condition: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x0_size[0]
        epsilon = torch.randn(x0_size, device="cuda")
        sigma_B = self.sde.sample_t(batch_size).to(device="cuda")
        sigma_B_1 = rearrange(sigma_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma
        is_video_batch = condition.data_type == DataType.VIDEO

        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        time_B_1 = torch.arctan(sigma_B_1 / self.sigma_data)
        return time_B_1.double(), epsilon

    def draw_training_time(self, x0_size: int, condition: Any) -> torch.Tensor:
        batch_size = x0_size[0]
        sigma_B = self.sde.sample_t(batch_size).to(device="cuda")
        sigma_B_1 = rearrange(sigma_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma
        is_video_batch = condition.data_type == DataType.VIDEO

        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        time_B_1 = torch.arctan(sigma_B_1 / self.sigma_data)
        return time_B_1.double()

    def draw_training_time_D(self, x0_size: Tuple[int, ...], condition: Any) -> torch.Tensor:
        batch_size = x0_size[0]
        if self.timestep_shift > 0:
            sigma_B = torch.rand(batch_size).to(device="cuda").double()
            sigma_B = self.timestep_shift * sigma_B / (1 + (self.timestep_shift - 1) * sigma_B)
            sigma_B_1 = rearrange(sigma_B, "b -> b 1")
            time_B_1 = torch.arctan(sigma_B_1 / (1 - sigma_B_1))
            return time_B_1
        sigma_B = self.sde_D.sample_t(batch_size).to(device="cuda")
        sigma_B_1 = rearrange(sigma_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma
        is_video_batch = condition.data_type == DataType.VIDEO
        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        time_B_1 = self.convert_sigma_to_time(sigma_B_1)
        return time_B_1.double()
