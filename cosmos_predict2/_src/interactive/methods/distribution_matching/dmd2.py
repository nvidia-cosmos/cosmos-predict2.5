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
import uuid
from typing import Any, Dict, Literal, Mapping

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.checkpointer import non_strict_load_model
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast_split_tensor
from cosmos_predict2._src.imaginaire.utils.object_store import download_from_s3_with_cache
from cosmos_predict2._src.imaginaire.utils.optim_instantiate import get_base_scheduler
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.interactive.configs.method_configs.config_dmd2 import DMD2Config
from cosmos_predict2._src.interactive.methods.cosmos2_interactive_model import Cosmos2InteractiveModel
from cosmos_predict2._src.interactive.utils.model_loader import get_storage_reader
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.denoise_prediction import DenoisePrediction
from cosmos_predict2._src.predict2.modules.denoiser_scaling import (
    RectifiedFlow_sCMWrapper as TrigFlow2RectifiedFlowScaling,
)


class DMD2Model(Cosmos2InteractiveModel):
    """
    DMD2 distillation model using TrigFlow parameterization.

    https://arxiv.org/abs/2405.14867
    """

    # ------------------------ Initialization & configuration ------------------------
    def __init__(self, config: DMD2Config):
        """
        Args:
            config (DMD2Config): The configuration for the DMD model
        """
        super().__init__(config)
        self.config: DMD2Config = config
        self.selected_sampling_time = config.selected_sampling_time
        self.noise_level_parameterization = config.noise_level_parameterization

        assert self.noise_level_parameterization.lower() == "trigflow", (
            "Currently only supports TrigFlow parameterization."
        )
        # Converterfrom Trigflow time to EDM scaling coefficients for student net and fake score net
        self.trigflow_scaler = TrigFlow2RectifiedFlowScaling(config.sigma_data)

    # ------------------------ Model construction & apply FSDP ------------------------
    def build_model(self):
        super().build_model()

        if self.config.load_teacher_weights:
            assert self.config.teacher_load_from.load_path, (
                "A pretrained teacher model checkpoint is required for distillation"
            )

        self.net_teacher = self.build_net(self.config.net_teacher)
        self.net_fake_score = self.build_net(self.config.net_fake_score)
        assert self.net_fake_score is not None, "DMD2 requires a fake_score network."

        log.info("==========Instantiating networks...==========")
        log.info(f"net: {self.net}")
        log.info(f"net_teacher: {self.net_teacher}")
        log.info(f"net_fake_score: {self.net_fake_score}")

        if self.config.load_teacher_weights:
            self._load_ckpt_to_net(
                self.net_teacher,
                self.config.net_teacher,
                self.config.teacher_load_from.load_path,
                credential_path=self.config.teacher_load_from.credentials,
            )
            self._copy_teacher_weights(target_net=self.net, target_name="student")
            self._copy_teacher_weights(target_net=self.net_fake_score, target_name="fake score")
            if self.config.ema.enabled and hasattr(self, "net_ema_worker"):
                self.net_ema_worker.copy_to(src_model=self.net_teacher, tgt_model=self.net_ema)

        if getattr(self.config, "student_load_from", None) and self.config.student_load_from.load_path:
            log.info(f"Initializing student net from checkpoint: {self.config.student_load_from.load_path}")
            self._load_ckpt_to_net(
                self.net,
                self.config,
                self.config.student_load_from.load_path,
                credential_path=self.config.student_load_from.credentials,
            )
            if self.config.ema.enabled and hasattr(self, "net_ema_worker"):
                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)

        # discriminator
        if self.config.net_discriminator_head:
            self.net_discriminator_head = self.build_net(self.config.net_discriminator_head)
            assert self.config.loss_scale_GAN_generator > 0, "GAN network enabled, but GAN loss is set to 0."
            assert self.config.net_fake_score, "GAN network enabled, but fake score net is not set."
            assert self.config.intermediate_feature_ids, (
                "GAN network enabled, but intermediate feature ids are not set."
            )
            assert self.net_discriminator_head.num_branches == len(self.config.intermediate_feature_ids)
        else:
            self.net_discriminator_head = None

        # freeze models
        if self.net.use_crossattn_projection:
            log.info("Freezing the CR1 embedding projection layer in student net..")
            self.net.crossattn_proj.requires_grad_(False)

        if self.net_fake_score and self.net_fake_score.use_crossattn_projection:
            log.info("Freezing the CR1 embedding projection layer in fake score net..")
            self.net_fake_score.crossattn_proj.requires_grad_(False)

        log.info("Freezing teacher net..")
        self.net_teacher.requires_grad_(False)

        self.denoiser_nets = {
            "teacher": self.net_teacher,
            "fake_score": self.net_fake_score,
            "student": self.net,
        }

        torch.cuda.empty_cache()

    def _copy_teacher_weights(self, target_net: torch.nn.Module, target_name: str) -> None:
        """Copy teacher weights into a target network, with logging on key mismatches.

        Args:
            target_net: The target network to copy weights to.
            target_name: The name of the target network.
        """
        log.info(f"==========Copying teacher weights to {target_name} net==========")
        to_load = {k: v for k, v in self.net_teacher.state_dict().items() if not k.endswith("_extra_state")}
        key_match_status = target_net.load_state_dict(to_load, strict=False)
        missing = [k for k in key_match_status.missing_keys if not k.endswith("_extra_state")]
        unexpected = [k for k in key_match_status.unexpected_keys if not k.endswith("_extra_state")]
        if missing or unexpected:
            log.warning(f"==========teacher -> {target_name}: Missing: {missing[:10]}, Unexpected: {unexpected}")
        if not missing and not unexpected:
            log.info(f"==========teacher -> {target_name}: All keys matched successfully.")

    def _load_ckpt_to_net(
        self,
        net: torch.nn.Module,
        config,
        ckpt_path: str,
        prefix: str = "net_ema",
        credential_path: str | None = None,
    ) -> None:
        if ckpt_path.endswith(".pt"):
            self._load_pt_ckpt(net, config, ckpt_path, credential_path)
        else:
            self._load_dcp_ckpt(net, ckpt_path, prefix, credential_path)

    def _load_pt_ckpt(self, net: torch.nn.Module, config: LazyDict, ckpt_path: str, credential_path) -> None:
        """
        Load a PT checkpoint into a network.  Recreates the net with the same config and loads the weights into it
        to allow for loading pt loading into fsdp nets.

        Args:
            net: The network to load the checkpoint into.
            config: The config for the network.
            ckpt_path: The path to the PT checkpoint.
        Returns:
            None
        """

        if ckpt_path.startswith("s3://"):
            local_ckpt_path = download_from_s3_with_cache(s3_path=ckpt_path, s3_credential_path=credential_path)
        else:
            local_ckpt_path = ckpt_path

        pt_state_dict = torch.load(local_ckpt_path, map_location="cpu", weights_only=False)

        def _get_common_prefix(strs: list[str]) -> str:
            """
            Get the common prefix of a list of strings.  Use to normalize weight names between different models.

            Example:
            >>> _get_common_prefix(["net.xxx.yyy", "net.xxx.zzz", "net.xxx.aaa"])
            "net.xxx."
            """
            common_prefix = ""
            first_str = strs[0]
            first_str_parts = first_str.split(".")
            n = 1
            while n <= len(first_str_parts):  # Stop when n exceeds available parts
                candidate_prefix = ".".join(first_str_parts[0:n]) + "."
                if all(key.startswith(candidate_prefix) for key in strs):
                    common_prefix = candidate_prefix
                    n += 1
                else:
                    break

            return common_prefix

        load_prefix = _get_common_prefix(list(pt_state_dict.keys()))
        target_prefix = _get_common_prefix(list(net.state_dict().keys()))

        log.info(f"==========Mapping pt checkpoint prefix from {load_prefix} to {target_prefix}==========")
        pt_state_dict = {k.replace(load_prefix, target_prefix): v for k, v in pt_state_dict.items()}

        # load a non fsdp net to load the pt checkpoint
        pt_net = self.build_net_without_fsdp(config)
        load_status = pt_net.load_state_dict(pt_state_dict, strict=True, assign=False)
        missing_keys = load_status.missing_keys
        unexpected_keys = load_status.unexpected_keys
        if missing_keys:
            log.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        pt_net = self.wrap_net_with_fsdp(pt_net)
        # load the weight loaded fsdp net to the original net
        net.load_state_dict(pt_net.state_dict(), strict=True, assign=False)
        log.info(f"==========Loaded PT checkpoint to {net} successfully.")

    def _load_dcp_to_net(
        self, net: torch.nn.Module, ckpt_path: str, prefix: str = "net_ema", credential_path: str | None = None
    ) -> None:
        """Load a DCP checkpoint into a single network."""
        if (
            credential_path is None
            and hasattr(self.config, "teacher_load_from")
            and self.config.teacher_load_from is not None
        ):
            credential_path = self.config.teacher_load_from.credentials

        storage_reader = get_storage_reader(ckpt_path, credential_path)
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
            _new_state_dict[f"{prefix}.{k}"] = _state_dict[k]
        dcp.load(_new_state_dict, storage_reader=storage_reader, planner=DefaultLoadPlanner(allow_partial_load=True))
        for k in _state_dict.keys():
            _state_dict[k] = _new_state_dict[f"{prefix}.{k}"]

        log.info(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=False)))
        del _state_dict, _new_state_dict

    # ------------------------ Optimizers & schedulers ------------------------
    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Create optimizers/schedulers for student, fake_score, and discriminator (if present)."""
        # Student optimizer/scheduler from base class
        net_optimizer, net_scheduler = super().init_optimizer_scheduler(optimizer_config, scheduler_config)

        # Fake-score optimizer/scheduler (required for DMD2)
        fake_score_optimizer = lazy_instantiate(self.config.optimizer_fake_score_config, model=self.net_fake_score)
        fake_score_scheduler = get_base_scheduler(fake_score_optimizer, self, scheduler_config)
        self.optimizer_dict["fake_score"] = fake_score_optimizer
        self.scheduler_dict["fake_score"] = fake_score_scheduler

        # Optional discriminator optimizer/scheduler
        if self.net_discriminator_head:
            discriminator_optimizer = lazy_instantiate(
                self.config.optimizer_discriminator_config,
                model=self.net_discriminator_head,
            )
            discriminator_scheduler = get_base_scheduler(discriminator_optimizer, self, scheduler_config)
            self.optimizer_dict["discriminator"] = discriminator_optimizer
            self.scheduler_dict["discriminator"] = discriminator_scheduler

        return net_optimizer, net_scheduler

    def is_student_phase(self, iteration: int) -> bool:
        """Return True when we are in the student update phase."""
        return (
            self.net_fake_score is None
            or iteration < self.config.warmup_steps
            or iteration % self.config.student_update_freq == 0
        )

    def get_effective_iteration(self, iteration: int) -> int:
        """Effective student iteration index used for EMA scheduling."""
        if self.net_fake_score is None or iteration < self.config.warmup_steps:
            return iteration
        return self.config.warmup_steps + (iteration - self.config.warmup_steps) // self.config.student_update_freq

    def get_effective_iteration_fake(self, iteration: int) -> int:
        """Effective critic/fake-score iteration index."""
        return iteration - self.get_effective_iteration(iteration) - 1

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        """Get the optimizers for the current iteration based on student/critic phase."""
        if self.is_student_phase(iteration):
            return [self.optimizer_dict["net"]]
        if self.config.loss_scale_GAN_generator > 0 and self.net_discriminator_head:
            return [self.optimizer_dict["fake_score"], self.optimizer_dict["discriminator"]]
        return [self.optimizer_dict["fake_score"]]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """Get the LR schedulers for the current iteration based on student/critic phase."""
        if self.is_student_phase(iteration):
            return [self.scheduler_dict["net"]]
        if self.config.loss_scale_GAN_generator > 0 and self.net_discriminator_head:
            return [self.scheduler_dict["fake_score"], self.scheduler_dict["discriminator"]]
        return [self.scheduler_dict["fake_score"]]

    # ------------------------ Core denoise function ------------------------
    def _apply_video_condition_mask(
        self,
        tensor_B_C_T_H_W: torch.Tensor,
        condition: Video2WorldCondition,
    ) -> torch.Tensor:
        """Apply video conditioning mask to zero out conditional frames."""
        if not condition.is_video:
            return tensor_B_C_T_H_W

        _, C, _, _, _ = tensor_B_C_T_H_W.shape
        condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
            tensor_B_C_T_H_W
        )
        return tensor_B_C_T_H_W * (1 - condition_video_mask)

    def denoise(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        noise_level: torch.Tensor,
        condition: Video2WorldCondition,
        net_type: Literal["teacher", "fake_score", "student"] = "student",
    ) -> DenoisePrediction:
        """
        Network forward to denoise the input noised data given noise level, and condition.

        The math formulation follows the generalized EDM formulation using the c_in, c_out, c_skip coefficients.
        By setting these coefficients differently (see self.scaling), this denoiser function supports different
        parameterizations of the noise level: EDM, Rectified Flow, TrigFlow, etc.

        Args:
            xt_B_C_T_H_W (torch.Tensor): The input noised-input data.
            noise_level (torch.Tensor): The noise level under the chosen parameterization. Generalized to support different parameterizations:
                - EDM: this is the sigma_B_T. Range from 0 to 1.
                - Rectified Flow: this is the normalized 'time' parameter. Range from 0 to 1.
                - TrigFlow: this is the angle (also called 'time') parameter in the trigonometry parameterization. Range from 0 to pi/2.
            condition (Video2WorldCondition): contains the GT frames, text embeddings, the conditiong frames & frame masks, etc.

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred) or velocity prediction (v_pred) if necessary.
        """
        B, C, T, H, W = xt_B_C_T_H_W.shape
        net = self.denoiser_nets[net_type]
        # Whether the network returns intermediate features from given block IDs
        # used by the fake score net to branch out the discriminator head.
        net_return_interm_feat = net_type == "fake_score" and getattr(self, "intermediate_feature_ids", None)

        # Prep noise level param for network input
        if noise_level.ndim == 1:
            noise_level_B_T = repeat(noise_level, "b -> b t", t=T)
        elif noise_level.ndim == 2:
            noise_level_B_T = noise_level
        else:
            raise ValueError(f"noise_level shape {noise_level.shape} is not supported")
        noise_level_B_1_T_1_1 = rearrange(noise_level_B_T, "b t -> b 1 t 1 1")

        # Convert noise level time to EDM-formulation coefficients
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.trigflow_scaler(
            noise_level_B_1_T_1_1
        )

        # Prepare noised signal for network input (aka preconditioning)
        net_state_in_B_C_T_H_W = xt_B_C_T_H_W * c_in_B_1_T_1_1

        # Apply vid2vid conditioning: replace chosen frames with GT frames as conditioning frames
        net_state_in_B_C_T_H_W, c_noise_B_1_T_1_1, condition_video_mask = self._apply_vid2vid_conditioning(
            net_state_in_B_C_T_H_W=net_state_in_B_C_T_H_W,
            c_noise_B_1_T_1_1=c_noise_B_1_T_1_1,
            condition=condition,
            num_channels=C,
            noise_level_B_1_T_1_1=noise_level_B_1_T_1_1,
        )

        # Denoiser net forward pass
        net_out = net(
            x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(
                **self.tensor_kwargs
            ),  # Match model precision to avoid dtype mismatch with FSDP
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(
                **self.tensor_kwargs
            ),  # Keep FP32 for numerical stability in timestep embeddings
            intermediate_feature_ids=self.config.intermediate_feature_ids if net_return_interm_feat else None,
            **condition.to_dict(),
        )
        if net_return_interm_feat:
            net_output_B_C_T_H_W, intermediate_features_outputs = net_out
        else:
            net_output_B_C_T_H_W, intermediate_features_outputs = net_out, []
        net_output_B_C_T_H_W = net_output_B_C_T_H_W.to(dtype=xt_B_C_T_H_W.dtype)

        # Reconstruction of x0 following generalized EDM formulation
        # Note: compatible with Rectified Flow if the c_* coefficients are set to rectified flow scaling
        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W

        # Replace GT on conditioned frames to avoid training on pinned frames (parity with base Video2WorldModel)
        if condition.is_video and self.config.replace_cond_output_with_gt:
            gt_frames = condition.gt_frames.type_as(x0_pred_B_C_T_H_W)
            condition_video_mask = condition_video_mask.type_as(x0_pred_B_C_T_H_W)
            x0_pred_B_C_T_H_W = gt_frames * condition_video_mask + x0_pred_B_C_T_H_W * (1 - condition_video_mask)

        # F prediction (TrigFlow, similar to velocity pred in Rectified Flow) for student and teacher
        F_pred_B_C_T_H_W = (torch.cos(noise_level_B_1_T_1_1) * xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / (
            torch.sin(noise_level_B_1_T_1_1) * self.sigma_data
        )

        return DenoisePrediction(
            x0=x0_pred_B_C_T_H_W, F=F_pred_B_C_T_H_W, intermediate_features=intermediate_features_outputs
        )

    def backward_simulation(
        self,
        condition: Video2WorldCondition,
        init_noise: torch.Tensor,
        n_steps: int,
        with_grad: bool = False,
        dump_iter: int | None = None,
    ):
        """
        Performs the backward (denoising) process with the student net to get the noisy
        examples x_t'. See Sec. 4.5 of https://arxiv.org/pdf/2405.14867.

        Works with EDM-scaling parameterization.
        """
        log.info(f"backward_simulation, n_steps: {n_steps}")
        t_steps = self.config.selected_sampling_time[:n_steps] + [0]
        _ones = torch.ones(init_noise.shape[0]).to(**self.tensor_kwargs)
        x = init_noise
        for count, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            context_fn = torch.enable_grad if with_grad and count == n_steps - 1 else torch.no_grad
            with context_fn():
                x = self.denoise(x, t_cur * _ones, condition, net_type="student").x0
            if t_next > 1e-5:
                x = math.cos(t_next) * x / self.sigma_data + math.sin(t_next) * init_noise

        # save backward simulation video for debugging
        if dump_iter is not None:
            video = self.decode(x)
            uid = uuid.uuid4()
            save_img_or_video((1.0 + video[0]) / 2, f"out-{dump_iter:06d}-{uid}", fps=10)

        return x.float()

    # ------------------------ Training step ------------------------
    def _setup_grad_requirements(self, iteration: int) -> None:
        if self.is_student_phase(iteration):
            # update the student
            self.net.train().requires_grad_(True)
            if self.net.use_crossattn_projection:
                self.net.crossattn_proj.requires_grad_(False)
            if self.net_fake_score:
                self.net_fake_score.eval().requires_grad_(False)
            if self.net_discriminator_head:
                self.net_discriminator_head.eval().requires_grad_(False)
        else:
            # update the fake_score and discriminator
            self.net.eval().requires_grad_(False)
            if self.net_fake_score:
                self.net_fake_score.train().requires_grad_(True)
                if self.net_fake_score.use_crossattn_projection:
                    self.net_fake_score.crossattn_proj.requires_grad_(False)
            if self.net_discriminator_head:
                self.net_discriminator_head.train().requires_grad_(True)

    def single_train_step(
        self, data_batch: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the diffusion model.

        This method is model-agnostic and delegates the core losses to
        `training_step_generator` and `training_step_critic`, which should
        be implemented by the inheriting EDM/RF model classes.
        """
        # Update stats on how many videos the model has seen
        self._update_train_stats(data_batch)

        # Get the input data to noise and denoise and the corresponding conditioner.
        _, x0_B_C_T_H_W, condition, uncondition = self.get_data_and_condition(data_batch)

        # Freeze / unfreeze networks according to current phase
        self._setup_grad_requirements(iteration)

        if self.is_student_phase(iteration):
            output_batch, loss = self.training_step_generator(x0_B_C_T_H_W, condition, uncondition, iteration)
        else:
            output_batch, loss = self.training_step_critic(x0_B_C_T_H_W, condition, uncondition, iteration)
        loss = loss.mean()  # each loss term has been separately scaled
        return output_batch, loss

    def training_step_generator(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: Video2WorldCondition,
        uncondition: Video2WorldCondition,
        iteration: int,
    ):
        """
        A naive impl of DMD for distillation. https://arxiv.org/pdf/2311.18828.
        Note: assume student is initialized (warmed-up) with noise-clean_data pair or consistency distillation.
        To Tune:
        - how to sample, time_B_T
        - weights for normalization and different loss weights based on time_B_T

        Notation:
            - with G_ prefix: input/output of student net (generator).
            - with D_ prefix: input/output of the critic nets fake score net, teacher net, and optionally discriminator.
        """
        # Use the critic net's time to sample noise level because the DMD loss comes fromt he critic net's grad.
        D_time_B_T = self.draw_training_time_critic(x0_B_C_T_H_W.shape, condition)
        G_epsilon_B_C_T_H_W, D_epsilon_B_C_T_H_W = torch.randn_like(x0_B_C_T_H_W), torch.randn_like(x0_B_C_T_H_W)
        (
            G_epsilon_B_C_T_H_W,
            condition,
            uncondition,
            D_epsilon_B_C_T_H_W,
            D_time_B_T,
        ) = self.broadcast_split_for_model_parallelsim(
            G_epsilon_B_C_T_H_W, condition, uncondition, D_epsilon_B_C_T_H_W, D_time_B_T
        )

        n_steps = torch.randint(
            low=0, high=len(self.config.selected_sampling_time), size=(1,), device=self.tensor_kwargs["device"]
        )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(n_steps, src=0)
        n_steps = int(n_steps.item()) + 1

        dump_iter = None
        if self.config.vis_debug and torch.distributed.get_rank() == 0:
            if iteration % 100 == 0:
                dump_iter = iteration

        # Generate student's few-step output G_x0_theta with gradients on the
        # last step (simulates inference-time few-step sampling).
        G_x0_theta_B_C_T_H_W = self.backward_simulation(
            condition, G_epsilon_B_C_T_H_W, n_steps, with_grad=True, dump_iter=dump_iter
        )

        # Re-noise student output to construct input to the discriminator
        # Discriminator is the fake score net, uses its intermediate feature and run GAN loss
        D_time_B_1_T_1_1 = rearrange(D_time_B_T, "b t -> b 1 t 1 1")
        D_cost_B_1_T_1_1, D_sint_B_1_T_1_1 = torch.cos(D_time_B_1_T_1_1), torch.sin(D_time_B_1_T_1_1)
        D_xt_theta_B_C_T_H_W = (
            G_x0_theta_B_C_T_H_W * D_cost_B_1_T_1_1 / self.sigma_data + D_epsilon_B_C_T_H_W * D_sint_B_1_T_1_1
        )

        # If GAN loss is enabled, need gradient to flow from discriminator logits to student's sample
        # D_xt_theta_B_C_T_H_W. If no GAN loss, turn off grad to save memory.
        context_fn_fake_score = torch.no_grad if self.net_discriminator_head is None else torch.enable_grad
        with context_fn_fake_score():
            fake_pred = self.denoise(D_xt_theta_B_C_T_H_W, D_time_B_T, condition, net_type="fake_score")
        x0_theta_fake_B_C_T_H_W, intermediate_features_outputs = fake_pred.x0, fake_pred.intermediate_features

        # Same noised input, get teacher denoising ouput
        with torch.no_grad():
            x0_theta_teacher_B_C_T_H_W = self.denoise(
                D_xt_theta_B_C_T_H_W, D_time_B_T, condition, net_type="teacher"
            ).x0
            if self.config.teacher_guidance > 0.0:
                x0_theta_teacher_B_C_T_H_W_uncond = self.denoise(
                    D_xt_theta_B_C_T_H_W, D_time_B_T, uncondition, net_type="teacher"
                ).x0
                x0_theta_teacher_B_C_T_H_W = x0_theta_teacher_B_C_T_H_W + self.config.teacher_guidance * (
                    x0_theta_teacher_B_C_T_H_W - x0_theta_teacher_B_C_T_H_W_uncond
                )

        # Per-sample (new in our sCM2, not in DMD) normalization weight to stablize grad scale
        # Mask out conditional frames to avoid corrupting the loss normalization
        with torch.no_grad():
            diff = torch.abs(G_x0_theta_B_C_T_H_W.double() - x0_theta_teacher_B_C_T_H_W.double())

            if condition.is_video:
                _, C, _, _, _ = G_x0_theta_B_C_T_H_W.shape
                condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                    diff
                )
                diff_masked = diff * (1 - condition_video_mask)
                num_non_cond_pixels = (1 - condition_video_mask).sum(dim=[1, 2, 3, 4], keepdim=True).clip(min=1)
                weight_factor = (diff_masked.sum(dim=[1, 2, 3, 4], keepdim=True) / num_non_cond_pixels).clip(
                    min=0.00001
                )
            else:
                weight_factor = diff.mean(dim=[1, 2, 3, 4], keepdim=True).clip(min=0.00001)

        # DMD's distribution matching loss by computing the the diff of teacher score and fake score
        # the grad of score func is the prediction from both nets
        # Mask out conditional frames from gradient computation
        grad_B_C_T_H_W = x0_theta_fake_B_C_T_H_W.double() - x0_theta_teacher_B_C_T_H_W.double()
        grad_B_C_T_H_W = self._apply_video_condition_mask(grad_B_C_T_H_W, condition)
        grad_B_C_T_H_W = grad_B_C_T_H_W / weight_factor
        # trick to let gradient flow into student only: current formulation let the value
        # of d (loss_dmd)/dG_x0 equal to grad_B_C_T_H_W (up to a constant). but since grad_B_C_T_H_W is detached,
        # gradient doesn't flow into teacher / fake score for this loss.
        loss_dmd = (G_x0_theta_B_C_T_H_W.double() - (G_x0_theta_B_C_T_H_W.double() - grad_B_C_T_H_W).detach()) ** 2
        loss_dmd[torch.isnan(loss_dmd).flatten(start_dim=1).any(dim=1)] = 0
        loss_dmd = loss_dmd.mean(dim=(1, 2, 3, 4))
        total_generator_loss = self.config.loss_scale_sid * loss_dmd

        if self.net_discriminator_head:
            logits_theta_B = self.net_discriminator_head(intermediate_features_outputs)[:, 0].float()  # type: ignore
            # train generator with BCE(fake, 1) gan loss to push generator generate like-real data
            loss_gan = F.binary_cross_entropy_with_logits(
                logits_theta_B, torch.ones_like(logits_theta_B), reduction="none"
            )
            loss_gan = torch.nan_to_num(loss_gan)

            total_generator_loss += self.config.loss_scale_GAN_generator * loss_gan
        else:
            loss_gan = 0.0

        return {
            "grad_B_C_T_H_W": grad_B_C_T_H_W.detach(),
            "dmd_loss_generator": total_generator_loss,
            "dmd_loss": loss_dmd,
            "gan_loss": loss_gan,
        }, total_generator_loss

    def training_step_critic(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: Video2WorldCondition,
        uncondition: Video2WorldCondition,
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the fake score net (critic) and optionally the discriminator head.
        """
        D_time_B_T = self.draw_training_time_critic(x0_B_C_T_H_W.shape, condition)
        G_epsilon_B_C_T_H_W, D_epsilon_B_C_T_H_W = torch.randn_like(x0_B_C_T_H_W), torch.randn_like(x0_B_C_T_H_W)
        (
            G_epsilon_B_C_T_H_W,
            condition,
            uncondition,
            D_epsilon_B_C_T_H_W,
            D_time_B_T,
        ) = self.broadcast_split_for_model_parallelsim(
            G_epsilon_B_C_T_H_W, condition, uncondition, D_epsilon_B_C_T_H_W, D_time_B_T
        )

        if self.net.is_context_parallel_enabled and self.net_fake_score is not None:  # type: ignore
            # need x0 for discrimiator loss
            x0_B_C_T_H_W = broadcast_split_tensor(
                tensor=x0_B_C_T_H_W, seq_dim=2, process_group=self.get_context_parallel_group()
            )

        n_steps = torch.randint(
            low=0, high=len(self.config.selected_sampling_time), size=(1,), device=self.tensor_kwargs["device"]
        )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(n_steps, src=0)
        n_steps = int(n_steps.item()) + 1

        # Generate student output G_x0_theta via backward_simulation with gradients on the
        # last step (simulates inference-time few-step sampling).
        G_x0_theta_B_C_T_H_W = self.backward_simulation(condition, G_epsilon_B_C_T_H_W, n_steps, with_grad=False)

        # Re-noise student output to construct input to the discriminator
        # Discriminator is the fake score net, uses its intermediate feature and run GAN loss
        D_time_B_1_T_1_1 = rearrange(D_time_B_T, "b t -> b 1 t 1 1")
        D_cost_B_1_T_1_1, D_sint_B_1_T_1_1 = torch.cos(D_time_B_1_T_1_1), torch.sin(D_time_B_1_T_1_1)
        D_xt_theta_B_C_T_H_W = (
            G_x0_theta_B_C_T_H_W * D_cost_B_1_T_1_1 / self.sigma_data + D_epsilon_B_C_T_H_W * D_sint_B_1_T_1_1
        )

        fake_denoise_prediction = self.denoise(D_xt_theta_B_C_T_H_W, D_time_B_T, condition, net_type="fake_score")
        x0_theta_fake_B_C_T_H_W = fake_denoise_prediction.x0
        intermediate_features_outputs = fake_denoise_prediction.intermediate_features

        # Denoising loss for the fake score net
        # Mask out conditional frames to avoid training critic on pinned frames
        diff_B_C_T_H_W = G_x0_theta_B_C_T_H_W - x0_theta_fake_B_C_T_H_W
        diff_B_C_T_H_W = self._apply_video_condition_mask(diff_B_C_T_H_W, condition)
        loss_fake_score = self.config.loss_scale_fake_score * (diff_B_C_T_H_W**2 / D_sint_B_1_T_1_1**2).mean(
            dim=(1, 2, 3, 4)
        )
        total_critic_loss = loss_fake_score

        if self.net_discriminator_head is not None:
            logits_theta_B = self.net_discriminator_head(intermediate_features_outputs)[:, 0].float()  # type: ignore

            # Prepare real data instance to the discriminator.
            # discriminator's logits = first_few_layers_of_fake_score_net(real_data) --> discriminator head
            xt_real_B_C_T_H_W = (
                x0_B_C_T_H_W * D_cost_B_1_T_1_1 / self.sigma_data + D_epsilon_B_C_T_H_W * D_sint_B_1_T_1_1
            )
            intermediate_features_outputs_real = self.denoise(
                xt_real_B_C_T_H_W, D_time_B_T, condition, net_type="fake_score"
            ).intermediate_features
            logits_real_B = self.net_discriminator_head(intermediate_features_outputs_real)[:, 0].float()  # type: ignore

            # train discriminator with BCE(real, 1) + BCE(fake, 0)
            loss_gan = F.binary_cross_entropy_with_logits(
                logits_real_B, torch.ones_like(logits_real_B), reduction="none"
            ) + F.binary_cross_entropy_with_logits(logits_theta_B, torch.zeros_like(logits_theta_B), reduction="none")
            loss_gan = torch.nan_to_num(loss_gan)

            total_critic_loss += self.config.loss_scale_GAN_discriminator * loss_gan
        else:
            loss_gan = 0.0

        output_batch = {
            "x0_pred": G_x0_theta_B_C_T_H_W * self.sigma_data,
            "dmd_loss_critic": total_critic_loss,
            "dmd_loss": loss_fake_score,
            "gan_loss": loss_gan,
        }
        return output_batch, total_critic_loss

    # ------------------------ Checkpointing helpers ------------------------
    def state_dict(self) -> Dict[str, Any]:
        net_state_dict = super().state_dict()
        fake_score_state_dict = self.net_fake_score.state_dict(prefix="net_fake_score.")
        net_state_dict.update(fake_score_state_dict)

        if self.net_discriminator_head:
            discriminator_state_dict = self.net_discriminator_head.state_dict(prefix="net_discriminator_head.")
            net_state_dict.update(discriminator_state_dict)
        return net_state_dict

    def model_dict(self) -> Dict[str, Any]:
        model_dict = super().model_dict()
        model_dict["fake_score"] = self.net_fake_score
        if self.net_discriminator_head:
            model_dict["discriminator"] = self.net_discriminator_head
        return model_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """Load weights for student/EMA (via base class) and for fake_score/discriminator heads."""
        base_results: _IncompatibleKeys | None = None
        # Load the student net and its EMA weights from the base class.
        if strict:
            base_results = super().load_state_dict(state_dict, strict=strict, assign=assign)
        else:
            super().load_state_dict(state_dict, strict=strict, assign=assign)

        # Load the fake score and discriminator heads.
        fake_score_state_dict = collections.OrderedDict()
        discriminator_state_dict = collections.OrderedDict()

        for k, v in state_dict.items():
            if k.startswith("net_fake_score."):
                fake_score_state_dict[k.replace("net_fake_score.", "")] = v
            elif k.startswith("net_discriminator_head."):
                discriminator_state_dict[k.replace("net_discriminator_head.", "")] = v

        if strict:
            assert base_results is not None
            missing_keys: list[str] = list(base_results.missing_keys)
            unexpected_keys: list[str] = list(base_results.unexpected_keys)

            fake_score_results: _IncompatibleKeys = self.net_fake_score.load_state_dict(
                fake_score_state_dict, strict=True, assign=assign
            )
            missing_keys += fake_score_results.missing_keys
            unexpected_keys += fake_score_results.unexpected_keys

            if self.net_discriminator_head and discriminator_state_dict:
                discriminator_results: _IncompatibleKeys = self.net_discriminator_head.load_state_dict(
                    discriminator_state_dict, strict=True, assign=assign
                )
                missing_keys += discriminator_results.missing_keys
                unexpected_keys += discriminator_results.unexpected_keys

            return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=unexpected_keys)

        # Non-strict branch: rely on non_strict_load_model for extra heads.
        log.critical("load fake score model in non-strict mode")
        log.critical(str(non_strict_load_model(self.net_fake_score, fake_score_state_dict)), rank0_only=False)

        if self.net_discriminator_head and discriminator_state_dict:
            log.critical("load discriminator model in non-strict mode")
            log.critical(
                str(non_strict_load_model(self.net_discriminator_head, discriminator_state_dict)),
                rank0_only=False,
            )

    def broadcast_split_for_model_parallelsim(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: Video2WorldCondition,
        uncondition: Video2WorldCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        time_B_T: torch.Tensor,
    ):
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            self.net_teacher.enable_context_parallel(cp_group)
            self.net_fake_score.enable_context_parallel(cp_group)
        else:
            self.net_teacher.disable_context_parallel()
            self.net_fake_score.disable_context_parallel()

        return super().broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, uncondition, epsilon_B_C_T_H_W, time_B_T
        )
