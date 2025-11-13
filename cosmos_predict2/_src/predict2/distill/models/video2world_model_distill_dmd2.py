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
A dmd2 distillation ImaginaireModel enabled for Cosmos-Predict2.5 Video2WorldModel.

Features
- Training uses TrigFlow formulation to sample t. Use a wrapper to convert the trigflow t to EDM scaling coefficients.
So all nets' output is still in EDM/Rectified Flow format.
- Consolidated the denoise function for student, teacher, and fake score net.
"""

from __future__ import annotations

import math
import uuid
from typing import Callable, Dict, List, Tuple

import attrs
import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.core import parallel_state

from cosmos_predict2._src.common.modules.denoiser_scaling import EDMScaling, RectifiedFlowScaling
from cosmos_predict2._src.common.modules.edm_sde import EDMSDE
from cosmos_predict2._src.common.modules.res_sampler import COMMON_SOLVER_OPTIONS
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast_split_tensor, cat_outputs_cp
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.distill.models.distillation_base_mixin import (
    BaseDistillConfig,
    DistillationCoreMixin,
    TrigFlowMixin,
)
from cosmos_predict2._src.predict2.models.video2world_model import Video2WorldConfig, Video2WorldModel
from cosmos_predict2._src.predict2.modules.denoiser_scaling import (
    EDM_sCMWrapper,
    RectifiedFlow_sCMWrapper,
)


@attrs.define(slots=False)
class Video2WorldModelDistillationConfigDMD2TrigFlow(BaseDistillConfig, Video2WorldConfig):
    """
    Configuration for DMD2 distillation using TrigFlow parameterization.
    Compatible with both EDM and Rectified Flow teacher models.
    """

    sde: LazyDict = L(EDMSDE)(  # p_mean -0.8 is an empirical best choice
        p_mean=-0.8,
        p_std=1.6,
        sigma_max=80,
        sigma_min=0.0002,
    )
    sde_D: LazyDict = L(EDMSDE)(  # same as base predict2 model
        p_mean=0.0,
        p_std=1.6,
        sigma_max=80,
        sigma_min=0.0002,
    )

    # Selected time below corresponds to a uniformly-spaced 4-step t in RF: [1.0, 0.75, 0.5, 0.25] with a shift of 5
    selected_sampling_time: List[float] = [math.pi / 2, math.atan(15), math.atan(5), math.atan(5 / 3)]

    conditional_frame_timestep: float = (
        -1.0
    )  # The timestep used for the conditional frames (keep consistent with the teacher model to be distilled)
    denoise_replace_gt_frames: bool = True  # Whether to denoise the ground truth frames

    vis_debug: bool = False  # flag for visualizing intermediate results during training


class Video2WorldModelDistillDMD2TrigFlow(DistillationCoreMixin, TrigFlowMixin, Video2WorldModel):
    """
    DMD2 distillation model using TrigFlow parameterization.

    Trains a student model to match the teacher's distribution using distribution matching
    distillation. Uses fake diffusion critic and teacher score to compute
    gradient signals for the student. Optionally uses discriminator head for GAN loss as
    described in DMD2 paper.

    Uses TrigFlow parameterization for noise level sampling, converting to EDM/Rectified Flow scaling
    coefficients internally, hence compatible with both EDM and Rectified Flow teacher models.
    """

    def __init__(self, config: Video2WorldModelDistillationConfigDMD2TrigFlow):
        """
        Args:
            config: DMD2 configuration object. Must use a teacher model trained with
                EDM or EDM-wrapped rectified flow for compatible scaling coefficients.
        """
        self.init_distillation_common(config)
        super().__init__(config)
        # Ensure float32 tensor kwargs exist for mixin utilities that expect it
        if not hasattr(self, "tensor_kwargs_fp32"):
            device = self.tensor_kwargs["device"] if hasattr(self, "tensor_kwargs") else "cuda"
            self.tensor_kwargs_fp32 = {"device": device, "dtype": torch.float32}
        self.config = config

        self.sde = lazy_instantiate(config.sde)
        self.sde_D = lazy_instantiate(config.sde_D)

        self.sigma_data = config.sigma_data
        self.sigma_conditional = config.sigma_conditional

        # Converterfrom Trigflow time to EDM scaling coefficients for student net and fake score net
        self.scaling_from_time = (
            EDM_sCMWrapper(config.sigma_data)
            if config.scaling == "edm"
            else RectifiedFlow_sCMWrapper(config.sigma_data)
        )
        self.scaling_teacher = (
            EDMScaling(self.sigma_data)
            if config.scaling == "edm"
            else RectifiedFlowScaling(self.sigma_data, config.rectified_flow_t_scaling_factor)
        )

        self.selected_sampling_time = config.selected_sampling_time
        log.info(f"==============================Student timesteps (trigflow): {self.selected_sampling_time}")

        self.vis_debug = config.vis_debug

    # ------------------------ training ------------------------

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
        D_time_B_T = self.draw_training_time_D(x0_B_C_T_H_W.shape, condition)
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
        if self.vis_debug and torch.distributed.get_rank() == 0:
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
            if self.teacher_guidance > 0.0:
                x0_theta_teacher_B_C_T_H_W_uncond = self.denoise(
                    D_xt_theta_B_C_T_H_W, D_time_B_T, uncondition, net_type="teacher"
                ).x0
                x0_theta_teacher_B_C_T_H_W = x0_theta_teacher_B_C_T_H_W + self.teacher_guidance * (
                    x0_theta_teacher_B_C_T_H_W - x0_theta_teacher_B_C_T_H_W_uncond
                )

        # Per-sample (new in our sCM2, not in DMD) normalization weight to stablize grad scale
        with torch.no_grad():
            weight_factor = (
                torch.abs(G_x0_theta_B_C_T_H_W.double() - x0_theta_teacher_B_C_T_H_W.double())
                .mean(dim=[1, 2, 3, 4], keepdim=True)
                .clip(min=0.00001)
            )

        # DMD's distribution matching loss by computing the the diff of teacher score and fake score
        # the grad of score func is the prediction from both nets
        grad_B_C_T_H_W = (x0_theta_fake_B_C_T_H_W.double() - x0_theta_teacher_B_C_T_H_W.double()) / weight_factor
        # trick to let gradient flow into student only: current formulation let the value
        # of d (loss_dmd)/dG_x0 equal to grad_B_C_T_H_W (up to a constant). but since grad_B_C_T_H_W is detached,
        # gradient doesn't flow into teacher / fake score for this loss.
        loss_dmd = (G_x0_theta_B_C_T_H_W.double() - (G_x0_theta_B_C_T_H_W.double() - grad_B_C_T_H_W).detach()) ** 2
        loss_dmd[torch.isnan(loss_dmd).flatten(start_dim=1).any(dim=1)] = 0
        loss_dmd = loss_dmd.mean(dim=(1, 2, 3, 4))
        kendall_loss = self.loss_scale_sid * loss_dmd

        if self.net_discriminator_head:
            logits_theta_B = self.net_discriminator_head(intermediate_features_outputs)[:, 0].float()  # type: ignore
            # train generator with BCE(fake, 1) gan loss to push generator generate like-real data
            loss_gan = F.binary_cross_entropy_with_logits(
                logits_theta_B, torch.ones_like(logits_theta_B), reduction="none"
            )
            loss_gan = torch.nan_to_num(loss_gan)

            kendall_loss += self.loss_scale_GAN_generator * loss_gan
        else:
            loss_gan = 0.0

        return {
            "grad_B_C_T_H_W": grad_B_C_T_H_W.detach(),
            "dmd_loss_generator": kendall_loss,
            "dmd_loss": kendall_loss,
            "gan_loss": loss_gan,
        }, kendall_loss

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
        D_time_B_T = self.draw_training_time_D(x0_B_C_T_H_W.shape, condition)
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
        kendall_loss = self.loss_scale_fake_score * (
            (G_x0_theta_B_C_T_H_W - x0_theta_fake_B_C_T_H_W) ** 2 / D_sint_B_1_T_1_1**2
        ).mean(dim=(1, 2, 3, 4))

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

            kendall_loss += self.loss_scale_GAN_discriminator * loss_gan
        else:
            loss_gan = 0.0

        output_batch = {
            "x0_pred": G_x0_theta_B_C_T_H_W * self.sigma_data,
            "dmd_loss_critic": kendall_loss,
            "dmd_loss": kendall_loss,
            "gan_loss": loss_gan,
        }
        return output_batch, kendall_loss

    # ------------------------ Sampling ------------------------
    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch.

        Note: For distilled models, there is no 'guidance' parameter as teacher guidance is distilled into the student model.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 prediction

        """
        if "num_conditional_frames" in data_batch:
            num_conditional_frames = data_batch["num_conditional_frames"]
        else:
            num_conditional_frames = 0

        _, x0, condition, _ = self.get_data_and_condition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=0,  # inference time will use the fixed num_conditional_frames
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)

        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            condition = condition.broadcast(cp_group)

        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        @torch.no_grad()
        def x0_fn(noise_x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            raw_x0 = self.denoise(noise_x, time, condition, net_type="student").x0
            return raw_x0

        return x0_fn

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
        init_noise: torch.Tensor = None,
        mid_t: List[float] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            solver_option (str): differential equation solver option, default to "2ab"~(mulitstep solver)
        """
        del kwargs
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]  # type: ignore

        x0_fn = self.get_x0_fn_from_batch(data_batch)

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)

        if init_noise is None:
            init_noise = torch.randn(
                n_sample,
                *state_shape,
                dtype=torch.float32,
                device=self.tensor_kwargs["device"],
                generator=generator,
            )

        if self.net.is_context_parallel_enabled:  # type: ignore
            init_noise = broadcast_split_tensor(init_noise, seq_dim=2, process_group=self.get_context_parallel_group())

        # Sampling steps
        x = init_noise.to(torch.float64)
        ones = torch.ones(x.size(0), device=x.device, dtype=x.dtype)
        t_steps = self.config.selected_sampling_time[:num_steps] + [
            0,
        ]
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            x = x0_fn(x.float(), t_cur * ones).to(torch.float64)
            if t_next > 1e-5:
                x = math.cos(t_next) * x / self.sigma_data + math.sin(t_next) * init_noise
        samples = x.float()
        if self.net.is_context_parallel_enabled:  # type: ignore
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())
        return torch.nan_to_num(samples)
