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

import math
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict2._src.common.types.high_sigma_strategy import HighSigmaStrategy as HighSigmaStrategy
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.text2world_model import (
    DenoisePrediction,
    Text2WorldCondition,
    Text2WorldModelConfig,
)
from cosmos_predict2._src.predict2.models.text2world_model import DiffusionModel as Text2WorldModel

NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


class ConditioningStrategy(str, Enum):
    FRAME_REPLACE = "frame_replace"  # First few frames of the video are replaced with the conditional frames

    def __str__(self) -> str:
        return self.value


@attrs.define(slots=False)
class Video2WorldConfig(Text2WorldModelConfig):
    min_num_conditional_frames: int = 1  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    sigma_conditional: float = 0.0001  # Noise level used for conditional frames
    conditioning_strategy: str = str(ConditioningStrategy.FRAME_REPLACE)  # What strategy to use for conditioning
    denoise_replace_gt_frames: bool = True  # Whether to denoise the ground truth frames
    high_sigma_strategy: str = str(HighSigmaStrategy.UNIFORM80_2000)  # What strategy to use for high sigma
    high_sigma_ratio: float = 0.05  # Ratio of high sigma frames
    low_sigma_ratio: float = 0.05  # Ratio of low sigma frames
    conditional_frames_probs: Optional[Dict[int, float]] = None  # Probability distribution for conditional frames

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.conditioning_strategy in [
            str(ConditioningStrategy.FRAME_REPLACE),
        ]
        assert self.high_sigma_strategy in [
            str(HighSigmaStrategy.NONE),
            str(HighSigmaStrategy.UNIFORM80_2000),
            str(HighSigmaStrategy.LOGUNIFORM200_100000),
            str(HighSigmaStrategy.BALANCED_TWO_HEADS_V1),
            str(HighSigmaStrategy.SHIFT24),
            str(HighSigmaStrategy.HARDCODED_20steps),
        ]


LOG_200 = math.log(200)
LOG_100000 = math.log(100000)


class Video2WorldModel(Text2WorldModel):
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Video2WorldCondition]:
        # generate random number of conditional frames for training
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return raw_state, latent_state, condition

    def draw_training_sigma_and_epsilon(self, x0_size: int, condition: Any) -> torch.Tensor:
        sigma_B_1, epsilon = super().draw_training_sigma_and_epsilon(x0_size, condition)
        is_video_batch = condition.data_type == DataType.VIDEO
        # if is_video_batch, with 5% ratio, we regenerate sigma_B_1 with uniformally from 80 to 2000
        # with remaining 95% ratio, we keep the original sigma_B_1
        if is_video_batch:
            if self.config.high_sigma_strategy == str(HighSigmaStrategy.UNIFORM80_2000):
                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
                new_sigma = torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * 1920 + 80
                sigma_B_1 = torch.where(mask, new_sigma, sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.LOGUNIFORM200_100000):
                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
                log_new_sigma = (
                    torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * (LOG_100000 - LOG_200)
                    + LOG_200
                )
                sigma_B_1 = torch.where(mask, log_new_sigma.exp(), sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.SHIFT24):
                # sample t from uniform distribution between 0 and 1, with same shape as sigma_B_1
                _t = torch.rand(sigma_B_1.shape, device=sigma_B_1.device).double()
                _t = 24 * _t / (24 * _t + 1 - _t)
                sigma_B_1 = (_t / (1.0 - _t)).float()

                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
                new_sigma = torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * 1920 + 80
                sigma_B_1 = torch.where(mask, new_sigma, sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.BALANCED_TWO_HEADS_V1):
                # replace high sigma parts
                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
                log_new_sigma = (
                    torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * (LOG_100000 - LOG_200)
                    + LOG_200
                )
                sigma_B_1 = torch.where(mask, log_new_sigma.exp(), sigma_B_1)
                # replace low sigma parts
                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.low_sigma_ratio
                low_sigma_B_1 = torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * 2.0 + 0.00001
                sigma_B_1 = torch.where(mask, low_sigma_B_1, sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.HARDCODED_20steps):
                if not hasattr(self, "hardcoded_20steps_sigma"):
                    from cosmos_predict2._src.common.modules.res_sampler import get_rev_ts

                    hardcoded_20steps_sigma = get_rev_ts(
                        t_min=self.sde.sigma_min, t_max=self.sde.sigma_max, num_steps=20, ts_order=7.0
                    )
                    # add extra 100000 to the beginning
                    self.hardcoded_20steps_sigma = torch.cat(
                        [torch.tensor([100000.0], device=hardcoded_20steps_sigma.device), hardcoded_20steps_sigma],
                        dim=0,
                    )
                sigma_B_1 = self.hardcoded_20steps_sigma[
                    torch.randint(0, len(self.hardcoded_20steps_sigma), sigma_B_1.shape)
                ].type_as(sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.NONE):
                pass
            else:
                raise ValueError(f"High sigma strategy {self.config.high_sigma_strategy} is not supported")
        return sigma_B_1, epsilon

    def denoise_with_velocity(
        self, noise_x_in_t_space: torch.Tensor, t_B_T: torch.Tensor, condition: Text2WorldCondition
    ) -> torch.Tensor:
        """
        This function is used when self.config.use_flowunipc_scheduler is set.
        """
        if t_B_T.ndim == 1:
            t_B_T = rearrange(t_B_T, "b -> b 1")
        elif t_B_T.ndim == 2:
            t_B_T = t_B_T
        else:
            raise ValueError(f"t_B_T shape {t_B_T.shape} is not supported")
        # our model expects input of sigma and x_sigma, so convert t -> sigma, x_t to x_sigma
        sigma_B_T = t_B_T / (1.0 - t_B_T)
        x_B_C_T_H_W_in_sigma_space = noise_x_in_t_space * (1.0 + rearrange(sigma_B_T, "b t -> b 1 t 1 1"))
        denoise_output_B_C_T_H_W = self.denoise(x_B_C_T_H_W_in_sigma_space, sigma_B_T, condition)
        x0_pred_B_C_T_H_W = denoise_output_B_C_T_H_W.x0
        eps_pred_B_C_T_H_W = denoise_output_B_C_T_H_W.eps
        return eps_pred_B_C_T_H_W - x0_pred_B_C_T_H_W

    def denoise(
        self, xt_B_C_T_H_W: torch.Tensor, sigma: torch.Tensor, condition: Text2WorldCondition
    ) -> DenoisePrediction:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (Text2WorldCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """

        if sigma.ndim == 1:
            sigma_B_T = rearrange(sigma, "b -> b 1")
        elif sigma.ndim == 2:
            sigma_B_T = sigma
        else:
            raise ValueError(f"sigma shape {sigma.shape} is not supported")

        sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")
        # get precondition for the network
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling(sigma=sigma_B_1_T_1_1)

        net_state_in_B_C_T_H_W = xt_B_C_T_H_W * c_in_B_1_T_1_1

        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(net_state_in_B_C_T_H_W) / self.config.sigma_data
            if not condition.use_video_condition:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                net_state_in_B_C_T_H_W
            )

            # Replace the first few frames of the video with the conditional frames
            # Update the c_noise as the conditional frames are clean and have very low noise

            # Make the first few frames of x_t be the ground truth frames
            net_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + net_state_in_B_C_T_H_W * (
                1 - condition_video_mask
            )
            # Adjust c_noise for the conditional frames
            sigma_cond_B_1_T_1_1 = torch.ones_like(sigma_B_1_T_1_1) * self.config.sigma_conditional
            _, _, _, c_noise_cond_B_1_T_1_1 = self.scaling(sigma=sigma_cond_B_1_T_1_1)
            condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
            c_noise_B_1_T_1_1 = c_noise_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + c_noise_B_1_T_1_1 * (
                1 - condition_video_mask_B_1_T_1_1
            )

        # forward pass through the network
        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(
                **{
                    **self.tensor_kwargs,
                    "dtype": torch.float32 if self.config.use_wan_fp32_strategy else self.tensor_kwargs["dtype"],
                },
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        ).float()

        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W
        if condition.is_video and self.config.denoise_replace_gt_frames:
            # Set the first few frames to the ground truth frames. This will ensure that the loss is not computed for the first few frames.
            x0_pred_B_C_T_H_W = condition.gt_frames.type_as(
                x0_pred_B_C_T_H_W
            ) * condition_video_mask + x0_pred_B_C_T_H_W * (1 - condition_video_mask)

        # get noise prediction based on sde
        eps_pred_B_C_T_H_W = (xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / sigma_B_1_T_1_1

        return DenoisePrediction(x0_pred_B_C_T_H_W, eps_pred_B_C_T_H_W, None)

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            if self.config.use_flowunipc_scheduler:
                cond_velocity = self.denoise_with_velocity(noise_x, sigma, condition)
                uncond_velocity = self.denoise_with_velocity(noise_x, sigma, uncondition)
                velocity = uncond_velocity + guidance * (cond_velocity - uncond_velocity)
                return velocity
            cond_x0 = self.denoise(noise_x, sigma, condition).x0
            uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
            raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
            if "guided_image" in data_batch:
                # replacement trick that enables inpainting with base model
                assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch["guided_image"]
                guide_mask = data_batch["guided_mask"]
                raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0
            return raw_x0

        return x0_fn
