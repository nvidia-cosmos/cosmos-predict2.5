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
Cosmos Policy Video2World Model - extends CosmosPolicyDiffusionModel with video conditioning.

IMPORTANT: This class inherits from CosmosPolicyDiffusionModel (not Video2WorldModel)
to ensure it gets all the policy-specific functionality (training_step, compute_loss, etc.)
"""

import math
from typing import Callable, Dict, Optional, Tuple

import attrs
import torch
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.cosmos_policy.conditioner import Text2WorldCondition
from cosmos_predict2._src.predict2.cosmos_policy.config.conditioner.video2world_conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.cosmos_policy.models.policy_text2world_model_rectified_flow import (
    CosmosPolicyDiffusionModelRectifiedFlow,
    CosmosPolicyModelConfigRectifiedFlow,
    replace_latent_with_action_chunk,
    replace_latent_with_proprio,
)
from cosmos_predict2._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY, ConditioningStrategy

LOG_200 = math.log(200)
LOG_100000 = math.log(100000)


@attrs.define(slots=False)
class CosmosPolicyVideo2WorldConfigRectifiedFlow(CosmosPolicyModelConfigRectifiedFlow):
    """
    Extended config for Cosmos Policy Video2World model.
    Inherits from CosmosPolicyModelConfig and adds some video-specific parameters in the same way that
    Video2WorldConfig adds video-specific parameters to Text2WorldModelConfig.

    """

    min_num_conditional_frames: int = 1  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    conditional_frame_timestep: float = -1.0  # Noise level used for conditional frames; -1 means not effective
    conditioning_strategy: str = str(ConditioningStrategy.FRAME_REPLACE)  # What strategy to use for conditioning
    denoise_replace_gt_frames: bool = True  # Whether to denoise the ground truth frames

    conditional_frames_probs: Optional[Dict[int, float]] = None  # Probability distribution for conditional frames

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.conditioning_strategy in [
            str(ConditioningStrategy.FRAME_REPLACE),
        ]


class CosmosPolicyVideo2WorldModelRectifiedFlow(CosmosPolicyDiffusionModelRectifiedFlow):
    """
    Cosmos Policy Video2World Model - extends CosmosPolicyDiffusionModel with video conditioning.

    Inheritance: CosmosPolicyDiffusionModel → Text2WorldModel

    Adds Video2World functionality to the policy base:
    - Video frame conditioning with gt_frames
    - High sigma sampling strategies
    - FlowUniPC scheduler support
    - Policy-specific mask manipulation for world model/value function
    """

    def __init__(self, config: CosmosPolicyVideo2WorldConfigRectifiedFlow):
        super().__init__(config)
        self.config: CosmosPolicyVideo2WorldConfigRectifiedFlow = config

    def _mask_latent_frame(
        self,
        condition_mask: Tensor,
        batch_indices: Tensor,
        latent_idx: Tensor,
        mask_value: float,
        sample_mask: Tensor | None = None,
    ) -> None:
        """
        Apply mask to a specific latent frame index.

        Args:
            condition_mask: The condition_video_input_mask_B_C_T_H_W tensor to modify in-place
            batch_indices: Tensor of batch indices
            latent_idx: Index of the latent frame to mask
            mask_value: Value to set (typically 0 to mask out)
            sample_mask: Optional per-sample mask for conditional masking (training).
                        If None, applies mask_value unconditionally (inference).
        """
        if sample_mask is not None:
            # Training: conditional masking per sample using torch.where
            condition_mask[batch_indices, :, latent_idx, :, :] = torch.where(
                sample_mask[:, :, 0, :, :].bool(),
                torch.full_like(condition_mask[batch_indices, :, latent_idx, :, :], mask_value),
                condition_mask[batch_indices, :, latent_idx, :, :],
            )
        else:
            # Inference: unconditional masking
            condition_mask[batch_indices, :, latent_idx, :, :] = mask_value

    def _apply_current_state_action_masks(
        self,
        condition: Video2WorldCondition,
        data_batch: dict[str, torch.Tensor],
        sample_mask: Tensor | None = None,
    ) -> None:
        """
        Mask out current state and action for V(s') prediction.

        This masks: current_proprio, current_wrist_image, current_wrist_image2,
        current_image, current_image2, and action frames.

        Args:
            condition: The condition object with mask to modify
            data_batch: Data batch containing latent indices
            sample_mask: Optional per-sample mask for conditional masking (training).
                        If None, applies unconditionally (inference).
        """
        B = condition.condition_video_input_mask_B_C_T_H_W.shape[0]
        batch_indices = torch.arange(B, device=condition.condition_video_input_mask_B_C_T_H_W.device)

        # Mask out current proprio frame
        if torch.all(data_batch["current_proprio_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["current_proprio_latent_idx"],
                0,
                sample_mask,
            )

        # Mask out current wrist image frame
        if torch.all(data_batch["current_wrist_image_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["current_wrist_image_latent_idx"],
                0,
                sample_mask,
            )

        # Mask out current wrist image #2 frame
        if "current_wrist_image2_latent_idx" in data_batch and torch.all(
            data_batch["current_wrist_image2_latent_idx"] != -1
        ):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["current_wrist_image2_latent_idx"],
                0,
                sample_mask,
            )

        # Mask out current image frame (primary image)
        if torch.all(data_batch["current_image_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["current_image_latent_idx"],
                0,
                sample_mask,
            )

        # Mask out current image #2 frame (secondary image)
        if "current_image2_latent_idx" in data_batch and torch.all(data_batch["current_image2_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["current_image2_latent_idx"],
                0,
                sample_mask,
            )

        # Mask out action frame
        self._mask_latent_frame(
            condition.condition_video_input_mask_B_C_T_H_W,
            batch_indices,
            data_batch["action_latent_idx"],
            0,
            sample_mask,
        )

    def _apply_future_state_masks(
        self,
        condition: Video2WorldCondition,
        data_batch: dict[str, torch.Tensor],
        sample_mask: Tensor | None = None,
    ) -> None:
        """
        Mask out future state for Q(s, a) prediction.

        This masks: future_proprio, future_wrist_image, future_wrist_image2,
        future_image, and future_image2 frames.

        Args:
            condition: The condition object with mask to modify
            data_batch: Data batch containing latent indices
            sample_mask: Optional per-sample mask for conditional masking (training).
                        If None, applies unconditionally (inference).
        """
        B = condition.condition_video_input_mask_B_C_T_H_W.shape[0]
        batch_indices = torch.arange(B, device=condition.condition_video_input_mask_B_C_T_H_W.device)

        # Mask out future proprio frame
        if torch.all(data_batch["future_proprio_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["future_proprio_latent_idx"],
                0,
                sample_mask,
            )

        # Mask out future wrist image frame
        if torch.all(data_batch["future_wrist_image_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["future_wrist_image_latent_idx"],
                0,
                sample_mask,
            )

        # Mask out future wrist image #2 frame
        if "future_wrist_image2_latent_idx" in data_batch and torch.all(
            data_batch["future_wrist_image2_latent_idx"] != -1
        ):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["future_wrist_image2_latent_idx"],
                0,
                sample_mask,
            )

        # Mask out future image frame (primary image)
        if torch.all(data_batch["future_image_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["future_image_latent_idx"],
                0,
                sample_mask,
            )

        # Mask out future image #2 frame (secondary image)
        if "future_image2_latent_idx" in data_batch and torch.all(data_batch["future_image2_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["future_image2_latent_idx"],
                0,
                sample_mask,
            )

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Video2WorldCondition]:
        """
        Extended get_data_and_condition with policy-specific conditioning logic.

        Adds:
        - Condition mask manipulation for world model and value function samples
        - Latent injection for actions and proprio in gt_frames
        - Input masking for different prediction modes (V(s'), Q(s,a))
        """
        # generate random number of conditional frames for training
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        # Set video conditioning (from Video2WorldModel base functionality)
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )

        # NOTE (user):
        # If training future state prediction or value function on rollout data, adjust the condition_video_input_mask_B_C_T_H_W so that the
        # proper elements are treated as additional conditioning input (beyond the usual conditioning inputs)
        # - World model (future state prediction): action chunk is additionally treated as conditioning input
        # - Value function (expected return prediction): action chunk and future state are additionally treated as conditioning input
        if "rollout_data_mask" in data_batch:
            # For world model, set the video input mask to 1 for the action frame (i.e. make it a clean/denoised conditioning frame)
            # For value function, set mask to 1 for everything except the value function return frame (should be 0)
            world_model_sample_mask = data_batch["world_model_sample_mask"]
            value_function_sample_mask = data_batch["value_function_sample_mask"]
            # Expand masks from (B,) to (B, 1, 1, H', W')
            H_latent, W_latent = condition.condition_video_input_mask_B_C_T_H_W.shape[-2:]
            world_model_sample_mask = (
                world_model_sample_mask.unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4)
                .expand(-1, 1, 1, H_latent, W_latent)
            ).to(condition.condition_video_input_mask_B_C_T_H_W.dtype)
            value_function_sample_mask = (
                value_function_sample_mask.unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4)
                .expand(-1, 1, 1, H_latent, W_latent)
            ).to(condition.condition_video_input_mask_B_C_T_H_W.dtype)
            # World model: Set the action frame to 1
            batch_indices = torch.arange(world_model_sample_mask.shape[0], device=world_model_sample_mask.device)
            condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, data_batch["action_latent_idx"], :, :] = (
                world_model_sample_mask[:, :, 0, :, :]
            )
            # Value function: Set everything except the value function return frame to 1
            # First, set all frames to 1 for value function samples
            T = condition.condition_video_input_mask_B_C_T_H_W.shape[2]
            value_mask_all_frames = value_function_sample_mask.expand(-1, -1, T, -1, -1)  # (B, 1, T, H', W')
            condition.condition_video_input_mask_B_C_T_H_W = torch.where(
                value_mask_all_frames.bool(),
                torch.ones_like(condition.condition_video_input_mask_B_C_T_H_W),
                condition.condition_video_input_mask_B_C_T_H_W,
            )
            # Then set the value function return frame to 0
            condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, data_batch["value_latent_idx"], :, :] = (
                torch.where(
                    value_function_sample_mask[:, :, 0, :, :].bool(),
                    torch.zeros_like(
                        condition.condition_video_input_mask_B_C_T_H_W[
                            batch_indices, :, data_batch["value_latent_idx"], :, :
                        ]
                    ),
                    condition.condition_video_input_mask_B_C_T_H_W[
                        batch_indices, :, data_batch["value_latent_idx"], :, :
                    ],
                )
            )
            # If we are predicting V(s') instead of V(s, a, s'), mask out the current state and action so that only the
            # future state is used for future state value prediction
            if self.config.mask_current_state_action_for_value_prediction:
                self._apply_current_state_action_masks(condition, data_batch, sample_mask=value_function_sample_mask)
            # If we are predicting Q(s, a) instead of V(s, a, s'), mask out the future state so that only the
            # current state and action are used for Q(s, a) prediction
            if self.config.mask_future_state_for_qvalue_prediction:
                self._apply_future_state_masks(condition, data_batch, sample_mask=value_function_sample_mask)
            # Additionally, add the action chunk to the gt_frames so that the actions are added in later based on the mask
            # No need to do this for the other frames; actions are special because they are manually injected
            condition.orig_gt_frames = condition.gt_frames.clone()  # Keep a backup of the original gt_frames
            condition.gt_frames = replace_latent_with_action_chunk(
                condition.gt_frames, data_batch["actions"], action_indices=data_batch["action_latent_idx"]
            )

        # Manually add in the current and future proprio to the condition.gt_frames as well
        if "proprio" in data_batch and torch.all(
            data_batch["current_proprio_latent_idx"] != -1
        ):  # -1 indicates proprio is not used
            condition.gt_frames = replace_latent_with_proprio(
                condition.gt_frames,
                data_batch["proprio"],
                proprio_indices=data_batch["current_proprio_latent_idx"],
            )
        if "future_proprio" in data_batch and torch.all(
            data_batch["future_proprio_latent_idx"] != -1
        ):  # -1 indicates proprio is not used
            condition.gt_frames = replace_latent_with_proprio(
                condition.gt_frames,
                data_batch["future_proprio"],
                proprio_indices=data_batch["future_proprio_latent_idx"],
            )

        # Manually add in value to the condition.gt_frames as well
        # (This is actually not needed for training because the value is not used as conditioning, but it may be useful
        # for visualizations when decoding the ground-truth latents to images)
        if torch.all(data_batch["value_latent_idx"] != -1) and "value_function_return" in data_batch:
            batch_indices = torch.arange(condition.gt_frames.shape[0], device=condition.gt_frames.device)
            _, C_latent, _, H_latent, W_latent = condition.gt_frames.shape
            condition.gt_frames[batch_indices, :, data_batch["value_latent_idx"], :, :] = (
                data_batch["value_function_return"]
                .reshape(-1, 1, 1, 1)
                .expand(-1, C_latent, H_latent, W_latent)
                .to(condition.gt_frames.dtype)
            )

        return raw_state, latent_state, condition

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Text2WorldCondition,
    ) -> torch.Tensor:
        """
        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (Text2WorldCondition): conditional information, generated from self.conditioner

        Returns:
            velocity prediction
        """
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )

            # Make the first few frames of x_t be the ground truth frames
            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )

                # Reshape timesteps_B_T to (B, 1, 1, 1, 1) for proper broadcasting with (B, 1, T, 1, 1)
                timesteps_B_1_1_1_1 = timesteps_B_T.view(timesteps_B_T.shape[0], 1, 1, 1, 1)
                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_1_1_1_1 * (
                    1 - condition_video_mask_B_1_T_1_1
                )

                # Squeeze to (B, T) - only squeeze dims 1, 3, 4 to preserve batch and time dimensions
                timesteps_B_T = timesteps_B_1_T_1_1.squeeze(dim=(1, 3, 4))

        # forward pass through the network
        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=timesteps_B_T,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        ).float()

        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise - gt_frames_x0
            net_output_B_C_T_H_W = gt_frames_velocity * condition_video_mask + net_output_B_C_T_H_W * (
                1 - condition_video_mask
            )

        return net_output_B_C_T_H_W

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 0,  # Unused, kept for API compatibility with parent class
        is_negative_prompt: bool = False,  # Unused, kept for API compatibility with parent class
    ) -> Callable:
        """
        Generates a callable velocity function based on the provided data batch.

        This function processes the input data batch through a conditioning workflow to obtain
        the conditioned state. It then defines a nested function `velocity_fn` which applies
        a denoising operation on an input `noise_x` at a given timestep.

        Note: CFG (Classifier-Free Guidance) is not used for cosmos policy video2world model,
        so we only compute the conditional velocity prediction without any guidance scaling.
        The `guidance` and `is_negative_prompt` parameters are kept for API compatibility
        with the parent class but are ignored.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content
          of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float): Unused, kept for API compatibility
        - is_negative_prompt (bool): Unused, kept for API compatibility

        Returns:
        - Callable: A function `velocity_fn(noise, noise_x, timestep)` that returns velocity prediction
        """
        del guidance, is_negative_prompt  # Unused - no CFG for cosmos policy

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        # Get condition only, ignore uncondition since we don't use CFG
        condition, _ = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)

        # NOTE (user):
        # The original gt_frames latent is useful for decoding latents to images without distortions caused by the latent injections below
        condition.orig_gt_frames = condition.gt_frames.clone()  # Keep a backup of the original gt_frames

        B = condition.condition_video_input_mask_B_C_T_H_W.shape[0]

        # NOTE (user):
        # If generating samples with current proprio fed as condition via latent injection, adjust the condition_video_input_mask_B_C_T_H_W so that the
        # current proprio latent frame is treated as conditioning input
        if "proprio" in data_batch and torch.all(
            data_batch["current_proprio_latent_idx"] != -1
        ):  # -1 indicates proprio is not used
            proprio = data_batch["proprio"]
            current_proprio_latent_idx = data_batch["current_proprio_latent_idx"]
            batch_indices = torch.arange(B, device=proprio.device)
            condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, current_proprio_latent_idx, :, :] = 1
            # Additionally, add the proprio to the gt_frames so that the proprio is added in later based on the mask
            condition.gt_frames = replace_latent_with_proprio(
                condition.gt_frames, proprio, proprio_indices=current_proprio_latent_idx
            )

        if (
            "mask_current_state_action_for_value_prediction" in data_batch
            and data_batch["mask_current_state_action_for_value_prediction"]
        ):
            # Inference: apply masks unconditionally (sample_mask=None)
            self._apply_current_state_action_masks(condition, data_batch, sample_mask=None)

        if (
            "mask_future_state_for_qvalue_prediction" in data_batch
            and data_batch["mask_future_state_for_qvalue_prediction"]
        ):
            # Inference: apply masks unconditionally (sample_mask=None)
            self._apply_future_state_masks(condition, data_batch, sample_mask=None)

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return self.denoise(noise, noise_x, timestep, condition)

        return velocity_fn
