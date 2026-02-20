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
Cosmos Policy Diffusion Model - extends Text2WorldModel with policy-specific functionality.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import attrs
import torch
from einops import rearrange

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_predict2._src.imaginaire.utils import misc
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast_split_tensor, cat_outputs_cp
from cosmos_predict2._src.predict2.cosmos_policy.conditioner import Text2WorldCondition
from cosmos_predict2._src.predict2.cosmos_policy.models.policy_text2world_model import (
    replace_latent_with_action_chunk,
    replace_latent_with_proprio,
)
from cosmos_predict2._src.predict2.cosmos_policy.modules.cosmos_sampler import CosmosPolicySampler
from cosmos_predict2._src.predict2.cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldModelRectifiedFlow,
    Text2WorldModelRectifiedFlowConfig,
)


@attrs.define(slots=False)
class CosmosPolicyModelConfigRectifiedFlow(Text2WorldModelRectifiedFlowConfig):
    """
    Extended config for Cosmos Policy diffusion model.
    Uses Cosmos Policy's HybridEDMSDE instead of the original EDMSDE.
    Also adds policy-specific parameters for loss masking and action prediction.
    """

    sde: LazyDict = L(HybridEDMSDE)(
        # Note: Most of these values get overridden later in the experiment configs
        p_mean=0.0,
        p_std=1.0,
        sigma_max=80,
        sigma_min=0.0002,
        hybrid_sigma_distribution=True,
        uniform_lower=1.0,
        uniform_upper=85.0,
    )

    # Whether to use loss masking to separate action, future state, and value prediction
    # - Policy prediction: only take loss on action predictions
    # - World model prediction: only take loss on future state predictions
    # - Value function prediction: only take loss on value predictions
    mask_loss_for_action_future_state_prediction: bool = False
    # Whether to use loss masking on value prediction during policy predictions (so we only take loss on action + future state predictions)
    mask_value_prediction_loss_for_policy_prediction: bool = False
    # Whether to mask out some inputs (current state and action) during future state value prediction
    mask_current_state_action_for_value_prediction: bool = False
    # Whether to mask out some inputs (future state) during Q(s,a) prediction
    mask_future_state_for_qvalue_prediction: bool = False

    # Action loss multiplier (if greater than 1, upweights loss on predicting actions relative to other losses)
    # (Must be an integer - or will be cast to an integer later!)
    action_loss_multiplier: int = 1

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert not (
            self.mask_loss_for_action_future_state_prediction and self.mask_value_prediction_loss_for_policy_prediction
        ), (
            "Cannot enable both mask_loss_for_action_future_state_prediction and mask_value_prediction_loss_for_policy_prediction!"
        )


class CosmosPolicyDiffusionModelRectifiedFlow(Text2WorldModelRectifiedFlow):
    """
    Cosmos Policy Diffusion Model - extends Text2WorldModel with policy-specific functionality.

    Adds support for:
    - Action chunk prediction and injection
    - Proprioception (proprio) prediction and injection
    - Value function prediction
    - Loss masking for different prediction types (action, future state, value)
    - Multi-component loss tracking
    """

    def __init__(self, config: CosmosPolicyModelConfigRectifiedFlow):
        super().__init__(config)
        self.config: CosmosPolicyModelConfigRectifiedFlow = config

        # Cosmos Policy SDE and Sampler
        self.sde = lazy_instantiate(config.sde)
        self.sampler = CosmosPolicySampler()

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the Cosmos Policy diffusion model using rectified flow.

        Uses velocity matching objective with flow matching formulation instead of EDM.

        Extended from base to pass policy-specific data (actions, proprio, values, masks).

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: The computed loss for the training step as a PyTorch Tensor.
        """
        self._update_train_stats(data_batch)

        # Obtain text embeddings online using reason1 encoder from ai_caption
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)

        # Sample N(0, 1) noise and training time for rectified flow
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.tensor_kwargs_fp32)
        batch_size = x0_B_C_T_H_W.size()[0]
        t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32)
        t_B = rearrange(t_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma

        # Broadcast and split the input data and condition for model parallelism
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B = self.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
        )

        # Get discrete timesteps and sigmas for rectified flow
        timesteps_B = self.rectified_flow.get_discrete_timestamp(t_B, self.tensor_kwargs_fp32)
        sigmas_B = self.rectified_flow.get_sigmas(timesteps_B, self.tensor_kwargs_fp32)
        timesteps_B_T = rearrange(timesteps_B, "b -> b 1")
        sigmas_B_T = rearrange(sigmas_B, "b -> b 1")

        output_batch, velocity_loss = self.compute_loss_rectified_flow(
            x0_B_C_T_H_W,
            condition,
            epsilon_B_C_T_H_W,
            timesteps_B_T,
            sigmas_B_T,
            action_chunk=data_batch["actions"],
            action_indices=data_batch["action_latent_idx"],
            proprio=data_batch["proprio"],
            current_proprio_indices=data_batch["current_proprio_latent_idx"],
            future_proprio=data_batch["future_proprio"],
            future_proprio_indices=data_batch["future_proprio_latent_idx"],
            future_wrist_image_indices=data_batch["future_wrist_image_latent_idx"],
            future_wrist_image2_indices=(
                data_batch["future_wrist_image2_latent_idx"] if "future_wrist_image2_latent_idx" in data_batch else None
            ),
            future_image_indices=data_batch["future_image_latent_idx"],
            future_image2_indices=(
                data_batch["future_image2_latent_idx"] if "future_image2_latent_idx" in data_batch else None
            ),
            rollout_data_mask=data_batch["rollout_data_mask"],
            world_model_sample_mask=data_batch["world_model_sample_mask"],
            value_function_sample_mask=data_batch["value_function_sample_mask"],
            value_function_return=data_batch["value_function_return"],
            value_indices=data_batch["value_latent_idx"],
        )

        velocity_loss = velocity_loss.mean()

        return output_batch, velocity_loss

    def compute_loss_rectified_flow(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: Text2WorldCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        sigmas_B_T: torch.Tensor,
        action_chunk: torch.Tensor,
        action_indices: torch.Tensor,
        proprio: torch.Tensor,
        current_proprio_indices: torch.Tensor,
        future_proprio: torch.Tensor,
        future_proprio_indices: torch.Tensor,
        future_wrist_image_indices: torch.Tensor,
        future_wrist_image2_indices: Optional[torch.Tensor],
        future_image_indices: torch.Tensor,
        future_image2_indices: Optional[torch.Tensor],
        rollout_data_mask: torch.Tensor,
        world_model_sample_mask: torch.Tensor,
        value_function_sample_mask: torch.Tensor,
        value_function_return: torch.Tensor,
        value_indices: torch.Tensor,
    ):
        """
        Compute loss using rectified flow (velocity matching) with policy-specific functionality.

        This method uses flow matching formulation instead of EDM, computing loss as
        MSE between predicted and target velocities.

        This method extends the base implementation to support:
        1. Latent injection for actions, proprioception, and values
        2. Loss masking for different prediction types (action, future state, value)
        3. Detailed per-component loss tracking

        Args:
            x0_B_C_T_H_W: image/video latent (clean data)
            condition: text condition
            epsilon_B_C_T_H_W: noise (N(0,1))
            timesteps_B_T: discrete timesteps for rectified flow
            sigmas_B_T: sigma values corresponding to timesteps
            action_chunk: ground truth action chunk
            action_indices: indices for action latent frames
            proprio: current proprioception
            current_proprio_indices: indices for current proprio latent frames
            future_proprio: future proprioception
            future_proprio_indices: indices for future proprio latent frames
            future_wrist_image_indices: indices for future wrist image latent frames
            future_wrist_image2_indices: indices for future wrist image #2 latent frames
            future_image_indices: indices for future primary image latent frames
            future_image2_indices: indices for future secondary image latent frames
            rollout_data_mask: mask for rollout vs demo data
            world_model_sample_mask: mask for world model samples
            value_function_sample_mask: mask for value function samples
            value_function_return: ground truth value function returns
            value_indices: indices for value latent frames

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data for debug / logging / callbacks
                - Tensor: velocity matching loss
        """
        # NOTE (user): Action chunk, proprio, value injection
        # x0_B_C_T_H_W: (B, C', T', H', W')
        # x0_B_C_T_H_W is the VAE-encoded latent representing several input images (which may be in a different order than below):
        # - conditional frames (e.g., proprio, wrist camera image, primary image, optional history)
        # - future image(s) and proprio state to predict
        # - action chunk to predict (blank image)
        # - value function return (rewards-to-go) to predict (blank image)
        condition.orig_x0_B_C_T_H_W = x0_B_C_T_H_W.clone()  # Keep a backup of the original gt_frames
        batch_indices = torch.arange(x0_B_C_T_H_W.shape[0], device=x0_B_C_T_H_W.device)
        C_latent, H_latent, W_latent = x0_B_C_T_H_W.shape[1], x0_B_C_T_H_W.shape[3], x0_B_C_T_H_W.shape[4]
        # Action
        x0_B_C_T_H_W = replace_latent_with_action_chunk(
            x0_B_C_T_H_W,
            action_chunk,
            action_indices=action_indices,
        )
        # Proprio
        if torch.all(current_proprio_indices != -1):  # -1 indicates proprio is not used
            x0_B_C_T_H_W = replace_latent_with_proprio(
                x0_B_C_T_H_W,
                proprio,
                proprio_indices=current_proprio_indices,
            )
        # Future proprio
        if torch.all(future_proprio_indices != -1):  # -1 indicates future proprio is not used
            x0_B_C_T_H_W = replace_latent_with_proprio(
                x0_B_C_T_H_W,
                future_proprio,
                proprio_indices=future_proprio_indices,
            )
        # Value
        x0_B_C_T_H_W[batch_indices, :, value_indices, :, :] = (
            value_function_return.reshape(-1, 1, 1, 1).expand(-1, C_latent, H_latent, W_latent).to(x0_B_C_T_H_W.dtype)
        )

        # Get interpolation for rectified flow: xt (noisy sample) and vt (target velocity)
        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas_B_T)

        # Get velocity prediction from the model
        vt_pred_B_C_T_H_W = self.denoise(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_B_T,
            condition=condition,
        )

        # Get time-based loss weights for rectified flow
        time_weights_B = self.rectified_flow.train_time_weight(timesteps_B_T, self.tensor_kwargs_fp32)

        # Construct mask to support masking out loss (or scaling it by some multiplier) for different types of predictions
        B, T = x0_B_C_T_H_W.shape[0], x0_B_C_T_H_W.shape[2]
        device = timesteps_B_T.device
        final_mask_B_T = torch.ones((B, T), dtype=torch.long, device=device)  # All 1s mask initially

        # If using input masking for value prediction, mask out the loss for everything except the value prediction
        # This is necessary since otherwise the loss will be computed for all latent frames, not just the value prediction frame
        if (
            self.config.mask_current_state_action_for_value_prediction
            or self.config.mask_future_state_for_qvalue_prediction
        ):
            mask_B_T = torch.ones((B, T), dtype=torch.long, device=device)  # All 1s mask
            # Rollout value-function samples (rollout_data_mask == 1 and value_function_sample_mask == 1)
            value_idx_B = ((rollout_data_mask == 1) & (value_function_sample_mask == 1)).to(torch.long).to(device)
            if torch.any(value_idx_B):
                value_batch_indices = torch.nonzero(value_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(device)
                # First set the mask to 0 for everything except the value prediction, but only do this for value prediction samples
                mask_B_T[value_batch_indices, :] = 0
                # Then set the mask to 1 for the value prediction
                mask_B_T[value_batch_indices, value_indices[value_batch_indices]] = 1
            final_mask_B_T = final_mask_B_T * mask_B_T

        # Build per-sample mask to select which frames contribute to loss
        # - Demo samples: only action prediction
        # - Rollout world-model samples: only future state (proprio, wrist image, primary image)
        # - Rollout value-function samples: only value prediction
        if self.config.mask_loss_for_action_future_state_prediction:
            B, T = x0_B_C_T_H_W.shape[0], x0_B_C_T_H_W.shape[2]
            mask_B_T = torch.zeros(
                (B, T), dtype=torch.long, device=device
            )  # All 0s mask, to be filled with 1s for the relevant timesteps
            # Demo samples (rollout_data_mask == 0)
            demo_idx_B = (rollout_data_mask == 0).to(torch.long).to(device)
            if torch.any(demo_idx_B):
                demo_batch_indices = torch.nonzero(demo_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(device)
                mask_B_T[demo_batch_indices, action_indices[demo_batch_indices]] = 1
            # Rollout world-model samples (rollout_data_mask == 1 and world_model_sample_mask == 1)
            world_idx_B = (rollout_data_mask == 1) & (world_model_sample_mask == 1).to(torch.long).to(device)
            if torch.any(world_idx_B):
                world_batch_indices = torch.nonzero(world_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(device)
                if torch.all(future_image_indices != -1):  # -1 indicates future image is not used
                    mask_B_T[world_batch_indices, future_image_indices[world_batch_indices]] = 1
                if future_image2_indices is not None and torch.all(
                    future_image2_indices != -1
                ):  # -1 indicates secondary image is not used
                    mask_B_T[world_batch_indices, future_image2_indices[world_batch_indices]] = 1
                if torch.all(future_wrist_image_indices != -1):  # -1 indicates future wrist image is not used
                    mask_B_T[world_batch_indices, future_wrist_image_indices[world_batch_indices]] = 1
                if future_wrist_image2_indices is not None and torch.all(
                    future_wrist_image2_indices != -1
                ):  # -1 indicates future wrist image #2 is not used
                    mask_B_T[world_batch_indices, future_wrist_image2_indices[world_batch_indices]] = 1
                if torch.all(future_proprio_indices != -1):  # -1 indicates future proprio is not used
                    mask_B_T[world_batch_indices, future_proprio_indices[world_batch_indices]] = 1
            # Rollout value-function samples (rollout_data_mask == 1 and value_function_sample_mask == 1)
            value_idx_B = ((rollout_data_mask == 1) & (value_function_sample_mask == 1)).to(torch.long).to(device)
            if torch.any(value_idx_B):
                value_batch_indices = torch.nonzero(value_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(device)
                mask_B_T[value_batch_indices, value_indices[value_batch_indices]] = 1
            final_mask_B_T = final_mask_B_T * mask_B_T

        # Build per-sample mask to select which frames contribute to loss
        # - Demo samples: only action prediction + future state prediction
        # - Rollout world-model samples: only future state (proprio, wrist image, primary image)
        # - Rollout value-function samples: N/A (assert that we don't encounter any value function samples here)
        if self.config.mask_value_prediction_loss_for_policy_prediction:
            assert value_function_sample_mask.sum() == 0, (
                "No value function samples should be present when mask_value_prediction_loss_for_policy_prediction==True!"
            )
            B, T = x0_B_C_T_H_W.shape[0], x0_B_C_T_H_W.shape[2]
            mask_B_T = torch.zeros(
                (B, T), dtype=torch.long, device=device
            )  # All 0s mask, to be filled with 1s for the relevant timesteps
            # Demo samples (rollout_data_mask == 0)
            demo_idx_B = (rollout_data_mask == 0).to(torch.long).to(device)
            if torch.any(demo_idx_B):
                demo_batch_indices = torch.nonzero(demo_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(device)
                mask_B_T[demo_batch_indices, action_indices[demo_batch_indices]] = 1
                if torch.all(future_image_indices != -1):  # -1 indicates future image is not used
                    mask_B_T[demo_batch_indices, future_image_indices[demo_batch_indices]] = 1
                if future_image2_indices is not None and torch.all(
                    future_image2_indices != -1
                ):  # -1 indicates secondary image is not used
                    mask_B_T[demo_batch_indices, future_image2_indices[demo_batch_indices]] = 1
                if torch.all(future_wrist_image_indices != -1):  # -1 indicates future wrist image is not used
                    mask_B_T[demo_batch_indices, future_wrist_image_indices[demo_batch_indices]] = 1
                if future_wrist_image2_indices is not None and torch.all(
                    future_wrist_image2_indices != -1
                ):  # -1 indicates future wrist image #2 is not used
                    mask_B_T[demo_batch_indices, future_wrist_image2_indices[demo_batch_indices]] = 1
                if torch.all(future_proprio_indices != -1):  # -1 indicates future proprio is not used
                    mask_B_T[demo_batch_indices, future_proprio_indices[demo_batch_indices]] = 1
            # Rollout world-model samples (rollout_data_mask == 1 and world_model_sample_mask == 1)
            world_idx_B = (rollout_data_mask == 1) & (world_model_sample_mask == 1).to(torch.long).to(device)
            if torch.any(world_idx_B):
                world_batch_indices = torch.nonzero(world_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(device)
                if torch.all(future_image_indices != -1):  # -1 indicates future image is not used
                    mask_B_T[world_batch_indices, future_image_indices[world_batch_indices]] = 1
                if future_image2_indices is not None and torch.all(
                    future_image2_indices != -1
                ):  # -1 indicates secondary image is not used
                    mask_B_T[world_batch_indices, future_image2_indices[world_batch_indices]] = 1
                if torch.all(future_wrist_image_indices != -1):  # -1 indicates future wrist image is not used
                    mask_B_T[world_batch_indices, future_wrist_image_indices[world_batch_indices]] = 1
                if future_wrist_image2_indices is not None and torch.all(
                    future_wrist_image2_indices != -1
                ):  # -1 indicates future wrist image #2 is not used
                    mask_B_T[world_batch_indices, future_wrist_image2_indices[world_batch_indices]] = 1
                if torch.all(future_proprio_indices != -1):  # -1 indicates future proprio is not used
                    mask_B_T[world_batch_indices, future_proprio_indices[world_batch_indices]] = 1
            final_mask_B_T = final_mask_B_T * mask_B_T

        # If applicable, upweight the loss on the action predictions by a factor of `action_loss_multiplier`
        if self.config.action_loss_multiplier != 1:
            # Only upweight the loss on the action indices
            final_mask_B_T[batch_indices, action_indices] = final_mask_B_T[batch_indices, action_indices] * int(
                self.config.action_loss_multiplier
            )

        # Compute velocity matching loss (MSE between predicted and target velocity)
        velocity_mse_B_C_T_H_W = (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2

        # Apply time-based weighting (expanded to match spatial dimensions)
        # time_weights_B has shape (B, 1), expand to (B, 1, 1, 1, 1) for broadcasting
        time_weights_expanded = rearrange(time_weights_B, "b 1 -> b 1 1 1 1")
        weighted_velocity_loss_B_C_T_H_W = velocity_mse_B_C_T_H_W * time_weights_expanded

        # Apply the loss mask to the loss
        if (
            self.config.mask_loss_for_action_future_state_prediction
            or self.config.mask_current_state_action_for_value_prediction
            or self.config.mask_future_state_for_qvalue_prediction
            or self.config.action_loss_multiplier != 1
        ):
            weighted_velocity_loss_B_C_T_H_W = weighted_velocity_loss_B_C_T_H_W * rearrange(
                final_mask_B_T, "b t -> b 1 t 1 1"
            )

        # Get losses for future third-person image prediction
        if torch.all(future_image_indices != -1):  # -1 indicates future third-person image is not used
            batch_indices = torch.arange(x0_B_C_T_H_W.shape[0], device=x0_B_C_T_H_W.device)
            future_image_diff = (
                vt_B_C_T_H_W[batch_indices, :, future_image_indices, :, :]
                - vt_pred_B_C_T_H_W[batch_indices, :, future_image_indices, :, :]
            )
            future_image_diff_demo = future_image_diff[rollout_data_mask == 0]
            future_image_diff_world_model = future_image_diff[world_model_sample_mask == 1]
            future_image_diff_value_function = future_image_diff[value_function_sample_mask == 1]

            demo_sample_future_image_mse_loss = (future_image_diff_demo**2).mean()
            demo_sample_future_image_l1_loss = torch.abs(future_image_diff_demo).mean()
            world_model_sample_future_image_mse_loss = (future_image_diff_world_model**2).mean()
            world_model_sample_future_image_l1_loss = torch.abs(future_image_diff_world_model).mean()
            all_samples_future_image_mse_loss = (future_image_diff**2).mean()
            all_samples_future_image_l1_loss = torch.abs(future_image_diff).mean()
        else:
            # If not generating future third-person images, set all future image losses to nan
            demo_sample_future_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            demo_sample_future_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)

        # Get losses for future wrist image prediction
        if torch.all(future_wrist_image_indices != -1):  # -1 indicates future wrist image is not used
            future_wrist_image_diff = (
                vt_B_C_T_H_W[batch_indices, :, future_wrist_image_indices, :, :]
                - vt_pred_B_C_T_H_W[batch_indices, :, future_wrist_image_indices, :, :]
            )
            future_wrist_image_diff_demo = future_wrist_image_diff[rollout_data_mask == 0]
            future_wrist_image_diff_world_model = future_wrist_image_diff[world_model_sample_mask == 1]
            future_wrist_image_diff_value_function = future_wrist_image_diff[value_function_sample_mask == 1]

            demo_sample_future_wrist_image_mse_loss = (future_wrist_image_diff_demo**2).mean()
            demo_sample_future_wrist_image_l1_loss = torch.abs(future_wrist_image_diff_demo).mean()
            world_model_sample_future_wrist_image_mse_loss = (future_wrist_image_diff_world_model**2).mean()
            world_model_sample_future_wrist_image_l1_loss = torch.abs(future_wrist_image_diff_world_model).mean()
            all_samples_future_wrist_image_mse_loss = (future_wrist_image_diff**2).mean()
            all_samples_future_wrist_image_l1_loss = torch.abs(future_wrist_image_diff).mean()
        else:
            # If not generating future wrist images, set all future wrist image losses to nan
            demo_sample_future_wrist_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            demo_sample_future_wrist_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_wrist_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_wrist_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_wrist_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_wrist_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)

        # Get losses for future proprio prediction
        if torch.all(future_proprio_indices != -1):  # -1 indicates future proprio is not used
            future_proprio_diff = (
                vt_B_C_T_H_W[batch_indices, :, future_proprio_indices, :, :]
                - vt_pred_B_C_T_H_W[batch_indices, :, future_proprio_indices, :, :]
            )
            future_proprio_diff_demo = future_proprio_diff[rollout_data_mask == 0]
            future_proprio_diff_world_model = future_proprio_diff[world_model_sample_mask == 1]
            future_proprio_diff_value_function = future_proprio_diff[value_function_sample_mask == 1]

            demo_sample_future_proprio_mse_loss = (future_proprio_diff_demo**2).mean()
            demo_sample_future_proprio_l1_loss = torch.abs(future_proprio_diff_demo).mean()
            world_model_sample_future_proprio_mse_loss = (future_proprio_diff_world_model**2).mean()
            world_model_sample_future_proprio_l1_loss = torch.abs(future_proprio_diff_world_model).mean()
            all_samples_future_proprio_mse_loss = (future_proprio_diff**2).mean()
            all_samples_future_proprio_l1_loss = torch.abs(future_proprio_diff).mean()
        else:
            # If not generating future proprio, set all future proprio losses to nan
            demo_sample_future_proprio_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            demo_sample_future_proprio_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_proprio_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_proprio_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_proprio_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_proprio_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)

        # Get losses for action prediction (velocity-based)
        action_diff = (
            vt_B_C_T_H_W[batch_indices, :, action_indices, :, :]
            - vt_pred_B_C_T_H_W[batch_indices, :, action_indices, :, :]
        )
        action_diff_demo = action_diff[rollout_data_mask == 0]
        action_diff_world_model = action_diff[world_model_sample_mask == 1]
        demo_sample_action_mse_loss = (action_diff_demo**2).mean()
        demo_sample_action_l1_loss = torch.abs(action_diff_demo).mean()
        all_samples_action_mse_loss = (action_diff**2).mean()
        all_samples_action_l1_loss = torch.abs(action_diff).mean()

        # Get losses for value function prediction (velocity-based)
        value_diff = (
            vt_B_C_T_H_W[batch_indices, :, value_indices, :, :]
            - vt_pred_B_C_T_H_W[batch_indices, :, value_indices, :, :]
        )
        value_diff_demo = value_diff[rollout_data_mask == 0]
        value_diff_world_model = value_diff[world_model_sample_mask == 1]
        value_diff_value_function = value_diff[value_function_sample_mask == 1]
        demo_sample_value_mse_loss = (value_diff_demo**2).mean()
        demo_sample_value_l1_loss = torch.abs(value_diff_demo).mean()
        world_model_sample_value_mse_loss = (value_diff_world_model**2).mean()
        world_model_sample_value_l1_loss = torch.abs(value_diff_world_model).mean()
        value_function_sample_value_mse_loss = (value_diff_value_function**2).mean()
        value_function_sample_value_l1_loss = torch.abs(value_diff_value_function).mean()
        all_samples_value_mse_loss = (value_diff**2).mean()
        all_samples_value_l1_loss = torch.abs(value_diff).mean()

        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "timesteps": timesteps_B_T,
            "sigmas": sigmas_B_T,
            "condition": condition,
            "vt_pred": vt_pred_B_C_T_H_W,
            "vt_target": vt_B_C_T_H_W,
            "velocity_mse_loss": velocity_mse_B_C_T_H_W.mean(),
            "edm_loss": weighted_velocity_loss_B_C_T_H_W.mean(),  # For callback compatibility
            "velocity_loss_per_frame": torch.mean(weighted_velocity_loss_B_C_T_H_W, dim=[1, 3, 4]),
            # Demo sample losses (velocity-based)
            "demo_sample_action_mse_loss": demo_sample_action_mse_loss,  # Main action loss for policy
            "demo_sample_action_l1_loss": demo_sample_action_l1_loss,  # Main action loss for policy
            "demo_sample_future_proprio_mse_loss": demo_sample_future_proprio_mse_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_proprio_l1_loss": demo_sample_future_proprio_l1_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_wrist_image_mse_loss": demo_sample_future_wrist_image_mse_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_wrist_image_l1_loss": demo_sample_future_wrist_image_l1_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_image_mse_loss": demo_sample_future_image_mse_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_image_l1_loss": demo_sample_future_image_l1_loss,  # Auxiliary future state loss for policy
            "demo_sample_value_mse_loss": demo_sample_value_mse_loss,  # Auxiliary value loss for policy
            "demo_sample_value_l1_loss": demo_sample_value_l1_loss,  # Auxiliary value loss for policy
            # World model sample losses (velocity-based)
            "world_model_sample_future_proprio_mse_loss": world_model_sample_future_proprio_mse_loss,  # Main future state loss for world model
            "world_model_sample_future_proprio_l1_loss": world_model_sample_future_proprio_l1_loss,  # Main future state loss for world model
            "world_model_sample_future_wrist_image_mse_loss": world_model_sample_future_wrist_image_mse_loss,  # Main future state loss for world model
            "world_model_sample_future_wrist_image_l1_loss": world_model_sample_future_wrist_image_l1_loss,  # Main future state loss for world model
            "world_model_sample_future_image_mse_loss": world_model_sample_future_image_mse_loss,  # Main future state loss for world model
            "world_model_sample_future_image_l1_loss": world_model_sample_future_image_l1_loss,  # Main future state loss for world model
            "world_model_sample_value_mse_loss": world_model_sample_value_mse_loss,  # Auxiliary value loss for world model
            "world_model_sample_value_l1_loss": world_model_sample_value_l1_loss,  # Auxiliary value loss for world model
            # Value function sample losses (velocity-based)
            "value_function_sample_value_mse_loss": value_function_sample_value_mse_loss,  # Main loss for value function
            "value_function_sample_value_l1_loss": value_function_sample_value_l1_loss,  # Main loss for value function
        }
        return output_batch, weighted_velocity_loss_B_C_T_H_W

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 0,  # cosmos policy uses 0 for guidance
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 1.0,
        use_variance_scale: bool = False,
        return_orig_clean_latent_frames: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch using rectified flow with Cosmos Policy extensions.

        Uses velocity prediction and euler integration via FlowUniPCMultistepScheduler.

        Extended to support:
        - Variance scaling for increased diversity (via shift parameter adjustment)
        - Returning original clean latent frames

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            guidance (float): guidance weights for classifier-free guidance
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            shift (float): shift parameter for the flow matching scheduler (default: 5.0)
            use_variance_scale (bool): use variance scale to increase diversity in outputs
            return_orig_clean_latent_frames (bool): Whether to return the clean latent frames
        """
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
            ]

        # Get original clean latent frames if requested (for conditioning or debugging)
        orig_clean_latent_frames = None
        if return_orig_clean_latent_frames:
            _, orig_clean_latent_frames, _ = self.get_data_and_condition(data_batch)

        # NOTE: Add random variance scaling to increase diversity in outputs
        # For rectified flow, we adjust the shift parameter instead of sigma
        if use_variance_scale:
            torch.manual_seed(seed)
            shift_variance_scale = torch.rand(1).item() * 4.0 + 3.0  # uniform between 3.0 and 7.0
            effective_shift = shift * shift_variance_scale / 5.0  # normalize around default shift of 5.0
        else:
            effective_shift = shift

        # Generate initial noise
        noise = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape),
            torch.float32,
            self.tensor_kwargs["device"],
            seed,
        )

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        # Set up the flow matching scheduler timesteps
        self.sample_scheduler.set_timesteps(
            num_steps,
            device=self.tensor_kwargs["device"],
            shift=effective_shift,
            use_kerras_sigma=self.config.use_kerras_sigma_at_inference,
        )

        timesteps = self.sample_scheduler.timesteps

        # Get velocity prediction function with classifier-free guidance
        velocity_fn = self.get_velocity_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        # Handle context parallelism
        if self.net.is_context_parallel_enabled:
            noise = broadcast_split_tensor(tensor=noise, seq_dim=2, process_group=self.get_context_parallel_group())

        latents = noise

        # Euler integration loop using rectified flow
        for _, t in enumerate(timesteps):
            latent_model_input = latents
            timestep = [t]
            timestep = torch.stack(timestep)

            velocity_pred = velocity_fn(noise, latent_model_input, timestep.unsqueeze(0))
            temp_x0 = self.sample_scheduler.step(
                velocity_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            latents = temp_x0.squeeze(0)

        # Gather outputs from context parallel ranks
        if self.net.is_context_parallel_enabled:
            latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())

        if return_orig_clean_latent_frames:
            return latents, orig_clean_latent_frames
        else:
            return latents
