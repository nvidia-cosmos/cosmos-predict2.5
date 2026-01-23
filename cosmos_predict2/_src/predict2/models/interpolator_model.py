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

from typing import Callable, Dict, Optional, Tuple

import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict2._src.imaginaire.modules.res_sampler import COMMON_SOLVER_OPTIONS
from cosmos_predict2._src.imaginaire.utils import misc
from cosmos_predict2._src.imaginaire.utils.context_parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.frame_interpolation.conditioner import (
    InterpolatorCondition,  # type: ignore[missing-import]
)
from cosmos_predict2._src.predict2.models.denoise_prediction import DenoisePrediction
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
    IS_PREPROCESSED_KEY,
    Text2WorldCondition,
)
from cosmos_predict2._src.predict2.models.video2world_model import (
    NUM_CONDITIONAL_FRAMES_KEY,
    ConditioningStrategy,
    Video2WorldConfig,
    Video2WorldModel,
)


@attrs.define(slots=False)
class InterpolatorConfig(Video2WorldConfig):
    """Configuration for interpolator model with frame interpolation specific settings."""

    sigma_conditional: float = 0.0001  # Noise level used for conditional frames
    frame_wise_encoding: bool = True  # Whether to use frame-wise encoding (True) or causal video encoding (False)
    interleaved_conditioning: bool = False  # Whether to use interleaved conditioning

    def __attrs_post_init__(self):
        super().__attrs_post_init__()


class InterpolatorModel(Video2WorldModel):
    """
    Interpolator model that extends Vid2VidModel with frame interpolation capabilities.

    This model inherits from Vid2VidModel and only overrides the methods that have
    been specifically adapted for frame interpolation functionality, while reusing
    all other functionality from the parent classes.
    """

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, InterpolatorCondition]:
        # generate random number of conditional frames for training
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            interleaved_conditioning=self.config.interleaved_conditioning,
        )
        return raw_state, latent_state, condition

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode input video frames to latent space with frame-by-frame processing.

        Args:
            state: Input video tensor of shape (B, C, T, H, W)

        Returns:
            Encoded latent tensor of shape (B, C, T, H, W)
        """
        if not self.config.frame_wise_encoding:
            # Use causal video encoding.
            return super().encode(state)

        # Use frame-wise encoding.
        input_state = rearrange(state, "b c t h w -> (b t) c 1 h w")
        encoded_state = [self.tokenizer.encode(one_state.unsqueeze(0)) * self.sigma_data for one_state in input_state]
        encoded_state = torch.cat(encoded_state, dim=0)
        return rearrange(encoded_state, "(b t) c 1 h w -> b c t h w", b=state.shape[0])

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations back to video frames with frame-by-frame processing.

        Args:
            latent: Latent tensor of shape (B, C, T, H, W)

        Returns:
            Decoded video tensor of shape (B, C, T, H, W)
        """
        if not self.config.frame_wise_encoding:
            # Use causal video decoding.
            return super().decode(latent)

        # Use frame-wise decoding.
        latent_batch = rearrange(latent, "b c t h w -> (b t) c 1 h w")
        decoded_batch = [
            self.tokenizer.decode(one_latent.unsqueeze(0) / self.sigma_data) for one_latent in latent_batch
        ]
        decoded_batch = torch.cat(decoded_batch, dim=0)
        return rearrange(decoded_batch, "(b t) c 1 h w -> b c t h w", b=latent.shape[0])

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        input_key = self.input_data_key if input_key is None else input_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
                assert torch.all((data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)), (
                    f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
                )
            else:
                assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

            expected_length = self.config.state_t
            original_length = data_batch[input_key].shape[2]
            assert original_length == expected_length, (
                f"Input video length doesn't match expected length specified by state_t: {original_length} != {expected_length}"
            )

    def denoise(
        self, xt_B_C_T_H_W: torch.Tensor, sigma: torch.Tensor, condition: Text2WorldCondition
    ) -> DenoisePrediction:
        """
        Performs denoising on the input noise data, noise level, and condition with interpolation-specific
        noise handling for conditional frames.

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

            if self.config.conditioning_strategy == str(ConditioningStrategy.FRAME_REPLACE):
                # In case of frame replacement strategy, replace the first few frames of the video with the conditional frames
                # ADD ACTUAL NOISE to conditional frames to match what we tell the model about noise levels (v1-style fix)

                # Add actual noise to conditional frames.
                condition_noise = torch.randn_like(condition_state_in_B_C_T_H_W) * self.config.sigma_conditional
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W + condition_noise

                # Make the first few frames of x_t be the (now properly noisy) ground truth frames
                net_state_in_B_C_T_H_W = (
                    condition_state_in_B_C_T_H_W * condition_video_mask
                    + net_state_in_B_C_T_H_W * (1 - condition_video_mask)
                )
                # Adjust c_noise for the conditional frames
                sigma_cond_B_1_T_1_1 = torch.ones_like(sigma_B_1_T_1_1) * self.config.sigma_conditional
                _, _, _, c_noise_cond_B_1_T_1_1 = self.scaling(sigma=sigma_cond_B_1_T_1_1)
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                c_noise_B_1_T_1_1 = c_noise_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + c_noise_B_1_T_1_1 * (
                    1 - condition_video_mask_B_1_T_1_1
                )
            elif self.config.conditioning_strategy == str(ConditioningStrategy.CHANNEL_CONCAT):
                # In case of channel concatenation strategy, concatenate the conditional frames in the channel dimension
                condition_state_in_masked_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask
                net_state_in_B_C_T_H_W = torch.cat([net_state_in_B_C_T_H_W, condition_state_in_masked_B_C_T_H_W], dim=1)

        else:
            # In case of image batch, simply concatenate the 0 frames when channel concat strategy is used
            if self.config.conditioning_strategy == str(ConditioningStrategy.CHANNEL_CONCAT):
                net_state_in_B_C_T_H_W = torch.cat(
                    [net_state_in_B_C_T_H_W, torch.zeros_like(net_state_in_B_C_T_H_W)], dim=1
                )

        # forward pass through the network
        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(
                **self.tensor_kwargs
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

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
        x_sigma_max: Optional[torch.Tensor] = None,
        sigma_max: float | None = None,
    ) -> torch.Tensor:
        """
        Generate interpolated samples from the batch with interpolation-specific logic.

        Args:
            data_batch: Raw data batch from the training data loader
            guidance: Guidance weight for classifier-free guidance
            seed: Random seed for reproducible generation
            state_shape: Shape of the state, defaults to data batch if not provided
            n_sample: Number of samples to generate
            is_negative_prompt: Whether to use negative prompt in unconditioning
            num_steps: Number of diffusion steps
            solver_option: Differential equation solver option
            x_sigma_max: Initial noise tensor
            sigma_max: Maximum sigma value for diffusion

        Returns:
            Generated latent samples
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
                _T,
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        # Get interpolation-specific x0 function
        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        if x_sigma_max is None:
            x_sigma_max = (
                misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    self.tensor_kwargs["device"],
                    seed,
                )
                * self.sde.sigma_max
            )

        # Handle context parallelism for interpolation
        if self.net.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.get_context_parallel_group())

        if sigma_max is None:
            sigma_max = self.sde.sigma_max

        # Generate samples using interpolation-aware sampling
        samples = self.sampler(
            x0_fn,
            x_sigma_max,
            num_steps=num_steps,
            sigma_max=sigma_max,
            sigma_min=self.sde.sigma_min,
            solver_option=solver_option,
        )

        if self.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())

        return samples

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generate x0 function with interpolation-specific conditioning logic.

        This method provides a clean, self-contained implementation that uses
        our custom denoise method with proper noise handling for conditional frames.

        Args:
            data_batch: Input data batch
            guidance: Classifier-free guidance scale
            is_negative_prompt: Whether to use negative prompts

        Returns:
            Function that generates x0 predictions for interpolation
        """
        # Set up conditioning logic
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

        # Set up both conditions with proper gt_frames for interpolation
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            interleaved_conditioning=self.config.interleaved_conditioning,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            interleaved_conditioning=self.config.interleaved_conditioning,
        )

        # condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        # uncondition = uncondition.edit_for_inference(
        #     is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        # )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def interpolation_x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            """
            Clean interpolation x0 function using our custom denoise method.
            """
            if guidance == -1:
                # Unconditional generation - use our denoise method
                raw_x0 = self.denoise(noise_x, sigma, uncondition).x0
            else:
                # Classifier-free guidance - use our denoise method for both paths
                cond_x0 = self.denoise(noise_x, sigma, condition).x0
                uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
                raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)

            # Apply guided interpolation if masks are provided
            if "guided_image" in data_batch:
                assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch["guided_image"]
                guide_mask = data_batch["guided_mask"]
                raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0

            return raw_x0

        return interpolation_x0_fn
