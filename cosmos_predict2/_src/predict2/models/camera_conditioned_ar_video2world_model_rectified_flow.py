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

from typing import Callable, Dict, Tuple

import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict2._src.imaginaire.utils import misc
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.camera_conditioned.conditioner import CameraConditionedCondition
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2.utils.context_parallel import (
    broadcast_split_tensor,
    cat_outputs_cp,
)

IS_PREPROCESSED_KEY = "is_preprocessed"


@attrs.define(slots=False)
class CameraConditionedARVideo2WorldRectifiedFlowConfig(Video2WorldModelRectifiedFlowConfig):
    pass


class CameraConditionedARVideo2WorldModelRectifiedFlow(Video2WorldModelRectifiedFlow):
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, CameraConditionedCondition]:
        self._normalize_multicam_video_databatch_inplace(data_batch)
        self._augment_multicam_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # Latent cond state
        split_size = data_batch["num_frames"].item()
        raw_state_cond = data_batch[self.input_data_key + "_cond"]
        raw_state_cond_chunks = torch.split(raw_state_cond, split_size_or_sections=split_size, dim=2)
        latent_state_cond_list = []
        for raw_state_cond_chunk in raw_state_cond_chunks:
            latent_state_cond_chunk = self.encode(raw_state_cond_chunk).contiguous().float()
            latent_state_cond_list.append(latent_state_cond_chunk)

        # Latent tgt state
        raw_state_src = data_batch[self.input_data_key]
        raw_state_src_chunks = torch.split(raw_state_src, split_size_or_sections=split_size, dim=2)
        latent_state_src_list = []
        for raw_state_src_chunk in raw_state_src_chunks:
            latent_state_src_chunk = self.encode(raw_state_src_chunk).contiguous().float()
            latent_state_src_list.append(latent_state_src_chunk)

        raw_state = torch.cat(
            (
                raw_state_cond_chunks[0],
                raw_state_cond_chunks[1],
                raw_state_src_chunks[0],
                raw_state_cond_chunks[2],
                raw_state_cond_chunks[3],
            ),
            dim=2,
        )
        latent_state = torch.cat(
            (
                latent_state_cond_list[0],
                latent_state_cond_list[1],
                latent_state_src_list[0],
                latent_state_cond_list[2],
                latent_state_cond_list[3],
            ),
            dim=2,
        )

        # Condition
        camera_list = torch.chunk(data_batch["camera"], len(latent_state_cond_list) + len(latent_state_src_list), dim=1)
        camera = torch.cat((camera_list[0], camera_list[1], camera_list[4], camera_list[2], camera_list[3]), dim=1)
        data_batch["camera"] = camera
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        condition = condition.set_camera_conditioned_ar_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
        )

        # torch.distributed.breakpoint()
        return raw_state, latent_state, condition

    def _normalize_multicam_video_databatch_inplace(
        self, data_batch: dict[str, torch.Tensor], input_key: str = None
    ) -> None:
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
                assert torch.all(
                    (data_batch[input_key] >= -1.0001)
                    & (data_batch[input_key] <= 1.0001)
                    & (data_batch[input_key + "_cond"] >= -1.0001)
                    & (data_batch[input_key + "_cond"] <= 1.0001)
                ), (
                    f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
                )
            else:
                assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[input_key + "_cond"] = data_batch[input_key + "_cond"].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

    def _augment_multicam_image_dim_inplace(self, data_batch: dict[str, torch.Tensor], input_key: str = None) -> None:
        input_key = self.input_image_key if input_key is None else input_key
        if input_key in data_batch:
            # Check if the data has already been augmented and avoid re-augmenting
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert data_batch[input_key].shape[2] == 1, (
                    f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                )
                return
            else:
                data_batch[input_key] = rearrange(data_batch[input_key], "b c h w -> b c 1 h w").contiguous()
                data_batch[input_key + "_cond"] = rearrange(
                    data_batch[input_key + "_cond"], "b c h w -> b c 1 h w"
                ).contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True

    @torch.no_grad()
    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        num_input_video: int = 2,
        num_output_video: int = 1,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - num_input_video (int): Number of input videos as condition. Defaults to 1
        - num_output_video (int): Number of generated output videos. Defaults to 2
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """

        camera_list = torch.chunk(data_batch["camera"], num_input_video + num_output_video, dim=1)
        camera = torch.cat((camera_list[0], camera_list[1], camera_list[4], camera_list[2], camera_list[3]), dim=1)
        data_batch["camera"] = camera

        x0_cond_chunks = torch.chunk(data_batch[self.input_data_key], num_input_video, dim=2)
        x0_cond_list = []
        for x0_cond_chunk in x0_cond_chunks:
            x0_cond = self.encode(x0_cond_chunk).contiguous().float()
            x0_cond_list.append(x0_cond)
        x0_conds = torch.cat(x0_cond_list, dim=2)
        data_batch["video_cond"] = x0_conds

        x0 = torch.cat(
            [x0_cond_list[0], x0_cond_list[1], torch.zeros_like(x0_cond), x0_cond_list[2], x0_cond_list[3]], dim=2
        )

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_camera_conditioned_ar_video_condition(
            gt_frames=x0,
            num_conditional_frames=data_batch[NUM_CONDITIONAL_FRAMES_KEY],
        )
        uncondition = uncondition.set_camera_conditioned_ar_video_condition(
            gt_frames=x0,
            num_conditional_frames=data_batch[NUM_CONDITIONAL_FRAMES_KEY],
        )

        # torch.distributed.breakpoint()
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            cond_v = self.denoise(noise, noise_x, timestep, condition)
            uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
            velocity_pred = cond_v + guidance * (cond_v - uncond_v)
            return velocity_pred

        return velocity_fn, x0_cond_list

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        num_input_video: int = 3,
        num_output_video: int = 1,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            num_input_video (int): Number of input videos as condition. Defaults to 1
            num_output_video (int): Number of generated output videos. Defaults to 2
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            solver_option (str): differential equation solver option, default to "2ab"~(mulitstep solver)
        """

        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            _T = _T // num_input_video
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        velocity_fn, x0_cond_list = self.get_velocity_fn_from_batch(
            data_batch, guidance, num_input_video, num_output_video, is_negative_prompt=is_negative_prompt
        )

        noise_list = []
        for i in range(num_output_video):
            noise = misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            noise_list.append(noise)

        noise = torch.cat([x0_cond_list[0], x0_cond_list[1], noise_list[0], x0_cond_list[2], x0_cond_list[3]], dim=2)

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.sample_scheduler.set_timesteps(num_steps, device=self.tensor_kwargs["device"], shift=shift)

        timesteps = self.sample_scheduler.timesteps

        if self.net.is_context_parallel_enabled:
            noise = broadcast_split_tensor(tensor=noise, seq_dim=2, process_group=self.get_context_parallel_group())
        latents = noise

        for _, t in enumerate(timesteps):
            latent_model_input = latents
            timestep = [t]

            timestep = torch.stack(timestep)

            velocity_pred = velocity_fn(noise, latent_model_input, timestep.unsqueeze(0))
            temp_x0 = self.sample_scheduler.step(
                velocity_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            latents = temp_x0.squeeze(0)

        if self.net.is_context_parallel_enabled:
            latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())

        sample_chunks = torch.chunk(latents, num_input_video + num_output_video, dim=2)
        sample_list = [sample_chunks[2]]

        return sample_list
