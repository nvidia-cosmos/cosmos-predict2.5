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

import random
from dataclasses import field
from typing import Callable, Dict, Optional, Tuple, cast

import attrs
import torch
import torch.distributed as dist
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor
from torch.distributed import get_process_group_ranks

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast, broadcast_split_tensor
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import IS_PREPROCESSED_KEY
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2_multiview.configs.vid2vid.defaults.conditioner import (
    ConditionLocationList,
    MultiViewCondition,
)

TRAIN_SAMPLE_N_VIEWS_KEY = "train_sample_n_views"
TRAIN_SAMPLING_APPLIED_KEY = "train_sampling_applied"
_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


@attrs.define(slots=False)
class MultiviewVid2VidModelRectifiedFlowConfig(Video2WorldModelRectifiedFlowConfig):
    min_num_conditional_frames_per_view: int = 1
    max_num_conditional_frames_per_view: int = 2
    train_sample_views_range: Tuple[int, int] | None = None
    condition_locations: ConditionLocationList = field(default_factory=lambda: ConditionLocationList([]))
    state_t: int = 0
    view_condition_dropout_max: int = 0
    online_text_embeddings_as_dict: bool = True  # For backward compatibility with old experiments
    conditional_frames_probs: Optional[Dict[int, float]] = None  # Probability distribution for conditional frames


class MultiviewVid2VidModelRectifiedFlow(Video2WorldModelRectifiedFlow):
    def __init__(self, config: MultiviewVid2VidModelRectifiedFlowConfig):
        super().__init__(config)
        self.state_t = config.state_t
        self.empty_string_text_embeddings = None
        self.neg_text_embeddings = None
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            compute_empty_and_negative_text_embeddings(self)

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        n_views = state.shape[2] // self.tokenizer.get_pixel_num_frames(self.state_t)
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        if n_views > 4 and n_views <= cp_size:
            return self.encode_cp(state)
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=n_views)
        encoded_state = super().encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        n_views = latent.shape[2] // self.state_t
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        if n_views > 4 and n_views <= cp_size:
            return self.decode_cp(latent)
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=n_views)
        decoded_state = super().decode(latent)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return decoded_state

    @torch.no_grad()
    def encode_cp(self, state: torch.Tensor) -> torch.Tensor:
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        cp_group = parallel_state.get_context_parallel_group()
        n_views = state.shape[2] // self.tokenizer.get_pixel_num_frames(self.state_t)
        assert n_views <= cp_size, f"n_views must be less than cp_size, got n_views={n_views} and cp_size={cp_size}"
        state_V_B_C_T_H_W = rearrange(state, "B C (V T) H W -> V B C T H W", V=n_views)
        state_input = torch.zeros((cp_size, *state_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        state_input[0:n_views] = state_V_B_C_T_H_W
        local_state_V_B_C_T_H_W = broadcast_split_tensor(state_input, seq_dim=0, process_group=cp_group)
        local_state = rearrange(local_state_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        encoded_state = super().encode(local_state)
        encoded_state_list = [torch.empty_like(encoded_state) for _ in range(cp_size)]
        dist.all_gather(encoded_state_list, encoded_state, group=cp_group)
        encoded_state = torch.cat(encoded_state_list[0:n_views], dim=2)  # [B, C, V * T, H, W]
        return encoded_state

    @torch.no_grad()
    def decode_cp(self, latent: torch.Tensor) -> torch.Tensor:
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        cp_group = parallel_state.get_context_parallel_group()
        n_views = latent.shape[2] // self.state_t
        assert n_views <= cp_size, f"n_views must be less than cp_size, got n_views={n_views} and cp_size={cp_size}"
        latent_V_B_C_T_H_W = rearrange(latent, "B C (V T) H W -> V B C T H W", V=n_views)
        latent_input = torch.zeros((cp_size, *latent_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        latent_input[0:n_views] = latent_V_B_C_T_H_W
        local_latent_V_B_C_T_H_W = broadcast_split_tensor(latent_input, seq_dim=0, process_group=cp_group)
        local_latent = rearrange(local_latent_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        decoded_state = super().decode(local_latent)
        decoded_state_list = [torch.empty_like(decoded_state) for _ in range(cp_size)]
        dist.all_gather(decoded_state_list, decoded_state, group=cp_group)
        decoded_state = torch.cat(decoded_state_list[0:n_views], dim=2)  # [B, C, V * T, H, W]
        return decoded_state

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return training_step_multiview(self, data_batch, iteration)

    def inplace_compute_text_embeddings_online(self, data_batch: dict[str, torch.Tensor]) -> None:
        inplace_compute_text_embeddings_online_multiview(self, data_batch)

    def broadcast_split_for_model_parallelsim(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: MultiViewCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ):
        n_views = x0_B_C_T_H_W.shape[2] // self.state_t
        x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views).contiguous()
        if epsilon_B_C_T_H_W is not None:
            epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views).contiguous()
        reshape_sigma_B_T = False
        if sigma_B_T is not None:
            assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
            if sigma_B_T.shape[-1] != 1:
                assert sigma_B_T.shape[-1] % n_views == 0, (
                    f"sigma_B_T temporal dimension T must either be 1 or a multiple of sample_n_views. Got T={sigma_B_T.shape[-1]} and sample_n_views={n_views}"
                )
                sigma_B_T = rearrange(sigma_B_T, "B (V T) -> (B V) T", V=n_views).contiguous()
                reshape_sigma_B_T = True
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = super().broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )
        x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
        if epsilon_B_C_T_H_W is not None:
            epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
        if reshape_sigma_B_T:
            sigma_B_T = rearrange(sigma_B_T, "(B V) T -> B (V T)", V=n_views)
        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T

    def get_data_batch_with_latent_view_indices(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        data_batch = self._preprocess_databatch(data_batch)
        num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
        n_views = data_batch["view_indices"].shape[1] // num_video_frames_per_view
        view_indices_B_V_T = rearrange(data_batch["view_indices"], "B (V T) -> B V T", V=n_views)

        latent_view_indices_B_V_T = view_indices_B_V_T[:, :, 0 : self.config.state_t]
        latent_view_indices_B_T = rearrange(latent_view_indices_B_V_T, "B V T -> B (V T)")
        data_batch_with_latent_view_indices = data_batch.copy()
        data_batch_with_latent_view_indices["latent_view_indices_B_T"] = latent_view_indices_B_T
        return data_batch_with_latent_view_indices

    def sample_first_n_views_from_data_batch(self, data_batch, n_views):
        if n_views == data_batch["sample_n_views"].cpu().item():
            return data_batch
        new_data_batch = {}
        num_video_frames_per_view = data_batch["num_video_frames_per_view"].cpu().item()
        log.debug(f"Sampling {n_views} views out of {data_batch['sample_n_views'].cpu().item()}")
        log.debug(f"num_video_frames_per_view: {num_video_frames_per_view}")
        new_total_frames = num_video_frames_per_view * n_views
        new_total_t5_dim = 512 * n_views
        new_data_batch["video"] = data_batch["video"][:, :, 0:new_total_frames]
        new_data_batch["view_indices"] = data_batch["view_indices"][:, 0:new_total_frames]
        new_data_batch["sample_n_views"] = 0 * data_batch["sample_n_views"] + n_views
        new_data_batch["fps"] = data_batch["fps"]
        new_data_batch["t5_text_embeddings"] = data_batch["t5_text_embeddings"][:, 0:new_total_t5_dim]
        new_data_batch["t5_text_mask"] = data_batch["t5_text_mask"][:, 0:new_total_t5_dim]
        split_captions = data_batch["ai_caption"][0].split(" -- ")
        assert len(split_captions) == 7, f"Expected 7 view captions, got {len(split_captions)}"
        new_data_batch["ai_caption"] = [" -- ".join(split_captions[0:n_views])]
        new_data_batch["n_orig_video_frames_per_view"] = data_batch["n_orig_video_frames_per_view"]
        assert data_batch["ref_cam_view_idx_sample_position"].item() == -1, (
            f"ref_cam_view_idx_sample_position is not supported by batch sampling, got {data_batch['ref_cam_view_idx_sample_position']}"
        )
        new_data_batch["ref_cam_view_idx_sample_position"] = data_batch["ref_cam_view_idx_sample_position"]
        new_data_batch["camera_keys_selection"] = data_batch["camera_keys_selection"][0:n_views]
        new_data_batch["view_indices_selection"] = data_batch["view_indices_selection"]
        for key in [
            "__url__",
            "__key__",
            "__t5_url__",
            "image_size",
            "num_video_frames_per_view",
            "aspect_ratio",
            "padding_mask",
        ]:
            new_data_batch[key] = data_batch[key]
        for key in [NUM_CONDITIONAL_FRAMES_KEY]:
            if key in data_batch:
                new_data_batch[key] = data_batch[key]
        old_keys = set(list(data_batch.keys()))
        new_keys = set(list(new_data_batch.keys()))
        diff = old_keys.difference(new_keys)
        # We use the following check to ensure that all newly introduced keys are correctly and explicitly handled and avoid bugs from omitted handling of new keys
        assert old_keys == new_keys, f"Expected old keys to equal new keys. Difference {diff}"
        return new_data_batch

    def _preprocess_databatch(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Preprocess data batch with dynamic view sampling"""
        if TRAIN_SAMPLING_APPLIED_KEY in data_batch and data_batch[TRAIN_SAMPLING_APPLIED_KEY] is True:
            return data_batch
        if self.config.train_sample_views_range is not None:
            min_views, max_views = self.config.train_sample_views_range
            log.debug(f"Randomly sampling {min_views} to {max_views} views")
            if TRAIN_SAMPLE_N_VIEWS_KEY in data_batch:
                train_sample_n_views = data_batch[TRAIN_SAMPLE_N_VIEWS_KEY]
                log.debug(f"Using {TRAIN_SAMPLE_N_VIEWS_KEY} from data batch: {train_sample_n_views}")
                if train_sample_n_views < 1:  # No sampling is applied
                    return data_batch
            else:
                # Sample and broadcast n_views across cp_groups so all ranks in the same group sample the same number of views
                train_sample_n_views = random.randint(min_views, max_views)
                if parallel_state.get_context_parallel_group() is not None:
                    train_sample_n_views = broadcast(train_sample_n_views, parallel_state.get_context_parallel_group())
                log.debug(f"Sampled and broadcasted {train_sample_n_views=} ")
            log.debug(f"Sampling {train_sample_n_views} views out of {data_batch['sample_n_views'].cpu().item()}")
            data_batch = self.sample_first_n_views_from_data_batch(data_batch, train_sample_n_views)
            data_batch[TRAIN_SAMPLING_APPLIED_KEY] = True
        return data_batch

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        data_batch = self._preprocess_databatch(data_batch)
        input_key = self.input_data_key if input_key is None else input_key
        is_preprocessed = IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True

        num_video_frames_per_view = (
            self.tokenizer.get_pixel_num_frames(self.state_t)
            if is_preprocessed
            else data_batch["num_video_frames_per_view"]
        )
        if isinstance(num_video_frames_per_view, torch.Tensor):
            num_video_frames_per_view = int(num_video_frames_per_view.cpu().item())
        n_views = data_batch[input_key].shape[2] // num_video_frames_per_view
        if input_key in data_batch:
            data_batch[input_key] = rearrange(data_batch[input_key], "B C (V T) H W -> (B V) C T H W", V=n_views)
            super()._normalize_video_databatch_inplace(data_batch, input_key)
            data_batch[input_key] = rearrange(data_batch[input_key], "(B V) C T H W -> B C (V T) H W", V=n_views)

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[Tensor, Tensor, MultiViewCondition]:
        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        raw_state, latent_state, condition = super(Video2WorldModelRectifiedFlow, self).get_data_and_condition(
            data_batch_with_latent_view_indices
        )
        condition = cast(MultiViewCondition, condition)
        condition = condition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=None,
            view_condition_dropout_max=self.config.view_condition_dropout_max,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return raw_state, latent_state, condition

    def get_velocity_fn_from_batch(
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
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return velocity predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """

        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch_with_latent_view_indices:
            num_conditional_frames = data_batch_with_latent_view_indices[NUM_CONDITIONAL_FRAMES_KEY]
            log.debug(f"Using {num_conditional_frames=} from data batch")
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(
                data_batch_with_latent_view_indices
            )
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch_with_latent_view_indices)

        is_image_batch = self.is_image_batch(data_batch_with_latent_view_indices)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch_with_latent_view_indices)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=x0,
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=x0,
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(
            is_cfg_conditional=True,
            condition_locations=self.config.condition_locations,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False,
            condition_locations=self.config.condition_locations,
            num_conditional_frames_per_view=num_conditional_frames,
        )
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

        return velocity_fn

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: dict[str, torch.Tensor],
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        **kwargs,
    ) -> torch.Tensor:
        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        process_group = parallel_state.get_context_parallel_group()
        cp_size = len(get_process_group_ranks(process_group))
        samples_B_C_T_H_W = super().generate_samples_from_batch(
            data_batch_with_latent_view_indices,
            guidance,
            seed,
            state_shape,
            n_sample,
            is_negative_prompt,
            num_steps,
            shift,
            **kwargs,
        )
        if cp_size > 1:
            samples_B_C_T_H_W = rearrange(
                samples_B_C_T_H_W, "B C (c V T) H W -> B C (V c T) H W", c=cp_size, T=self.state_t // cp_size
            )
        return samples_B_C_T_H_W


def compute_text_embeddings_online_multiview_single_caption(
    model, data_batch: dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_preprocessed = IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True
    num_video_frames_per_view = (
        model.tokenizer.get_pixel_num_frames(model.state_t)
        if is_preprocessed
        else data_batch["num_video_frames_per_view"]
    )
    if isinstance(num_video_frames_per_view, torch.Tensor):
        num_video_frames_per_view = int(num_video_frames_per_view.cpu().item())
    n_views = data_batch[model.input_data_key].shape[2] // num_video_frames_per_view
    B, _, _, _, _ = data_batch[model.input_data_key].shape

    # compute prompt embeddings
    view0_text_embeddings_B_L_D = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
    assert view0_text_embeddings_B_L_D.shape[1] == 512, (
        f"view0_text_embeddings should be of shape (B, 512, D), got {view0_text_embeddings_B_L_D.shape}"
    )

    output_text_embeddings = model.empty_string_text_embeddings.clone().repeat(B, n_views, 1)
    output_neg_text_embeddings = model.empty_string_text_embeddings.clone().repeat(B, n_views, 1)
    output_text_embeddings = rearrange(output_text_embeddings, "B (V L) D -> V B L D", V=n_views)
    output_neg_text_embeddings = rearrange(output_neg_text_embeddings, "B (V L) D -> V B L D", V=n_views)
    # Assign prompt embeddings to the front camera view
    for i_b in range(B):
        front_cam_view_idx_sample_position = data_batch["front_cam_view_idx_sample_position"][i_b]
        output_text_embeddings[front_cam_view_idx_sample_position, i_b] = view0_text_embeddings_B_L_D[i_b]
        output_neg_text_embeddings[front_cam_view_idx_sample_position, i_b] = model.neg_text_embeddings[0]
    output_text_embeddings = rearrange(output_text_embeddings, "V B L D -> B (V L) D")
    output_neg_text_embeddings = rearrange(output_neg_text_embeddings, "V B L D -> B (V L) D")

    dropout_text_embeddings = model.empty_string_text_embeddings.clone().repeat(B, n_views, 1)

    if not model.config.conditioner.text.use_empty_string:
        dropout_text_embeddings *= 0.0

    return output_text_embeddings, output_neg_text_embeddings, dropout_text_embeddings


def compute_text_embeddings_online_multiview_multiple_captions(
    model, data_batch: dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_preprocessed = IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True
    num_video_frames_per_view = (
        model.tokenizer.get_pixel_num_frames(model.state_t)
        if is_preprocessed
        else data_batch["num_video_frames_per_view"]
    )
    if isinstance(num_video_frames_per_view, torch.Tensor):
        num_video_frames_per_view = int(num_video_frames_per_view.cpu().item())
    n_views = data_batch[model.input_data_key].shape[2] // num_video_frames_per_view
    B, _, _, _, _ = data_batch[model.input_data_key].shape

    # compute each view's caption separately
    if not len(data_batch["ai_caption"]) == 1:
        raise NotImplementedError(f"Expected batch size of 1, got {len(data_batch['ai_caption'])}")

    captions = data_batch["ai_caption"][0].split(" -- ")
    if len(captions) != n_views:
        raise ValueError(f"Expected {n_views} captions, got {len(captions)}: {captions}")
    view_text_embeddings = []
    for caption in captions:
        data_batch_per_view = {model.input_caption_key: [caption]}
        view_text_embedding = model.text_encoder.compute_text_embeddings_online(
            data_batch_per_view, model.input_caption_key
        )
        view_text_embeddings.append(view_text_embedding)

    view_text_embeddings_B_V_L_D = torch.stack(view_text_embeddings, dim=1)
    assert view_text_embeddings_B_V_L_D.shape[:3] == (
        B,
        n_views,
        512,
    ), f"view_text_embeddings_B_V_L_D should be of shape (B, n_views, 512, D), got {view_text_embeddings_B_V_L_D.shape}"
    output_text_embeddings = rearrange(view_text_embeddings_B_V_L_D, "B V L D -> B (V L) D")

    # repeat negative embedding for each per view
    output_neg_text_embeddings_L_D = model.neg_text_embeddings[0].clone()
    output_neg_text_embeddings_B_V_L_D = rearrange(output_neg_text_embeddings_L_D, "(B V L) D -> B V L D", B=1, V=1)
    output_neg_text_embeddings_B_V_L_D = output_neg_text_embeddings_B_V_L_D.repeat(B, n_views, 1, 1)
    assert output_neg_text_embeddings_B_V_L_D.shape[:3] == (
        B,
        n_views,
        512,
    ), (
        f"output_neg_text_embeddings_B_V_L_D should be of shape (B, n_views, 512, D), got {output_neg_text_embeddings_B_V_L_D.shape}"
    )
    output_neg_text_embeddings = rearrange(output_neg_text_embeddings_B_V_L_D, "B V L D -> B (V L) D")

    dropout_text_embeddings = model.empty_string_text_embeddings.clone().repeat(B, n_views, 1)
    if not model.config.conditioner.text.use_empty_string:
        dropout_text_embeddings *= 0.0

    return output_text_embeddings, output_neg_text_embeddings, dropout_text_embeddings


def compute_text_embeddings_online_multiview(
    model, data_batch: dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    captions = data_batch["ai_caption"][0].split(" -- ")
    is_preprocessed = IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True
    num_video_frames_per_view = (
        model.tokenizer.get_pixel_num_frames(model.state_t)
        if is_preprocessed
        else data_batch["num_video_frames_per_view"]
    )
    if isinstance(num_video_frames_per_view, torch.Tensor):
        num_video_frames_per_view = int(num_video_frames_per_view.cpu().item())
    n_views = data_batch[model.input_data_key].shape[2] // num_video_frames_per_view
    assert len(captions) == 1 or len(captions) == n_views, f"Expected 1 or {n_views} captions, got {len(captions)}"
    if len(captions) == 1:
        return compute_text_embeddings_online_multiview_single_caption(model, data_batch)
    else:
        return compute_text_embeddings_online_multiview_multiple_captions(model, data_batch)


def inplace_compute_text_embeddings_online_multiview(model, data_batch: dict[str, torch.Tensor]) -> None:
    output_text_embeddings, output_neg_text_embeddings, dropout_text_embeddings = (
        compute_text_embeddings_online_multiview(model, data_batch)
    )
    t5_text_embeddings = {
        "text_embeddings": output_text_embeddings,
        "dropout_text_embeddings": dropout_text_embeddings,
    }
    neg_t5_text_embeddings = {
        "text_embeddings": output_neg_text_embeddings,
        "dropout_text_embeddings": dropout_text_embeddings,
    }
    data_batch["t5_text_embeddings"] = (
        t5_text_embeddings if model.config.online_text_embeddings_as_dict else t5_text_embeddings["text_embeddings"]
    )
    data_batch["neg_t5_text_embeddings"] = (
        neg_t5_text_embeddings
        if model.config.online_text_embeddings_as_dict
        else neg_t5_text_embeddings["text_embeddings"]
    )
    data_batch["t5_text_mask"] = torch.ones(
        output_text_embeddings.shape[0], output_text_embeddings.shape[1], device="cuda"
    )


def compute_empty_and_negative_text_embeddings(model):
    # Compute empty string embeddings for text embedding dropout
    if model.empty_string_text_embeddings is None:
        empty_string_data_batch = {
            model.input_caption_key: [" "],
        }
        model.empty_string_text_embeddings = model.text_encoder.compute_text_embeddings_online(
            empty_string_data_batch, model.input_caption_key
        )

    # compute negative prompt embeddings for sampling
    if model.neg_text_embeddings is None:
        neg_promt_data_batch = {
            model.input_caption_key: [_DEFAULT_NEGATIVE_PROMPT],
        }
        model.neg_text_embeddings = model.text_encoder.compute_text_embeddings_online(
            neg_promt_data_batch, model.input_caption_key
        )


def training_step_multiview(
    model, data_batch: dict[str, torch.Tensor], iteration: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Performs a single training step for the diffusion model.

    This method is responsible for executing one iteration of the model's training. It involves:
    1. Adding noise to the input data using the SDE process.
    2. Passing the noisy data through the network to generate predictions.
    3. Computing the loss based on the difference between the predictions and the original data, \
        considering any configured loss weighting.

    Args:
        data_batch (dict): raw data batch draw from the training data loader.
        iteration (int): Current iteration number.

    Returns:
        tuple: A tuple containing two elements:
            - dict: additional data that used to debug / logging / callbacks
            - Tensor: The computed loss for the training step as a PyTorch Tensor.

    Raises:
        AssertionError: If the class is conditional, \
            but no number of classes is specified in the network configuration.

    Notes:
        - The method handles different types of conditioning
        - The method also supports Kendall's loss
    """
    model._update_train_stats(data_batch)

    if model.config.text_encoder_config is not None and model.config.text_encoder_config.compute_online:
        model.inplace_compute_text_embeddings_online(data_batch)

    # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
    _, x0_B_C_T_H_W, condition = model.get_data_and_condition(data_batch)

    # Sample pertubation noise levels and N(0, 1) noises
    epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **model.tensor_kwargs_fp32)
    batch_size = x0_B_C_T_H_W.size()[0]
    t_B = model.rectified_flow.sample_train_time(batch_size).to(**model.tensor_kwargs_fp32)
    t_B = rearrange(t_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma
    x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B = model.broadcast_split_for_model_parallelsim(
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
    )
    timesteps = model.rectified_flow.get_discrete_timestamp(t_B, model.tensor_kwargs_fp32)
    sigmas = model.rectified_flow.get_sigmas(
        timesteps,
        model.tensor_kwargs_fp32,
    )
    timesteps = rearrange(timesteps, "b -> b 1")
    sigmas = rearrange(sigmas, "b -> b 1")
    xt_B_C_T_H_W, vt_B_C_T_H_W = model.rectified_flow.get_interpolation(epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas)

    vt_pred_B_C_T_H_W = model.denoise(
        noise=epsilon_B_C_T_H_W,
        xt_B_C_T_H_W=xt_B_C_T_H_W.to(**model.tensor_kwargs),
        timesteps_B_T=timesteps,
        condition=condition,
    )

    time_weights_B = model.rectified_flow.train_time_weight(timesteps, model.tensor_kwargs_fp32)
    per_instance_loss = torch.mean((vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2, dim=list(range(1, vt_pred_B_C_T_H_W.dim())))

    loss = torch.mean(time_weights_B * per_instance_loss)
    output_batch = {
        "x0": x0_B_C_T_H_W,
        "xt": xt_B_C_T_H_W,
        "sigma": sigmas,
        "condition": condition,
        "model_pred": vt_pred_B_C_T_H_W,
        "edm_loss": loss,
    }

    return output_batch, loss
