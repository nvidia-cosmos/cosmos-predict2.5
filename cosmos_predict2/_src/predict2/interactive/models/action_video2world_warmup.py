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

from typing import List, Tuple

import attrs
import torch
from einops import rearrange

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldModelRectifiedFlow,
    Text2WorldModelRectifiedFlowConfig,
)


@attrs.define(slots=False)
class ActionConditionedSFWarmupModelRFConfig(Text2WorldModelRectifiedFlowConfig):
    denoising_step_list: List[int] = attrs.field(default=attrs.Factory(lambda: [0, 9, 18, 27]))
    fps: int = 4
    # Ensure no conditional frames are used during warmup
    min_num_conditional_frames: int = 0
    max_num_conditional_frames: int = 0
    # # This field is not used in warmup but needed for config composition compatibility
    conditional_frames_probs: dict[int, float] | None = None


class ActionConditionedSFWarmupModelRF(Text2WorldModelRectifiedFlow):
    # Narrow the config type for this subclass
    config: ActionConditionedSFWarmupModelRFConfig
    net: torch.nn.Module

    def __init__(self, config: ActionConditionedSFWarmupModelRFConfig):
        super().__init__(config)
        # Cache fps locally to satisfy type-checkers
        self.fps: int = config.fps

        self.sample_scheduler.set_timesteps(
            35,
            device=self.tensor_kwargs["device"],
            shift=config.shift,
        )
        timesteps_all = self.sample_scheduler.timesteps.clone()
        self.t_list = [timesteps_all[i] for i in config.denoising_step_list]
        self.t_list.append(torch.tensor(0, device=timesteps_all.device, dtype=timesteps_all.dtype))
        self.t_list = torch.stack(self.t_list)  # Convert list to tensor for indexing
        log.info(f"============================== timesteps: {self.t_list}")

        # Latest decoded video for visualization callbacks
        self.latest_backward_simulation_video = None

    def is_image_batch(self, data_batch: dict) -> bool:
        """Always returns False (video batch) since we're processing video sequences."""
        return False

    @torch.no_grad()
    def _prepare_generator_input_output(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_image = data_batch["input_image"].to(**self.tensor_kwargs) / 127.5 - 1.0
        input_image = rearrange(input_image, "b c h w -> b c 1 h w")
        input_latent = self.encode(input_image).contiguous().float()  # (B, 16, 1, lh, lw)

        ode_latents = data_batch["ode_latents"].to(**self.tensor_kwargs)  # (B, 5, 16, lt, lh, lw)
        batch_size, num_denoising_steps_plus_one, num_channels, num_frames, height, width = ode_latents.shape

        index = torch.randint(
            0,
            num_denoising_steps_plus_one - 1,
            [batch_size, num_frames],
            device=self.tensor_kwargs["device"],
            dtype=torch.long,
        )
        index[:, 0] = num_denoising_steps_plus_one - 1

        timesteps = self.t_list[index].to(self.tensor_kwargs["device"])

        noisy_input = torch.gather(
            ode_latents,
            dim=1,
            index=index.reshape(batch_size, 1, 1, num_frames, 1, 1)
            .expand(-1, -1, num_channels, -1, height, width)
            .to(self.tensor_kwargs["device"]),
        ).squeeze(1)  # (B, 16, lt, lh, lw)
        noisy_input[:, :, 0] = input_latent.squeeze(2)  # set the first frame to the input image clean latent

        target_latents = ode_latents[:, -1]  # (B, 16, lt, lh, lw)

        return timesteps, noisy_input, target_latents

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        action = data_batch["action"].to(**self.tensor_kwargs)  # (B, 12, 29)

        timesteps, input_latents, target_latents = self._prepare_generator_input_output(data_batch)
        batch_size, _, num_frames, lh, lw = input_latents.shape

        # set first frame to 1 and other frames to 0
        condition_mask = torch.zeros(batch_size, 1, num_frames, lh, lw).to(**self.tensor_kwargs)
        # condition_mask[:, :, 0] = 1

        # set all zeros padding mask
        padding_mask = torch.zeros(batch_size, 1, lh, lw).to(**self.tensor_kwargs)

        velocity_pred = self.net(
            x_B_C_T_H_W=input_latents.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            crossattn_emb=data_batch["t5_text_embeddings"],
            condition_video_input_mask_B_C_T_H_W=condition_mask,
            fps=self.fps,
            padding_mask=padding_mask,
            data_type=DataType.VIDEO,
            action=action,
        ).float()  # type: ignore

        timesteps_normalized = timesteps.unsqueeze(1).unsqueeze(-1).unsqueeze(-1) / 1000.0  # (B, 1, T, 1, 1)
        pred_x0 = input_latents - timesteps_normalized * velocity_pred

        loss = torch.mean((pred_x0[:, :, 1:] - target_latents[:, :, 1:]) ** 2)

        output_batch = {
            "x0": pred_x0,
            "xt": input_latents,
            "model_pred": velocity_pred,
            "edm_loss": loss,
        }

        # Best-effort visualization every 100 steps: generate a short causal rollout
        # and expose it for the interactive wandb callback.
        if iteration % 100 == 0:
            try:
                cond, init_noise, n_steps_vis = self._prepare_backward_simulation(
                    data_batch=data_batch,
                    target_latents=target_latents,
                    padding_mask=padding_mask,
                    action=action,
                )

                _ = self.backward_simulation(
                    cond,
                    init_noise,
                    n_steps=n_steps_vis,
                    with_grad=False,
                    dump_iter=iteration,
                )
            except Exception:
                # best-effort visualization; never break training
                pass

        return output_batch, loss
