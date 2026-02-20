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

import os
from typing import Callable, Dict, Tuple

import attrs
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

from cosmos_predict2._src.imaginaire.checkpointer.s3_filesystem import S3StorageReader
from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.distributed import get_rank, get_world_size
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.checkpointer.dcp import ModelWrapper
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)
from cosmos_predict2._src.transfer2.configs.vid2vid_transfer.defaults.conditioner import ControlVideo2WorldCondition
from cosmos_predict2._src.transfer2.datasets.augmentors.control_input import CTRL_HINT_KEYS

IS_PREPROCESSED_KEY = "is_preprocessed"


@attrs.define(slots=False)
class ControlVideo2WorldRectifiedFlowConfig(Video2WorldModelRectifiedFlowConfig):
    base_load_from: LazyDict = None
    min_num_conditional_frames: int = 0  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    copy_weight_strategy: str = (
        "first_n"  # How to copy weights from base model to control branch. "first_n" or "spaced_n"
    )
    hint_keys: str = "_".join([key.replace("control_input_", "") for key in CTRL_HINT_KEYS.keys()])
    use_reference_image: bool = False  # Whether to use reference image as control input


class ControlVideo2WorldModelRectifiedFlow(Video2WorldModelRectifiedFlow):
    """
    ImaginaireModel instance of the VACE-styled controlnet for training.
    """

    def __init__(self, config: ControlVideo2WorldRectifiedFlowConfig, *args, **kwargs):
        self.is_new_training = True
        self.copy_weight_strategy = config.copy_weight_strategy
        self.hint_keys = ["control_input_" + key for key in config.hint_keys.split("_")]
        super().__init__(config, *args, **kwargs)
        log.info(self.net, rank0_only=True)

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, ControlVideo2WorldCondition]:
        # Get base data and condition
        if self.input_data_key in data_batch and data_batch[self.input_data_key].shape[2] == 1:
            data_batch[self.input_image_key] = data_batch[self.input_data_key].squeeze(2)
            assert data_batch[self.input_image_key].dtype == torch.uint8, "Image data is not in uint8 format."
            data_batch[self.input_image_key] = data_batch[self.input_image_key].to(**self.tensor_kwargs) / 127.5 - 1.0
            del data_batch[self.input_data_key]
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        # Add control conditioning
        latent_control_input = []
        control_weight = data_batch.get("control_weight", [1.0] * len(self.hint_keys))
        if len(control_weight) == 1:
            control_weight = control_weight * len(self.hint_keys)
        control_weight_maps = [None] * len(self.hint_keys)  # spatio-temporal control weight
        for hi, hint_key in enumerate(self.hint_keys):
            control_input = getattr(condition, hint_key, None)
            control_input_mask = getattr(condition, hint_key + "_mask", None)
            latent_control_input += self.get_control_latent(latent_state, control_input, control_input_mask)
            if not torch.is_grad_enabled() and not self.net.vace_has_mask:  # inference mode
                if control_input is None:  # set control weight to 0 if no control input
                    if len(control_weight) == len(self.hint_keys):
                        control_weight[hi] = 0.0
                    else:
                        control_weight.insert(hi, 0.0)
                if (
                    control_input_mask is not None and (control_input_mask != 1).any()
                ):  # use control weight to implement masking operation
                    assert control_input_mask.shape[1] == 1, (
                        f"control_input_mask.shape[1] != 1: {control_input_mask.shape[1]}"
                    )
                    control_weight_maps[hi] = control_input_mask * control_weight[hi]
        # If any control mask exists, use spatio-temporal control weight instead of scalar control weight.
        if any(c is not None for c in control_weight_maps):
            for hi in range(len(self.hint_keys)):
                if control_weight_maps[hi] is None:  # convert scalar control weight to spatio-temporal control weight
                    control_weight_maps[hi] = control_weight[hi] * torch.ones_like(
                        next(c for c in control_weight_maps if c is not None)
                    )
            control_weight_maps = torch.stack(control_weight_maps)
            # resize spatio-temporal control weight to match latent_state shape
            control_weight = self.resize_control_weight(control_weight_maps, latent_state)

        # assert num_modalities > 0, "No control input found"
        latent_control_input = torch.cat(latent_control_input, dim=1)
        condition = condition.set_control_condition(
            latent_control_input=latent_control_input,
            control_weight=control_weight,
        )

        return raw_state, latent_state, condition

    def resize_control_weight(self, control_context_scale: Tensor, latent_state: Tensor) -> Tensor:
        temporal_compression_factor = self.tokenizer.temporal_compression_factor
        control_weight_maps = [w for w in control_context_scale]  # Keep as tensor
        _, _, T, H, W = latent_state.shape
        H = H // self.net.patch_spatial  # spatial patch size
        W = W // self.net.patch_spatial  # spatial patch size
        weight_maps = []
        for weight_map in control_weight_maps:  # [B, 1, T, H, W]
            if weight_map.shape[2:5] != (T, H, W):
                assert weight_map.shape[2] == temporal_compression_factor * (T - 1) + 1, (
                    f"{weight_map.shape[2]} != {temporal_compression_factor * (T - 1) + 1}"
                )
                weight_map_i = [
                    torch.nn.functional.interpolate(
                        weight_map[:, :, :1, :, :],
                        size=(1, H, W),
                        mode="trilinear",
                        align_corners=False,
                    )
                ]
                weight_map_i += [
                    torch.nn.functional.interpolate(
                        weight_map[:, :, 1:],
                        size=(T - 1, H, W),
                        mode="trilinear",
                        align_corners=False,
                    )
                ]
                weight_map = torch.cat(weight_map_i, dim=2)

            # Reshape to match BTHWD format
            weight_map = weight_map.permute(0, 2, 3, 4, 1)  # [B, T, H, W, 1]
            weight_maps.append(weight_map)
        control_weight_maps = weight_maps
        control_weight_maps = torch.stack(control_weight_maps)
        # Cap the sum over dim0 at each T,H,W position to be at most 1.0
        # control_weight_maps shape: [num_modalities, B, T, H, W, 1]
        max_control_weight_sum = 1.0
        sum_over_modalities = control_weight_maps.sum(dim=0)  # [B, T, H, W, 1]
        max_values = torch.clamp_min(sum_over_modalities, max_control_weight_sum)  # [B, T, H, W, 1]
        scale_factors = max_control_weight_sum / max_values  # [B, T, H, W, 1]
        control_weight_maps = control_weight_maps * scale_factors[None]  # [num_modalities, B, T, H, W, 1]
        return control_weight_maps

    def get_control_latent(self, latent_state: Tensor, control_input: Tensor, control_input_mask: Tensor) -> Tensor:
        latent_control_input = []
        if control_input is not None and not (control_input == -1).all():
            if self.net.vace_has_mask:
                if control_input_mask is None or (control_input_mask == 0).all():
                    control_input_mask = torch.ones_like(control_input[:, :1])
                assert control_input_mask.shape[1] == 1, (
                    f"control_input_mask.shape[1] != 1: {control_input_mask.shape[1]}"
                )
                fg = (control_input + 1) / 2 * control_input_mask * 2 - 1
                latent_control_input.append(self.encode(fg).contiguous().to(**self.tensor_kwargs))

                # reshape 8x8 spatial patch to channel dimension
                ph = pw = self.tokenizer.spatial_compression_factor
                mask = rearrange(control_input_mask, "b c t (h ph) (w pw) -> b (c ph pw) t h w", ph=ph, pw=pw)
                if mask.shape[2] > 1:
                    # interpolate to t frames
                    t = self.config.state_t
                    assert control_input_mask.shape[2] == 4 * (t - 1) + 1, (
                        f"control_input_mask.shape[2] != 4 * (t - 1) + 1: {control_input_mask.shape[2]} != {4 * (t - 1) + 1}"
                    )
                    H, W = mask.shape[-2:]
                    mask = [
                        mask[:, :, :1],
                        nn.functional.interpolate(mask[:, :, 1:], size=(t - 1, H, W), mode="nearest-exact"),
                    ]
                    mask = torch.cat(mask, dim=2)
                latent_control_input.append(mask.contiguous().to(**self.tensor_kwargs))
            else:
                latent_control_input.append(self.encode(control_input).contiguous().to(**self.tensor_kwargs))
        else:
            if self.net.vace_has_mask:
                ch = latent_state.shape[1] + self.tokenizer.spatial_compression_factor**2
                zero_latent_state = (
                    torch.zeros_like(latent_state[:, :1]).repeat(1, ch, 1, 1, 1).to(**self.tensor_kwargs)
                )
            else:
                zero_latent_state = torch.zeros_like(latent_state).to(**self.tensor_kwargs)
            latent_control_input.append(zero_latent_state)
        return latent_control_input

    def denoise(self, noise: torch.Tensor, xt_B_C_T_H_W: torch.Tensor, timesteps_B_T: torch.Tensor, condition):
        """
        Override denoise method for control branch support in rectified flow.
        """
        # Handle control conditioning
        if hasattr(condition, "latent_control_input"):
            # The control conditioning is already set in the condition object
            pass

        # Call parent's denoise method
        return super().denoise(noise, xt_B_C_T_H_W, timesteps_B_T, condition)

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `velocity_fn` based on the provided data batch and guidance factor for rectified flow.
        """
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
            log.info(
                f"num_conditional_frames: {num_conditional_frames} is set by data_batch[NUM_CONDITIONAL_FRAMES_KEY]"
            )
        else:
            num_conditional_frames = 0

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, control_condition = self.get_data_and_condition(data_batch)

        # Set video condition
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
        )

        # Set control condition
        latent_control_input = control_condition.latent_control_input
        control_weight = control_condition.control_context_scale
        condition = condition.set_control_condition(
            latent_control_input=latent_control_input, control_weight=control_weight
        )
        uncondition = uncondition.set_control_condition(
            latent_control_input=latent_control_input, control_weight=control_weight
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        world_size = get_world_size()
        rank = get_rank()

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            noise_x = noise_x.to(**self.tensor_kwargs)
            """
            Use CFG parallel with 2 independent CP groups, each performing one denoising step.
            It allows for better scaling, as we get an additional 2x scaling increase. For example if standard
            inference of Cosmos-Transfer2 model hits a scaling wall at 32 GPUs with 8.4s inference time and using 64 GPUs
            provides no additional benefit, we can enable CFG-parallelism to scale the model to 64 GPUs with 4.2s inference time.
            """
            if getattr(self.net, "cfg_parallel", False):
                second_cp_start_rank = world_size // 2

                if rank < second_cp_start_rank:
                    cond_v = self.denoise(noise, noise_x, timestep, condition)
                else:
                    uncond_v = self.denoise(noise, noise_x, timestep, uncondition)

                rec_tensor = torch.empty_like(cond_v if rank < second_cp_start_rank else uncond_v)
                if rank < second_cp_start_rank:
                    torch.distributed.isend(cond_v, dst=second_cp_start_rank + rank)
                    res = torch.distributed.irecv(rec_tensor, src=second_cp_start_rank + rank)
                    res.wait()
                    uncond_v = rec_tensor

                else:
                    torch.distributed.irecv(rec_tensor, src=rank - second_cp_start_rank)
                    res = torch.distributed.isend(uncond_v, dst=rank - second_cp_start_rank)
                    res.wait()
                    cond_v = rec_tensor
            else:
                # Standard path without CFG parallelism
                cond_v = self.denoise(noise, noise_x, timestep, condition)
                uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
            velocity_pred = cond_v + guidance * (cond_v - uncond_v)
            return velocity_pred

        return velocity_fn

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str | None = None) -> None:
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
        super()._normalize_video_databatch_inplace(data_batch, input_key)

        # Handle control_input if it exists
        for key in data_batch.keys():
            if key.startswith("control_input_") and data_batch[key] is not None:
                hint_key = key
                # Normalize control_input if not already normalized
                if data_batch[hint_key].dtype == torch.uint8:
                    data_batch[hint_key] = data_batch[hint_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                elif data_batch[hint_key].dtype == torch.bool:
                    data_batch[hint_key] = data_batch[hint_key].to(**self.tensor_kwargs)

                if data_batch[hint_key].dim() == 5 and data_batch[hint_key].shape[2] > 1:
                    expected_length = self.tokenizer.get_pixel_num_frames(self.config.state_t)
                    original_length = data_batch[hint_key].shape[2]
                    assert original_length == expected_length, (
                        "Input control_input length doesn't match expected length specified by state_t."
                    )

    def _augment_image_dim_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        super()._augment_image_dim_inplace(data_batch, input_key)
        # Handle control_input if it exists
        for key in data_batch.keys():
            if key.startswith("control_input_") and data_batch[key] is not None and data_batch[key].dim() == 4:
                data_batch[key] = rearrange(data_batch[key], "b c h w -> b c 1 h w").contiguous()

    def copy_weights_to_control_branch(self) -> None:
        """
        VACE has the skip design of control blocks: control block i output modulates base block 2i
        In ControlNet training beginning, we copy base model weights to control branch. There are two strategies:
        1. copy base model's i-th block weight to control net's i-th block (more intuitive, the control blocks is a trainable
         copy of the first N layers of the base model)
        2. copy base model's 2i-th block weight to control net's i-th block (follow the correspondence of skip connection, \
           but the block-to-block connection in the control branch is weird.)
        Here we adopt the first strategy.
        """
        if self.is_new_training:
            control_blocks = (
                self.net.control_blocks if self.net.num_control_branches == 1 else self.net.control_blocks_0
            )
            if self.copy_weight_strategy == "first_n":
                # copy base model's i-th block weight to control net's i-th block
                control_to_base_layer_maping = {i: i for i in range(len(control_blocks))}
                assert len(control_to_base_layer_maping) == len(control_blocks)
            elif self.copy_weight_strategy == "spaced_n":
                # copy base model's 2i-th block weight to control net's i-th block
                control_to_base_layer_maping = {v: k for k, v in self.net.control_layers_mapping.items()}
                assert len(control_to_base_layer_maping) == len(control_blocks)
            else:
                raise ValueError("Other copy weight strategy doesn't seem to make sense.")

            # 1. First copy weights from base model to control net
            for control_layer_idx, base_layer_idx in control_to_base_layer_maping.items():
                log.info(
                    f"======Copying base model's {base_layer_idx}-th block weight to control net's {control_layer_idx}-th block"
                )

                if self.net.num_control_branches > 1:
                    for nc in range(self.net.num_control_branches):
                        missing_keys, unexpected_keys = getattr(self.net, f"control_blocks_{nc}")[
                            control_layer_idx
                        ].load_state_dict(self.net.blocks[base_layer_idx].state_dict(), strict=False)
                else:
                    missing_keys, unexpected_keys = self.net.control_blocks[control_layer_idx].load_state_dict(
                        self.net.blocks[base_layer_idx].state_dict(), strict=False
                    )
                assert len(unexpected_keys) == 0, f"unexpected_keys: {unexpected_keys}"
                assert set(missing_keys).issubset(
                    {
                        "before_proj.weight",
                        "before_proj.bias",
                        "after_proj.weight",
                        "after_proj.bias",
                        "_checkpoint_wrapped_module.before_proj.weight",
                        "_checkpoint_wrapped_module.before_proj.bias",
                        "_checkpoint_wrapped_module.after_proj.weight",
                        "_checkpoint_wrapped_module.after_proj.bias",
                    }
                ), f"missing_keys: {missing_keys}"

            if self.net.separate_embedders:
                self.net.t_embedder_for_control_branch.load_state_dict(self.net.t_embedder.state_dict(), strict=True)
                self.net.t_embedding_norm_for_control_branch.load_state_dict(
                    self.net.t_embedding_norm.state_dict(), strict=True
                )
                self.net.x_embedder_for_control_branch.load_state_dict(self.net.x_embedder.state_dict(), strict=True)

            self.is_new_training = False

    def freeze_base_model(self):
        log.info("\nFreezing base model\n")
        # 1. freeze everything
        for param in self.net.parameters():
            param.requires_grad = False

        # 2. unfreeze control-specific parameters: the blocks and patch embedding
        if self.net.num_control_branches > 1:
            for nc in range(self.net.num_control_branches):
                for param in getattr(self.net, f"control_blocks_{nc}").parameters():
                    param.requires_grad = True
            if hasattr(self.net, "after_proj"):
                for param in self.net.after_proj.parameters():
                    param.requires_grad = True
        else:
            for block in self.net.control_blocks:
                for param in block.parameters():
                    param.requires_grad = True

        for param in self.net.control_embedder.parameters():
            param.requires_grad = True

        if self.net.separate_embedders:
            for param in self.net.t_embedder_for_control_branch.parameters():
                param.requires_grad = True
            for param in self.net.t_embedding_norm_for_control_branch.parameters():
                param.requires_grad = True
            for param in self.net.x_embedder_for_control_branch.parameters():
                param.requires_grad = True

        if self.net.use_input_hint_block:
            for param in self.net.input_hint_block.parameters():
                param.requires_grad = True

        # 3. unfreeze reference image weights if we use reference image control
        if self.config.use_reference_image:
            if hasattr(self.net, "img_context_proj"):
                for param in self.net.img_context_proj.parameters():
                    param.requires_grad = True
                log.info("✓ Unfroze img_context_proj")

            # 3.1 Unfreeze reference image weights in each ControlAwareDiTBlock
            if hasattr(self.net, "blocks"):
                for i, block in enumerate(self.net.blocks):
                    # Access the actual block inside CheckpointWrapper
                    actual_block = block._checkpoint_wrapped_module
                    cross_attn = actual_block.cross_attn

                    # Unfreeze k_img, v_img, k_img_norm
                    for param_key in ["k_img", "v_img", "k_img_norm", "q_img", "q_img_norm"]:
                        if hasattr(cross_attn, param_key):
                            for param in getattr(cross_attn, param_key).parameters():
                                param.requires_grad = True

                    log.info(f"✓ Unfroze reference image weights in ControlAwareDiTBlock {i}")

            # 3.2 Unfreeze reference image weights in each ControlEncoderDiTBlock
            if hasattr(self.net, "control_blocks"):
                for i, block in enumerate(self.net.control_blocks):
                    # Access the actual block inside CheckpointWrapper
                    actual_block = block._checkpoint_wrapped_module
                    cross_attn = actual_block.cross_attn

                    # Unfreeze k_img, v_img, k_img_norm
                    for param_key in ["k_img", "v_img", "k_img_norm", "q_img", "q_img_norm"]:
                        if hasattr(cross_attn, param_key):
                            for param in getattr(cross_attn, param_key).parameters():
                                param.requires_grad = True

                    log.info(f"✓ Unfroze reference image weights in ControlEncoderDiTBlock {i}")

    def set_up_model(self):
        super().set_up_model()
        self.freeze_base_model()
        self.load_base_model()
        self.copy_weights_to_control_branch()

    def load_multi_branch_checkpoints(self, checkpoint_paths: list[str]):
        """
        Load control blocks from multiple checkpoint paths into control_blocks_0, control_blocks_1, etc.

        Args:
            checkpoint_paths (list[str]): List of checkpoint paths containing control blocks
        """
        if not checkpoint_paths:
            log.warning("No checkpoint paths provided for control branches")
            return

        # Use the same credentials as base model if available
        credential_path = "credentials/s3_checkpoint.secret"
        if hasattr(self.config, "base_load_from") and self.config.base_load_from is not None:
            credential_path = self.config.base_load_from.credentials

        load_planner = DefaultLoadPlanner(allow_partial_load=False)
        _model_wrapper = ModelWrapper(self)
        _state_dict = _model_wrapper.state_dict()

        # Filter out _extra_state entries to avoid metadata mismatch
        checkpoint_state_dict = {k: v for k, v in _state_dict.items() if "_extra_state" not in k}
        # Replace control_blocks_{nc} with control_blocks in the state dict
        for k in list(checkpoint_state_dict.keys()):
            for nc in range(self.net.num_control_branches):
                if f"control_blocks_{nc}" in k:
                    new_key = k.replace(f"control_blocks_{nc}", "control_blocks")
                    checkpoint_state_dict[new_key] = checkpoint_state_dict.pop(k)
                elif f"control_embedder.{nc}" in k:
                    new_key = k.replace(f"control_embedder.{nc}", "control_embedder")
                    checkpoint_state_dict[new_key] = checkpoint_state_dict.pop(k)

        for nc, checkpoint_path in enumerate(checkpoint_paths):
            if checkpoint_path is None:
                log.warning(f"No checkpoint path provided for control branch {nc}")
                continue

            if not INTERNAL:
                from cosmos_predict2._src.imaginaire.utils.checkpoint_db import download_checkpoint

                checkpoint_path = download_checkpoint(checkpoint_path)

            checkpoint_format = "pt" if checkpoint_path.endswith(".pt") else "dcp"
            # Handle checkpoint path with or without "model" suffix
            cur_key_ckpt_full_path = (
                checkpoint_path
                if checkpoint_path.endswith("model") or checkpoint_format == "pt"
                else os.path.join(checkpoint_path, "model")
            )
            log.critical(f"Start loading checkpoint for control branch {nc} from {checkpoint_path}")

            if "s3://" in checkpoint_path:
                storage_reader = S3StorageReader(
                    credential_path=credential_path,
                    path=cur_key_ckpt_full_path,
                )
            else:
                storage_reader = FileSystemReader(cur_key_ckpt_full_path)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            if checkpoint_format == "dcp":  # load dcp checkpoint
                dcp.load(
                    checkpoint_state_dict,
                    storage_reader=storage_reader,
                    planner=load_planner,
                )
            else:
                # load pytorch checkpoint appending all keys to checkpoint_to_model_keys
                checkpoint_state_dict = torch.load(checkpoint_path)

            # Create mapping from checkpoint keys to model keys
            # Checkpoint has "control_blocks" but we want to load into "control_blocks_{nc}"
            checkpoint_to_model_keys = {}
            for k, v in checkpoint_state_dict.items():
                if "control_blocks." in k:
                    # Replace "control_blocks" with "control_blocks_{nc}" in the key
                    new_key = k.replace("control_blocks", f"control_blocks_{nc}")
                    checkpoint_to_model_keys[new_key] = v
                elif "control_embedder" in k:
                    new_key = k.replace("control_embedder", f"control_embedder.{nc}")
                    checkpoint_to_model_keys[new_key] = v
                else:
                    checkpoint_to_model_keys[k] = v

            assert checkpoint_to_model_keys, f"No control_blocks keys found in checkpoint for branch {nc}"

            log.info(f"Checkpoint to model keys: {checkpoint_to_model_keys}")
            _model_wrapper.load_state_dict(checkpoint_to_model_keys)
            log.info(f"Done loading the control branch {nc} checkpoint.")

    def load_base_model(self, load_ema_to_reg: bool = False) -> None:
        config = self.config
        if config.base_load_from is not None:
            checkpoint_path = config.base_load_from["load_path"]
        else:
            checkpoint_path = None
        if checkpoint_path is not None:
            load_planner = DefaultLoadPlanner(allow_partial_load=True)
            if config.base_load_from.get("credentials", None):
                cur_key_ckpt_full_path = os.path.join("s3://", checkpoint_path, "model")
                if INTERNAL:
                    storage_reader = S3StorageReader(
                        credential_path=config.base_load_from.credentials,
                        path=cur_key_ckpt_full_path,
                    )
                else:
                    from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path

                    checkpoint_path = get_checkpoint_path(cur_key_ckpt_full_path)
            else:
                storage_reader = FileSystemReader(checkpoint_path)

            log.critical(f"Start loading checkpoint for base model from {checkpoint_path}")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            checkpoint_format = "pt" if checkpoint_path.endswith(".pt") else "dcp"

            _model_wrapper = ModelWrapper(
                self, load_ema_to_reg=load_ema_to_reg if checkpoint_format == "dcp" else False
            )
            _state_dict = _model_wrapper.state_dict()

            # Filter out _extra_state entries to avoid metadata mismatch
            filtered_state_dict = {k: v for k, v in _state_dict.items() if "_extra_state" not in k}

            # Copy EMA weights to regular weights
            all_keys = list(filtered_state_dict.keys())
            # log.info(f"All keys: {all_keys}")
            for k in all_keys:
                if k.startswith("net.") and k.replace("net.", "net_ema.") in filtered_state_dict:
                    filtered_state_dict[k] = filtered_state_dict[k.replace("net.", "net_ema.")]
            self.load_state_dict(easy_io.load(checkpoint_path), strict=False)
        log.info("Done loading the base model checkpoint.")

    def get_x_from_clean(
        self,
        in_clean_img: torch.Tensor,
        sigma_max: float | None,
        seed: int = 1,
    ) -> Tensor:
        """
        in_clean_img (torch.Tensor): input clean image for image-to-image/video-to-video by adding noise then denoising
        sigma_max (float): maximum sigma applied to in_clean_image for image-to-image/video-to-video
        """
        if in_clean_img is None:
            return None
        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)
        noise = torch.randn(*in_clean_img.shape, **self.tensor_kwargs, generator=generator)
        if sigma_max is None:
            sigma_max = self.sde.sigma_max
        x_sigma_max = in_clean_img + noise * sigma_max
        return x_sigma_max
