# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly  prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""
Condition postprocessors for model-specific conditioning logic.

This module obviates the need to modify the get_data_and_condition method
in the distillation/interactive method classes (e.g. dmd2 class).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from cosmos_predict2._src.interactive.configs.method_configs.config_cosmos2_interactive_base import IS_PREPROCESSED_KEY
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.transfer2.configs.vid2vid_transfer.defaults.conditioner import ControlVideo2WorldCondition


class ControlConditionPostprocessor:
    """
    Adds control latent and weights to the condition object for Cosmos-Transfer2 models.

    The implementation mirrors the get_data_and_condition method in the Cosmos-Transfer2 model class:
        cosmos_predict2/_src/transfer2/models/vid2vid_model_control_vace.py

    This postprocessor tokenizes (encodes) control inputs from the data batch (e.g., depth, edge) into latent space
    and attaches them to the condition objects.

    Usage in config:
        condition_postprocessor = L(ControlConditionPostprocessor)(
            hint_keys=["depth"],
        )
    """

    def __init__(self, hint_keys: list[str]):
        """
        Args:
            hint_keys: List of control modality keys (e.g., ["depth", "edge"]).
                       These correspond to attributes on the condition object.
        """
        self.hint_keys = hint_keys

    def __call__(
        self,
        model: Any,
        condition: ControlVideo2WorldCondition,
        uncondition: ControlVideo2WorldCondition,
        latent_state: torch.Tensor,
        data_batch: dict[str, torch.Tensor],
    ) -> tuple[ControlVideo2WorldCondition, ControlVideo2WorldCondition]:
        """
        Process conditions to add control latent and weights.
        Consequently, this turns the condition and uncondition into ControlVideo2WorldCondition objects.

        Args:
            model: The model instance (provides encode(), tokenizer, net, tensor_kwargs, config)
            condition: The positive condition object
            uncondition: The negative/unconditional condition object
            latent_state: The encoded latent state tensor
            data_batch: The input data batch

        Returns:
            Tuple of (modified_condition, modified_uncondition)
        """
        # Handle single-frame video inputs by treating them as images.
        if (
            getattr(model, "input_data_key", None) in data_batch
            and data_batch[model.input_data_key] is not None
            and data_batch[model.input_data_key].dim() == 5
            and data_batch[model.input_data_key].shape[2] == 1
            and getattr(model, "input_image_key", None) not in data_batch
        ):
            video = data_batch[model.input_data_key]
            if video.dtype == torch.uint8:
                video = video.to(**model.tensor_kwargs) / 127.5 - 1.0
            image = video.squeeze(2)  # (B, C, H, W)
            data_batch[model.input_image_key] = image
            data_batch[IS_PREPROCESSED_KEY] = True
            del data_batch[model.input_data_key]

            # Re-encode to keep latent_state aligned; mutate in-place to propagate back.
            image_5d = image.unsqueeze(2)  # (B, C, 1, H, W)
            new_latent = model.encode(image_5d).contiguous().to(**model.tensor_kwargs)
            if latent_state.shape == new_latent.shape:
                latent_state.copy_(new_latent)
            else:
                latent_state = new_latent

            # Treat as image batch for downstream conditioning.
            condition = condition.edit_data_type(DataType.IMAGE)
            uncondition = uncondition.edit_data_type(DataType.IMAGE)

        # Compute control latents and control weights once from the positive condition
        # and share them between condition and uncondition for classifier-free guidance,
        # matching the teacher model behavior.
        condition_with_control = self.update_condition_with_control_input_latent(
            model, condition, latent_state, data_batch
        )
        latent_control_input = condition_with_control.latent_control_input
        control_weight = condition_with_control.control_context_scale
        condition = condition.set_control_condition(
            latent_control_input=latent_control_input,
            control_weight=control_weight,  # type: ignore
        )
        uncondition = uncondition.set_control_condition(
            latent_control_input=latent_control_input,
            control_weight=control_weight,  # type: ignore
        )
        return condition, uncondition

    def _normalize_control_input_inplace(
        self, data_batch: dict[str, torch.Tensor], parsed_hint_key: str, tensor_kwargs: dict[str, Any]
    ) -> torch.Tensor | None:
        """Normalize control input to [-1, 1] on the correct device/dtype."""
        # Handle control_input if it exists
        control_input = data_batch.get(parsed_hint_key, None)
        if control_input is not None:
            # Normalize control_input if not already normalized
            if control_input.dtype == torch.uint8:
                control_input = control_input.to(**tensor_kwargs) / 127.5 - 1.0
            elif control_input.dtype == torch.bool:
                control_input = control_input.to(**tensor_kwargs)
            data_batch[parsed_hint_key] = control_input
        return control_input

    def update_condition_with_control_input_latent(
        self,
        model: Any,
        condition: "ControlVideo2WorldCondition",
        latent_state: torch.Tensor,
        data_batch: dict[str, torch.Tensor],
    ) -> "ControlVideo2WorldCondition":
        """
        Update the condition with the control input latent.

        Args:
            model: The model instance (provides encode(), tokenizer, net, tensor_kwargs, config)
            condition: The condition to update.
            latent_state: The latent state of the video
            data_batch: The data batch containing the control input and control input mask

        Returns:
            The updated condition with control latent and weights.
        """

        latent_control_input = []
        control_weight = data_batch.get("control_weight", [1.0] * len(self.hint_keys))
        if len(control_weight) == 1:
            control_weight = control_weight * len(self.hint_keys)
        control_weight_maps: list[torch.Tensor | None] = [None] * len(self.hint_keys)

        for hi, hint_key in enumerate(self.hint_keys):
            parsed_hint_key = self._parse_hint_key(condition, hint_key)  # e.g. turn "edge" -> "control_input_edge"
            control_input = self._normalize_control_input_inplace(data_batch, parsed_hint_key, model.tensor_kwargs)
            control_input_mask = getattr(condition, parsed_hint_key + "_mask", None)
            if control_input_mask is not None and (
                control_input_mask.dtype == torch.uint8 or control_input_mask.dtype == torch.bool
            ):
                control_input_mask = control_input_mask.to(**model.tensor_kwargs)

            if control_input is not None and control_input.dim() == 5 and control_input.shape[2] > 1:
                expected_length = model.tokenizer.get_pixel_num_frames(model.config.state_t)
                original_length = control_input.shape[2]
                assert original_length == expected_length, (
                    "Input control_input length doesn't match expected length specified by state_t."
                )
            latent_control_input += self.get_control_latent(model, latent_state, control_input, control_input_mask)

            if not torch.is_grad_enabled() and not model.net.vace_has_mask:  # inference mode
                if control_input is None:  # set control weight to 0 if no control input
                    if len(control_weight) == len(self.hint_keys):
                        control_weight[hi] = 0.0
                    else:
                        control_weight.insert(hi, 0.0)
                if control_input_mask is not None and (control_input_mask != 1).any():
                    # use control weight to implement masking operation
                    assert control_input_mask.shape[1] == 1, (
                        f"control_input_mask.shape[1] != 1: {control_input_mask.shape[1]}"
                    )
                    control_weight_maps[hi] = control_input_mask * control_weight[hi]

        # If any control mask exists, use spatio-temporal control weight instead of scalar
        if any(c is not None for c in control_weight_maps):
            for hi in range(len(self.hint_keys)):
                if control_weight_maps[hi] is None:
                    # convert scalar control weight to spatio-temporal control weight
                    control_weight_maps[hi] = control_weight[hi] * torch.ones_like(
                        next(c for c in control_weight_maps if c is not None)
                    )
            stacked_maps = torch.stack(control_weight_maps)
            control_weight = self.resize_control_weight(model, stacked_maps, latent_state)

        latent_control_input = torch.cat(latent_control_input, dim=1)
        condition = condition.set_control_condition(
            latent_control_input=latent_control_input,
            control_weight=control_weight,
        )
        return condition

    def get_control_latent(
        self,
        model: Any,
        latent_state: torch.Tensor,
        control_input: torch.Tensor | None,
        control_input_mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Encode control input into latent space."""
        latent_control_input = []
        if control_input is not None and not (control_input == -1).all():
            if model.net.vace_has_mask:
                if control_input_mask is None or (control_input_mask == 0).all():
                    control_input_mask = torch.ones_like(control_input[:, :1])
                assert control_input_mask.shape[1] == 1, (
                    f"control_input_mask.shape[1] != 1: {control_input_mask.shape[1]}"
                )
                fg = (control_input + 1) / 2 * control_input_mask * 2 - 1
                latent_control_input.append(model.encode(fg).contiguous().to(**model.tensor_kwargs))

                # reshape spatial patch to channel dimension
                ph = pw = model.tokenizer.spatial_compression_factor
                mask = rearrange(control_input_mask, "b c t (h ph) (w pw) -> b (c ph pw) t h w", ph=ph, pw=pw)
                if mask.shape[2] > 1:
                    # interpolate to t frames
                    temporal_compression_factor = model.tokenizer.temporal_compression_factor
                    target_t = latent_state.shape[2]
                    assert control_input_mask.shape[2] == temporal_compression_factor * (target_t - 1) + 1, (
                        f"control_input_mask.shape[2] {control_input_mask.shape[2]} != target_t_expanded {temporal_compression_factor * (target_t - 1) + 1}"
                    )
                    H, W = mask.shape[-2:]
                    mask = torch.cat(
                        [
                            mask[:, :, :1],
                            F.interpolate(mask[:, :, 1:], size=(target_t - 1, H, W), mode="nearest-exact"),
                        ],
                        dim=2,
                    )
                latent_control_input.append(mask.contiguous().to(**model.tensor_kwargs))
            else:
                latent_control_input.append(model.encode(control_input).contiguous().to(**model.tensor_kwargs))
        else:
            if model.net.vace_has_mask:
                ch = latent_state.shape[1] + model.tokenizer.spatial_compression_factor**2
                zero_latent_state = (
                    torch.zeros_like(latent_state[:, :1]).repeat(1, ch, 1, 1, 1).to(**model.tensor_kwargs)
                )
            else:
                zero_latent_state = torch.zeros_like(latent_state).to(**model.tensor_kwargs)
            latent_control_input.append(zero_latent_state)
        return latent_control_input

    def resize_control_weight(
        self,
        model: Any,
        control_context_scale: torch.Tensor,
        latent_state: torch.Tensor,
    ) -> torch.Tensor:
        """Resize spatio-temporal control weight maps to match latent state shape."""
        temporal_compression_factor = model.tokenizer.temporal_compression_factor
        control_weight_maps = [w for w in control_context_scale]
        _, _, T, H, W = latent_state.shape
        H = H // model.net.patch_spatial
        W = W // model.net.patch_spatial
        weight_maps = []

        for weight_map in control_weight_maps:  # [B, 1, T, H, W]
            if weight_map.shape[2:5] != (T, H, W):
                assert weight_map.shape[2] == temporal_compression_factor * (T - 1) + 1, (
                    f"{weight_map.shape[2]} != {temporal_compression_factor * (T - 1) + 1}"
                )
                weight_map_i = [
                    F.interpolate(weight_map[:, :, :1, :, :], size=(1, H, W), mode="trilinear", align_corners=False)
                ]
                weight_map_i += [
                    F.interpolate(weight_map[:, :, 1:], size=(T - 1, H, W), mode="trilinear", align_corners=False)
                ]
                weight_map = torch.cat(weight_map_i, dim=2)

            # Reshape to match BTHWD format
            weight_map = weight_map.permute(0, 2, 3, 4, 1)  # [B, T, H, W, 1]
            weight_maps.append(weight_map)

        control_weight_maps = torch.stack(weight_maps).to(dtype=model.precision)
        # Cap the sum over dim0 at each T,H,W position to be at most 1.0
        max_control_weight_sum = 1.0
        sum_over_modalities = control_weight_maps.sum(dim=0)
        max_values = torch.clamp_min(sum_over_modalities, max_control_weight_sum)
        scale_factors = max_control_weight_sum / max_values
        control_weight_maps = control_weight_maps * scale_factors[None]
        return control_weight_maps

    def _parse_hint_key(self, condition: "ControlVideo2WorldCondition", hint_key: str) -> str:
        """
        if passed hint_key uses "control_input_" prefix, return the prefixed key
        otherwise (e.g. "edge"), add the 'control_input_' prefix to match the Cosmos-Transfer
        style hint key naming convention.
        """
        if hasattr(condition, hint_key):
            return hint_key
        prefixed = f"control_input_{hint_key}"
        if hasattr(condition, prefixed):
            return prefixed
        return hint_key


class ActionConditionPostprocessor:
    def __init__(self):
        raise NotImplementedError


class CameraConditionPostprocessor:
    def __init__(self):
        raise NotImplementedError
