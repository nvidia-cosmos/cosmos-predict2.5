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
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import torch
from einops import rearrange
from hydra.core.config_store import ConfigStore
from omegaconf import ListConfig

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast_split_tensor
from cosmos_predict2._src.imaginaire.utils.validator import Validator
from cosmos_predict2._src.predict2.conditioner import Text2WorldCondition, TextAttr
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import (
    _SHARED_CONFIG,
    GeneralConditioner,
    ReMapkey,
    Video2WorldCondition,
)


class ConditionLocation(Enum):
    """
    Enum representing different camera condition locations for anymulti-to-multiview video generation.

    Attributes:
        NO_CAM: Indicates no camera is used for conditioning (i.e text2world)
        REF_CAM: Indicates a reference camera is used for conditioning. (i.e single-to-multiview-text2world)
        ANY_CAM: Indicates any camera can be used for conditioning. (i.e any-to-multiview-text2world)
        FIRST_RANDOM_N: Indicates a random number of frames from all cameras are used for conditioning. (i.e video2world-multiview)

    Note: Multiple locations can be set together when compatible.
        - NO_CAM cannot be set with any other location.
        - ANY_CAM and REF_CAM cannot be set simultaneously.
        - FIRST_RANDOM_N can be set with ANY_CAM or REF_CAM.
    """

    NO_CAM = "no_cam"
    REF_CAM = "ref_cam"
    ANY_CAM = "any_cam"
    FIRST_RANDOM_N = "first_random_n"


class ConditionLocationListValidator(Validator):
    """
    Validator for a list of ConditionLocation objects.
    Validates that:
        - NO_CAM is not set with any other location
        - ANY_CAM and REF_CAM are not set together
    """

    def __init__(self, default: List[ConditionLocation], hidden=False, tooltip=None):
        self.default = default
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value: List[ConditionLocation]):
        for v in value:
            if not isinstance(v, ConditionLocation):
                raise TypeError(f"All elements must be ConditionLocation enums, got {type(v)}: {v}")
        if ConditionLocation.NO_CAM in value:
            assert len(value) == 1, f"Cannot set ConditionLocation.NO_CAM and other locations together. Got {value=}"
        elif ConditionLocation.ANY_CAM in value and ConditionLocation.REF_CAM in value:
            raise ValueError("ConditionLocation.ANY_CAM and ConditionLocation.REF_CAM cannot be set together.")
        return value

    def __repr__(self) -> str:
        return f"ConditionLocationValidator({self.default=}, {self.hidden=})"

    def json(self):
        return {
            "type": ConditionLocationListValidator.__name__,
            "default": self.default,
            "tooltip": self.tooltip,
        }


class ConditionLocationList(list):
    def __init__(self, locations: List[ConditionLocation]):
        enum_locations = []
        for loc in locations:
            if not isinstance(loc, ConditionLocation):
                loc = ConditionLocation(loc)  # Will raise ValueError if invalid
            enum_locations.append(loc)
        super().__init__(enum_locations)
        self.validator = ConditionLocationListValidator(default=[])
        self.validator.validate(self)

    def __repr__(self) -> str:
        return f"ConditionLocationList({super().__repr__()})"

    def to_json(self):
        return {
            "type": ConditionLocationList.__name__,
            "locations": [location.value for location in self],
        }


@dataclass(frozen=True)
class MultiViewCondition(Video2WorldCondition):
    state_t: Optional[int] = None
    view_indices_B_T: Optional[torch.Tensor] = None
    ref_cam_view_idx_sample_position: Optional[torch.Tensor] = None

    def set_video_condition(
        self,
        state_t: int,
        gt_frames: torch.Tensor,
        condition_locations: Union[ConditionLocationList, ListConfig] = field(
            default_factory=lambda: ConditionLocationList([])
        ),
        random_min_num_conditional_frames_per_view: Optional[int] = None,
        random_max_num_conditional_frames_per_view: Optional[int] = None,
        num_conditional_frames_per_view: Optional[int] = None,
        condition_cam_idx: Optional[int] = None,
        view_condition_dropout_max: int = 0,
        conditional_frames_probs: Optional[Dict[int, float]] = None,
    ) -> "MultiViewCondition":
        """
        Sets the video conditioning frames for anymulti-to-multiview generation.

        This method creates a conditioning mask for the input video frames that determines
        which frames will be used as context frames for generating new frames. The method
        handles video batches (T>1) and does not support images (T=1).

        Args:
            gt_frames: A tensor of ground truth frames with shape [B, C, T, H, W], where:
                B = batch size
                C = number of channels
                T = number of frames per view * self.sample_n_views
                H = height
                W = width

            random_min_num_conditional_frames_per_view: Minimum number of frames per view to use for conditioning
                when randomly selecting a number of conditioning frames.

            random_max_num_conditional_frames_per_view: Maximum number of frames per view to use for conditioning
                when randomly selecting a number of conditioning frames.

            num_conditional_frames_per_view: Optional; If provided, all examples in the batch will use
                exactly this many frames per view for conditioning. If None, a random number of frames per view
                between random_min_num_conditional_frames_per_view and random_max_num_conditional_frames_per_view
                will be selected for each example in the batch.

            condition_cam_idx: Optional; Used only if ConditionLocation.ANY_CAM is in condition_locations.
                If provided, all examples in the batch will use the same cam_idx for conditioning. If None,
                a random cam_idx will be selected for each example in the batch.
            view_condition_dropout_max: Optional; If provided and > 0, then a random number of views will be dropped from the conditioning.

            conditional_frames_probs: Optional; Dictionary mapping number of frames to probabilities.
                If provided, overrides the random_min/max_num_conditional_frames with weighted sampling.
                Example: {0: 0.5, 1: 0.25, 2: 0.25} for 50% chance of 0 frames, 25% for 1, 25% for 2.

        Returns:
            A new MultiViewCondition object with the gt_frames and conditioning mask set.
            The conditioning mask (condition_video_input_mask_B_C_T_H_W) is a binary tensor
            of shape [B, 1, T, H, W] where 1 indicates frames used for conditioning and 0
            indicates frames to be generated.

        Notes:
            - Image batches are not supported.
            - For video batches multiple condition_locations can be provided and combined:
                - If num_conditional_frames_per_view is provided and "random_n" is in condition_locations,
                then all examples will use the same number of frames per view for conditioning,
                otherwise, if num_conditional_frames_per_view is not provided,
                then each example will randomly uses between random_min_num_conditional_frames_per_view
                and random_max_num_conditional_frames_per_view frames per view.
                - If "ref_cam" is in condition_locations, then for each example,
                all frames of the first view will be used for conditioning.
        """
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["state_t"] = state_t
        kwargs["gt_frames"] = gt_frames
        B, _, T, H, W = gt_frames.shape

        if not isinstance(condition_locations, ConditionLocationList):
            condition_locations = ConditionLocationList(condition_locations)
        assert len(condition_locations) > 0, "condition_locations must be provided."
        assert state_t is not None, "state_t must be provided."
        assert T > 1, "Image batches are not supported."
        assert T % state_t == 0, f"T must be a multiple of state_t. Got T={T} and state_t={state_t}."
        sample_n_views = T // state_t
        condition_video_input_mask_B_C_V_T_H_W = torch.zeros(
            B, 1, sample_n_views, state_t, H, W, dtype=gt_frames.dtype, device=gt_frames.device
        )
        views_eligible_for_dropout = list(range(sample_n_views))

        if ConditionLocation.REF_CAM in condition_locations:
            ref_cam_view_idx_sample_position = kwargs["ref_cam_view_idx_sample_position"]
            ref_cam_idx_B = (
                torch.ones(B, dtype=torch.int32, device=ref_cam_view_idx_sample_position.device)
                * ref_cam_view_idx_sample_position
            )
            condition_video_input_mask_B_C_V_T_H_W = self.enable_ref_cam_condition(
                ref_cam_idx_B, condition_video_input_mask_B_C_V_T_H_W
            )
            assert (ref_cam_view_idx_sample_position == ref_cam_view_idx_sample_position[0]).all(), (
                f"ref_cam_view_idx_sample_position must be the same for all examples. Got {ref_cam_view_idx_sample_position=}"
            )
            ref_cam_view_idx_sample_position_int = ref_cam_view_idx_sample_position[0].cpu().item()
            views_eligible_for_dropout.remove(ref_cam_view_idx_sample_position_int)
        elif ConditionLocation.ANY_CAM in condition_locations:
            if condition_cam_idx is None:
                assert kwargs["view_indices_B_T"].shape[-1] % sample_n_views == 0, (
                    f"view_indices_B_T last dimension must be a multiple of sample_n_views. Got view_indices_B_T.shape={kwargs['view_indices_B_T'].shape}, sample_n_views={sample_n_views}"
                )
                view_indices = kwargs["view_indices_B_T"]
                selected_cam_latent_t_index = torch.randint(0, state_t, size=(B,))
                any_cam_idx_B = view_indices[torch.arange(B), selected_cam_latent_t_index]
            else:
                any_cam_idx_B = torch.full((B,), condition_cam_idx, dtype=torch.int32)
            condition_video_input_mask_B_C_V_T_H_W = self.enable_ref_cam_condition(
                any_cam_idx_B, condition_video_input_mask_B_C_V_T_H_W
            )
            assert (any_cam_idx_B == any_cam_idx_B[0]).all(), (
                f"any_cam_idx_B must be the same for all examples. Got {any_cam_idx_B=}"
            )
            any_cam_idx_B_int = any_cam_idx_B[0].cpu().item()
            views_eligible_for_dropout.remove(any_cam_idx_B_int)
        if ConditionLocation.FIRST_RANDOM_N in condition_locations:
            if (
                num_conditional_frames_per_view is None
                and random_min_num_conditional_frames_per_view == random_max_num_conditional_frames_per_view
            ):
                num_conditional_frames_per_view = random_min_num_conditional_frames_per_view
            if num_conditional_frames_per_view is not None:
                num_conditional_frames_per_view_B = torch.ones(B, dtype=torch.int32) * num_conditional_frames_per_view
            elif conditional_frames_probs is not None:
                # Use weighted sampling based on provided probabilities
                frames_options = list(conditional_frames_probs.keys())
                weights = list(conditional_frames_probs.values())
                num_conditional_frames_per_view_B = torch.tensor(
                    random.choices(frames_options, weights=weights, k=B), dtype=torch.int32
                )
            else:
                assert (
                    random_min_num_conditional_frames_per_view is not None
                    and random_max_num_conditional_frames_per_view is not None
                ), (
                    f"random_min_num_conditional_frames_per_view and random_max_num_conditional_frames_per_view must be provided if num_conditional_frames_per_view is None. Got {random_min_num_conditional_frames_per_view=}, {random_max_num_conditional_frames_per_view=}, {num_conditional_frames_per_view=}"
                )
                num_conditional_frames_per_view_B = torch.randint(
                    random_min_num_conditional_frames_per_view,
                    random_max_num_conditional_frames_per_view + 1,
                    size=(B,),
                )
            condition_video_input_mask_B_C_V_T_H_W = self.enable_first_random_n_condition(
                condition_video_input_mask_B_C_V_T_H_W, num_conditional_frames_per_view_B
            )
        if view_condition_dropout_max > 0:
            random.shuffle(views_eligible_for_dropout)
            n_views_to_dropout = random.randint(0, view_condition_dropout_max)
            views_to_dropout = views_eligible_for_dropout[:n_views_to_dropout]
            for view_idx in views_to_dropout:
                condition_video_input_mask_B_C_V_T_H_W[:, :, view_idx] = 0

        condition_video_input_mask_B_C_T_H_W = rearrange(
            condition_video_input_mask_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W", V=sample_n_views
        )
        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        return type(self)(**kwargs)

    def enable_ref_cam_condition(self, cam_idx_B: torch.Tensor, condition_video_input_mask_B_C_V_T_H_W: torch.Tensor):
        """
        Sets condition video input mask to 1 for all frames of the cam_idx[i] view in each example i
        Args:
            cam_idx_B: A tensor of shape [B]
            condition_video_input_mask_B_C_V_T_H_W: A tensor of shape [B, 1, V, T, H, W]
            where V is the number of views, T is the number of frames per view, H is the height, and W is the width
        Returns:
            A copy of the condition video input mask with the cam_idx[i] view set to 1 for example i
        """
        assert condition_video_input_mask_B_C_V_T_H_W.ndim == 6, (
            f"condition_video_input_mask_B_C_V_T_H_W must have 6 dimensions. Got {condition_video_input_mask_B_C_V_T_H_W.shape=}"
        )
        assert cam_idx_B.ndim == 1, f"cam_idx_B must have 1 dimension. Got {cam_idx_B.shape=}"
        copy_condition_video_input_mask_B_C_V_T_H_W = condition_video_input_mask_B_C_V_T_H_W.clone()
        for i in range(copy_condition_video_input_mask_B_C_V_T_H_W.shape[0]):
            copy_condition_video_input_mask_B_C_V_T_H_W[i, :, cam_idx_B[i]] = 1
        return copy_condition_video_input_mask_B_C_V_T_H_W

    def enable_first_random_n_condition(
        self, condition_video_input_mask_B_C_V_T_H_W: torch.Tensor, num_conditional_frames_per_view_B: torch.Tensor
    ):
        """
        Sets condition video input mask to 1 for the first num_conditional_frames_per_view_B frames of each view
        Args:
            condition_video_input_mask_B_C_V_T_H_W: A tensor of shape [B, 1, V, T, H, W]
            num_conditional_frames_per_view_B: A tensor of shape [B]
        Returns:
            A copy of the condition video input mask with the first num_conditional_frames_per_view_B frames of each view set to 1
        """
        assert condition_video_input_mask_B_C_V_T_H_W.ndim == 6, (
            "condition_video_input_mask_B_C_V_T_H_W must have 6 dimensions"
        )
        B, _, _, _, _, _ = condition_video_input_mask_B_C_V_T_H_W.shape
        copy_condition_video_input_mask_B_C_V_T_H_W = condition_video_input_mask_B_C_V_T_H_W.clone()
        for idx in range(B):
            copy_condition_video_input_mask_B_C_V_T_H_W[idx, :, :, : num_conditional_frames_per_view_B[idx]] = 1
        return copy_condition_video_input_mask_B_C_V_T_H_W

    def edit_for_inference(
        self,
        condition_locations: Union[ConditionLocationList, ListConfig] = field(
            default_factory=lambda: ConditionLocationList([])
        ),
        is_cfg_conditional: bool = True,
        num_conditional_frames_per_view: int = 1,
    ) -> "MultiViewCondition":
        _condition = self.set_video_condition(
            state_t=self.state_t,
            gt_frames=self.gt_frames,
            condition_locations=condition_locations,
            random_min_num_conditional_frames_per_view=0,
            random_max_num_conditional_frames_per_view=0,
            num_conditional_frames_per_view=num_conditional_frames_per_view,
            view_condition_dropout_max=0,
        )
        if not is_cfg_conditional:
            # Do not use classifier free guidance on conditional frames.
            # YB found that it leads to worse results.
            _condition.use_video_condition.fill_(True)
        return _condition

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> "MultiViewCondition":
        if self.is_broadcasted:
            return self
        gt_frames_B_C_T_H_W = self.gt_frames
        view_indices_B_T = self.view_indices_B_T
        condition_video_input_mask_B_C_T_H_W = self.condition_video_input_mask_B_C_T_H_W
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = None
        kwargs["condition_video_input_mask_B_C_T_H_W"] = None
        kwargs["view_indices_B_T"] = None
        new_condition = Text2WorldCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )

        kwargs = new_condition.to_dict(skip_underscore=False)
        _, _, T, _, _ = gt_frames_B_C_T_H_W.shape
        n_views = T // self.state_t
        assert T % self.state_t == 0, f"T must be a multiple of state_t. Got T={T} and state_t={self.state_t}."
        if process_group is not None:
            if T > 1 and process_group.size() > 1:
                log.debug(f"Broadcasting {gt_frames_B_C_T_H_W.shape=} to {n_views=} views")
                gt_frames_B_C_V_T_H_W = rearrange(gt_frames_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=n_views)
                condition_video_input_mask_B_C_V_T_H_W = rearrange(
                    condition_video_input_mask_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=n_views
                )
                view_indices_B_V_T = rearrange(view_indices_B_T, "B (V T) -> B V T", V=n_views)

                gt_frames_B_C_V_T_H_W = broadcast_split_tensor(
                    gt_frames_B_C_V_T_H_W, seq_dim=3, process_group=process_group
                )
                condition_video_input_mask_B_C_V_T_H_W = broadcast_split_tensor(
                    condition_video_input_mask_B_C_V_T_H_W, seq_dim=3, process_group=process_group
                )
                view_indices_B_V_T = broadcast_split_tensor(view_indices_B_V_T, seq_dim=2, process_group=process_group)

                gt_frames_B_C_T_H_W = rearrange(gt_frames_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W", V=n_views)
                condition_video_input_mask_B_C_T_H_W = rearrange(
                    condition_video_input_mask_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W", V=n_views
                )
                view_indices_B_T = rearrange(view_indices_B_V_T, "B V T -> B (V T)", V=n_views)

        kwargs["gt_frames"] = gt_frames_B_C_T_H_W
        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        kwargs["view_indices_B_T"] = view_indices_B_T
        return type(self)(**kwargs)


class MultiViewConditioner(GeneralConditioner):
    def forward(self, batch: Dict, override_dropout_rate: Optional[Dict[str, float]] = None) -> MultiViewCondition:
        output = super()._forward(batch, override_dropout_rate)
        return MultiViewCondition(**output)


MultiViewConditionerConfig: LazyDict = L(MultiViewConditioner)(
    **_SHARED_CONFIG,
    view_indices_B_T=L(ReMapkey)(
        input_key="latent_view_indices_B_T",
        output_key="view_indices_B_T",
        dropout_rate=0.0,
        dtype=None,
    ),
    ref_cam_view_idx_sample_position=L(ReMapkey)(
        input_key="ref_cam_view_idx_sample_position",
        output_key="ref_cam_view_idx_sample_position",
        dropout_rate=0.0,
        dtype=None,
    ),
)


class TextAttrEmptyStringDropout(TextAttr):
    def __init__(
        self,
        input_key: str,
        pos_input_key: str,
        dropout_input_key: str,
        dropout_rate: Optional[float] = 0.0,
        use_empty_string: bool = False,
        **kwargs,
    ):
        self._input_key = input_key
        self._pos_input_key = pos_input_key
        self._dropout_input_key = dropout_input_key
        self._dropout_rate = dropout_rate
        self._use_empty_string = use_empty_string
        super().__init__(input_key, dropout_rate)

    def forward(self, tensor: torch.Tensor):
        return {"crossattn_emb": tensor}

    def random_dropout_input(
        self,
        in_tensor_dict: torch.Tensor | Dict[str, torch.Tensor],
        dropout_rate: Optional[float] = None,
        key: Optional[str] = None,
    ) -> torch.Tensor:
        if key is not None and "mask" in key:
            return in_tensor_dict
        del key
        assert isinstance(in_tensor_dict, dict), f"in_tensor_dict must be a dict. Got {type(in_tensor_dict)}"
        in_tensor = in_tensor_dict[self._pos_input_key]
        B = in_tensor.shape[0]
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        keep_mask = torch.bernoulli((1.0 - dropout_rate) * torch.ones(B)).type_as(in_tensor)
        if self._use_empty_string:
            empty_prompt = in_tensor_dict[self._dropout_input_key]
            if empty_prompt.shape[0] != B:
                empty_prompt = empty_prompt.repeat(B, 1, 1)
        else:
            empty_prompt = torch.zeros_like(in_tensor)

        return keep_mask * in_tensor + (1 - keep_mask) * empty_prompt

    def details(self) -> str:
        return "Output key: [crossattn_emb]"


def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_multiview_conditioner",
        node=MultiViewConditionerConfig,
    )
