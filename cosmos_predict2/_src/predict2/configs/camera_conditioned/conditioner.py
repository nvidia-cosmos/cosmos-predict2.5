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

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.conditioner import (
    ReMapkey,
    Text2WorldCondition,
)
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import (
    _SHARED_CONFIG,
    Video2WorldCondition,
    Video2WorldConditioner,
    VideoPredictionConditioner,
)
from cosmos_predict2._src.predict2.utils.context_parallel import broadcast_split_tensor


@dataclass(frozen=True)
class CameraConditionedCondition(Video2WorldCondition):
    camera: Optional[torch.Tensor] = None

    def set_camera_conditioned_video_condition(
        self,
        gt_frames: torch.Tensor,
        num_conditional_frames: Optional[int] = None,
    ) -> "CameraConditionedCondition":
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = gt_frames

        # condition_video_input_mask_B_C_T_H_W
        B, _, T, H, W = gt_frames.shape
        condition_video_input_mask_B_C_T_H_W = torch.zeros(
            B, 1, T, H, W, dtype=gt_frames.dtype, device=gt_frames.device
        )
        if T == 1:  # handle image batch
            num_conditional_frames_B = torch.zeros(B, dtype=torch.int32)
        else:  # handle video batch
            if isinstance(num_conditional_frames, torch.Tensor):
                num_conditional_frames_B = torch.ones(B, dtype=torch.int32) * num_conditional_frames.cpu()
            else:
                num_conditional_frames_B = torch.ones(B, dtype=torch.int32) * num_conditional_frames
        for idx in range(B):
            condition_video_input_mask_B_C_T_H_W[
                idx, :, num_conditional_frames_B[idx] : num_conditional_frames_B[idx] * 2, :, :
            ] += 1

        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        return type(self)(**kwargs)

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> "CameraConditionedCondition":
        if self.is_broadcasted:
            return self
        # extra efforts
        gt_frames = self.gt_frames
        condition_video_input_mask_B_C_T_H_W = self.condition_video_input_mask_B_C_T_H_W
        camera = self.camera
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = None
        kwargs["condition_video_input_mask_B_C_T_H_W"] = None
        new_condition = Text2WorldCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )

        kwargs = new_condition.to_dict(skip_underscore=False)
        _, _, T, _, _ = gt_frames.shape
        if process_group is not None:
            if T > 1 and process_group.size() > 1:
                gt_frames = broadcast_split_tensor(gt_frames, seq_dim=2, process_group=process_group)
                condition_video_input_mask_B_C_T_H_W = broadcast_split_tensor(
                    condition_video_input_mask_B_C_T_H_W, seq_dim=2, process_group=process_group
                )
                camera = broadcast_split_tensor(camera, seq_dim=1, process_group=process_group)
        kwargs["gt_frames"] = gt_frames
        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        kwargs["camera"] = camera
        return type(self)(**kwargs)


class CameraConditionedConditioner(Video2WorldConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> CameraConditionedCondition:
        output = super()._forward(batch, override_dropout_rate)
        assert "camera" in output, "CameraConditionedConditioner requires 'camera' in output"
        return CameraConditionedCondition(**output)


@dataclass(frozen=True)
class CameraConditionedARCondition(Video2WorldCondition):
    camera: Optional[torch.Tensor] = None

    def set_camera_conditioned_ar_video_condition(
        self,
        gt_frames: torch.Tensor,
        num_conditional_frames: Optional[int] = None,
    ) -> "CameraConditionedARCondition":
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = gt_frames

        # condition_video_input_mask_B_C_T_H_W
        B, _, T, H, W = gt_frames.shape
        condition_video_input_mask_B_C_T_H_W = torch.zeros(
            B, 1, T, H, W, dtype=gt_frames.dtype, device=gt_frames.device
        )

        if T == 1:  # handle image batch
            num_conditional_frames_B = torch.zeros(B, dtype=torch.int32)
        else:  # handle video batch
            if isinstance(num_conditional_frames, torch.Tensor):
                num_conditional_frames_B = torch.ones(B, dtype=torch.int32) * num_conditional_frames.cpu()
            else:
                num_conditional_frames_B = torch.ones(B, dtype=torch.int32) * num_conditional_frames
        for idx in range(B):
            condition_video_input_mask_B_C_T_H_W[idx, :, : num_conditional_frames_B[idx] * 2, :, :] += 1
            condition_video_input_mask_B_C_T_H_W[idx, :, (-num_conditional_frames_B[idx] * 2) :, :, :] += 1

        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W

        return type(self)(**kwargs)

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> "CameraConditionedARCondition":
        if self.is_broadcasted:
            return self
        # extra efforts
        gt_frames = self.gt_frames
        condition_video_input_mask_B_C_T_H_W = self.condition_video_input_mask_B_C_T_H_W
        camera = self.camera
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = None
        kwargs["condition_video_input_mask_B_C_T_H_W"] = None
        new_condition = Text2WorldCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )

        kwargs = new_condition.to_dict(skip_underscore=False)
        _, _, T, _, _ = gt_frames.shape
        if process_group is not None:
            if T > 1 and process_group.size() > 1:
                gt_frames = broadcast_split_tensor(gt_frames, seq_dim=2, process_group=process_group)
                condition_video_input_mask_B_C_T_H_W = broadcast_split_tensor(
                    condition_video_input_mask_B_C_T_H_W, seq_dim=2, process_group=process_group
                )
                camera = broadcast_split_tensor(camera, seq_dim=1, process_group=process_group)
        kwargs["gt_frames"] = gt_frames
        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        kwargs["camera"] = camera
        return type(self)(**kwargs)


class CameraConditionedARConditioner(Video2WorldConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> CameraConditionedARCondition:
        output = super()._forward(batch, override_dropout_rate)
        assert "camera" in output, "CameraConditionedARConditioner requires 'camera' in output"
        return CameraConditionedARCondition(**output)


CameraConditionedConditionerConfig: LazyDict = L(CameraConditionedConditioner)(
    **_SHARED_CONFIG,
    camera=L(ReMapkey)(
        input_key="camera",
        output_key="camera",
        dropout_rate=0.0,
        dtype=None,
    ),
)

CameraConditionedARConditionerConfig: LazyDict = L(CameraConditionedARConditioner)(
    **_SHARED_CONFIG,
    camera=L(ReMapkey)(
        input_key="camera",
        output_key="camera",
        dropout_rate=0.0,
        dtype=None,
    ),
)


def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_conditioner",
        node=VideoPredictionConditioner,
    )

    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="camera_conditioned_video_conditioner",
        node=CameraConditionedConditionerConfig,
    )

    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="camera_conditioned_ar_video_conditioner",
        node=CameraConditionedARConditionerConfig,
    )
