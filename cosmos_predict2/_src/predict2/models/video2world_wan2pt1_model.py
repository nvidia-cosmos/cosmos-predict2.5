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

from typing import Tuple

import torch
from torch import Tensor

from cosmos_predict2._src.predict2.models.text2world_model import Text2WorldCondition
from cosmos_predict2._src.predict2.models.text2world_wan2pt1_model import (
    DataType,
    Text2WorldModelWan2pt1Config,
    WANDiffusionModel,
)

WAN2PT1_I2V_COND_LATENT_KEY = "i2v_WAN2PT1_cond_latents"  # gitleaks:allow


class I2VWan2pt1Model(WANDiffusionModel):
    def __init__(self, config: Text2WorldModelWan2pt1Config):
        # Note that I2V config.shift has better value {"480p": 3.0, "720p": 5.0}
        super().__init__(config)

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[Tensor, Text2WorldCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # Latent state
        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_data_key]
        latent_state = self.encode(raw_state).contiguous().float()
        if WAN2PT1_I2V_COND_LATENT_KEY not in data_batch:
            conditional_content = torch.zeros_like(raw_state).to(**self.tensor_kwargs)
            if not is_image_batch:
                conditional_content[:, :, 0] = raw_state[:, :, 0]

            data_batch[WAN2PT1_I2V_COND_LATENT_KEY] = self.encode(conditional_content).contiguous()

        # Condition
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        return raw_state, latent_state, condition
