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
Conversion from the TrigFlow (sCM paper) parameterization trigflow_t to
the four c_xxx scaling coefficients as in EDM fomulation.
"""

from typing import Tuple

import torch


# xt (under TrigFlow) = cost*x0/sigma_d + sint*eps
# xt' = x0 + sigma*eps
class EDM_sCMWrapper:
    def __init__(self, sigma_data: float = 1.0):
        self.sigma_data = sigma_data

    def __call__(self, trigflow_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = trigflow_t.dtype
        trigflow_t = trigflow_t.to(torch.float64)
        sigma = torch.tan(trigflow_t) * self.sigma_data
        # c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        # c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        # c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_skip = self.sigma_data * torch.cos(trigflow_t)
        c_out = self.sigma_data * torch.sin(trigflow_t)
        c_in = torch.ones_like(trigflow_t)
        c_noise = 0.25 * sigma.log()
        return c_skip.to(dtype), c_out.to(dtype), c_in.to(dtype), c_noise.to(dtype)


class RectifiedFlow_sCMWrapper:
    def __init__(self, sigma_data: float = 1.0):
        self.sigma_data = sigma_data

    def __call__(self, trigflow_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = trigflow_t.dtype
        trigflow_t = trigflow_t.to(torch.float64)
        # sigma = torch.tan(trigflow_t) * self.sigma_data
        # t = sigma / (sigma + 1)
        # c_skip = 1.0 - t
        # c_out = -t
        # c_in = 1.0 - t
        # c_noise = t
        c_skip = self.sigma_data / (torch.cos(trigflow_t) + self.sigma_data * torch.sin(trigflow_t))
        c_out = (
            -self.sigma_data * torch.sin(trigflow_t) / (torch.cos(trigflow_t) + self.sigma_data * torch.sin(trigflow_t))
        )
        c_in = self.sigma_data / (torch.cos(trigflow_t) + self.sigma_data * torch.sin(trigflow_t))
        c_noise = (
            self.sigma_data * torch.sin(trigflow_t) / (torch.cos(trigflow_t) + self.sigma_data * torch.sin(trigflow_t))
        )
        return c_skip.to(dtype), c_out.to(dtype), c_in.to(dtype), c_noise.to(dtype)
