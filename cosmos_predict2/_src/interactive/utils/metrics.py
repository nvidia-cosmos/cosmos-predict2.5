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

import torch


def cos_similarity(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x1_flat = x1.flatten(start_dim=1)
    x2_flat = x2.flatten(start_dim=1)

    dot_product = torch.sum(x1_flat * x2_flat, dim=1)

    norm1 = torch.linalg.norm(x1_flat, ord=2, dim=1)
    norm2 = torch.linalg.norm(x2_flat, ord=2, dim=1)

    similarity = dot_product / (norm1 * norm2 + eps)

    return similarity
