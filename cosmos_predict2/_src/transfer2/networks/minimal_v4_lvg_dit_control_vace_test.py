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

import copy

import pytest
import torch

from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.transfer2.configs.vid2vid_transfer.defaults.net import TRANSFER2_CONTROL2WORLD_NET_2B

"""
Usage:
    pytest -s cosmos_predict2/_src/transfer2/networks/minimal_v4_lvg_dit_control_vace_test.py --all -k test_minimal_v1_lvg_edge_dit
"""


@pytest.mark.L1
def test_minimal_v1_lvg_edge_control_vace_dit():
    dtype = torch.bfloat16
    net_config = copy.deepcopy(TRANSFER2_CONTROL2WORLD_NET_2B)
    net_config.num_blocks = 8
    net = instantiate(net_config).cuda().to(dtype=dtype)
    print(net)

    batch_size = 2
    t = 8
    x_B_C_T_H_W = torch.randn(batch_size, 16, t, 40, 40).cuda().to(dtype=dtype)
    noise_labels_B = torch.randn(batch_size).cuda().to(dtype=dtype)
    crossattn_emb_B_T_D = torch.randn(batch_size, 512, 1024).cuda().to(dtype=dtype)
    fps_B = torch.randint(size=(1,), low=2, high=30).cuda().float().repeat(batch_size)
    padding_mask_B_T_H_W = torch.zeros(batch_size, 1, 40, 40).cuda().to(dtype=dtype)
    condition_video_input_mask_B_C_T_H_W = torch.ones(batch_size, 1, t, 40, 40).cuda().to(dtype=dtype)
    latent_control_input_edge = torch.randn(batch_size, 16, t, 40, 40).cuda().to(dtype=dtype)

    output_B = net(
        x_B_C_T_H_W,
        noise_labels_B,
        crossattn_emb_B_T_D,
        latent_control_input=latent_control_input_edge,
        condition_video_input_mask_B_C_T_H_W=condition_video_input_mask_B_C_T_H_W,
        fps=fps_B,
        padding_mask=padding_mask_B_T_H_W,
    )

    loss = output_B.sum()
    print("starting backward pass")
    loss.backward()

    assert output_B.shape == x_B_C_T_H_W.shape
    print("Model test passed.")


if __name__ == "__main__":
    test_minimal_v1_lvg_edge_control_vace_dit()
