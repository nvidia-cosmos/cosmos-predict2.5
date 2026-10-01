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

import pytest
import torch

from cosmos_predict2._src.imaginaire.functional.runge_kutta import get_runge_kutta_fn


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_heun_uses_slopes_at_the_initial_and_predicted_states(dtype):
    x_s = torch.linspace(-2, 3, 48, dtype=dtype).reshape(3, 2, 2, 4)
    s = torch.tensor([1.0, 0.8, 2.0], dtype=dtype)
    t = torch.tensor([0.7, 1.1, 0.4], dtype=dtype)
    rate = torch.tensor([0.7, -0.4, 0.2], dtype=dtype).reshape(3, 1, 1, 1)
    forcing = torch.linspace(-0.2, 0.3, 16, dtype=dtype).reshape(1, 2, 2, 4)
    calls = []

    def slope(x, sigma):
        return rate * x + sigma.reshape(3, 1, 1, 1).square() * forcing

    def denoise(x, sigma):
        calls.append((x.clone(), sigma.clone()))
        return x - sigma.reshape(3, 1, 1, 1) * slope(x, sigma)

    dt = (t - s).reshape(3, 1, 1, 1)
    first_slope = slope(x_s, s)
    predictor = x_s + dt * first_slope
    average_slope = (first_slope + slope(predictor, t)) / 2
    expected = x_s + dt * average_slope
    expected_x0 = x_s - s.reshape(3, 1, 1, 1) * average_slope

    result, x0 = get_runge_kutta_fn("2heun_naive")(x_s, s, t, denoise)

    assert len(calls) == 2
    torch.testing.assert_close(calls[0][0], x_s)
    torch.testing.assert_close(calls[0][1], s)
    torch.testing.assert_close(calls[1][0], predictor)
    torch.testing.assert_close(calls[1][1], t)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(x0, expected_x0)
    assert result.dtype == dtype
    assert result.shape == x_s.shape


def test_heun_has_second_order_global_convergence():
    # x0(x, sigma) = (1 - rate * sigma) * x makes the diffusion ODE
    # dx/dsigma = (x - x0) / sigma exactly dx/dsigma = rate * x.
    rate = 0.7
    initial = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    exact = initial * torch.exp(torch.tensor(-0.8 * rate, dtype=torch.float64))
    errors = []

    def denoise(x, sigma):
        return (1 - rate * sigma[:, None]) * x

    for num_steps in (16, 32, 64):
        sigmas = torch.linspace(1.0, 0.2, num_steps + 1, dtype=torch.float64)
        x = initial.clone()
        for s, t in zip(sigmas[:-1], sigmas[1:]):
            x, _ = get_runge_kutta_fn("2heun_naive")(x, s[None], t[None], denoise)
        errors.append((x - exact).abs().max().item())

    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert 3.8 < coarse / fine < 4.2


def test_heun_gradients_match_the_second_order_update():
    rate = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    x_s = torch.tensor([[1.0, -2.0], [0.3, 0.9]], dtype=torch.float64, requires_grad=True)
    s = torch.tensor([1.0, 0.9], dtype=torch.float64)
    t = torch.tensor([0.8, 0.4], dtype=torch.float64)

    def denoise(x, sigma):
        return (1 - rate * sigma[:, None]) * x

    result, _ = get_runge_kutta_fn("2heun_naive")(x_s, s, t, denoise)
    grad_x, grad_rate = torch.autograd.grad(result.sum(), (x_s, rate))
    dt = (t - s)[:, None]
    expected_grad_x = (1 + rate * dt + 0.5 * rate.square() * dt.square()).expand_as(x_s)
    expected_grad_rate = (x_s * (dt + rate * dt.square())).sum()

    torch.testing.assert_close(grad_x, expected_grad_x)
    torch.testing.assert_close(grad_rate, expected_grad_rate)
