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

import math

import pytest
import torch

from cosmos_predict2._src.imaginaire.functional.runge_kutta import phi1, phi2


def _series_reference(t, order, derivative=0):
    # phi_order(t) = sum(t**k / (k + order)!), evaluated independently
    # with enough terms for the test interval [-1, 1].
    return sum(
        math.factorial(k + derivative) / (math.factorial(k) * math.factorial(k + derivative + order)) * t**k
        for k in range(40)
    )


@pytest.mark.parametrize("fn,order", [(phi1, 1), (phi2, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_phi_values_include_the_removable_singularity(fn, order, dtype):
    t = torch.tensor([[0.0, 1e-18, -1e-18, 1e-12, -1e-12], [1e-8, -1e-8, 0.05, -0.05, 0.4]], dtype=dtype)
    expected = _series_reference(t.double(), order).to(dtype)
    result = fn(t)

    assert result.dtype == dtype
    assert result.shape == t.shape
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("fn,order", [(phi1, 1), (phi2, 2)])
@pytest.mark.parametrize("derivative", [1, 2])
def test_phi_derivatives_near_zero(fn, order, derivative):
    t = torch.tensor([0.0, 1e-18, -1e-18, 1e-12, -1e-12, 1e-8, -1e-8], dtype=torch.float64, requires_grad=True)
    result = fn(t)
    for _ in range(derivative):
        result = torch.autograd.grad(result.sum(), t, create_graph=True)[0]
    expected = _series_reference(t.detach(), order, derivative)

    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("fn,order", [(phi1, 1), (phi2, 2)])
def test_phi_values_and_gradients_across_the_series_boundary(fn, order):
    t = torch.tensor([-1.0, -0.100001, -0.099999, 0.099999, 0.100001, 1.0], dtype=torch.float64, requires_grad=True)
    result = fn(t)
    gradient = torch.autograd.grad(result.sum(), t)[0]

    torch.testing.assert_close(result, _series_reference(t.detach(), order), rtol=1e-13, atol=1e-15)
    torch.testing.assert_close(gradient, _series_reference(t.detach(), order, 1), rtol=1e-11, atol=1e-14)


@pytest.mark.parametrize("fn", [phi1, phi2])
def test_phi_gradcheck_at_zero_and_small_arguments(fn):
    t = torch.tensor([0.0, -1e-8, 1e-8, -0.05, 0.05, -0.3, 0.3], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (t,))
    assert torch.autograd.gradgradcheck(fn, (t,))


@pytest.mark.parametrize("fn,order", [(phi1, 1), (phi2, 2)])
def test_phi_inactive_series_does_not_overflow_for_large_negative_arguments(fn, order):
    t = torch.tensor([-1e40], dtype=torch.float64, requires_grad=True)
    result = fn(t)
    gradient = torch.autograd.grad(result.sum(), t)[0]

    assert torch.isfinite(result).all()
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(result, torch.tensor([1e-40], dtype=torch.float64), rtol=1e-12, atol=0)
    torch.testing.assert_close(gradient, torch.tensor([1e-80], dtype=torch.float64), rtol=1e-12, atol=0)
