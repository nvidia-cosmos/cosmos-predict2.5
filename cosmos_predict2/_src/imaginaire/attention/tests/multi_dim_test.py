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
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Multi-Dimensional Attention unit tests.
"""

import math
import random
import unittest
from functools import partial
from itertools import product
from typing import Callable

import pytest
import torch
from torch import Tensor

from cosmos_predict2._src.imaginaire.attention import multi_dimensional_attention
from cosmos_predict2._src.imaginaire.attention.natten import NATTEN_SUPPORTED
from cosmos_predict2._src.imaginaire.attention.utils import is_blackwell_dc, is_fp8
from cosmos_predict2._src.imaginaire.attention.utils import safe_log as log
from cosmos_predict2._src.imaginaire.utils.device import with_torch_device

RAND_SWEEP_TESTS = 1000

skip_if_natten_not_supported = partial(
    pytest.mark.skipif,
    not NATTEN_SUPPORTED,
    reason="NATTEN is disabled, not available, or too old in this environment.",
)


def _reset_everything():
    torch.manual_seed(42)
    torch.cuda.empty_cache()


class MultiDimTester:
    def __init__(
        self,
        reference_fn: Callable,
        batch: int,
        heads: int,
        token_layout_shape: tuple,
        head_dim: int,
        window_size: tuple,
        stride: tuple,
        dilation: tuple,
        is_causal: tuple,
        test_backward: bool = True,
        scale: float | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
        heads_kv: int | None = None,
        head_dim_v: int | None = None,
    ):
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv or heads
        self.token_layout_shape = token_layout_shape
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v or head_dim
        self.test_backward = test_backward
        self.scale = scale if scale is not None else head_dim**-0.5
        self.dtype = dtype
        self.device = device

        self.window_size = window_size
        self.stride = stride
        self.dilation = dilation
        self.is_causal = is_causal

        # Initialize input tensors
        self.q = torch.randn(
            self.batch,
            *self.token_layout_shape,
            self.heads,
            self.head_dim,
            dtype=dtype,
            device=device,
            requires_grad=test_backward,
        )
        self.k = torch.randn(
            self.batch,
            *self.token_layout_shape,
            self.heads_kv,
            self.head_dim,
            dtype=dtype,
            device=device,
            requires_grad=test_backward,
        )
        self.v = torch.randn(
            self.batch,
            *self.token_layout_shape,
            self.heads_kv,
            self.head_dim_v,
            dtype=dtype,
            device=device,
            requires_grad=test_backward,
        )
        self.d_output = (
            torch.randn(self.batch, *self.token_layout_shape, self.heads, self.head_dim_v, dtype=dtype, device=device)
            if test_backward
            else None
        )

        # Run reference implementation
        q_ref = self.q.clone().detach().requires_grad_(self.test_backward)
        k_ref = self.k.clone().detach().requires_grad_(self.test_backward)
        v_ref = self.v.clone().detach().requires_grad_(self.test_backward)

        output_ref, lse_ref = reference_fn(
            query=q_ref,
            key=k_ref,
            value=v_ref,
            scale=self.scale,
            window_size=self.window_size,
            stride=self.stride,
            dilation=self.dilation,
            is_causal=self.is_causal,
            return_lse=True,
        )

        self.output_ref = output_ref.detach().to(torch.float32)
        self.lse_ref = lse_ref.detach().to(torch.float32)

        # Reference backward pass
        if self.test_backward:
            d_output = self.d_output.clone().detach()
            output_ref.backward(d_output)
            self.dq_ref = q_ref.grad.detach().to(torch.float32)
            self.dk_ref = k_ref.grad.detach().to(torch.float32)
            self.dv_ref = v_ref.grad.detach().to(torch.float32)

    def test(
        self,
        target_fn: Callable,
        dtype: torch.dtype,
        atol_fwd: float,
        atol_bwd: tuple[float, float, float] | None = None,
        rtol_fwd: float = 0.0,
        rtol_bwd: float = 0.0,
        test_backward: bool | None = None,
    ):
        test_backward = self.test_backward if test_backward is None else test_backward

        q = self.q.clone().detach().to(dtype).requires_grad_(test_backward)
        k = self.k.clone().detach().to(dtype).requires_grad_(test_backward)
        v = self.v.clone().detach().to(dtype).requires_grad_(test_backward)

        output, lse = target_fn(
            query=q,
            key=k,
            value=v,
            scale=self.scale,
            window_size=self.window_size,
            stride=self.stride,
            dilation=self.dilation,
            is_causal=self.is_causal,
            return_lse=True,
        )

        torch.testing.assert_close(output.to(torch.float32), self.output_ref, atol=atol_fwd, rtol=rtol_fwd)
        torch.testing.assert_close(lse.to(torch.float32), self.lse_ref, atol=atol_fwd, rtol=rtol_fwd)

        # Backward pass
        if test_backward:
            assert atol_bwd is not None
            assert rtol_bwd is not None
            atol_dq, atol_dk, atol_dv = atol_bwd

            d_output = self.d_output.clone().detach().to(dtype)
            output.backward(d_output)

            dq = q.grad.detach().to(torch.float32)
            dk = k.grad.detach().to(torch.float32)
            dv = v.grad.detach().to(torch.float32)

            torch.testing.assert_close(dq, self.dq_ref, atol=atol_dq, rtol=rtol_bwd)
            torch.testing.assert_close(dk, self.dk_ref, atol=atol_dk, rtol=rtol_bwd)
            torch.testing.assert_close(dv, self.dv_ref, atol=atol_dv, rtol=rtol_bwd)


def idx2crd(index, shape) -> tuple:
    rank = len(shape)
    coord = []
    residual = index
    for i in range(rank - 1, -1, -1):
        coord.append(residual % shape[i])
        residual = residual // shape[i]

    # assert residual == 0
    return tuple(coord[::-1])


def multi_dim_mask(
    q_idx: int,
    kv_idx: int,
    token_layout_shape: tuple,
    window_size: tuple,
    stride: tuple,
    dilation: tuple,
    is_causal: tuple,
) -> bool:
    assert len(token_layout_shape) == len(window_size) == len(stride) == len(dilation) == len(is_causal)

    # Reconstruct global Q and KV coordinates
    q_crd = idx2crd(q_idx, token_layout_shape)
    kv_crd = idx2crd(kv_idx, token_layout_shape)

    masks = []
    for q, kv, x, w, s, d, c in zip(q_crd, kv_crd, token_layout_shape, window_size, stride, dilation, is_causal):
        # Coordinates within dilation group
        q_crd_di = q // d
        kv_crd_di = kv // d

        # Dilation group coordinates
        q_dilation_group_crd = q % d
        kv_dilation_group_crd = kv % d

        # Fixup input shape according to dilation group
        dilation_group_padding = 1 - ((q_dilation_group_crd + (d - (x % d))) // d)
        qkv_shape_corrected = (x // d) + dilation_group_padding

        if c:
            # Leader is the last (right-most) query in the stride group.
            stride_group_leader = min(
                (q_crd_di // s) * s + s - 1,
                qkv_shape_corrected - 1,
            )

            if not (
                (q_crd_di - kv_crd_di >= 0)  # window still ends at query index
                and (stride_group_leader - kv_crd_di < w)
                and (q_dilation_group_crd == kv_dilation_group_crd)
            ):
                return False

        else:
            # Window size left and right (non-causal only)
            window_size_left = w // 2
            window_size_right = w // 2 + (w % 2 - 1)

            # Leader is the center-most query in the stride group.
            # If stride is even, choose the right hand side center query.
            stride_group_leader = min(
                (q_crd_di // s) * s + (s // 2),
                qkv_shape_corrected - 1,
            )

            window_center = min(max(stride_group_leader, window_size_left), qkv_shape_corrected - 1 - window_size_right)
            w0 = window_center - kv_crd_di
            w1 = kv_crd_di - window_center
            if not (
                (((0 <= w0) and (w0 <= window_size_left)) or ((0 <= w1) and (w1 <= window_size_right)))
                and (q_dilation_group_crd == kv_dilation_group_crd)
            ):
                return False

    return True


def multi_dim_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    window_size: tuple,
    stride: tuple,
    dilation: tuple,
    is_causal: tuple,
    scale: float,
    return_lse: bool = False,
):
    assert query.dim() in [4, 5, 6]
    B, *token_layout_shape, H, D = query.shape
    H_kv, _ = key.shape[-2:]
    D_v = value.shape[-1]
    seqlen = math.prod(token_layout_shape)

    # cast from torch shape to tuple
    token_layout_shape = tuple(x for x in token_layout_shape)

    num_dims = len(token_layout_shape)

    assert H % H_kv == 0
    h_k = H // H_kv

    query_t = query.flatten(1, num_dims).transpose(1, 2)
    key_t = key.flatten(1, num_dims).transpose(1, 2)
    value_t = value.flatten(1, num_dims).transpose(1, 2)

    assert query_t.dim() == key_t.dim() == value_t.dim() == 4
    assert query_t.shape[2] == key_t.shape[2] == value_t.shape[2] == seqlen

    # Decomposed GQA/MQA implementation
    if h_k > 1:
        key_t = torch.repeat_interleave(key_t, repeats=h_k, dim=1, output_size=H)
        value_t = torch.repeat_interleave(value_t, repeats=h_k, dim=1, output_size=H)

    attn_scores = torch.matmul(query_t, key_t.transpose(-2, -1)) * scale

    mask = torch.zeros((seqlen, seqlen), dtype=torch.bool)
    is_valid = partial(
        multi_dim_mask,
        token_layout_shape=token_layout_shape,
        window_size=window_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
    )
    for q, kv in product(range(mask.shape[0]), range(mask.shape[1])):
        mask[q, kv] = not is_valid(q, kv)

    mask_cu = mask.unsqueeze(0).unsqueeze(0).to(attn_scores.device)
    attn_scores = attn_scores.masked_fill(mask_cu, float("-inf"))

    # Compute logsumexp (LSE) before softmax
    lse = torch.logsumexp(attn_scores, dim=-1)  # Shape: (B, H, seqlen)
    lse = lse.transpose(1, 2)  # Shape: (B, seqlen, H)
    lse = lse.reshape(B, *token_layout_shape, H)  # Shape: (B, *token_layout_shape, H)

    attn_weights = attn_scores.softmax(dim=-1)

    out = torch.matmul(attn_weights, value_t)

    out = out.transpose(1, 2)

    out = out.reshape(B, *token_layout_shape, H, D_v)

    if return_lse:
        return out, lse
    return out


class MultiDimTest(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    @with_torch_device(device="cuda")
    def _test_against_bmm_reference(
        self,
        batch: int,
        heads: int,
        token_layout_shape: tuple,
        head_dim: int,
        window_size: tuple,
        stride: tuple,
        dilation: tuple,
        is_causal: tuple,
        test_backward: bool,
        backend: str,
        scale: float | None = None,
        heads_kv: int | None = None,
        head_dim_v: int | None = None,
    ):
        reference_dtype = torch.float16
        device = "cuda"
        attention_fn = partial(multi_dimensional_attention, backend=backend)

        log.debug(
            "Running reference Multi-Dimensional Attention on: "
            f"{batch=}, {heads=}, {heads_kv=}, {head_dim=}, {head_dim_v=}, "
            f"{token_layout_shape=}, {window_size=}, {stride=}, {dilation=}, {is_causal=}."
        )
        tester = MultiDimTester(
            reference_fn=multi_dim_reference,
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            token_layout_shape=token_layout_shape,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            window_size=window_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            dtype=reference_dtype,
            test_backward=test_backward,
            scale=scale,
            device=device,
        )

        ALLOWED_DTYPES = [
            # dtype, atol_out, (atol_dq, atol_dk, atol_dv), rtol_fwd, rtol_bwd
            (torch.float16, 1e-2, (4e-2, 4e-2, 4e-2), 0, 0),
            (torch.bfloat16, 1e-1, (2e-1, 2e-1, 2e-1), 0, 0),
        ]
        if backend == "natten" and is_blackwell_dc():
            ALLOWED_DTYPES += [
                (torch.float8_e4m3fn, 4e-1, None, 1e-1, 0),
                (torch.float8_e5m2, 8e-1, None, 5e-1, 0),
            ]

        for dtype, atol_fwd, atol_bwd, rtol_fwd, rtol_bwd in ALLOWED_DTYPES:
            test_backward_ = test_backward and not is_fp8(dtype)
            log.debug(
                f"Testing Multi-Dimensional Attention ({backend}): {batch=}, {heads=}, {heads_kv=}, {head_dim=}, {head_dim_v=}, "
                f"{token_layout_shape=}, {window_size=}, {stride=}, {dilation=}, "
                f"{is_causal=}, {dtype=}, {test_backward_=}."
            )
            tester.test(
                target_fn=attention_fn,
                dtype=dtype,
                atol_fwd=atol_fwd,
                atol_bwd=atol_bwd,
                rtol_fwd=rtol_fwd,
                rtol_bwd=rtol_bwd,
                test_backward=test_backward_,
            )

    def _test_randsweep(self, num_dims: int, backend: str, max_tests: int = 1000, max_seqlen: int = 2**17):
        random.seed(42)

        for i in range(max_tests):
            batch = random.choice(range(1, 2))

            supports_mla = False
            supports_gqa_mqa = False
            if backend == "natten":
                head_dim_choices = [32, 64, 128]
                heads_choices = range(1, 4 + 1)
                # GQA/MQA is not supported in FNA ops yet
                supports_gqa_mqa = False

                # Enable MLA when supported in hopper or blackwell
                head_dim = random.choice(head_dim_choices)
                head_dim_v = None
                # head_dim_v = random.choice(head_dim_choices)

            else:
                raise NotImplementedError()

            heads = random.choice(heads_choices)
            heads_kv = (
                heads
                if not supports_gqa_mqa
                else random.choice([1] + [i for i in range(1, heads + 1) if heads % i == 0])
            )
            assert heads >= heads_kv and heads % heads_kv == 0

            token_layout_shape = []
            for j in range(num_dims):
                max_size = (
                    min(max_seqlen, 16384)
                    if j == 0
                    else min(16384, max(10, max_seqlen - math.prod(token_layout_shape)))
                )
                token_layout_shape.append(random.choice(range(4, max_size)))

            while math.prod(token_layout_shape) > max_seqlen:
                dim_to_cut = random.choice(range(num_dims))
                token_layout_shape[dim_to_cut] = max(4, int(token_layout_shape[dim_to_cut] * 0.1))

            token_layout_shape = tuple(token_layout_shape)
            window_size = tuple(random.choice(range(2, x + 1)) for x in token_layout_shape)
            stride = tuple(random.choice(range(1, k + 1)) for k in window_size)
            dilation = tuple(random.choice(range(1, x // k + 1)) for x, k in zip(token_layout_shape, window_size))
            is_causal = tuple(random.choice([False, True]) for _ in range(num_dims))

            self._test_against_bmm_reference(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                token_layout_shape=token_layout_shape,
                window_size=window_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                backend=backend,
                test_backward=True,
            )

    @pytest.mark.L0
    @skip_if_natten_not_supported()
    def test_natten_fast(self):
        random.seed(83)
        torch.manual_seed(83)
        self._test_randsweep(num_dims=1, backend="natten", max_tests=2, max_seqlen=2**9)
        self._test_randsweep(num_dims=2, backend="natten", max_tests=2, max_seqlen=2**9)
        self._test_randsweep(num_dims=3, backend="natten", max_tests=2, max_seqlen=2**9)

    @pytest.mark.L1
    @pytest.mark.skip("Extended rand sweep is disabled until we have a faster reference for multi-dim")
    @skip_if_natten_not_supported()
    def test_natten_randsweep(self):
        random.seed(84)
        torch.manual_seed(84)
        self._test_randsweep(num_dims=1, backend="natten", max_tests=RAND_SWEEP_TESTS // 3, max_seqlen=2**11)
        self._test_randsweep(num_dims=2, backend="natten", max_tests=RAND_SWEEP_TESTS // 3, max_seqlen=2**11)
        self._test_randsweep(num_dims=3, backend="natten", max_tests=RAND_SWEEP_TESTS // 3, max_seqlen=2**11)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    unittest.main()
