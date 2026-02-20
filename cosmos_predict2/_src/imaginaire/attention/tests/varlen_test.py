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

SDPA unit tests.
"""

import random
import unittest
from functools import partial

import pytest
import torch

from cosmos_predict2._src.imaginaire.attention import attention as i4_attention
from cosmos_predict2._src.imaginaire.attention.flash3 import FLASH3_SUPPORTED
from cosmos_predict2._src.imaginaire.attention.masks import CausalType
from cosmos_predict2._src.imaginaire.attention.natten import NATTEN_SUPPORTED
from cosmos_predict2._src.imaginaire.attention.utils import is_blackwell_dc, is_fp8, is_hopper
from cosmos_predict2._src.imaginaire.attention.utils import safe_log as log
from cosmos_predict2._src.imaginaire.utils.device import with_torch_device

RAND_SWEEP_TESTS = 1000

skip_if_natten_not_supported = partial(
    pytest.mark.skipif,
    not NATTEN_SUPPORTED,
    reason="NATTEN is disabled, not available, or too old in this environment.",
)

skip_if_flash2_not_supported = partial(
    pytest.mark.skipif,
    True,
    reason="Flash2 varlen is banned.",
)

skip_if_flash3_not_supported = partial(
    pytest.mark.skipif,
    not FLASH3_SUPPORTED,
    reason="Flash3 is disabled, not available, or too old in this environment.",
)

# Tests are only enabled on Hopper and Blackwell DC-class for now.
# Will extend to other arches as we integrate more backends.
skip_if_not_supported = partial(
    pytest.mark.skipif,
    not is_blackwell_dc() and not is_hopper(),
    reason="SDPA tests are only allowed for Hopper and Blackwell DC-class GPUs for now.",
)

skip_if_not_blackwell = partial(
    pytest.mark.skipif, not is_blackwell_dc(), reason="This test is only allowed for Blackwell DC-class GPUs."
)

skip_if_not_hopper = partial(pytest.mark.skipif, not is_hopper(), reason="This test is only allowed for Hopper GPUs.")


def _reset_everything():
    torch.manual_seed(42)
    torch.cuda.empty_cache()


# Computes varlen by breaking up into individual attention calls
def compute_split_reference(
    batch: int,
    heads: int,
    head_dim: int,
    seqlens_Q_list: list[int],
    seqlens_KV_list: list[int],
    is_causal: bool,
    causal_type: CausalType | None,
    backend: str,
    test_backward: bool,
    dtype: torch.dtype = torch.float32,
    heads_kv: int | None = None,
    head_dim_v: int | None = None,
    backend_kwargs: dict | None = None,
):
    heads_kv = heads_kv or heads
    head_dim_v = head_dim_v or head_dim

    assert len(seqlens_Q_list) == len(seqlens_KV_list) == batch

    seqlen_q_total = sum(seqlens_Q_list)
    seqlen_kv_total = sum(seqlens_KV_list)
    dtype_safe = torch.float16
    with torch.no_grad():
        q_ref, k_ref, v_ref, d_out_ref = (
            torch.randn((1, seqlen_q_total, heads, head_dim), device="cuda", dtype=dtype_safe).to(dtype),
            torch.randn(
                (1, seqlen_kv_total, heads_kv, head_dim),
                device="cuda",
                dtype=dtype_safe,
            ).to(dtype),
            torch.randn(
                (1, seqlen_kv_total, heads_kv, head_dim_v),
                device="cuda",
                dtype=dtype_safe,
            ).to(dtype),
            torch.randn((1, seqlen_q_total, heads, head_dim_v), device="cuda", dtype=dtype_safe).to(dtype),
        )
        q, k, v, d_out = (
            q_ref.clone(),
            k_ref.clone(),
            v_ref.clone(),
            d_out_ref.clone(),
        )

    out_list = []
    lse_list = []
    d_q_list = []
    d_k_list = []
    d_v_list = []

    q_start, kv_start = 0, 0
    for b in range(batch):
        seqlen_q = seqlens_Q_list[b]
        seqlen_kv = seqlens_KV_list[b]

        q_ = q_ref[:, q_start : q_start + seqlen_q, :, :].clone()
        k_ = k_ref[:, kv_start : kv_start + seqlen_kv, :, :].clone()
        v_ = v_ref[:, kv_start : kv_start + seqlen_kv, :, :].clone()

        if test_backward:
            q_ = q_.requires_grad_(True)
            k_ = k_.requires_grad_(True)
            v_ = v_.requires_grad_(True)
            d_out_ = d_out_ref[:, q_start : q_start + seqlen_q, :, :].clone().requires_grad_(True)

        out_, lse_ = i4_attention(
            q_,
            k_,
            v_,
            is_causal=is_causal,
            causal_type=causal_type,
            backend=backend,
            backend_kwargs=backend_kwargs,
            return_lse=True,
        )

        if test_backward:
            out_.backward(d_out_)

        with torch.no_grad():
            out_list.append(out_.data.clone().float())
            lse_list.append(lse_.data.clone().float())
            if test_backward:
                assert q_.grad is not None
                assert k_.grad is not None
                assert v_.grad is not None
                d_q_list.append(q_.grad.clone().float())
                d_k_list.append(k_.grad.clone().float())
                d_v_list.append(v_.grad.clone().float())

        q_start += seqlen_q
        kv_start += seqlen_kv

    assert q_start == seqlen_q_total
    assert kv_start == seqlen_kv_total

    out_ref = torch.cat(out_list, dim=1)
    lse_ref = torch.cat(lse_list, dim=1)
    assert out_ref.shape[:3] == q_ref.shape[:3]
    dq_ref = None
    dk_ref = None
    dv_ref = None
    if test_backward:
        dq_ref = torch.cat(d_q_list, dim=1)
        dk_ref = torch.cat(d_k_list, dim=1)
        dv_ref = torch.cat(d_v_list, dim=1)

        assert dq_ref.shape == q_ref.shape
        assert dk_ref.shape == k_ref.shape
        assert dv_ref.shape == v_ref.shape

    return (q, k, v, d_out), (out_ref, lse_ref, dq_ref, dk_ref, dv_ref)


class VarlenTest(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    @with_torch_device(device="cuda")
    def _test_against_manual_varlen(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlens_Q_list: list[int],
        seqlens_KV_list: list[int],
        is_causal: bool,
        causal_type: CausalType | None,
        dtype: torch.dtype,
        atol_fwd: tuple[float, float],
        atol_bwd: tuple[float, float, float] | None,
        backend: str,
        reference_backend: str,
        test_backward: bool,
        heads_kv: int | None = None,
        head_dim_v: int | None = None,
        reference_backend_kwargs: dict | None = None,
        backend_kwargs: dict | None = None,
    ):
        heads_kv = heads_kv or heads
        head_dim_v = head_dim_v or head_dim

        log.debug(
            f"Testing varlen ({backend}) against manual varlen ({reference_backend}): "
            f"{batch=}, {heads=}, {heads_kv=}, {head_dim=}, {head_dim_v=}, "
            f"{seqlens_Q_list=}, {seqlens_KV_list=}, {is_causal=}, {causal_type=}, {dtype=}."
        )

        inputs, reference = compute_split_reference(
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            seqlens_Q_list=seqlens_Q_list,
            seqlens_KV_list=seqlens_KV_list,
            is_causal=is_causal,
            causal_type=causal_type,
            dtype=dtype,
            backend=reference_backend,
            backend_kwargs=reference_backend_kwargs,
            test_backward=test_backward,
        )

        q, k, v, d_out = inputs
        out_ref, lse_ref, dq_ref, dk_ref, dv_ref = reference
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
        d_out = d_out.to(dtype)

        # Run target
        if test_backward:
            q.requires_grad_(test_backward)
            k.requires_grad_(test_backward)
            v.requires_grad_(test_backward)
            d_out.requires_grad_(test_backward)

        seqlens_Q = torch.tensor(seqlens_Q_list, dtype=torch.int32, device=q.device)
        seqlens_KV = torch.tensor(seqlens_KV_list, dtype=torch.int32, device=q.device)

        out_, lse_ = i4_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            causal_type=causal_type,
            backend=backend,
            return_lse=True,
            seqlens_Q=seqlens_Q,
            seqlens_KV=seqlens_KV,
            backend_kwargs=backend_kwargs,
        )
        out = out_.float()
        lse = lse_.float()

        if test_backward:
            dq, dk, dv = None, None, None
            out_.backward(d_out)
            with torch.no_grad():
                dq, dk, dv = (
                    q.grad.clone().float(),
                    k.grad.clone().float(),
                    v.grad.clone().float(),
                )

        atol_out, atol_lse = atol_fwd
        assert out.shape == out_ref.shape

        torch.testing.assert_close(out, out_ref, atol=atol_out, rtol=0)
        torch.testing.assert_close(lse, lse_ref, atol=atol_lse, rtol=0)

        if test_backward:
            assert atol_bwd is not None
            atol_dq, atol_dk, atol_dv = atol_bwd
            torch.testing.assert_close(dq, dq_ref, atol=atol_dq, rtol=0)
            torch.testing.assert_close(dk, dk_ref, atol=atol_dk, rtol=0)
            torch.testing.assert_close(dv, dv_ref, atol=atol_dv, rtol=0)

    def _test_natten_varlen(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlens_Q_list: list[int],
        seqlens_KV_list: list[int],
        is_causal: bool,
        head_dim_v: int | None = None,
        heads_kv: int | None = None,
    ):
        # We're testing against the same backend and same dtype,
        # but with varlen implemented as multiple kernel calls, so
        # error thresholds should be much smaller here.
        # This is therefore only a test of the varlen functionality.
        # Correctness per dtype is expected to be verified in the main
        # fmha tests.
        # dQ still needs a more relaxed threshold because of the non-determinism
        ALLOWED_DTYPES = [
            # dtype, (atol_out, atol_lse), (atol_dq, atol_dk, atol_dv)
            (torch.float16, (1e-6, 1e-6), (1e-2, 1e-6, 1e-6)),
            (torch.bfloat16, (1e-6, 1e-6), (1e-2, 1e-6, 1e-6)),
        ]

        if is_blackwell_dc():
            ALLOWED_DTYPES += [
                (torch.float8_e4m3fn, (1e-6, 1e-6), None),
                (torch.float8_e5m2, (1e-6, 1e-6), None),
            ]

        # NOTE: Hopper FMHA does not support varlen, so natten falls back
        # to cutlass-fmha, which means the reference may target hopper-fmha,
        # while the varlen target is cutlass-fmha, and this will throw off the
        # error limits.
        backend_kwargs = None
        if is_hopper():
            backend_kwargs = {"backend": "cutlass-fmha"}

        for dtype, atol_fwd, atol_bwd in ALLOWED_DTYPES:
            self._test_against_manual_varlen(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                seqlens_Q_list=seqlens_Q_list,
                seqlens_KV_list=seqlens_KV_list,
                is_causal=is_causal,
                causal_type=CausalType.TopLeft,  # Top-left is the only supported mask in natten (for now)
                dtype=dtype,
                atol_fwd=atol_fwd,
                atol_bwd=atol_bwd,
                backend="natten",
                reference_backend="natten",
                backend_kwargs=backend_kwargs,
                reference_backend_kwargs=backend_kwargs,
                test_backward=not is_fp8(dtype),
            )

    def _test_flash2_varlen(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlens_Q_list: list[int],
        seqlens_KV_list: list[int],
        is_causal: bool,
        head_dim_v: int | None = None,
        heads_kv: int | None = None,
    ):
        # Flash2 weirdly has higher error rates than all other backends in varlen.
        # We observe very clear instability in training, so we're banning it for now.
        # Setting deterministic=True doesn't seem to help either
        backend_kwargs = None
        # backend_kwargs = {"deterministic": True}
        ALLOWED_DTYPES = [
            # dtype, (atol_out, atol_lse), (atol_dq, atol_dk, atol_dv)
            (torch.float16, (1e-2, 1e-2), (1e-1, 1e-2, 1e-2)),
            (torch.bfloat16, (1e-1, 1e-2), (1e-1, 1e-1, 1e-1)),
        ]

        for dtype, atol_fwd, atol_bwd in ALLOWED_DTYPES:
            self._test_against_manual_varlen(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                seqlens_Q_list=seqlens_Q_list,
                seqlens_KV_list=seqlens_KV_list,
                is_causal=is_causal,
                causal_type=CausalType.BottomRight,  # Bottom-right is the only supported mask in flash2
                dtype=dtype,
                atol_fwd=atol_fwd,
                atol_bwd=atol_bwd,
                backend="flash2",
                reference_backend="flash2",
                backend_kwargs=backend_kwargs,
                reference_backend_kwargs=backend_kwargs,
                test_backward=True,
            )

    def _test_flash3_varlen(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlens_Q_list: list[int],
        seqlens_KV_list: list[int],
        is_causal: bool,
        head_dim_v: int | None = None,
        heads_kv: int | None = None,
    ):
        # We're testing against the same backend and same dtype,
        # but with varlen implemented as multiple kernel calls, so
        # error thresholds should be much smaller here.
        # This is therefore only a test of the varlen functionality.
        # Correctness per dtype is expected to be verified in the main
        # fmha tests.
        # dQ still needs a more relaxed threshold because of the non-determinism
        ALLOWED_DTYPES = [
            # dtype, (atol_out, atol_lse), (atol_dq, atol_dk, atol_dv)
            (torch.float16, (1e-6, 1e-6), (1e-2, 1e-6, 1e-6)),
            (torch.bfloat16, (1e-6, 1e-6), (1e-2, 1e-6, 1e-6)),
        ]
        test_backward = True

        # GQA/MQA + varlen needs to use deterministic mode if we want to keep our thresholds at 1e-6
        backend_kwargs = {}
        reference_backend_kwargs = {}
        if heads_kv is not None and heads != heads_kv:
            backend_kwargs["deterministic"] = True
            reference_backend_kwargs["deterministic"] = True

        for dtype, atol_fwd, atol_bwd in ALLOWED_DTYPES:
            self._test_against_manual_varlen(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                seqlens_Q_list=seqlens_Q_list,
                seqlens_KV_list=seqlens_KV_list,
                is_causal=is_causal,
                causal_type=CausalType.BottomRight,  # Bottom-right is the only supported mask in flash3
                dtype=dtype,
                atol_fwd=atol_fwd,
                atol_bwd=atol_bwd,
                backend="flash3",
                reference_backend="flash3",
                test_backward=test_backward,
                backend_kwargs=backend_kwargs,
                reference_backend_kwargs=reference_backend_kwargs,
            )

    def _test_varlen(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlens_Q_list: list[int],
        seqlens_KV_list: list[int],
        is_causal: bool,
        backend: str,
        head_dim_v: int | None = None,
        heads_kv: int | None = None,
    ):
        if backend == "natten":
            self._test_natten_varlen(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                seqlens_Q_list=seqlens_Q_list,
                seqlens_KV_list=seqlens_KV_list,
                is_causal=is_causal,
            )
        elif backend == "flash2":
            self._test_flash2_varlen(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                seqlens_Q_list=seqlens_Q_list,
                seqlens_KV_list=seqlens_KV_list,
                is_causal=is_causal,
            )
        elif backend == "flash3":
            self._test_flash3_varlen(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                seqlens_Q_list=seqlens_Q_list,
                seqlens_KV_list=seqlens_KV_list,
                is_causal=is_causal,
            )
        else:
            raise NotImplementedError()

    def _test_varlen_randsweep(self, backend: str, max_tests: int = 1000):
        random.seed(42)

        max_seqlen = 2**17
        for i in range(max_tests):
            batch = random.choice(range(1, 12))

            supports_gqa_mqa = False
            if backend == "natten":
                head_dim_choices = [32, 64, 128]
                heads_choices = range(1, 8 + 1)
                # GQA/MQA is only supported in NATTEN's Blackwell FMHA backend for now
                supports_gqa_mqa = is_blackwell_dc()
            elif backend in ["flash2", "flash3"]:
                head_dim_choices = range(16, 256 + 1, 8)
                heads_choices = range(1, 8 + 1)
                supports_gqa_mqa = True
            else:
                raise NotImplementedError()

            heads = random.choice(heads_choices)
            heads_kv = (
                heads
                if not supports_gqa_mqa
                else random.choice([1] + [i for i in range(1, heads + 1) if heads % i == 0])
            )
            assert heads >= heads_kv and heads % heads_kv == 0

            head_dim = random.choice(head_dim_choices)
            head_dim_v = None

            # Flash3 backward head dim 256 doesn't support deterministic mode
            if backend == "flash3" and head_dim > 128 and heads != heads_kv:
                heads_kv = heads

            seqlens_Q_list = []
            seqlens_KV_list = []
            for i in range(batch):
                max_q = min(2**12, max(max_seqlen - sum(seqlens_Q_list), 24))
                max_k = min(2**12, max(max_seqlen - sum(seqlens_KV_list), 24))
                new_q = random.choice(range(8, max_q, 1))
                new_k = random.choice(range(8, max_k, 1))
                seqlens_Q_list.append(new_q)
                seqlens_KV_list.append(new_k)

            for is_causal in [False, True]:
                self._test_varlen(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    seqlens_Q_list=seqlens_Q_list,
                    seqlens_KV_list=seqlens_KV_list,
                    is_causal=is_causal,
                    backend=backend,
                )

    @pytest.mark.L0
    @skip_if_natten_not_supported()
    @skip_if_not_supported()
    def test_natten_varlen_fast(self):
        problem_sizes = [
            (
                9,
                4,
                128,
                [2669, 2240, 910, 2421, 3323, 34, 3308, 2867, 1401],
                [2880, 1726, 1847, 1147, 3568, 3116, 661, 1739, 1146],
            ),
            (6, 1, 128, [128, 128, 135, 121, 128, 128], [128, 128, 135, 121, 128, 128]),
            (5, 1, 128, [128, 128, 135, 128, 128], [128, 128, 135, 128, 128]),
            (2, 1, 128, [135, 200], [128, 768]),
            (2, 1, 128, [1024, 200], [128, 768]),
            (2, 1, 128, [135, 200], [135, 768]),
            (2, 1, 128, [1024, 200], [135, 768]),
            (2, 1, 128, [1024, 256], [128, 768]),
            (4, 1, 128, [1024, 8, 17, 2048], [10, 20, 512, 16]),
            (3, 2, 128, [268, 1584, 1571], [2448, 4088, 1925]),
            (2, 1, 128, [1024, 256], [512, 768]),
        ]
        for (
            batch,
            heads,
            head_dim,
            seqlens_Q_list,
            seqlens_KV_list,
        ) in problem_sizes:
            for is_causal in [False, True]:
                self._test_varlen(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    seqlens_Q_list=seqlens_Q_list,
                    seqlens_KV_list=seqlens_KV_list,
                    is_causal=is_causal,
                    backend="natten",
                )

    @pytest.mark.L1
    @skip_if_natten_not_supported()
    @skip_if_not_supported()
    def test_natten_varlen_randsweep(self):
        self._test_varlen_randsweep(backend="natten", max_tests=RAND_SWEEP_TESTS)

    @pytest.mark.L0
    @skip_if_flash2_not_supported()
    @skip_if_not_supported()
    def test_flash2_varlen_fast(self):
        problem_sizes = [
            (
                9,
                4,
                128,
                [2669, 2240, 910, 2421, 3323, 34, 3308, 2867, 1401],
                [2880, 1726, 1847, 1147, 3568, 3116, 661, 1739, 1146],
            ),
            (6, 1, 128, [128, 128, 135, 121, 128, 128], [128, 128, 135, 121, 128, 128]),
            (5, 1, 128, [128, 128, 135, 128, 128], [128, 128, 135, 128, 128]),
            (2, 1, 128, [135, 200], [128, 768]),
            (2, 1, 128, [1024, 200], [128, 768]),
            (2, 1, 128, [135, 200], [135, 768]),
            (2, 1, 128, [1024, 200], [135, 768]),
            (2, 1, 128, [1024, 256], [128, 768]),
            (4, 1, 128, [1024, 8, 17, 2048], [10, 20, 512, 16]),
            (3, 2, 128, [268, 1584, 1571], [2448, 4088, 1925]),
            (2, 1, 128, [1024, 256], [512, 768]),
        ]
        for (
            batch,
            heads,
            head_dim,
            seqlens_Q_list,
            seqlens_KV_list,
        ) in problem_sizes:
            for is_causal in [False, True]:
                self._test_varlen(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    seqlens_Q_list=seqlens_Q_list,
                    seqlens_KV_list=seqlens_KV_list,
                    is_causal=is_causal,
                    backend="flash2",
                )

    @pytest.mark.L1
    @skip_if_flash2_not_supported()
    @skip_if_not_supported()
    def test_flash2_varlen_randsweep(self):
        self._test_varlen_randsweep(backend="flash2", max_tests=RAND_SWEEP_TESTS)

    @pytest.mark.L0
    @skip_if_flash3_not_supported()
    @skip_if_not_hopper()
    def test_flash3_varlen_fast(self):
        problem_sizes = [
            (
                9,
                4,
                128,
                [2669, 2240, 910, 2421, 3323, 34, 3308, 2867, 1401],
                [2880, 1726, 1847, 1147, 3568, 3116, 661, 1739, 1146],
            ),
            (6, 1, 128, [128, 128, 135, 121, 128, 128], [128, 128, 135, 121, 128, 128]),
            (5, 1, 128, [128, 128, 135, 128, 128], [128, 128, 135, 128, 128]),
            (2, 1, 128, [135, 200], [128, 768]),
            (2, 1, 128, [1024, 200], [128, 768]),
            (2, 1, 128, [135, 200], [135, 768]),
            (2, 1, 128, [1024, 200], [135, 768]),
            (2, 1, 128, [1024, 256], [128, 768]),
            (4, 1, 128, [1024, 8, 17, 2048], [10, 20, 512, 16]),
            (3, 2, 128, [268, 1584, 1571], [2448, 4088, 1925]),
            (2, 1, 128, [1024, 256], [512, 768]),
        ]
        for (
            batch,
            heads,
            head_dim,
            seqlens_Q_list,
            seqlens_KV_list,
        ) in problem_sizes:
            for is_causal in [False, True]:
                self._test_varlen(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    seqlens_Q_list=seqlens_Q_list,
                    seqlens_KV_list=seqlens_KV_list,
                    is_causal=is_causal,
                    backend="flash3",
                )

    @pytest.mark.L1
    @skip_if_flash3_not_supported()
    @skip_if_not_hopper()
    def test_flash3_varlen_randsweep(self):
        self._test_varlen_randsweep(backend="flash3", max_tests=RAND_SWEEP_TESTS)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    unittest.main()
