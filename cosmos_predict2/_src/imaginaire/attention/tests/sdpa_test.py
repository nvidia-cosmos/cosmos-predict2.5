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
from typing import Callable

import pytest
import torch
from torch import Tensor

from cosmos_predict2._src.imaginaire.attention import attention as i4_attention
from cosmos_predict2._src.imaginaire.attention.flash2 import FLASH2_SUPPORTED
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
    not FLASH2_SUPPORTED,
    reason="Flash2 is disabled, not available, or too old in this environment.",
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


class SdpaTester:
    def __init__(
        self,
        reference_fn: Callable,
        batch: int,
        heads: int,
        seqlen_q: int,
        seqlen_kv: int,
        head_dim: int,
        test_backward: bool = True,
        scale: float | None = None,
        is_causal: bool = False,
        causal_type: CausalType | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
        heads_kv: int | None = None,
        head_dim_v: int | None = None,
    ):
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv or heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v or head_dim
        self.test_backward = test_backward
        self.scale = scale if scale is not None else head_dim**-0.5
        self.is_causal = is_causal
        self.causal_type = causal_type
        self.dtype = dtype
        self.device = device

        # Initialize input tensors
        self.q = torch.randn(
            self.batch,
            self.seqlen_q,
            self.heads,
            self.head_dim,
            dtype=dtype,
            device=device,
            requires_grad=test_backward,
        )
        self.k = torch.randn(
            self.batch,
            self.seqlen_kv,
            self.heads_kv,
            self.head_dim,
            dtype=dtype,
            device=device,
            requires_grad=test_backward,
        )
        self.v = torch.randn(
            self.batch,
            self.seqlen_kv,
            self.heads_kv,
            self.head_dim_v,
            dtype=dtype,
            device=device,
            requires_grad=test_backward,
        )
        self.d_output = (
            torch.randn(self.batch, self.seqlen_q, self.heads, self.head_dim_v, dtype=dtype, device=device)
            if test_backward
            else None
        )

        # Run reference implementation
        q_ref = self.q.clone().detach().requires_grad_(self.test_backward)
        k_ref = self.k.clone().detach().requires_grad_(self.test_backward)
        v_ref = self.v.clone().detach().requires_grad_(self.test_backward)

        output_ref = reference_fn(
            query=q_ref,
            key=k_ref,
            value=v_ref,
            scale=self.scale,
            is_causal=self.is_causal,
            causal_type=self.causal_type,
        )

        self.output_ref = output_ref.detach().to(torch.float32)

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

        output = target_fn(
            query=q, key=k, value=v, scale=self.scale, is_causal=self.is_causal, causal_type=self.causal_type
        )

        torch.testing.assert_close(output.to(torch.float32), self.output_ref, atol=atol_fwd, rtol=rtol_fwd)

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


def torch_sdpa_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
    is_causal: bool,
    causal_type: CausalType,
):
    heads = query.shape[2]
    heads_kv = key.shape[2]
    assert heads % heads_kv == 0
    h_k = heads // heads_kv

    assert not is_causal or causal_type == CausalType.TopLeft, "Torch SDPA only supports top-left causal mask."

    # Torch requires heads-first layout
    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()

    k_final, v_final = key, value
    # Decomposed GQA/MQA implementation for torch SDPA via explicit repeats
    if h_k > 1:
        k_final = torch.repeat_interleave(key, repeats=h_k, dim=1, output_size=heads)
        v_final = torch.repeat_interleave(value, repeats=h_k, dim=1, output_size=heads)

        assert k_final.shape[:2] == query.shape[:2]
        assert v_final.shape[:2] == query.shape[:2]
        assert k_final.shape[-1] == query.shape[-1]
        assert v_final.shape[-1] == query.shape[-1]

    with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
        out = torch.nn.functional.scaled_dot_product_attention(
            query, k_final, v_final, is_causal=is_causal, scale=scale
        )

    out = out.permute(0, 2, 1, 3).contiguous()
    return out


# NOTE: Use ONLY when seqlen_{q,kv} are small!
# Supports MLA and bottom-right causal mask, unlike SDPA
def bmm_sdpa_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
    is_causal: bool,
    causal_type: CausalType,
    MAX_QK: int = 16384**2,
):
    B, S_q, H, D = query.shape
    _, S_kv, H_kv, _ = key.shape

    assert H % H_kv == 0
    h_k = H // H_kv

    if S_q * S_kv > MAX_QK:
        raise ValueError(f"Query-key matmul too large: {S_q}*{S_kv} > MAX_QK={MAX_QK}")

    query_t = query.transpose(1, 2)
    key_t = key.transpose(1, 2)
    value_t = value.transpose(1, 2)

    # Decomposed GQA/MQA implementation
    if h_k > 1:
        key_t = torch.repeat_interleave(key_t, repeats=h_k, dim=1, output_size=H)
        value_t = torch.repeat_interleave(value_t, repeats=h_k, dim=1, output_size=H)

    attn_scores = torch.matmul(query_t, key_t.transpose(-2, -1)) * scale

    if is_causal:
        if causal_type == CausalType.TopLeft:
            diagonal_offset = 1
        elif causal_type == CausalType.BottomRight:
            diagonal_offset = S_kv - S_q + 1
        else:
            raise NotImplementedError()
        mask = torch.triu(torch.ones(S_q, S_kv, device=query.device, dtype=torch.bool), diagonal=diagonal_offset)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

    attn_weights = attn_scores.softmax(dim=-1)

    # We can have entirely masked rows (queries) with this mask
    if causal_type == CausalType.BottomRight:
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    out = torch.matmul(attn_weights, value_t)

    out = out.transpose(1, 2)

    return out


class SdpaTest(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    @with_torch_device(device="cuda")
    def _test_against_torch_sdpa(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlen_q: int,
        seqlen_kv: int,
        is_causal: bool,
        causal_type: CausalType,
        test_backward: bool,
        backend: str,
        scale: float | None = None,
        heads_kv: int | None = None,
        head_dim_v: int | None = None,
    ):
        reference_dtype = torch.float16
        device = "cuda"
        attention_fn = partial(i4_attention, backend=backend)

        reference_fn = torch_sdpa_reference
        if (is_causal and causal_type == CausalType.BottomRight) or (head_dim_v is not None and head_dim_v != head_dim):
            reference_fn = bmm_sdpa_reference

        tester = SdpaTester(
            reference_fn=reference_fn,
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            dtype=reference_dtype,
            test_backward=test_backward,
            scale=scale,
            is_causal=is_causal,
            causal_type=causal_type,
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
                f"Testing SDPA ({backend}) vs torch SDPA: {batch=}, {heads=}, {heads_kv=}, {head_dim=}, {head_dim_v=}, "
                f"{seqlen_q=}, {seqlen_kv=}, {is_causal=}, {causal_type=}, {dtype=}, {test_backward_=}"
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

    def _test_backend_against_torch_sdpa(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlen_q: int,
        seqlen_kv: int,
        is_causal: bool,
        backend: str,
        scale: float | None = None,
        heads_kv: int | None = None,
        head_dim_v: int | None = None,
    ):
        assert backend in [
            "natten",
            "flash2",
            "flash3",
        ]
        test_backward = True
        self._test_against_torch_sdpa(
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            is_causal=is_causal,
            causal_type=CausalType.TopLeft if backend not in ["flash2", "flash3"] else CausalType.BottomRight,
            scale=scale,
            test_backward=test_backward,
            backend=backend,
        )

    def _test_randsweep_against_torch_sdpa(self, backend: str, max_tests: int = 1000):
        random.seed(42)

        max_qk = 2**21
        for i in range(max_tests):
            batch = random.choice(range(1, 4))

            supports_mla = False
            supports_gqa_mqa = False
            if backend == "natten":
                head_dim_choices = [32, 64, 128]
                heads_choices = range(1, 8 + 1)
                # GQA/MQA is only supported in NATTEN's Blackwell FMHA backend for now
                supports_gqa_mqa = is_blackwell_dc()

                # Enable MLA when supported in hopper or blackwell
                head_dim = random.choice(head_dim_choices)
                head_dim_v = None
                # head_dim_v = random.choice(head_dim_choices)

            elif backend in ["flash2", "flash3"]:
                head_dim_choices = range(16, 256 + 1, 8)
                heads_choices = range(1, 8 + 1)
                supports_gqa_mqa = True

                # NOTE: Flash 3 MLA fails a static check in bwd, seems like an FA bug
                ## Flash 3 supports MLA, but with some extra constraints
                # if backend == "flash3" and random.choice([True, False]):
                #    # Either head_dim_qk <= 64 and head_dim_v <= 512, or
                #    # 128 <= head_dim_qk <= 192 and 96 <= head_dim_v <= 128
                #    if random.choice([True, False]):
                #        head_dim = random.choice(range(16, 64 + 1, 8))
                #        head_dim_v = random.choice(head_dim_choices)
                #    else:
                #        head_dim = random.choice(range(128, 192 + 1, 8))
                #        head_dim_v = random.choice(range(96, 128 + 1, 8))

                # else:
                head_dim = random.choice(head_dim_choices)
                head_dim_v = None

            else:
                raise NotImplementedError()

            heads = random.choice(heads_choices)
            heads_kv = (
                heads
                if not supports_gqa_mqa
                else random.choice([1] + [i for i in range(1, heads + 1) if heads % i == 0])
            )
            assert heads >= heads_kv and heads % heads_kv == 0

            seqlen_q = random.choice(range(16, 2**14, 1))
            seqlen_kv = random.choice(range(16, 2**14, 1))

            is_causal = random.choice([True, False])

            while seqlen_q * seqlen_kv > max_qk:
                cut_kv = random.choice([True, False])
                if cut_kv:
                    seqlen_kv = int(seqlen_kv * 0.75)
                else:
                    seqlen_q = int(seqlen_q * 0.75)

            self._test_backend_against_torch_sdpa(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                is_causal=is_causal,
                backend=backend,
            )

    @pytest.mark.L0
    @skip_if_natten_not_supported()
    @skip_if_not_blackwell()
    def test_natten_blackwell_fast(self):
        problem_sizes = [
            (1, 8, 8, 128, 16384, 16384),
            (1, 8, 4, 128, 16384, 16384),
            (1, 8, 2, 128, 16384, 16384),
            (1, 8, 1, 128, 16384, 16384),
            (1, 12, 12, 128, 7688, 256),
            (1, 12, 6, 128, 7688, 256),
            (1, 12, 4, 128, 7688, 256),
            (1, 12, 3, 128, 7688, 256),
            (1, 12, 2, 128, 7688, 256),
            (1, 12, 1, 128, 7688, 256),
            (6, 2, 2, 128, 12244, 123),
            (6, 2, 1, 128, 12244, 123),
            (2, 1, 1, 128, 2048, 2048),
            (2, 1, 1, 64, 2048, 2048),
            (4, 1, 1, 64, 2048, 2048),
            (1, 1, 1, 64, 1411, 1375),
            (2, 1, 1, 64, 1536, 1280),
            (2, 1, 1, 64, 1536, 1536),
            (2, 1, 1, 64, 1536, 1376),
            (2, 1, 1, 64, 1416, 1376),
            (2, 1, 1, 64, 1411, 1375),
            (1, 1, 1, 64, 10240, 512),
            (2, 1, 1, 64, 10240, 512),
            (4, 1, 1, 64, 10240, 512),
            (8, 1, 1, 64, 10240, 512),
            (3, 1, 1, 64, 10240, 512),
            (4, 1, 1, 64, 10240, 512),
            (5, 1, 1, 64, 10240, 512),
            (6, 1, 1, 64, 10240, 512),
            (7, 1, 1, 64, 10240, 512),
            (8, 1, 1, 64, 10240, 512),
            (3, 3, 3, 64, 9197, 512),
            (3, 3, 1, 64, 9197, 512),
            (7, 1, 1, 64, 10240, 256),
            (3, 3, 3, 64, 10240, 256),
            (3, 3, 3, 64, 9216, 256),
            (3, 3, 3, 64, 9200, 256),
            (3, 3, 3, 64, 9200, 192),
            (3, 3, 3, 64, 9200, 168),
            (3, 3, 3, 64, 9200, 166),
            (3, 3, 3, 64, 9198, 166),
            (3, 3, 3, 64, 9197, 166),
            (4, 1, 1, 64, 10240, 10240),
            (4, 1, 1, 64, 10240, 1024),
            (1, 1, 1, 64, 9197, 166),
            (3, 3, 3, 64, 2560, 256),
            (1, 1, 1, 128, 128, 128),
            (2, 1, 1, 128, 128, 128),
            (1, 2, 2, 128, 128, 128),
            (2, 2, 2, 128, 128, 128),
            (2, 2, 2, 64, 128, 128),
            (1, 1, 1, 32, 32, 32),
            (1, 1, 1, 32, 128, 128),
            (1, 1, 1, 32, 128, 128),
            (1, 1, 1, 128, 128, 64),
            (1, 1, 1, 32, 128, 258),
            (1, 2, 2, 64, 128, 15),
            (1, 1, 1, 32, 8, 17),
            (1, 1, 1, 64, 17, 49),
            (2, 4, 4, 32, 128, 237),
            (2, 4, 2, 32, 128, 237),
            (2, 4, 1, 32, 128, 237),
            (4, 3, 3, 64, 256, 33),
            (4, 3, 1, 64, 256, 33),
            (1, 1, 1, 128, 128, 75),
            (1, 1, 1, 32, 125, 444),
            (1, 2, 2, 64, 125, 231),
            (1, 2, 1, 64, 125, 231),
            (1, 1, 1, 128, 256, 10240),
            (1, 1, 1, 32, 128, 4096),
            (1, 1, 1, 128, 3584, 381),
            (1, 1, 1, 128, 12072, 1680),
        ]
        for (
            batch,
            heads,
            heads_kv,
            head_dim,
            seqlen_q,
            seqlen_kv,
        ) in problem_sizes:
            for is_causal in [False, True]:
                self._test_backend_against_torch_sdpa(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    backend="natten",
                )

    @pytest.mark.L0
    @skip_if_natten_not_supported()
    @skip_if_not_hopper()
    def test_natten_hopper_fast(self):
        # No GQA/MQA
        # No MLA (except when using Ampere kernels)
        # No causal masking (except when using Ampere kernels)
        problem_sizes = [
            (1, 8, 8, 128, 16384, 16384),
            (1, 12, 12, 128, 7688, 256),
            (6, 2, 2, 128, 12244, 123),
            (2, 1, 1, 128, 2048, 2048),
            (2, 1, 1, 64, 2048, 2048),
            (4, 1, 1, 64, 2048, 2048),
            (1, 1, 1, 64, 1411, 1375),
            (2, 1, 1, 64, 1536, 1280),
            (2, 1, 1, 64, 1536, 1536),
            (2, 1, 1, 64, 1536, 1376),
            (2, 1, 1, 64, 1416, 1376),
            (2, 1, 1, 64, 1411, 1375),
            (1, 1, 1, 64, 10240, 512),
            (2, 1, 1, 64, 10240, 512),
            (4, 1, 1, 64, 10240, 512),
            (8, 1, 1, 64, 10240, 512),
            (3, 1, 1, 64, 10240, 512),
            (4, 1, 1, 64, 10240, 512),
            (5, 1, 1, 64, 10240, 512),
            (6, 1, 1, 64, 10240, 512),
            (7, 1, 1, 64, 10240, 512),
            (8, 1, 1, 64, 10240, 512),
            (3, 3, 3, 64, 9197, 512),
            (7, 1, 1, 64, 10240, 256),
            (3, 3, 3, 64, 10240, 256),
            (3, 3, 3, 64, 9216, 256),
            (3, 3, 3, 64, 9200, 256),
            (3, 3, 3, 64, 9200, 192),
            (3, 3, 3, 64, 9200, 168),
            (3, 3, 3, 64, 9200, 166),
            (3, 3, 3, 64, 9198, 166),
            (3, 3, 3, 64, 9197, 166),
            (4, 1, 1, 64, 10240, 10240),
            (4, 1, 1, 64, 10240, 1024),
            (1, 1, 1, 64, 9197, 166),
            (3, 3, 3, 64, 2560, 256),
            (1, 1, 1, 128, 128, 128),
            (2, 1, 1, 128, 128, 128),
            (1, 2, 2, 128, 128, 128),
            (2, 2, 2, 128, 128, 128),
            (2, 2, 2, 64, 128, 128),
            (1, 1, 1, 32, 32, 32),
            (1, 1, 1, 32, 128, 128),
            (1, 1, 1, 32, 128, 128),
            (1, 1, 1, 128, 128, 64),
            (1, 1, 1, 32, 128, 258),
            (1, 2, 2, 64, 128, 15),
            (1, 1, 1, 32, 8, 17),
            (1, 1, 1, 64, 17, 49),
            (2, 4, 4, 32, 128, 237),
            (4, 3, 3, 64, 256, 33),
            (1, 1, 1, 128, 128, 75),
            (1, 1, 1, 32, 125, 444),
            (1, 2, 2, 64, 125, 231),
            (1, 1, 1, 128, 256, 10240),
            (1, 1, 1, 32, 128, 4096),
            (1, 1, 1, 128, 3584, 381),
            (1, 1, 1, 128, 12072, 1680),
        ]
        for (
            batch,
            heads,
            heads_kv,
            head_dim,
            seqlen_q,
            seqlen_kv,
        ) in problem_sizes:
            for is_causal in [False, True]:
                self._test_backend_against_torch_sdpa(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    backend="natten",
                )

    @pytest.mark.L1
    @skip_if_natten_not_supported()
    @skip_if_not_supported()
    def test_natten_randsweep(self):
        self._test_randsweep_against_torch_sdpa(backend="natten", max_tests=RAND_SWEEP_TESTS)

    @pytest.mark.L0
    @skip_if_flash2_not_supported()
    @skip_if_not_supported()
    def test_flash2_fast(self):
        problem_sizes = [
            (1, 8, 8, 128, 16384, 16384),
            (1, 8, 4, 128, 16384, 16384),
            (1, 8, 2, 128, 16384, 16384),
            (1, 8, 1, 128, 16384, 16384),
            (1, 12, 12, 128, 7688, 256),
            (1, 12, 6, 128, 7688, 256),
            (1, 12, 4, 128, 7688, 256),
            (1, 12, 3, 128, 7688, 256),
            (1, 12, 2, 128, 7688, 256),
            (1, 12, 1, 128, 7688, 256),
            (6, 2, 2, 128, 12244, 123),
            (6, 2, 1, 128, 12244, 123),
            (2, 1, 1, 128, 2048, 2048),
            (2, 1, 1, 64, 2048, 2048),
            (4, 1, 1, 64, 2048, 2048),
            (1, 1, 1, 64, 1411, 1375),
            (2, 1, 1, 64, 1536, 1280),
            (2, 1, 1, 64, 1536, 1536),
            (2, 1, 1, 64, 1536, 1376),
            (2, 1, 1, 64, 1416, 1376),
            (2, 1, 1, 64, 1411, 1375),
            (1, 1, 1, 64, 10240, 512),
            (2, 1, 1, 64, 10240, 512),
            (4, 1, 1, 64, 10240, 512),
            (8, 1, 1, 64, 10240, 512),
            (3, 1, 1, 64, 10240, 512),
            (4, 1, 1, 64, 10240, 512),
            (5, 1, 1, 64, 10240, 512),
            (6, 1, 1, 64, 10240, 512),
            (7, 1, 1, 64, 10240, 512),
            (8, 1, 1, 64, 10240, 512),
            (3, 3, 3, 64, 9197, 512),
            (3, 3, 1, 64, 9197, 512),
            (7, 1, 1, 64, 10240, 256),
            (3, 3, 3, 64, 10240, 256),
            (3, 3, 3, 64, 9216, 256),
            (3, 3, 3, 64, 9200, 256),
            (3, 3, 3, 64, 9200, 192),
            (3, 3, 3, 64, 9200, 168),
            (3, 3, 3, 64, 9200, 166),
            (3, 3, 3, 64, 9198, 166),
            (3, 3, 3, 64, 9197, 166),
            (4, 1, 1, 64, 10240, 10240),
            (4, 1, 1, 64, 10240, 1024),
            (1, 1, 1, 64, 9197, 166),
            (3, 3, 3, 64, 2560, 256),
            (1, 1, 1, 128, 128, 128),
            (2, 1, 1, 128, 128, 128),
            (1, 2, 2, 128, 128, 128),
            (2, 2, 2, 128, 128, 128),
            (2, 2, 2, 64, 128, 128),
            (1, 1, 1, 32, 32, 32),
            (1, 1, 1, 32, 128, 128),
            (1, 1, 1, 32, 128, 128),
            (1, 1, 1, 128, 128, 64),
            (1, 1, 1, 32, 128, 258),
            (1, 2, 2, 64, 128, 15),
            (1, 1, 1, 32, 8, 17),
            (1, 1, 1, 64, 17, 49),
            (2, 4, 4, 32, 128, 237),
            (2, 4, 2, 32, 128, 237),
            (2, 4, 1, 32, 128, 237),
            (4, 3, 3, 64, 256, 33),
            (4, 3, 1, 64, 256, 33),
            (1, 1, 1, 128, 128, 75),
            (1, 1, 1, 32, 125, 444),
            (1, 2, 2, 64, 125, 231),
            (1, 2, 1, 64, 125, 231),
            (1, 1, 1, 128, 256, 10240),
            (1, 1, 1, 32, 128, 4096),
            (1, 1, 1, 128, 3584, 381),
            (1, 1, 1, 128, 12072, 1680),
        ]
        for (
            batch,
            heads,
            heads_kv,
            head_dim,
            seqlen_q,
            seqlen_kv,
        ) in problem_sizes:
            for is_causal in [False, True]:
                self._test_backend_against_torch_sdpa(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    backend="flash2",
                )

    @pytest.mark.L1
    @skip_if_flash2_not_supported()
    @skip_if_not_supported()
    def test_flash2_randsweep(self):
        self._test_randsweep_against_torch_sdpa(backend="flash2", max_tests=RAND_SWEEP_TESTS)

    @pytest.mark.L0
    @skip_if_flash3_not_supported()
    @skip_if_not_supported()
    def test_flash3_fast(self):
        problem_sizes = [
            (1, 8, 8, 128, 16384, 16384),
            (1, 8, 4, 128, 16384, 16384),
            (1, 8, 2, 128, 16384, 16384),
            (1, 8, 1, 128, 16384, 16384),
            (1, 12, 12, 128, 7688, 256),
            (1, 12, 6, 128, 7688, 256),
            (1, 12, 4, 128, 7688, 256),
            (1, 12, 3, 128, 7688, 256),
            (1, 12, 2, 128, 7688, 256),
            (1, 12, 1, 128, 7688, 256),
            (6, 2, 2, 128, 12244, 123),
            (6, 2, 1, 128, 12244, 123),
            (2, 1, 1, 128, 2048, 2048),
            (2, 1, 1, 64, 2048, 2048),
            (4, 1, 1, 64, 2048, 2048),
            (1, 1, 1, 64, 1411, 1375),
            (2, 1, 1, 64, 1536, 1280),
            (2, 1, 1, 64, 1536, 1536),
            (2, 1, 1, 64, 1536, 1376),
            (2, 1, 1, 64, 1416, 1376),
            (2, 1, 1, 64, 1411, 1375),
            (1, 1, 1, 64, 10240, 512),
            (2, 1, 1, 64, 10240, 512),
            (4, 1, 1, 64, 10240, 512),
            (8, 1, 1, 64, 10240, 512),
            (3, 1, 1, 64, 10240, 512),
            (4, 1, 1, 64, 10240, 512),
            (5, 1, 1, 64, 10240, 512),
            (6, 1, 1, 64, 10240, 512),
            (7, 1, 1, 64, 10240, 512),
            (8, 1, 1, 64, 10240, 512),
            (3, 3, 3, 64, 9197, 512),
            (3, 3, 1, 64, 9197, 512),
            (7, 1, 1, 64, 10240, 256),
            (3, 3, 3, 64, 10240, 256),
            (3, 3, 3, 64, 9216, 256),
            (3, 3, 3, 64, 9200, 256),
            (3, 3, 3, 64, 9200, 192),
            (3, 3, 3, 64, 9200, 168),
            (3, 3, 3, 64, 9200, 166),
            (3, 3, 3, 64, 9198, 166),
            (3, 3, 3, 64, 9197, 166),
            (4, 1, 1, 64, 10240, 10240),
            (4, 1, 1, 64, 10240, 1024),
            (1, 1, 1, 64, 9197, 166),
            (3, 3, 3, 64, 2560, 256),
            (1, 1, 1, 128, 128, 128),
            (2, 1, 1, 128, 128, 128),
            (1, 2, 2, 128, 128, 128),
            (2, 2, 2, 128, 128, 128),
            (2, 2, 2, 64, 128, 128),
            (1, 1, 1, 32, 32, 32),
            (1, 1, 1, 32, 128, 128),
            (1, 1, 1, 32, 128, 128),
            (1, 1, 1, 128, 128, 64),
            (1, 1, 1, 32, 128, 258),
            (1, 2, 2, 64, 128, 15),
            (1, 1, 1, 32, 8, 17),
            (1, 1, 1, 64, 17, 49),
            (2, 4, 4, 32, 128, 237),
            (2, 4, 2, 32, 128, 237),
            (2, 4, 1, 32, 128, 237),
            (4, 3, 3, 64, 256, 33),
            (4, 3, 1, 64, 256, 33),
            (1, 1, 1, 128, 128, 75),
            (1, 1, 1, 32, 125, 444),
            (1, 2, 2, 64, 125, 231),
            (1, 2, 1, 64, 125, 231),
            (1, 1, 1, 128, 256, 10240),
            (1, 1, 1, 32, 128, 4096),
            (1, 1, 1, 128, 3584, 381),
            (1, 1, 1, 128, 12072, 1680),
        ]
        for (
            batch,
            heads,
            heads_kv,
            head_dim,
            seqlen_q,
            seqlen_kv,
        ) in problem_sizes:
            for is_causal in [False, True]:
                self._test_backend_against_torch_sdpa(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    backend="flash3",
                )

    @pytest.mark.L1
    @skip_if_flash3_not_supported()
    @skip_if_not_hopper()
    def test_flash3_randsweep(self):
        self._test_randsweep_against_torch_sdpa(backend="flash3", max_tests=RAND_SWEEP_TESTS)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    unittest.main()
