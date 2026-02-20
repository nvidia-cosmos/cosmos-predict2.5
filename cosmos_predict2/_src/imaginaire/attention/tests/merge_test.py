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

Attention merging unit tests.
"""

import random
import unittest
from functools import partial

import pytest
import torch

from cosmos_predict2._src.imaginaire.attention import attention as i4_attention
from cosmos_predict2._src.imaginaire.attention import merge_attentions
from cosmos_predict2._src.imaginaire.attention.flash2 import FLASH2_SUPPORTED
from cosmos_predict2._src.imaginaire.attention.flash3 import FLASH3_SUPPORTED
from cosmos_predict2._src.imaginaire.attention.natten import NATTEN_SUPPORTED
from cosmos_predict2._src.imaginaire.attention.utils import is_blackwell_dc, is_hopper
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
    reason="Attention merge tests are only allowed for Hopper and Blackwell DC-class GPUs for now.",
)

skip_if_not_blackwell = partial(
    pytest.mark.skipif, not is_blackwell_dc(), reason="This test is only allowed for Blackwell DC-class GPUs."
)

skip_if_not_hopper = partial(pytest.mark.skipif, not is_hopper(), reason="This test is only allowed for Hopper GPUs.")


def _reset_everything():
    torch.manual_seed(42)
    torch.cuda.empty_cache()


def sdpa_split(
    q: torch.Tensor,
    k_0: torch.Tensor,
    v_0: torch.Tensor,
    k_1: torch.Tensor,
    v_1: torch.Tensor,
    do: torch.Tensor,
    backend: str,
    backend_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute attention by splitting KV into two branches and merging.
    """
    q = q.requires_grad_(True)
    k_0 = k_0.requires_grad_(True)
    v_0 = v_0.requires_grad_(True)
    k_1 = k_1.requires_grad_(True)
    v_1 = v_1.requires_grad_(True)

    out1, lse1 = i4_attention(q, k_0, v_0, return_lse=True, backend=backend, backend_kwargs=backend_kwargs)
    out2, lse2 = i4_attention(q, k_1, v_1, return_lse=True, backend=backend, backend_kwargs=backend_kwargs)

    out, _ = merge_attentions([out1, out2], [lse1, lse2], torch_compile=False)

    out.backward(do)

    with torch.no_grad():
        output = out.data
        assert q.grad is not None
        assert k_0.grad is not None
        assert k_1.grad is not None
        assert v_0.grad is not None
        assert v_1.grad is not None
        dq, dk1, dv1 = q.grad.data, k_0.grad.data, v_0.grad.data
        dk2, dv2 = k_1.grad.data, v_1.grad.data

        dk = torch.cat([dk1, dk2], dim=1)
        dv = torch.cat([dv1, dv2], dim=1)

        return output, dq, dk, dv


def sdpa_ref(
    q: torch.Tensor,
    k_0: torch.Tensor,
    v_0: torch.Tensor,
    k_1: torch.Tensor,
    v_1: torch.Tensor,
    do: torch.Tensor,
    backend: str,
    backend_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute reference attention by concatenating KV.
    """
    with torch.no_grad():
        k = torch.cat([k_0, k_1], dim=1)
        v = torch.cat([v_0, v_1], dim=1)

    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)

    out = i4_attention(q, k, v, backend=backend, backend_kwargs=backend_kwargs)
    out.backward(do)

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None

    return out.data, q.grad.data, k.grad.data, v.grad.data


class AttentionMergeTest(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    @with_torch_device(device="cuda")
    def _test(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlen_Q: int,
        seqlen_KV_0: int,
        seqlen_KV_1: int,
        backend: str,
        heads_kv: int | None = None,
        head_dim_v: int | None = None,
        backend_kwargs: dict | None = None,
    ):
        heads_kv = heads_kv or heads
        head_dim_v = head_dim_v or head_dim

        # We're testing against the same backend, so we can use higher tolerance
        ALLOWED_DTYPES = [
            # (dtype, atol_out, (atol_dq, atol_dk, atol_dv))
            (torch.float32, 1e-3, (1e-2, 1e-3, 1e-3)),
            (torch.float16, 1e-2, (1e-2, 1e-2, 1e-2)),
            (torch.bfloat16, 5e-2, (5e-2, 5e-2, 5e-2)),
        ]

        # Most backends only support float16 and bfloat16
        SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]

        # NATTEN supports float32 in the cutlass-fmha backend
        if backend == "natten":
            SUPPORTED_DTYPES.append(torch.float32)

        for dtype, atol_out, (atol_dq, atol_dk, atol_dv) in ALLOWED_DTYPES:
            if dtype not in SUPPORTED_DTYPES:
                continue

            log.debug(
                f"Testing Attention Merging ({backend}): "
                f"{batch=}, {heads=}, {heads_kv=}, {head_dim=}, {head_dim_v=}, "
                f"{seqlen_Q=}, {seqlen_KV_0=}, {seqlen_KV_1=}, "
                f"{dtype=}."
            )

            q = torch.randn(batch, seqlen_Q, heads, head_dim, device="cuda", dtype=dtype)
            k_0 = torch.randn(batch, seqlen_KV_0, heads_kv, head_dim, device="cuda", dtype=dtype)
            v_0 = torch.randn(batch, seqlen_KV_0, heads_kv, head_dim_v, device="cuda", dtype=dtype)
            k_1 = torch.randn(batch, seqlen_KV_1, heads_kv, head_dim, device="cuda", dtype=dtype)
            v_1 = torch.randn(batch, seqlen_KV_1, heads_kv, head_dim_v, device="cuda", dtype=dtype)
            do = torch.randn(batch, seqlen_Q, heads, head_dim_v, device="cuda", dtype=dtype)

            q_ref = q.clone()
            k_0_ref = k_0.clone()
            v_0_ref = v_0.clone()
            k_1_ref = k_1.clone()
            v_1_ref = v_1.clone()
            do_ref = do.clone()

            output_ref, dq_ref, dk_ref, dv_ref = sdpa_ref(
                q_ref,
                k_0_ref,
                v_0_ref,
                k_1_ref,
                v_1_ref,
                do_ref,
                backend=backend,
                backend_kwargs=backend_kwargs,
            )

            output, dq, dk, dv = sdpa_split(q, k_0, v_0, k_1, v_1, do, backend=backend, backend_kwargs=backend_kwargs)

            torch.testing.assert_close(output.float(), output_ref.float(), atol=atol_out, rtol=0)
            torch.testing.assert_close(dk.float(), dk_ref.float(), atol=atol_dk, rtol=0)
            torch.testing.assert_close(dv.float(), dv_ref.float(), atol=atol_dv, rtol=0)
            torch.testing.assert_close(dq.float(), dq_ref.float(), atol=atol_dq, rtol=0)

    def _test_randsweep(self, backend: str, max_tests: int = 1000, backend_kwargs: dict | None = None):
        random.seed(42)

        max_Q = 16384
        max_KV = 16384

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

            seqlen_Q = random.choice(range(8, max_Q + 1))
            seqlen_KV_0 = random.choice(range(8, max_KV + 1))
            seqlen_KV_1 = random.choice(range(8, max_KV + 1))

            self._test(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                seqlen_Q=seqlen_Q,
                seqlen_KV_0=seqlen_KV_0,
                seqlen_KV_1=seqlen_KV_1,
                backend=backend,
                backend_kwargs=backend_kwargs,
            )

    @pytest.mark.L0
    @skip_if_natten_not_supported()
    @skip_if_not_supported()
    def test_natten_merge_fast(self):
        problem_sizes = [
            (1, 2, 128, 512, 256, 256),
            (2, 4, 64, 1024, 512, 768),
            (1, 1, 128, 256, 128, 128),
            (4, 2, 128, 512, 256, 512),
        ]

        backend_kwargs = None

        for batch, heads, head_dim, seqlen_Q, seqlen_KV_0, seqlen_KV_1 in problem_sizes:
            self._test(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_Q=seqlen_Q,
                seqlen_KV_0=seqlen_KV_0,
                seqlen_KV_1=seqlen_KV_1,
                backend="natten",
                backend_kwargs=backend_kwargs,
            )

    @pytest.mark.L1
    @skip_if_natten_not_supported()
    @skip_if_not_supported()
    def test_natten_merge_randsweep(self):
        backend_kwargs = None
        self._test_randsweep(backend="natten", max_tests=RAND_SWEEP_TESTS, backend_kwargs=backend_kwargs)

    @pytest.mark.L0
    @skip_if_flash2_not_supported()
    @skip_if_not_supported()
    def test_flash2_merge_fast(self):
        problem_sizes = [
            (1, 2, 128, 512, 256, 256),
            (2, 4, 64, 1024, 512, 768),
            (1, 1, 128, 256, 128, 128),
            (4, 2, 128, 512, 256, 512),
        ]

        for batch, heads, head_dim, seqlen_Q, seqlen_KV_0, seqlen_KV_1 in problem_sizes:
            self._test(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_Q=seqlen_Q,
                seqlen_KV_0=seqlen_KV_0,
                seqlen_KV_1=seqlen_KV_1,
                backend="flash2",
            )

    @pytest.mark.L1
    @skip_if_flash2_not_supported()
    @skip_if_not_supported()
    def test_flash2_merge_randsweep(self):
        self._test_randsweep(backend="flash2", max_tests=RAND_SWEEP_TESTS)

    @pytest.mark.L0
    @skip_if_flash3_not_supported()
    @skip_if_not_hopper()
    def test_flash3_merge_fast(self):
        problem_sizes = [
            (1, 2, 128, 512, 256, 256),
            (2, 4, 64, 1024, 512, 768),
            (1, 1, 128, 256, 128, 128),
            (4, 2, 128, 512, 256, 512),
        ]

        for batch, heads, head_dim, seqlen_Q, seqlen_KV_0, seqlen_KV_1 in problem_sizes:
            self._test(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_Q=seqlen_Q,
                seqlen_KV_0=seqlen_KV_0,
                seqlen_KV_1=seqlen_KV_1,
                backend="flash3",
            )

    @pytest.mark.L1
    @skip_if_flash3_not_supported()
    @skip_if_not_hopper()
    def test_flash3_merge_randsweep(self):
        self._test_randsweep(backend="flash3", max_tests=RAND_SWEEP_TESTS)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    unittest.main()
