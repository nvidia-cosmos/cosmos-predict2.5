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

import unittest
from functools import partial

import pytest
import torch
from torch import nn

from cosmos_predict2._src.imaginaire.attention import attention as i4_attention
from cosmos_predict2._src.imaginaire.attention.flash2 import FLASH2_SUPPORTED
from cosmos_predict2._src.imaginaire.attention.flash3 import FLASH3_SUPPORTED
from cosmos_predict2._src.imaginaire.attention.masks import CausalType
from cosmos_predict2._src.imaginaire.attention.natten import NATTEN_SUPPORTED
from cosmos_predict2._src.imaginaire.attention.utils import is_blackwell_dc, is_hopper
from cosmos_predict2._src.imaginaire.attention.utils import safe_log as log
from cosmos_predict2._src.imaginaire.attention.varlen import generate_varlen_parameters
from cosmos_predict2._src.imaginaire.utils.device import with_torch_device

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
    reason="Attention tests are only allowed for Hopper and Blackwell DC-class GPUs for now.",
)

skip_if_not_blackwell = partial(
    pytest.mark.skipif, not is_blackwell_dc(), reason="This test is only allowed for Blackwell DC-class GPUs."
)

skip_if_not_hopper = partial(pytest.mark.skipif, not is_hopper(), reason="This test is only allowed for Hopper GPUs.")


def reset_torch_compile(cache_size_limit):
    # Torch compile reset and sensible settings for unit testing
    log.debug(f"Resetting torch compile cache. New cache size limit: {cache_size_limit}")
    torch.compiler.reset()
    torch._dynamo.config.cache_size_limit = cache_size_limit
    torch._dynamo.config.accumulated_recompile_limit = cache_size_limit * 4
    torch._dynamo.config.fail_on_recompile_limit_hit = True


def _reset_everything():
    torch.manual_seed(42)
    reset_torch_compile(1024)
    torch.cuda.empty_cache()


class Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        qkv_bias: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.mlp_dim = int(self.embed_dim * self.mlp_ratio)
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(self.embed_dim, self.embed_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.mlp_dim),
            nn.GELU(),
            nn.Linear(self.mlp_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, *args, **kwargs):
        B, sQ, D = x.shape
        B, sK, D = c.shape
        q = self.q(x).reshape(B, sQ, self.num_heads, self.head_dim)
        k, v = self.kv(c).reshape(B, sK, 2, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)

        x0 = i4_attention(q, k, v, *args, **kwargs)
        assert isinstance(x0, torch.Tensor)
        x0 = x0.reshape(B, sQ, D)

        return self.mlp(x0)


class TorchCompileTests(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    @with_torch_device(device="cuda")
    def _test_module(
        self,
        batch: int,
        seqlens_Q: list[int],
        seqlens_KV: list[int],
        num_heads: int,
        head_dim: int,
        is_causal: bool,
        causal_type: CausalType,
        atol: float,
        backend: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        embed_dim = num_heads * head_dim

        assert len(seqlens_Q) == len(seqlens_KV)
        assert len(seqlens_Q) >= 1
        assert len(seqlens_Q) == 1 or batch == len(seqlens_Q)

        seqlen_q = sum(seqlens_Q)
        seqlen_kv = sum(seqlens_KV)
        is_varlen = len(seqlens_Q) > 1

        batch_ = 1 if is_varlen else batch
        seqlens_Q_ = torch.tensor(seqlens_Q, device=device, dtype=torch.int32) if is_varlen else None
        seqlens_KV_ = torch.tensor(seqlens_KV, device=device, dtype=torch.int32) if is_varlen else None

        dummy_q = torch.randn(
            (batch_, seqlen_q, num_heads, head_dim),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        dummy_kv = torch.randn(
            (batch_, seqlen_kv, num_heads, head_dim),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        # seq maxes MUST be computed ahead of time when using torch compile
        # because need to be copied to host
        (
            cumulative_seqlen_Q,
            cumulative_seqlen_KV,
            max_seqlen_Q,
            max_seqlen_KV,
        ) = generate_varlen_parameters(
            query=dummy_q,
            key=dummy_kv,
            value=dummy_kv,
            seqlens_Q=seqlens_Q_,
            seqlens_KV=seqlens_KV_,
        )

        _reset_everything()

        log.debug(
            f"Testing torch compile on Attention module with input shapes: "
            f"{batch=}, {num_heads=}, {head_dim=}, {seqlens_Q=}, {seqlens_KV=}, "
            f"{is_causal=}, {causal_type=}, {dtype=}, {device=}, {backend=}."
        )

        model_eager = (
            Block(
                embed_dim=embed_dim,
                mlp_ratio=2,
                num_heads=num_heads,
            )
            .to(dtype)
            .to(device)
        )

        model_compiled = torch.compile(model_eager, fullgraph=True, backend="inductor")

        x = torch.randn((batch_, seqlen_q, embed_dim), dtype=dtype, device=device)
        c = torch.randn((batch_, seqlen_kv, embed_dim), dtype=dtype, device=device)
        dy = torch.randn((batch_, seqlen_q, embed_dim), dtype=dtype, device=device) * 0.1

        x_ref = x.clone().requires_grad_(True)
        c_ref = c.clone().requires_grad_(True)
        dy_ref = dy.clone()

        # eager
        y_ref = model_eager(
            x_ref,
            c_ref,
            is_causal=is_causal,
            causal_type=causal_type,
            cumulative_seqlen_Q=cumulative_seqlen_Q,
            cumulative_seqlen_KV=cumulative_seqlen_KV,
            max_seqlen_Q=max_seqlen_Q,
            max_seqlen_KV=max_seqlen_KV,
            backend=backend,
        )
        y_ref.backward(dy_ref)
        dx_ref = x_ref.grad
        dc_ref = c_ref.grad

        # compile on first attempt
        x = x.requires_grad_(True)
        c = c.requires_grad_(True)
        y = model_compiled(
            x,
            c,
            is_causal=is_causal,
            causal_type=causal_type,
            cumulative_seqlen_Q=cumulative_seqlen_Q,
            cumulative_seqlen_KV=cumulative_seqlen_KV,
            max_seqlen_Q=max_seqlen_Q,
            max_seqlen_KV=max_seqlen_KV,
            backend=backend,
        )
        y.backward(dy)
        dx = x.grad
        dc = c.grad

        torch.testing.assert_close(y, y_ref, atol=atol, rtol=0)
        torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=0)
        torch.testing.assert_close(dc, dc_ref, atol=atol, rtol=0)

        # Second run, just to make sure it doesn't crash
        y = model_compiled(
            x,
            c,
            is_causal=is_causal,
            causal_type=causal_type,
            cumulative_seqlen_Q=cumulative_seqlen_Q,
            cumulative_seqlen_KV=cumulative_seqlen_KV,
            max_seqlen_Q=max_seqlen_Q,
            max_seqlen_KV=max_seqlen_KV,
            backend=backend,
        )
        y.backward(dy)
        dx = x.grad
        dc = c.grad

    @pytest.mark.L0
    @skip_if_natten_not_supported()
    @skip_if_not_supported()
    def test_compiled_natten(self):
        problem_sizes = [
            (1, 4, 128, [128], [128]),
            (1, 1, 128, [128], [1024]),
            (1, 1, 128, [128], [13568]),
            (1, 1, 128, [128], [13496]),
            (1, 1, 32, [128], [13496]),
            (1, 1, 32, [32], [13496]),
            (3, 1, 32, [77], [8504]),
            (1, 1, 32, [77], [8504]),
            (1, 1, 64, [40], [12296]),
            (1, 2, 64, [40], [12296]),
            (1, 2, 64, [40], [12296]),
            (1, 1, 128, [128], [128]),
        ]
        for (
            batch,
            num_heads,
            head_dim,
            seqlens_Q,
            seqlens_KV,
        ) in problem_sizes:
            for is_causal in [False, True]:
                self._test_module(
                    batch=batch,
                    seqlens_Q=seqlens_Q,
                    seqlens_KV=seqlens_KV,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    is_causal=is_causal,
                    causal_type=CausalType.TopLeft,
                    atol=1e-3,
                    backend="natten",
                )

    @pytest.mark.L0
    @skip_if_flash2_not_supported()
    @skip_if_not_supported()
    def test_compiled_flash2(self):
        problem_sizes = [
            (1, 4, 128, [128], [128]),
            (1, 1, 128, [128], [1024]),
            (1, 1, 128, [128], [13568]),
            (1, 1, 128, [128], [13496]),
            (1, 1, 32, [128], [13496]),
            (1, 1, 32, [32], [13496]),
            (3, 1, 32, [77], [8504]),
            (1, 1, 32, [77], [8504]),
            (1, 1, 64, [40], [12296]),
            (1, 2, 64, [40], [12296]),
            (1, 2, 64, [40], [12296]),
            (1, 1, 128, [128], [128]),
        ]
        for (
            batch,
            num_heads,
            head_dim,
            seqlens_Q,
            seqlens_KV,
        ) in problem_sizes:
            for is_causal in [False, True]:
                self._test_module(
                    batch=batch,
                    seqlens_Q=seqlens_Q,
                    seqlens_KV=seqlens_KV,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    is_causal=is_causal,
                    causal_type=CausalType.BottomRight,
                    atol=1e-3,
                    backend="flash2",
                )

    @pytest.mark.L0
    @skip_if_flash3_not_supported()
    @skip_if_not_supported()
    def test_compiled_flash3(self):
        problem_sizes = [
            (1, 4, 128, [128], [128]),
            (1, 1, 128, [128], [1024]),
            (1, 1, 128, [128], [13568]),
            (1, 1, 128, [128], [13496]),
            (1, 1, 32, [128], [13496]),
            (1, 1, 32, [32], [13496]),
            (3, 1, 32, [77], [8504]),
            (1, 1, 32, [77], [8504]),
            (1, 1, 64, [40], [12296]),
            (1, 2, 64, [40], [12296]),
            (1, 2, 64, [40], [12296]),
            (1, 1, 128, [128], [128]),
        ]
        for (
            batch,
            num_heads,
            head_dim,
            seqlens_Q,
            seqlens_KV,
        ) in problem_sizes:
            for is_causal in [False, True]:
                self._test_module(
                    batch=batch,
                    seqlens_Q=seqlens_Q,
                    seqlens_KV=seqlens_KV,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    is_causal=is_causal,
                    causal_type=CausalType.BottomRight,
                    atol=1e-3,
                    backend="flash3",
                )


if __name__ == "__main__":
    torch.manual_seed(42)
    unittest.main()
