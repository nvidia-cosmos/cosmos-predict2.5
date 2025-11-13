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
Flash Attention with JVP
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Taken from https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py (2025/03)

Simplified version, combining Triton forward and official backward (Kaiwen Zheng)

Modified to support Jacobian-vector-product (JVP) computation (Kaiwen Zheng)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import torch
import triton
import triton.language as tl
from einops import rearrange
from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_varlen_backward

DEVICE = "cuda"


@triton.jit
def _attn_fwd_inner(
    acc,
    acc_A,
    acc_B,
    l_i,
    m_i,
    r_i,
    q,
    tq,  #
    K_block_ptr,
    V_block_ptr,
    tK_block_ptr,
    tV_block_ptr,  #
    start_m,
    sm_scale,  #
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    SEQ_LEN_KV: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    bf16_v: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, min(start_m * BLOCK_M, SEQ_LEN_KV)
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, min((start_m + 1) * BLOCK_M, SEQ_LEN_KV)
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, SEQ_LEN_KV
    qk_scale = sm_scale * 1.44269504
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    tK_block_ptr = tl.advance(tK_block_ptr, (0, lo))
    tV_block_ptr = tl.advance(tV_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k, tk = (
            tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero"),
            tl.load(tK_block_ptr, boundary_check=(0, 1), padding_option="zero"),
        )
        qk = tl.dot(q, k)

        tS_ij = tl.dot(tq, k)
        tS_ij = tl.dot(q, tk, tS_ij)
        tS_ij *= sm_scale
        if STAGE == 2:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(causal_mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        # mask if SEQ_LEN_KV % BLOCK_N != 0
        boundary_m = tl.full([BLOCK_M], hi, dtype=tl.int32)
        size_n = start_n + offs_n[None, :]
        mask = size_n < boundary_m[:, None]
        qk = tl.where(mask, qk, float("-inf"))
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        tS_ij = tl.where(mask, tS_ij, float("0"))
        H_ij = p * tS_ij
        r_ij = tl.sum(H_ij, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        r_i = r_i * alpha + r_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        acc_A = acc_A * alpha[:, None]
        acc_B = acc_B * alpha[:, None]
        # update acc
        v, tv = (
            tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero"),
            tl.load(tV_block_ptr, boundary_check=(0, 1), padding_option="zero"),
        )
        # boundary_v = tl.full([HEAD_DIM_V], hi, dtype=tl.int32)
        # size_n = start_n + offs_n
        # mask_v = size_n[:, None] < boundary_v[None, :]
        # v = tl.where(mask_v, v, float("0"))
        # tv = tl.where(mask_v, tv, float("0"))
        if bf16_v:
            p = p.to(tl.bfloat16)
            H_ij = H_ij.to(tl.bfloat16)
            v = v.to(tl.bfloat16)
            tv = tv.to(tl.bfloat16)
        else:
            p = p.to(tl.float16)
            H_ij = H_ij.to(tl.float16)
            v = v.to(tl.float16)
            tv = tv.to(tl.float16)
        acc = tl.dot(p, v, acc)
        acc_A = tl.dot(p, tv, acc_A)
        acc_B = tl.dot(H_ij, v, acc_B)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        tV_block_ptr = tl.advance(tV_block_ptr, (BLOCK_N, 0))
        tK_block_ptr = tl.advance(tK_block_ptr, (0, BLOCK_N))
    return acc, acc_A, acc_B, l_i, m_i, r_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [16, 32, 64]
    for s in [3, 4, 7]
    for w in [4, 8]
]


@triton.autotune(configs, key=["SEQ_LEN_Q", "SEQ_LEN_KV", "HEAD_DIM_QK", "HEAD_DIM_V"])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    tQ,
    tK,
    tV,
    sm_scale,
    M,
    Out,
    tOut,  #
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qd,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,  #
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vd,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,  #
    Z,
    H,  #
    SEQ_LEN_Q,
    SEQ_LEN_KV,  #
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    start_m_idx = start_m * BLOCK_M
    # end_m_idx = (start_m + 1) * BLOCK_M

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(SEQ_LEN_Q, HEAD_DIM_QK),
        strides=(stride_qm, stride_qd),
        offsets=(start_m_idx, 0),
        block_shape=(BLOCK_M, HEAD_DIM_QK),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(SEQ_LEN_KV, HEAD_DIM_V),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM_V),
        order=(1, 0),
    )
    # load transposed K
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM_QK, SEQ_LEN_KV),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM_QK, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(SEQ_LEN_Q, HEAD_DIM_V),
        strides=(stride_om, stride_od),
        offsets=(start_m_idx, 0),
        block_shape=(BLOCK_M, HEAD_DIM_V),
        order=(1, 0),
    )
    tQ_block_ptr = tl.make_block_ptr(
        base=tQ + q_offset,
        shape=(SEQ_LEN_Q, HEAD_DIM_QK),
        strides=(stride_qm, stride_qd),
        offsets=(start_m_idx, 0),
        block_shape=(BLOCK_M, HEAD_DIM_QK),
        order=(1, 0),
    )
    tV_block_ptr = tl.make_block_ptr(
        base=tV + v_offset,
        shape=(SEQ_LEN_KV, HEAD_DIM_V),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM_V),
        order=(1, 0),
    )
    # load transposed K
    tK_block_ptr = tl.make_block_ptr(
        base=tK + k_offset,
        shape=(HEAD_DIM_QK, SEQ_LEN_KV),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM_QK, BLOCK_N),
        order=(0, 1),
    )
    tO_block_ptr = tl.make_block_ptr(
        base=tOut + o_offset,
        shape=(SEQ_LEN_Q, HEAD_DIM_V),
        strides=(stride_om, stride_od),
        offsets=(start_m_idx, 0),
        block_shape=(BLOCK_M, HEAD_DIM_V),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m_idx + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_qk, offs_d_v = tl.arange(0, HEAD_DIM_QK), tl.arange(0, HEAD_DIM_V)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)
    r_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_A = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)
    acc_B = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q, tq = (
        tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero"),
        tl.load(tQ_block_ptr, boundary_check=(0, 1), padding_option="zero"),
    )
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, acc_A, acc_B, l_i, m_i, r_i = _attn_fwd_inner(
            acc,
            acc_A,
            acc_B,
            l_i,
            m_i,
            r_i,
            q,
            tq,  #
            K_block_ptr,
            V_block_ptr,
            tK_block_ptr,
            tV_block_ptr,  #
            start_m,
            sm_scale,  #
            BLOCK_M,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            SEQ_LEN_KV,
            HEAD_DIM_V,
            V.dtype.element_ty == tl.bfloat16,  #
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, acc_A, acc_B, l_i, m_i, r_i = _attn_fwd_inner(
            acc,
            acc_A,
            acc_B,
            l_i,
            m_i,
            r_i,
            q,
            tq,  #
            K_block_ptr,
            V_block_ptr,
            tK_block_ptr,
            tV_block_ptr,  #
            start_m,
            sm_scale,  #
            BLOCK_M,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            SEQ_LEN_KV,
            HEAD_DIM_V,
            V.dtype.element_ty == tl.bfloat16,  #
        )

    # epilogue
    # m_i += tl.math.log2(l_i)
    empty_mask = l_i == 0.0
    # NOTE: This happens if the entire block is masked out.
    l_i = tl.where(empty_mask, 1.0, l_i)
    # NOTE: This is needed to compute the logsumexp for the backward pass.
    m_i = m_i + tl.where(
        empty_mask,
        0.0,
        tl.math.log2(l_i),
    )

    acc = acc / l_i[:, None]
    tO_i = (acc_A + acc_B - (r_i[:, None] * acc)) / l_i[:, None]
    m_ptrs = M + off_hz * SEQ_LEN_Q + offs_m
    O_block_ptr = Out + o_offset + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_od
    tO_block_ptr = tOut + o_offset + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_od
    # mask if SEQ_LEN_Q % BLOCK_M != 0
    mask_lse = offs_m < SEQ_LEN_Q
    mask = offs_m[:, None] < SEQ_LEN_Q
    tl.store(m_ptrs, m_i * 0.69314718, mask=mask_lse)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=mask)
    tl.store(tO_block_ptr, tO_i.to(tOut.type.element_ty), mask=mask)


def generate_qkv(q, k, v):
    """
    Arguments:
        q: (batch_size, nheads, seqlen_q, d)
        k: (batch_size, nheads_k, seqlen_k, d)
        v: (batch_size, nheads_k, seqlen_k, d)
    """
    batch_size, _, seqlen_q, d = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    assert k.shape == (batch_size, nheads_k, seqlen_k, d)
    assert v.shape == (batch_size, nheads_k, seqlen_k, d)

    def unpad_fn(x):
        return rearrange(x, "b h s d -> (b s) h d")

    def lse_unpad_fn(x):
        return rearrange(x, "b h s -> (b s) h")

    def pad_fn(x):
        return rearrange(x, "(b s) h d -> b h s d", b=batch_size)

    # unpad_fn = lambda x: rearrange(x, "b h s d -> (b s) h d")
    # lse_unpad_fn = lambda x: rearrange(x, "b h s -> (b s) h")
    # pad_fn = lambda x: rearrange(x, "(b s) h d -> b h s d", b=batch_size)

    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)
    max_seqlen_q = seqlen_q

    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=q.device)
    max_seqlen_k = seqlen_k

    return (
        unpad_fn,
        lse_unpad_fn,
        pad_fn,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )


class _attention(torch.autograd.Function):
    """
    Arguments:
        q, tq: (batch_size, nheads, seqlen_q, d_qk)
        k, tk: (batch_size, nheads, seqlen_kv, d_qk)
        v, tv: (batch_size, nheads, seqlen_kv, d_v)
    Returns:
        o, to: (batch_size, nheads, seqlen_q, d_v)

    Backward is only supported when d_qk=d_v.
    """

    @staticmethod
    def forward(ctx, q, k, v, tq, tk, tv, causal=False, sm_scale=None):
        is_grad = any(x.requires_grad for x in [q, k, v])
        # shape constraints
        assert q.shape[:-2] == k.shape[:-2] and k.shape[:-2] == v.shape[:-2]
        assert k.shape[-2] == v.shape[-2] and q.shape[-1] == k.shape[-1]
        Z, H = q.shape[:-2]
        SEQ_LEN_Q, SEQ_LEN_KV = q.shape[-2], k.shape[-2]
        HEAD_DIM_QK, HEAD_DIM_V = q.shape[-1], v.shape[-1]
        assert HEAD_DIM_QK in {16, 32, 64, 128, 256}
        assert HEAD_DIM_V in {16, 32, 64, 128, 256}
        assert (SEQ_LEN_Q == SEQ_LEN_KV) or (not causal), "Causal cross-attention is currently not supported."
        assert tq.shape == q.shape and tk.shape == k.shape and tv.shape == v.shape
        assert tq.stride() == q.stride() and tk.stride() == k.stride() and tv.stride() == v.stride()
        if sm_scale is None:
            sm_scale = HEAD_DIM_QK ** (-0.5)
        o = torch.empty((Z, H, SEQ_LEN_Q, HEAD_DIM_V), device=q.device, dtype=q.dtype)
        to = torch.empty_like(o)
        stage = 3 if causal else 1

        M = torch.empty((Z, H, SEQ_LEN_Q), device=q.device, dtype=torch.float32)

        def grid(args):
            return (triton.cdiv(SEQ_LEN_Q, args["BLOCK_M"]), Z * H, 1)

        # grid = lambda args: (triton.cdiv(SEQ_LEN_Q, args["BLOCK_M"]), Z * H, 1)
        ctx.grid = grid
        _attn_fwd[grid](
            q,
            k,
            v,
            tq,
            tk,
            tv,
            sm_scale,
            M,
            o,
            to,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),  #
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            Z,
            H,  #
            SEQ_LEN_Q,
            SEQ_LEN_KV,  #
            HEAD_DIM_QK,
            HEAD_DIM_V,  #
            STAGE=stage,
        )

        if is_grad:
            ctx.save_for_backward(q, k, v, o, M)
            ctx.sm_scale = sm_scale
            ctx.causal = causal
        return o, to

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        assert q.shape[-1] == k.shape[-1] and k.shape[-1] == v.shape[-1], (
            "Backward not supported with different headdim."
        )
        # flash_attn uses the shape (batch_size, seqlen, nheads, headdim)
        # torch.nn.functional.scaled_dot_product_attention and this implementation use (batch_size, nheads, seqlen, headdim)
        if q.shape[-2] == k.shape[-2]:
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            _flash_attn_backward(
                dout.transpose(1, 2),
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                out.transpose(1, 2),
                softmax_lse,
                dq.transpose(1, 2),
                dk.transpose(1, 2),
                dv.transpose(1, 2),
                dropout_p=0.0,
                softmax_scale=ctx.sm_scale,
                causal=ctx.causal,
                window_size=(-1, -1),
                # softcap=0,
                alibi_slopes=None,
                deterministic=False,
            )
        else:
            unpad_fn, lse_unpad_fn, pad_fn, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = generate_qkv(
                q, k, v
            )
            q_unpad, k_unpad, v_unpad = unpad_fn(q), unpad_fn(k), unpad_fn(v)
            dq, dk, dv = torch.empty_like(q_unpad), torch.empty_like(k_unpad), torch.empty_like(v_unpad)
            _flash_attn_varlen_backward(
                unpad_fn(dout),
                q_unpad,
                k_unpad,
                v_unpad,
                unpad_fn(out),
                lse_unpad_fn(softmax_lse),
                dq,
                dk,
                dv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=ctx.sm_scale,
                causal=ctx.causal,
                window_size=(-1, -1),
                # softcap=0,
                alibi_slopes=None,
                deterministic=False,
            )
            dq, dk, dv = pad_fn(dq), pad_fn(dk), pad_fn(dv)
        return dq, dk, dv, None, None, None, None, None


attention = _attention.apply


# def _test_fwd_bwd(Z, H, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
#     torch.manual_seed(20)
#     q = torch.empty((Z, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
#     k = torch.empty((Z, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
#     v = torch.empty((Z, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
#     tq = torch.zeros_like(q)
#     tk = torch.zeros_like(k)
#     tv = torch.zeros_like(v)
#     sm_scale = 0.5
#     dout = torch.randn_like(q)
#     # reference implementation
#     M = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=DEVICE))
#     p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
#     if causal:
#         p[:, :, M == 0] = float("-inf")
#     p = torch.softmax(p.float(), dim=-1).to(dtype)
#     ref_out = torch.matmul(p, v)
#     ref_out.backward(dout)
#     ref_dv, v.grad = v.grad.clone(), None
#     ref_dk, k.grad = k.grad.clone(), None
#     ref_dq, q.grad = q.grad.clone(), None
#     # triton implementation
#     tri_out = attention(q, k, v, tq, tk, tv, causal, sm_scale)[0].to(dtype)
#     tri_out.backward(dout)
#     tri_dv, v.grad = v.grad.clone(), None
#     tri_dk, k.grad = k.grad.clone(), None
#     tri_dq, q.grad = q.grad.clone(), None
#     # compare
#     rtol = 2e-2 if dtype == torch.bfloat16 else 0
#     torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
#     torch.testing.assert_close(ref_dq, tri_dq, atol=1e-2, rtol=rtol / 2)
#     torch.testing.assert_close(ref_dk, tri_dk, atol=1e-2, rtol=rtol / 2)
#     torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=rtol)


# def test_fwd_bwd():
#     for shape in [(1, 2, 1024, 64), (1, 2, 999, 64)]:
#         for causal in [True, False]:
#             for dtype in [torch.float16, torch.bfloat16]:
#                 _test_fwd_bwd(*shape, causal, dtype)
#                 print(f"Shape={shape}, Causal={causal}, Dtype={dtype} Passed (SA fwd/bwd).")


# def _test_jvp(Z, H, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
#     torch.manual_seed(20)
#     q = torch.empty((Z, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
#     k = torch.empty((Z, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
#     v = torch.empty((Z, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
#     tq = torch.empty((Z, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
#     tk = torch.empty((Z, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
#     tv = torch.empty((Z, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
#     sm_scale = 0.5

#     def naive_attention(q, k, v):
#         # reference implementation
#         M = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=DEVICE))
#         p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
#         if causal:
#             p[:, :, M == 0] = float("-inf")
#         p = torch.softmax(p.float(), dim=-1).to(dtype)
#         ref_out = torch.matmul(p, v)
#         return ref_out

#     _, ref_tout = torch.func.jvp(naive_attention, (q, k, v), (tq, tk, tv))
#     # triton implementation
#     tri_tout = attention(q, k, v, tq, tk, tv, causal, sm_scale)[1].to(dtype)
#     # compare
#     torch.testing.assert_close(ref_tout, tri_tout, atol=1e-2, rtol=1e-2)


# def test_jvp():
#     for shape in [(1, 2, 1024, 64), (1, 2, 999, 64)]:
#         for causal in [True, False]:
#             for dtype in [torch.float16, torch.bfloat16]:
#                 _test_jvp(*shape, causal, dtype)
#                 print(f"Shape={shape}, Causal={causal}, Dtype={dtype} Passed (SA JVP).")


# BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
# # vary seq length for fixed head and batch=4
# configs = []
# for mode in ["fwd", "bwd"]:
#     for causal in [True, False]:
#         if mode == "bwd" and not causal:
#             continue
#         configs.append(
#             triton.testing.Benchmark(
#                 x_names=["SEQ_LEN"],
#                 x_vals=[2**i for i in range(10, 15)],
#                 line_arg="provider",
#                 line_vals=["triton-fp16", "flash"],
#                 line_names=["Triton [FP16]", "FlashAttn-2"],
#                 styles=[("red", "-"), ("blue", "-"), ("green", "-")],
#                 ylabel="TFLOPS",
#                 plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
#                 args={
#                     "H": N_HEADS,
#                     "BATCH": BATCH,
#                     "HEAD_DIM": HEAD_DIM,
#                     "mode": mode,
#                     "causal": causal,
#                 },
#             )
#         )


# @triton.testing.perf_report(configs)
# def bench_flash_attention(BATCH, H, SEQ_LEN, HEAD_DIM, causal, mode, provider, device=DEVICE):
#     assert mode in ["fwd", "bwd"]
#     dtype = torch.float16
#     if "triton" in provider:
#         q = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
#         k = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
#         v = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
#         tq = torch.zeros_like(q)
#         tk = torch.zeros_like(k)
#         tv = torch.zeros_like(v)
#         sm_scale = 1.3
#         fn = lambda: attention(q, k, v, tq, tk, tv, causal, sm_scale)[0]
#         if mode == "bwd":
#             o = fn()
#             do = torch.randn_like(o)
#             fn = lambda: o.backward(do, retain_graph=True)
#         ms = triton.testing.do_bench(fn)
#     if provider == "flash":
#         from flash_attn.flash_attn_interface import flash_attn_func

#         q = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
#         k = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
#         v = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
#         fn = lambda: flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal)
#         if mode == "bwd":
#             o = fn()
#             do = torch.randn_like(o)
#             fn = lambda: o.backward(do, retain_graph=True)
#         ms = triton.testing.do_bench(fn)
#     # there are 2 matmuls in the forward pass
#     flops_per_matmul = 2.0 * BATCH * H * SEQ_LEN * SEQ_LEN * HEAD_DIM
#     total_flops = 2 * flops_per_matmul
#     if causal:
#         total_flops *= 0.5
#     if mode == "bwd":
#         # there are 5 matmuls in the backward pass
#         total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
#     elif "triton" in provider:
#         # there are 6 matmuls in the forward pass with JVP computation
#         total_flops *= 3
#     return total_flops * 1e-12 / (ms * 1e-3)


# def _test_fwd_bwd_ca(Z, H, SEQ_LEN_Q, SEQ_LEN_KV, HEAD_DIM, dtype=torch.float16):
#     torch.manual_seed(20)
#     q = torch.empty((Z, H, SEQ_LEN_Q, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
#     k = (
#         torch.empty((Z, H, SEQ_LEN_KV, HEAD_DIM), dtype=dtype, device=DEVICE)
#         .normal_(mean=0.0, std=0.5)
#         .requires_grad_()
#     )
#     v = (
#         torch.empty((Z, H, SEQ_LEN_KV, HEAD_DIM), dtype=dtype, device=DEVICE)
#         .normal_(mean=0.0, std=0.5)
#         .requires_grad_()
#     )
#     tq = torch.zeros_like(q)
#     tk = torch.zeros_like(k)
#     tv = torch.zeros_like(v)
#     sm_scale = 0.5
#     dout = torch.randn((Z, H, SEQ_LEN_Q, HEAD_DIM), device=q.device, dtype=q.dtype)
#     # reference implementation
#     p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
#     p = torch.softmax(p.float(), dim=-1).to(dtype)
#     ref_out = torch.matmul(p, v)
#     ref_out.backward(dout)
#     ref_dv, v.grad = v.grad.clone(), None
#     ref_dk, k.grad = k.grad.clone(), None
#     ref_dq, q.grad = q.grad.clone(), None
#     # triton implementation
#     tri_out = attention(q, k, v, tq, tk, tv, False, sm_scale)[0].to(dtype)
#     tri_out.backward(dout)
#     tri_dv, v.grad = v.grad.clone(), None
#     tri_dk, k.grad = k.grad.clone(), None
#     tri_dq, q.grad = q.grad.clone(), None
#     # compare
#     atol = 2e-2 if dtype == torch.bfloat16 else 1e-2
#     rtol = 2e-2 if dtype == torch.bfloat16 else 0
#     torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
#     torch.testing.assert_close(ref_dq, tri_dq, atol=atol, rtol=rtol / 2)
#     torch.testing.assert_close(ref_dk, tri_dk, atol=atol, rtol=rtol / 2)
#     torch.testing.assert_close(ref_dv, tri_dv, atol=atol, rtol=rtol)


# def test_fwd_bwd_ca():
#     for shape in [(1, 2, 256, 1024, 128), (1, 2, 1024, 256, 128), (1, 2, 1024, 512, 64), (1, 2, 1000, 515, 64)]:
#         for dtype in [torch.float16, torch.bfloat16]:
#             _test_fwd_bwd_ca(*shape, dtype)
#             print(f"Shape={shape}, Dtype={dtype} Passed (CA fwd/bwd with the same headdim).")


# def _test_jvp_ca(Z, H, SEQ_LEN_Q, SEQ_LEN_KV, HEAD_DIM_QK, HEAD_DIM_V, dtype=torch.float16):
#     torch.manual_seed(20)
#     q = (
#         torch.empty((Z, H, SEQ_LEN_Q, HEAD_DIM_QK), dtype=dtype, device=DEVICE)
#         .normal_(mean=0.0, std=0.5)
#         .requires_grad_()
#     )
#     k = (
#         torch.empty((Z, H, SEQ_LEN_KV, HEAD_DIM_QK), dtype=dtype, device=DEVICE)
#         .normal_(mean=0.0, std=0.5)
#         .requires_grad_()
#     )
#     v = (
#         torch.empty((Z, H, SEQ_LEN_KV, HEAD_DIM_V), dtype=dtype, device=DEVICE)
#         .normal_(mean=0.0, std=0.5)
#         .requires_grad_()
#     )
#     tq = torch.empty((Z, H, SEQ_LEN_Q, HEAD_DIM_QK), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
#     tk = torch.empty((Z, H, SEQ_LEN_KV, HEAD_DIM_QK), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
#     tv = torch.empty((Z, H, SEQ_LEN_KV, HEAD_DIM_V), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
#     sm_scale = 0.5

#     def naive_attention(q, k, v):
#         # reference implementation
#         p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
#         p = torch.softmax(p.float(), dim=-1).to(dtype)
#         ref_out = torch.matmul(p, v)
#         return ref_out

#     ref_out, ref_tout = torch.func.jvp(naive_attention, (q, k, v), (tq, tk, tv))
#     # triton implementation
#     tri_out, tri_tout = attention(q, k, v, tq, tk, tv, False, sm_scale)
#     # compare
#     atol = 2e-2 if dtype == torch.bfloat16 else 1e-2
#     torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
#     torch.testing.assert_close(ref_tout, tri_tout, atol=atol, rtol=1e-2)


# def test_jvp_ca():
#     for shape in [
#         (1, 2, 256, 1024, 64, 128),
#         (1, 2, 1000, 15, 128, 32),
#         (1, 2, 512, 512, 16, 32),
#         (1, 2, 515, 999, 16, 32),
#     ]:
#         for dtype in [torch.float16, torch.bfloat16]:
#             _test_jvp_ca(*shape, dtype)
#             print(f"Shape={shape}, Dtype={dtype} Passed (CA fwd/JVP with different headdim).")


# if __name__ == "__main__":
#     test_fwd_bwd()
#     test_jvp()
#     # only works on post-Ampere GPUs right now
#     bench_flash_attention.run(save_path=".", print_data=True)
#     test_fwd_bwd_ca()
#     test_jvp_ca()
