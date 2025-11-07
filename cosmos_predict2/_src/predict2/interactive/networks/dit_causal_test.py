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

# torchrun --nproc_per_node=4 -m pytest -s cosmos_predict2/_src/predict2/interactive/networks/dit_causal_test.py --L1

from __future__ import annotations

import os

import pytest
import torch

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils.helper_test import RunIf
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.interactive.networks.blockmask import prepare_temporal_only_causal_blockmask
from cosmos_predict2._src.predict2.interactive.networks.dit_causal import (
    CausalAttention,
    CausalBlock,
    CausalDIT,
    CausalDITKVCache,
    KVContextConfig,
    VideoSeqPos,
)
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import CheckpointMode, MiniTrainDIT, SACConfig

try:
    import torch._dynamo as dynamo

    disable_dynamo = dynamo.disable
except Exception:  # pragma: no cover

    def disable_dynamo(fn):
        return fn


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)


@pytest.fixture()
def dit_cfg():
    return L(CausalDIT)(
        max_img_h=240,
        max_img_w=240,
        max_frames=128,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=256,
        num_blocks=2,
        num_heads=4,
        concat_padding_mask=True,
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True,
        adaln_lora_dim=256,
        extra_per_block_abs_pos_emb=True,
        rope_h_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
        rope_t_extrapolation_ratio=2.0,
        sac_config=L(SACConfig)(mode=CheckpointMode.NONE),
    )


@pytest.fixture()
def dit_cfg_non_causal():
    return L(MiniTrainDIT)(
        max_img_h=240,
        max_img_w=240,
        max_frames=128,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=256,
        num_blocks=2,
        num_heads=4,
        concat_padding_mask=True,
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True,
        adaln_lora_dim=256,
        extra_per_block_abs_pos_emb=True,
        rope_h_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
        rope_t_extrapolation_ratio=2.0,
    )


@RunIf(min_gpus=4)
@pytest.mark.skip
@pytest.mark.L1
def test_causal_dit_context_parallel(dit_cfg):
    # Auto-skip when not launched with torchrun or insufficient world size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 4:
        pytest.skip("context-parallel test requires torchrun with WORLD_SIZE>=4")

    # Import distributed-only dependencies lazily to allow non-CP tests to run with plain pytest
    import torch.distributed as dist
    from megatron.core import parallel_state
    from torch.nn.parallel import DistributedDataParallel as DDP

    import cosmos_predict2._src.imaginaire.utils.distributed
    from cosmos_predict2._src.imaginaire.utils.context_parallel import cat_outputs_cp, split_inputs_cp

    cosmos_predict2._src.imaginaire.utils.distributed.init()

    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    # First perform non-context parallel inference
    model = instantiate(dit_cfg).cuda().to(dtype=dtype)
    model_state_dict = model.state_dict()

    # Create dataset
    batch_size = 1
    t_length = 40
    shape = [t_length, 512, 1024]  # [T, H, W]
    noise_labels = (
        torch.randn(
            batch_size,
            t_length // 8,
        )
        .cuda()
        .to(dtype=dtype)
    )
    crossattn_emb = torch.randn(batch_size, 512, 1024).cuda().to(dtype=dtype)
    padding_mask = torch.zeros(batch_size, 1, shape[1] // 16, shape[2] // 16).cuda().to(dtype=dtype)
    x = torch.randn(batch_size, 16, shape[0] // 8, shape[1] // 16, shape[2] // 16).cuda().to(dtype=dtype)
    fps = torch.randint(size=(batch_size,), low=2, high=30).cuda().to(dtype=dtype)

    dist.broadcast(x, 0)
    dist.broadcast(noise_labels, 0)
    dist.broadcast(crossattn_emb, 0)
    dist.broadcast(fps, 0)
    dist.broadcast(padding_mask, 0)

    with torch.no_grad():
        model_output_non_cp = model(
            x_B_C_T_H_W=x,
            timesteps_B_T=noise_labels,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            data_type=DataType.VIDEO,
        )

        assert model_output_non_cp.shape == x.shape, f"Expected shape: {x.shape}, got: {model_output_non_cp.shape}"

    # Now perform context parallel inference
    del model
    parallel_state.initialize_model_parallel(context_parallel_size=4)
    process_group = parallel_state.get_context_parallel_group()
    model = instantiate(dit_cfg).cuda().to(dtype=dtype)
    model.load_state_dict(model_state_dict)
    model = DDP(model, process_group=process_group)
    model.module.enable_context_parallel(parallel_state.get_context_parallel_group())

    with torch.no_grad():
        # Split inputs for CP
        x_orig = torch.clone(x)
        x = split_inputs_cp(x, seq_dim=2, cp_group=process_group)
        noise_labels = split_inputs_cp(noise_labels, seq_dim=1, cp_group=process_group)

        model_output_cp = model(
            x_B_C_T_H_W=x,
            timesteps_B_T=noise_labels,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            data_type=DataType.VIDEO,
        )

        if model.module.is_context_parallel_enabled:
            model_output_cp = cat_outputs_cp(model_output_cp, seq_dim=2, cp_group=process_group)

        assert model_output_cp.shape == x_orig.shape, f"Expected shape: {x_orig.shape}, got: {model_output_cp.shape}"

    relative_error = torch.norm(model_output_non_cp - model_output_cp) / torch.norm(model_output_non_cp)
    assert relative_error < 5e-3, "Relative error between CP and non-CP outputs is too large"


@RunIf(min_gpus=1)
@pytest.mark.L1
@disable_dynamo
def test_non_causal_dit_output_shape(dit_cfg_non_causal):
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    model = instantiate(dit_cfg_non_causal).cuda().to(dtype=dtype)

    batch_size = 1
    t_length = 40
    shape = [t_length, 512, 1024]  # [T, H, W]
    noise_labels = (
        torch.randn(
            batch_size,
            t_length // 8,
        )
        .cuda()
        .to(dtype=dtype)
    )
    crossattn_emb = torch.randn(batch_size, 512, 1024).cuda().to(dtype=dtype)
    padding_mask = torch.zeros(batch_size, 1, shape[1] // 16, shape[2] // 16).cuda().to(dtype=dtype)
    x = torch.randn(batch_size, 16, shape[0] // 8, shape[1] // 16, shape[2] // 16).cuda().to(dtype=dtype)
    fps = torch.randint(size=(batch_size,), low=2, high=30).cuda().to(dtype=dtype)

    with torch.no_grad():
        model_output = model(
            x_B_C_T_H_W=x,
            timesteps_B_T=noise_labels,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            data_type=DataType.VIDEO,
        )

        expected_shape = x.shape
        actual_shape = model_output.shape
        assert actual_shape == expected_shape, f"Expected shape: {expected_shape}, got: {actual_shape}"


@RunIf(min_gpus=1)
@pytest.mark.L1
@disable_dynamo
def test_equivalent_BT_vs_B_noise(dit_cfg):
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    net = instantiate(dit_cfg).cuda().to(dtype=dtype)
    net.eval()

    batch_size = 2
    t = 37
    x_B_C_T_H_W = torch.randn(batch_size, 16, t, 40, 40).cuda().to(dtype=dtype)
    noise_labels_B = torch.randn(batch_size).cuda().to(dtype=dtype)
    noise_labels_BT = noise_labels_B.unsqueeze(1).repeat(1, t)
    crossattn_emb_B_T_D = torch.randn(batch_size, 512, 1024).cuda().to(dtype=dtype)
    fps_B = torch.randint(size=(1,), low=2, high=30).cuda().float().repeat(batch_size)
    padding_mask_B_T_H_W = torch.zeros(batch_size, 1, 40, 40).cuda().to(dtype=dtype)

    output_BT = net(
        x_B_C_T_H_W,
        noise_labels_BT,
        crossattn_emb_B_T_D,
        fps=fps_B,
        padding_mask=padding_mask_B_T_H_W,
    )
    output_B = net(
        x_B_C_T_H_W,
        noise_labels_B,
        crossattn_emb_B_T_D,
        fps=fps_B,
        padding_mask=padding_mask_B_T_H_W,
    )
    torch.testing.assert_close(output_BT, output_B, rtol=1e-3, atol=1e-3)


@RunIf(min_gpus=1)
@pytest.mark.L1
@disable_dynamo
def test_causal_attention_equivalence_self_attn_blockmask():
    dtype = torch.float32  # use fp32 to avoid tolerance issues across backends
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    device = torch.device("cuda")
    b, s, d = 2, 12, 64
    heads = 4
    head_dim = d // heads

    attn_ulysses = CausalAttention(
        query_dim=d, context_dim=None, n_heads=heads, head_dim=head_dim, backend="ulysses"
    ).to(device)
    attn_flex = CausalAttention(
        query_dim=d, context_dim=None, n_heads=heads, head_dim=head_dim, backend="torch-flex"
    ).to(device)
    attn_ulysses_flex = CausalAttention(
        query_dim=d, context_dim=None, n_heads=heads, head_dim=head_dim, backend="ulysses-flex"
    ).to(device)

    attn_flex.load_state_dict(attn_ulysses.state_dict(), strict=True)
    attn_ulysses_flex.load_state_dict(attn_ulysses.state_dict(), strict=True)
    attn_ulysses.eval()
    attn_flex.eval()
    attn_ulysses_flex.eval()

    x = torch.randn(b, s, d, device=device, dtype=dtype)
    # Simulate Video DiT temporal-only causal mask with frame_seqlen=4
    block_mask = prepare_temporal_only_causal_blockmask(device=device, num_frames=s // 4, frame_seqlen=4)
    # Flex backends use BlockMask; Ulysses uses equivalent dense boolean mask
    attn_flex.mask = block_mask
    attn_ulysses_flex.mask = block_mask
    T = s // 4
    causal_mask = torch.tril(torch.ones(T, T, device=device), diagonal=0).bool()
    dense_mask = causal_mask.repeat_interleave(4, dim=0).repeat_interleave(4, dim=1)
    attn_ulysses.mask = dense_mask
    with torch.no_grad():
        out_u = attn_ulysses(x)
        out_f = attn_flex(x)
        out_uf = attn_ulysses_flex(x)

    torch.testing.assert_close(out_u, out_f, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_u, out_uf, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_f, out_uf, rtol=1e-3, atol=1e-3)


@RunIf(min_gpus=1)
@pytest.mark.L1
@disable_dynamo
def test_causal_block_equivalence():
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(2)

    device = torch.device("cuda")
    b, t, h, w = 1, 3, 2, 2
    x_dim = 64
    heads = 4
    context_dim = 64
    mlp_ratio = 2.0

    blk_u = CausalBlock(
        x_dim=x_dim,
        context_dim=context_dim,
        num_heads=heads,
        mlp_ratio=mlp_ratio,
        backend="ulysses",
        image_context_dim=None,
    ).to(device)

    blk_f = CausalBlock(
        x_dim=x_dim,
        context_dim=context_dim,
        num_heads=heads,
        mlp_ratio=mlp_ratio,
        backend="torch-flex",
        image_context_dim=None,
    ).to(device)
    blk_uf = CausalBlock(
        x_dim=x_dim,
        context_dim=context_dim,
        num_heads=heads,
        mlp_ratio=mlp_ratio,
        backend="ulysses-flex",
        image_context_dim=None,
    ).to(device)

    blk_f.load_state_dict(blk_u.state_dict(), strict=True)
    blk_uf.load_state_dict(blk_u.state_dict(), strict=True)
    blk_u.eval()
    blk_f.eval()
    blk_uf.eval()

    x = torch.randn(b, t, h, w, x_dim, device=device, dtype=dtype)
    emb = torch.randn(b, t, x_dim, device=device, dtype=dtype)
    cross = torch.randn(b, 5, context_dim, device=device, dtype=dtype)

    # Simulate Video DiT temporal-only causal mask with frame_seqlen=h*w
    frame_seqlen = max(4, h * w)
    block_mask = prepare_temporal_only_causal_blockmask(device=device, num_frames=t, frame_seqlen=frame_seqlen)
    # Flex backends use BlockMask; Ulysses uses equivalent dense boolean mask
    blk_f.self_attn.mask = block_mask
    blk_uf.self_attn.mask = block_mask
    tokens_per_frame = h * w
    causal_mask = torch.tril(torch.ones(t, t, device=device), diagonal=0).bool()
    dense_mask = causal_mask.repeat_interleave(tokens_per_frame, dim=0).repeat_interleave(tokens_per_frame, dim=1)
    blk_u.self_attn.mask = dense_mask

    with torch.no_grad():
        out_u = blk_u(x, emb, cross)
        out_f = blk_f(x, emb, cross)
        out_uf = blk_uf(x, emb, cross)

    torch.testing.assert_close(out_u, out_f, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_u, out_uf, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_f, out_uf, rtol=1e-3, atol=1e-3)


@RunIf(min_gpus=1)
@pytest.mark.L1
def test_dit_kvcache_ar_parity() -> None:
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    device = torch.device("cuda")

    # Small config for quick runtime
    B = 1
    T = 5
    H, W = 8, 8
    tokens_per_frame = H * W
    in_channels = 3
    out_channels = 3
    num_heads = 4
    head_dim = 16
    model_dim = num_heads * head_dim
    num_blocks = 2

    # Base model (no KV) to provide a common state_dict for consistent weights
    dit_base = (
        CausalDIT(
            max_img_h=H,
            max_img_w=W,
            max_frames=T,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_spatial=1,
            patch_temporal=1,
            model_channels=model_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            atten_backend="torch-flex",
            pos_emb_cls="rope3d",
            concat_padding_mask=True,
            rope_enable_fps_modulation=False,
        )
        .to(device)
        .to(dtype)
        .eval()
    )

    dit_kv1 = (
        CausalDITKVCache(
            max_img_h=H,
            max_img_w=W,
            max_frames=T,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_spatial=1,
            patch_temporal=1,
            model_channels=model_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            atten_backend="torch-flex",
            pos_emb_cls="rope3d",
            concat_padding_mask=True,
            rope_enable_fps_modulation=False,
        )
        .to(device)
        .to(dtype)
        .eval()
    )

    _ = dit_kv1.load_state_dict(dit_base.state_dict(), strict=True)

    # Inputs / conditioning
    x_B_C_T_H_W = torch.randn(B, in_channels, T, H, W, device=device, dtype=dtype)
    timesteps = torch.tensor([128.0], device=device, dtype=dtype)
    crossattn_emb = torch.randn(B, 32, 1024, device=device, dtype=dtype)
    padding_mask = torch.zeros(B, 1, H, W, device=device, dtype=dtype)

    cache_size = T * tokens_per_frame
    # Global absolute positions used to build per-frame VideoSeqPos
    full_video_pos = VideoSeqPos(T=T, H=H, W=W)

    # Denoising process across a small schedule per frame
    # Initialize identical noisy videos for two separate runs
    initial_noise = torch.randn(B, out_channels, T, H, W, device=device, dtype=dtype)
    c_t_scale = 1000.0
    rev_t = torch.tensor([1.0, 0.5, 0.25, 0.0], device=device, dtype=dtype)
    # Precompute scaled timestep tensors reused across loops
    rev_ts_scaled_1d = rev_t * c_t_scale  # [S]
    rev_ts_scaled_2d = [rev_ts_scaled_1d[i].view(1, 1) for i in range(len(rev_t))]

    # --- Run 1: CausalDIT (no KV) baseline ---
    with torch.no_grad():
        seq_video_base = initial_noise.clone()
        generated_frames = []
        for frame_idx in range(T):
            start_token = frame_idx * tokens_per_frame
            end_token = start_token + tokens_per_frame

            # Per-frame denoising steps
            for step_idx in range(len(rev_t) - 1):
                cur_t = rev_t[step_idx]
                next_t = rev_t[step_idx + 1]

                pred_video_full = dit_base(
                    x_B_C_T_H_W=seq_video_base[:, :, : frame_idx + 1],
                    timesteps_B_T=rev_ts_scaled_1d[step_idx].unsqueeze(0),
                    crossattn_emb=crossattn_emb,
                    padding_mask=padding_mask,
                )  # [B, C, frame_idx+1, H, W]
                pred_frame_kv = pred_video_full[:, :, -1:]

                # Update current frame (Euler-like)
                seq_frame = seq_video_base[:, :, frame_idx : frame_idx + 1]
                seq_x0 = seq_frame - cur_t * pred_frame_kv
                noise = torch.zeros_like(seq_x0)  # deterministic path
                vel = noise - seq_x0
                seq_video_base[:, :, frame_idx : frame_idx + 1] = seq_x0 + next_t * vel

            # After full denoising steps, collect the current frame for both paths
            generated_frames.append(seq_video_base[:, :, frame_idx : frame_idx + 1].clone())

    y_kv = torch.cat(generated_frames, dim=2)

    # --- Run 2: CausalDITKVCache ---
    with torch.no_grad():
        # reset KV caches for KVCache
        dit_kv1.make_it_kv_cache(batch_size=B, seq_len=cache_size, dtype=dtype, device=device)
        generated_frames1 = []
        for frame_idx in range(T):
            start_token = frame_idx * tokens_per_frame
            end_token = start_token + tokens_per_frame
            # Build per-frame absolute positions once
            cur_video_pos = VideoSeqPos(
                T=1,
                H=H,
                W=W,
                pos_h=full_video_pos.pos_h[start_token:end_token],
                pos_w=full_video_pos.pos_w[start_token:end_token],
                pos_t=full_video_pos.pos_t[start_token:end_token],
            )

            # KV context reused across steps (match rollout): don't store during denoising
            kv_cfg_denoise = KVContextConfig(start_idx=start_token, run_with_kv=True, store_kv=False)
            # Initialize single-frame state from initial noise (avoid slicing a full sequence)
            frame_seq = initial_noise[:, :, frame_idx : frame_idx + 1].clone()
            for step_idx in range(len(rev_t) - 1):
                cur_t = rev_t[step_idx]
                next_t = rev_t[step_idx + 1]

                # Embed current 1-frame input
                cur_frame_video = frame_seq
                x_1f_T_H_W_D, _, _ = dit_kv1.prepare_embedded_sequence(
                    cur_frame_video,
                    padding_mask=padding_mask,
                )
                cur_B_L_D = x_1f_T_H_W_D.reshape(B, tokens_per_frame, model_dim)

                pred_tokens = dit_kv1.forward_seq(
                    x_B_L_D=cur_B_L_D,
                    video_pos=cur_video_pos,
                    timesteps_B_T=rev_ts_scaled_2d[step_idx],
                    crossattn_emb=crossattn_emb,
                    kv_context_cfg=kv_cfg_denoise,
                )
                pred_B_T_H_W_O = pred_tokens.view(B, 1, H, W, out_channels)
                pred_frame_kv1 = dit_kv1.unpatchify(pred_B_T_H_W_O)

                seq_x01 = frame_seq - cur_t * pred_frame_kv1
                noise1 = torch.zeros_like(seq_x01)
                vel1 = noise1 - seq_x01
                frame_seq = seq_x01 + next_t * vel1

            # After finishing the frame, prefill KV cache for history
            cur_frame_video = frame_seq
            x_1f_T_H_W_D, _, _ = dit_kv1.prepare_embedded_sequence(
                cur_frame_video,
                padding_mask=padding_mask,
            )
            cur_B_L_D = x_1f_T_H_W_D.reshape(B, tokens_per_frame, model_dim)
            kv_cfg_prefill = KVContextConfig(start_idx=start_token, run_with_kv=True, store_kv=True)
            _ = dit_kv1.forward_seq(
                x_B_L_D=cur_B_L_D,
                video_pos=cur_video_pos,
                timesteps_B_T=rev_ts_scaled_2d[-1],
                crossattn_emb=crossattn_emb,
                kv_context_cfg=kv_cfg_prefill,
            )

            generated_frames1.append(frame_seq.clone())

    y_kv1 = torch.cat(generated_frames1, dim=2)

    assert y_kv.shape == y_kv1.shape
    torch.testing.assert_close(y_kv1, y_kv, rtol=1e-4, atol=1e-4)
