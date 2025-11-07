# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

from __future__ import annotations

import pytest
import torch

from cosmos_predict2._src.imaginaire.utils.helper_test import RunIf
from cosmos_predict2._src.predict2.interactive.networks.dit_action_causal import (
    ActionChunkCausalDIT,
    ActionChunkCausalDITKVCache,
)
from cosmos_predict2._src.predict2.interactive.networks.dit_causal import (
    KVContextConfig,
    VideoSeqPos,
)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)


@RunIf(min_gpus=1)
@pytest.mark.L1
def test_action_chunk_dit_kvcache_parity() -> None:
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
    action_dim = 80  # default in ActionChunkCausalDIT if unspecified

    # Base model (no KV wrapper) to provide a shared state_dict
    dit_base = (
        ActionChunkCausalDIT(
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
            action_dim=action_dim,
        )
        .to(device)
        .to(dtype)
        .eval()
    )

    dit_kv1 = (
        ActionChunkCausalDITKVCache(
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
            action_dim=action_dim,
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
    action_full = torch.randn(B, T, action_dim, device=device, dtype=dtype)

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

    # --- Run 1: Full ActionChunkCausalDIT ---
    with torch.no_grad():
        seq_video_full = initial_noise.clone()
        generated_frames_full = []
        for frame_idx in range(T):
            for step_idx in range(len(rev_t) - 1):
                cur_t = rev_t[step_idx]
                next_t = rev_t[step_idx + 1]

                pred_video_prefix = dit_base(
                    x_B_C_T_H_W=seq_video_full[:, :, : frame_idx + 1],
                    timesteps_B_T=rev_ts_scaled_1d[step_idx].unsqueeze(0),
                    crossattn_emb=crossattn_emb,
                    padding_mask=padding_mask,
                    action=action_full[:, : frame_idx + 1],
                )  # [B, C, <=T, H, W]
                pred_frame = pred_video_prefix[:, :, -1:]

                # Euler-like update on the current frame
                seq_frame = seq_video_full[:, :, frame_idx : frame_idx + 1]
                seq_x0 = seq_frame - cur_t * pred_frame
                noise = torch.zeros_like(seq_x0)
                vel = noise - seq_x0
                seq_video_full[:, :, frame_idx : frame_idx + 1] = seq_x0 + next_t * vel

            generated_frames_full.append(seq_video_full[:, :, frame_idx : frame_idx + 1].clone())

    y_full = torch.cat(generated_frames_full, dim=2)

    # --- Run 2: ActionChunkCausalDITKVCache (per-frame KV streaming) ---
    with torch.no_grad():
        dit_kv1.make_it_kv_cache(batch_size=B, seq_len=cache_size, dtype=dtype, device=device)
        generated_frames_kv1 = []
        for frame_idx in range(T):
            start_token = frame_idx * tokens_per_frame
            end_token = start_token + tokens_per_frame
            cur_video_pos = VideoSeqPos(
                T=1,
                H=H,
                W=W,
                pos_h=full_video_pos.pos_h[start_token:end_token],
                pos_w=full_video_pos.pos_w[start_token:end_token],
                pos_t=full_video_pos.pos_t[start_token:end_token],
            )

            # KV context reused across steps (do not store during denoising)
            kv_cfg_denoise = KVContextConfig(start_idx=start_token, run_with_kv=True, store_kv=False)
            # Initialize single-frame state from initial noise (avoid slicing a full sequence)
            frame_seq = initial_noise[:, :, frame_idx : frame_idx + 1].clone()
            for step_idx in range(len(rev_t) - 1):
                cur_t = rev_t[step_idx]
                next_t = rev_t[step_idx + 1]

                # Embed current 1-frame input with padding mask
                x_1f_T_H_W_D, _, _ = dit_kv1.prepare_embedded_sequence(
                    frame_seq,
                    padding_mask=padding_mask,
                )
                cur_B_L_D = x_1f_T_H_W_D.reshape(B, tokens_per_frame, model_dim)

                pred_tokens = dit_kv1.forward_seq(
                    x_B_L_D=cur_B_L_D,
                    video_pos=cur_video_pos,
                    timesteps_B_T=rev_ts_scaled_2d[step_idx],
                    crossattn_emb=crossattn_emb,
                    kv_context_cfg=kv_cfg_denoise,
                    action=action_full[:, frame_idx : frame_idx + 1],
                )
                pred_B_T_H_W_O = pred_tokens.view(B, 1, H, W, out_channels)
                pred_frame_kv1 = dit_kv1.unpatchify(pred_B_T_H_W_O)

                seq_x01 = frame_seq - cur_t * pred_frame_kv1
                noise1 = torch.zeros_like(seq_x01)
                vel1 = noise1 - seq_x01
                frame_seq = seq_x01 + next_t * vel1

            # After finishing the frame, prefill KV cache for history
            x_1f_T_H_W_D, _, _ = dit_kv1.prepare_embedded_sequence(
                frame_seq,
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
                action=action_full[:, frame_idx : frame_idx + 1],
            )

            generated_frames_kv1.append(frame_seq.clone())

    y_kv1 = torch.cat(generated_frames_kv1, dim=2)

    assert y_full.shape == y_kv1.shape
    torch.testing.assert_close(y_kv1, y_full, rtol=1e-4, atol=1e-4)
