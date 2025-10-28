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

# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import time
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from megatron.core import parallel_state

from cosmos_predict2._src.imaginaire.flags import INTERNAL, SMOKE
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.distributed import broadcast, get_rank, sync_model_states
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.tokenizers.interface import VideoTokenizerInterface
from cosmos_predict2._src.predict2.tokenizers.wan2pt1_2d_plugins import plugin_mount
from cosmos_predict2._src.predict2.utils.tokenizer_benchmarking import BenchmarkTimes

__all__ = [
    "WanVAE",
]

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        # conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        # init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[: c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2 :, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim, dropout)
        )

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1)
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0], dropout)
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        temporal_window=4,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.temporal_window = temporal_window
        # modules
        self.encoder = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale, clear_encoder_cache=True):
        if clear_encoder_cache:
            self.clear_cache()
        # cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // self.temporal_window
        # 对encode输入的x，按时间拆分为1、self.temporal_stride、self.temporal_stride、self.temporal_window....
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self._i0_encode(x)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + self.temporal_window * (i - 1) : 1 + self.temporal_window * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        if (t - 1) % self.temporal_window:
            self._enc_conv_idx = [0]
            out_ = self.encoder(
                x[:, :, 1 + self.temporal_window * (iter_ - 1) :, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        if clear_encoder_cache:
            self.clear_cache()
        return mu

    @torch.compiler.disable
    def _i0_encode(self, x):
        """
        If enabled torch.compile uses significantly more memory for this step, so we disable it
        """
        out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        return out

    @torch.compiler.disable
    def _i0_decode(self, x):
        return self.decoder(x[:, :, 0:1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)

    def decode(self, z, scale, clear_decoder_cache=True):
        if clear_decoder_cache:
            self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self._i0_decode(x)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        if clear_decoder_cache:
            self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(
    pretrained_path=None,
    z_dim=None,
    device="cpu",
    s3_credential_path: str = "credentials/s3_training.secret",
    load_mean_std=False,
    mean_std_path: str = "s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/Wan2.1_VAE.pth",
    **kwargs,
):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    cfg.update(**kwargs)

    # init model
    with torch.device("meta"):
        model = WanVAE_(**cfg)

    if SMOKE or pretrained_path is None:
        model.to_empty(device=device)
        if load_mean_std:
            img_mean, img_std = torch.randn(1, 16, 1, 1, 1, device=device), torch.randn(1, 16, 1, 1, 1, device=device)
            video_mean, video_std = (
                torch.randn(1, 16, 32, 1, 1, device=device),
                torch.randn(1, 16, 32, 1, 1, device=device),
            )
    else:
        if get_rank() == 0:
            if not INTERNAL:
                from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path

                pretrained_path = get_checkpoint_path(pretrained_path)
            if pretrained_path.startswith("s3://"):
                backend_key = "wan2pt1_vae"
                easy_io.set_s3_backend(
                    key=backend_key,
                    backend_args={
                        "backend": "s3",
                        "s3_credential_path": s3_credential_path,
                    },
                )
            else:
                backend_key = None

            ckpt = easy_io.load(
                pretrained_path,
                backend_key=backend_key,
                map_location=device,
            )
            if load_mean_std:
                img_mean_std = mean_std_path.replace("mean_std.pt", "images_mean_std.pt")
                video_mean_std = mean_std_path.replace("mean_std.pt", "video_mean_std.pt")
                if not INTERNAL:
                    from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path

                    img_mean_std = get_checkpoint_path(img_mean_std)
                    video_mean_std = get_checkpoint_path(video_mean_std)
                img_mean, img_std = easy_io.load(img_mean_std, backend_key=backend_key, map_location=device)
                video_mean, video_std = easy_io.load(video_mean_std, backend_key=backend_key, map_location=device)
                img_mean = img_mean.reshape(1, 16, 1, 1, 1)
                img_std = img_std.reshape(1, 16, 1, 1, 1)
                video_mean = video_mean.reshape(1, 16, 32, 1, 1)
                video_std = video_std.reshape(1, 16, 32, 1, 1)

            # load checkpoint
            log.info(f"loading {pretrained_path}")
            model.load_state_dict(ckpt, assign=True)
        else:
            model.to_empty(device=device)
            if load_mean_std:
                img_mean, img_std = (
                    torch.randn(1, 16, 1, 1, 1, device=device),
                    torch.randn(1, 16, 1, 1, 1, device=device),
                )
                video_mean, video_std = (
                    torch.randn(1, 16, 32, 1, 1, device=device),
                    torch.randn(1, 16, 32, 1, 1, device=device),
                )
    sync_model_states(model)

    if load_mean_std:
        log.info("broadcast mean and std for wan2pt1")
        broadcast(img_mean, 0)
        broadcast(img_std, 0)
        broadcast(video_mean, 0)
        broadcast(video_std, 0)
        return model, img_mean, img_std, video_mean, video_std

    return (
        model,
        torch.zeros(1, 1, 1, 1, 1, device=device),
        torch.ones(1, 1, 1, 1, 1, device=device),
        torch.zeros(1, 1, 50, 1, 1, device=device),
        torch.ones(1, 1, 50, 1, 1, device=device),
    )


class WanVAE:
    def __init__(
        self,
        z_dim=16,
        vae_pth="s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/Wan2.1_VAE.pth",
        s3_credential_path: str = "credentials/s3_training.secret",
        load_mean_std=False,
        mean_std_path: str = "s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/mean_std.pt",
        dtype=torch.bfloat16,
        device="cuda",
        is_amp=True,
        benchmark: bool = False,
        temporal_window: int = 4,
        is_parallel: bool = False,
        cp_grid_shape: Optional[tuple[int, int]] = None,
    ):
        self.dtype = dtype
        self.device = device
        self.benchmark = benchmark
        self.temporal_window = temporal_window
        self.is_parallel = is_parallel
        self.cp_grid_shape = cp_grid_shape
        self.context_parallel_enabled = False
        self.cp_group_initialized = False

        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model, self.img_mean, self.img_std, self.video_mean, self.video_std = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
            s3_credential_path=s3_credential_path,
            load_mean_std=load_mean_std,
            mean_std_path=mean_std_path,
            device=device,
            temporal_window=temporal_window,
        )

        if is_parallel:
            cp_group = None
            if parallel_state.is_initialized():
                cp_group = parallel_state.get_context_parallel_group()
                if cp_grid_shape is None:
                    cp_grid_shape = (1, cp_group.size())
            else:
                assert False, "is_parallel set, but context parallelism is initialized"

            self._initialize_context_parallel(cp_group, cp_grid_shape)

        self.model = self.model.eval().requires_grad_(False)
        self.is_amp = is_amp
        if not is_amp:
            self.model = self.model.to(dtype=dtype)
            self.context = nullcontext()
        else:
            self.context = torch.amp.autocast("cuda", dtype=dtype)

    def count_param(self):
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def encode(self, videos, clear_encoder_cache=True):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        if self.is_parallel:
            if self._is_image_batch(videos):
                self._disable_context_parallel()
            else:
                # Latents are concatenated before attention so we won't need to gather chunks after execution
                try:
                    videos = self._broadcast_split_for_model_parallelsim(videos)
                    self._enable_context_parallel()
                except ValueError as e:
                    log.warning(str(e))
                    self._disable_context_parallel()
        if self.benchmark:
            torch.cuda.synchronize()
            benchmark_times = BenchmarkTimes()
            total_time = time.perf_counter()
        in_dtype = videos.dtype
        with self.context:
            if not self.is_amp:
                videos = videos.to(self.dtype)
            if self.benchmark:
                torch.cuda.synchronize()
                model_time = time.perf_counter()
            latent = self.model.encode(videos, self.scale, clear_encoder_cache)
            if self.benchmark:
                torch.cuda.synchronize()
                benchmark_times.model_invocation = time.perf_counter() - model_time
        latent = latent.to(in_dtype)
        if self.benchmark:
            torch.cuda.synchronize()
            benchmark_times.total = time.perf_counter() - total_time
            return latent, benchmark_times
        return latent

    @torch.no_grad()
    def decode(self, zs, clear_decoder_cache=True):
        if self.benchmark:
            torch.cuda.synchronize()
            benchmark_times = BenchmarkTimes()
            total_time = time.perf_counter()
        if self.is_parallel:
            if self._is_image_batch(zs):
                self._disable_context_parallel()
            else:
                # Make sure height and width divisible by CP factors
                can_apply_cp = (zs.shape[3] % self.cp_grid_shape[0] == 0) and (zs.shape[4] % self.cp_grid_shape[1] == 0)
                if not can_apply_cp:
                    log.warning(
                        f"For parallel encoding with grid_shape {self.cp_grid_shape} latent height should be divisible by grid_shape[0], got {zs.shape[3]} / {self.cp_grid_shape[0]} and width should be divisible by grid_shape[1], got {zs.shape[4]} / {self.cp_grid_shape[1]}, falling back to non CP"
                    )
                    self._disable_context_parallel()
                else:
                    self._enable_context_parallel()
        in_dtype = zs.dtype
        with self.context:
            if not self.is_amp:
                zs = zs.to(self.dtype)
            if self.benchmark:
                torch.cuda.synchronize()
                model_time = time.perf_counter()
            video_recon = self.model.decode(zs, self.scale, clear_decoder_cache)
            if self.benchmark:
                torch.cuda.synchronize()
                benchmark_times.model_invocation = time.perf_counter() - model_time
        video_recon = video_recon.to(in_dtype)
        if self.is_parallel and self.context_parallel_enabled:
            # Decoder splits tensors into CP chunks after attention (it is assumed all ranks in CP group have same data before execution), so we only need to gather at the end
            video_recon = self._cat_outputs_cp(video_recon)
        if self.benchmark:
            torch.cuda.synchronize()
            benchmark_times.total = time.perf_counter() - total_time
            return video_recon, benchmark_times
        return video_recon

    @property
    def spatial_compression_factor(self):
        return 8

    @property
    def temporal_compression_factor(self):
        return 4

    @property
    def _cp_dim(self):
        return 3

    def _broadcast_split_for_model_parallelsim(self, state: torch.Tensor) -> torch.Tensor:
        # All ranks from CP group get different data to encode, later when data is split before calling `compute_loss_with_epsilon_and_sigma`, they get data broadcasted from min rank in group
        # So we have to broadcast data now
        assert len(state.shape) == 5, "State should be of shape BCTHW"
        cp_rows, cp_cols = self.cp_grid_shape
        can_cp_be_applied_to_shape = (
            state.shape[3] % (cp_rows * self.spatial_compression_factor) == 0
            and state.shape[4] % (cp_cols * self.spatial_compression_factor) == 0
        )

        if not can_cp_be_applied_to_shape:
            raise ValueError(
                f"For parallel encoding with grid_shape {self.cp_grid_shape} height should be divisible by compression_factor*grid_shape[0], got {state.shape[3]} / ({self.cp_grid_shape[0]} * {self.spatial_compression_factor}) and width should be divisible by compression_factor*grid_shape[1], got {state.shape[4]} / ({self.cp_grid_shape[1]} * {self.spatial_compression_factor}), falling back to non CP"
            )

        # distributed.broadcast doesn't work with torch.export so we use distributed.all_gather
        state = state.contiguous()
        state_list = [torch.zeros_like(state) for _ in range(cp_rows * cp_cols)]
        distributed.all_gather(state_list, state, group=self.cp_group)
        state = state_list[0]
        # state = context_parallel.broadcast(state.contiguous(), self.cp_group)

        chunk_h = state.shape[3] // cp_rows
        chunk_w = state.shape[4] // cp_cols
        group_rank = distributed.get_rank(group=self.cp_group)

        row_id = group_rank // cp_cols
        col_id = group_rank % cp_cols

        return state[:, :, :, row_id * chunk_h : (row_id + 1) * chunk_h, col_id * chunk_w : (col_id + 1) * chunk_w]

    def _cat_outputs_cp(self, local_video_recon: torch.Tensor):
        video_recon_chunks = [torch.zeros_like(local_video_recon) for _ in range(self.cp_group_size)]
        distributed.all_gather(video_recon_chunks, local_video_recon, group=self.cp_group)

        # Concatenate chunks vertically then horizontaly
        video_recon = torch.cat(
            [torch.cat(video_recon_chunks[c :: self.cp_grid_shape[1]], dim=3) for c in range(self.cp_grid_shape[1])],
            dim=4,
        )

        return video_recon

    def _enable_context_parallel(self):
        self.context_parallel_enabled = True
        for _, plugin_list in self.plugins.items():
            for _, plugin in plugin_list.items():
                plugin.set_enable(True)

    def _disable_context_parallel(self):
        self.context_parallel_enabled = False
        for _, plugin_list in self.plugins.items():
            for _, plugin in plugin_list.items():
                plugin.set_enable(False)

    def _is_image_batch(self, x: torch.Tensor) -> bool:
        assert len(x.shape) == 5, "Expected tensor's shape to be BCTHW"
        return x.shape[2] == 1

    def _initialize_context_parallel(self, cp_group: distributed.ProcessGroup, cp_grid_shape) -> None:
        assert self.cp_group_initialized is False
        self.is_parallel = True
        self.cp_grid_shape = cp_grid_shape
        self.context_parallel_enabled = False
        self.cp_group = cp_group

        self.cp_group_size = len(distributed.get_process_group_ranks(self.cp_group))
        self.plugins: dict = plugin_mount(self.model, self.cp_group, cp_grid_shape)
        self._enable_context_parallel()
        log.info(f"Enabled CP with grid_shape: {cp_grid_shape} for Wan2.1 tokenizer")


class Wan2pt1VAEInterface(VideoTokenizerInterface):
    def __init__(self, chunk_duration: int = 81, load_mean_std=False, **kwargs):
        self.keep_decoder_cache = kwargs.get("keep_decoder_cache", False)
        self.keep_encoder_cache = kwargs.get("keep_encoder_cache", False)
        self.model = WanVAE(
            dtype=torch.bfloat16,
            is_amp=False,
            load_mean_std=load_mean_std,
            vae_pth=kwargs.get(
                "vae_pth",
                "s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/Wan2.1_VAE.pth",
            ),
            s3_credential_path=kwargs.get("s3_credential_path", "credentials/s3_training.secret"),
            temporal_window=kwargs.get("temporal_window", 4),
            is_parallel=kwargs.get("is_parallel", False),
            cp_grid_shape=kwargs.get("cp_grid_shape", None),
        )
        del kwargs
        self.chunk_duration = chunk_duration
        self.cp_initialized = False

    def initialize_context_parallel(self, cp_group: distributed.ProcessGroup, cp_grid_shape: tuple[int, int]) -> None:
        assert self.cp_initialized is False
        self.cp_initialized = True
        self.model._initialize_context_parallel(cp_group, cp_grid_shape)

    @property
    def dtype(self):
        return self.model.dtype

    def reset_dtype(self):
        pass

    def clear_cache(self):
        """Clear the feature cache for both encoder and decoder."""
        self.model.model.clear_cache()

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        latents = self.model.encode(state, clear_encoder_cache=not self.keep_encoder_cache)
        num_frames = latents.shape[2]
        if num_frames == 1:
            return (latents - self.model.img_mean.type_as(latents)) / self.model.img_std.type_as(latents)
        else:
            return (latents - self.model.video_mean[:, :, :num_frames].type_as(latents)) / self.model.video_std[
                :, :, :num_frames
            ].type_as(latents)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        num_frames = latent.shape[2]
        if num_frames == 1:
            recon = self.model.decode(
                ((latent * self.model.img_std.type_as(latent)) + self.model.img_mean.type_as(latent)).contiguous()
            )
        else:
            recon = self.model.decode(
                (
                    (latent * self.model.video_std[:, :, :num_frames].type_as(latent))
                    + self.model.video_mean[:, :, :num_frames].type_as(latent)
                ).contiguous()
            )

        if isinstance(recon, list):
            # torch.export makes batch_size=1 to be returned as list so we take first element and create batch dimension back
            assert len(recon) == 1, "Assuming batch_size=1 was used"
            recon = recon[0].unsqueeze(0)
        return recon

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return 1 + (num_pixel_frames - 1) // 4

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return (num_latent_frames - 1) * 4 + 1

    @property
    def spatial_compression_factor(self):
        return 8

    @property
    def temporal_compression_factor(self):
        return 4

    @property
    def pixel_chunk_duration(self):
        return self.chunk_duration

    @property
    def latent_chunk_duration(self):
        return self.get_latent_num_frames(self.chunk_duration)

    @property
    def latent_ch(self):
        return 16

    @property
    def spatial_resolution(self):
        return 512

    @property
    def name(self):
        return "wan2pt1_tokenizer"
