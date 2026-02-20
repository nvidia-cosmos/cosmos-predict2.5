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
Cosmos Policy Wan2pt1 VAE with deterministic seeding support.

This module provides a complete wrapper around the wan2pt1 tokenizer to add
deterministic seeding without modifying the main branch files.
"""

import os
import sys
import time
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as distributed
from megatron.core import parallel_state

from cosmos_predict2._src.imaginaire.flags import INTERNAL, SMOKE
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.distributed import broadcast, get_rank, sync_model_states
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.cosmos_policy.utils.checkpoint_utils import resolve_checkpoint_path
from cosmos_predict2._src.predict2.tokenizers.interface import VideoTokenizerInterface
from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import WanVAE_ as BaseWanVAE_
from cosmos_predict2._src.predict2.tokenizers.wan2pt1_2d_plugins import plugin_mount
from cosmos_predict2._src.predict2.utils.tokenizer_benchmarking import BenchmarkTimes


class WanVAE_(BaseWanVAE_):
    """
    Extended WanVAE_ for Cosmos Policy with deterministic seeding support.

    Adds optional deterministic seeding in the encode method for reproducibility.
    """

    def encode(self, x, scale, clear_encoder_cache=True):
        """
        Extended encode with optional deterministic seeding for reproducibility.

        When DETERMINISTIC environment variable is set to "true", uses fixed seed
        for reproducible VAE encoding.
        """
        # NOTE (user): Deterministic seeding for reproducibility
        #                 WARNING: DO NOT ENABLE DURING TRAINING - messes up all random sampling!
        deterministic_enabled = os.environ.get("DETERMINISTIC", "").lower() == "true"
        if deterministic_enabled:
            # Check if we're in a training run by examining command line arguments
            cmd_args = " ".join(sys.argv).lower()
            is_training = (
                "scripts.train" in cmd_args
                or "scripts/train.py" in cmd_args
                or "/train.py" in cmd_args
                or "-m cosmos_predict2._src.predict2.cosmos_policy.scripts.train" in cmd_args
            )
            # Hard-fail to make sure we aren't in a training run with determinism enabled
            assert not is_training, (
                "DETERMINISTIC mode is enabled but this is a training run! "
                "Deterministic seeding breaks random sampling during training. "
                "Please unset the DETERMINISTIC environment variable before training."
            )
            # Proceed with deterministic seeding
            from cosmos_predict2._src.predict2.cosmos_policy.utils.utils import set_seed_everywhere

            seed = 7  # Arbitrary seed value
            set_seed_everywhere(seed)

        # Call parent implementation
        return super().encode(x, scale, clear_encoder_cache)


def _policy_video_vae(
    pretrained_path=None,
    z_dim=None,
    device="cpu",
    s3_credential_path: str = "credentials/s3_training.secret",
    load_mean_std=False,
    mean_std_path: str = "hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
    **kwargs,
):
    """
    Modified _video_vae that uses the policy WanVAE_ with deterministic seeding.

    This is a copy of the original _video_vae function, but instantiates our
    WanVAE_ subclass instead of the base one.
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

    # init model - use our WanVAE_ subclass
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

            resolved_path = resolve_checkpoint_path(pretrained_path)
            ckpt = easy_io.load(
                resolved_path,
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


class CosmosPolicyWanVAE:
    """
    Cosmos Policy WanVAE wrapper that uses deterministic WanVAE_.

    This is a complete reimplementation of WanVAE that uses our
    _policy_video_vae() function to instantiate the deterministic WanVAE_.
    """

    def __init__(
        self,
        z_dim=16,
        vae_pth="hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
        s3_credential_path: str = "credentials/s3_training.secret",
        load_mean_std=False,
        mean_std_path: str = "hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/mean_std.pt",
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

        # init model - use our _policy_video_vae function
        self.model, self.img_mean, self.img_std, self.video_mean, self.video_std = _policy_video_vae(
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
    """
    Cosmos Policy Wan2pt1VAE Interface with deterministic seeding support.

    Uses CosmosPolicyWanVAE which instantiates our deterministic WanVAE_ subclass.
    """

    def __init__(self, chunk_duration: int = 81, load_mean_std=False, **kwargs):
        self.keep_decoder_cache = kwargs.get("keep_decoder_cache", False)
        self.keep_encoder_cache = kwargs.get("keep_encoder_cache", False)
        # Use our CosmosPolicyWanVAE instead of the base WanVAE
        self.model = CosmosPolicyWanVAE(
            dtype=torch.bfloat16,
            is_amp=False,
            load_mean_std=load_mean_std,
            vae_pth=kwargs.get(
                "vae_pth",
                "hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
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
                ((latent * self.model.img_std.type_as(latent)) + self.model.img_mean.type_as(latent)).contiguous(),
                clear_decoder_cache=not self.keep_decoder_cache,
            )
        else:
            recon = self.model.decode(
                (
                    (latent * self.model.video_std[:, :, :num_frames].type_as(latent))
                    + self.model.video_mean[:, :, :num_frames].type_as(latent)
                ).contiguous(),
                clear_decoder_cache=not self.keep_decoder_cache,
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
        return "cosmos_policy_wan2pt1_tokenizer"
