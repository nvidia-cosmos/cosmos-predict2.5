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

from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import wandb

from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import distributed, misc
from cosmos_predict2._src.imaginaire.utils.callback import Callback
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io


def _get_normal_quantile_bins():
    """Get predefined bins based on exp(N(0,1)) distribution quantiles"""
    # Using torch.special.ndtri (inverse of standard normal CDF)
    # and taking exponential to get exponentially spaced bins
    probs = torch.linspace(0.0, 1.0, 11)  # 11 points gives 10 bins
    points = torch.special.ndtri(probs)
    # Take exponential to get exponentially spaced bins
    points = torch.exp(points)
    # Replace extreme values at boundaries
    points[0] = points[1] / (points[2] / points[1])  # Extrapolate left boundary
    points[-1] = points[-2] * (points[-2] / points[-3])  # Extrapolate right boundary
    return points.numpy()


@dataclass
class _SigmaLossCache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sigma_list: List[torch.Tensor] = []
        self.loss_list: List[torch.Tensor] = []

    def add(self, sigma: torch.Tensor, loss: torch.Tensor):
        # Convert to bf16 and store on CPU
        self.sigma_list.append(sigma.detach().cpu().to(torch.bfloat16))
        self.loss_list.append(loss.detach().cpu().to(torch.bfloat16))

    def get_arrays(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        if not self.sigma_list:
            return torch.tensor([], dtype=torch.bfloat16), torch.tensor([], dtype=torch.bfloat16), None

        sigma_arr = torch.cat(self.sigma_list, dim=0)  # [B*N, T] or [B*N, 1]
        loss_arr = torch.cat(self.loss_list, dim=0)  # [B*N, T]

        # Handle broadcasting case where sigma is shape [B, 1]
        if sigma_arr.shape[-1] == 1 and loss_arr.shape[-1] > 1:
            sigma_arr = sigma_arr.expand(-1, loss_arr.shape[-1])

        num_frames = loss_arr.shape[-1] if len(loss_arr.shape) > 1 else 1

        assert sigma_arr.shape == loss_arr.shape, (sigma_arr.shape, loss_arr.shape)

        return sigma_arr, loss_arr, num_frames


class SigmaLossAnalysisPerFrame(Callback):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
        logging_viz_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.save_s3 = save_s3
        self.logging_iter_multipler = logging_iter_multipler
        assert logging_viz_iter_multipler % logging_iter_multipler == 0
        self.logging_viz_iter_multipler = logging_viz_iter_multipler
        self.name = self.__class__.__name__

        self.image_cache = _SigmaLossCache()
        self.video_cache = _SigmaLossCache()

    def _create_analysis_plots(
        self,
        sigma_arr: torch.Tensor,
        loss_arr: torch.Tensor,
        frame_idx: Optional[int] = None,  # [N]  # [N]
    ) -> Optional[wandb.Image]:
        if len(sigma_arr) == 0:
            return None

        # Convert to numpy for plotting
        sigma_np = sigma_arr.cpu().float().numpy()[:800]
        loss_np = loss_arr.cpu().float().numpy()[:800]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Get predefined bins based on normal distribution quantiles
        sigma_bins = _get_normal_quantile_bins()

        y_tick_min, y_tick_max = 0, 1.0
        # 2D histogram with exponential sigma bins and fixed [0,1] loss range
        loss_bins = np.linspace(y_tick_min, y_tick_max, 20)

        counts, xedges, yedges = np.histogram2d(sigma_np, loss_np, bins=(sigma_bins, loss_bins))
        if counts.max() < 0.1:
            return None

        # Plot heatmap with exponential scale colormap
        im = ax1.imshow(
            counts.T,
            origin="lower",
            aspect="auto",
            extent=[sigma_bins[0], sigma_bins[-1], y_tick_min, y_tick_max],
            norm=matplotlib.colors.LogNorm(vmin=1, vmax=counts.max()),
        )
        plt.colorbar(im, ax=ax1)

        # Set fixed loss ticks from 0 to 1
        yticks = np.linspace(y_tick_min, y_tick_max, 6)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([f"{y:.1f}" for y in yticks])

        ax1.set_xlabel("Sigma (Standard Normal Quantiles)")
        ax1.set_ylabel("Loss")
        title = "Sigma vs Loss Distribution"
        if frame_idx is not None:
            title += f" (Frame {frame_idx})"
        ax1.set_title(title)

        # Sigma histogram with loss statistics per bin
        hist_counts, _ = np.histogram(sigma_np, bins=sigma_bins)
        bin_indices = np.digitize(sigma_np, sigma_bins) - 1

        # Calculate statistics per bin
        n_bins = len(sigma_bins) - 1
        means = np.zeros(n_bins)
        stds = np.zeros(n_bins)
        for i in range(n_bins):
            bin_mask = bin_indices == i
            if bin_mask.any():
                means[i] = loss_np[bin_mask].mean()
                stds[i] = loss_np[bin_mask].std()
            else:
                means[i] = np.nan
                stds[i] = np.nan

        # Plot histogram
        bin_centers = (sigma_bins[:-1] + sigma_bins[1:]) / 2
        ax2.bar(bin_centers, hist_counts, width=np.diff(sigma_bins), alpha=0.3, align="center")

        # Plot loss statistics on twin axis
        ax2_twin = ax2.twinx()
        valid_mask = ~np.isnan(means)
        ax2_twin.errorbar(
            bin_centers[valid_mask], means[valid_mask], yerr=stds[valid_mask], color="red", fmt="o-", alpha=0.5
        )

        ax2.set_xlabel("Sigma (Standard Normal Quantiles)")
        ax2.set_ylabel("Count")
        ax2_twin.set_ylabel("Loss (mean ± std)")
        title = "Sigma Distribution with Loss Statistics"
        if frame_idx is not None:
            title += f" (Frame {frame_idx})"
        ax2.set_title(title)

        # Add grid for better readability
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)

        # Add quantile labels
        probs = torch.linspace(0.0, 1.0, 11)  # 10 points for 9 internal quantiles
        quantile_labels = [f"{p:.1%}" for p in probs]
        ax1.set_xticks(sigma_bins[1:-1])  # Skip boundary bins
        ax1.set_xticklabels(quantile_labels[1:-1], rotation=45)
        ax1.set_xscale("log")
        ax2.set_xticks(sigma_bins[1:-1])
        ax2.set_xticklabels(quantile_labels[1:-1], rotation=45)
        ax2.set_xscale("log")

        plt.tight_layout()
        fig_img = wandb.Image(fig)
        plt.close(fig)

        return fig_img

    def _process_frame_stats(self, sigma: torch.Tensor, loss: torch.Tensor, frame_idx: int) -> dict:
        """Calculate statistics for a specific frame"""
        return {
            "sigma_log_mean": float(sigma.log().mean()),
            "sigma_log_std": float(sigma.log().std()),
            "loss_mean": float(loss.mean()),
            "loss_std": float(loss.std()),
            "loss_min": float(loss.min()),
            "loss_max": float(loss.max()),
            "loss_median": float(loss.median()),
            "loss_q1": float(torch.quantile(loss.float(), 0.25)),
            "loss_q3": float(torch.quantile(loss.float(), 0.75)),
        }

    def _gather_and_save(self, cache: _SigmaLossCache, iteration: int, prefix: str, log_viz: bool = True) -> dict:
        info = {}

        # Gather data from all ranks
        local_sigma, local_loss, num_frames = cache.get_arrays()
        world_size = dist.get_world_size()

        if world_size > 1:
            # Gather sizes first
            local_size = torch.tensor([len(local_sigma)], dtype=torch.long, device="cuda")
            sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
            dist.all_gather(sizes, local_size)
            sizes = [s.item() for s in sizes]

            # Gather data
            max_size = max(sizes)
            if max_size > 0:
                # Move to GPU for gathering
                padded_sigma = torch.zeros(max_size, num_frames or 1, dtype=torch.bfloat16, device="cuda")
                padded_loss = torch.zeros(max_size, num_frames or 1, dtype=torch.bfloat16, device="cuda")

                if len(local_sigma) > 0:
                    padded_sigma[: len(local_sigma)] = local_sigma.cuda()
                    padded_loss[: len(local_loss)] = local_loss.cuda()

                all_sigma = [torch.zeros_like(padded_sigma) for _ in range(world_size)]
                all_loss = [torch.zeros_like(padded_loss) for _ in range(world_size)]

                dist.all_gather(all_sigma, padded_sigma)
                dist.all_gather(all_loss, padded_loss)

                if distributed.is_rank0():
                    # Combine data from all ranks
                    valid_sigma = []
                    valid_loss = []
                    for sigma, loss, size in zip(all_sigma, all_loss, sizes):
                        if size > 0:
                            valid_sigma.append(sigma[:size])
                            valid_loss.append(loss[:size])

                    if valid_sigma:
                        sigma_arr = torch.cat(valid_sigma)
                        loss_arr = torch.cat(valid_loss)

                        # Overall statistics
                        info[f"{prefix}/total_samples"] = sigma_arr.shape[0]

                        # Per-frame statistics
                        if num_frames and num_frames > 1:
                            for t in range(num_frames):
                                frame_stats = self._process_frame_stats(sigma_arr[:, t], loss_arr[:, t], t)
                                frame_prefix = f"{prefix}/frame_{t}"
                                info.update({f"{frame_prefix}/{k}": v for k, v in frame_stats.items()})

                                # Create per-frame visualization
                                if log_viz:
                                    fig_img = self._create_analysis_plots(sigma_arr[:, t], loss_arr[:, t], t)
                                    if fig_img is not None:
                                        info[f"{frame_prefix}/distribution_plot"] = fig_img
                        else:
                            # Single frame case (images or single-frame stats)
                            frame_stats = self._process_frame_stats(sigma_arr.squeeze(), loss_arr.squeeze(), None)
                            info.update({f"{prefix}/{k}": v for k, v in frame_stats.items()})

                            # Create visualization
                            if log_viz:
                                fig_img = self._create_analysis_plots(sigma_arr.squeeze(), loss_arr.squeeze())
                                if fig_img is not None:
                                    info[f"{prefix}/distribution_plot"] = fig_img

                        if self.save_s3:
                            save_data = {
                                "sigma": sigma_arr.cpu(),
                                "loss": loss_arr.cpu(),
                                "stats": {k: v for k, v in info.items() if not isinstance(v, wandb.Image)},
                            }
                            easy_io.dump(
                                save_data,
                                f"s3://rundir/{self.name}/{prefix}_Iter{iteration:09d}.pkl",
                            )

        cache.reset()
        return info

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ):
        sigma = output_batch["sigma"]
        loss_per_frame = output_batch["edm_loss_per_frame"]

        if model.is_image_batch(data_batch):
            self.image_cache.add(sigma, loss_per_frame)
        else:
            self.video_cache.add(sigma, loss_per_frame)

        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0:
            info = {}

            with misc.timer("sigma_loss_analysis"):
                log_viz = iteration % (self.config.trainer.logging_iter * self.logging_viz_iter_multipler) == 0
                # Process image data
                if len(self.image_cache.sigma_list) > 0:
                    info.update(self._gather_and_save(self.image_cache, iteration, "sigma_loss_image", log_viz=log_viz))

                # Process video data
                if len(self.video_cache.sigma_list) > 0:
                    info.update(self._gather_and_save(self.video_cache, iteration, "sigma_loss_video", log_viz=log_viz))

                if distributed.is_rank0() and info and wandb.run:
                    wandb.log(info, step=iteration)
