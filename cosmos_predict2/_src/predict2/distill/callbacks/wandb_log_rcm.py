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

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.distributed as dist
import torch.utils.data
import wandb
from einops import rearrange

from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.distill.callbacks.wandb_log_base import WandBCallback as WandBCallbackBase


@dataclass
class _LossRecord:
    """Records and tracks various loss metrics during training."""

    # Initialize metrics with default values
    loss: torch.Tensor = 0
    nan_mask_gradient_ratio: torch.Tensor = 0  # More descriptive name for nan_mask_g_ratio
    nan_mask_model_ratio: torch.Tensor = 0  # More descriptive name for nan_mask_F_theta_ratio
    l2_df_dt: torch.Tensor = 0  # Original name as requested
    l2_df_dt_no_nan: torch.Tensor = 0  # L2 norm of df/dt with NaN values removed
    iteration_count: int = 0  # More descriptive name for iter_count

    def reset(self) -> None:
        """Reset all metrics to their default values."""
        self.loss = 0
        self.nan_mask_gradient_ratio = 0
        self.nan_mask_model_ratio = 0
        self.l2_df_dt = 0
        self.l2_df_dt_no_nan = 0
        self.iteration_count = 0

    def update(
        self,
        loss: torch.Tensor,
        nan_mask_gradient_ratio: torch.Tensor,
        nan_mask_model_ratio: torch.Tensor,
        l2_df_dt: torch.Tensor,
        l2_df_dt_no_nan: torch.Tensor,
    ) -> None:
        """Update the loss record with new values.

        Args:
            loss: The loss value to add
            nan_mask_gradient_ratio: NaN ratio in gradient mask
            nan_mask_model_ratio: NaN ratio in model mask
            l2_df_dt: L2 norm of df/dt
            l2_df_dt_no_nan: L2 norm of df/dt with NaN values removed
        """
        self.loss += loss.detach().float()
        self.nan_mask_gradient_ratio += nan_mask_gradient_ratio
        self.nan_mask_model_ratio += nan_mask_model_ratio
        self.l2_df_dt += l2_df_dt
        self.l2_df_dt_no_nan += l2_df_dt_no_nan
        self.iteration_count += 1

    def get_stats(self) -> Dict[str, float]:
        """Calculate and return statistics across all metrics.

        Returns:
            Dictionary containing averaged metrics
        """
        stats = {}

        if self.iteration_count > 0:
            # Calculate average for standard metrics
            metrics = {
                "loss": self.loss,
                "nan_mask_gradient_ratio": self.nan_mask_gradient_ratio,
                "nan_mask_model_ratio": self.nan_mask_model_ratio,
                "l2_df_dt": self.l2_df_dt,
                "l2_df_dt_no_nan": self.l2_df_dt_no_nan,
            }

            # Process each metric
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    avg_value = value / self.iteration_count
                    # Distribute across processes
                    dist.all_reduce(avg_value, op=dist.ReduceOp.AVG)
                    stats[name] = avg_value.item()
                else:
                    stats[name] = value / self.iteration_count
        else:
            # Default values if no iterations
            stats = {
                "loss": 0.0,
                "nan_mask_gradient_ratio": 0.0,
                "nan_mask_model_ratio": 0.0,
                "l2_df_dt": 0.0,
                "l2_df_dt_no_nan": 0.0,
            }

        # Reset after collecting stats
        self.reset()
        return stats


class WandbCallback(WandBCallbackBase):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.train_image_log = _LossRecord()
        self.train_video_log = _LossRecord()
        self.final_loss_log = _LossRecord()
        self.img_unstable_count = torch.zeros(1, device="cuda")
        self.video_unstable_count = torch.zeros(1, device="cuda")
        self.logging_iter_multiplier = logging_iter_multipler
        self.save_logging_iter_multiplier = save_logging_iter_multipler
        assert self.logging_iter_multiplier > 0, "logging_iter_multiplier should be greater than 0"
        self.save_s3 = save_s3
        self.wandb_extra_tag = f"@{self.logging_iter_multiplier}" if self.logging_iter_multiplier > 1 else ""
        self.name = "wandb_loss_log" + self.wandb_extra_tag

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        skip_update_due_to_unstable_loss = False
        if torch.isnan(loss) or torch.isinf(loss):
            skip_update_due_to_unstable_loss = True
            log.critical(
                f"Unstable loss {loss} at iteration {iteration} with is_image_batch: {model.is_image_batch(data_batch)}",
                rank0_only=False,
            )

        if not skip_update_due_to_unstable_loss:
            if "df_dt" in output_batch.keys():
                # Calculate l2_df_dt metric
                l2_df_dt_value = torch.pow(output_batch["df_dt"], 2).mean()

                # Get common metrics
                nan_mask_g_ratio = output_batch["nan_mask_g"].float().mean()
                nan_mask_F_theta_ratio = output_batch["nan_mask_F_theta"].float().mean()

                _df_dt = output_batch["df_dt"]
                # Create a mask for non-NaN values
                valid_mask = ~output_batch["nan_mask_g"]
                # Calculate mean only over non-NaN values
                if valid_mask.sum() > 0:
                    l2_df_dt_value_no_nan = torch.pow(_df_dt[valid_mask], 2).mean()
                else:
                    l2_df_dt_value_no_nan = torch.tensor(0.0, device=_df_dt.device, dtype=_df_dt.dtype)

                # Update the appropriate log based on batch type
                if model.is_image_batch(data_batch):
                    self.train_image_log.update(
                        loss.detach().float(),
                        nan_mask_g_ratio,
                        nan_mask_F_theta_ratio,
                        l2_df_dt_value,
                        l2_df_dt_value_no_nan,
                    )
                else:
                    self.train_video_log.update(
                        loss.detach().float(),
                        nan_mask_g_ratio,
                        nan_mask_F_theta_ratio,
                        l2_df_dt_value,
                        l2_df_dt_value_no_nan,
                    )

                # Always update the final loss log
                self.final_loss_log.update(
                    loss.detach().float(),
                    nan_mask_g_ratio,
                    nan_mask_F_theta_ratio,
                    l2_df_dt_value,
                    l2_df_dt_value_no_nan,
                )
            # Log DMD loss immediately at step end if provided by the model
            if "dmd_loss" in output_batch.keys():
                _dmd_loss_value = output_batch["dmd_loss"]
                if not isinstance(_dmd_loss_value, torch.Tensor):
                    _dmd_loss_value = torch.tensor(_dmd_loss_value, device="cuda", dtype=torch.float32)
                else:
                    _dmd_loss_value = _dmd_loss_value.detach().float()
                # Average across processes for consistency
                dist.all_reduce(_dmd_loss_value, op=dist.ReduceOp.AVG)
                if distributed.is_rank0():
                    wandb.log({f"train{self.wandb_extra_tag}/dmd_loss": _dmd_loss_value.item()}, step=iteration)
            # Log DMD loss immediately at step end if provided by the model
            if "dmd_loss_critic" in output_batch.keys():
                _dmd_loss_value = output_batch["dmd_loss_critic"]
                if not isinstance(_dmd_loss_value, torch.Tensor):
                    _dmd_loss_value = torch.tensor(_dmd_loss_value, device="cuda", dtype=torch.float32)
                else:
                    _dmd_loss_value = _dmd_loss_value.detach().float()
                # Average across processes for consistency
                dist.all_reduce(_dmd_loss_value, op=dist.ReduceOp.AVG)
                if distributed.is_rank0():
                    wandb.log({f"train{self.wandb_extra_tag}/dmd_loss_critic": _dmd_loss_value.item()}, step=iteration)
            if "dmd_loss_generator" in output_batch.keys():
                _dmd_loss_value = output_batch["dmd_loss_generator"]
                if not isinstance(_dmd_loss_value, torch.Tensor):
                    _dmd_loss_value = torch.tensor(_dmd_loss_value, device="cuda", dtype=torch.float32)
                else:
                    _dmd_loss_value = _dmd_loss_value.detach().float()
                # Average across processes for consistency
                dist.all_reduce(_dmd_loss_value, op=dist.ReduceOp.AVG)
                if distributed.is_rank0():
                    wandb.log(
                        {f"train{self.wandb_extra_tag}/dmd_loss_generator": _dmd_loss_value.item()}, step=iteration
                    )
        else:
            # Track unstable losses
            if model.is_image_batch(data_batch):
                self.img_unstable_count += 1
            else:
                self.video_unstable_count += 1

        # Log at specified intervals
        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multiplier) == 0:
            if self.logging_iter_multiplier > 1:
                timer_results = {}
            else:
                timer_results = self.trainer.training_timer.compute_average_results()

            # Get statistics from each loss record
            image_stats = self.train_image_log.get_stats()
            video_stats = self.train_video_log.get_stats()
            final_stats = self.final_loss_log.get_stats()

            # Reduce unstable counts across all processes
            dist.all_reduce(self.img_unstable_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.video_unstable_count, op=dist.ReduceOp.SUM)

            if distributed.is_rank0():
                # Create info dictionary for logging
                info = {f"timer/{key}": value for key, value in timer_results.items()}

                # Add image statistics
                for key, value in image_stats.items():
                    info[f"train{self.wandb_extra_tag}/image_{key}"] = value

                # Add video statistics
                for key, value in video_stats.items():
                    info[f"train{self.wandb_extra_tag}/video_{key}"] = value

                # Add final statistics
                for key, value in final_stats.items():
                    info[f"train{self.wandb_extra_tag}/{key}"] = value

                # Add unstable counts
                info.update(
                    {
                        f"train{self.wandb_extra_tag}/img_unstable_count": self.img_unstable_count.item(),
                        f"train{self.wandb_extra_tag}/video_unstable_count": self.video_unstable_count.item(),
                        "iteration": iteration,
                        "sample_counter": getattr(self.trainer, "sample_counter", iteration),
                    }
                )

                # Save to S3 if enabled
                if self.save_s3:
                    save_interval = (
                        self.config.trainer.logging_iter
                        * self.logging_iter_multiplier
                        * self.save_logging_iter_multiplier
                    )
                    if iteration % save_interval == 0:
                        easy_io.dump(
                            info,
                            f"s3://rundir/{self.name}/Train_Iter{iteration:09d}.json",
                        )

                if wandb:
                    wandb.log(info, step=iteration)
            if self.logging_iter_multiplier == 1:
                self.trainer.training_timer.reset()

            # reset unstable count
            self.img_unstable_count.zero_()
            self.video_unstable_count.zero_()

    @torch.no_grad
    def log_to_wandb(
        self,
        model,
        data_tensor: torch.Tensor,
        wandb_key: str,
        iteration: int,
        n_viz_sample: int = 3,
        fps: int = 8,
        caption: str = None,
    ):
        """
        Logs image or video data to wandb from a [b, c, t, h, w] tensor.

        It normalizes the tensor, selects the first n_viz_sample from the batch (b)
        dimension, and arranges them into a single row grid (n rows, n_viz_sample columns).
        Logs as wandb.Image if t=1, otherwise logs as wandb.Video.

        Args:
            data_tensor (torch.Tensor): Input tensor of shape [b, c, t, h, w].
                                        Values are expected to be in the range [-1, 1].
            wandb_key (str): The key (name) for the log entry in wandb.
            n_viz_sample (int): Max number of samples from the batch dimension (b)
                                to visualize side-by-side. Defaults to 3.
            fps (int): Frames per second to use when logging video. Defaults to 8.
            caption (str, optional): Caption for the logged image/video in wandb. Defaults to None.
        """
        if hasattr(model, "decode"):
            data_tensor = model.decode(data_tensor)
        # Move tensor to CPU and detach from graph (important for logging)
        data_tensor = data_tensor.cpu().float()  # Ensure float for normalization

        _b, _c, _t, _h, _w = data_tensor.shape

        # Clamp and normalize tensor values from [-1, 1] to [0, 1]
        # wandb.Image/Video expect data in [0, 1] range for float or [0, 255] for uint8
        normalized_tensor = (1.0 + data_tensor.clamp(-1, 1)) / 2.0

        actual_n_viz_sample = min(n_viz_sample, _b)

        to_show = normalized_tensor[:actual_n_viz_sample]  # Shape: [actual_n_viz_sample, c, t, h, w]

        is_single_frame = _t == 1

        log_data = {}
        if is_single_frame:
            grid_tensor = rearrange(to_show.squeeze(2), "b c h w -> c h (b w)")
            log_data[wandb_key] = wandb.Image(grid_tensor, caption=caption)
            print(f"Prepared image grid for wandb key '{wandb_key}' with shape {grid_tensor.shape}")

        else:
            # wandb.Video expects time dimension first.
            grid_tensor = rearrange(to_show, "b c t h w -> t c h (b w)")

            # Optional: Convert to uint8 [0, 255] if preferred or if float causes issues
            # grid_tensor = (grid_tensor * 255).to(torch.uint8)

            log_data[wandb_key] = wandb.Video(grid_tensor, fps=fps, caption=caption)
            print(f"Prepared video grid for wandb key '{wandb_key}' with shape {grid_tensor.shape}")

        wandb.log(log_data, step=iteration)
        print(f"Successfully logged to wandb key: {wandb_key}")
