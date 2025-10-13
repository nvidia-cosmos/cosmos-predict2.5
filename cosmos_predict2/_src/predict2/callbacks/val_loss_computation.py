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
ValLossComputation Callback

A validation loss computation callback that inherits from EveryNDrawSample
and reuses its x0_pred functionality with a different dataset (PromptVideoItemDataset).

Key Features:
- Inherits from EveryNDrawSample: Reuses x0_pred method and infrastructure
- Real Video Evaluation: Uses PromptVideoItemDataset with actual video ground truth data
- Same Noise Schedule: Uses identical log-space noise schedule as EveryNDrawSample.x0_pred
- EMA Support: Inherited EMA functionality from parent class
- Distributed Training: Inherited distributed handling from parent class
- Comprehensive Logging: Logs validation loss metrics to console and WandB
"""

from functools import partial

import torch
import wandb
from megatron.core import parallel_state
from torch.utils.data import DataLoader

from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.predict2.callbacks.every_n_draw_sample import EveryNDrawSample
from cosmos_predict2._src.predict2.datasets.data_sources.item_datasets_for_validation import get_itemdataset_option
from cosmos_predict2._src.predict2.datasets.item_dataset import (
    ItemDatasetConfig,
    PromptVideoItemDataset,
    calculate_indices,
)
from cosmos_predict2._src.predict2.models.text2world_model import DiffusionModel


class ValLossComputation(EveryNDrawSample):
    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        """Override to add debug logging before calling parent."""
        log.critical(f"DEBUG: ValLossComputation.on_training_step_end called at iteration {iteration}", rank0_only=True)
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    """
    Validation loss computation callback inheriting from EveryNDrawSample.

    This callback reuses EveryNDrawSample.x0_pred functionality but evaluates on
    PromptVideoItemDataset instead of training data. It computes validation loss
    using the same noise schedule and methodology as the parent class.

    Key differences from parent:
    - Uses PromptVideoItemDataset for validation data instead of training batches
    - Only runs x0_pred (no sampling generation)
    - Logs validation loss metrics instead of visual samples
    """

    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        val_dataset_name: str = "ptbench_video_val",
        batch_size: int = 1,
        n_sigmas_for_x0_prediction: int = 4,  # Match EveryNDrawSample parameter
        is_ema: bool = True,
        is_debug: bool = False,
        max_eval_samples: int = 100,
        **kwargs,  # Pass any additional EveryNDrawSample parameters
    ):
        """Initialize ValLossComputation by inheriting from EveryNDrawSample."""
        print(f"DEBUG: ValLossComputation.__init__ called with every_n={every_n}, max_eval_samples={max_eval_samples}")
        log.critical(f"DEBUG: Initializing ValLossComputation with every_n={every_n}, is_ema={is_ema}", rank0_only=True)
        # Initialize parent with x0_prediction enabled, sampling disabled
        super().__init__(
            every_n=every_n,
            step_size=step_size,
            n_viz_sample=batch_size,  # Match batch_size for consistency
            do_x0_prediction=True,  # Always enable x0_prediction
            n_sigmas_for_x0_prediction=n_sigmas_for_x0_prediction,
            is_ema=is_ema,
            save_s3=False,  # Disable S3 saving for validation
            guidance=[0.0],  # Minimal guidance (not used since we skip sampling)
            **kwargs,
        )

        # Validation-specific parameters
        self.val_dataset_name = val_dataset_name
        self.batch_size = batch_size
        self.is_debug = is_debug
        self.max_eval_samples = max_eval_samples

        # Override parent's name for logging
        self.name = self.__class__.__name__

        # Will be initialized in on_train_start
        self.val_dataloader = None

    def on_train_start(self, model: DiffusionModel, iteration: int = 0) -> None:
        """Initialize validation dataset and call parent's on_train_start."""
        log.critical(f"DEBUG: ValLossComputation.on_train_start called at iteration {iteration}", rank0_only=True)

        # Call parent's on_train_start to initialize base functionality
        super().on_train_start(model, iteration)

        # Set up validation dataset
        self._setup_validation_dataset(model)

        log.critical(
            f"DEBUG: ValLossComputation.on_train_start completed, dataloader: {self.val_dataloader is not None}",
            rank0_only=True,
        )

    def _setup_validation_dataset(self, model: DiffusionModel) -> None:
        """Set up the validation dataset for evaluation."""
        log.critical(f"DEBUG: Starting _setup_validation_dataset for {self.val_dataset_name}", rank0_only=True)

        # Get video dimensions from model configuration
        video_height, video_width = model.get_video_height_width()
        num_video_frames = model.tokenizer.get_pixel_num_frames(model.get_num_video_latent_frames())
        log.critical(
            f"DEBUG: Video dimensions: {video_height}x{video_width}, frames: {num_video_frames}", rank0_only=True
        )

        # Get dataset configuration - fail fast if not found
        dataset_option: ItemDatasetConfig = get_itemdataset_option(self.val_dataset_name)
        dataset_path = dataset_option.path
        dataset_length = dataset_option.length
        log.critical(f"DEBUG: Found dataset option: path={dataset_path}, length={dataset_length}", rank0_only=True)

        log.warning(
            f"Using validation dataset: {self.val_dataset_name} at path: {dataset_path}. "
            f"It is user's responsibility to set up the correct credentials."
        )

        # Limit evaluation samples and ensure FSDP compatibility
        dataset_length = min(dataset_length, self.max_eval_samples)
        dataset_length = int(dataset_length // model.config.fsdp_shard_size * model.config.fsdp_shard_size)
        log.critical(f"DEBUG: Final dataset_length after FSDP alignment: {dataset_length}", rank0_only=True)

        # Distribute dataset across ranks
        if torch.distributed.get_world_size() > parallel_state.get_data_parallel_world_size():
            num_replicate = parallel_state.get_data_parallel_world_size()
            data_parallel_id = parallel_state.get_data_parallel_rank()
            start_idx, end_idx, is_overflow = calculate_indices(dataset_length, num_replicate, data_parallel_id)
            log.critical(
                f"DEBUG: Using data parallel: replicate={num_replicate}, id={data_parallel_id}", rank0_only=True
            )
        else:
            world_size, rank = distributed.get_world_size(), distributed.get_rank()
            start_idx, end_idx, is_overflow = calculate_indices(dataset_length, world_size, rank)
            log.critical(f"DEBUG: Using world distributed: world_size={world_size}, rank={rank}", rank0_only=True)

        log.critical(f"DEBUG: Calculated indices: {start_idx}-{end_idx}, overflow={is_overflow}", rank0_only=True)

        # Debug mode: only evaluate 2 samples
        if self.is_debug:
            end_idx = min(start_idx + 2, end_idx)

        if is_overflow:
            log.critical("DEBUG: Overflow in calculating indices, SKIPPING.", rank0_only=True)
            self.val_dataloader = None
        else:
            log.critical(
                f"DEBUG: Creating PromptVideoItemDataset with indices {start_idx}-{end_idx}...", rank0_only=True
            )
            # Create validation dataloader
            self.val_dataloader = DataLoader(
                PromptVideoItemDataset(
                    path=dataset_path,
                    start_index=start_idx,
                    end_index=end_idx,
                    height=video_height,
                    width=video_width,
                    num_video_frames=num_video_frames,
                ),
                batch_size=self.batch_size,
                num_workers=4,
                prefetch_factor=2,
                persistent_workers=False,
                shuffle=False,
            )
            log.critical(f"DEBUG: Successfully created dataloader: {self.val_dataloader is not None}", rank0_only=True)

        log.critical(
            f"ValLoss: Finished setting up validation dataloader for {self.val_dataset_name} "
            f"with video shape {num_video_frames}x{video_height}x{video_width}",
            rank0_only=True,
        )

    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        """
        Compute validation loss by reusing parent's x0_pred on validation dataset.

        This method iterates through the validation dataset and applies the parent's
        x0_pred method to each batch, then aggregates and logs the validation metrics.
        """
        log.critical(f"DEBUG: ValLossComputation.every_n_impl called at iteration {iteration}", rank0_only=True)
        del data_batch, output_batch, loss  # Not used, we use validation data

        if self.val_dataloader is None:
            log.critical(
                f"DEBUG: ValLoss: Skipping validation at iteration {iteration} (no dataloader)", rank0_only=True
            )
            return

        tag = "ema" if self.is_ema else "reg"
        log.critical(f"ValLoss: {tag} Starting validation loss computation at iteration {iteration}", rank0_only=True)

        # Use parent's EMA context handling
        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "val_loss")
        else:
            from contextlib import nullcontext

            context = nullcontext

        total_mse_losses = []
        num_samples = 0

        with context():
            # Iterate through validation dataset
            for i, val_data_batch in enumerate(self.val_dataloader):
                log.debug(f"ValLoss: {tag} Processing validation batch {i}")

                # Prepare validation batch (same as training batch format)
                val_data_batch = self._prepare_validation_batch(val_data_batch, model)

                # Create dummy output_batch (required by parent's x0_pred)
                dummy_output_batch = {"x0": val_data_batch["video"]}

                # Reuse parent's x0_pred method - this does the heavy lifting!
                _, mse_loss, sigmas = self.x0_pred(trainer, model, val_data_batch, dummy_output_batch, None, iteration)

                total_mse_losses.append(mse_loss)
                num_samples += val_data_batch["video"].shape[0]

        # Aggregate validation metrics
        if total_mse_losses:
            # Stack and average MSE losses across all validation batches
            all_mse = torch.stack(total_mse_losses)  # [num_batches, num_noise_levels]
            avg_mse_per_noise = all_mse.mean(dim=0)  # [num_noise_levels]
            overall_val_loss = avg_mse_per_noise.mean().item()  # Single scalar

            # Distributed aggregation
            if torch.distributed.is_initialized():
                # Reduce using the same pattern as parent callback
                samples_tensor = torch.tensor(num_samples, device="cuda", dtype=torch.float32)
                total_samples_tensor = distributed.dist_reduce_tensor(samples_tensor, reduce="sum")
                total_samples = int(total_samples_tensor.item())

                # Reduce validation loss (mean across ranks)
                loss_tensor = torch.tensor(overall_val_loss, device="cuda", dtype=torch.float32)
                overall_val_loss_tensor = distributed.dist_reduce_tensor(loss_tensor, reduce="mean")
                overall_val_loss = overall_val_loss_tensor.item()

                # Also reduce per-noise-level losses for WandB logging
                avg_mse_per_noise = distributed.dist_reduce_tensor(avg_mse_per_noise, reduce="mean")
            else:
                total_samples = num_samples

            # Log validation results
            log.critical(
                f"ValLoss: {tag} Iteration {iteration} - Validation Loss: {overall_val_loss:.6f}, "
                f"Samples: {total_samples}",
                rank0_only=False,
            )

            # WandB logging (rank 0 only)
            if wandb.run and distributed.get_rank() == 0:
                wandb.log(
                    {
                        f"val_loss/{tag}": overall_val_loss,
                        f"val_samples/{tag}": total_samples,
                    },
                    step=iteration,
                )

                # Log per-noise-level MSE (same format as parent class)
                for i, sigma_val in enumerate(sigmas):
                    wandb.log({f"val_mse_{tag}/Sigma{sigma_val:0.5f}": avg_mse_per_noise[i].item()}, step=iteration)

        log.critical(f"ValLoss: {tag} Completed validation loss computation at iteration {iteration}", rank0_only=False)
        distributed.barrier()
        torch.cuda.empty_cache()

    def _prepare_validation_batch(self, val_data_batch: dict, model: DiffusionModel) -> dict:
        """
        Prepare validation batch to match training data format expected by parent's x0_pred.

        This method converts PromptVideoItemDataset format to the format expected by
        the parent class's x0_pred method, ensuring compatibility.
        """
        # Move to correct device, but DON'T convert video dtype (keep uint8)
        for key, value in val_data_batch.items():
            if isinstance(value, torch.Tensor):
                if key == "video":
                    # Keep video as uint8, only move to device
                    val_data_batch[key] = value.to(device=model.tensor_kwargs["device"])
                else:
                    # Convert other tensors to model precision
                    val_data_batch[key] = value.to(**model.tensor_kwargs)

        # Video stays in uint8 format - model.get_data_and_condition() expects this
        # The model's _normalize_video_databatch_inplace() will handle uint8 -> float conversion

        # Handle text embeddings (same as parent's sample method)
        if model.config.text_encoder_config is not None and model.config.text_encoder_config.compute_online:
            if "prompt" in val_data_batch:
                val_data_batch["ai_caption"] = val_data_batch["prompt"]  # Use prompt as ai_caption
                text_embeddings = model.text_encoder.compute_text_embeddings_online(
                    val_data_batch, model.input_caption_key
                )
                val_data_batch["t5_text_embeddings"] = text_embeddings
                val_data_batch["t5_text_mask"] = torch.ones(
                    text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
                )

        return val_data_batch
