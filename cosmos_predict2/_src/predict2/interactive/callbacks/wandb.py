# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict

import torch
import wandb
from einops import rearrange

from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.predict2.callbacks.wandb_log import WandbCallback as Predict2WandbCallback
from cosmos_predict2._src.predict2.distill.callbacks.wandb_log_scm import WandbCallback as DistillWandbCallback


@torch.no_grad()
def log_video_tensor(
    video_tensor: torch.Tensor,
    key: str,
    iteration: int,
    viz_max_batch: int,
    viz_fps: int,
) -> None:
    """
    Log a video tensor in shape [B, C, T, H, W] with values in [-1, 1].
    """
    if video_tensor is None:
        return

    # Move to CPU and cast to float for wandb
    v = video_tensor.detach().cpu().float()
    # clamp and convert to [0, 1]
    v = (1.0 + v.clamp(-1, 1)) / 2.0

    if v.dim() != 5:
        return

    b = min(max(1, int(viz_max_batch)), v.shape[0])
    v = v[:b]

    # Arrange samples horizontally; wandb.Video expects numpy array [T, C, H, W]
    grid = rearrange(v, "b c t h w -> t c h (b w)")
    grid_uint8 = (grid * 255.0).clamp(0.0, 255.0).to(torch.uint8).numpy()
    payload = {key: wandb.Video(grid_uint8, fps=int(viz_fps), format="mp4")}
    wandb.log(payload, step=iteration)


class WandbCallback(DistillWandbCallback):
    """
    Interactive WandB callback that extends predict2_distill WandbCallback with
    additional logging for the interactive models. In particular, when the model
    exposes a recently generated video from `backward_simulation`, we visualize
    it on wandb.
    """

    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
        viz_every_n: int = 100,
        viz_max_batch: int = 2,
        viz_fps: int = 10,
    ) -> None:
        super().__init__(
            logging_iter_multipler=logging_iter_multipler,
            save_logging_iter_multipler=save_logging_iter_multipler,
            save_s3=save_s3,
        )
        self.viz_every_n = max(1, int(viz_every_n))
        self.viz_max_batch = max(1, int(viz_max_batch))
        self.viz_fps = int(viz_fps)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: Dict[str, torch.Tensor],
        output_batch: Dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # keep original logging behavior
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

        # Additional interactive logging: visualize latest backward_simulation video if available
        if distributed.is_rank0() and iteration % self.viz_every_n == 0:
            latest = getattr(model, "latest_backward_simulation_video", None)
            if isinstance(latest, torch.Tensor) and latest.ndim == 5:
                try:
                    log_video_tensor(
                        latest,
                        key=f"train{self.wandb_extra_tag}/backward_simulation_video",
                        iteration=iteration,
                        viz_max_batch=self.viz_max_batch,
                        viz_fps=self.viz_fps,
                    )
                except Exception:
                    # best-effort visualization; do not crash training
                    pass


class WarmupWandbCallback(Predict2WandbCallback):
    """
    Warm-up WandB callback that extends predict2 WandbCallback with additional
    logging for interactive models' backward_simulation videos.
    """

    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
        viz_every_n: int = 100,
        viz_max_batch: int = 2,
        viz_fps: int = 10,
    ) -> None:
        super().__init__(
            logging_iter_multipler=logging_iter_multipler,
            save_logging_iter_multipler=save_logging_iter_multipler,
            save_s3=save_s3,
        )
        self.viz_every_n = max(1, int(viz_every_n))
        self.viz_max_batch = max(1, int(viz_max_batch))
        self.viz_fps = int(viz_fps)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: Dict[str, torch.Tensor],
        output_batch: Dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # keep original predict2 logging behavior
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

        # Additional warmup logging: visualize latest backward_simulation video if available
        if distributed.is_rank0() and iteration % self.viz_every_n == 0:
            latest = getattr(model, "latest_backward_simulation_video", None)
            if isinstance(latest, torch.Tensor) and latest.ndim == 5:
                try:
                    log_video_tensor(
                        latest,
                        key=f"train{self.wandb_extra_tag}/backward_simulation_video",
                        iteration=iteration,
                        viz_max_batch=self.viz_max_batch,
                        viz_fps=self.viz_fps,
                    )
                except Exception:
                    # best-effort visualization; do not crash training
                    pass
