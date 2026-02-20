# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist
import torch.utils.data
import wandb
from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.callback import WandBCallback as WandBCallbackImage
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.callbacks.wandb_log import _LossRecord


@dataclass
class _LossRecordNoEDM:
    loss: float = 0
    iter_count: int = 0

    def reset(self) -> None:
        self.loss = 0
        self.iter_count = 0

    def get_stat(self) -> Tuple[float, float]:
        if self.iter_count > 0:
            avg_loss = self.loss / self.iter_count
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss.item()
        else:
            avg_loss = 0
        self.reset()
        return avg_loss


class WandbCallback(WandBCallbackImage):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.train_image_log = _LossRecord()
        self.train_video_log = _LossRecord()
        self.train_final_loss_log = _LossRecord()
        self.train_demo_sample_action_mse_loss_log = _LossRecordNoEDM()
        self.train_demo_sample_action_l1_loss_log = _LossRecordNoEDM()
        self.train_demo_sample_future_proprio_mse_loss_log = _LossRecordNoEDM()
        self.train_demo_sample_future_proprio_l1_loss_log = _LossRecordNoEDM()
        self.train_demo_sample_future_wrist_image_mse_loss_log = _LossRecordNoEDM()
        self.train_demo_sample_future_wrist_image_l1_loss_log = _LossRecordNoEDM()
        self.train_demo_sample_future_image_mse_loss_log = _LossRecordNoEDM()
        self.train_demo_sample_future_image_l1_loss_log = _LossRecordNoEDM()
        self.train_demo_sample_value_mse_loss_log = _LossRecordNoEDM()
        self.train_demo_sample_value_l1_loss_log = _LossRecordNoEDM()
        self.train_world_model_sample_future_proprio_mse_loss_log = _LossRecordNoEDM()
        self.train_world_model_sample_future_proprio_l1_loss_log = _LossRecordNoEDM()
        self.train_world_model_sample_future_wrist_image_mse_loss_log = _LossRecordNoEDM()
        self.train_world_model_sample_future_wrist_image_l1_loss_log = _LossRecordNoEDM()
        self.train_world_model_sample_future_image_mse_loss_log = _LossRecordNoEDM()
        self.train_world_model_sample_future_image_l1_loss_log = _LossRecordNoEDM()
        self.train_world_model_sample_value_mse_loss_log = _LossRecordNoEDM()
        self.train_world_model_sample_value_l1_loss_log = _LossRecordNoEDM()
        self.train_value_function_sample_value_mse_loss_log = _LossRecordNoEDM()
        self.train_value_function_sample_value_l1_loss_log = _LossRecordNoEDM()
        self.train_img_unstable_count = torch.zeros(1, device="cuda")
        self.train_video_unstable_count = torch.zeros(1, device="cuda")

        self.val_image_log = _LossRecord()
        self.val_video_log = _LossRecord()
        self.val_final_loss_log = _LossRecord()
        self.val_demo_sample_action_mse_loss_log = _LossRecordNoEDM()
        self.val_demo_sample_action_l1_loss_log = _LossRecordNoEDM()
        self.val_demo_sample_future_proprio_mse_loss_log = _LossRecordNoEDM()
        self.val_demo_sample_future_proprio_l1_loss_log = _LossRecordNoEDM()
        self.val_demo_sample_future_wrist_image_mse_loss_log = _LossRecordNoEDM()
        self.val_demo_sample_future_wrist_image_l1_loss_log = _LossRecordNoEDM()
        self.val_demo_sample_future_image_mse_loss_log = _LossRecordNoEDM()
        self.val_demo_sample_future_image_l1_loss_log = _LossRecordNoEDM()
        self.val_demo_sample_value_mse_loss_log = _LossRecordNoEDM()
        self.val_demo_sample_value_l1_loss_log = _LossRecordNoEDM()
        self.val_world_model_sample_future_proprio_mse_loss_log = _LossRecordNoEDM()
        self.val_world_model_sample_future_proprio_l1_loss_log = _LossRecordNoEDM()
        self.val_world_model_sample_future_wrist_image_mse_loss_log = _LossRecordNoEDM()
        self.val_world_model_sample_future_wrist_image_l1_loss_log = _LossRecordNoEDM()
        self.val_world_model_sample_future_image_mse_loss_log = _LossRecordNoEDM()
        self.val_world_model_sample_future_image_l1_loss_log = _LossRecordNoEDM()
        self.val_world_model_sample_value_mse_loss_log = _LossRecordNoEDM()
        self.val_world_model_sample_value_l1_loss_log = _LossRecordNoEDM()
        self.val_value_function_sample_value_mse_loss_log = _LossRecordNoEDM()
        self.val_value_function_sample_value_l1_loss_log = _LossRecordNoEDM()
        self.val_img_unstable_count = torch.zeros(1, device="cuda")
        self.val_video_unstable_count = torch.zeros(1, device="cuda")

        self.logging_iter_multipler = logging_iter_multipler
        self.save_logging_iter_multipler = save_logging_iter_multipler
        assert self.logging_iter_multipler > 0, "logging_iter_multipler should be greater than 0"
        self.save_s3 = save_s3
        self.wandb_extra_tag = f"@{logging_iter_multipler}" if logging_iter_multipler > 1 else ""
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
            if model.is_image_batch(data_batch):
                self.train_image_log.loss += loss.detach().float()
                self.train_image_log.iter_count += 1
                self.train_image_log.edm_loss += output_batch["edm_loss"].detach().float()
            else:
                self.train_video_log.loss += loss.detach().float()
                self.train_video_log.iter_count += 1
                self.train_video_log.edm_loss += output_batch["edm_loss"].detach().float()

            self.train_final_loss_log.loss += loss.detach().float()
            self.train_final_loss_log.iter_count += 1
            self.train_final_loss_log.edm_loss += output_batch["edm_loss"].detach().float()

            demo_sample_action_mse_loss = output_batch["demo_sample_action_mse_loss"].detach().float()
            if not torch.isnan(demo_sample_action_mse_loss):
                self.train_demo_sample_action_mse_loss_log.loss += demo_sample_action_mse_loss
                self.train_demo_sample_action_mse_loss_log.iter_count += 1
            demo_sample_action_l1_loss = output_batch["demo_sample_action_l1_loss"].detach().float()
            if not torch.isnan(demo_sample_action_l1_loss):
                self.train_demo_sample_action_l1_loss_log.loss += demo_sample_action_l1_loss
                self.train_demo_sample_action_l1_loss_log.iter_count += 1
            demo_sample_future_proprio_mse_loss = output_batch["demo_sample_future_proprio_mse_loss"].detach().float()
            if not torch.isnan(demo_sample_future_proprio_mse_loss):
                self.train_demo_sample_future_proprio_mse_loss_log.loss += demo_sample_future_proprio_mse_loss
                self.train_demo_sample_future_proprio_mse_loss_log.iter_count += 1
            demo_sample_future_proprio_l1_loss = output_batch["demo_sample_future_proprio_l1_loss"].detach().float()
            if not torch.isnan(demo_sample_future_proprio_l1_loss):
                self.train_demo_sample_future_proprio_l1_loss_log.loss += demo_sample_future_proprio_l1_loss
                self.train_demo_sample_future_proprio_l1_loss_log.iter_count += 1
            demo_sample_future_wrist_image_mse_loss = (
                output_batch["demo_sample_future_wrist_image_mse_loss"].detach().float()
            )
            if not torch.isnan(demo_sample_future_wrist_image_mse_loss):
                self.train_demo_sample_future_wrist_image_mse_loss_log.loss += demo_sample_future_wrist_image_mse_loss
                self.train_demo_sample_future_wrist_image_mse_loss_log.iter_count += 1
            demo_sample_future_wrist_image_l1_loss = (
                output_batch["demo_sample_future_wrist_image_l1_loss"].detach().float()
            )
            if not torch.isnan(demo_sample_future_wrist_image_l1_loss):
                self.train_demo_sample_future_wrist_image_l1_loss_log.loss += demo_sample_future_wrist_image_l1_loss
                self.train_demo_sample_future_wrist_image_l1_loss_log.iter_count += 1
            demo_sample_future_image_mse_loss = output_batch["demo_sample_future_image_mse_loss"].detach().float()
            if not torch.isnan(demo_sample_future_image_mse_loss):
                self.train_demo_sample_future_image_mse_loss_log.loss += demo_sample_future_image_mse_loss
                self.train_demo_sample_future_image_mse_loss_log.iter_count += 1
            demo_sample_future_image_l1_loss = output_batch["demo_sample_future_image_l1_loss"].detach().float()
            if not torch.isnan(demo_sample_future_image_l1_loss):
                self.train_demo_sample_future_image_l1_loss_log.loss += demo_sample_future_image_l1_loss
                self.train_demo_sample_future_image_l1_loss_log.iter_count += 1
            demo_sample_value_mse_loss = output_batch["demo_sample_value_mse_loss"].detach().float()
            if not torch.isnan(demo_sample_value_mse_loss):
                self.train_demo_sample_value_mse_loss_log.loss += demo_sample_value_mse_loss
                self.train_demo_sample_value_mse_loss_log.iter_count += 1
            demo_sample_value_l1_loss = output_batch["demo_sample_value_l1_loss"].detach().float()
            if not torch.isnan(demo_sample_value_l1_loss):
                self.train_demo_sample_value_l1_loss_log.loss += demo_sample_value_l1_loss
                self.train_demo_sample_value_l1_loss_log.iter_count += 1

            world_model_sample_future_proprio_mse_loss = (
                output_batch["world_model_sample_future_proprio_mse_loss"].detach().float()
            )
            if not torch.isnan(world_model_sample_future_proprio_mse_loss):
                self.train_world_model_sample_future_proprio_mse_loss_log.loss += (
                    world_model_sample_future_proprio_mse_loss
                )
                self.train_world_model_sample_future_proprio_mse_loss_log.iter_count += 1
            world_model_sample_future_proprio_l1_loss = (
                output_batch["world_model_sample_future_proprio_l1_loss"].detach().float()
            )
            if not torch.isnan(world_model_sample_future_proprio_l1_loss):
                self.train_world_model_sample_future_proprio_l1_loss_log.loss += (
                    world_model_sample_future_proprio_l1_loss
                )
                self.train_world_model_sample_future_proprio_l1_loss_log.iter_count += 1
            world_model_sample_future_wrist_image_mse_loss = (
                output_batch["world_model_sample_future_wrist_image_mse_loss"].detach().float()
            )
            if not torch.isnan(world_model_sample_future_wrist_image_mse_loss):
                self.train_world_model_sample_future_wrist_image_mse_loss_log.loss += (
                    world_model_sample_future_wrist_image_mse_loss
                )
                self.train_world_model_sample_future_wrist_image_mse_loss_log.iter_count += 1
            world_model_sample_future_wrist_image_l1_loss = (
                output_batch["world_model_sample_future_wrist_image_l1_loss"].detach().float()
            )
            if not torch.isnan(world_model_sample_future_wrist_image_l1_loss):
                self.train_world_model_sample_future_wrist_image_l1_loss_log.loss += (
                    world_model_sample_future_wrist_image_l1_loss
                )
                self.train_world_model_sample_future_wrist_image_l1_loss_log.iter_count += 1
            world_model_sample_future_image_mse_loss = (
                output_batch["world_model_sample_future_image_mse_loss"].detach().float()
            )
            if not torch.isnan(world_model_sample_future_image_mse_loss):
                self.train_world_model_sample_future_image_mse_loss_log.loss += world_model_sample_future_image_mse_loss
                self.train_world_model_sample_future_image_mse_loss_log.iter_count += 1
            world_model_sample_future_image_l1_loss = (
                output_batch["world_model_sample_future_image_l1_loss"].detach().float()
            )
            if not torch.isnan(world_model_sample_future_image_l1_loss):
                self.train_world_model_sample_future_image_l1_loss_log.loss += world_model_sample_future_image_l1_loss
                self.train_world_model_sample_future_image_l1_loss_log.iter_count += 1
            world_model_sample_value_mse_loss = output_batch["world_model_sample_value_mse_loss"].detach().float()
            if not torch.isnan(world_model_sample_value_mse_loss):
                self.train_world_model_sample_value_mse_loss_log.loss += world_model_sample_value_mse_loss
                self.train_world_model_sample_value_mse_loss_log.iter_count += 1
            world_model_sample_value_l1_loss = output_batch["world_model_sample_value_l1_loss"].detach().float()
            if not torch.isnan(world_model_sample_value_l1_loss):
                self.train_world_model_sample_value_l1_loss_log.loss += world_model_sample_value_l1_loss
                self.train_world_model_sample_value_l1_loss_log.iter_count += 1

            value_function_sample_value_mse_loss = output_batch["value_function_sample_value_mse_loss"].detach().float()
            if not torch.isnan(value_function_sample_value_mse_loss):
                self.train_value_function_sample_value_mse_loss_log.loss += value_function_sample_value_mse_loss
                self.train_value_function_sample_value_mse_loss_log.iter_count += 1
            value_function_sample_value_l1_loss = output_batch["value_function_sample_value_l1_loss"].detach().float()
            if not torch.isnan(value_function_sample_value_l1_loss):
                self.train_value_function_sample_value_l1_loss_log.loss += value_function_sample_value_l1_loss
                self.train_value_function_sample_value_l1_loss_log.iter_count += 1

        else:
            if model.is_image_batch(data_batch):
                self.train_img_unstable_count += 1
            else:
                self.train_video_unstable_count += 1

        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0:
            if self.logging_iter_multipler > 1:
                timer_results = {}
            else:
                timer_results = self.trainer.training_timer.compute_average_results()
            avg_image_loss, avg_image_edm_loss = self.train_image_log.get_stat()
            avg_video_loss, avg_video_edm_loss = self.train_video_log.get_stat()
            avg_final_loss, avg_final_edm_loss = self.train_final_loss_log.get_stat()

            avg_demo_sample_action_mse_loss = self.train_demo_sample_action_mse_loss_log.get_stat()
            avg_demo_sample_action_l1_loss = self.train_demo_sample_action_l1_loss_log.get_stat()
            avg_future_proprio_mse_loss = self.train_demo_sample_future_proprio_mse_loss_log.get_stat()
            avg_future_proprio_l1_loss = self.train_demo_sample_future_proprio_l1_loss_log.get_stat()
            avg_future_wrist_image_mse_loss = self.train_demo_sample_future_wrist_image_mse_loss_log.get_stat()
            avg_future_wrist_image_l1_loss = self.train_demo_sample_future_wrist_image_l1_loss_log.get_stat()
            avg_demo_sample_future_image_mse_loss = self.train_demo_sample_future_image_mse_loss_log.get_stat()
            avg_demo_sample_future_image_l1_loss = self.train_demo_sample_future_image_l1_loss_log.get_stat()
            avg_demo_sample_value_mse_loss = self.train_demo_sample_value_mse_loss_log.get_stat()
            avg_demo_sample_value_l1_loss = self.train_demo_sample_value_l1_loss_log.get_stat()

            avg_world_model_sample_future_proprio_mse_loss = (
                self.train_world_model_sample_future_proprio_mse_loss_log.get_stat()
            )
            avg_world_model_sample_future_proprio_l1_loss = (
                self.train_world_model_sample_future_proprio_l1_loss_log.get_stat()
            )
            avg_world_model_sample_future_wrist_image_mse_loss = (
                self.train_world_model_sample_future_wrist_image_mse_loss_log.get_stat()
            )
            avg_world_model_sample_future_wrist_image_l1_loss = (
                self.train_world_model_sample_future_wrist_image_l1_loss_log.get_stat()
            )
            avg_world_model_sample_future_image_mse_loss = (
                self.train_world_model_sample_future_image_mse_loss_log.get_stat()
            )
            avg_world_model_sample_future_image_l1_loss = (
                self.train_world_model_sample_future_image_l1_loss_log.get_stat()
            )
            avg_world_model_sample_value_mse_loss = self.train_world_model_sample_value_mse_loss_log.get_stat()
            avg_world_model_sample_value_l1_loss = self.train_world_model_sample_value_l1_loss_log.get_stat()

            avg_value_function_sample_value_mse_loss = self.train_value_function_sample_value_mse_loss_log.get_stat()
            avg_value_function_sample_value_l1_loss = self.train_value_function_sample_value_l1_loss_log.get_stat()

            dist.all_reduce(self.train_img_unstable_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.train_video_unstable_count, op=dist.ReduceOp.SUM)

            if distributed.is_rank0():
                info = {f"timer/{key}": value for key, value in timer_results.items()}
                info.update(
                    {
                        f"train{self.wandb_extra_tag}/image_loss": avg_image_loss,
                        f"train{self.wandb_extra_tag}/image_edm_loss": avg_image_edm_loss,
                        f"train{self.wandb_extra_tag}/video_loss": avg_video_loss,
                        f"train{self.wandb_extra_tag}/video_edm_loss": avg_video_edm_loss,
                        f"train{self.wandb_extra_tag}/loss": avg_final_loss,
                        f"train{self.wandb_extra_tag}/edm_loss": avg_final_edm_loss,
                        f"train{self.wandb_extra_tag}/demo_sample_action_mse_loss": avg_demo_sample_action_mse_loss,
                        f"train{self.wandb_extra_tag}/demo_sample_action_l1_loss": avg_demo_sample_action_l1_loss,
                        f"train{self.wandb_extra_tag}/demo_sample_future_proprio_mse_loss": avg_future_proprio_mse_loss,
                        f"train{self.wandb_extra_tag}/demo_sample_future_proprio_l1_loss": avg_future_proprio_l1_loss,
                        f"train{self.wandb_extra_tag}/demo_sample_future_wrist_image_mse_loss": avg_future_wrist_image_mse_loss,
                        f"train{self.wandb_extra_tag}/demo_sample_future_wrist_image_l1_loss": avg_future_wrist_image_l1_loss,
                        f"train{self.wandb_extra_tag}/demo_sample_future_image_mse_loss": avg_demo_sample_future_image_mse_loss,
                        f"train{self.wandb_extra_tag}/demo_sample_future_image_l1_loss": avg_demo_sample_future_image_l1_loss,
                        f"train{self.wandb_extra_tag}/demo_sample_value_mse_loss": avg_demo_sample_value_mse_loss,
                        f"train{self.wandb_extra_tag}/demo_sample_value_l1_loss": avg_demo_sample_value_l1_loss,
                        f"train{self.wandb_extra_tag}/world_model_sample_future_proprio_mse_loss": avg_world_model_sample_future_proprio_mse_loss,
                        f"train{self.wandb_extra_tag}/world_model_sample_future_proprio_l1_loss": avg_world_model_sample_future_proprio_l1_loss,
                        f"train{self.wandb_extra_tag}/world_model_sample_future_wrist_image_mse_loss": avg_world_model_sample_future_wrist_image_mse_loss,
                        f"train{self.wandb_extra_tag}/world_model_sample_future_wrist_image_l1_loss": avg_world_model_sample_future_wrist_image_l1_loss,
                        f"train{self.wandb_extra_tag}/world_model_sample_future_image_mse_loss": avg_world_model_sample_future_image_mse_loss,
                        f"train{self.wandb_extra_tag}/world_model_sample_future_image_l1_loss": avg_world_model_sample_future_image_l1_loss,
                        f"train{self.wandb_extra_tag}/world_model_sample_value_mse_loss": avg_world_model_sample_value_mse_loss,
                        f"train{self.wandb_extra_tag}/world_model_sample_value_l1_loss": avg_world_model_sample_value_l1_loss,
                        f"train{self.wandb_extra_tag}/value_function_sample_value_mse_loss": avg_value_function_sample_value_mse_loss,
                        f"train{self.wandb_extra_tag}/value_function_sample_value_l1_loss": avg_value_function_sample_value_l1_loss,
                        f"train{self.wandb_extra_tag}/train_img_unstable_count": self.train_img_unstable_count.item(),
                        f"train{self.wandb_extra_tag}/train_video_unstable_count": self.train_video_unstable_count.item(),
                        "iteration": iteration,
                        "sample_counter": getattr(self.trainer, "sample_counter", iteration),
                    }
                )
                if self.save_s3:
                    if (
                        iteration
                        % (
                            self.config.trainer.logging_iter
                            * self.logging_iter_multipler
                            * self.save_logging_iter_multipler
                        )
                        == 0
                    ):
                        easy_io.dump(
                            info,
                            f"s3://rundir/{self.name}/Train_Iter{iteration:09d}.json",
                        )

                if wandb:
                    wandb.log(info, step=iteration)
            if self.logging_iter_multipler == 1:
                self.trainer.training_timer.reset()

            # reset unstable count
            self.train_img_unstable_count.zero_()
            self.train_video_unstable_count.zero_()

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        """
        Callback that is run after validation step is executed; similar to self.on_train_step_end().

        Things that are different from self.on_train_step_end():
            - No use of training timer
            - Using validation_iter instead of logging_iter
            - Doesn't do the push to WandB here; see self.on_validation_end() for that
        """
        skip_update_due_to_unstable_loss = False
        if torch.isnan(loss) or torch.isinf(loss):
            skip_update_due_to_unstable_loss = True
            log.critical(
                f"Unstable loss {loss} at iteration {iteration} with is_image_batch: {model.is_image_batch(data_batch)}",
                rank0_only=False,
            )

        if not skip_update_due_to_unstable_loss:
            if model.is_image_batch(data_batch):
                self.val_image_log.loss += loss.detach().float()
                self.val_image_log.iter_count += 1
                self.val_image_log.edm_loss += output_batch["edm_loss"].detach().float()
            else:
                self.val_video_log.loss += loss.detach().float()
                self.val_video_log.iter_count += 1
                self.val_video_log.edm_loss += output_batch["edm_loss"].detach().float()

            self.val_final_loss_log.loss += loss.detach().float()
            self.val_final_loss_log.iter_count += 1
            self.val_final_loss_log.edm_loss += output_batch["edm_loss"].detach().float()

            demo_sample_action_mse_loss = output_batch["demo_sample_action_mse_loss"].detach().float()
            if not torch.isnan(demo_sample_action_mse_loss):
                self.val_demo_sample_action_mse_loss_log.loss += demo_sample_action_mse_loss
                self.val_demo_sample_action_mse_loss_log.iter_count += 1
            demo_sample_action_l1_loss = output_batch["demo_sample_action_l1_loss"].detach().float()
            if not torch.isnan(demo_sample_action_l1_loss):
                self.val_demo_sample_action_l1_loss_log.loss += demo_sample_action_l1_loss
                self.val_demo_sample_action_l1_loss_log.iter_count += 1
            demo_sample_future_proprio_mse_loss = output_batch["demo_sample_future_proprio_mse_loss"].detach().float()
            if not torch.isnan(demo_sample_future_proprio_mse_loss):
                self.val_demo_sample_future_proprio_mse_loss_log.loss += demo_sample_future_proprio_mse_loss
                self.val_demo_sample_future_proprio_mse_loss_log.iter_count += 1
            demo_sample_future_proprio_l1_loss = output_batch["demo_sample_future_proprio_l1_loss"].detach().float()
            if not torch.isnan(demo_sample_future_proprio_l1_loss):
                self.val_demo_sample_future_proprio_l1_loss_log.loss += demo_sample_future_proprio_l1_loss
                self.val_demo_sample_future_proprio_l1_loss_log.iter_count += 1
            demo_sample_future_wrist_image_mse_loss = (
                output_batch["demo_sample_future_wrist_image_mse_loss"].detach().float()
            )
            if not torch.isnan(demo_sample_future_wrist_image_mse_loss):
                self.val_demo_sample_future_wrist_image_mse_loss_log.loss += demo_sample_future_wrist_image_mse_loss
                self.val_demo_sample_future_wrist_image_mse_loss_log.iter_count += 1
            demo_sample_future_wrist_image_l1_loss = (
                output_batch["demo_sample_future_wrist_image_l1_loss"].detach().float()
            )
            if not torch.isnan(demo_sample_future_wrist_image_l1_loss):
                self.val_demo_sample_future_wrist_image_l1_loss_log.loss += demo_sample_future_wrist_image_l1_loss
                self.val_demo_sample_future_wrist_image_l1_loss_log.iter_count += 1
            demo_sample_future_image_mse_loss = output_batch["demo_sample_future_image_mse_loss"].detach().float()
            if not torch.isnan(demo_sample_future_image_mse_loss):
                self.val_demo_sample_future_image_mse_loss_log.loss += demo_sample_future_image_mse_loss
                self.val_demo_sample_future_image_mse_loss_log.iter_count += 1
            demo_sample_future_image_l1_loss = output_batch["demo_sample_future_image_l1_loss"].detach().float()
            if not torch.isnan(demo_sample_future_image_l1_loss):
                self.val_demo_sample_future_image_l1_loss_log.loss += demo_sample_future_image_l1_loss
                self.val_demo_sample_future_image_l1_loss_log.iter_count += 1
            demo_sample_value_mse_loss = output_batch["demo_sample_value_mse_loss"].detach().float()
            if not torch.isnan(demo_sample_value_mse_loss):
                self.val_demo_sample_value_mse_loss_log.loss += demo_sample_value_mse_loss
                self.val_demo_sample_value_mse_loss_log.iter_count += 1
            demo_sample_value_l1_loss = output_batch["demo_sample_value_l1_loss"].detach().float()
            if not torch.isnan(demo_sample_value_l1_loss):
                self.val_demo_sample_value_l1_loss_log.loss += demo_sample_value_l1_loss
                self.val_demo_sample_value_l1_loss_log.iter_count += 1

            world_model_sample_future_proprio_mse_loss = (
                output_batch["world_model_sample_future_proprio_mse_loss"].detach().float()
            )
            if not torch.isnan(world_model_sample_future_proprio_mse_loss):
                self.val_world_model_sample_future_proprio_mse_loss_log.loss += (
                    world_model_sample_future_proprio_mse_loss
                )
                self.val_world_model_sample_future_proprio_mse_loss_log.iter_count += 1
            world_model_sample_future_proprio_l1_loss = (
                output_batch["world_model_sample_future_proprio_l1_loss"].detach().float()
            )
            if not torch.isnan(world_model_sample_future_proprio_l1_loss):
                self.val_world_model_sample_future_proprio_l1_loss_log.loss += world_model_sample_future_proprio_l1_loss
                self.val_world_model_sample_future_proprio_l1_loss_log.iter_count += 1
            world_model_sample_future_wrist_image_mse_loss = (
                output_batch["world_model_sample_future_wrist_image_mse_loss"].detach().float()
            )
            if not torch.isnan(world_model_sample_future_wrist_image_mse_loss):
                self.val_world_model_sample_future_wrist_image_mse_loss_log.loss += (
                    world_model_sample_future_wrist_image_mse_loss
                )
                self.val_world_model_sample_future_wrist_image_mse_loss_log.iter_count += 1
            world_model_sample_future_wrist_image_l1_loss = (
                output_batch["world_model_sample_future_wrist_image_l1_loss"].detach().float()
            )
            if not torch.isnan(world_model_sample_future_wrist_image_l1_loss):
                self.val_world_model_sample_future_wrist_image_l1_loss_log.loss += (
                    world_model_sample_future_wrist_image_l1_loss
                )
                self.val_world_model_sample_future_wrist_image_l1_loss_log.iter_count += 1
            world_model_sample_future_image_mse_loss = (
                output_batch["world_model_sample_future_image_mse_loss"].detach().float()
            )
            if not torch.isnan(world_model_sample_future_image_mse_loss):
                self.val_world_model_sample_future_image_mse_loss_log.loss += world_model_sample_future_image_mse_loss
                self.val_world_model_sample_future_image_mse_loss_log.iter_count += 1
            world_model_sample_future_image_l1_loss = (
                output_batch["world_model_sample_future_image_l1_loss"].detach().float()
            )
            if not torch.isnan(world_model_sample_future_image_l1_loss):
                self.val_world_model_sample_future_image_l1_loss_log.loss += world_model_sample_future_image_l1_loss
                self.val_world_model_sample_future_image_l1_loss_log.iter_count += 1
            world_model_sample_value_mse_loss = output_batch["world_model_sample_value_mse_loss"].detach().float()
            if not torch.isnan(world_model_sample_value_mse_loss):
                self.val_world_model_sample_value_mse_loss_log.loss += world_model_sample_value_mse_loss
                self.val_world_model_sample_value_mse_loss_log.iter_count += 1
            world_model_sample_value_l1_loss = output_batch["world_model_sample_value_l1_loss"].detach().float()
            if not torch.isnan(world_model_sample_value_l1_loss):
                self.val_world_model_sample_value_l1_loss_log.loss += world_model_sample_value_l1_loss
                self.val_world_model_sample_value_l1_loss_log.iter_count += 1

            value_function_sample_value_mse_loss = output_batch["value_function_sample_value_mse_loss"].detach().float()
            if not torch.isnan(value_function_sample_value_mse_loss):
                self.val_value_function_sample_value_mse_loss_log.loss += value_function_sample_value_mse_loss
                self.val_value_function_sample_value_mse_loss_log.iter_count += 1
            value_function_sample_value_l1_loss = output_batch["value_function_sample_value_l1_loss"].detach().float()
            if not torch.isnan(value_function_sample_value_l1_loss):
                self.val_value_function_sample_value_l1_loss_log.loss += value_function_sample_value_l1_loss
                self.val_value_function_sample_value_l1_loss_log.iter_count += 1

        else:
            if model.is_image_batch(data_batch):
                self.val_img_unstable_count += 1
            else:
                self.val_video_unstable_count += 1

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        """Computes and logs averages of all the validation metrics."""
        if iteration % (self.config.trainer.validation_iter * self.logging_iter_multipler) == 0:
            avg_image_loss, avg_image_edm_loss = self.val_image_log.get_stat()
            avg_video_loss, avg_video_edm_loss = self.val_video_log.get_stat()
            avg_final_loss, avg_final_edm_loss = self.val_final_loss_log.get_stat()

            avg_demo_sample_action_mse_loss = self.val_demo_sample_action_mse_loss_log.get_stat()
            avg_demo_sample_action_l1_loss = self.val_demo_sample_action_l1_loss_log.get_stat()
            avg_future_proprio_mse_loss = self.val_demo_sample_future_proprio_mse_loss_log.get_stat()
            avg_future_proprio_l1_loss = self.val_demo_sample_future_proprio_l1_loss_log.get_stat()
            avg_future_wrist_image_mse_loss = self.val_demo_sample_future_wrist_image_mse_loss_log.get_stat()
            avg_future_wrist_image_l1_loss = self.val_demo_sample_future_wrist_image_l1_loss_log.get_stat()
            avg_demo_sample_future_image_mse_loss = self.val_demo_sample_future_image_mse_loss_log.get_stat()
            avg_demo_sample_future_image_l1_loss = self.val_demo_sample_future_image_l1_loss_log.get_stat()
            avg_demo_sample_value_mse_loss = self.val_demo_sample_value_mse_loss_log.get_stat()
            avg_demo_sample_value_l1_loss = self.val_demo_sample_value_l1_loss_log.get_stat()

            avg_world_model_sample_future_proprio_mse_loss = (
                self.val_world_model_sample_future_proprio_mse_loss_log.get_stat()
            )
            avg_world_model_sample_future_proprio_l1_loss = (
                self.val_world_model_sample_future_proprio_l1_loss_log.get_stat()
            )
            avg_world_model_sample_future_wrist_image_mse_loss = (
                self.val_world_model_sample_future_wrist_image_mse_loss_log.get_stat()
            )
            avg_world_model_sample_future_wrist_image_l1_loss = (
                self.val_world_model_sample_future_wrist_image_l1_loss_log.get_stat()
            )
            avg_world_model_sample_future_image_mse_loss = (
                self.val_world_model_sample_future_image_mse_loss_log.get_stat()
            )
            avg_world_model_sample_future_image_l1_loss = (
                self.val_world_model_sample_future_image_l1_loss_log.get_stat()
            )
            avg_world_model_sample_value_mse_loss = self.val_world_model_sample_value_mse_loss_log.get_stat()
            avg_world_model_sample_value_l1_loss = self.val_world_model_sample_value_l1_loss_log.get_stat()

            avg_value_function_sample_value_mse_loss = self.val_value_function_sample_value_mse_loss_log.get_stat()
            avg_value_function_sample_value_l1_loss = self.val_value_function_sample_value_l1_loss_log.get_stat()

            dist.all_reduce(self.val_img_unstable_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.val_video_unstable_count, op=dist.ReduceOp.SUM)

            if distributed.is_rank0():
                info = {}
                info.update(
                    {
                        f"val{self.wandb_extra_tag}/image_loss": avg_image_loss,
                        f"val{self.wandb_extra_tag}/image_edm_loss": avg_image_edm_loss,
                        f"val{self.wandb_extra_tag}/video_loss": avg_video_loss,
                        f"val{self.wandb_extra_tag}/video_edm_loss": avg_video_edm_loss,
                        f"val{self.wandb_extra_tag}/loss": avg_final_loss,
                        f"val{self.wandb_extra_tag}/edm_loss": avg_final_edm_loss,
                        f"val{self.wandb_extra_tag}/demo_sample_action_mse_loss": avg_demo_sample_action_mse_loss,
                        f"val{self.wandb_extra_tag}/demo_sample_action_l1_loss": avg_demo_sample_action_l1_loss,
                        f"val{self.wandb_extra_tag}/demo_sample_future_proprio_mse_loss": avg_future_proprio_mse_loss,
                        f"val{self.wandb_extra_tag}/demo_sample_future_proprio_l1_loss": avg_future_proprio_l1_loss,
                        f"val{self.wandb_extra_tag}/demo_sample_future_wrist_image_mse_loss": avg_future_wrist_image_mse_loss,
                        f"val{self.wandb_extra_tag}/demo_sample_future_wrist_image_l1_loss": avg_future_wrist_image_l1_loss,
                        f"val{self.wandb_extra_tag}/demo_sample_future_image_mse_loss": avg_demo_sample_future_image_mse_loss,
                        f"val{self.wandb_extra_tag}/demo_sample_future_image_l1_loss": avg_demo_sample_future_image_l1_loss,
                        f"val{self.wandb_extra_tag}/demo_sample_value_mse_loss": avg_demo_sample_value_mse_loss,
                        f"val{self.wandb_extra_tag}/demo_sample_value_l1_loss": avg_demo_sample_value_l1_loss,
                        f"val{self.wandb_extra_tag}/world_model_sample_future_proprio_mse_loss": avg_world_model_sample_future_proprio_mse_loss,
                        f"val{self.wandb_extra_tag}/world_model_sample_future_proprio_l1_loss": avg_world_model_sample_future_proprio_l1_loss,
                        f"val{self.wandb_extra_tag}/world_model_sample_future_wrist_image_mse_loss": avg_world_model_sample_future_wrist_image_mse_loss,
                        f"val{self.wandb_extra_tag}/world_model_sample_future_wrist_image_l1_loss": avg_world_model_sample_future_wrist_image_l1_loss,
                        f"val{self.wandb_extra_tag}/world_model_sample_future_image_mse_loss": avg_world_model_sample_future_image_mse_loss,
                        f"val{self.wandb_extra_tag}/world_model_sample_future_image_l1_loss": avg_world_model_sample_future_image_l1_loss,
                        f"val{self.wandb_extra_tag}/world_model_sample_value_mse_loss": avg_world_model_sample_value_mse_loss,
                        f"val{self.wandb_extra_tag}/world_model_sample_value_l1_loss": avg_world_model_sample_value_l1_loss,
                        f"val{self.wandb_extra_tag}/value_function_sample_value_mse_loss": avg_value_function_sample_value_mse_loss,
                        f"val{self.wandb_extra_tag}/value_function_sample_value_l1_loss": avg_value_function_sample_value_l1_loss,
                        f"val{self.wandb_extra_tag}/val_img_unstable_count": self.val_img_unstable_count.item(),
                        f"val{self.wandb_extra_tag}/val_video_unstable_count": self.val_video_unstable_count.item(),
                    }
                )
                if self.save_s3:
                    if (
                        iteration
                        % (
                            self.config.trainer.validation_iter
                            * self.logging_iter_multipler
                            * self.save_logging_iter_multipler
                        )
                        == 0
                    ):
                        easy_io.dump(
                            info,
                            f"s3://rundir/{self.name}/Val_Iter{iteration:09d}.json",
                        )

                if wandb:
                    wandb.log(info, step=iteration)

                log.info(f"Validation final loss (iteration {iteration}): {avg_final_loss:4f}")

            # reset unstable count
            self.val_img_unstable_count.zero_()
            self.val_video_unstable_count.zero_()


WANDB_CALLBACK_ACTIONS = dict(
    wandb=L(WandbCallback)(
        save_s3="${upload_reproducible_setup}",
        logging_iter_multipler=1,
        save_logging_iter_multipler=10,
    ),
    wandb_10x=L(WandbCallback)(
        logging_iter_multipler=10,
        save_logging_iter_multipler=1,
        save_s3="${upload_reproducible_setup}",
    ),
)


def register_configs():
    cs = ConfigStore.instance()
    cs.store(
        group="callbacks",
        package="trainer.callbacks",
        name="wandb_callback_actions",
        node=WANDB_CALLBACK_ACTIONS,
    )
