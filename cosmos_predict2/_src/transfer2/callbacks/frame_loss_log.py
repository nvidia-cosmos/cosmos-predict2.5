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
from typing import Tuple

import torch
import torch.distributed as dist
import wandb

from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.callback import Callback
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io


@dataclass
class _LossRecord:
    iter_count: int = 0
    edm_loss_m1: float = 0
    edm_loss_m2: float = 0

    def reset(self) -> None:
        self.iter_count = 0
        self.edm_loss_m1 = 0
        self.edm_loss_m2 = 0

    def get_stat(self) -> Tuple[float, float]:
        if self.iter_count > 0:
            edm_loss_m1 = self.edm_loss_m1 / self.iter_count
            edm_loss_m2 = self.edm_loss_m2 / self.iter_count
            dist.all_reduce(edm_loss_m1, op=dist.ReduceOp.AVG)
            dist.all_reduce(edm_loss_m2, op=dist.ReduceOp.AVG)
        else:
            edm_loss_m1 = torch.ones(1)
            edm_loss_m2 = torch.ones(1)
        iter_count = self.iter_count
        self.reset()
        return edm_loss_m1.tolist(), edm_loss_m2.tolist(), iter_count


class FrameLossLog(Callback):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.save_s3 = save_s3
        self.logging_iter_multipler = logging_iter_multipler
        self.save_logging_iter_multipler = save_logging_iter_multipler
        self.name = self.__class__.__name__

        self.train_image_log = _LossRecord()
        self.train_video_log = _LossRecord()

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ):
        skip_update_due_to_unstable_loss = False
        if torch.isnan(loss) or torch.isinf(loss):
            skip_update_due_to_unstable_loss = True
            log.critical(
                f"Unstable loss {loss} at iteration {iteration} with is_image_batch: {model.is_image_batch(data_batch)}",
                rank0_only=False,
            )

        if not skip_update_due_to_unstable_loss and "edm_loss_per_frame" in output_batch:
            _loss = output_batch["edm_loss_per_frame"].detach().mean(dim=0)
            if model.is_image_batch(data_batch):
                self.train_image_log.iter_count += 1
                self.train_image_log.edm_loss_m1 += _loss
                self.train_image_log.edm_loss_m2 += _loss**2

            else:
                self.train_video_log.iter_count += 1
                self.train_video_log.edm_loss_m1 += _loss
                self.train_video_log.edm_loss_m2 += _loss**2

        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0:
            world_size = dist.get_world_size()
            image_edm_loss_m1, image_edm_loss_m2, img_iter_count = self.train_image_log.get_stat()
            video_edm_loss_m1, video_edm_loss_m2, vid_iter_count = self.train_video_log.get_stat()
            img_iter_count *= world_size
            vid_iter_count *= world_size

            if distributed.is_rank0():
                info = {}
                if vid_iter_count > 0:
                    info["frame_loss_log/video_sample"] = vid_iter_count
                    for i, (m1, m2) in enumerate(zip(video_edm_loss_m1, video_edm_loss_m2)):
                        info[f"frame_loss_log/video_edm_loss_{i}"] = m1
                        info[f"frame_loss_log_sq/video_edm_loss_{i}"] = m2
                if img_iter_count > 0:
                    info["frame_loss_log/image_sample"] = img_iter_count
                    for i, (m1, m2) in enumerate(zip(image_edm_loss_m1, image_edm_loss_m2)):
                        info[f"frame_loss_log/image_edm_loss_{i}"] = m1
                        info[f"frame_loss_log_sq/image_edm_loss_{i}"] = m2

                if info:
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

                    if wandb.run:
                        wandb.log(info, step=iteration)
