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

import math
import os
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as torchvision_F
import wandb
from einops import rearrange, repeat
from megatron.core import parallel_state

from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import distributed, log, misc
from cosmos_predict2._src.imaginaire.utils.callback import Callback
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video


def resize_image(image: torch.Tensor, size: int = 1024) -> torch.Tensor:
    """
    Resize the image to the given size. This is done so that wandb can display the image correctly.
    """
    _, h, w = image.shape
    ratio = size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    return torchvision_F.resize(image, (new_h, new_w))


def is_primitive(value):
    return isinstance(value, (int, float, str, bool, type(None)))


def convert_to_primitive(value):
    if isinstance(value, (list, tuple)):
        return [convert_to_primitive(v) for v in value if is_primitive(v) or isinstance(v, (list, dict))]
    elif isinstance(value, dict):
        return {k: convert_to_primitive(v) for k, v in value.items() if is_primitive(v) or isinstance(v, (list, dict))}
    elif is_primitive(value):
        return value
    else:
        return "non-primitive"  # Skip non-primitive types


class ValidationDrawSample(Callback):
    """
    This callback sample condition inputs from validation data, run inference and save the results to wandb and s3.

    Args:
        n_samples (int): The number of samples to run inference on.
        n_viz_sample (int, optional): for each batch, min(n_viz_sample, batch_size) samples will be saved to wandb. Defaults to 3.
        n_sample_to_save (int, optional): number of samples to save. The actual number of samples to save is min(n_sample_to_save, data parallel instances). Defaults to 128.
        num_sampling_step (int, optional): number of sampling steps. Defaults to 35.
        guidance (List[float], optional): guidance scale. Defaults to [0.0, 3.0, 7.0].
        do_x0_prediction (bool, optional): whether to do x0 prediction. Defaults to True.
        n_sigmas_for_x0_prediction (int, optional): number of sigmas to use for x0 prediction. Defaults to 4.
        save_s3 (bool, optional): whether to save to s3. Defaults to False.
        is_ema (bool, optional): whether the callback is run for ema model. Defaults to False.
        use_negative_prompt (bool, optional): whether to use negative prompt. Defaults to False.
        fps (int, optional): frames per second when saving the video. Defaults to 16.
    """

    def __init__(
        self,
        n_samples: int,
        n_viz_sample: int = 3,
        n_sample_to_save: int = 128,
        num_sampling_step: int = 35,
        guidance: List[float] = [0.0, 3.0, 7.0],
        do_x0_prediction: bool = True,
        n_sigmas_for_x0_prediction: int = 4,
        save_s3: bool = False,
        is_ema: bool = False,
        use_negative_prompt: bool = False,
        prompt_type: str = "t5_xxl",
        fps: int = 16,
        run_at_start: bool = False,
        barrier_after_run: bool = True,
    ):
        # s3: # files: min(n_sample_to_save, data instance)  # per file: min(batch_size, n_viz_sample)
        # wandb: 1 file, # per file: min(batch_size, n_viz_sample)
        self.barrier_after_run = barrier_after_run
        self.run_at_start = run_at_start

        self.n_viz_sample = n_viz_sample
        self.n_sample_to_save = n_sample_to_save
        self.save_s3 = save_s3
        self.do_x0_prediction = do_x0_prediction
        self.n_sigmas_for_x0_prediction = n_sigmas_for_x0_prediction
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.use_negative_prompt = use_negative_prompt
        self.prompt_type = prompt_type
        self.guidance = guidance
        self.num_sampling_step = num_sampling_step
        self.rank = distributed.get_rank()
        self.fps = fps
        self.n_samples = n_samples
        self.sample_counter = 0

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        self.wandb_online = self.config.job.wandb_mode == "online"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")

        if parallel_state.is_initialized():
            self.data_parallel_id = parallel_state.get_data_parallel_rank()
        else:
            self.data_parallel_id = self.rank

        if self.use_negative_prompt:
            if self.prompt_type == "t5_xxl":
                self.negative_prompt_data = easy_io.load(
                    "s3://bucket/edify_video/v4/validation/item_dataset/negative_prompt/000000.pkl"
                )
            elif self.prompt_type == "umt5_xxl":
                self.negative_prompt_data = easy_io.load(
                    "s3://bucket/edify_video/v4/validation/item_dataset/negative_prompt/umt5_neg.pt"
                )
            else:
                raise ValueError(f"Invalid prompt type: {self.prompt_type}")

    @misc.timer("ValidationDrawSample: x0")
    @torch.no_grad()
    def x0_pred(self, model, data_batch, iteration):
        tag = "ema" if self.is_ema else "reg"

        log.debug("starting data and condition model", rank0_only=False)

        raw_data, x0, condition = model.get_data_and_condition(data_batch)
        _, condition, x0, _ = model.broadcast_split_for_model_parallelsim(None, condition, x0, None)

        log.debug("done data and condition model", rank0_only=False)
        batch_size = x0.shape[0]
        sigmas = np.exp(
            np.linspace(
                math.log(model.sde.sigma_min), math.log(model.sde.sigma_max), self.n_sigmas_for_x0_prediction + 1
            )[1:]
        )

        to_show = []
        generator = torch.Generator(device="cuda")
        generator.manual_seed(0)
        random_noise = torch.randn(*x0.shape, generator=generator, **model.tensor_kwargs)
        _ones = torch.ones(batch_size, **model.tensor_kwargs)
        mse_loss_list = []
        for _, sigma in enumerate(sigmas):
            x_sigma = sigma * random_noise + x0
            log.debug(f"starting denoising {sigma}", rank0_only=False)
            sample = model.denoise(x_sigma, _ones * sigma, condition).x0
            log.debug(f"done denoising {sigma}", rank0_only=False)
            mse_loss = distributed.dist_reduce_tensor(F.mse_loss(sample, x0))
            mse_loss_list.append(mse_loss)

            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())
        to_show.append(
            raw_data.float().cpu(),
        )

        local_path = self.run_save(to_show, batch_size, iteration, "x0")
        return local_path, torch.tensor(mse_loss_list).cuda(), sigmas

    def on_validation_start(
        self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        self.sample_counter = 0

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        trainer = self.trainer

        past_samples = self.sample_counter * distributed.get_world_size()
        current_samples = past_samples + self.rank

        should_run = (iteration > 0 or self.run_at_start) and past_samples < self.n_samples
        should_save = current_samples < self.n_samples

        if should_run:
            save_log = "" if should_save else " (skipping save/log, running only for GPU synchronization)"
            log.debug(
                f"Callback {self.__class__.__name__} fired on validation_step_end step {iteration} [{current_samples + 1}/{self.n_samples}]{save_log}",
                rank0_only=False,
            )
            self.run_sample(trainer, model, data_batch, iteration, should_save)
            self.sample_counter += 1
            log.debug(
                f"Callback {self.__class__.__name__} finished on validation_step_end step {iteration} [{current_samples + 1}/{self.n_samples}]{save_log}",
                rank0_only=False,
            )

            if self.barrier_after_run:
                distributed.barrier()

    def run_sample(self, trainer, model, data_batch, iteration, should_save):
        tag = "ema" if self.is_ema else "reg"
        sample_counter = getattr(trainer, "sample_counter", iteration)
        batch_info = {
            "data": {
                k: convert_to_primitive(v)
                for k, v in data_batch.items()
                if is_primitive(v) or isinstance(v, (list, dict))
            },
            "sample_counter": sample_counter,
            "iteration": iteration,
        }
        if is_tp_cp_pp_rank0():
            if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
                easy_io.dump(
                    batch_info,
                    f"s3://rundir/{self.name}/BatchInfo_ReplicateID{self.data_parallel_id:04d}_Iter{iteration:09d}.json",
                )

        log.debug("entering, every_n_impl", rank0_only=False)
        if self.do_x0_prediction:
            log.debug("entering, x0_pred", rank0_only=False)
            x0_save_dir, mse_loss, sigmas = self.x0_pred(
                model,
                data_batch,
                iteration,
            )
            log.debug("done, x0_pred", rank0_only=False)
            if self.save_s3 and self.rank == 0:
                easy_io.dump(
                    {
                        "mse_loss": mse_loss.tolist(),
                        "sigmas": sigmas.tolist(),
                        "iteration": iteration,
                    },
                    f"s3://rundir/{self.name}/{tag}_MSE_Iter{iteration:09d}.json",
                )

        log.debug("entering, sample", rank0_only=False)
        sample_save_dir = self.sample(
            model,
            data_batch,
            iteration,
            should_save,
        )
        log.debug("done, sample", rank0_only=False)

        log.debug("waiting for all ranks to finish", rank0_only=False)
        dist.barrier()
        if self.rank == 0 and wandb.run:
            sample_counter = getattr(trainer, "sample_counter", iteration)
            data_type = "image" if model.is_image_batch(data_batch) else "video"
            tag += f"_{data_type}"
            info = {
                "trainer/global_step": iteration,
                "sample_counter": sample_counter,
            }
            if self.do_x0_prediction:
                imgs = []

                assert x0_save_dir is not None
                for fp in os.listdir(x0_save_dir):
                    imgs.append(wandb.Image(os.path.join(x0_save_dir, fp), caption=f"{sample_counter}"))
                info[f"{self.name}/{tag}_x0"] = imgs
                # convert mse_loss to a dict
                mse_loss = mse_loss.tolist()
                info.update({f"x0_pred_mse_{tag}/Sigma{sigmas[i]:0.5f}": mse_loss[i] for i in range(len(mse_loss))})

            assert sample_save_dir is not None
            imgs = []
            for idx, fp in enumerate(os.listdir(sample_save_dir)):
                imgs.append(wandb.Image(os.path.join(sample_save_dir, fp), caption=f"{sample_counter}"))

            info[f"{self.name}/{tag}_sample"] = imgs
            wandb.log(
                info,
                step=iteration,
            )
        torch.cuda.empty_cache()

    @misc.timer("ValidationDrawSample: sample")
    def sample(self, model, data_batch, iteration, should_save):
        tag = "ema" if self.is_ema else "reg"

        # Obtain text embeddings online
        text_encoder_config = getattr(model.config, "text_encoder_config", None)
        if text_encoder_config is not None and text_encoder_config.compute_online:
            text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        raw_data, x0, _ = model.get_data_and_condition(data_batch)
        if self.use_negative_prompt:
            batch_size = x0.shape[0]
            if self.negative_prompt_data["t5_text_embeddings"].shape != data_batch["t5_text_embeddings"].shape:
                data_batch["neg_t5_text_embeddings"] = misc.to(
                    repeat(
                        self.negative_prompt_data["t5_text_embeddings"],
                        "... -> b ...",
                        b=batch_size,
                    ),
                    **model.tensor_kwargs,
                )
            else:
                data_batch["neg_t5_text_embeddings"] = misc.to(
                    self.negative_prompt_data["t5_text_embeddings"],
                    **model.tensor_kwargs,
                )

            assert data_batch["neg_t5_text_embeddings"].shape == data_batch["t5_text_embeddings"].shape, (
                f"{data_batch['neg_t5_text_embeddings'].shape} != {data_batch['t5_text_embeddings'].shape}"
            )
            data_batch["neg_t5_text_mask"] = data_batch["t5_text_mask"]

        to_show = []
        for guidance in self.guidance:
            sample = model.generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                # make sure no mismatch and also works for cp
                state_shape=x0.shape[1:],
                n_sample=x0.shape[0],
                num_steps=self.num_sampling_step,
                is_negative_prompt=self.use_negative_prompt,
            )
            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())

        to_show.append(raw_data.float().cpu())

        batch_size = x0.shape[0]
        if should_save:
            local_path = self.run_save(to_show, batch_size, iteration, "sample")
            return local_path
        return None

    def run_save(self, to_show, batch_size, iteration, save_name) -> Optional[str]:
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]
        is_single_frame = to_show.shape[3] == 1
        n_viz_sample = min(self.n_viz_sample, batch_size)

        save_path = os.path.join(str(iteration), str(self.sample_counter), save_name)

        # ! we only save first n_sample_to_save video!
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            save_img_or_video(
                rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
                f"s3://rundir/{self.name}/{save_path}/{self.rank}",
                fps=self.fps,
            )

        local_save_dir = os.path.join(self.local_dir, save_path)
        os.makedirs(local_save_dir, exist_ok=True)
        local_path = os.path.join(local_save_dir, f"{self.rank}.jpg")

        if self.wandb_online:
            if is_single_frame:  # image case
                to_show = rearrange(
                    to_show[:, :n_viz_sample],
                    "n b c t h w -> t c (n h) (b w)",
                )
                image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
                # resize so that wandb can handle it
                torchvision.utils.save_image(resize_image(image_grid, 1024), local_path, nrow=1, scale_each=True)
            else:
                to_show = to_show[:, :n_viz_sample]  # [n, b, c, 3, h, w]

                # resize 3 frames frames so that we can display them on wandb
                _T = to_show.shape[3]
                three_frames_list = [0, _T // 2, _T - 1]
                to_show = to_show[:, :, :, three_frames_list]
                log_image_size = 1024
                to_show = rearrange(
                    to_show,
                    "n b c t h w -> 1 c (n h) (b t w)",
                )

                # resize so that wandb can handle it
                image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
                torchvision.utils.save_image(
                    resize_image(image_grid, log_image_size), local_path, nrow=1, scale_each=True
                )

        return local_save_dir
