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
from contextlib import nullcontext
from functools import partial
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

from cosmos_predict2._src.imaginaire.callbacks.every_n import EveryN
from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import distributed, log, misc
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


class EveryNDrawSample(EveryN):
    """
    This callback sample condition inputs from training data, run inference and save the results to wandb and s3.

    Args:
        every_n (int): The frequency at which the callback is invoked.
        step_size (int, optional): The step size for the callback. Defaults to 1.
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
        every_n: int,
        step_size: int = 1,
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
    ):
        # s3: # files: min(n_sample_to_save, data instance)  # per file: min(batch_size, n_viz_sample)
        # wandb: 1 file, # per file: min(batch_size, n_viz_sample)
        super().__init__(every_n, step_size)

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

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
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

    @misc.timer("EveryNDrawSample: x0")
    @torch.no_grad()
    def x0_pred(self, trainer, model, data_batch, output_batch, loss, iteration):
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

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_x0_Iter{iteration:09d}"

        local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
        return local_path, torch.tensor(mse_loss_list).cuda(), sigmas

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "every_n_sampling")
        else:
            context = nullcontext

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
        with context():
            if self.do_x0_prediction:
                log.debug("entering, x0_pred", rank0_only=False)
                x0_img_fp, mse_loss, sigmas = self.x0_pred(
                    trainer,
                    model,
                    data_batch,
                    output_batch,
                    loss,
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
            sample_img_fp = self.sample(
                trainer,
                model,
                data_batch,
                output_batch,
                loss,
                iteration,
            )
            log.debug("done, sample", rank0_only=False)

            log.debug("waiting for all ranks to finish", rank0_only=False)
            dist.barrier()
        if wandb.run:
            sample_counter = getattr(trainer, "sample_counter", iteration)
            data_type = "image" if model.is_image_batch(data_batch) else "video"
            tag += f"_{data_type}"
            info = {
                "trainer/global_step": iteration,
                "sample_counter": sample_counter,
            }
            if self.do_x0_prediction:
                info[f"{self.name}/{tag}_x0"] = wandb.Image(x0_img_fp, caption=f"{sample_counter}")
                # convert mse_loss to a dict
                mse_loss = mse_loss.tolist()
                info.update({f"x0_pred_mse_{tag}/Sigma{sigmas[i]:0.5f}": mse_loss[i] for i in range(len(mse_loss))})

            info[f"{self.name}/{tag}_sample"] = wandb.Image(sample_img_fp, caption=f"{sample_counter}")
            wandb.log(
                info,
                step=iteration,
            )
        torch.cuda.empty_cache()

    @misc.timer("EveryNDrawSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        tag = "ema" if self.is_ema else "reg"

        # Obtain text embeddings online
        text_encoder_config = getattr(model.config, "text_encoder_config", None)
        if text_encoder_config is not None and text_encoder_config.compute_online:
            text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        raw_data, x0, condition = model.get_data_and_condition(data_batch)
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
                is_negative_prompt=True if self.use_negative_prompt else False,
            )
            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())

        to_show.append(raw_data.float().cpu())

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"

        batch_size = x0.shape[0]
        if is_tp_cp_pp_rank0():
            local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
            return local_path
        return None

    def run_save(self, to_show, batch_size, base_fp_wo_ext) -> Optional[str]:
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]
        is_single_frame = to_show.shape[3] == 1
        n_viz_sample = min(self.n_viz_sample, batch_size)

        # ! we only save first n_sample_to_save video!
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            save_img_or_video(
                rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
                f"s3://rundir/{self.name}/{base_fp_wo_ext}",
                fps=self.fps,
            )

        file_base_fp = f"{base_fp_wo_ext}_resize.jpg"
        local_path = f"{self.local_dir}/{file_base_fp}"

        if self.rank == 0 and wandb.run:
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

            return local_path
        return None
