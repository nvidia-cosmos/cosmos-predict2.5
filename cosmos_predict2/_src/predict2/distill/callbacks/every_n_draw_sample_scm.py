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

import os
from contextlib import nullcontext
from functools import partial
from typing import List, Optional

import torch
import torch.distributed as dist
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
from cosmos_predict2._src.predict2.datasets.data_sources.item_datasets_for_validation import get_itemdataset_option
from cosmos_predict2._src.predict2.distill.datasets.val_data_distillation_t2i_t5_v0 import get_val_data


# use first two rank to generate some images for visualization
def resize_image(image: torch.Tensor, size: int = 1024) -> torch.Tensor:
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


def sample_batch_image(resolution: str = "512", batch_size: int = 1):
    h, w = int(resolution), int(resolution)
    data_batch = {
        "dataset_name": "image_data",
        "images": torch.randn(batch_size, 3, h, w).cuda(),
        "t5_text_embeddings": torch.randn(batch_size, 512, 1024).cuda(),
        "fps": torch.randint(16, 32, (batch_size,)).cuda(),
        "padding_mask": torch.zeros(batch_size, 1, h, w).cuda(),
    }
    return data_batch


def sample_batch_video(resolution: str = "512", batch_size: int = 1, num_frames: int = 17):
    h, w = int(resolution), int(resolution)
    data_batch = {
        "dataset_name": "video_data",
        "video": torch.randint(0, 256, (batch_size, 3, num_frames, h, w), dtype=torch.uint8).cuda(),
        "t5_text_embeddings": torch.randn(batch_size, 512, 1024).cuda(),
        "fps": torch.randint(16, 32, (batch_size,)).cuda(),
        "padding_mask": torch.zeros(batch_size, 1, h, w).cuda(),
    }
    return data_batch


def get_sample_batch(
    num_frames: int = 17,
    resolution: str = "512",
    batch_size: int = 1,
) -> torch.Tensor:
    if num_frames == 1:
        data_batch = sample_batch_image(resolution, batch_size)
    else:
        data_batch = sample_batch_video(resolution, batch_size, num_frames)

    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
            data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

    return data_batch


class EveryNDrawSample(EveryN):
    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        fix_batch_fp: Optional[str] = None,
        n_x0_level: int = 4,
        n_viz_sample: int = 3,
        n_sample_to_save: int = 128,
        num_sampling_step: int = 35,
        guidance: List[float] = [3.0, 7.0, 9.0, 13.0],
        do_x0_prediction: bool = True,
        is_sample: bool = True,
        save_s3: bool = False,
        is_ema: bool = False,
        use_negative_prompt: bool = False,
        show_all_frames: bool = False,
        is_image: bool = True,
        num_samples: int = 10,
        **kwargs,
    ):
        super().__init__(every_n, step_size)
        self.fix_batch = fix_batch_fp
        self.n_x0_level = n_x0_level
        self.n_viz_sample = n_viz_sample
        self.n_sample_to_save = n_sample_to_save
        self.save_s3 = save_s3
        self.do_x0_prediction = do_x0_prediction
        self.is_sample = is_sample
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.use_negative_prompt = use_negative_prompt
        self.show_all_frames = show_all_frames
        self.guidance = guidance
        self.num_sampling_step = num_sampling_step
        self.rank = distributed.get_rank()
        assert not use_negative_prompt
        self.is_image = is_image
        self.num_samples = num_samples

        log.critical("only use val_data_distillation_t2i_t5_v0.py to get val data")

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")

        if self.fix_batch is not None:
            with misc.timer(f"loading fix_batch {self.fix_batch}"):
                self.fix_batch = misc.co(easy_io.load(self.fix_batch), "cpu")

        if parallel_state.is_initialized():
            self.data_parallel_id = parallel_state.get_data_parallel_rank()
        else:
            self.data_parallel_id = self.rank

        if self.use_negative_prompt:
            self.negative_prompt_data = easy_io.load(get_itemdataset_option("negative_prompt_v0_s3").path)

        if model.model_replicate_rank() == 0:
            self.kv_prompt_to_emb = get_val_data()
        else:
            self.kv_prompt_to_emb = None

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
            log.debug("entering, ema", rank0_only=False)
            # we only use rank0 and rank to generate images and save
            # other rank run forward pass to make sure it works for FSDP
            log.debug("entering, fsdp", rank0_only=False)
            if self.is_sample:
                log.debug("entering, sample", rank0_only=False)
                sample_img_fp, MSE = self.sample(
                    trainer,
                    model,
                    data_batch,
                    output_batch,
                    loss,
                    iteration,
                )
                if self.kv_prompt_to_emb:
                    self.sample_fixed(
                        trainer,
                        model,
                        data_batch,
                        output_batch,
                        loss,
                        iteration,
                    )
                log.debug("done, sample", rank0_only=False)
            if self.fix_batch is not None:
                misc.to(self.fix_batch, "cpu")

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

            if self.is_sample:
                info[f"{self.name}/{tag}_sample"] = wandb.Image(sample_img_fp, caption=f"{sample_counter}")
                info[f"{self.name}/{tag}_MSE"] = MSE
            wandb.log(
                info,
                step=iteration,
            )
        torch.cuda.empty_cache()

    @misc.timer("EveryNDrawSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        """
        Args:
            skip_save: to make sure FSDP can work, we run forward pass on all ranks even though we only save on rank 0 and 1
        """
        if self.fix_batch is not None:
            data_batch = misc.to(self.fix_batch, **model.tensor_kwargs)

        tag = "ema" if self.is_ema else "reg"
        raw_data, x0, _, _ = model.get_data_and_condition(data_batch)

        to_show = []
        sample_1 = model.generate_samples_from_batch(
            data_batch,
            # make sure no mismatch and also works for cp
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
        )
        if hasattr(model, "decode"):
            sample_1 = model.decode(sample_1)
        to_show.append(sample_1.float().cpu())

        sample_2 = model.generate_samples_from_batch(
            data_batch,
            # make sure no mismatch and also works for cp
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            mid_t=[1.0],
        )
        if hasattr(model, "decode"):
            sample_2 = model.decode(sample_2)
        to_show.append(sample_2.float().cpu())

        sample_teacher = model.generate_samples_from_batch_teacher(
            data_batch,
            # make sure no mismatch and also works for cp
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            num_steps=self.num_sampling_step,
        )
        if hasattr(model, "decode"):
            sample_teacher = model.decode(sample_teacher)

        to_show.append(sample_teacher.float().cpu())

        MSE = torch.mean((sample_1.float() - sample_teacher.float()) ** 2)
        dist.all_reduce(MSE, op=dist.ReduceOp.AVG)

        to_show.append(raw_data.float().cpu())
        # visualize input video
        if "hint_key" in data_batch:
            hint = data_batch[data_batch["hint_key"]]
            for idx in range(0, hint.size(1), 3):
                x_rgb = hint[:, idx : idx + 3]
                to_show.append(x_rgb.float().cpu())

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"

        batch_size = x0.shape[0]
        if is_tp_cp_pp_rank0():
            local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
            return local_path, MSE.cpu().item()
        return None, None

    @misc.timer("EveryNDrawSample: sample_fixed")
    def sample_fixed(self, trainer, model, data_batch, output_batch, loss, iteration):
        """
        Args:
            skip_save: to make sure FSDP can work, we run forward pass on all ranks even though we only save on rank 0 and 1
        """

        tag = "ema" if self.is_ema else "reg"

        to_show = []

        data_batch = get_sample_batch(
            num_frames=(
                1 if self.is_image else model.tokenizer.get_pixel_num_frames(model.get_num_video_latent_frames())
            ),
            resolution=model.config.resolution,
            batch_size=self.num_samples,
        )

        for prompt, text_emb in self.kv_prompt_to_emb.items():
            log.debug(f"Generating with prompt: {prompt}")
            data_batch["t5_text_embeddings"] = repeat(
                text_emb.to(**model.tensor_kwargs), "b l d -> (k b) l d", k=self.num_samples
            )

            # generate samples
            sample = model.generate_samples_from_batch(data_batch, seed=1, mid_t=[1.3])

            if hasattr(model, "decode"):
                video = model.decode(sample)

            to_show.append(video.float().cpu())

        base_fp_wo_ext = f"0_{tag}_Sample_Iter{iteration:09d}"

        if is_tp_cp_pp_rank0() and self.data_parallel_id == 0:
            self.run_save(to_show, self.num_samples, base_fp_wo_ext)

    def run_save(self, to_show, batch_size, base_fp_wo_ext) -> Optional[str]:
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]
        is_single_frame = to_show.shape[3] == 1
        # n_viz_sample = min(self.n_viz_sample, batch_size)
        n_viz_sample = batch_size

        # ! we only save first n_sample_to_save video!
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            save_img_or_video(
                rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
                f"s3://rundir/{self.name}/{base_fp_wo_ext}",
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
                if not self.show_all_frames:
                    # resize 3 frames frames so that we can display them on wandb
                    _T = to_show.shape[3]
                    three_frames_list = [0, _T // 2, _T - 1]
                    to_show = to_show[:, :, :, three_frames_list]
                    log_image_size = 1024
                else:
                    log_image_size = 512 * to_show.shape[3]
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
