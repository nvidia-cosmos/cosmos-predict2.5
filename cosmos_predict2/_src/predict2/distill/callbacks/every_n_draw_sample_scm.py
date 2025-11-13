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
import random
from contextlib import nullcontext
from functools import partial
from typing import Optional

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


resolution2hw = {"720": (704, 1280)}


def sample_batch_image(resolution: str = "512", batch_size: int = 1):
    h, w = resolution2hw[resolution]
    data_batch = {
        "dataset_name": "image_data",
        "images": torch.randn(batch_size, 3, h, w).cuda(),
        "t5_text_embeddings": torch.randn(batch_size, 512, 1024).cuda(),
        "fps": torch.randint(16, 32, (batch_size,)).cuda(),
        "padding_mask": torch.zeros(batch_size, 1, h, w).cuda(),
    }
    return data_batch


def sample_batch_video(resolution: str = "512", batch_size: int = 1, num_frames: int = 17):
    h, w = resolution2hw[resolution]
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


video_prompts = [
    "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
    "A dramatic and dynamic scene in the style of a disaster movie, depicting a powerful tsunami rushing through a narrow alley in Bulgaria. The water is turbulent and chaotic, with waves crashing violently against the walls and buildings on either side. The alley is lined with old, weathered houses, their facades partially submerged and splintered. The camera angle is low, capturing the full force of the tsunami as it surges forward, creating a sense of urgency and danger. People can be seen running frantically, adding to the chaos. The background features a distant horizon, hinting at the larger scale of the tsunami. A dynamic, sweeping shot from a low-angle perspective, emphasizing the movement and intensity of the event.",
    "Animated scene features a close-up of a short fluffy monster kneeling beside a melting red candle. The art style is 3D and realistic, with a focus on lighting and texture. The mood of the painting is one of wonder and curiosity, as the monster gazes at the flame with wide eyes and open mouth. Its pose and expression convey a sense of innocence and playfulness, as if it is exploring the world around it for the first time. The use of warm colors and dramatic lighting further enhances the cozy atmosphere of the image.",
    "The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it’s tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds.",
    "A close up view of a glass sphere that has a zen garden within it. There is a small dwarf in the sphere who is raking the zen garden and creating patterns in the sand.",
    "The camera rotates around a large stack of vintage televisions all showing different programs — 1950s sci-fi movies, horror movies, news, static, a 1970s sitcom, etc, set inside a large New York museum gallery.",
    "A close-up shot of a ceramic teacup slowly pouring water into a glass mug. The water flows smoothly from the spout of the teacup into the mug, creating gentle ripples as it fills up. Both cups have detailed textures, with the teacup having a matte finish and the glass mug showcasing clear transparency. The background is a blurred kitchen countertop, adding context without distracting from the central action. The pouring motion is fluid and natural, emphasizing the interaction between the two cups.",
    "A dynamic and chaotic scene in a dense forest during a heavy rainstorm, capturing a real girl frantically running through the foliage. Her wild hair flows behind her as she sprints, her arms flailing and her face contorted in fear and desperation. Behind her, various animals—rabbits, deer, and birds—are also running, creating a frenzied atmosphere. The girl's clothes are soaked, clinging to her body, and she is screaming and shouting as she tries to escape. The background is a blur of greenery and rain-drenched trees, with occasional glimpses of the darkening sky. A wide-angle shot from a low angle, emphasizing the urgency and chaos of the moment.",
    "A playful raccoon is seen playing an electronic guitar, strumming the strings with its front paws. The raccoon has distinctive black facial markings and a bushy tail. It sits comfortably on a small stool, its body slightly tilted as it focuses intently on the instrument. The setting is a cozy, dimly lit room with vintage posters on the walls, adding a retro vibe. The raccoon's expressive eyes convey a sense of joy and concentration. Medium close-up shot, focusing on the raccoon's face and hands interacting with the guitar.",
]


class EveryNDrawSample(EveryN):
    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        n_sample_to_save: int = 128,
        num_sampling_step: int = 35,
        is_sample: bool = True,
        save_s3: bool = False,
        is_ema: bool = False,
        use_negative_prompt: bool = False,
        show_all_frames: bool = False,
        is_image: bool = False,
        num_samples: int = 10,
        sample_fix: bool = True,
    ):
        super().__init__(every_n, step_size, run_at_start=True)
        self.n_sample_to_save = n_sample_to_save
        self.save_s3 = save_s3
        self.is_sample = is_sample
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.use_negative_prompt = use_negative_prompt
        self.show_all_frames = show_all_frames
        self.num_sampling_step = num_sampling_step
        self.rank = distributed.get_rank()
        assert not use_negative_prompt
        self.is_image = is_image
        self.num_samples = num_samples
        self.sample_fix = sample_fix

        log.critical("only use val_data_distillation_t2i_t5_v0.py to get val data")

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
            self.negative_prompt_data = easy_io.load(get_itemdataset_option("negative_prompt_v0_s3").path)

        if self.sample_fix:
            assert model.config.text_encoder_config is not None and model.config.text_encoder_config.compute_online
            self.kv_prompt_to_emb = {}
            for prompt in video_prompts:
                log.info(f"Computing embedding for prompt: {prompt}")
                data_batch = {model.input_caption_key: [prompt]}
                text_emb = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
                self.kv_prompt_to_emb[prompt] = text_emb

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
                if self.sample_fix:
                    self.sample_fixed(
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

        tag = "ema" if self.is_ema else "reg"
        frames_options = list(model.config.conditional_frames_probs.keys())
        weights = list(model.config.conditional_frames_probs.values())
        data_batch["num_conditional_frames"] = random.choices(frames_options, weights=weights, k=1)[0]
        raw_data, x0, _, _ = model.get_data_and_condition(data_batch)

        to_show = []
        sample_1 = model.generate_samples_from_batch(
            data_batch,
            # make sure no mismatch and also works for cp
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            mid_t=[],
        )
        if hasattr(model, "decode"):
            sample_1 = model.decode(sample_1)
        to_show.append(sample_1.float().cpu())

        sample_2 = model.generate_samples_from_batch(
            data_batch,
            # make sure no mismatch and also works for cp
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            mid_t=[1.3],
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
            log.info(f"Generating with prompt: {prompt}")
            # set prompt and make embeddings match batch
            data_batch[model.input_caption_key] = [prompt] * self.num_samples
            # create embeddings and move to model tensor kwargs (likely cuda + bfloat16)
            emb = repeat(text_emb.to(**model.tensor_kwargs), "b l d -> (k b) l d", k=self.num_samples)
            # ensure no autograd graph / no extra refs
            emb = emb.detach()
            data_batch["t5_text_embeddings"] = emb
            data_batch["t5_text_mask"] = torch.ones(
                data_batch["t5_text_embeddings"].shape[0], data_batch["t5_text_embeddings"].shape[1], device="cuda"
            )

            # generate samples (we are already inside a @torch.no_grad() scope from every_n_impl)
            sample = model.generate_samples_from_batch(data_batch, seed=1)

            # decode if needed
            if hasattr(model, "decode"):
                video = model.decode(sample)
            else:
                video = sample

            # move final video to CPU and detach to break GPU refs
            video_cpu = video.detach().float().cpu()
            to_show.append(video_cpu)

            # delete large GPU objects and free cache immediately
            del sample
            del video
            del emb
            torch.cuda.synchronize()  # optional but helpful to ensure kernels finished
            torch.cuda.empty_cache()

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
