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

import argparse
import os
import time

import torch
import torch._dynamo
from einops import repeat

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_uri
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.datasets.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO
from cosmos_predict2._src.predict2.distill.utils.model_loader import load_model_from_checkpoint

torch._dynamo.config.suppress_errors = True

"""
Example command with 4 GPUs:

save_root=results_vis/distillation_inference_results_$(date +%Y%m%d)
sigma_max=80
experiment=dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V
iter=iter_000007500
resolution=720
cp_size=1

PYTHONPATH=. torchrun --nproc_per_node=${cp_size} cosmos_predict2/_src/predict2/distill/inference/eval_distilled_predict2p5_model_cli_single.py \
    --experiment ${experiment} \
    --ckpt_iter ${iter} \
    --num_samples 1 \
    --net_type student \
    --cp_size=${cp_size} \
    --save_root ${save_root} \
    --sigma_max ${sigma_max} \
    --tag ${experiment}_${iter}
"""

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


torch.enable_grad(False)

DEFAULT_POSITIVE_PROMPT = "The vibrant and detailed illustration features a dragon with a lizard's head. The dragon, central to the image, exhibits a dynamic pose with its head turned to the left and its body facing right. Its scales are a blend of blue, red, and orange, with intricate patterns. The dragon's eyes are striking blue, and it has a small horn on its head. Surrounding the dragon are splashes of water and paint, creating a sense of movement and energy. Additionally, various objects, including a skull and a flower, add complexity and richness to the image. Signed in the bottom right corner, the artwork showcases the artist's signature and creates a visually striking and engaging piece of art."

ratio = "9,16"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="simple t2v inference script")
    parser.add_argument("--experiment", type=str, default="???", help="inference only config")
    parser.add_argument("--tag", type=str, default="???", help="Tag for the run")
    parser.add_argument(
        "--s3_checkpoint_dir",
        type=str,
        default="",
        help="Path to the checkpoint. If not provided, will use the one specify in the config",
    )
    parser.add_argument(
        "--generate_image",
        dest="is_image",
        action="store_true",
        default=False,
        help="Generate image (default: generate video)",
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument("--ckpt_iter", type=str, default="iter_000004500", help="Checkpoint iteration")
    parser.add_argument("--cp_size", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--num_steps", type=int, default=4, help="Number of steps to generate")
    parser.add_argument("--cache_dir", type=str, default="./predict2_distill_cache_ckpts", help="Cache directory")
    parser.add_argument("--sigma_max", type=int, default=80, help="Sigma max")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--resolution", type=str, default="720", help="Resolution to generate")
    parser.add_argument(
        "--net_type", choices=["student", "teacher"], default="teacher", help="Net type, choose from student or teacher"
    )
    parser.add_argument(
        "--save_root", type=str, default="results_vis/rcm_inference_results", help="Root directory to save the samples"
    )
    parser.add_argument("--save_prompt", action="store_true", default=False, help="Save prompt as txt file")
    parser.add_argument(
        "--load_teacher",
        action="store_true",
        default=False,
        help="Load teacher checkpoint during model init (slower, not needed for inference)",
    )
    return parser.parse_args()


def sample_batch_image(resolution: str = "512", batch_size: int = 1):
    h, w = IMAGE_RES_SIZE_INFO[resolution][ratio]
    data_batch = {
        "dataset_name": "image_data",
        "images": torch.randn(batch_size, 3, h, w).cuda(),
        "t5_text_embeddings": torch.randn(batch_size, 512, 1024).cuda(),
        "fps": torch.randint(16, 32, (batch_size,)).cuda(),
        "padding_mask": torch.zeros(batch_size, 1, h, w).cuda(),
    }
    return data_batch


def sample_batch_video(resolution: str = "512", batch_size: int = 1, num_frames: int = 17):
    ratio = "9,16"
    h, w = VIDEO_RES_SIZE_INFO[resolution][ratio]
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


def main(args):
    save_folder = os.path.join(
        args.save_root, f"{args.experiment}_{args.ckpt_iter}_res{args.resolution}_smax{args.sigma_max}", args.net_type
    )
    os.makedirs(save_folder, exist_ok=True)
    ckpt_path = get_checkpoint_uri("575edf0f-d973-4c74-b52c-69929a08d0a5")
    registered_exp_name = args.experiment
    exp_override_opts = []

    if "dmd2" not in args.experiment:
        exp_override_opts.append(f"model.config.sde.sigma_max={args.sigma_max}")
    if exp_override_opts is None:
        exp_override_opts = []

    device_rank = 0
    process_group = None
    if args.cp_size > 1:
        from megatron.core import parallel_state

        from cosmos_predict2._src.imaginaire.utils import distributed

        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.cp_size)
        process_group = parallel_state.get_context_parallel_group()
        device_rank = distributed.get_rank(process_group)

    # instantiate model, config and load checkpoint
    model, config = load_model_from_checkpoint(
        experiment_name=registered_exp_name,
        s3_checkpoint_dir=ckpt_path,
        enable_fsdp=False,
        load_ema_to_reg=True,
        local_cache_dir=None,
        config_file="cosmos_predict2/_src/predict2/distill/configs/registry_predict2p5.py",
        experiment_opts=exp_override_opts,
        skip_teacher_init=not args.load_teacher,  # only needed if want to inference with teacher for comparison
    )

    log.info("\n\n============Setting net_fake_score to None")
    model.net_fake_score = None

    # Only enable CP for student network here, teacher CP is handled inside velocity_fn
    if process_group is not None:
        log.info("Enabling CP in student network\n")
        model.net.enable_context_parallel(process_group)

    data_batch = get_sample_batch(
        num_frames=1 if args.is_image else model.tokenizer.get_pixel_num_frames(model.get_num_video_latent_frames()),
        resolution=args.resolution,
        batch_size=args.num_samples,
    )
    data_batch["num_conditional_frames"] = 0  # text2world model

    for prompt_idx, prompt in enumerate(video_prompts):
        log.info(f"Generating with prompt {prompt_idx}: {prompt}")
        text_embeddings = model.text_encoder.compute_text_embeddings_online(
            {"ai_caption": [prompt], "images": None}, input_caption_key="ai_caption"
        )
        data_batch["t5_text_embeddings"] = repeat(
            text_embeddings.to(**model.tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples
        )

        # generate samples
        if args.net_type == "teacher":
            log.info("Generating with teacher model")
            start_gen = time.perf_counter()
            sample = model.generate_samples_from_batch_teacher(data_batch, num_steps=args.num_steps, seed=1)
            end_gen = time.perf_counter()
            print(f"==============TEACHER============ Generation took {end_gen - start_gen:.2f} seconds")
        else:
            log.info("Generating with student model")
            start_gen = time.perf_counter()
            sample = model.generate_samples_from_batch(data_batch, seed=1)
            end_gen = time.perf_counter()
            print(f"==============STUDENT============ Generation took {end_gen - start_gen:.2f} seconds")

        if hasattr(model, "decode"):
            start_dec = time.perf_counter()
            video = model.decode(sample)
            end_dec = time.perf_counter()
            print(f"Decoding took {end_dec - start_dec:.2f} seconds")

        video_normalized = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0

        # Save each video individually instead of stacking
        if device_rank == 0:
            for sample_idx in range(args.num_samples):
                individual_save_fn = os.path.join(
                    save_folder,
                    f"{args.net_type}_{args.tag}_prompt{prompt_idx:02d}_steps{args.num_steps}_sample{sample_idx:02d}",
                )
                # Use the same format as the original working code
                save_img_or_video(video_normalized[sample_idx], individual_save_fn)
                log.info(f"Saved individual video to {individual_save_fn}")

                # Also save a text file with the prompt for reference
                if args.save_prompt:
                    prompt_file = individual_save_fn + "_prompt.txt"
                    with open(prompt_file, "w") as f:
                        f.write(prompt)
                    log.info(f"Saved prompt to {prompt_file}")

    # clean up properly
    if args.cp_size > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
