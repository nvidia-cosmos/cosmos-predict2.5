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
Please run the script with the following command:

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t_warmup.py \
--experiment=cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes_release \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000014000/model \
  --input_video_root /project/cosmos/user/gr00t_gr1 \
  --save_root datasets/gr1_warmup_regenerated_4step \
  --base_path_gr00t_gr1_local /project/cosmos/user/datasets/gr1_unified/gr1_unified.RU0226RemoveStaticFreq20 \
  --resolution 480,832 --guidance 0 --chunk_size 12 --start 0 --end 1000 --query_steps 0,9,18,27,34
"""

import argparse
import json
import os

import mediapy
import numpy as np
import torch
import tqdm
from loguru import logger

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import LeRobotDataset
from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (
    ActionVideo2WorldInference,
)

_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", "webp"]
_VIDEO_EXTENSIONS = [".mp4"]

_ACTION_SCALER = 20.0


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the Video2World inference script."""
    parser = argparse.ArgumentParser(description="Image2World/Video2World inference script")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config")
    parser.add_argument("--chunk_size", type=int, default=12, help="Chunk size for action conditioning")
    parser.add_argument("--guidance", type=int, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=1, help="Guidance value")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the checkpoint. If not provided, will use the one specify in the config",
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument(
        "--resolution",
        type=str,
        default="none",
        help="Resolution of the video (H,W). Be default it will use model trained resolution. 9:16",
    )
    parser.add_argument("--input_video_root", type=str, default="bridge/annotation/test_100", help="Action root")
    parser.add_argument("--save_root", type=str, default="results/image2world", help="Save root")
    parser.add_argument(
        "--base_path_gr00t_gr1_local",
        type=str,
        default="/project/cosmos/user/datasets/gr1_unified/gr1_unified.RU0226RemoveStaticFreq20",
        help="Base path to the GR00T GR1 dataset",
    )

    parser.add_argument("--start", type=int, default=0, help="Start index for processing files")
    parser.add_argument("--end", type=int, default=100, help="End index for processing files")

    parser.add_argument(
        "--num_latent_conditional_frames",
        type=int,
        default=1,
        help="Number of latent conditional frames (0, 1 or 2). For images, both values work by duplicating frames. For videos, uses the first N frames.",
    )

    parser.add_argument(
        "--query_steps",
        type=lambda x: [int(i) for i in x.split(",")],
        default="0,9,18,27,34",
        help="Query steps for the diffusion process",
    )
    # Context parallel arguments
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (number of GPUs to split context over). Set to 8 for 8 GPUs",
    )
    return parser.parse_args()


def main():
    torch.enable_grad(False)  # Disable gradient calculations for inference
    args = parse_arguments()

    # Determine supported extensions based on num_latent_conditional_frames
    if args.num_latent_conditional_frames > 1:
        supported_extensions = _VIDEO_EXTENSIONS
        # Check if input folder contains any videos
        has_videos = False
        for file_name in os.listdir(args.input_root):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in _VIDEO_EXTENSIONS:
                has_videos = True
                break

        if not has_videos:
            raise ValueError(
                f"num_latent_conditional_frames={args.num_latent_conditional_frames} > 1 requires video inputs, "
                f"but no videos found in {args.input_root}. Found extensions: "
                f"{set(os.path.splitext(f)[1].lower() for f in os.listdir(args.input_root) if os.path.splitext(f)[1])}"
            )

        logger.info(f"Using video-only mode with {args.num_latent_conditional_frames} conditional frames")
    elif args.num_latent_conditional_frames == 1:
        supported_extensions = _IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS
        logger.info(f"Using image+video mode with {args.num_latent_conditional_frames} conditional frame")

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = LeRobotDataset(
        num_frames=13,
        time_division_factor=4,
        time_division_remainder=1,
        max_pixels=1920 * 1080,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
        dataset_path=args.base_path_gr00t_gr1_local,
        data_split="full",
        embodiment="gr1",
        downscaled_res=False,
    )

    # Initialize the inference handler with context parallel support
    video2world_cli = ActionVideo2WorldInference(
        args.experiment, args.ckpt_path, args.s3_cred, context_parallel_size=args.context_parallel_size
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory usage after model dcp.load: {mem_bytes / (1024**3):.2f} GB")

    os.makedirs(os.path.join(args.save_root, "latents"), exist_ok=True)
    os.makedirs(os.path.join(args.save_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.save_root, "actions"), exist_ok=True)
    os.makedirs(os.path.join(args.save_root, "videos"), exist_ok=True)

    print(len(dataset))
    for idx in tqdm.tqdm((range(args.start, args.end))):
        data = dataset[idx]
        img_np_array = data["video"][:, 0, :, :].permute(1, 2, 0).cpu().numpy()
        video_np_array = data["video"].permute(1, 2, 3, 0).cpu().numpy()
        action = data["action"].cpu().numpy()

        next_img_array, video_clamped, latents_to_save = video2world_cli.step_inference_with_latents(
            img_array=img_np_array,
            action=action,
            guidance=args.guidance,
            seed=0,
            num_latent_conditional_frames=args.num_latent_conditional_frames,
            query_steps=args.query_steps,
        )

        for k in latents_to_save:
            latents_to_save[k] = latents_to_save[k].squeeze(0).cpu()

        torch.save(latents_to_save, os.path.join(args.save_root, "latents", f"{idx}.pt"))
        mediapy.write_image(os.path.join(args.save_root, "images", f"{idx}.png"), img_np_array)
        mediapy.write_video(os.path.join(args.save_root, "videos", f"{idx}.mp4"), video_np_array)
        with open(os.path.join(args.save_root, "actions", f"{idx}.json"), "w") as f:
            json.dump(action.tolist(), f, indent=4)

    exit()


if __name__ == "__main__":
    main()
