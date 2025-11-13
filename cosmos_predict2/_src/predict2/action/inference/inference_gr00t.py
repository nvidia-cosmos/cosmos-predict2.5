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


# ---------------------------------- benchmark ----------------------------------


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t.py \
--experiment=cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame/checkpoints/iter_000014000 \
  --input_video_root results/gr00t_gr1/gt \
  --save_root results/gr00t_gr1/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame-14k\
  --resolution 480,832 --guidance 0 --chunk_size 12

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t.py \
--experiment=cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame/checkpoints/iter_000020000 \
  --input_video_root results/gr00t_gr1/gt \
  --save_root results/gr00t_gr1/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame-20k\
  --resolution 480,832 --guidance 0 --chunk_size 12 --start 80 --end 100

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t.py \
--experiment=cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame/checkpoints/iter_000028000 \
  --input_video_root results/gr00t_gr1/gt \
  --save_root results/gr00t_gr1/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame-28k\
  --resolution 480,832 --guidance 0 --chunk_size 12 --start 0 --end 100

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t.py \
--experiment=cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000004000 \
  --input_video_root results/gr00t_gr1/gt \
  --save_root results/gr00t_gr1/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes-4k\
  --resolution 480,832 --guidance 0 --chunk_size 12 --start 80 --end 100

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t.py \
--experiment=cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_49frame_full \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_49frame_full/checkpoints/iter_000004000 \
  --input_video_root results/gr00t_gr1/gt \
  --save_root results/gr00t_gr1/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_49frame_full-6k\
  --resolution 480,832 --guidance 0 --chunk_size 48 --start 80 --end 100

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t.py \
--experiment=cosmos_predict2p5_2B_conditioned_posttrained_rl_merged_action_gr00t_gr1_customized_13frame_full \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_conditioned_posttrained_rl_merged_action_gr00t_gr1_customized_13frame_full/checkpoints/iter_000004000 \
  --input_video_root results/gr00t_gr1/gt \
  --save_root results/gr00t_gr1/cosmos_predict2p5_2B_conditioned_posttrained_rl_merged_action_gr00t_gr1_customized_13frame_full-4k\
  --resolution 480,832 --guidance 0 --chunk_size 12 --start 0 --end 100

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t.py \
--experiment=cosmos_predict2p5_2B_conditioned_posttrained_rl_merged_action_gr00t_gr1_customized_13frame_full \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_conditioned_posttrained_rl_merged_action_gr00t_gr1_customized_13frame_full/checkpoints/iter_000008000 \
  --input_video_root results/gr00t_gr1/gt \
  --save_root results/gr00t_gr1/cosmos_predict2p5_2B_conditioned_posttrained_rl_merged_action_gr00t_gr1_customized_13frame_full-8k\
  --resolution 480,832 --guidance 0 --chunk_size 12 --start 0 --end 100

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t.py \
--experiment=cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_49frame_full \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_49frame_full/checkpoints/iter_000010000 \
  --input_video_root results/gr00t_gr1/gt \
  --save_root results/gr00t_gr1/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_49frame_full-10k\
  --resolution 480,832 --guidance 0 --chunk_size 48 --start 90 --end 100

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t.py \
--experiment=cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_73frame_full \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_73frame_full/checkpoints/iter_000006000 \
  --input_video_root results/gr00t_gr1/gt \
  --save_root results/gr00t_gr1/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_73frame_full-6k\
  --resolution 480,832 --guidance 0 --chunk_size 72 --start 70 --end 80

CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t.py \
--experiment=cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000008000 \
  --input_video_root results/gr00t_gr1/gt \
  --save_root results/gr00t_gr1/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes-8k\
  --resolution 480,832 --guidance 0 --chunk_size 12 --start 75 --end 100


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t.py \
--experiment=cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000010000 \
  --input_video_root results/gr00t_gr1/gt \
  --save_root results/gr00t_gr1/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes-10k\
  --resolution 480,832 --guidance 0 --chunk_size 12 --start 90 --end 100

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t.py \
--experiment=cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000014000 \
  --input_video_root results/gr00t_gr1/gt \
  --save_root results/gr00t_gr1/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes-14k\
  --resolution 480,832 --guidance 0 --chunk_size 12 --start 90 --end 100


CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t.py \
--experiment=cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000016000 \
  --input_video_root results/gr00t_gr1/gt \
  --save_root results/gr00t_gr1/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes-16k\
  --resolution 480,832 --guidance 0 --chunk_size 12 --start 0 --end 100
"""

import argparse
import os
from glob import glob

import mediapy
import numpy as np
import torch
from loguru import logger

from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (
    _DEFAULT_NEGATIVE_PROMPT,
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
    parser.add_argument(
        "--num_chunks", type=int, default=12, help="Number of chunks to generate (-1 for all available chunks)"
    )
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

    # for pi dataset
    parser.add_argument("--camera_id", type=str, default="base", help="Camera id")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--fps_downsample_ratio", type=int, default=1)
    parser.add_argument("--gripper_scale", type=float, default=1.0)
    parser.add_argument("--gripper_key", type=str, default="continuous_gripper_state", help="Gripper key")
    parser.add_argument("--state_key", type=str, default="state", help="State key")

    parser.add_argument("--reverse", action="store_true", help="Reverse the video")
    parser.add_argument("--single_chunk", action="store_true", help="Single chunk")
    parser.add_argument("--start_frame_idx", type=int, default=0, help="Start frame index")
    parser.add_argument("--save_fps", type=int, default=10, help="Save fps")

    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Custom negative prompt for classifier-free guidance. If not specified, uses default embeddings from S3.",
    )
    parser.add_argument(
        "--num_latent_conditional_frames",
        type=int,
        default=1,
        help="Number of latent conditional frames (0, 1 or 2). For images, both values work by duplicating frames. For videos, uses the first N frames.",
    )
    # Context parallel arguments
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (number of GPUs to split context over). Set to 8 for 8 GPUs",
    )
    return parser.parse_args()


def get_action_sequence_from_states(
    data,
    fps_downsample_ratio=1,
    use_quat=False,
    state_key="state",
    gripper_scale=1.0,
    gripper_key="continuous_gripper_state",
):
    """
    Get the action sequence from the states.
    """

    actions = np.array(data["action"])[::fps_downsample_ratio][:-1]
    return actions


def get_video_id(img_path: str):
    """Extract video ID from image path by removing directory and extension."""
    return img_path.split("/")[-1].split(".")[0]


def main():
    torch.enable_grad(False)  # Disable gradient calculations for inference
    args = parse_arguments()

    # Validate num_latent_conditional_frames at the very beginning
    if args.num_latent_conditional_frames not in [0, 1, 2]:
        raise ValueError(
            f"num_latent_conditional_frames must be 0, 1 or 2, but got {args.num_latent_conditional_frames}"
        )

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

    # Initialize the inference handler with context parallel support
    video2world_cli = ActionVideo2WorldInference(
        args.experiment, args.ckpt_path, args.s3_cred, context_parallel_size=args.context_parallel_size
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory usage after model dcp.load: {mem_bytes / (1024**3):.2f} GB")

    # get input video and annotation path
    input_video_path = os.path.join(args.input_video_root)

    # Only process files on rank 0 if using distributed processing
    rank0 = True
    if args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    # pdb.set_trace()
    video_list = glob(os.path.join(input_video_path, "*.mp4"))
    input_json_list = [video_path.replace(".mp4", "_actions.npy") for video_path in video_list]

    # Ensure save directory exists
    os.makedirs(args.save_root, exist_ok=True)

    # Process each file in the input directory
    for annotation_path, video_path in zip(input_json_list[args.start : args.end], video_list[args.start : args.end]):
        actions = np.load(annotation_path)

        # Convert camera_id to integer if it's a string and can be converted to an integer

        actions = actions[: len(actions)]
        video_array = mediapy.read_video(video_path)

        # Resize video_array with arg.resolution if specified
        if args.resolution != "none":
            try:
                h, w = map(int, args.resolution.split(","))
                video_array = np.stack([mediapy.resize_image(frame, (h, w)) for frame in video_array], axis=0)
            except Exception as e:
                logger.warning(f"Failed to resize video to {args.resolution}: {e}")

        img_array = video_array[args.start_frame_idx]
        # img_name = annotation_path.split("/")[-1].split(".")[0]
        img_name = video_path.split("/")[-1].split(".")[0]

        frames = [img_array]
        chunk_video = []
        video_array = video_array[:: args.fps_downsample_ratio]

        video_name = f"{args.save_root}/{img_name.replace('.jpg', '.mp4')}"
        chunk_video_name = f"{args.save_root}/{img_name + '.mp4'}"
        logger.info(f"Saving video to {video_name}")
        if os.path.exists(chunk_video_name):
            logger.info(f"Video already exists: {chunk_video_name}")
            continue

        # Calculate the maximum number of chunks to generate
        max_chunks = len(actions) // args.chunk_size
        if args.num_chunks > 0:
            max_chunks = min(max_chunks, args.num_chunks)

        logger.info(f"Generating {max_chunks} chunks (chunk_size={args.chunk_size}, total_actions={len(actions)})")

        chunk_count = 0
        for i in range(args.start_frame_idx, len(actions), args.chunk_size):
            if actions[i : i + args.chunk_size].shape[0] != args.chunk_size:
                break

            # Check if we've reached the desired number of chunks
            if args.num_chunks > 0 and chunk_count >= args.num_chunks:
                logger.info(f"Reached target number of chunks ({args.num_chunks}), stopping generation")
                break

            logger.info(f"Generating chunk {chunk_count + 1}/{max_chunks}")
            next_img_array, video_clamped = video2world_cli.step_inference(
                img_array=img_array,
                action=actions[i : i + args.chunk_size],
                guidance=args.guidance,
                seed=i,
            )
            frames.append(next_img_array)
            img_array = next_img_array
            chunk_video.append(video_clamped)
            chunk_count += 1

            if args.single_chunk:
                break

        chunk_list = [chunk_video[0]] + [chunk_video[i][: args.chunk_size] for i in range(1, len(chunk_video))]
        chunk_video = np.concatenate(chunk_list, axis=0)
        if args.single_chunk:
            chunk_video_name = f"{args.save_root}/{img_name + '_single_chunk.mp4'}"
        else:
            # chunk_video_name = f"{args.save_root}/{img_name + '_chunk.mp4'}"
            chunk_video_name = f"{args.save_root}/{img_name + '.mp4'}"
        mediapy.write_video(chunk_video_name, chunk_video, fps=args.save_fps)

        # concat_video = np.concatenate([chunk_video, video_array[: chunk_video.shape[0]]], axis=2)
        # concat_video_name = f"{args.save_root}/{img_name + '_concat.mp4'}"
        # mediapy.write_video(concat_video_name, concat_video, fps=args.save_fps)

        logger.info(f"Saved video to {chunk_video_name}")

    # Synchronize all processes before cleanup
    if args.context_parallel_size > 1:
        torch.distributed.barrier()

    # Clean up distributed resources
    video2world_cli.cleanup()


if __name__ == "__main__":
    main()
