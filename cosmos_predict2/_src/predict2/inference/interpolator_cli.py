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

"""Command-line interface for video frame interpolation using diffusion models.

This script processes video files and generates interpolated frames between existing frames,
effectively increasing the frame rate of videos using trained diffusion models.

Multi-GPU Modes:
1. Context Parallelism (--context_parallel_size > 1): All GPUs work together on each video.
   Use this for large videos or when you have OOM issues. Videos are processed sequentially.

2. Data Parallelism (--context_parallel_size == 1, multiple GPUs via torchrun): Each GPU
   processes different videos independently. Use this for batch processing many videos.

Example usage:
# 720p 2X FRUC
run_docker -g 3 -i nvcr.io/nvidian/imaginaire4:v10.1.0 \
    "python3 -m cosmos_predict2._src.predict2.inference.interpolator_cli \
        --experiment=Interpolation-2B-720p-16fps-to-32fps-HQ_V6_from_22 \
        --ckpt_path s3://bucket/predict2/frame_interpolation/Interpolation-2B-720p-16fps-to-32fps-HQ_V6_from_22/checkpoints/iter_000142000 \
        --ckpt_cred credentials/pbss_dir_share.secret \
        --video_pattern 'tmp/panda70m_test_0000071_00000.mp4' \
        --output_dir tmp/panda70m_test_0000071_00000 \
        --upsample_factor 2 \
        --num_frame_pairs 2 \
        --output_frames"

    # For Multi-GPU with context parallelism
    append `--context_parallel_size <num_gpus>` to the command above.

# 1080p 4X FRUC
run_docker -g 3 -i nvcr.io/nvidian/imaginaire4:v10.1.0 \
    "python3 -m cosmos_predict2._src.predict2.inference.interpolator_cli \
        --experiment=Interpolation-2B-1080p-8fps-to-32fps-HQ_V6_from_22 \
        --ckpt_path s3://bucket/predict2/frame_interpolation/Interpolation-2B-1080p-16fps-to-48fps-HQ_V6_from_22/checkpoints/iter_000116000 \
        --ckpt_cred credentials/pbss_dir_share.secret \
        --video_pattern 'tmp/panda70m_test_0000071_00000.mp4' \
        --output_dir tmp/panda70m_test_0000071_00000 \
        --upsample_factor 4 \
        --num_frame_pairs 2 \
        --output_frames"


# 1080p 8-to-32fps
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m cosmos_predict2._src.predict2.inference.interpolator_cli \
    --experiment=Interpolation-2B-1080p-8fps-to-32fps-HQ_V6_from_22 \
    --ckpt_path s3://bucket/cosmos_diffusion_v2/frame_interpolation/Interpolation-2B-1080p-8fps-to-32fps-HQ_V6_from_22/checkpoints/iter_000206000 \
    --ckpt_cred credentials/s3_training.secret \
    --video_pattern 'assets/interpolator/test_1_first1000_v2_trimmed.mp4' \
    --output_dir results/interpolator/Interpolation-2B-1080p-8fps-to-32fps-HQ_V6_from_22/iter206k/test_1_first1000_v2 \
    --upsample_factor 4 \
    --num_frame_pairs 300 \
    --output_frames

# 1080p 8-to-32fps Rectified Flow
CUDA_VISIBLE_DEVICES=1 torchrun --master_port=29501 --nproc_per_node=1 -m cosmos_predict2._src.predict2.inference.interpolator_cli \
    --experiment=Interpolation-2B-1080p-8fps-to-32fps-HQ_V6_from_22_rectified_flow \
    --ckpt_path s3://bucket/cosmos_diffusion_v2/frame_interpolation/Interpolation-2B-1080p-8fps-to-32fps-HQ_V6_from_22_rectified_flow_non_consecutive/checkpoints/iter_000010000 \
    --ckpt_cred credentials/s3_training.secret \
    --video_pattern 'assets/interpolator/test_1_first1000_v2_trimmed.mp4' \
    --output_dir results/interpolator/Interpolation-2B-1080p-8fps-to-32fps-HQ_V6_from_22_rectified_flow_non_consecutive/iter10k/test_1_first1000_v2 \
    --upsample_factor 4 \
    --num_frame_pairs 300 \
    --output_frames


# 1080p 24-to-30fps
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m cosmos_predict2._src.predict2.inference.interpolator_cli \
    --experiment=Interpolation-2B-1080p-24fps-to-30fps-HQ_V6_from_22 \
    --ckpt_path s3://bucket/cosmos_diffusion_v2/frame_interpolation/Interpolation-2B-1080p-24fps-to-30fps-HQ_V6_from_22/checkpoints/iter_000010000 \
    --ckpt_cred credentials/s3_training.secret \
    --video_pattern 'assets/upscaler/000005.mp4' \
    --output_dir results/interpolator/interleave/iter10k \
    --num_interleaved_frames 4 \
    --num_frame_pairs 2 \
    --output_frames

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m cosmos_predict2._src.predict2.inference.interpolator_cli \
    --experiment=Interpolation-2B-1080p-24fps-to-30fps-HQ_V6_from_22_rectified_flow \
    --ckpt_path s3://bucket/cosmos_diffusion_v2/frame_interpolation/Interpolation-2B-1080p-24fps-to-30fps-HQ_V6_from_22_rectified_flow/checkpoints/iter_000010000 \
    --ckpt_cred credentials/s3_training.secret \
    --video_pattern 'assets/upscaler/000005.mp4' \
    --output_dir results/interpolator/interleave_rectified_flow/iter10k \
    --num_interleaved_frames 4 \
    --num_frame_pairs 2 \
    --output_frames


Expected input structure:
    input_root/
     ├── video1.mp4
     ├── video1.txt (optional text prompt)
     ├── video2.mp4
     ├── video2.txt (optional text prompt)
     └── ...

Generated output structure
(if `--output_dir` is provided, the filename subdirectory is omitted):
    output_dir/
     ├── video1/
        ├── interpolated.mp4
        ├── interpolated_frames/
            ├── frame_000000.jpg
            ├── frame_000001.jpg
            └── ...
     ├── video2/
        ├── interpolated.mp4
        ├── interpolated_frames/
            └── ...
     └── ...

# Method 1: Direct python for 1 GPU
run_docker -g 1 -i nvcr.io/nvidian/imaginaire4:v10.1.0 \
    "python3 -m cosmos_predict2._src.predict2.inference.interpolator_cli \
        --experiment=Interpolation-2B-720p-16fps-to-32fps-HQ_V6_from_22 \
        --ckpt_path s3://bucket/cosmos_diffusion_v2/frame_interpolation/Interpolation-2B-720p-16fps-to-32fps-HQ_V6_from_22/checkpoints/iter_000370000 \
        --ckpt_cred credentials/s3_checkpoint.secret \
        --video_pattern 's3://cosmos2_results/qinshengz_Stage-c_pt_4-Index-22-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_20_iter-26000_task1_dataset-transition_change_issue_upsampled_prompts_v1/*/0.mp4' \
        --input_cred credentials/pdx_cosmos_benchmark.secret \
        --upsample_factor 2 \
        --num_frame_pairs -1 \
        --output_frames"

# Method 2: torchrun with 4 GPUs (should work if mean_std_cli works)
run_docker -g 0,1,2,3 -i nvcr.io/nvidian/imaginaire4:v10.1.0 \
    "torchrun --nproc_per_node=4 -m cosmos_predict2._src.predict2.inference.interpolator_cli \
        --experiment=Interpolation-2B-720p-16fps-to-32fps-HQ_V6_from_22 \
        --ckpt_path s3://bucket/cosmos_diffusion_v2/frame_interpolation/Interpolation-2B-720p-16fps-to-32fps-HQ_V6_from_22/checkpoints/iter_000370000 \
        --ckpt_cred credentials/s3_checkpoint.secret \
        --video_pattern 's3://cosmos2_results/qinshengz_Stage-c_pt_4-Index-22-Size-2B-Res-720-Fps-16-Note-HQ_V3_from_20_iter-26000_task1_dataset-transition_change_issue_upsampled_prompts_v1/*/0.mp4' \
        --input_cred credentials/pdx_cosmos_benchmark.secret \
        --upsample_factor 2 \
        --num_frame_pairs -1 \
        --output_frames"
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from tqdm import tqdm

from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.context_managers import distributed_init
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.inference.interpolator_lib import Interpolator
from cosmos_predict2._src.predict2.inference.utils import (
    get_filepaths,
    numpy2tensor,
    read_video,
    set_s3_backend,
    tensor2numpy,
    write_image,
    write_video,
)

_DEFAULT_FPS = 24.0


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the interpolator inference script."""
    parser = argparse.ArgumentParser(description="Video frame interpolation inference script")

    # Model and experiment configuration
    parser.add_argument("--experiment", type=str, required=True, help="Experiment configuration name")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the model checkpoint (local or S3). If not provided, uses config default",
    )
    parser.add_argument(
        "--ckpt_cred",
        type=str,
        default="credentials/s3_checkpoint.secret",
        help="Path to S3 credentials for checkpoint access",
    )

    # Input/output configuration
    parser.add_argument(
        "--video_pattern", type=str, default="path/to/videos/*.mp4", help="Glob pattern for input videos (local or S3)"
    )
    parser.add_argument(
        "--input_cred",
        type=str,
        default="credentials/pbss_dir_share.secret",
        help="Path to S3 credentials for input access",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (local or S3). Defaults to same directory as input",
    )
    parser.add_argument(
        "--output_frames",
        action="store_true",
        help="Save individual interpolated frames as JPEG files",
    )

    # Interpolation parameters
    parser.add_argument(
        "--upsample_factor",
        type=int,
        default=2,
        help="Temporal framerate upsampling factor (e.g., 2 for 2X FRUC, 4 for 4X FRUC)",
    )
    parser.add_argument(
        "--num_frame_pairs",
        type=int,
        default=-1,
        help="Number of consecutive frame pairs to process from each input video. If -1, process all frame pairs",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="Target resolution as 'H,W'. Uses model's default resolution if not specified",
    )
    parser.add_argument(
        "--num_interleaved_frames",
        type=int,
        default=0,
        choices=[0, 4],
        help="Number of interleaved frames for interpolation. 0 means no interleaved frames.",
    )

    # Model inference parameters
    parser.add_argument("--guidance", type=int, default=-1, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Custom negative prompt for classifier-free guidance. Uses default S3 embeddings if not specified",
    )

    # Distributed processing
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for context parallelism. Use 2+ if encountering OOM errors",
    )

    return parser.parse_args()


def _read_prompt(prompt_path: str) -> str | None:
    """Read text prompt from file if it exists.

    Args:
        prompt_path: Path to the prompt text file.

    Returns:
        Text prompt content if file exists, None otherwise.
    """
    if easy_io.exists(prompt_path):
        logger.info(f"Loading prompt from {prompt_path}")
        prompt = easy_io.load(prompt_path, file_format="txt")
        return prompt.strip()
    return None


def _get_output_video_dir(input_video_filepath: str, output_dir: str = None, output_frames: bool = False) -> str:
    """Generate output directory path for processed video.

    Args:
        input_video_filepath: Path to input video file.
        output_dir: Base output directory (optional).
        output_frames: Whether frame output directory should be created.

    Returns:
        Path to the output directory for this video.
    """
    video_filename = os.path.basename(input_video_filepath).split(".")[0]
    video_dirname = os.path.dirname(input_video_filepath)
    output_video_dir = output_dir or os.path.join(video_dirname, video_filename)

    # Create directories for local output
    if not output_video_dir.startswith("s3://"):
        os.makedirs(output_video_dir, exist_ok=True)
        if output_frames:
            output_frames_dir = os.path.join(output_video_dir, "interpolated_frames")
            os.makedirs(output_frames_dir, exist_ok=True)

    return output_video_dir


def _generate_interpolated_frames(
    input_video,
    interpolator,
    upsample_factor: int,
    num_frame_pairs: int,
    num_interleaved_frames: int = 0,
    prompt: str = None,
    guidance: int = -1,
    resolution: str = None,
    seed: int = 1,
    negative_prompt: str = None,
    show_progress: bool = True,
    output_frames_dir: str = None,
) -> list:
    """Generate interpolated frames for consecutive frame pairs.

    Args:
        input_video: Input video frames array.
        interpolator: Interpolator instance for frame generation.
        upsample_factor: Temporal framerate upsampling factor.
        num_frame_pairs: Number of consecutive frame pairs to process.
        prompt: Optional text prompt for interpolation.
        guidance: Classifier-free guidance scale.
        resolution: Target resolution as 'H,W'.
        seed: Random seed for reproducibility.
        negative_prompt: Custom negative prompt for classifier-free guidance.
        show_progress: Whether to show progress bar (default True).
        output_frames_dir: Directory to write frames incrementally (optional).

    Returns:
        List of interpolated frames as numpy arrays.
    """
    interpolated_frames = []
    total_frame_idx = 0  # Track total frame count for output filenames

    if num_interleaved_frames > 0:
        actual_num_pairs = (len(input_video) - 1) // num_interleaved_frames
    else:
        actual_num_pairs = len(input_video) - 1
    if num_frame_pairs > 0:
        actual_num_pairs = min(num_frame_pairs, actual_num_pairs)

    # Show progress bar for frame pair processing
    frame_range = range(1, actual_num_pairs + 1)
    if show_progress:
        frame_iter = tqdm(frame_range, desc="Interpolating frames", unit="pair")
    else:
        frame_iter = frame_range

    for frame_idx in frame_iter:
        if num_interleaved_frames > 0:
            start_idx = (frame_idx - 1) * num_interleaved_frames
            end_idx = start_idx + num_interleaved_frames + 1
            input_frames = input_video[start_idx:end_idx]
            zeros = np.zeros_like(input_frames[0])
            concat_frames = [input_frames[0]]
            for i in range(1, num_interleaved_frames + 1):
                concat_frames.append(zeros)
                concat_frames.append(input_frames[i])
            assert len(concat_frames) == 9, f"Only support 9 frames for now, got {len(concat_frames)}"
            video_batch = np.stack(concat_frames)
        else:
            # Get consecutive frame pair
            first_frame, last_frame = input_video[frame_idx - 1 : frame_idx + 1]

            # Create interpolation sequence: first frame, zeros, last frame
            zeros = np.zeros_like(first_frame)
            middle_frames = [zeros] * (upsample_factor - 1)  # List of zero frames
            video_batch = np.stack([first_frame] + middle_frames + [last_frame])

        # Convert to tensor and resize
        video_batch = numpy2tensor(video_batch[np.newaxis, ...])

        # Generate interpolated frames
        curr_frames = interpolator(
            prompt=prompt,
            input_video=video_batch,
            guidance=guidance,
            resolution=resolution,
            seed=seed,
            negative_prompt=negative_prompt,
        )

        # Convert to numpy and accumulate frames
        curr_frames = tensor2numpy(curr_frames)[0]
        # Skip first frame for subsequent pairs to avoid duplication
        if num_interleaved_frames > 0:  # remove input frames
            indices = [0] + list(range(1, curr_frames.shape[0], 2)) + [curr_frames.shape[0] - 1]
            curr_frames = curr_frames[indices]
        curr_frames_ = curr_frames if frame_idx == 1 else curr_frames[1:]

        # Write frames incrementally if output directory is provided
        if output_frames_dir is not None:
            for frame in curr_frames_:
                frame_path = f"{output_frames_dir}/frame_{total_frame_idx:06d}.jpg"
                write_image(frame_path, frame)
                total_frame_idx += 1

        interpolated_frames.extend(curr_frames_)

    return np.stack(interpolated_frames)


def main():
    """Main entry point for the interpolator CLI."""
    torch.enable_grad(False)  # Disable gradients for inference
    args = parse_arguments()

    # Initialize distributed processing if environment is set up for it
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        with distributed_init():
            distributed.init()

    world_size = distributed.get_world_size()
    rank = distributed.get_rank()

    # Determine if we're using context parallelism (all GPUs work on same video)
    # vs data parallelism (different GPUs work on different videos)
    use_context_parallel = args.context_parallel_size > 1

    # Initialize the interpolator
    interpolator = Interpolator(
        args.experiment, args.ckpt_path, args.ckpt_cred, context_parallel_size=args.context_parallel_size
    )

    # Set S3 backend for all ranks (needed for video file access)
    set_s3_backend(credentials=args.input_cred)

    # Discover input video files (only rank 0 does the search)
    if rank == 0:
        filepaths = get_filepaths(args.video_pattern)
        log.info(f"Found {len(filepaths)} video files")
    else:
        filepaths = []

    # Broadcast the file list to all ranks
    if world_size > 1:
        filepaths_list = [filepaths]
        dist.broadcast_object_list(filepaths_list, src=0)
        filepaths = filepaths_list[0]

    if use_context_parallel:
        # Context parallelism: all GPUs work together on each video sequentially
        # All ranks process the same video list
        rank_filepaths = filepaths
        log.info(f"Context parallel mode: All {world_size} GPUs will process {len(filepaths)} videos together")
    else:
        # Data parallelism: distribute videos across ranks using round-robin
        if len(filepaths) < world_size:
            log.error(f"Found {len(filepaths)} files but need at least {world_size} for {world_size} GPUs")
            exit(1)

        # Trim to be evenly divisible by world_size
        num_files_per_rank = len(filepaths) // world_size
        filepaths = filepaths[: num_files_per_rank * world_size]
        rank_filepaths = filepaths[rank::world_size]
        log.info(f"Data parallel mode: Rank {rank} processing {len(rank_filepaths)} videos independently")

    # Process each video file
    for idx, input_video_filepath in enumerate(rank_filepaths):
        if use_context_parallel:
            log.info(f"Processing video {idx + 1}/{len(rank_filepaths)}: {input_video_filepath}")
        else:
            log.info(f"Rank {rank}: Processing video {idx + 1}/{len(rank_filepaths)}: {input_video_filepath}")

        # Load input video and metadata
        input_video = read_video(input_video_filepath)
        input_fps = getattr(input_video.metadata, "fps", _DEFAULT_FPS)
        if args.num_interleaved_frames > 0:
            output_fps = input_fps * (args.num_interleaved_frames + 1) / args.num_interleaved_frames
        else:
            output_fps = input_fps * args.upsample_factor

        # Load optional text prompt
        prompt = _read_prompt(input_video_filepath.replace(".mp4", ".txt"))

        # Prepare output directory (only rank 0 saves when using context parallelism)
        should_save = not use_context_parallel or rank == 0
        output_frames_dir = None
        if should_save:
            output_video_dir = _get_output_video_dir(input_video_filepath, args.output_dir, args.output_frames)
            if args.output_frames:
                output_frames_dir = f"{output_video_dir}/interpolated_frames"

        # Generate interpolated frames for consecutive frame pairs
        # Frames are written incrementally if output_frames_dir is provided
        interpolated_frames = _generate_interpolated_frames(
            input_video=input_video,
            interpolator=interpolator,
            upsample_factor=args.upsample_factor,
            num_frame_pairs=args.num_frame_pairs,
            num_interleaved_frames=args.num_interleaved_frames,
            prompt=prompt,
            guidance=args.guidance,
            resolution=args.resolution,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
            show_progress=(rank == 0),  # Only show progress on rank 0
            output_frames_dir=output_frames_dir,  # Write frames incrementally
        )

        # Save interpolated video
        if should_save:
            output_video_path = f"{output_video_dir}/interpolated.mp4"
            write_video(output_video_path, interpolated_frames, fps=output_fps)
            log.info(f"Completed output video {idx + 1}/{len(rank_filepaths)}: {output_video_path}")

        # Synchronize after each video in context parallel mode
        if use_context_parallel and world_size > 1:
            dist.barrier()

    log.info(f"Finished processing all {len(rank_filepaths)} videos")

    # Synchronize before cleanup
    if world_size > 1:
        dist.barrier()

    # Clean up distributed resources
    interpolator.cleanup()


if __name__ == "__main__":
    main()
