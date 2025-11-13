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
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference.py \
--experiment=cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320 \
  --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320/checkpoints/iter_000016000 \
  --input_video_root /project/cosmos/user/bridge/ \
  --input_json_sub_folder annotation/test_100 \
  --save_root results/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320-val-16k \
  --resolution 256,320 --guidance 0 --chunk_size 12 --camera_id 0 --save_fps 4
"""

import argparse
import json
import os
from glob import glob

import mediapy
import numpy as np
import torch
from loguru import logger

from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.predict2.action.datasets.dataset_utils import euler2rotm, rotm2euler, rotm2quat
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
    parser.add_argument("--input_json_sub_folder", type=str, default="bridge/annotation/test_100", help="Action root")
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
    parser.add_argument("--save_fps", type=int, default=20, help="Save fps")

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


def _get_robot_states(label, state_key="state", gripper_key="continuous_gripper_state"):
    """
    Extracts the robot arm and gripper states from the label dictionary for the specified frame indices.

    Args:
        label (dict): Dictionary containing robot state information, with keys "state" and "continuous_gripper_state".
        frame_ids (list or np.ndarray): List or array of frame indices to extract.

    Returns:
        tuple:
            - np.ndarray: Array of arm states for the selected frames, shape (len(frame_ids), state_dim).
            - np.ndarray: Array of gripper states for the selected frames, shape (len(frame_ids),).
    """

    all_states = np.array(label[state_key])
    all_cont_gripper_states = np.array(label[gripper_key])

    return all_states, all_cont_gripper_states


def _get_actions(arm_states, gripper_states, sequence_length, use_quat=False):
    """
    Compute the relative actions between consecutive robot states.

    Args:
        arm_states (np.ndarray): Array of arm states with shape (sequence_length, 6), where each state contains
            [x, y, z, roll, pitch, yaw] or similar.
        gripper_states (np.ndarray): Array of gripper states with shape (sequence_length,).
        sequence_length (int): Number of states in the sequence.
        use_quat (bool): If True, represent rotation as quaternion; otherwise, use Euler angles.

    Returns:
        np.ndarray: Array of actions with shape (sequence_length - 1, 7), where each action contains
            [relative_xyz (3), relative_rotation (3), gripper_state (1)].
    """

    if use_quat:
        action = np.zeros((sequence_length - 1, 8))
    else:
        action = np.zeros((sequence_length - 1, 7))

    for k in range(1, sequence_length):
        prev_xyz = arm_states[k - 1, 0:3]
        prev_rpy = arm_states[k - 1, 3:6]
        prev_rotm = euler2rotm(prev_rpy)
        curr_xyz = arm_states[k, 0:3]
        curr_rpy = arm_states[k, 3:6]
        curr_gripper = gripper_states[k]
        curr_rotm = euler2rotm(curr_rpy)
        rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
        rel_rotm = prev_rotm.T @ curr_rotm

        if use_quat:
            rel_rot = rotm2quat(rel_rotm)
            action[k - 1, 0:3] = rel_xyz
            action[k - 1, 3:7] = rel_rot
            action[k - 1, 7] = curr_gripper
        else:
            rel_rot = rotm2euler(rel_rotm)
            action[k - 1, 0:3] = rel_xyz
            action[k - 1, 3:6] = rel_rot
            action[k - 1, 6] = curr_gripper
    return action  # (l - 1, act_dim)


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

    arm_states, cont_gripper_states = _get_robot_states(data, state_key, gripper_key)
    actions = _get_actions(
        arm_states[::fps_downsample_ratio],
        cont_gripper_states[::fps_downsample_ratio],
        len(data[state_key][::fps_downsample_ratio]),
        use_quat=use_quat,
    )
    actions *= np.array(
        [_ACTION_SCALER, _ACTION_SCALER, _ACTION_SCALER, _ACTION_SCALER, _ACTION_SCALER, _ACTION_SCALER, gripper_scale]
    )

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
    input_json_path = os.path.join(args.input_video_root, args.input_json_sub_folder)
    input_json_list = glob(os.path.join(input_json_path, "*.json"))

    # Only process files on rank 0 if using distributed processing
    rank0 = True
    if args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    # Ensure save directory exists
    os.makedirs(args.save_root, exist_ok=True)

    # Process each file in the input directory
    for annotation_path in input_json_list[args.start : args.end]:
        with open(annotation_path, "r") as f:
            json_data = json.load(f)

        # Convert camera_id to integer if it's a string and can be converted to an integer
        camera_id = (
            int(args.camera_id) if isinstance(args.camera_id, str) and args.camera_id.isdigit() else args.camera_id
        )

        if isinstance(json_data["videos"][camera_id], dict):
            video_path = os.path.join(input_video_path, json_data["videos"][camera_id]["video_path"])
        else:
            video_path = os.path.join(input_video_path, json_data["videos"][camera_id])

        actions = get_action_sequence_from_states(
            json_data,
            fps_downsample_ratio=args.fps_downsample_ratio,
            state_key=args.state_key,
            gripper_scale=args.gripper_scale,
            gripper_key=args.gripper_key,
        )

        actions = actions[: len(actions)]
        video_array = mediapy.read_video(video_path)
        img_array = video_array[args.start_frame_idx]

        # Resize img_array with arg.resolution if specified
        if args.resolution != "none":
            try:
                h, w = map(int, args.resolution.split(","))
                img_array = mediapy.resize_image(img_array, (h, w))
            except Exception as e:
                logger.warning(f"Failed to resize image to {args.resolution}: {e}")

        img_name = annotation_path.split("/")[-1].split(".")[0]

        frames = [img_array]
        chunk_video = []

        video_name = f"{args.save_root}/{img_name.replace('.jpg', '.mp4')}"
        chunk_video_name = f"{args.save_root}/{img_name + '_chunk.mp4'}"
        logger.info(f"Saving video to {video_name}")
        if os.path.exists(chunk_video_name):
            logger.info(f"Video already exists: {chunk_video_name}")
            continue

        for i in range(args.start_frame_idx, len(actions), args.chunk_size):
            next_img_array, video_clamped = video2world_cli.step_inference(
                img_array=img_array,
                action=actions[i : i + args.chunk_size],
                guidance=args.guidance,
                seed=i,
            )
            frames.append(next_img_array)
            img_array = next_img_array
            chunk_video.append(video_clamped)

            if args.single_chunk:
                break

        chunk_list = [chunk_video[0]] + [chunk_video[i][: args.chunk_size] for i in range(1, len(chunk_video))]
        chunk_video = np.concatenate(chunk_list, axis=0)
        if args.single_chunk:
            chunk_video_name = f"{args.save_root}/{img_name + '_single_chunk.mp4'}"
        else:
            chunk_video_name = f"{args.save_root}/{img_name + '_chunk.mp4'}"

        if rank0:
            mediapy.write_video(chunk_video_name, chunk_video, fps=args.save_fps)
        logger.info(f"Saved video to {chunk_video_name}")

    # Synchronize all processes before cleanup
    if args.context_parallel_size > 1:
        torch.distributed.barrier()

    # Clean up distributed resources
    video2world_cli.cleanup()


if __name__ == "__main__":
    main()
