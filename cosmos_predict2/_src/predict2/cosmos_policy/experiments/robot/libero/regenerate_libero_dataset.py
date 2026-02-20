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
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.

Usage:
    python experiments/robot/libero/regenerate_libero_dataset.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR> \
        [--data_collection] [--jpeg_compress] [--local_log_dir <PATH>]

    Examples:
        uv run -m cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.libero.regenerate_libero_dataset \
            --libero_task_suite libero_spatial \
            --libero_raw_data_dir cosmos_policy/LIBERO/libero/datasets/libero_spatial \
            --libero_target_dir ./regen_data/libero_spatial_regen \
            --data_collection True --jpeg_compress True --local_log_dir ./regen_data/ --deterministic True

        uv run -m cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.libero.regenerate_libero_dataset \
            --libero_task_suite libero_object \
            --libero_raw_data_dir cosmos_policy/LIBERO/libero/datasets/libero_object \
            --libero_target_dir ./regen_data/libero_object_regen \
            --data_collection True --jpeg_compress True --local_log_dir ./regen_data/ --deterministic True

        uv run -m cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.libero.regenerate_libero_dataset \
            --libero_task_suite libero_goal \
            --libero_raw_data_dir cosmos_policy/LIBERO/libero/datasets/libero_goal \
            --libero_target_dir ./regen_data/libero_goal_regen \
            --data_collection True --jpeg_compress True --local_log_dir ./regen_data/ --deterministic True

        uv run -m cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.libero.regenerate_libero_dataset \
            --libero_task_suite libero_10 \
            --libero_raw_data_dir cosmos_policy/LIBERO/libero/datasets/libero_10 \
            --libero_target_dir ./regen_data/libero_10_regen \
            --data_collection True --jpeg_compress True --local_log_dir ./regen_data/ --deterministic True

"""

import argparse
import json
import os

import h5py
import imageio
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark

from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.robot_utils import DATE_TIME
from cosmos_predict2._src.predict2.cosmos_policy.utils.utils import set_seed_everywhere

IMAGE_RESOLUTION = 256


def jpeg_encode_image(image, quality=95):
    """Encode image as JPEG bytes."""
    import io

    from PIL import Image

    img = Image.fromarray(image)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def str2bool(v):
    """Convert string representation of truth to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def main(args):
    if args.deterministic:
        set_seed_everywhere(seed=0)

    print(f"Regenerating {args.libero_task_suite} dataset!")

    # Create target directory
    if os.path.isdir(args.libero_target_dir):
        user_input = input(
            f"Target directory already exists at path: {args.libero_target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: "
        )
        if user_input != "y":
            exit()
    os.makedirs(args.libero_target_dir, exist_ok=True)

    # Create rollout data directory if data collection is enabled
    if args.data_collection:
        rollout_data_dir = os.path.join(args.local_log_dir, "rollout_data")
        os.makedirs(rollout_data_dir, exist_ok=True)
        print(f"Data collection enabled. Rollout data will be saved to: {rollout_data_dir}")

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = os.path.join(args.libero_target_dir, f"{args.libero_task_suite}_metainfo.json")
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0
    total_episodes = 0  # Global episode counter for data collection

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        # Create new HDF5 file for regenerated demos
        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        for i in range(len(orig_data.keys())):
            # Get demo data
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])
            for _ in range(10):
                # Deterministic seeding for reproducibility
                if args.deterministic:
                    set_seed_everywhere(seed=0)
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []

            # Data collection buffers
            if args.data_collection:
                primary_images_list = []  # For primary third-person images
                wrist_images_list = []  # For wrist camera images
                proprio_list = []  # For proprioceptive state
                actions_list = []  # For actions

            # Replay original demo actions in environment and record observations
            for _, action in enumerate(orig_actions):
                # Deterministic seeding for reproducibility
                if args.deterministic:
                    set_seed_everywhere(seed=0)

                # Skip transitions with no-op actions
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    print(f"\tSkipping no-op action: {action}")
                    num_noops += 1
                    continue

                if states == []:
                    # In the first timestep, since we're using the original initial state to initialize the environment,
                    # copy the initial state (first state in episode) over from the original HDF5 to the new one
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    # For all other timesteps, get state from environment and record it
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )

                # Record original action (from demo)
                actions.append(action)

                # Record data returned by environment
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )
                # Flip images vertically to correct rendering orientation
                agentview_image_flipped = np.flipud(obs["agentview_image"])
                eye_in_hand_image_flipped = np.flipud(obs["robot0_eye_in_hand_image"])

                agentview_images.append(agentview_image_flipped)
                eye_in_hand_images.append(eye_in_hand_image_flipped)

                # Collect data for rollout dataset (if data collection mode is enabled)
                if args.data_collection:
                    primary_images_list.append(agentview_image_flipped)
                    wrist_images_list.append(eye_in_hand_image_flipped)
                    # Combine proprioceptive data (gripper position + end-effector position + end-effector quaternion)
                    proprio = np.concatenate(
                        [obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
                    )
                    proprio_list.append(proprio)
                    actions_list.append(action)

                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())

            # Save episodic data for rollout dataset (if data collection mode is enabled)
            # This captures ALL episodes regardless of success/failure
            if args.data_collection and len(primary_images_list) > 0:
                # Increment total episode counter
                total_episodes += 1

                # Prepare collected data dictionary
                collected_data = dict(
                    primary_images=np.stack(primary_images_list, axis=0),  # (T, H, W, C)
                    wrist_images=np.stack(wrist_images_list, axis=0),  # (T, H, W, C)
                    proprio=np.stack(proprio_list, axis=0),  # (T, D)
                    actions=np.stack(actions_list, axis=0),  # (T, action_dim)
                    success=bool(done),  # True for successful episodes, False for unsuccessful
                )

                def _save_episode_data():
                    """Save collected episode data to HDF5 file."""
                    ep_filename = f"episode_data--suite={args.libero_task_suite}--{DATE_TIME}--task={task_id}--ep={total_episodes}--success={done}--regen_demo.hdf5"
                    rollout_data_dir = os.path.join(args.local_log_dir, "rollout_data")
                    ep_filepath = os.path.join(rollout_data_dir, ep_filename)
                    with h5py.File(ep_filepath, "w") as f:
                        for k, v in collected_data.items():
                            if isinstance(v, np.ndarray):
                                is_image = v.ndim == 4 and v.shape[-1] == 3 and v.dtype == np.uint8
                                if is_image and args.jpeg_compress:
                                    jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in v]
                                    dt = h5py.vlen_dtype(np.dtype("uint8"))
                                    f.create_dataset(k + "_jpeg", data=jpeg_list, dtype=dt)
                                else:
                                    f.create_dataset(k, data=v)
                            else:
                                f.attrs[k] = v
                        f.attrs["task_description"] = task_description

                _save_episode_data()
                print(f"Saved rollout episode data: task={task_id}, episode={total_episodes}, success={done}")

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_images)

                ep_data_grp = grp.create_group(f"demo_{i}")
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])

                # Save image data with optional JPEG compression
                agentview_images_array = np.stack(agentview_images, axis=0)
                eye_in_hand_images_array = np.stack(eye_in_hand_images, axis=0)

                if args.jpeg_compress:
                    # Apply JPEG compression to demonstration images
                    agentview_jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in agentview_images_array]
                    eye_in_hand_jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in eye_in_hand_images_array]

                    dt = h5py.vlen_dtype(np.dtype("uint8"))
                    obs_grp.create_dataset("agentview_rgb_jpeg", data=agentview_jpeg_list, dtype=dt)
                    obs_grp.create_dataset("eye_in_hand_rgb_jpeg", data=eye_in_hand_jpeg_list, dtype=dt)
                else:
                    # Save uncompressed images
                    obs_grp.create_dataset("agentview_rgb", data=agentview_images_array)
                    obs_grp.create_dataset("eye_in_hand_rgb", data=eye_in_hand_images_array)
                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)

                num_success += 1

            # Save a video of the episode (for visualization)
            video_out_dir = f"./temp_regen/{args.libero_task_suite}"
            os.makedirs(video_out_dir, exist_ok=True)
            mp4_path = os.path.join(video_out_dir, f"ep={total_episodes}--success={bool(done)}.mp4")
            video_writer = imageio.get_writer(mp4_path, fps=30)
            for img in agentview_images:
                video_writer.append_data(img)
            video_writer.close()
            print(f"Saved rollout MP4 at path {mp4_path}")

            num_replays += 1

            # Record success/false and initial environment state in metainfo dict
            task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{i}"
            if task_key not in metainfo_json_dict:
                metainfo_json_dict[task_key] = {}
            if episode_key not in metainfo_json_dict[task_key]:
                metainfo_json_dict[task_key][episode_key] = {}
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()

            # Write metainfo dict to JSON file
            # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")

        # Close HDF5 files
        orig_data_file.close()
        new_data_file.close()
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")

    print(f"Dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")

    # Report data collection summary if enabled
    if args.data_collection:
        rollout_data_dir = os.path.join(args.local_log_dir, "rollout_data")
        print(f"Data collection complete! Saved {total_episodes} rollout episodes to: {rollout_data_dir}")
        print(f"JPEG compression: {'Enabled' if args.jpeg_compress else 'Disabled'}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite. Example: libero_spatial",
        required=True,
    )
    parser.add_argument(
        "--libero_raw_data_dir",
        type=str,
        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial",
        required=True,
    )
    parser.add_argument(
        "--libero_target_dir",
        type=str,
        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops",
        required=True,
    )
    parser.add_argument(
        "--data_collection",
        type=str2bool,
        default=False,
        help="If True, save replayed demonstration episodes as rollout episodes (same format as run_libero_eval.py). Accepts: yes/no, true/false, t/f, y/n, 1/0",
    )
    parser.add_argument(
        "--jpeg_compress",
        type=str2bool,
        default=True,
        help="If True, apply JPEG compression to images before saving (default: True). Accepts: yes/no, true/false, t/f, y/n, 1/0",
    )
    parser.add_argument(
        "--local_log_dir",
        type=str,
        default="./experiments/logs",
        help="Local directory for rollout data logs (used when data_collection=True)",
    )
    parser.add_argument(
        "--deterministic",
        type=str2bool,
        default=True,
        help="If True, use deterministic seed for environment (default: True). Accepts: yes/no, true/false, t/f, y/n, 1/0",
    )
    args = parser.parse_args()

    # Start data regeneration
    main(args)
