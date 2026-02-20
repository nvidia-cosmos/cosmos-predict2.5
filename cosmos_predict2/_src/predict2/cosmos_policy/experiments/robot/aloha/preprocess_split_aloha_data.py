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
Adapted from:
https://github.com/user/openvla-oft/blob/main/experiments/robot/aloha/preprocess_split_aloha_data.py

Preprocesses ALOHA dataset(s) and splits them into train/val sets.

Preprocessing includes downsizing images from 480x640 to 256x256 and
compressing image sequences into MP4 videos for efficient storage.
Splits happen at the episode level (not step level), which means that
an episode is treated as an atomic unit that entirely goes to either
the train set or val set.

Original ALOHA data layout:
    /PATH/TO/DATASET/dataset_name/
        - episode_0.hdf5
        - episode_1.hdf5
        - ...
        - episode_N.hdf5

Preprocessed data layout (after running this script):
    /PATH/TO/PREPROCESSED_DATASETS/dataset_name/
        - train/
            - episode_0.hdf5  (contains paths to video files)
            - episode_0_cam_high.mp4
            - episode_0_cam_left_wrist.mp4
            - episode_0_cam_right_wrist.mp4
            - episode_1.hdf5
            - episode_1_cam_high.mp4
            - ...
        - val/
            - episode_0.hdf5  (contains paths to video files)
            - episode_0_cam_high.mp4
            - episode_0_cam_left_wrist.mp4
            - episode_0_cam_right_wrist.mp4
            - ...

    where N > M > K

Usage examples:
    python experiments/robot/aloha/preprocess_split_aloha_data.py \
        --dataset_path /scr/user/data/aloha2/fold_shirt_1500_steps/ \
        --out_base_dir /scr/user/data/aloha2_preprocessed/ \
        --percent_val 0.01
    python experiments/robot/aloha/preprocess_split_aloha_data.py \
        --dataset_path /scr/user/data/aloha2/put_candies_in_bowl_1000_steps/ \
        --out_base_dir /scr/user/data/aloha2_preprocessed/ \
        --percent_val 0.01
    python experiments/robot/aloha/preprocess_split_aloha_data.py \
        --dataset_path /scr/user/data/aloha2/put_candy_in_bag_1000_steps/ \
        --out_base_dir /scr/user/data/aloha2_preprocessed/ \
        --percent_val 0.01
    python experiments/robot/aloha/preprocess_split_aloha_data.py \
        --dataset_path /scr/user/data/aloha2/put_purple_eggplant_on_plate_250_steps/ \
        --out_base_dir /scr/user/data/aloha2_preprocessed/ \
        --percent_val 0.01
    python experiments/robot/aloha/preprocess_split_aloha_data.py \
        --dataset_path /scr/user/data/aloha2/put_brown_chicken_wing_on_plate_250_steps/ \
        --out_base_dir /scr/user/data/aloha2_preprocessed/ \
        --percent_val 0.01

"""

import argparse
import glob
import json
import os
import random

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def create_video_from_images(images, video_path, fps=30, quality=23):
    """
    Creates an MP4 video from a sequence of images.

    Args:
        images: numpy array of shape (num_frames, height, width, channels)
        video_path: path where to save the MP4 file
        fps: frames per second for the video
        quality: CRF value for H.264 encoding (lower = higher quality, 18-28 is good range)
    """
    if len(images) == 0:
        raise ValueError("No images provided")

    height, width, channels = images[0].shape

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for img in images:
        # Convert RGB to BGR for OpenCV
        if channels == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        out.write(img_bgr)

    out.release()
    print(f"Video saved: {video_path}")


def load_video_as_images(video_path):
    """
    Loads an MP4 video back into a numpy array of images.

    Args:
        video_path: path to the MP4 file

    Returns:
        numpy array of shape (num_frames, height, width, channels) in RGB format
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    return np.array(frames)


def load_hdf5(demo_path):
    """Loads single episode."""
    if not os.path.isfile(demo_path):
        print(f"Dataset does not exist at \n{demo_path}\n")
        exit()

    print(f"Loading {demo_path}...")
    with h5py.File(demo_path, "r") as root:
        is_sim = root.attrs["sim"]
        qpos = root["/observations/qpos"][()]
        qvel = root["/observations/qvel"][()]
        effort = root["/observations/effort"][()]
        action = root["/action"][()]
        image_dict = dict()
        for cam_name in root["/observations/images/"].keys():
            image_dict[cam_name] = root[f"/observations/images/{cam_name}"][()]
    print(f"Loading episode complete: {demo_path}")

    return qpos, qvel, effort, action, image_dict, is_sim


def load_and_preprocess_all_episodes(demo_paths, out_dataset_dir, split_name):
    """
    Loads and preprocesses all episodes.
    Resizes all images in one episode and converts them to MP4 videos before loading the next,
    to reduce memory usage and storage space.

    Args:
        demo_paths: List of paths to original episode files
        out_dataset_dir: Directory to save preprocessed episodes
        split_name: Name of the split ("train" or "val")

    Returns:
        List of metadata dictionaries tracking original->preprocessed mapping
    """
    cam_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    metadata_list = []

    for idx, demo in enumerate(tqdm(demo_paths, desc=f"Processing {split_name} episodes")):
        qpos, qvel, effort, action, image_dict, is_sim = load_hdf5(demo)
        # Save non-image info
        episode_len = image_dict["cam_high"].shape[0]

        # Resize all images and create video paths dict
        print("Resizing images and creating videos for episode...")
        video_paths = {}
        for k in cam_names:
            resized_images = []
            for i in range(episode_len):
                resized_images.append(
                    np.array(
                        Image.fromarray(image_dict[k][i]).resize(
                            (args.img_resize_size, args.img_resize_size), resample=Image.BICUBIC
                        )
                    )  # BICUBIC is default; specify explicitly to make it clear
                )
            resized_images = np.stack(resized_images)

            # Create video from resized images
            video_filename = f"episode_{idx}_{k}.mp4"
            video_path = os.path.join(out_dataset_dir, video_filename)
            create_video_from_images(resized_images, video_path, fps=args.video_fps)

            # Store relative path for portability
            video_paths[k] = video_filename

        print("Resizing images and video creation complete!")

        # Save preprocessed episode (now with video paths instead of raw images)
        data_dict = dict(
            qpos=qpos,
            qvel=qvel,
            effort=effort,
            action=action,
            video_paths=video_paths,  # Changed from image_dict to video_paths
            is_sim=is_sim,
        )
        preprocessed_episode_name = f"episode_{idx}.hdf5"
        save_new_hdf5(out_dataset_dir, data_dict, idx)

        # Track metadata for this episode
        episode_metadata = {
            "original_file_path": demo,
            "original_file_name": os.path.basename(demo),
            "preprocessed_episode_name": preprocessed_episode_name,
            "preprocessed_video_files": list(video_paths.values()),
            "split": split_name,
            "preprocessed_index": idx,
            "episode_length": episode_len,
            "is_simulation": bool(is_sim),
        }
        metadata_list.append(episode_metadata)

    return metadata_list


def save_new_hdf5(out_dataset_dir, data_dict, episode_idx):
    """Saves an HDF5 file for a new episode with video file references instead of raw image data."""
    out_path = os.path.join(out_dataset_dir, f"episode_{episode_idx}.hdf5")

    # Save HDF5 with same structure as original demos but with video paths instead of image arrays
    with h5py.File(
        out_path, "w", rdcc_nbytes=1024**2 * 2
    ) as root:  # Magic constant for rdcc_nbytes comes from ALOHA codebase
        episode_len = data_dict["qpos"].shape[0]
        root.attrs["sim"] = data_dict["is_sim"]

        # Save observations (non-image data)
        obs = root.create_group("observations")
        _ = obs.create_dataset("qpos", (episode_len, 14))
        _ = obs.create_dataset("qvel", (episode_len, 14))
        _ = obs.create_dataset("effort", (episode_len, 14))
        root["/observations/qpos"][...] = data_dict["qpos"]
        root["/observations/qvel"][...] = data_dict["qvel"]
        root["/observations/effort"][...] = data_dict["effort"]

        # Save video paths instead of raw image data
        video_paths_group = obs.create_group("video_paths")
        for cam_name, video_path in data_dict["video_paths"].items():
            # Store as string dataset
            video_paths_group.create_dataset(cam_name, data=video_path.encode("utf-8"))

        # Save actions
        _ = root.create_dataset("action", (episode_len, 14))
        root["/action"][...] = data_dict["action"]

        # Compute and save *relative* actions as well
        actions = data_dict["action"]
        relative_actions = np.zeros_like(actions)
        relative_actions[:-1] = actions[1:] - actions[:-1]  # Relative actions are the changes in joint pos
        relative_actions[-1] = relative_actions[-2]  # Just copy the second-to-last action for the last action
        _ = root.create_dataset("relative_action", (episode_len, 14))
        root["/relative_action"][...] = relative_actions

    print(f"Saved dataset: {out_path}")


def randomly_split_episode_paths(all_demo_paths, percent_val):
    """
    Randomly splits episode file paths into train and validation sets.

    Args:
        all_demo_paths: List of file paths to episode HDF5 files
        percent_val: Fraction of episodes to use for validation

    Returns:
        train_demo_paths: List of file paths for training episodes
        val_demo_paths: List of file paths for validation episodes
    """
    # Create a list of episode indices
    num_episodes_total = len(all_demo_paths)
    indices = list(range(num_episodes_total))

    # Shuffle the episode indices
    random.shuffle(indices)

    # Split into train and val sets
    num_episodes_val = int(num_episodes_total * percent_val)
    print(f"Total # episodes: {num_episodes_total}; using {num_episodes_val} ({percent_val:.2f}%) for val set")
    num_episodes_train = num_episodes_total - num_episodes_val

    train_indices = indices[:num_episodes_train]
    val_indices = indices[num_episodes_train:]

    train_demo_paths = [all_demo_paths[i] for i in train_indices]
    val_demo_paths = [all_demo_paths[i] for i in val_indices]

    return train_demo_paths, val_demo_paths


def main(args):
    # Create directory to save preprocessed dataset (if it doesn't exist already)
    os.makedirs(args.out_base_dir, exist_ok=True)
    out_dataset_dir = os.path.join(args.out_base_dir, os.path.basename(args.dataset_path.rstrip("/")))
    os.makedirs(out_dataset_dir, exist_ok=True)
    # Get list of filepaths of all episodes
    all_demo_paths = glob.glob(os.path.join(args.dataset_path, "*.hdf5"))  # List of HDF5 filepaths
    all_demo_paths.sort()

    # Split episodes into train and val sets using the modular function
    train_demo_paths, val_demo_paths = randomly_split_episode_paths(all_demo_paths, args.percent_val)

    # Preprocess all episodes and save the result
    out_dataset_dir_train = os.path.join(out_dataset_dir, "train")
    out_dataset_dir_val = os.path.join(out_dataset_dir, "val")
    os.makedirs(out_dataset_dir_train, exist_ok=True)
    os.makedirs(out_dataset_dir_val, exist_ok=True)

    # Process episodes and collect metadata
    train_metadata = load_and_preprocess_all_episodes(train_demo_paths, out_dataset_dir_train, "train")
    val_metadata = load_and_preprocess_all_episodes(val_demo_paths, out_dataset_dir_val, "val")

    # Combine all metadata and save to JSON file
    all_metadata = {
        "dataset_info": {
            "original_dataset_path": args.dataset_path,
            "preprocessed_dataset_path": out_dataset_dir,
            "total_episodes": len(all_demo_paths),
            "train_episodes": len(train_demo_paths),
            "val_episodes": len(val_demo_paths),
            "validation_percentage": args.percent_val,
            "preprocessing_settings": {"img_resize_size": args.img_resize_size, "video_fps": args.video_fps},
        },
        "episode_mapping": train_metadata + val_metadata,
    }

    # Save metadata to JSON file
    metadata_file_path = os.path.join(out_dataset_dir, "preprocessing_metadata.json")
    with open(metadata_file_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\nPreprocessing complete!")
    print(f"Metadata saved to: {metadata_file_path}")
    print(f"Total episodes processed: {len(all_metadata['episode_mapping'])}")
    print(f"Train episodes: {len(train_metadata)}")
    print(f"Val episodes: {len(val_metadata)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to raw ALOHA dataset directory. Example: /PATH/TO/USER/data/aloha_raw/put_green_pepper_into_pot/",
    )
    parser.add_argument(
        "--out_base_dir",
        required=True,
        help="Path to directory in which to save preprocessed dataset. Example: /PATH/TO/USER/data/aloha_preprocessed/",
    )
    parser.add_argument(
        "--percent_val",
        type=float,
        help="Percent of dataset to use as validation set (measured in episodes, not steps).",
        default=0.05,
    )
    parser.add_argument(
        "--img_resize_size",
        type=int,
        help="Size to resize images to. Final images will be square (img_resize_size x img_resize_size pixels).",
        default=256,
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        help="Frames per second for the output MP4 videos.",
        default=25,
    )
    args = parser.parse_args()

    main(args)
