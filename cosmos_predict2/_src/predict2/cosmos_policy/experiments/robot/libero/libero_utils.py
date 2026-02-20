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

"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image, ImageDraw, ImageFont

from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.robot_utils import DATE, DATE_TIME


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs, flip_images: bool = False):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    if flip_images:
        img = np.flipud(img)
    return img


def get_libero_wrist_image(obs, flip_images: bool = False):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    if flip_images:
        img = np.flipud(img)
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, ckpt_config_name=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:40]
    ckpt_str = f"--ckpt={ckpt_config_name}" if ckpt_config_name else ""
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}{ckpt_str}--task={processed_task_description}.mp4"
    )
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def save_rollout_video_with_future_image_predictions(
    rollout_images,
    idx,
    success,
    task_description,
    chunk_size,
    num_open_loop_steps,
    rollout_wrist_images=None,
    future_primary_image_predictions=None,
    future_wrist_image_predictions=None,
    show_diff=False,
    log_file=None,
    ckpt_config_name=None,
):
    """Saves an MP4 replay of an episode with future image predictions shown on the right."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:35]
    ckpt_str = f"--ckpt={ckpt_config_name}" if ckpt_config_name else ""
    mp4_path = f"{rollout_dir}/{DATE_TIME}--with_future_img--episode={idx}--success={success}{ckpt_str}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)

    # Determine availability of predictions
    has_primary_predictions = future_primary_image_predictions is not None and len(future_primary_image_predictions) > 0
    has_wrist_replay = rollout_wrist_images is not None
    has_wrist_predictions = has_wrist_replay and (
        future_wrist_image_predictions is not None and len(future_wrist_image_predictions) > 0
    )

    # Determine target dimensions to use for resizing
    if has_primary_predictions:
        target_h, target_w, c = future_primary_image_predictions[0].shape
    elif has_wrist_predictions:
        target_h, target_w, c = future_wrist_image_predictions[0].shape
    else:
        # Fall back to the rollout image size
        target_h, target_w, c = rollout_images[0].shape

    # Define text parameters
    text_height = 60  # Height for text area
    font_size = 18

    # Define column labels dynamically based on availability and configuration
    if show_diff:
        column_labels = []
        if has_wrist_replay:
            column_labels.append("replay wrist")
        if has_wrist_predictions:
            column_labels.append("future wrist")
            column_labels.append("wrist difference")
        column_labels.append("replay primary")
        if has_primary_predictions:
            column_labels.append("future primary")
            column_labels.append("primary difference")
    else:
        column_labels = []
        if has_wrist_replay:
            column_labels.append("real wrist image")
        if has_wrist_predictions:
            column_labels.append("predicted wrist image")
        column_labels.append("real current image")
        if has_primary_predictions:
            column_labels.append("predicted future image")
    num_columns = len(column_labels)

    # Iterate through images
    if has_wrist_replay:
        image_iterator = zip(rollout_images, rollout_wrist_images)
    else:
        image_iterator = rollout_images

    for i, image_data in enumerate(image_iterator):
        if has_wrist_replay:
            img, wrist_img = image_data
        else:
            img = image_data

        # Resize rollout image to match future prediction image dimensions if needed
        if img.shape[:2] != (target_h, target_w):
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(img)
            # Resize with PIL
            pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
            # Convert back to numpy array
            img = np.array(pil_img)

        # Resize wrist image if provided
        if has_wrist_replay:
            if wrist_img.shape[:2] != (target_h, target_w):
                # Convert numpy array to PIL Image
                pil_wrist_img = Image.fromarray(wrist_img)
                # Resize with PIL
                pil_wrist_img = pil_wrist_img.resize((target_w, target_h), Image.LANCZOS)
                # Convert back to numpy array
                wrist_img = np.array(pil_wrist_img)

        # Determine which future prediction images to use (if available)
        future_idx = i // num_open_loop_steps
        future_img = None
        future_wrist_img = None
        if has_primary_predictions:
            future_idx = min(future_idx, len(future_primary_image_predictions) - 1)
            future_img = future_primary_image_predictions[future_idx]
        if has_wrist_predictions:
            # If primary predictions are unavailable, still use the computed future_idx pattern
            future_wrist_idx = min(future_idx, len(future_wrist_image_predictions) - 1)
            future_wrist_img = future_wrist_image_predictions[future_wrist_idx]

        # Compute difference images if show_diff is True
        if show_diff:
            if has_primary_predictions and future_img is not None:
                # Compute primary image difference
                primary_diff = np.abs(img.astype(np.float32) - future_img.astype(np.float32))
                primary_diff = np.clip(primary_diff, 0, 255).astype(np.uint8)
            if has_wrist_predictions and future_wrist_img is not None:
                # Compute wrist image difference
                wrist_diff = np.abs(wrist_img.astype(np.float32) - future_wrist_img.astype(np.float32))
                wrist_diff = np.clip(wrist_diff, 0, 255).astype(np.uint8)

        # Create a combined image with the appropriate number of columns
        combined_img = np.zeros((target_h, target_w * num_columns, c), dtype=np.uint8)

        col = 0
        if show_diff:
            if has_wrist_replay:
                combined_img[:, target_w * col : target_w * (col + 1), :] = wrist_img
                col += 1
            if has_wrist_predictions and future_wrist_img is not None:
                combined_img[:, target_w * col : target_w * (col + 1), :] = future_wrist_img
                col += 1
                combined_img[:, target_w * col : target_w * (col + 1), :] = wrist_diff
                col += 1
            # Primary replay is always shown
            combined_img[:, target_w * col : target_w * (col + 1), :] = img
            col += 1
            if has_primary_predictions and future_img is not None:
                combined_img[:, target_w * col : target_w * (col + 1), :] = future_img
                col += 1
                combined_img[:, target_w * col : target_w * (col + 1), :] = primary_diff
                col += 1
        else:
            if has_wrist_replay:
                combined_img[:, target_w * col : target_w * (col + 1), :] = wrist_img
                col += 1
            if has_wrist_predictions and future_wrist_img is not None:
                combined_img[:, target_w * col : target_w * (col + 1), :] = future_wrist_img
                col += 1
            # Always show real primary image
            combined_img[:, target_w * col : target_w * (col + 1), :] = img
            col += 1
            if has_primary_predictions and future_img is not None:
                combined_img[:, target_w * col : target_w * (col + 1), :] = future_img
                col += 1

        # Create a blank area for text (white background)
        text_area = np.ones((text_height, target_w * num_columns, 3), dtype=np.uint8) * 255

        # Convert numpy array to PIL Image for text drawing
        text_img = Image.fromarray(text_area)
        draw = ImageDraw.Draw(text_img)

        # Try to use a standard font, fall back to default if not available
        try:
            font = ImageFont.truetype("Arial", font_size)
        except IOError:
            # If Arial is not available, try some other common fonts
            try:
                font = ImageFont.truetype("DejaVuSans", font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("Verdana", font_size)
                except IOError:
                    # Last resort: use default font
                    font = ImageFont.load_default()

        # Add column labels for all columns
        for col_idx, label in enumerate(column_labels):
            # Calculate center position for each column
            x_pos = col_idx * target_w + target_w // 2

            # Draw text centered in each column
            text_width = draw.textlength(label, font=font)
            draw.text((x_pos - text_width // 2, 8), label, font=font, fill=(0, 0, 0))

            # Add "K=X" as a second line under the prediction columns dynamically
            if ("predicted" in label) or ("future" in label):
                k_text = f"(K={chunk_size} timesteps)"
                k_text_width = draw.textlength(k_text, font=font)
                draw.text((x_pos - k_text_width // 2, 35), k_text, font=font, fill=(0, 0, 0))

        # Add timestep indicator in the center bottom of the text area
        timestep_text = f"t = {i}"
        timestep_text_width = draw.textlength(timestep_text, font=font)
        center_x = (target_w * num_columns) // 2
        draw.text((center_x - timestep_text_width // 2, 36), timestep_text, font=font, fill=(255, 0, 0))

        # Convert back to numpy array
        text_area = np.array(text_img)

        # Combine text area and images
        final_frame = np.vstack((text_area, combined_img))

        video_writer.append_data(final_frame)

    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
