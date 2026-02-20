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

"""Utils for evaluating policies in real-world ALOHA environments."""

import os
import time

import imageio
import numpy as np
from experiments.robot.aloha.real_env import make_real_env
from PIL import Image, ImageDraw, ImageFont

DATE_TIME = time.strftime("%Y_%m_%d--%H_%M_%S")


def get_next_task_label(task_label):
    """Prompt the user to input the next task."""
    if task_label == "":
        user_input = ""
        while user_input == "":
            user_input = input("Enter the task name: ")
        task_label = user_input
    else:
        user_input = input("Enter the task name (or leave blank to repeat the previous task): ")
        if user_input == "":
            pass  # Do nothing -> Let task_label be the same
        else:
            task_label = user_input
    print(f"Task: {task_label}")
    return task_label


def get_aloha_env():
    """Initializes and returns the ALOHA environment."""
    env = make_real_env(init_node=True)
    return env


def resize_image_for_preprocessing(img):
    """
    Takes numpy array corresponding to a single image and resizes to 256x256, exactly as done
    in the ALOHA data preprocessing script, which is used before converting the dataset to RLDS.
    """
    ALOHA_PREPROCESS_SIZE = 256
    img = np.array(
        Image.fromarray(img).resize((ALOHA_PREPROCESS_SIZE, ALOHA_PREPROCESS_SIZE), resample=Image.BICUBIC)
    )  # BICUBIC is default; specify explicitly to make it clear
    return img


def get_aloha_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    # obs: dm_env._environment.TimeStep
    img = obs.observation["images"]["cam_high"]
    img = resize_image_for_preprocessing(img)
    return img


def get_aloha_wrist_images(obs):
    """Extracts both wrist camera images from observations and preprocesses them."""
    # obs: dm_env._environment.TimeStep
    left_wrist_img = obs.observation["images"]["cam_left_wrist"]
    right_wrist_img = obs.observation["images"]["cam_right_wrist"]
    left_wrist_img = resize_image_for_preprocessing(left_wrist_img)
    right_wrist_img = resize_image_for_preprocessing(right_wrist_img)
    return left_wrist_img, right_wrist_img


def save_rollout_video(
    rollout_images, idx, success, task_description, policy_name, rollout_dir, log_file=None, notes=None
):
    """Saves an MP4 replay of an episode."""
    rollout_dir = os.path.join(rollout_dir, "videos")
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    filetag = (
        f"{rollout_dir}/{DATE_TIME}--{policy_name}--episode={idx}--success={success}--task={processed_task_description}"
    )
    if notes is not None:
        filetag += f"--{notes}"
    mp4_path = f"{filetag}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=25)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def save_rollout_video_with_future_image_predictions(
    rollout_images,
    rollout_images_left_wrist,
    rollout_images_right_wrist,
    future_image_predictions,
    idx,
    success,
    task_description,
    chunk_size,
    num_open_loop_steps,
    trained_with_image_aug,
    policy_name,
    rollout_dir,
    log_file=None,
):
    """Saves an MP4 replay of an episode with four columns: left wrist, right wrist, current image, and future image predictions.

    Args:
        rollout_images: List of main camera images
        rollout_images_left_wrist: List of left wrist camera images
        rollout_images_right_wrist: List of right wrist camera images
        future_image_predictions: List of predicted future images
        idx: Episode index
        success: Whether the episode was successful
        task_description: Description of the task
        chunk_size: Number of timesteps for future prediction
        num_open_loop_steps: Number of open loop steps
        trained_with_image_aug: If True, apply a 90% area center crop to all images
        policy_name: Model family (e.g. "cosmos")
        rollout_dir: Rollout directory
        log_file: Optional file for logging
    """
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    rollout_dir = os.path.join(rollout_dir, "videos")
    os.makedirs(rollout_dir, exist_ok=True)
    mp4_path = f"{rollout_dir}/{DATE_TIME}--{policy_name}--with_future_img--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=25)

    # Ensure future_image_predictions has at least one element
    if not future_image_predictions:
        raise ValueError("future_image_predictions must have at least one element")

    # Get dimensions from future_image_predictions to use for resizing
    target_h, target_w, c = future_image_predictions[0].shape

    # Define text parameters
    text_height = 60  # Height for text area
    font_size = 16

    # Define column labels
    column_labels = ["current left wrist img", "current right wrist img", "current main img", "future main img (pred)"]

    # Ensure all lists have elements
    if not rollout_images_left_wrist or not rollout_images_right_wrist:
        raise ValueError("rollout_images_left_wrist and rollout_images_right_wrist must have at least one element each")

    for i, img in enumerate(rollout_images):
        # Get corresponding left and right wrist images
        left_wrist_img = rollout_images_left_wrist[min(i, len(rollout_images_left_wrist) - 1)]
        right_wrist_img = rollout_images_right_wrist[min(i, len(rollout_images_right_wrist) - 1)]

        # Process all images - resize and optionally center crop
        images_to_process = [left_wrist_img, right_wrist_img, img]
        processed_images = []

        for current_img in images_to_process:
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(current_img)

            # Resize if needed
            if pil_img.size != (target_w, target_h):
                pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)

            # If center cropping, resize to the preprocess size (256x256)
            if trained_with_image_aug:
                # Calculate crop dimensions (90% of width and height)
                crop_factor = 0.9**0.5
                orig_w, orig_h = pil_img.size
                crop_w = int(orig_w * crop_factor)
                crop_h = int(orig_h * crop_factor)

                # Calculate crop coordinates to center the crop
                left = (orig_w - crop_w) // 2
                top = (orig_h - crop_h) // 2
                right = left + crop_w
                bottom = top + crop_h

                # Perform the crop
                pil_img = pil_img.crop((left, top, right, bottom))

                # Resize back to target dimensions
                pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)

            # Convert back to numpy array
            processed_images.append(np.array(pil_img))

        # Unpack processed images
        left_wrist_img_resized, right_wrist_img_resized, img_resized = processed_images

        # Determine which future prediction image to use
        future_idx = i // num_open_loop_steps
        future_idx = min(future_idx, len(future_image_predictions) - 1)  # Prevent index out of range
        future_img = future_image_predictions[future_idx]

        # Create a combined image with all four frames side by side (without text yet)
        combined_img = np.zeros((target_h, target_w * 4, c), dtype=np.uint8)
        combined_img[:, :target_w, :] = left_wrist_img_resized
        combined_img[:, target_w : target_w * 2, :] = right_wrist_img_resized
        combined_img[:, target_w * 2 : target_w * 3, :] = img_resized
        combined_img[:, target_w * 3 :, :] = future_img

        # Create a blank area for text (white background)
        text_area = np.ones((text_height, target_w * 4, 3), dtype=np.uint8) * 255

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

        # Add column labels
        for col_idx, label in enumerate(column_labels):
            # Calculate center position for each column
            x_pos = col_idx * target_w + target_w // 2

            # Set text color - red for the fourth column, black for others
            text_color = (255, 0, 0) if col_idx == 3 else (0, 0, 0)

            # Draw text centered in each column
            text_width = draw.textlength(label, font=font)
            draw.text((x_pos - text_width // 2, 8), label, font=font, fill=text_color)

            # Add "K=X" as a second line under the fourth column (predicted future)
            if col_idx == 3:  # Fourth column (predicted future)
                k_text = f"(K={chunk_size} timesteps)"
                k_text_width = draw.textlength(k_text, font=font)
                draw.text(
                    (x_pos - k_text_width // 2, 35), k_text, font=font, fill=text_color
                )  # Using the same red color

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


def save_rollout_video_with_all_types_future_image_predictions(
    rollout_images,
    rollout_images_left_wrist,
    rollout_images_right_wrist,
    future_left_wrist_image_predictions,
    future_right_wrist_image_predictions,
    future_primary_image_predictions,
    idx,
    success,
    task_description,
    chunk_size,
    num_open_loop_steps,
    trained_with_image_aug,
    policy_name,
    rollout_dir,
    log_file=None,
):
    """Saves an MP4 replay of an episode with 2 rows and 3 columns:
    Top row: current left wrist, current right wrist, current main image
    Bottom row: future left wrist, future right wrist, future main image predictions.

    Args:
        rollout_images: List of main camera images
        rollout_images_left_wrist: List of left wrist camera images
        rollout_images_right_wrist: List of right wrist camera images
        future_left_wrist_image_predictions: List of predicted future left wrist images
        future_right_wrist_image_predictions: List of predicted future right wrist images
        future_primary_image_predictions: List of predicted future main images
        idx: Episode index
        success: Whether the episode was successful
        task_description: Description of the task
        chunk_size: Number of timesteps for future prediction
        num_open_loop_steps: Number of open loop steps
        trained_with_image_aug: If True, apply a 90% area center crop to all images
        policy_name: Model family (e.g. "cosmos")
        rollout_dir: Rollout directory
        log_file: Optional file for logging
    """
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    rollout_dir = os.path.join(rollout_dir, "videos")
    os.makedirs(rollout_dir, exist_ok=True)
    mp4_path = f"{rollout_dir}/{DATE_TIME}--{policy_name}--with_all_future_imgs--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=25)

    # Ensure future prediction lists have at least one element
    if not future_left_wrist_image_predictions:
        raise ValueError("future_left_wrist_image_predictions must have at least one element")
    if not future_right_wrist_image_predictions:
        raise ValueError("future_right_wrist_image_predictions must have at least one element")
    if not future_primary_image_predictions:
        raise ValueError("future_primary_image_predictions must have at least one element")

    # Get dimensions from future_primary_image_predictions to use for resizing
    target_h, target_w, c = future_primary_image_predictions[0].shape

    # Define text parameters
    text_height = 30  # Height for text area
    font_size = 16

    # Define row and column labels
    column_labels = ["left wrist image", "right wrist image", "primary image"]

    # Ensure all lists have elements
    if not rollout_images_left_wrist or not rollout_images_right_wrist:
        raise ValueError("rollout_images_left_wrist and rollout_images_right_wrist must have at least one element each")

    for i, img in enumerate(rollout_images):
        # Get corresponding left and right wrist images
        left_wrist_img = rollout_images_left_wrist[min(i, len(rollout_images_left_wrist) - 1)]
        right_wrist_img = rollout_images_right_wrist[min(i, len(rollout_images_right_wrist) - 1)]

        # Process current images - resize and optionally center crop
        current_images_to_process = [left_wrist_img, right_wrist_img, img]
        processed_current_images = []

        for current_img in current_images_to_process:
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(current_img)

            # Resize if needed
            if pil_img.size != (target_w, target_h):
                pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)

            # If center cropping, resize to the preprocess size (256x256)
            if trained_with_image_aug:
                # Calculate crop dimensions (90% of width and height)
                crop_factor = 0.9**0.5
                orig_w, orig_h = pil_img.size
                crop_w = int(orig_w * crop_factor)
                crop_h = int(orig_h * crop_factor)

                # Calculate crop coordinates to center the crop
                left = (orig_w - crop_w) // 2
                top = (orig_h - crop_h) // 2
                right = left + crop_w
                bottom = top + crop_h

                # Perform the crop
                pil_img = pil_img.crop((left, top, right, bottom))

                # Resize back to target dimensions
                pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)

            # Convert back to numpy array
            processed_current_images.append(np.array(pil_img))

        # Unpack processed current images
        left_wrist_img_resized, right_wrist_img_resized, img_resized = processed_current_images

        # Determine which future prediction images to use
        future_idx = i // num_open_loop_steps
        future_left_idx = min(future_idx, len(future_left_wrist_image_predictions) - 1)
        future_right_idx = min(future_idx, len(future_right_wrist_image_predictions) - 1)
        future_primary_idx = min(future_idx, len(future_primary_image_predictions) - 1)

        future_left_wrist_img = future_left_wrist_image_predictions[future_left_idx]
        future_right_wrist_img = future_right_wrist_image_predictions[future_right_idx]
        future_primary_img = future_primary_image_predictions[future_primary_idx]

        # Create a combined image with 2 rows and 3 columns (without text yet)
        combined_img = np.zeros((target_h * 2, target_w * 3, c), dtype=np.uint8)

        # Top row: current images
        combined_img[:target_h, :target_w, :] = left_wrist_img_resized
        combined_img[:target_h, target_w : target_w * 2, :] = right_wrist_img_resized
        combined_img[:target_h, target_w * 2 : target_w * 3, :] = img_resized

        # Bottom row: future predictions
        combined_img[target_h:, :target_w, :] = future_left_wrist_img
        combined_img[target_h:, target_w : target_w * 2, :] = future_right_wrist_img
        combined_img[target_h:, target_w * 2 : target_w * 3, :] = future_primary_img

        # Create a blank area for text (white background)
        text_area = np.ones((text_height, target_w * 3, 3), dtype=np.uint8) * 255

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

        # Add top row labels
        for col_idx, label in enumerate(column_labels):
            # Calculate center position for each column
            x_pos = col_idx * target_w + target_w // 2
            text_color = (0, 0, 0)  # Black for current images

            # Draw text centered in each column
            text_width = draw.textlength(label, font=font)
            draw.text((x_pos - text_width // 2, 8), label, font=font, fill=text_color)

        # # Add "K=X timesteps" info at the bottom right
        # k_text = f"(K={chunk_size} timesteps)"
        # k_text_width = draw.textlength(k_text, font=font)
        # # Position at the right side of the text area
        # draw.text((target_w * 3 - k_text_width - 10, 50), k_text, font=font, fill=(255, 0, 0))

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
