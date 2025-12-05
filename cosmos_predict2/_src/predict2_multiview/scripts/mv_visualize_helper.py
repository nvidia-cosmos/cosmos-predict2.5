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

import torch as th

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video

# Visualization layouts for multi-view video arrangement
VISUALIZE_LAYOUTS = {
    "width": [
        [
            "camera_rear_left_70fov",
            "camera_cross_left_120fov",
            "camera_front_wide_120fov",
            "camera_cross_right_120fov",
            "camera_rear_right_70fov",
            "camera_rear_tele_30fov",
            "camera_front_tele_30fov",
        ]
    ],
    "height": [
        ["camera_rear_left_70fov"],
        ["camera_cross_left_120fov"],
        ["camera_front_wide_120fov"],
        ["camera_cross_right_120fov"],
        ["camera_rear_right_70fov"],
        ["camera_rear_tele_30fov"],
        ["camera_front_tele_30fov"],
    ],
    "grid": [
        [None, "camera_front_tele_30fov", None],
        ["camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov"],
        ["camera_rear_left_70fov", "camera_rear_tele_30fov", "camera_rear_right_70fov"],
    ],
}


def arrange_video_visualization(mv_video, data_batch, method="width"):
    """
    Rearrange multi-view video based on specified layout method.

    Args:
        mv_video: (B, C, V * T, H, W) - Multi-view video tensor
        data_batch: Batch containing camera order information
        method: Method to arrange video visualization. Can be "width", "height", "grid", or "time".
                - "width": Arrange all 7 views in a single horizontal row
                - "height": Arrange all 7 views in a single vertical column
                - "grid": Arrange views in a 3x3 grid with None values for empty positions
                - "time": Keep original format (V*T in time dimension, no spatial rearrangement)
    Returns:
        Video tensor arranged according to the layout:
        - For "width": (B, C, T, H, V*W) where V=7
        - For "height": (B, C, T, V*H, W) where V=7
        - For "grid": (B, C, T, 3*H, 3*W) with black padding for None positions
        - For "time": (B, C, V*T, H, W) (unchanged)
    """
    # Handle "time" mode - return video unchanged
    if method == "time":
        return mv_video

    if method not in VISUALIZE_LAYOUTS:
        raise ValueError(
            f"Unsupported visualization method: {method}. Choose from {list(VISUALIZE_LAYOUTS.keys()) + ['time']}"
        )

    current_view_order = data_batch["camera_keys_selection"][0]
    n_views = len(current_view_order)
    B, C, VT, H, W = mv_video.shape
    T = VT // n_views

    # Reshape to separate view and time dimensions: B C (V T) H W -> B C V T H W
    video = mv_video.view(B, C, n_views, T, H, W)

    # Create mapping from view name to tensor index
    view_name_to_video_tensor_idx = {view_name: idx for idx, view_name in enumerate(current_view_order)}

    # Create black view for None positions (used in grid layout)
    black_view = th.zeros(B, C, T, H, W, dtype=video.dtype, device=video.device)

    # Get layout definition
    layout_definition = VISUALIZE_LAYOUTS[method]

    # Arrange video according to layout
    grid_rows = []
    for row_of_view_names in layout_definition:
        row_tensors = []
        for view_name in row_of_view_names:
            if view_name is not None and view_name in view_name_to_video_tensor_idx:
                tensor_idx = view_name_to_video_tensor_idx[view_name]
                # video is B C V T H W. Get tensor for view: B C T H W
                row_tensors.append(video[:, :, tensor_idx])
            else:
                # Use black view for None positions or missing views
                row_tensors.append(black_view)
        grid_rows.append(th.cat(row_tensors, dim=-1))  # Concat on W dimension

    # Concatenate rows on H dimension
    video = th.cat(grid_rows, dim=-2)  # Concat on H dimension

    return video


def save_each_view_separately(
    mv_video: th.Tensor,
    data_batch: dict,
    save_dir: str,
    fps: float = 10.0,
    prefix: str = "",
) -> None:
    """
    Save each camera view as a separate video file.

    Args:
        mv_video: Multi-view video tensor with shape (C, V*T, H, W) where V is number of views
        data_batch: Data batch containing camera_keys_selection with actual camera order
        save_dir: Directory to save individual view videos
        fps: Frames per second for saved videos
        prefix: Optional prefix for saved filenames (e.g., "video_" or "control_")
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract actual camera order from data_batch
    camera_order = data_batch["camera_keys_selection"][0]

    C, VT, H, W = mv_video.shape
    n_views = len(camera_order)
    T = VT // n_views

    # Reshape to separate views: (C, V*T, H, W) -> (C, V, T, H, W)
    video_views = mv_video.view(C, n_views, T, H, W)

    # Save each view
    for view_idx, camera_name in enumerate(camera_order):
        view_video = video_views[:, view_idx, :, :, :]  # (C, T, H, W)
        filename = f"{prefix}{camera_name}" if prefix else camera_name
        view_save_path = os.path.join(save_dir, filename)
        save_img_or_video(view_video, view_save_path, fps=fps)
        log.info(f"Saved view {camera_name} to {view_save_path}")
