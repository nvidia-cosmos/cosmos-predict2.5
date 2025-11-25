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

"""Functions for sampling views from a multiview control video data batch."""

import torch

from cosmos_predict2._src.imaginaire.utils import log


def sample_n_views_from_data_batch(data_batch: dict, keep_view_indices: list[int]) -> dict:
    """Sample views from data batch, handling both video and control inputs."""

    keep_view_indices = sorted(keep_view_indices)
    n_keep_views = len(keep_view_indices)
    n_orig_views = data_batch["sample_n_views"].cpu().item()
    if n_keep_views == n_orig_views:
        log.debug("All views are requested to be kept, returning original data batch!")
        return data_batch
    num_video_frames_per_view = data_batch["num_video_frames_per_view"].cpu().item()

    select_ids = []
    for view_id in keep_view_indices:
        select_ids.extend(list(range(view_id * num_video_frames_per_view, (view_id + 1) * num_video_frames_per_view)))
    select_ids = torch.tensor(select_ids, device=data_batch["sample_n_views"].device, dtype=torch.int64)

    data_batch["video"] = data_batch["video"][:, :, select_ids]  # (B, C, V * T, H, W)
    data_batch["view_indices"] = data_batch["view_indices"][:, select_ids]  # (B, V * T)
    data_batch["view_indices_selection"] = data_batch["view_indices_selection"][:, keep_view_indices]
    data_batch["sample_n_views"] = 0 * data_batch["sample_n_views"] + n_keep_views

    # process control input keys
    for key in data_batch.keys():
        if key.startswith("control_input_"):
            data_batch[key] = data_batch[key][:, :, select_ids]  # (B, C, V * T, H, W)

    # process list-like keys
    for sample_captions in data_batch["ai_caption"]:
        if len(sample_captions) != n_orig_views:
            raise ValueError(
                f"Expected {n_orig_views} captions, got {len(sample_captions)} for key {key}. "
                "If using single caption, view sampling is not currently supported."
            )
    data_batch["ai_caption"] = [
        [caption for i, caption in enumerate(sample_captions) if i in keep_view_indices]
        for sample_captions in data_batch["ai_caption"]
    ]
    data_batch["camera_keys_selection"] = [
        [camera_key for i, camera_key in enumerate(sample_camera_keys) if i in keep_view_indices]
        for sample_camera_keys in data_batch["camera_keys_selection"]
    ]

    return data_batch
