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

from typing import Optional

import torch
from einops import rearrange

from cosmos_predict2._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos_predict2._src.imaginaire.utils import log


class AVMultiviewAdapter(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

        self.driving_dataloader_config = args["driving_dataloader_config"]
        self.embedding_type = args["embedding_type"]

    def __call__(self, data_dict: dict) -> dict:
        r""" """
        n_views = self.driving_dataloader_config.n_views
        num_video_frames_per_view = self.driving_dataloader_config.num_video_frames_per_view
        batch_video_n_frames_per_view = data_dict["video"].shape[1] // n_views
        assert batch_video_n_frames_per_view == num_video_frames_per_view, (
            f"Video must have {num_video_frames_per_view} frames, got {batch_video_n_frames_per_view}"
        )
        if self.embedding_type is not None:
            # Zero out embeddings of all views except the first one :
            t5_emb_V_L_D = rearrange(data_dict["t5_text_embeddings"], "L (V D) -> V L D", L=512, V=n_views, D=1024)
            if self.driving_dataloader_config.single_caption_only:
                t5_emb_V_L_D[1:] = 0
            t5_emb = rearrange(t5_emb_V_L_D, "V L D -> (V L) D")
            t5_mask = data_dict["t5_text_mask"]
            assert t5_mask.shape[0] == 512, "t5_text_mask should be of shape (512,)"
            t5_mask = t5_mask.repeat(n_views)
            log.debug(f"AVMultiviewAdapter: T5_emb shape: {t5_emb.shape}")
            log.debug(f"AVMultiviewAdapter: T5_mask shape: {t5_mask.shape}")
        else:
            assert "t5_text_embeddings" not in data_dict, (
                "t5_text_embeddings should not be in data_dict if embeddings are not loaded"
            )
            assert "t5_text_mask" not in data_dict, (
                "t5_text_mask should not be in data_dict if embeddings are not loaded"
            )

        view_indices_selection = [i for i in range(n_views)]
        view_indices = torch.tensor(view_indices_selection).repeat_interleave(num_video_frames_per_view)
        log.debug(f"AVMultiviewAdapter: view_indices: {view_indices.shape}")
        camera_to_view_id = self.driving_dataloader_config.camera_to_view_id
        front_cam_key = self.driving_dataloader_config.front_cam_key
        data_dict["front_cam_view_idx_sample_position"] = view_indices_selection.index(camera_to_view_id[front_cam_key])
        # data_dict["video"] = video
        if self.embedding_type is not None:
            data_dict["t5_text_embeddings"] = t5_emb
            data_dict["t5_text_mask"] = t5_mask

        data_dict["n_orig_video_frames_per_view"] = [data_dict["n_orig_video_frames"]] * n_views
        data_dict["num_video_frames_per_view"] = num_video_frames_per_view
        data_dict["view_indices_selection"] = view_indices_selection
        data_dict["camera_keys_selection"] = [
            "camera_front_wide_120fov",
            "camera_cross_left_120fov",
            "camera_cross_right_120fov",
            "camera_rear_left_70fov",
            "camera_rear_right_70fov",
            "camera_rear_tele_30fov",
            "camera_front_tele_30fov",
        ]
        data_dict["view_indices"] = view_indices
        data_dict["sample_n_views"] = torch.tensor(n_views)
        data_dict["ref_cam_view_idx_sample_position"] = torch.tensor(-1)
        data_dict["aspect_ratio"] = "16,9"

        del data_dict["num_frames"]
        return data_dict
