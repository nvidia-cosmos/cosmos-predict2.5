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
from typing import Optional

import torch

from cosmos_predict2._src.common.datasets.augmentors.v3_text_transforms import pad_and_resize
from cosmos_predict2._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos_predict2._src.predict2_multiview.datasets.cache_utils import get_cam_t5_cache_dir


class TextTransformForMultiviewVideo(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.driving_dataloader_config = args["driving_dataloader_config"]
        self.embedding_type = args["embedding_type"]
        self.t5_tokens_num = args["t5_tokens"]["num"]  # number of tokens we cap after padding
        self.is_mask_all_ones = args["is_mask_all_ones"]  # if true, set mask for t5 to all ones
        self.concat_viewt5 = self.driving_dataloader_config.concat_viewt5  # if true, concat view t5 embeddings
        self.camera_to_caption_prefix = self.driving_dataloader_config.camera_to_caption_prefix
        self.camera_to_view_id = self.driving_dataloader_config.camera_to_view_id
        self.no_view_prefix = self.driving_dataloader_config.no_view_prefix
        self.single_caption_only = self.driving_dataloader_config.single_caption_only
        self.front_cam_key = self.driving_dataloader_config.front_tele_and_front_cam_keys[1]

        t5_cache_dir = get_cam_t5_cache_dir()

        self.cam_prefix_prompt_t5_embeddings = {
            k: torch.load(os.path.join(t5_cache_dir, f"video_camera_embeddings_v0_{k}.pt")).cpu()
            for k in self.camera_to_caption_prefix.keys()
        }

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs text transformation.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with captions and t5 embeddings added
        """
        camera_keys_selection = data_dict["camera_keys_selection"]
        view_caption_prefix = {cam_key: self.camera_to_caption_prefix[cam_key] for cam_key in camera_keys_selection}
        t5_embeddings = []
        t5_masks = []
        view_captions = []
        cameras_with_t5 = [self.front_cam_key] if self.single_caption_only else camera_keys_selection
        for cam_key in camera_keys_selection:
            view_prefix_t5_embedding = self.cam_prefix_prompt_t5_embeddings[cam_key]
            view_caption_prefix = "" if self.no_view_prefix else self.camera_to_caption_prefix[cam_key]
            if cam_key in cameras_with_t5:
                view_caption_t5_embedding = data_dict["view_t5_embeddings"][cam_key]
                view_caption = f"{view_caption_prefix} {data_dict['view_captions'][cam_key]}".strip()
            else:
                view_caption_t5_embedding = torch.zeros_like(data_dict["view_t5_embeddings"][self.front_cam_key])
                view_caption = None
            view_t5_embedding = (
                torch.cat([view_prefix_t5_embedding, view_caption_t5_embedding], dim=0)
                if not self.no_view_prefix
                else view_caption_t5_embedding
            )
            if view_caption is not None:
                view_captions.append(view_caption)
            elif not self.no_view_prefix:
                view_captions.append(view_caption_prefix)

            if self.concat_viewt5:
                t5_embeddings.append(view_t5_embedding)
            else:
                view_t5_embedding, view_t5_mask = pad_and_resize(
                    view_t5_embedding, self.t5_tokens_num, is_mask_all_ones=self.is_mask_all_ones
                )
                t5_embeddings.append(view_t5_embedding)
                t5_masks.append(view_t5_mask)
        if self.concat_viewt5:
            t5_embeddings, t5_masks = pad_and_resize(
                torch.cat(t5_embeddings, dim=0), self.t5_tokens_num, is_mask_all_ones=self.is_mask_all_ones
            )
        else:
            t5_embeddings = torch.cat(t5_embeddings, dim=0)
            t5_masks = torch.cat(t5_masks, dim=0)

        if self.embedding_type is not None:
            data_dict["t5_text_embeddings"] = t5_embeddings
            data_dict["t5_text_mask"] = t5_masks
        data_dict["ai_caption"] = " -- ".join(view_captions)
        del data_dict["view_t5_embeddings"]
        del data_dict["view_captions"]

        return data_dict
