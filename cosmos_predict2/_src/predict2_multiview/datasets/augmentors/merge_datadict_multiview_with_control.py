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


class DataDictRewriter(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        assert len(input_keys) == len(output_keys)
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Rename the dictionary associated with the input keys into data_dict.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with dictionary associated with the input keys merged.
        """
        for i, key in enumerate(self.output_keys):
            if key not in data_dict:
                log.warning(
                    f"DataDictMerger dataloader error: missing {key}, {data_dict['__url__']}, {data_dict['__key__']}",
                    rank0_only=False,
                )
                return None
            key_dict = data_dict.pop(key)

            data_dict[self.output_keys[i]] = key_dict[self.input_keys[i]]
            del key_dict
        return data_dict


class DataDictConcatenator(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.concat_dim = args.get("concat_dim", 1) if args else 1  # Default to concatenating along temporal dimension
        self.concat_patterns = args.get("concat_patterns", {}) if args else {}

    def __call__(self, data_dict: dict) -> dict:
        """Concatenate tensors from multiple input keys.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with concatenated tensors
        """
        # Group input keys by pattern (e.g., video_0, video_1, etc.)
        grouped_keys = {}
        for key in self.input_keys:
            if key not in data_dict:
                log.warning(
                    f"DataDictConcatenator dataloader error: missing {key}, {data_dict['__url__']}, {data_dict['__key__']}",
                    rank0_only=False,
                )
                return None

            # Extract base name and index (e.g., video_0 -> video, 0)
            if "_" in key:
                base_name = "_".join(key.split("_")[:-1])
                try:
                    index = int(key.split("_")[-1])
                except ValueError:
                    base_name = key
                    index = 0
            else:
                base_name = key
                index = 0

            if base_name not in grouped_keys:
                grouped_keys[base_name] = []
            grouped_keys[base_name].append((index, key))

        # Sort by index and concatenate
        for base_name, key_list in grouped_keys.items():
            key_list.sort(key=lambda x: x[0])  # Sort by index

            tensors_to_concat = []
            metadata_dict = {}

            for index, key in key_list:
                key_dict = data_dict.pop(key)

                # Extract video tensor and metadata from the dictionary
                if isinstance(key_dict, dict) and "video" in key_dict:
                    tensor = key_dict["video"]
                    # Store metadata from the first video
                    if index == 0 and base_name == "video":
                        for meta_key in [
                            "fps",
                            "num_frames",
                            "chunk_index",
                            "frame_start",
                            "frame_end",
                            "n_orig_video_frames",
                            "frame_indices",
                        ]:
                            if meta_key in key_dict:
                                metadata_dict[meta_key] = key_dict[meta_key]
                elif isinstance(key_dict, torch.Tensor):
                    tensor = key_dict
                else:
                    # Handle other data types if needed
                    tensor = key_dict

                if isinstance(tensor, torch.Tensor):
                    tensors_to_concat.append(tensor)

            # Add metadata to data_dict if this is a video concatenation
            if base_name == "video" and metadata_dict:
                for meta_key, meta_value in metadata_dict.items():
                    if meta_key not in data_dict:
                        data_dict[meta_key] = meta_value

            # Concatenate tensors
            if tensors_to_concat:
                if len(tensors_to_concat) == 1:
                    concatenated_tensor = tensors_to_concat[0]
                else:
                    concatenated_tensor = torch.cat(tensors_to_concat, dim=self.concat_dim)

                # Determine output key name
                if self.output_keys and base_name in self.output_keys:
                    output_key = base_name
                else:
                    output_key = f"{base_name}_concat"

                data_dict[output_key] = concatenated_tensor
        # clean up
        for item in self.input_keys:
            if item in data_dict:
                data_dict.pip(item)

        return data_dict


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
        batch_hdmap_bbox_n_frames_per_view = data_dict["hdmap_bbox"].shape[1] // n_views
        assert batch_video_n_frames_per_view == num_video_frames_per_view, (
            f"Video must have {num_video_frames_per_view} frames, got {batch_video_n_frames_per_view}"
        )
        assert batch_hdmap_bbox_n_frames_per_view == num_video_frames_per_view, (
            f"Hdmap_bbox must have {num_video_frames_per_view} frames, got {batch_hdmap_bbox_n_frames_per_view}"
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
        # data_dict["hdmap_bbox"] = hdmap_bbox
        if self.embedding_type is not None:
            data_dict["t5_text_embeddings"] = t5_emb
            data_dict["t5_text_mask"] = t5_mask

        data_dict["control_weight"] = 1.0
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
        del data_dict["hdmap_bbox"]
        return data_dict


class OptionalKeyRenamer(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        assert len(input_keys) == len(output_keys)
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict | None:
        r"""Rename the dictionary associated with the input keys into data_dict if available.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with keys renamed.
        """
        for i, key in enumerate(self.input_keys):
            elem = data_dict.pop(self.input_keys[i], None)
            if elem is None:
                continue
            data_dict[self.output_keys[i]] = elem
        return data_dict


class SelectViews(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.views = args["views"]
        self.driving_dataloader_config = args["driving_dataloader_config"]

    def __call__(self, data_dict: dict) -> dict:
        r"""Select only some views from the data_dict."""
        return self.select_views(data_dict, self.views, self.driving_dataloader_config.camera_to_view_id)

    @staticmethod
    def select_views(data_batch: dict, views_to_keep: list[str], camera_to_view_id: dict[str, int]) -> dict:
        view_ids_to_keep = [camera_to_view_id[cam] for cam in views_to_keep]
        view_mask = [view_id in view_ids_to_keep for view_id in data_batch["view_indices_selection"]]
        assert sum(view_mask) == len(view_ids_to_keep), "Could not mask all requested views!"

        def filter_with_mask(lst):
            return [value for value, keep in zip(lst, view_mask, strict=True) if keep]

        # process dict elements
        for key in ["view_indices_selection", "camera_keys_selection", "n_orig_video_frames_per_view"]:
            data_batch[key] = filter_with_mask(data_batch[key])
        data_batch["sample_n_views"] = torch.tensor([len(views_to_keep)], dtype=torch.int64)
        captions = data_batch["ai_caption"]
        if isinstance(captions, (tuple, list)) and len(captions) > 1:
            data_batch["ai_caption"] = filter_with_mask(captions)
        elif isinstance(captions, str):
            maybe_multiple_captions = captions.split(" -- ")
            if len(maybe_multiple_captions) > 1:
                data_batch["ai_caption"] = " -- ".join(filter_with_mask(maybe_multiple_captions))

        view_index_mask = (data_batch["view_indices"].unsqueeze(0) == torch.tensor(view_ids_to_keep).unsqueeze(1)).any(
            dim=0
        )
        data_batch["view_indices"] = data_batch["view_indices"][view_index_mask]

        for key in ["video", "control_input_hdmap_bbox"]:
            if key in data_batch:
                arr = data_batch[key]
                data_batch[key] = arr[:, view_index_mask]

        return data_batch
