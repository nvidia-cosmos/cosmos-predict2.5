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

"""WebDataset loader for multi-view action-conditional robot data from S3.

This module extends the base WebDataset loader to support multi-view camera setups.
It builds on webdataset_s3.py the same way dataset_mv_local.py builds on dataset_local.py.

Supports multi-view wdinfo format where videos are organized by camera subdirectories:
  videos/base_0/00000000.tar
  videos/base_1/00000000.tar
  videos/wrist/00000000.tar
  annotations/00000000.tar

The wdinfo.json should include:
{
    "multi_view": true,
    "camera_ids": ["base_0", "base_1", "wrist"],
    "data_keys": ["videos", "annotations"],
    ...
}

Run this command to interactively debug:
PYTHONPATH=. python cosmos_predict2/_src/predict2/action/datasets/webdataset_mv_s3.py
"""

import json
import random
import time
from typing import Callable

import torch

from cosmos_predict2._src.imaginaire.datasets.webdataset.config.schema import DatasetInfo, TarSample
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.object_store import ObjectStore
from cosmos_predict2._src.predict2.action.datasets.webdataset_s3 import (
    ActionConditionedWebDatasetS3,
    ActionDataAugmentor,
)


class ActionDataAugmentorMultiView(ActionDataAugmentor):
    """Multi-view augmentor that concatenates frames from multiple cameras."""

    def _get_obs(
        self, sample: dict, frame_ids: list[int], cam_id: list | None = None, pre_encode: bool = False
    ) -> tuple[torch.Tensor, list[str]]:
        """Get observation frames from multiple camera views.

        Args:
            sample: WebDataset sample containing video data
            frame_ids: List of frame indices to extract
            pre_encode: Whether to use pre-encoded videos

        Returns:
            Tuple of (concatenated video tensor [T, C, H, W*num_views], camera IDs used)
        """
        del cam_id  # Unused - multi-view always uses self.cam_ids configuration
        # cam_ids format: [["base_0", "base_1"], "wrist_0"]
        # First element: list to randomly sample from
        # Second element: fixed camera
        # Note: Check for list-like objects (including OmegaConf ListConfig), not just Python list
        first_cam = self.cam_ids[0]
        is_list_like = not isinstance(first_cam, str) and hasattr(first_cam, "__iter__")
        temp_cam_id_0 = random.choice(list(first_cam)) if is_list_like else first_cam
        temp_cam_id_1 = self.cam_ids[1]

        frames_0 = self._get_frames(sample, frame_ids, cam_id=temp_cam_id_0, pre_encode=pre_encode)
        frames_1 = self._get_frames(sample, frame_ids, cam_id=temp_cam_id_1, pre_encode=pre_encode)
        # Concatenate along width dimension (dim=3 for [T, C, H, W])
        frames = torch.cat([frames_0, frames_1], dim=3)
        return frames, [temp_cam_id_0, temp_cam_id_1]


class MultiViewVideoOrganizer:
    """Organizes multi-view video data from WebDataset samples.

    When loading multi-view data, the base WebDataset creates keys like
    "videos/base_0", "videos/base_1", etc. This organizer restructures
    the sample to have:
        sample["videos"] = {
            "base_0": <video_bytes>,
            "base_1": <video_bytes>,
            ...
        }
    """

    def __init__(self, camera_ids: list[str]):
        """Initialize the multi-view video organizer.

        Args:
            camera_ids: List of camera IDs to organize
        """
        self.camera_ids = self._flatten_camera_ids(camera_ids)
        self.is_generator = True

    def _flatten_camera_ids(self, cam_ids: list) -> list[str]:
        """Flatten nested camera ID lists (e.g., [["base_0", "base_1"], "wrist"])."""
        flat_ids = []
        for cam_id in cam_ids:
            # Check for list-like types (including OmegaConf ListConfig), but not strings
            if not isinstance(cam_id, str) and hasattr(cam_id, "__iter__"):
                flat_ids.extend(cam_id)
            else:
                flat_ids.append(cam_id)
        return flat_ids

    def __call__(self, data_stream):
        """Reorganize video data in samples."""
        for sample in data_stream:
            try:
                # Look for video data with camera-specific keys
                videos_dict = {}
                missing_cams = []

                for cam_id in self.camera_ids:
                    video_key = f"videos_{cam_id}"
                    video_data = sample.get(video_key)
                    if video_data is not None:
                        videos_dict[cam_id] = video_data
                    else:
                        missing_cams.append(cam_id)

                # If we found any camera videos, organize them
                if videos_dict:
                    sample["videos"] = videos_dict

                yield sample

            except Exception as e:
                log.warning(f"Error organizing multi-view sample: {e}")
                yield sample


class ActionConditionedMultiViewWebDatasetS3(ActionConditionedWebDatasetS3):
    """Multi-view WebDataset loader extending the base single-view loader.

    Supports multi-view wdinfo format where videos are organized by camera subdirectories.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the multi-view WebDataset S3 loader."""
        # Multi-view specific attributes (set during wdinfo parsing)
        self.multi_view = False
        self.camera_ids_from_wdinfo: list[str] = []

        super().__init__(*args, **kwargs)

    def parse_dataset_info(self, dataset_info: list[DatasetInfo], use_multithread: bool = True) -> None:
        """Parse metadata about the list of tar files with multi-view support.

        This overrides the base method to handle multi-view wdinfo format where
        videos are stored in camera-specific subdirectories.
        """
        log.info(f"[MultiView] Start parsing dataset info with {len(dataset_info)} entries")
        tic = time.time()

        for dset_num, dset_info in enumerate(dataset_info):
            if len(dset_info.wdinfo) == 0:
                log.warning(f"No wdinfo found for dataset {dset_num}, skipping...")
                continue

            use_object_store = dset_info.object_store_config.enabled
            self.use_object_store = use_object_store
            dset_id = f"dset: {dset_num}"

            if use_object_store:
                object_store_reader = ObjectStore(config_object_storage=dset_info.object_store_config)
                bucket_dset = dset_info.object_store_config.bucket
                s3_client_dset = object_store_reader.client
            else:
                object_store_reader = None
                s3_client_dset = None
                bucket_dset = None

            tar_samples = []
            total_key_count = 0
            chunk_sizes = []

            for wdinfo_path in dset_info.wdinfo:
                log.info(f"[MultiView] Processing wdinfo: {wdinfo_path}")

                if use_object_store:
                    if not object_store_reader.object_exists(wdinfo_path):
                        raise FileNotFoundError(f"{wdinfo_path} not found")
                    cur_dset_info = object_store_reader.load_object(key=wdinfo_path, type="json")
                else:
                    with open(wdinfo_path) as fp:
                        cur_dset_info = json.load(fp)

                data_root = cur_dset_info["root"]
                # Strip s3://bucket/ prefix from root if present
                if data_root.startswith("s3://"):
                    parts = data_root[5:].split("/", 1)
                    data_root = parts[1] if len(parts) > 1 else ""

                tar_files_list = cur_dset_info["data_list"]
                is_multi_view = cur_dset_info.get("multi_view", False)
                camera_ids = cur_dset_info.get("camera_ids", [])
                data_keys = cur_dset_info.get("data_keys", self.data_keys)

                if is_multi_view:
                    self.multi_view = True
                    self.camera_ids_from_wdinfo = camera_ids
                    log.info(f"[MultiView] Detected multi-view dataset with cameras: {camera_ids}")
                    log.info(f"[MultiView] Data keys: {data_keys}")

                    # For multi-view, we need to create separate "virtual" keys for each camera
                    # The base WebDataset will construct paths as: root/key/tar_file
                    # We need: root/videos/camera_id/tar_file for videos
                    #          root/annotations/tar_file for other data

                    # Create modified keys list for multi-view
                    multi_view_keys = []
                    for key in data_keys:
                        if key == "videos":
                            # Add a key for each camera: "videos/base_0", "videos/base_1", etc.
                            for cam_id in camera_ids:
                                multi_view_keys.append(f"videos/{cam_id}")
                        else:
                            multi_view_keys.append(key)

                    log.info(f"[MultiView] Expanded keys for loading: {multi_view_keys}")

                    local_tar_samples = [
                        TarSample(
                            path=tar_file,
                            root=data_root,
                            keys=(dset_info.per_dataset_keys if dset_info.per_dataset_keys else multi_view_keys),
                            meta=dset_info,
                            dset_id=dset_id,
                            sample_keys_full_list=None,
                        )
                        for tar_file in tar_files_list
                    ]
                else:
                    # Standard single-view handling - delegate to parent
                    local_tar_samples = [
                        TarSample(
                            path=tar_file,
                            root=data_root,
                            keys=(dset_info.per_dataset_keys if dset_info.per_dataset_keys else self.data_keys),
                            meta=dset_info,
                            dset_id=dset_id,
                            sample_keys_full_list=None,
                        )
                        for tar_file in tar_files_list
                    ]

                tar_samples.extend(local_tar_samples)
                total_key_count += cur_dset_info["total_key_count"]
                chunk_sizes.append(cur_dset_info["chunk_size"])

            # Store results
            self.wdinfo.tar_files.extend(tar_samples)
            self.wdinfo.total_key_count += total_key_count
            if chunk_sizes:
                self.wdinfo.chunk_size = chunk_sizes[0]
            if s3_client_dset:
                self.s3_client[dset_id] = s3_client_dset
            if bucket_dset:
                self.bucket[dset_id] = bucket_dset

        toc = time.time()
        log.info(
            f"[MultiView] Parsed {len(dataset_info)} wdinfos "
            f"(num_keys={self.wdinfo.total_key_count}, num_tars={len(self.wdinfo.tar_files)}) "
            f"in {(toc - tic):.2f}s"
        )

    def build_data_augmentor(self, augmentor_cfg: dict) -> Callable:
        """Build multi-view data augmentor with video organizer."""
        from functools import partial

        from cosmos_predict2._src.imaginaire.datasets.webdataset.webdataset import Dataset as WebDatasetBase
        from cosmos_predict2._src.imaginaire.lazy_config import instantiate

        augmentations: list = []

        # Add multi-view video organizer if using multi-view wdinfo
        if self.multi_view:
            multi_view_organizer = MultiViewVideoOrganizer(camera_ids=self.cam_ids)
            augmentations.append(multi_view_organizer)

        action_augmentor = ActionDataAugmentorMultiView(
            fps_downsample_ratio=self.fps_downsample_ratio,
            num_action_per_chunk=self.num_action_per_chunk,
            accumulate_action=self.accumulate_action,
            video_size=self.video_size,
            normalize=self.normalize,
            load_action=self.load_action,
            load_t5_embeddings=self.load_t5_embeddings,
            state_key=self.state_key,
            gripper_key=self.gripper_key,
            gripper_rescale_factor=self.gripper_rescale_factor,
            cam_ids=self.cam_ids,
        )
        augmentations.append(action_augmentor)

        for aug in augmentor_cfg.keys():
            augmentations.append(instantiate(augmentor_cfg[aug]))

        return partial(WebDatasetBase.augmentor_fn, augmentations=augmentations)


if __name__ == "__main__":
    from cosmos_predict2._src.imaginaire.config import ObjectStoreConfig
    from cosmos_predict2._src.imaginaire.datasets.webdataset.config.schema import DatasetConfig, DatasetInfo
    from cosmos_predict2._src.imaginaire.datasets.webdataset.distributors import ShardlistBasic
    from cosmos_predict2._src.imaginaire.utils import log

    dataset_info = DatasetInfo(
        wdinfo=[
            "user/sync_gcp/pi_ablation_20251010/wdinfo/short_high_gripper_movement_segmented_episodes_30h_webdataset/wdinfo.json"
        ],
        object_store_config=ObjectStoreConfig(
            enabled=True, bucket="debug", credentials="credentials/s3_robotics.secret"
        ),
        per_dataset_keys=[],
    )

    config = DatasetConfig(
        dataset_info=[dataset_info],
        keys=["videos", "annotations"],
        streaming_download=True,
        buffer_size=100,
        augmentation={},
        distributor=ShardlistBasic(),
        decoders=["rgb"],
        remove_extension_from_keys=True,
    )

    dataset = ActionConditionedMultiViewWebDatasetS3(
        config=config,
        fps_downsample_ratio=2,
        num_action_per_chunk=12,
        cam_ids=[["base_0", "base_1"], "wrist"],
        accumulate_action=False,
        video_size=[480, 640],
        load_action=True,
        load_t5_embeddings=False,
        state_key="ee_pose",
        gripper_key="gripper_chunk",
        gripper_rescale_factor=10.0,
    )

    webdataset = dataset.build_dataset()
    log.info(f"Created multi-view WebDataset with {dataset.wdinfo.total_key_count} total keys")

    for i, sample in enumerate(webdataset):
        log.info(f"Sample {i}: keys={list(sample.keys())}")
        if "video" in sample:
            log.info(f"  video shape: {sample['video'].shape}")
        if "action" in sample:
            log.info(f"  action shape: {sample['action'].shape}")
        if i >= 2:
            break
