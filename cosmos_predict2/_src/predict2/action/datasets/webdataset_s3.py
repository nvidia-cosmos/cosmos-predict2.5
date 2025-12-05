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

"""WebDataset loader for action-conditional robot data from S3.

This module provides a WebDataset-based loader for reading robot action data
stored in tar files on S3. It reuses the WebDataset infrastructure from
cosmos_predict2._src.imaginaire and maintains compatibility with the output format of dataset_s3.py.

Supports both single-view and multi-view datasets:
- Single-view: videos stored in videos/{tar_file}
- Multi-view: videos stored in videos/{camera_id}/{tar_file}

For multi-view, the wdinfo.json should include:
{
    "multi_view": true,
    "camera_ids": ["base_0", "base_1", "wrist"],
    ...
}
"""

import io
import json
import time
from typing import Callable

import numpy as np
import torch
from decord import VideoReader, cpu
from omegaconf import DictConfig
from torchvision import transforms as T
from webdataset.handlers import reraise_exception

from cosmos_predict2._src.imaginaire.datasets.webdataset.config.schema import (
    AugmentorConfig,
    DatasetConfig,
    DatasetInfo,
    TarSample,
)
from cosmos_predict2._src.imaginaire.datasets.webdataset.webdataset import Dataset as WebDatasetBase
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.dataset_utils import Resize_Preprocess, ToTensorVideo, euler2rotm, rotm2euler
from cosmos_predict2._src.imaginaire.utils.object_store import ObjectStore


class ActionDataAugmentor:
    """Augmentor to transform WebDataset samples into action dataset format.

    This augmentor processes WebDataset tar samples containing robot trajectories
    and formats them to match the output of dataset_s3.py.
    """

    def __init__(
        self,
        fps_downsample_ratio: int = 1,
        num_action_per_chunk: int = 1,
        accumulate_action: bool = False,
        video_size: list[int] | None = None,
        normalize: bool = False,
        load_action: bool = True,
        load_t5_embeddings: bool = False,
        state_key: str = "state",
        gripper_key: str = "gripper_chunk",
        gripper_rescale_factor: float = 1.0,
        cam_ids: list | None = None,
        c_act_scaler: np.ndarray | None = None,
    ):
        """Initialize the action data augmentor.

        Args:
            fps_downsample_ratio: Interval between sampled frames
            num_action_per_chunk: Number of actions per sequence (NOT frames - we need num_action_per_chunk + 1 frames)
            accumulate_action: Whether to accumulate actions relative to first frame
            video_size: Target [H, W] for video frames
            normalize: Whether to normalize video frames
            load_action: Whether to load actions
            load_t5_embeddings: Whether to load T5 embeddings
            state_key: Key to access robot states
            gripper_key: Key to access gripper states
            gripper_rescale_factor: Scaling factor for gripper actions
            cam_ids: List of camera IDs to sample from
            c_act_scaler: Action scaling factors
        """
        self.fps_downsample_ratio = fps_downsample_ratio
        self.num_action_per_chunk = num_action_per_chunk
        # sequence_length is the number of frames, which is num_action_per_chunk + 1
        # This matches the behavior of dataset_local.py
        self.sequence_length = 1 + num_action_per_chunk
        self.accumulate_action = accumulate_action
        self.video_size = video_size
        self.normalize = normalize
        self.load_action = load_action
        self.load_t5_embeddings = load_t5_embeddings
        self.state_key = state_key
        self.gripper_key = gripper_key
        self.gripper_rescale_factor = gripper_rescale_factor
        self.cam_ids = cam_ids or ["base_0"]
        self.is_generator = True  # Mark as generator for WebDataset

        # Default action scaler if not provided
        if c_act_scaler is None:
            self.c_act_scaler = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, gripper_rescale_factor])
        else:
            self.c_act_scaler = c_act_scaler

        # Initialize video transforms (matching dataset_local.py)
        self.to_tensor_video = ToTensorVideo()
        video_size_tuple = tuple(self.video_size) if self.video_size else (256, 256)
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
                Resize_Preprocess(video_size_tuple),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.not_norm_preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(video_size_tuple)])

    def _get_robot_states(self, data: dict, frame_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Extract robot arm and gripper states for specified frames."""
        # Get states from the data dict - try multiple possible keys
        state_keys_to_try = [
            self.state_key,
        ]
        states = None
        for sk in state_keys_to_try:
            if sk in data and data[sk] is not None:
                states = data[sk]
                break

        gripper_keys_to_try = [self.gripper_key]
        gripper_states = None
        for gk in gripper_keys_to_try:
            if gk in data and data[gk] is not None:
                gripper_states = data[gk]
                break

        if states is None:
            available_keys = list(data.keys())
            raise KeyError(
                f"Could not find state data with key '{self.state_key}'. "
                f"Tried keys: {state_keys_to_try}. Available keys in data: {available_keys}"
            )

        # Extract states for the requested frames
        arm_states = []
        gripper_values = []

        for frame_id in frame_ids:
            if isinstance(states, (list, np.ndarray)):
                state = states[frame_id]
            else:
                state = states

            # Handle different state formats
            if isinstance(state, dict):
                # Dictionary format with position and rotation
                position = state.get("position", state.get("xyz", state.get("pos")))
                rotation = state.get("rotation", state.get("euler", state.get("rot")))
                arm_state = np.concatenate([position, rotation])
            else:
                # Direct array format [x, y, z, rx, ry, rz]
                arm_state = np.array(state[:6])

            arm_states.append(arm_state)

            # Get gripper state
            if gripper_states is not None:
                if isinstance(gripper_states, (list, np.ndarray)):
                    gripper = gripper_states[frame_id]
                else:
                    gripper = gripper_states
                gripper_values.append(float(gripper))
            else:
                # Default gripper state if not provided
                gripper_values.append(0.0)

        return np.array(arm_states), np.array(gripper_values)

    def _get_actions(self, arm_states: np.ndarray, gripper_states: np.ndarray, accumulate_action: bool) -> torch.Tensor:
        """Compute relative actions between consecutive frames.

        This matches the implementation in dataset_local.py exactly.
        """
        l, _ = arm_states.shape
        action = np.zeros((l - 1, 7), dtype=np.float32)

        if accumulate_action:
            # Accumulate actions relative to base frame, with reset every 4 frames
            # This matches dataset_local.py behavior
            base_xyz = arm_states[0, 0:3]
            base_rpy = arm_states[0, 3:6]
            base_rotm = euler2rotm(base_rpy)
            for k in range(1, l):
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(base_rotm.T, curr_xyz - base_xyz)
                rel_rotm = base_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
                if k % 4 == 0:
                    base_xyz = arm_states[k, 0:3]
                    base_rpy = arm_states[k, 3:6]
                    base_rotm = euler2rotm(base_rpy)
        else:
            # Compute relative actions between consecutive frames
            for k in range(1, l):
                prev_xyz = arm_states[k - 1, 0:3]
                prev_rpy = arm_states[k - 1, 3:6]
                prev_rotm = euler2rotm(prev_rpy)
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                rel_rotm = prev_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper

        return torch.from_numpy(action)

    def _get_frames(self, sample: dict, frame_ids: list[int], cam_id: str, pre_encode: bool) -> torch.Tensor:
        """Get video frames for a specific camera.

        Args:
            sample: WebDataset sample containing video data. For multi-view,
                    videos are organized as sample["videos"][cam_id] = video_bytes.
                    For single-view, sample["video"] or sample["videos"]["video"] = video_bytes.
            frame_ids: List of frame indices to extract
            cam_id: Camera ID to look for
            pre_encode: Whether to use pre-encoded videos

        Returns:
            Video tensor of shape [T, C, H, W]
        """
        if pre_encode:
            raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")

        # Try to get video data using the expected key format: videos_<cam_id>
        video_key = f"videos_{cam_id}"
        video_data = sample.get(video_key)

        # If not found, try alternative structures
        if video_data is None and "videos" in sample and isinstance(sample["videos"], dict):
            if cam_id in sample["videos"]:
                video_data = sample["videos"][cam_id]
            elif "video" in sample["videos"]:
                # Single video in videos dict
                video_data = sample["videos"]["video"]

        # If still not found, raise a helpful error with available keys
        if video_data is None:
            available_keys = [k for k in sample.keys() if not k.startswith("__")]
            raise KeyError(
                f"Video data not found for camera '{cam_id}'. Tried key '{video_key}'. Available keys: {available_keys}"
            )

        # Handle nested dict structure - extract bytes from dict
        if video_data is not None and isinstance(video_data, dict):
            # Try common keys that might contain video bytes
            for video_key in ["video", "video_path", "mp4", "data"]:
                if video_key in video_data and isinstance(video_data[video_key], bytes):
                    video_data = video_data[video_key]
                    break
            else:
                # If no known key found, look for any bytes value
                for v in video_data.values():
                    if isinstance(v, bytes):
                        video_data = v
                        break

        if video_data is not None and isinstance(video_data, bytes):
            frames = self._load_video(video_data, frame_ids)
            frames = frames.astype(np.uint8)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
        elif video_data is not None:
            extra_info = ""
            if isinstance(video_data, dict):
                extra_info = f", dict keys: {list(video_data.keys())}"
            log.warning(f"Unexpected video data type: {type(video_data)}, expected bytes{extra_info}")
            H, W = self.video_size or [256, 256]
            frames = torch.zeros(len(frame_ids), 3, H, W, dtype=torch.uint8)
        else:
            H, W = self.video_size or [256, 256]
            frames = torch.zeros(len(frame_ids), 3, H, W, dtype=torch.uint8)

        if self.normalize:
            frames = self.preprocess(frames)
        else:
            frames = self.not_norm_preprocess(frames)
            frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames

    def _get_obs(self, sample: dict, frame_ids: list[int], cam_id: str | None, pre_encode: bool):
        """Get observation frames from the sample.

        Args:
            sample: WebDataset sample containing video data
            frame_ids: List of frame indices to extract
            cam_id: Camera ID to use, or None to randomly select
            pre_encode: Whether to use pre-encoded videos

        Returns:
            Tuple of (video tensor [T, C, H, W], camera ID used)
        """
        if cam_id is None:
            selected_cam_id: str = np.random.choice(self.cam_ids) if len(self.cam_ids) > 1 else self.cam_ids[0]
        else:
            selected_cam_id = cam_id
        frames = self._get_frames(sample, frame_ids, cam_id=selected_cam_id, pre_encode=pre_encode)
        return frames, selected_cam_id

    def __call__(self, data_stream):
        """Process WebDataset samples and yield formatted data."""
        sample_count = 0
        for sample in data_stream:
            try:
                # Debug: Log first few samples to understand what keys are present
                if sample_count < 3:
                    all_keys = list(sample.keys())
                    sample_count += 1

                # Extract key information
                key = sample.get("__key__", "unknown")
                url = sample.get("__url__")  # Preserve URL for downstream processing

                annotation = {}
                ann_data = sample["annotations"]
                # Handle both raw JSON strings and already-parsed dicts
                if isinstance(ann_data, str):
                    annotation = json.loads(ann_data)
                elif isinstance(ann_data, bytes):
                    annotation = json.loads(ann_data.decode("utf-8"))
                elif isinstance(ann_data, dict):
                    annotation = ann_data

                # Determine frame indices to use
                if "frame_ids" in sample:
                    frame_ids = sample["frame_ids"]
                else:
                    # Create frame indices based on sequence length
                    # sequence_length = num_action_per_chunk + 1 (matching dataset_local.py behavior)
                    total_frames = len(annotation.get(self.state_key, []))
                    frame_ids = list(
                        range(
                            0,
                            min(total_frames, self.sequence_length * self.fps_downsample_ratio),
                            self.fps_downsample_ratio,
                        )
                    )

                # Get video frames using _get_obs (can be overridden for multi-view)
                video_tensor, _ = self._get_obs(sample, frame_ids, cam_id="base_0", pre_encode=False)
                # Permute from [T, C, H, W] to [C, T, H, W]
                video_tensor = video_tensor.permute(1, 0, 2, 3)

                # Get robot states and compute actions
                output_data = {"__key__": key}
                if url is not None:
                    output_data["__url__"] = url

                if self.load_action:
                    # Merge annotation data with sample for state extraction
                    merged_data = {**sample, **annotation}
                    arm_states, gripper_states = self._get_robot_states(merged_data, frame_ids)
                    actions = self._get_actions(arm_states, gripper_states, self.accumulate_action)
                    actions *= self.c_act_scaler
                    output_data["action"] = actions.float()

                # Add video data
                output_data["video"] = video_tensor.to(dtype=torch.uint8)

                # Add annotation file reference
                output_data["annotation_file"] = f"tar:{key}"

                # Add T5 embeddings and metadata
                if self.load_t5_embeddings and "t5_embeddings.npy" in sample:
                    t5_embeddings = np.load(sample["t5_embeddings.npy"])
                    output_data["t5_text_embeddings"] = torch.from_numpy(t5_embeddings)
                else:
                    output_data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16)
                    output_data["ai_caption"] = annotation.get("caption", "")

                output_data["t5_text_mask"] = torch.ones(512, dtype=torch.int64)
                output_data["fps"] = 4  # Default FPS
                output_data["image_size"] = 256 * torch.ones(4)
                output_data["num_frames"] = len(frame_ids)
                output_data["padding_mask"] = torch.zeros(1, 256, 256)

                yield output_data

            except Exception as e:
                log.warning(f"Error processing sample {sample.get('__key__', 'unknown')}: {e}")
                continue

    def _load_video(self, video_data: bytes, frame_ids: list[int]) -> np.ndarray:
        """Process raw video data and extract frames.

        Args:
            video_data: Raw video bytes
            frame_ids: List of frame indices to extract

        Returns:
            Video frames as numpy array of shape [T, H, W, C]
        """
        vr = VideoReader(io.BytesIO(video_data), ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        return frame_data


class ActionConditionedWebDatasetS3(WebDatasetBase):
    """WebDataset loader for action-conditional robot data from S3.

    This class extends the base WebDataset to load robot action data stored
    in WebDataset tar format on S3, maintaining compatibility with the
    output format of ActionConditionedDatasetS3.

    Supports both single-view and multi-view folder structures:
    - Videos are expected at: {root}/videos/{camera_id}/{tar_file}
    - Other data (annotations, etc.) at: {root}/{key}/{tar_file}

    For single-view, pass cam_ids=["base_0"] (or whichever single camera).
    For multi-view, pass cam_ids=["base_0", "base_1", "wrist"].
    """

    def __init__(
        self,
        config: DatasetConfig | DictConfig,
        fps_downsample_ratio: int = 1,
        num_action_per_chunk: int = 1,
        cam_ids: list | None = None,
        accumulate_action: bool = False,
        video_size: list[int] | None = None,
        normalize: bool = False,
        load_action: bool = True,
        load_t5_embeddings: bool = False,
        state_key: str = "state",
        gripper_key: str = "gripper_chunk",
        gripper_rescale_factor: float = 1.0,
        handler: Callable | None = None,
        **kwargs,
    ):
        """Initialize the WebDataset S3 loader.

        Args:
            config: WebDataset configuration with S3 paths
            fps_downsample_ratio: Interval between sampled frames
            num_action_per_chunk: Number of actions per sequence (NOT frames - we need num_action_per_chunk + 1 frames)
            cam_ids: List of camera IDs to sample from (e.g., ["base_0"] for single-view)
            accumulate_action: Whether to accumulate actions
            video_size: Target [H, W] for video frames
            normalize: Whether to normalize video frames
            load_action: Whether to load actions
            load_t5_embeddings: Whether to load T5 embeddings
            state_key: Key to access robot states
            gripper_key: Key to access gripper states
            gripper_rescale_factor: Scaling factor for gripper
            handler: Error handler
            **kwargs: Additional arguments passed to base class
        """
        # Store cam_ids BEFORE calling super().__init__() since parse_dataset_info needs it
        self.cam_ids = cam_ids or ["base_0"]

        # Initialize base WebDataset with S3 support
        if handler is None:
            handler = reraise_exception
        super().__init__(config=config, handler=handler)

        # Store action dataset specific parameters
        self.fps_downsample_ratio = fps_downsample_ratio
        self.num_action_per_chunk = num_action_per_chunk
        self.accumulate_action = accumulate_action
        self.video_size = video_size
        self.normalize = normalize
        self.load_action = load_action
        self.load_t5_embeddings = load_t5_embeddings
        self.state_key = state_key
        self.gripper_key = gripper_key
        self.gripper_rescale_factor = gripper_rescale_factor

    def parse_dataset_info(self, dataset_info: list[DatasetInfo], use_multithread: bool = True) -> None:
        """Parse metadata about the list of tar files with camera-aware path expansion.

        This overrides the base method to expand video keys with camera IDs.
        For example, if keys=["videos", "annotations"] and cam_ids=["base_0"],
        the expanded keys become ["videos/base_0", "annotations"].

        This allows loading from folder structures like:
            videos/base_0/00000000.tar
            annotations/00000000.tar
        """
        log.info(f"[ActionDataset] Start parsing dataset info with {len(dataset_info)} entries")
        log.info(f"[ActionDataset] Camera IDs: {self.cam_ids}")
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
                log.info(f"[ActionDataset] Processing wdinfo: {wdinfo_path}")

                if use_object_store:
                    if not object_store_reader.object_exists(wdinfo_path):
                        raise FileNotFoundError(f"{wdinfo_path} not found")
                    cur_dset_info = object_store_reader.load_object(key=wdinfo_path, type="json")
                else:
                    with open(wdinfo_path) as fp:
                        cur_dset_info = json.load(fp)

                # Debug: Log wdinfo content for troubleshooting
                log.info(
                    f"[ActionDataset] wdinfo content: root={cur_dset_info.get('root')}, "
                    f"data_keys={cur_dset_info.get('data_keys')}, "
                    f"num_tars={len(cur_dset_info.get('data_list', []))}"
                )

                data_root = cur_dset_info["root"]
                # Strip s3://bucket/ prefix from root if present
                if data_root.startswith("s3://"):
                    parts = data_root[5:].split("/", 1)
                    data_root = parts[1] if len(parts) > 1 else ""

                tar_files_list = cur_dset_info["data_list"]

                # Get data keys from wdinfo or fall back to config keys
                data_keys = cur_dset_info.get("data_keys", self.data_keys)

                # Expand video keys with camera IDs
                # e.g., ["videos", "annotations"] -> ["videos/base_0", "annotations"]
                expanded_keys = []
                for key in data_keys:
                    if key == "videos":
                        # Add a key for each camera: "videos/base_0", "videos/base_1", etc.
                        for cam_id in self.cam_ids:
                            expanded_keys.append(f"videos/{cam_id}")
                    else:
                        expanded_keys.append(key)

                log.info(f"[ActionDataset] Original keys: {data_keys}")
                log.info(f"[ActionDataset] Expanded keys: {expanded_keys}")

                # Debug: Show example tar paths that will be constructed
                if tar_files_list:
                    example_tar = tar_files_list[0]
                    log.info(f"[ActionDataset] Example tar paths for '{example_tar}':")
                    for key in expanded_keys:
                        full_path = f"{data_root}/{key}/{example_tar}"
                        log.info(f"[ActionDataset]   -> {full_path}")

                local_tar_samples = [
                    TarSample(
                        path=tar_file,
                        root=data_root,
                        keys=(dset_info.per_dataset_keys if dset_info.per_dataset_keys else expanded_keys),
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
            f"[ActionDataset] Parsed {len(dataset_info)} wdinfos "
            f"(num_keys={self.wdinfo.total_key_count}, num_tars={len(self.wdinfo.tar_files)}) "
            f"in {(toc - tic):.2f}s"
        )

    def build_data_augmentor(self, augmentor_cfg: dict[str, AugmentorConfig]) -> Callable:
        """Build data augmentors including the action data processor.

        This overrides the base method to add our custom ActionDataAugmentor
        at the beginning of the augmentation pipeline.
        """
        # Create action data augmentor
        action_augmentor = ActionDataAugmentor(
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

        # Build other augmentors from config
        augmentations = [action_augmentor]
        for aug in augmentor_cfg.keys():
            augmentations.append(instantiate(augmentor_cfg[aug]))

        # Return the augmentor function
        from functools import partial

        return partial(WebDatasetBase.augmentor_fn, augmentations=augmentations)


# Example usage and testing
if __name__ == "__main__":
    """Test the WebDataset S3 loader.
    
    Run with:
    PYTHONPATH=. python cosmos_predict2/_src/predict2/action/datasets/webdataset_s3.py
    """

    from cosmos_predict2._src.imaginaire.config import ObjectStoreConfig
    from cosmos_predict2._src.imaginaire.datasets.webdataset.config.schema import DatasetConfig, DatasetInfo
    from cosmos_predict2._src.imaginaire.datasets.webdataset.distributors import ShardlistBasic

    # Create dataset info objects
    dataset_info = DatasetInfo(
        wdinfo=[
            "user/sync_gcp/pi_ablation_20251010/wdinfo/short_high_gripper_movement_segmented_episodes_30h_webdataset/wdinfo.json"
        ],
        object_store_config=ObjectStoreConfig(
            enabled=True, bucket="debug", credentials="credentials/s3_robotics.secret"
        ),
        per_dataset_keys=[],  # Use empty list instead of None
    )

    # Create the dataset configuration
    config = DatasetConfig(
        dataset_info=[dataset_info],
        keys=["video", "json", "states"],
        streaming_download=True,
        buffer_size=100,
        augmentation={},
        distributor=ShardlistBasic(),
        decoders=["rgb", "json"],
        remove_extension_from_keys=True,
    )

    # Create dataset
    dataset = ActionConditionedWebDatasetS3(
        config=config,
        fps_downsample_ratio=1,
        num_action_per_chunk=16,
        cam_ids=["base_0"],
        accumulate_action=False,
        video_size=[256, 256],
        load_action=True,
        load_t5_embeddings=False,
        state_key="ee_pose",
    )

    # Build the actual dataset
    webdataset = dataset.build_dataset()

    log.info(f"Created WebDataset with {dataset.wdinfo.total_key_count} total keys")

    # Example of expected tar file structure:
    # Each tar file should contain samples with the following files per sample:
    # - {sample_id}.json: annotation file with robot states, episode metadata
    # - {sample_id}.base_0.mp4: video from camera "base_0"
    # - {sample_id}.t5_embeddings.npy: (optional) T5 text embeddings
    #
    # The json file should contain:
    # {
    #     "episode_id": "episode_001",
    #     "ee_pose": [...],  // or "state" - list of robot states per frame
    #     "gripper_state": [...],  // gripper states per frame
    #     "caption": "robot picking up object"  // optional text description
    # }
    #
    # === SINGLE-VIEW wdinfo.json ===
    # The wdinfo.json file should follow the standard WebDataset format:
    # {
    #     "root": "s3://bucket/path/to/tars/",
    #     "data_list": ["shard_000.tar", "shard_001.tar", ...],
    #     "total_key_count": 10000,
    #     "chunk_size": 1000
    # }
    #
    # === MULTI-VIEW ===
    # For multi-view datasets, see webdataset_mv_s3.py which provides:
    # - ActionConditionedMultiViewWebDatasetS3: extends this class for multi-view
    # - ActionDataAugmentorMultiView: augmentor that concatenates camera views
    #
    # Multi-view wdinfo.json format:
    # {
    #     "root": "s3://bucket/path/to/tars/",
    #     "data_list": ["00000000.tar", "00000001.tar", ...],
    #     "data_keys": ["videos", "annotations", "actions"],
    #     "multi_view": true,
    #     "camera_ids": ["base_0", "base_1", "wrist"],
    #     "total_key_count": 10000,
    #     "chunk_size": 100
    # }
    #
    # Multi-view directory structure:
    #   output_dir/
    #   ├── videos/
    #   │   ├── base_0/
    #   │   │   ├── 00000000.tar
    #   │   │   └── 00000001.tar
    #   │   ├── base_1/
    #   │   │   └── ...
    #   │   └── wrist/
    #   │       └── ...
    #   ├── annotations/
    #   │   ├── 00000000.tar
    #   │   └── 00000001.tar
    #   └── actions/
    #       └── ...
