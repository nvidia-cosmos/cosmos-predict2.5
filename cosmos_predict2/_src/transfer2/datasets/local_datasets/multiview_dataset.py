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


import json
import os
import pickle
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import (
    ResizePreprocess,
    ToTensorVideo,
)
from cosmos_predict2._src.transfer2.datasets.augmentor_provider import get_hdmap_augmentor_for_local_datasets
from cosmos_predict2._src.transfer2.utils.input_handling import detect_aspect_ratio

# mappings between control types and corresponding sub-folders names in the data folder
CTRL_TYPE_INFO = {
    "keypoint": {"folder": "keypoint", "format": "pickle", "data_dict_key": "keypoint"},
    "depth": {"folder": "depth", "format": "mp4", "data_dict_key": "depth"},
    "lidar": {"folder": "lidar", "format": "mp4", "data_dict_key": "lidar"},
    "hdmap_bbox": {"folder": "control_input_hdmap_bbox", "format": "mp4", "data_dict_key": "hdmap"},
    "seg": {"folder": "seg", "format": "pickle", "data_dict_key": "segmentation"},
    "edge": {"folder": None},  # Canny edge, computed on-the-fly
    "vis": {"folder": None},  # Blur, computed on-the-fly
    "upscale": {"folder": None},  # Computed on-the-fly
}

AUTO_MV_DEFAULT_PROMPT = 'This multi-camera perspective captures a drive along a multi-lane urban freeway during the daytime under a hazy or partly cloudy sky. The vehicle travels in one of the right lanes, flanked on one side by a high retaining wall featuring a concrete base and a brown, brick-patterned upper section with some climbing vines, and on the other side by a concrete median barrier. As the car moves forward, it approaches and passes under a large concrete overpass, which frames a view of a distant downtown city skyline with numerous high-rise buildings. A green freeway sign for the "Hill St / Grand Ave" exit is briefly visible, and another overpass, distinguished by its overhead catenary power lines suggesting a light rail system, also crosses the roadway. The flow of traffic is moderate, with the camera vehicle sharing the road with other cars, including a white Dodge Grand Caravan minivan, a dark-colored SUV, and a black sedan, which are visible at various points ahead and behind. The asphalt road surface shows some visible cracks and wear, contributing to the overall scene of a typical day on a major metropolitan highway.'


class MultiviewTransferDataset(Dataset):
    """
    A robust dataset for multi-view, control-based video generation.

    This class merges the multi-view loading logic from 'predict2' with the
    control signal handling and augmentation from 'transfer2'. It loads multiple
    camera views, their corresponding control signals, and processes them using
    the transfer2 augmentor pipeline.
    """

    def __init__(
        self,
        dataset_dir,
        num_frames,
        resolution,
        video_size,
        hint_key,
        camera_keys,
        front_camera_key=None,
        camera_to_view_id: Dict = None,
        sequence_interval=1,
        start_frame_interval=1,
        state_t=8,
        front_view_caption_only=True,
        is_train=True,
        **kwargs,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.resolution = resolution
        self.hint_key = hint_key
        self.camera_keys = camera_keys
        self.camera_to_view_id = camera_to_view_id
        self.front_camera_key = front_camera_key
        self.sequence_interval = sequence_interval
        self.start_frame_interval = start_frame_interval
        self.state_t = state_t
        self.front_view_caption_only = front_view_caption_only
        self.is_train = is_train
        self.H, self.W = video_size

        self.ctrl_type = hint_key.replace("control_input_", "")
        ctrl_types = []
        for ctrl_type in self.ctrl_type.split("_"):
            if ctrl_type == "hdmap":
                ctrl_types.append("hdmap_bbox")
            elif ctrl_type == "bbox":
                continue
            else:
                ctrl_types.append(ctrl_type)
        self.ctrl_types = ctrl_types
        for ctrl_type in self.ctrl_types:
            if ctrl_type not in CTRL_TYPE_INFO:
                raise ValueError(f"Control type '{ctrl_type}' not defined in CTRL_TYPE_INFO.")

        augmentor_cfg = get_hdmap_augmentor_for_local_datasets(resolution=resolution, control_input_type=self.ctrl_type)
        self.augmentor = {k: instantiate(v) for k, v in augmentor_cfg.items()}

        video_dir = os.path.join(self.dataset_dir, "videos")

        # Use the front camera to define the list of video episodes
        main_camera_video_dir = os.path.join(video_dir, self.front_camera_key)
        self.captions_dir = os.path.join(self.dataset_dir, "captions", self.front_camera_key)
        self.video_paths = sorted([os.path.join(main_camera_video_dir, f) for f in os.listdir(main_camera_video_dir)])

        # Train/val split
        cutoff = int(len(self.video_paths) * 0.1) + 1
        self.video_paths = self.video_paths[:-cutoff] if is_train else self.video_paths[-cutoff:]
        log.info(f"Using {len(self.video_paths)} videos for {'training' if is_train else 'validation'}.")

        self.samples = self._init_samples(self.video_paths)
        log.info(f"{len(self.samples)} total samples initialized.")

        self.num_failed_loads = 0
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess((video_size[0], video_size[1]))])

    def __str__(self) -> str:
        return f"{len(self.samples)} samples from {self.dataset_dir}"

    def __len__(self):
        return len(self.samples)

    def _init_samples(self, video_paths):
        """Efficiently pre-computes all possible valid samples using a thread pool."""
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_video_path = {
                executor.submit(self._load_and_process_video_path, video_path): video_path for video_path in video_paths
            }
            for future in tqdm(as_completed(future_to_video_path), total=len(video_paths), desc="Initializing samples"):
                samples.extend(future.result())
        return sorted(samples, key=lambda x: (x["video_path"], x["frame_ids"][0]))

    def _load_and_process_video_path(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        n_frames = len(vr)

        samples = []
        for frame_i in range(0, n_frames, self.start_frame_interval):
            sample = dict()
            sample["video_path"] = video_path
            sample["frame_ids"] = []
            curr_frame_i = frame_i
            while True:
                if curr_frame_i > (n_frames - 1):
                    break
                sample["frame_ids"].append(curr_frame_i)
                if len(sample["frame_ids"]) == self.sequence_length:
                    break
                curr_frame_i += self.sequence_interval
            if len(sample["frame_ids"]) == self.sequence_length:
                samples.append(sample)
        return samples

    def _load_video(self, video_path, frame_ids):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS
            fps = 24
        return frame_data, fps

    def _load_control_data(self, video_name, camera_key, frame_ids):
        """Load control data for the video clip."""
        data_dict = {}
        for ctrl_type in self.ctrl_types:
            config = CTRL_TYPE_INFO[ctrl_type]
            if config.get("folder") is None:
                continue
            ctrl_path = os.path.join(self.dataset_dir, config["folder"], camera_key, f"{video_name}.{config['format']}")
            try:
                if ctrl_type == "seg":
                    with open(ctrl_path, "rb") as f:
                        ctrl_data = pickle.load(f)
                    data_dict["segmentation"] = ctrl_data
                elif ctrl_type == "keypoint":
                    with open(ctrl_path, "rb") as f:
                        ctrl_data = pickle.load(f)
                    data_dict["keypoint"] = ctrl_data
                elif ctrl_type == "depth":
                    vr = VideoReader(ctrl_path, ctx=cpu(0))
                    # Ensure the depth video has the same number of frames
                    assert len(vr) >= frame_ids[-1] + 1, f"Depth video {ctrl_path} has fewer frames than main video"

                    # Load the corresponding frames
                    depth_frames = vr.get_batch(frame_ids).asnumpy()  # [T,H,W,C]
                    depth_frames = torch.from_numpy(depth_frames).permute(3, 0, 1, 2)  # [C,T,H,W], same as rgb video
                    data_dict["depth"] = {
                        "video": depth_frames,
                        "frame_start": frame_ids[0],
                        "frame_end": frame_ids[-1],
                    }
                elif ctrl_type == "lidar":
                    vr = VideoReader(ctrl_path, ctx=cpu(0))
                    # Ensure the lidar depth video has the same number of frames
                    assert len(vr) >= frame_ids[-1] + 1, f"Lidar video {ctrl_path} has fewer frames than main video"
                    # Load the corresponding frames
                    lidar_frames = vr.get_batch(frame_ids).asnumpy()  # [T,H,W,C]
                    lidar_frames = torch.from_numpy(lidar_frames).permute(3, 0, 1, 2)  # [C,T,H,W], same as rgb video
                    data_dict["lidar"] = {
                        "video": lidar_frames,
                        "frame_start": frame_ids[0],
                        "frame_end": frame_ids[-1],
                    }
                elif ctrl_type == "hdmap_bbox":
                    vr = VideoReader(ctrl_path, ctx=cpu(0))
                    # Ensure the hdmap video has the same number of frames
                    assert len(vr) >= frame_ids[-1] + 1, f"Hdmap video {ctrl_path} has fewer frames than main video"
                    # Load the corresponding frames
                    hdmap_frames = vr.get_batch(frame_ids).asnumpy()  # [T,H,W,C]
                    hdmap_frames = torch.from_numpy(hdmap_frames).permute(3, 0, 1, 2)  # [C,T,H,W], same as rgb video
                    data_dict["hdmap_bbox"] = hdmap_frames

            except Exception as e:
                warnings.warn(f"Failed to load control data from {ctrl_path}: {str(e)}")
                return None

        return data_dict

    def __getitem__(self, index):
        try:
            sample = self.samples[index]
            base_video_path = sample["video_path"]
            frame_ids = sample["frame_ids"]
            video_name = os.path.basename(base_video_path).replace(".mp4", "")

            videos, control_inputs = [], []
            fps = 24  # Default FPS

            for camera_key in self.camera_keys:
                video_path = os.path.join(self.dataset_dir, "videos", camera_key, f"{video_name}.mp4")
                frames_np, fps = self._load_video(video_path, frame_ids)

                h, w = frames_np.shape[1], frames_np.shape[2]
                aspect_ratio = detect_aspect_ratio((self.W, self.H))

                frames_t = torch.from_numpy(frames_np.astype(np.uint8)).permute(0, 3, 1, 2)
                frames_t = self.preprocess(frames_t)
                frames_t = torch.clamp(frames_t * 255.0, 0, 255).to(torch.uint8)
                video = frames_t.permute(1, 0, 2, 3)  # C, T, H, W

                data_for_augmentor = {
                    "video": video,
                    "frame_start": frame_ids[0],
                    "frame_end": frame_ids[-1] + 1,
                    "frame_indices": frame_ids,
                    "aspect_ratio": aspect_ratio,
                    "fps": fps,
                }
                data_for_augmentor["video_name"] = {
                    "video_path": video_path,
                }

                caption_path = os.path.join(self.captions_dir, f"{video_name}.json")
                data_for_augmentor["ai_caption"] = AUTO_MV_DEFAULT_PROMPT
                if os.path.exists(caption_path):
                    with open(caption_path, "r") as f:
                        metadata = json.load(f)
                    if "caption" in metadata and len(metadata["caption"]) > 0:
                        data_for_augmentor["ai_caption"] = metadata["caption"]

                if self.ctrl_types:
                    ctrl_data = self._load_control_data(video_name, camera_key, frame_ids)
                    if ctrl_data is None:
                        raise ValueError(f"Failed to load control data for {video_name} view {camera_key}")
                    data_for_augmentor.update(ctrl_data)

                for _, aug_fn in self.augmentor.items():
                    augmented_data = aug_fn(data_for_augmentor)

                videos.append(augmented_data["video"])
                control_inputs.append(augmented_data[self.hint_key])

            # 5. Collate all views into a single sample dictionary
            final_data = dict()
            final_data["video"] = torch.cat(videos, dim=1)  # Stack along channel dim
            final_data[self.hint_key] = torch.cat(control_inputs, dim=1)  # Stack along channel dim
            if torch.isnan(final_data["video"]).any() or torch.isinf(final_data["video"]).any():
                log.critical("NaN or Inf found in input video data!")
            if torch.isnan(final_data[self.hint_key]).any() or torch.isinf(final_data[self.hint_key]).any():
                log.critical("NaN or Inf found in control_input_edge data!")

            _, _, h, w = final_data["video"].shape
            final_data["image_size"] = torch.tensor([self.H, self.W, self.H, self.W])
            final_data["fps"] = fps
            final_data["sample_n_views"] = len(self.camera_keys)
            final_data["num_video_frames_per_view"] = self.sequence_length

            view_indices = [self.camera_to_view_id[key] for key in self.camera_keys]
            final_data["view_indices"] = torch.tensor(view_indices).repeat_interleave(self.sequence_length).contiguous()
            final_data["latent_view_indices_B_T"] = (
                torch.tensor(view_indices).repeat_interleave(self.state_t).contiguous()
            )
            final_data["video_name"] = {
                "video_path": base_video_path,
            }
            final_data["aspect_ratio"] = data_for_augmentor["aspect_ratio"]
            final_data["ai_caption"] = data_for_augmentor["ai_caption"]
            final_data["padding_mask"] = torch.zeros(1, self.H, self.W)
            final_data["ref_cam_view_idx_sample_position"] = -1
            final_data["front_cam_view_idx_sample_position"] = torch.tensor([0])
            return final_data

        except Exception as e:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load video {self.video_paths[index]} (total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
                rank0_only=False,
            )
            return self[np.random.randint(len(self.video_paths))]


if __name__ == "__main__":
    camera_keys = [
        "ftheta_camera_front_wide_120fov",
        "ftheta_camera_cross_left_120fov",
        "ftheta_camera_cross_right_120fov",
        "ftheta_camera_rear_left_70fov",
        "ftheta_camera_rear_right_70fov",
        "ftheta_camera_rear_tele_30fov",
        "ftheta_camera_front_tele_30fov",
    ]
    camera_to_view_id = {
        "ftheta_camera_front_wide_120fov": 0,
        "ftheta_camera_cross_left_120fov": 1,
        "ftheta_camera_cross_right_120fov": 2,
        "ftheta_camera_rear_left_70fov": 3,
        "ftheta_camera_rear_right_70fov": 4,
        "ftheta_camera_rear_tele_30fov": 5,
        "ftheta_camera_front_tele_30fov": 6,
    }
    dataset = L(MultiviewTransferDataset)(
        dataset_dir="datasets/multiview_hdmap_posttrain_dataset",
        hint_key="control_input_hdmap_bbox",
        resolution="720",
        state_t=8,
        num_frames=29,
        sequence_interval=1,
        camera_keys=camera_keys,
        video_size=(704, 1280),
        front_camera_key="ftheta_camera_front_wide_120fov",
        camera_to_view_id=camera_to_view_id,
        front_view_caption_only=True,
        is_train=True,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=8, pin_memory=True, drop_last=True)
    data = next(iter(dataloader))
    print(
        (
            f"{data['video'].sum()=}\n"
            f"{data['video'].shape=}\n"
            f"{data['video_name']=}\n"
            f"{data['control_input_hdmap_bbox'].shape=}\n"
            f"{data['latent_view_indices_B_T'].shape=}\n"
            "---"
        )
    )
