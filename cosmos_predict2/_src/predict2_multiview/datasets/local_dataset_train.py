# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import glob
import io
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterable, List, Optional

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2_multiview.datasets.augmentor_provider import get_video_augmentor_v2_multiview_local
from cosmos_predict2._src.predict2_multiview.datasets.data_sources.data_registration import (
    _get_contiguous_view_indices_options,
    _get_noncontiguous_view_indices_options,
)


@dataclass
class LocalDrivingVideoDataloaderConfig:
    resolution: str
    sample_n_views: int
    num_video_frames_per_view: int
    num_video_frames_loaded_per_view: int
    sample_noncontiguous_views: bool
    ref_cam_view_idx: int
    overfit_firstn: int
    camera_to_view_id: dict
    camera_to_caption_prefix: dict
    no_view_prefix: bool
    single_caption_only: bool
    H: int
    W: int
    front_tele_cam_key: Optional[str] = None
    front_cam_key: Optional[str] = None
    min_fps: int = 10
    max_fps: int = 60
    caption_tag_ratios: Optional[dict[str, float]] = None
    align_last_view_frames_and_clip_from_front: bool = False
    minimum_start_index: int = 3
    video_decode_num_threads: int = 8


def create_selection_data(driving_dataloader_config: LocalDrivingVideoDataloaderConfig):
    n_cameras = len(driving_dataloader_config.camera_to_view_id)
    view_id_to_camera_key = {v: k for k, v in driving_dataloader_config.camera_to_view_id.items()}
    if driving_dataloader_config.sample_noncontiguous_views:
        view_indices_options = _get_noncontiguous_view_indices_options(
            driving_dataloader_config.sample_n_views, driving_dataloader_config.ref_cam_view_idx, n_cameras
        )
    else:
        view_indices_options = _get_contiguous_view_indices_options(
            driving_dataloader_config.sample_n_views, driving_dataloader_config.ref_cam_view_idx, n_cameras
        )

    view_indices_selection = random.choice(view_indices_options)
    # in the case that view_ids are not contiguous, we use the indices selected and map them to view_ids based on their sorted order
    view_id_sorted = {i: view_id for i, view_id in enumerate(sorted(list(view_id_to_camera_key.keys())))}
    camera_keys_selection = [view_id_to_camera_key[view_id_sorted[view_idx]] for view_idx in view_indices_selection]
    return view_indices_selection, camera_keys_selection


def wrap_augmentor_func_as_generator(func: Callable, data: Iterable):
    """Wrap a regular augmentor function as a generator."""
    for data_dict in data:
        data_dict_out = func(data_dict)
        if data_dict_out is None:
            # Skip "unhealthy" samples
            continue
        yield data_dict_out


def augmentor_fn(data, augmentations):
    """
    Apply augmentors in sequence, handling both generator and regular functions.
    This matches the webdataset augmentor_fn behavior exactly.
    """
    # Build augmentor chain
    for aug_fn in augmentations:
        # Use generator function as augmentor
        # (recommended, allows skipping or replicating samples inside the augmentor)
        if getattr(aug_fn, "is_generator", False):
            data = aug_fn(data)
        else:  # Use regular function as augmentor (backward compatibility)
            data = wrap_augmentor_func_as_generator(aug_fn, data)
    yield from data


class LocalMultiviewPredictDataset(Dataset):
    def __init__(
        self,
        video_file_dirs: List[str],
        driving_dataloader_config: LocalDrivingVideoDataloaderConfig,
        shuffle: bool = True,
        gc_every_n: int = 100,
    ):
        self.driving_dataloader_config = driving_dataloader_config

        # Set frequently used variables from driving_dataloader_config
        self.front_tele_cam_key = self.driving_dataloader_config.front_tele_cam_key
        self.front_cam_key = self.driving_dataloader_config.front_cam_key
        self.num_video_frames_loaded_per_view = self.driving_dataloader_config.num_video_frames_loaded_per_view
        self.num_video_frames_per_view = self.driving_dataloader_config.num_video_frames_per_view

        # Create selection data
        self.view_indices_selection, self.camera_keys_selection = create_selection_data(self.driving_dataloader_config)

        # Initialize variables for garbage collection
        self.batch_counter = 0
        self.gc_every_n = gc_every_n

        # Build augmentors
        augmentor_cfg = get_video_augmentor_v2_multiview_local(
            resolution=self.driving_dataloader_config.resolution,
        )
        self.augmentor_fn = self.build_data_augmentor(augmentor_cfg)

        # Build sample dirs
        self.sample_dirs = self.build_sample_dirs(video_file_dirs)

        # Shuffle dataset
        if shuffle:
            random.shuffle(self.sample_dirs)

    def validate_config(self):
        # Validate num_video_frames_loaded_per_view and num_video_frames_per_view
        mult = self.num_video_frames_loaded_per_view - 1
        div = self.num_video_frames_per_view - 1
        assert mult % div == 0, (
            f"num_video_frames_loaded_per_view ({self.num_video_frames_loaded_per_view}) - 1 must be divisible by num_video_frames_per_view ({self.num_video_frames_per_view}) - 1. Got {mult} % {div} = {mult % div}"
        )

        # Validate front_tele_cam_key and front_cam_key
        if self.front_tele_cam_key is not None and self.front_tele_cam_key in self.camera_keys_selection:
            if self.front_cam_key not in self.camera_keys_selection:
                raise ValueError(
                    f"front_cam_key {self.front_cam_key} not found in camera_keys_selection. Skipping.",
                )

    def build_sample_dirs(self, video_file_dirs: List[str]) -> List[str]:
        sample_dirs = []
        for video_file_dir in video_file_dirs:
            for sample_dir in glob.glob(os.path.join(video_file_dir, "**")):
                if os.path.isdir(sample_dir):
                    sample_dirs.append(sample_dir)
        return sample_dirs

    def build_data_augmentor(self, augmentor_cfg: Dict[str, Any]) -> Callable:
        """
        Build data augmentors from config, similar to webdataset.

        Args:
            augmentor_cfg: Augmentor configuration

        Returns:
            Partial function that applies all augmentors
        """
        augmentations = []
        if augmentor_cfg:
            from cosmos_predict2._src.imaginaire.lazy_config import instantiate

            for aug in augmentor_cfg:
                augmentations.append(instantiate(augmentor_cfg[aug]))

        # This is the function that calls each augmentor in sequence.
        return partial(augmentor_fn, augmentations=augmentations)

    def select_captions_with_ratios(self, view_captions: pd.DataFrame) -> str:
        def get_prob(tag):
            if self.driving_dataloader_config.caption_tag_ratios:
                return self.driving_dataloader_config.caption_tag_ratios[tag]
            else:
                return 1 / len(choices)

        choices = list(view_captions.tag.unique())
        p = np.array([get_prob(k) for k in choices])
        p = p / p.sum()
        selected_key = np.random.choice(choices, p=p)
        caption = view_captions[view_captions.tag == selected_key].iloc[0]["caption"]
        return caption

    def convert_captions_to_string(self, all_view_captions: dict[str, str]) -> str:
        view_captions_list = []
        for cam_key in self.camera_keys_selection:
            view_caption_prefix = (
                ""
                if self.driving_dataloader_config.no_view_prefix
                else self.driving_dataloader_config.camera_to_caption_prefix[cam_key]
            )
            if cam_key == self.front_cam_key or not self.driving_dataloader_config.single_caption_only:
                view_captions_list.append(f"{view_caption_prefix} {all_view_captions[cam_key]}".strip())
        return " -- ".join(view_captions_list)

    def load_captions(self, captions_path: str) -> str:
        with open(captions_path, "r") as f:
            caption_df = pd.read_json(f, lines=True, orient="records")

        if self.driving_dataloader_config.single_caption_only:
            if len(caption_df[caption_df.view == self.front_cam_key]) == 0:
                raise ValueError(f"{self.front_cam_key} not found.")

        # Get captions per view
        empty_caption = ""
        all_view_captions = {}
        for cam_key in self.camera_keys_selection:
            if cam_key == self.front_tele_cam_key:
                continue
            if self.driving_dataloader_config.single_caption_only and cam_key != self.front_cam_key:
                log.debug(f"`single_caption_only=True` so setting {cam_key} to caption '{empty_caption}'")
                all_view_captions[cam_key] = empty_caption
                continue
            view_captions = caption_df[caption_df.view == cam_key]
            if len(view_captions) == 1:
                all_view_captions[cam_key] = view_captions.iloc[0]["caption"]
            elif len(view_captions) == 1:
                all_view_captions[cam_key] = self.select_captions_with_ratios(view_captions)
            else:
                log.debug(f"{cam_key} not found. Setting to '{empty_caption}'.", rank0_only=False)
                all_view_captions[cam_key] = empty_caption

        # Updating front tele cam key caption if exists
        if self.front_tele_cam_key is not None and self.front_tele_cam_key in self.camera_keys_selection:
            all_view_captions[self.front_tele_cam_key] = all_view_captions[self.front_cam_key]

        return self.convert_captions_to_string(all_view_captions)

    def probe_video_length(self, video_bytes: bytes) -> tuple[int, float] | None:
        """
        Return (num_frames, fps) for a video contained in `video_bytes`.
        This opens the container, looks at the stream header, and closes it
        — no frame decoding is performed.

        Args:
            video_bytes: Binary contents of an .mp4/.mkv/etc. file.

        Returns:
            n_frames (int): total number of frames in the stream.
            fps      (float): average frame rate reported by the container.
        """
        with av.open(io.BytesIO(video_bytes)) as container:
            stream = container.streams.video[0]
            n_frames = stream.frames  # populated from header
            fps = float(stream.average_rate) if stream.average_rate is not None else None
            if not n_frames or n_frames <= 0:  # extremely rare fallback
                n_frames = int(stream.duration * stream.average_rate) if stream.average_rate is not None else None
            if fps < self.driving_dataloader_config.min_fps:
                log.warning(
                    f"Video FPS {fps} is less than min_fps {self.driving_dataloader_config.min_fps}", rank0_only=False
                )
                return None
            if fps > self.driving_dataloader_config.max_fps:
                log.warning(
                    f"Video FPS {fps} is greater than max_fps {self.driving_dataloader_config.max_fps}",
                    rank0_only=False,
                )
                return None
            if fps is None or n_frames is None:
                return None
            return n_frames, fps

    def _decode_view(
        self,
        cam_key: str,
        video_bytes: bytes,
        start_idx: int,
        end_idx: int,
        H_out: int,
        W_out: int,
    ) -> torch.Tensor:
        n_frames = end_idx - start_idx
        if n_frames <= 0:
            raise ValueError("end_idx must be > start_idx")

        video_buffer = io.BytesIO(video_bytes)
        frames_np = np.empty((self.num_video_frames_per_view, H_out, W_out, 3), dtype=np.uint8)

        mult = self.num_video_frames_loaded_per_view - 1
        div = self.num_video_frames_per_view - 1
        temporal_subsampling_factor = mult // div

        log.debug(f"MultiViewVideoParsing: Temporal subsampling factor: {temporal_subsampling_factor}")
        with av.open(
            video_buffer, options={"threads": str(self.driving_dataloader_config.video_decode_num_threads)}
        ) as container:
            stream = container.streams.video[0]
            frame_pos = 0
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx < start_idx:
                    continue
                if (frame_idx - start_idx) % temporal_subsampling_factor != 0:
                    continue
                if frame_idx >= end_idx:
                    break

                # Resize + Color convert in C (libswscale)
                frame_resized = frame.reformat(width=W_out, height=H_out, format="rgb24")
                # Directly write the C-contiguous ndarray into pre-allocated buffer.
                frames_np[frame_pos] = frame_resized.to_ndarray()
                frame_pos += 1

        if frame_pos != self.num_video_frames_per_view:
            raise RuntimeError(f"Decoded {frame_pos} frames for {cam_key}, expected {self.num_video_frames_per_view} ")

        # Convert once to torch (shares memory) and permute to (C, T, H, W).
        view_video_frames = torch.from_numpy(frames_np).permute(3, 0, 1, 2)
        return view_video_frames

    def load_videos(self, sample_dir: str):
        videos = {}
        for filename in os.listdir(sample_dir):
            if filename.endswith(".mp4"):
                with open(os.path.join(sample_dir, filename), "rb") as f:
                    videos[filename] = f.read()

        # Get video frames and fps per view
        n_orig_video_frames_per_view = []
        all_view_video_fps = []
        for cam_key in self.camera_keys_selection:
            output = self.probe_video_length(videos[f"{cam_key}.mp4"])
            if output is None:
                raise ValueError(f"Video {cam_key} not found or invalid.")
            n_orig_video_frames, video_fps = output
            n_orig_video_frames_per_view.append(n_orig_video_frames)
            all_view_video_fps.append(video_fps)

        # Validate video fps
        fps = set(all_view_video_fps)
        assert len(fps) == 1, f"All view video fps must be the same. Got {fps}"
        fps = fps.pop()

        frame_range = (0, 150)

        # Validate minimum number of frames per view
        min_num_frames_per_view = min(n_orig_video_frames_per_view)
        if min_num_frames_per_view < self.num_video_frames_per_view:
            raise ValueError(
                f"This sample has {min_num_frames_per_view} frames in the shortest view which is less than target {self.num_video_frames_per_view=}",
            )

        # Select frame ranges
        relative_start_index = max(frame_range[0], self.driving_dataloader_config.minimum_start_index)
        all_view_selected_frame_ranges = []
        for i, cam_key in enumerate(self.camera_keys_selection):
            view_orig_n_frames = n_orig_video_frames_per_view[i]
            view_offset = relative_start_index
            diff = view_orig_n_frames - min_num_frames_per_view
            if diff > 0 and self.driving_dataloader_config.align_last_view_frames_and_clip_from_front:
                view_offset += diff
            all_view_selected_frame_ranges.append(
                (
                    view_offset,
                    view_offset + self.num_video_frames_loaded_per_view,
                )
            )
        log.debug(f"Selected view frame ranges: {all_view_selected_frame_ranges}")

        # Decode video frames
        C, T, H, W = (
            3,
            self.num_video_frames_per_view * len(self.camera_keys_selection),
            self.driving_dataloader_config.H,
            self.driving_dataloader_config.W,
        )
        all_view_video_frames_sampled = torch.empty((C, T, H, W), dtype=torch.uint8)

        for idx, cam_key in enumerate(self.camera_keys_selection):
            start_idx, end_idx = all_view_selected_frame_ranges[idx]
            view_video_frames = self._decode_view(
                cam_key,
                videos[f"{cam_key}.mp4"],
                start_idx,
                end_idx,
                H,
                W,
            )
            v_start_idx = idx * self.num_video_frames_per_view
            v_end_idx = v_start_idx + self.num_video_frames_per_view
            all_view_video_frames_sampled[:, v_start_idx:v_end_idx] = view_video_frames
            log.debug(
                f"Placed view {cam_key} into tensor slice [{v_start_idx}:{v_end_idx}] with shape {view_video_frames.shape}"
            )
            del view_video_frames

        return all_view_video_frames_sampled, fps, n_orig_video_frames_per_view

    def load_data(self, sample_dir: str) -> Dict[str, Any]:
        try:
            sample_id = sample_dir.split("/")[-1]

            video, fps, n_orig_video_frames_per_view = self.load_videos(sample_dir)

            if self.driving_dataloader_config.ref_cam_view_idx >= 0:
                # Position of the reference camera in the sample_n_views views selection. This is useful for conditioning.
                ref_cam_view_idx_sample_position = self.view_indices_selection.index(
                    self.driving_dataloader_config.ref_cam_view_idx
                )
            else:
                ref_cam_view_idx_sample_position = -1

            data_dict: Dict[str, Any] = {
                "__key__": sample_id,
                "ai_caption": self.load_captions(os.path.join(sample_dir, "caption.jsonl")),
                "camera_keys_selection": self.camera_keys_selection,
                "view_indices_selection": self.view_indices_selection,
                "aspect_ratio": "16,9",
                "num_video_frames_per_view": self.num_video_frames_per_view,
                "control_weight": 1.0,
                "view_indices": torch.tensor(self.view_indices_selection).repeat_interleave(
                    self.num_video_frames_per_view
                ),
                "sample_n_views": self.driving_dataloader_config.sample_n_views,
                "front_cam_view_idx_sample_position": self.view_indices_selection.index(
                    self.driving_dataloader_config.camera_to_view_id[self.front_cam_key]
                ),
                "video": video,
                "fps": fps,
                "n_orig_video_frames_per_view": n_orig_video_frames_per_view,
                "ref_cam_view_idx_sample_position": ref_cam_view_idx_sample_position,
            }

            return data_dict
        except Exception as e:
            log.error(f"Failed to load data for sample {sample_id}")
            raise e

    def gc(self):
        if self.batch_counter % self.gc_every_n == 0:
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        self.batch_counter = (self.batch_counter + 1) % (self.gc_every_n + 1)

    def __getitem__(self, idx):
        self.gc()
        sample_dir = self.sample_dirs[idx]
        try:
            data_dict = self.load_data(sample_dir)

            # Apply augmentors
            if self.augmentor_fn:
                # augmentor_fn is a generator, consume and return first sample
                for sample in self.augmentor_fn([data_dict]):
                    return sample
            else:
                return data_dict

        except Exception as e:
            log.error(f"Failed to load files from {sample_dir}")
            raise e

    def __len__(self):
        return len(self.sample_dirs)
