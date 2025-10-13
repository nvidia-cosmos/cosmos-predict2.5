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

import ctypes
import gc
import io
import random
import traceback
from typing import Optional

import av
import numpy as np
import torch

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.augmentors.video_parsing import VideoParsing


class MultiViewVideoParsing(VideoParsing):
    """
    Merge multiple views of the same video into a single video.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.args = args
        self.driving_dataloader_config = args["driving_dataloader_config"]
        self.camera_to_view_id = self.driving_dataloader_config.camera_to_view_id
        self.view_id_to_caption_id = self.driving_dataloader_config.view_id_to_caption_id
        self.view_id_to_camera_key = {v: k for k, v in self.camera_to_view_id.items()}
        self.front_tele_and_front_cam_keys = self.driving_dataloader_config.front_tele_and_front_cam_keys
        assert len(self.front_tele_and_front_cam_keys) in [
            0,
            2,
        ], f"front_tele_and_front_cam_keys must be a tuple of length 0 or 2. Got {self.front_tele_and_front_cam_keys}"
        if len(self.front_tele_and_front_cam_keys) == 2:
            self.front_tele_cam_key = self.front_tele_and_front_cam_keys[0]
            self.front_cam_key = self.front_tele_and_front_cam_keys[1]
        else:
            self.front_tele_cam_key = None
            self.front_cam_key = None

        self.n_cameras = len(self.camera_to_view_id)
        self.sample_n_views = self.driving_dataloader_config.sample_n_views
        self.num_video_frames_per_view = self.driving_dataloader_config.num_video_frames_per_view
        self.num_video_frames_loaded_per_view = self.driving_dataloader_config.num_video_frames_loaded_per_view
        mult = self.num_video_frames_loaded_per_view - 1
        div = self.num_video_frames_per_view - 1
        assert mult % div == 0, (
            f"num_video_frames_loaded_per_view ({self.num_video_frames_loaded_per_view}) - 1 must be divisible by num_video_frames_per_view ({self.num_video_frames_per_view}) - 1. Got {mult} % {div} = {mult % div}"
        )
        self.temporal_subsampling_factor = mult // div
        self.ref_cam_view_idx = (
            self.driving_dataloader_config.ref_cam_view_idx
        )  # If > 0, then ref_cam_view_idx must be in the view_indices
        assert self.ref_cam_view_idx < 0 or self.ref_cam_view_idx < self.n_cameras, (
            f"ref_cam_view_idx must be less than the number of cameras ({self.n_cameras}) or -1. Got {self.ref_cam_view_idx}"
        )

        self.video_decode_num_threads = args["video_decode_num_threads"]
        self.min_fps = args["min_fps"]
        self.max_fps = args["max_fps"]

        # minimum start index. From alpamayo dataverse
        # NOTE V2 data might have issue in the first few frames, set a minimum to exclude them.
        self.minimum_start_index = args["minimum_start_index"]

        # In some samples, camera_front_tele_30fov has 605 frames while the other views have 610 frames.
        # By default, we clip the last view frames to the minimum number of frames per view.
        # If this parameter is set to True, we align the last view frames and clip the first view frames to the minimum number of frames per view.
        self.align_last_view_frames_and_clip_from_front = args["align_last_view_frames_and_clip_from_front"]
        self.H = self.driving_dataloader_config.H
        self.W = self.driving_dataloader_config.W

        self.batch_counter = 0
        self.gc_every_n = 100

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
            if fps < self.min_fps:
                log.warning(f"Video FPS {fps} is less than min_fps {self.min_fps}", rank0_only=False)
                return None
            if fps > self.max_fps:
                log.warning(f"Video FPS {fps} is greater than max_fps {self.max_fps}", rank0_only=False)
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
        log.debug(f"MultiViewVideoParsing: Temporal subsampling factor: {self.temporal_subsampling_factor}")
        with av.open(video_buffer, options={"threads": str(self.video_decode_num_threads)}) as container:
            stream = container.streams.video[0]
            frame_pos = 0
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx < start_idx:
                    continue
                if (frame_idx - start_idx) % self.temporal_subsampling_factor != 0:
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

    def __call__(self, data_dict: dict) -> Optional[dict]:
        try:
            if self.batch_counter % self.gc_every_n == 0:
                gc.collect()
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            self.batch_counter = (self.batch_counter + 1) % (self.gc_every_n + 1)
            sample_key_with_rand_idx = data_dict["__key__"]
            sample_key = "-".join(sample_key_with_rand_idx.split("-")[:-1])

            view_indices_selection = data_dict["selection_data.json"]["view_indices_selection"]
            camera_keys_selection = data_dict["selection_data.json"]["camera_keys_selection"]
            assert self.ref_cam_view_idx == -1 or self.ref_cam_view_idx in view_indices_selection, (
                f"ref_cam_view_idx must be -1 or in view_indices_selection. Got {self.ref_cam_view_idx=} and {view_indices_selection=}"
            )

            if "caption.json" not in data_dict or sample_key not in data_dict["caption.json"]:
                log.warning(f"{sample_key} not found in caption JSON. Skipping.", rank0_only=False)
                return None

            sample_dict = data_dict["caption.json"][sample_key]

            if self.driving_dataloader_config.single_caption_only:
                view_id = self.camera_to_view_id[self.front_cam_key]
                view_caption_id = str(self.view_id_to_caption_id[view_id])
                if view_caption_id not in sample_dict:
                    log.warning(
                        f"{self.front_cam_key} [key='{view_caption_id}'] not found for {sample_key}. Skipping.",
                        rank0_only=False,
                    )
                    return None

            # Get the number of caption windows for each view
            caption_pos_id, start_pos_id, end_pos_id = 1, 2, 3
            num_caption_windows_per_view = dict()
            captions_per_view, starts_per_view, ends_per_view = dict(), dict(), dict()
            for cam_key in camera_keys_selection:
                if cam_key == self.front_tele_cam_key:
                    continue
                view_id = self.camera_to_view_id[cam_key]
                view_caption_id = str(self.view_id_to_caption_id[view_id])
                if view_caption_id not in sample_dict:
                    continue

                view_sample = sample_dict[view_caption_id]
                captions_per_view[cam_key] = view_sample[caption_pos_id]
                starts_per_view[cam_key] = view_sample[start_pos_id]
                ends_per_view[cam_key] = view_sample[end_pos_id]

                num_captions = len(captions_per_view[cam_key])
                num_starts = len(starts_per_view[cam_key])
                num_ends = len(ends_per_view[cam_key])
                if num_captions != num_starts or num_captions != num_ends:
                    log.error(
                        f"Inconsistent number of caption windows, starts, and ends for {cam_key} in {sample_key}. Skipping.",
                        rank0_only=False,
                    )
                    return None
                num_caption_windows_per_view[cam_key] = num_captions

            if self.driving_dataloader_config.single_caption_only:
                num_caption_windows = num_caption_windows_per_view.get(self.front_cam_key, 1)
            else:
                num_caption_windows = min(num_caption_windows_per_view.values())
            selected_window_id = random.randint(0, num_caption_windows - 1)

            # extract t5 embeddings, captions and frames
            all_view_t5_embeddings = {}
            all_view_captions = {}
            all_view_frame_ranges = {}
            all_view_video_fps = []
            n_orig_video_frames_per_view = []
            for i, cam_key in enumerate(camera_keys_selection):
                if cam_key == self.front_tele_cam_key:
                    continue
                if f"{cam_key}.bin" not in data_dict:
                    log.debug(
                        f"t5 embedding not found for {data_dict['__key__']} camera {cam_key}. Setting to zero tensor."
                    )
                    all_view_t5_embeddings[cam_key] = torch.zeros(512, 1024)
                else:
                    all_view_t5_embeddings[cam_key] = torch.from_numpy(data_dict[f"{cam_key}.bin"]).squeeze(0)

                empty_caption = ""
                if self.driving_dataloader_config.single_caption_only and cam_key != self.front_cam_key:
                    if (self.front_cam_key not in starts_per_view) or (self.front_cam_key not in ends_per_view):
                        available_keys = list(starts_per_view.keys()) + list(ends_per_view.keys())
                        log.warning(
                            f"front_cam_key {self.front_cam_key} not found in `starts_per_view` or `ends_per_view`. Only found {available_keys}.",
                            rank0_only=False,
                        )
                        return None
                    log.debug(
                        f"`single_caption_only=True` so setting {cam_key} in {sample_key} to caption '{empty_caption}'"
                    )
                    all_view_captions[cam_key] = empty_caption
                    all_view_frame_ranges[cam_key] = (
                        starts_per_view[self.front_cam_key][selected_window_id],
                        ends_per_view[self.front_cam_key][selected_window_id],
                    )
                    continue

                if cam_key not in captions_per_view:
                    log.debug(f"{cam_key} not found. Setting to '{empty_caption}'.", rank0_only=False)
                    all_view_captions[cam_key] = empty_caption
                    all_view_frame_ranges[cam_key] = (
                        starts_per_view[self.front_cam_key][selected_window_id],
                        ends_per_view[self.front_cam_key][selected_window_id],
                    )
                    continue

                maybe_dict = captions_per_view[cam_key][selected_window_id]
                if isinstance(maybe_dict, str):
                    caption = maybe_dict
                elif isinstance(maybe_dict, dict):
                    sampling_probs = {
                        "qwen2p5_7b_caption": 0.7,
                        "qwen2p5_7b_caption_medium": 0.2,
                        "qwen2p5_7b_caption_short": 0.1,
                    }
                    choices = list(maybe_dict.keys())
                    p = np.array([sampling_probs[k] for k in choices])
                    p = p / p.sum()
                    selected_key = np.random.choice(choices, p=p)
                    caption = maybe_dict[selected_key]
                else:
                    raise TypeError(f"Caption must be a string or a dict. Got {type(maybe_dict)} for {maybe_dict}")
                all_view_captions[cam_key] = caption
                all_view_frame_ranges[cam_key] = (
                    starts_per_view[cam_key][selected_window_id],
                    ends_per_view[cam_key][selected_window_id],
                )

            if self.front_tele_cam_key is not None and self.front_tele_cam_key in camera_keys_selection:
                if self.front_cam_key not in camera_keys_selection:
                    log.warning(
                        f"front_cam_key {self.front_cam_key} not found in camera_keys_selection. Skipping.",
                        rank0_only=False,
                    )
                    return None
                else:
                    all_view_captions[self.front_tele_cam_key] = str(all_view_captions[self.front_cam_key])
                    all_view_t5_embeddings[self.front_tele_cam_key] = all_view_t5_embeddings[self.front_cam_key].clone()

            for cam_key in camera_keys_selection:
                output = self.probe_video_length(data_dict[f"{cam_key}.mp4"])
                if output is None:
                    log.warning(f"Video {cam_key} not found or invalid. Skipping.", rank0_only=False)
                    return None
                n_orig_video_frames, video_fps = output
                n_orig_video_frames_per_view.append(n_orig_video_frames)
                all_view_video_fps.append(video_fps)

            fps = set(all_view_video_fps)
            assert len(fps) == 1, f"All view video fps must be the same. Got {fps}"
            fps = fps.pop()

            frame_range = set(all_view_frame_ranges.values())
            # (temporary hack to mix up the v1 and v2 caption ranges when _not_ using single_caption_only)
            if frame_range == {(0, 150), (2, 130)} and not self.driving_dataloader_config.single_caption_only:
                frame_range = {(2, 130)}
            assert len(frame_range) == 1, f"All frame ranges must be the same. Got {frame_range}"
            frame_range = frame_range.pop()

            # Get the absolute frame indices of the selected frames for each view
            min_num_frames_per_view = min(n_orig_video_frames_per_view)
            if min_num_frames_per_view < self.num_video_frames_per_view:
                log.warning(
                    f"This sample has {min_num_frames_per_view} frames in the shortest view which is less than target {self.num_video_frames_per_view=}",
                    rank0_only=False,
                )
                return None

            relative_start_index = max(frame_range[0], self.minimum_start_index)
            all_view_selected_frame_ranges = []
            for i, cam_key in enumerate(camera_keys_selection):
                view_orig_n_frames = n_orig_video_frames_per_view[i]
                view_offset = relative_start_index
                diff = view_orig_n_frames - min_num_frames_per_view
                if diff > 0 and self.align_last_view_frames_and_clip_from_front:
                    view_offset += diff
                all_view_selected_frame_ranges.append(
                    (
                        view_offset,
                        view_offset + self.num_video_frames_loaded_per_view,
                    )
                )
            log.debug(f"Selected view frame ranges: {all_view_selected_frame_ranges}")

            C, T, H, W = 3, self.num_video_frames_per_view * len(camera_keys_selection), self.H, self.W
            all_view_video_frames_sampled = torch.empty((C, T, H, W), dtype=torch.uint8)

            for idx, cam_key in enumerate(camera_keys_selection):
                start_idx, end_idx = all_view_selected_frame_ranges[idx]
                view_video_frames = self._decode_view(
                    cam_key,
                    data_dict[f"{cam_key}.mp4"],
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

            view_indices = torch.tensor(view_indices_selection).repeat_interleave(self.num_video_frames_per_view)
            data_dict["view_indices"] = view_indices
            data_dict["sample_n_views"] = self.sample_n_views
            if self.ref_cam_view_idx >= 0:
                # Position of the reference camera in the sample_n_views views selection. This is useful for conditioning.
                data_dict["ref_cam_view_idx_sample_position"] = view_indices_selection.index(self.ref_cam_view_idx)
            else:
                data_dict["ref_cam_view_idx_sample_position"] = -1
            data_dict["front_cam_view_idx_sample_position"] = view_indices_selection.index(
                self.camera_to_view_id[self.front_cam_key]
            )
            data_dict["video"] = {
                "video": all_view_video_frames_sampled,
                "fps": fps,
                "camera_keys_selection": camera_keys_selection,
                "n_orig_video_frames_per_view": n_orig_video_frames_per_view,
                "aspect_ratio": "16,9",
                "view_indices_selection": view_indices_selection,
                "view_indices": view_indices,
                "sample_n_views": self.sample_n_views,
                "view_t5_embeddings": all_view_t5_embeddings,
                "view_captions": all_view_captions,
                "num_video_frames_per_view": self.num_video_frames_per_view,
                "control_weight": 1.0,
                "all_view_selected_frame_ranges": all_view_selected_frame_ranges,
            }

            keys_to_delete = [k for k in list(data_dict.keys()) if "." in k]
            for key in keys_to_delete:
                del data_dict[key]

            return data_dict
        except Exception:
            log.error(traceback.format_exc(), rank0_only=False)
            log.error(f"Sample keys: {data_dict.keys()}", rank0_only=False)
            log.error(f"Sample key: {sample_key}", rank0_only=False)
            if "__url__" in data_dict:
                log.error(f"__url__: {data_dict['__url__']}", rank0_only=False)
            if "__key__" in data_dict:
                log.error(f"__key__: {data_dict['__key__']}", rank0_only=False)
            return None
