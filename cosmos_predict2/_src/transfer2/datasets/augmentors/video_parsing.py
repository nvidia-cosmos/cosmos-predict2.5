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

import io
import random
from typing import Optional

import decord
import numpy as np
import torch

from cosmos_predict2._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_predict2._src.transfer2.inference.utils import detect_aspect_ratio


class VideoParsing(Augmentor):
    """
    This augmentor is used to parse the video bytes and get the video frames.
    the return dict is back-compatible with old datasets, which video decoding happens in the decoder stage.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert len(input_keys) == 2, "VideoParsing augmentor only supports two input keys"
        self.meta_key = input_keys[0]
        self.video_key = input_keys[1]

        self.key_for_caption = args["key_for_caption"]
        assert self.key_for_caption in [
            "t2w_windows",
            "i2w_windows_later_frames",
        ], "key_for_caption must be either t2w_windows or i2w_windows_later_frames"
        self.min_duration = args["min_duration"]
        self.min_fps = args["min_fps"]
        self.max_fps = args["max_fps"]
        self.num_frames = args["num_video_frames"]
        self.use_native_fps = args["use_native_fps"]
        if self.use_native_fps:
            assert self.num_frames > 0, "num_frames must be greater than 0 when use_native_fps is True"
        self.video_decode_num_threads = args["video_decode_num_threads"]
        self.resolution = args.get("resolution", "720")

    def __call__(self, data_dict: dict) -> dict:
        try:
            meta_dict = data_dict[self.meta_key]
            video = data_dict[self.video_key]
        except Exception as e:
            log.warning(
                f"Cannot find video. url: {data_dict['__url__']}, key: {data_dict['__key__']}", rank0_only=False
            )
            return None

        if not isinstance(video, bytes):
            return data_dict

        video_info = {
            "fps": meta_dict["framerate"],
            "n_orig_video_frames": meta_dict["nb_frames"],
        }

        if video_info["fps"] < self.min_fps:
            # log.warning(f"Video FPS {video_info['fps']} is less than min_fps {self.min_fps}", rank0_only=False)
            return None
        if video_info["fps"] > self.max_fps:
            log.warning(f"Video FPS {video_info['fps']} is greater than max_fps {self.max_fps}", rank0_only=False)
            return None

        options: list = list((i, item) for i, item in enumerate(meta_dict[self.key_for_caption]))

        if "video" in data_dict and type(data_dict["video"]) == dict and "chunk_index" in data_dict["video"]:
            chunk_index = data_dict["video"]["chunk_index"]
            # log.info(f"Using chunk_index {chunk_index} for depth parsing")
            options = [options[chunk_index]]

        # Skip the last window if possible.
        # All windows except the last are 5 seconds long. The last window has a duration in the range [2.5s, 7.5), which is less preferred.
        if len(options) > 1:
            options = options[:-1]

        # shuffle options
        random.shuffle(options)
        video_frames = None
        for chunk_index, option in options:
            start_frame = option["start_frame"]
            end_frame = option["end_frame"]
            if (end_frame - start_frame) < self.min_duration * video_info["fps"]:
                continue

            if self.use_native_fps:
                if (end_frame - start_frame) < self.num_frames:
                    continue

            # video_buffer = io.BytesIO(video)
            # video_reader = decord.VideoReader(video_buffer, num_threads=self.video_decode_num_threads)
            video_buffer = io.BytesIO(video)
            video_reader = decord.VideoReader(video_buffer)
            orig_h, orig_w, _ = video_reader[0].shape  # Get original dimensions
            aspect_ratio = detect_aspect_ratio((orig_w, orig_h))
            new_w, new_h = VIDEO_RES_SIZE_INFO[self.resolution][aspect_ratio]
            scaling_ratio = min((new_w / orig_w), (new_h / orig_h))
            new_w, new_h = int(scaling_ratio * orig_w + 0.5), int(scaling_ratio * orig_h + 0.5)
            # Reload video with resized dimensions
            video_buffer = io.BytesIO(video)
            video_reader = decord.VideoReader(
                video_buffer, num_threads=self.video_decode_num_threads, width=new_w, height=new_h
            )

            if self.use_native_fps:
                if "alpamayo" in data_dict["__url__"].root:
                    start_frame += 5
                if (end_frame - start_frame) < self.num_frames:
                    continue

                # take mid self.num_frames frames from start frame to end frame.
                total_frames = end_frame - start_frame
                # always try lower fps if possible.
                num_multiplier = total_frames // self.num_frames
                expected_length = self.num_frames * num_multiplier
                _start_frame = start_frame + (total_frames - expected_length) // 2
                _end_frame = _start_frame + expected_length
                frame_indices = np.arange(_start_frame, _end_frame, num_multiplier).tolist()
                assert len(frame_indices) == self.num_frames, "frame_indices length is not equal to num_frames"
                try:
                    video_frames = video_reader.get_batch(frame_indices).asnumpy()
                except Exception as e:
                    log.warning(
                        f"Video is not long enough, return None. url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                        rank0_only=False,
                    )
                    return None
                video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
                video_reader.seek(0)  # set video reader point back to 0 to clean up cache
                del video_reader  # delete the reader to avoid memory leak
                break

            else:
                frame_indices = np.arange(start_frame, end_frame).tolist()

                # online hot-fix for alpamayo data. Skip the first 5 frames as there is chance that the first five frames contain black frames.
                if "alpamayo" in data_dict["__url__"].root:
                    assert len(frame_indices) >= 5, (
                        "Getting less than 5 frames for alpamayo videos. There is no way to skip the first five frames."
                    )
                    frame_indices = frame_indices[5:]
                    start_frame += 5
                video_frames = video_reader.get_batch(frame_indices).asnumpy()

                video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

                # Clean up
                video_reader.seek(0)  # set video reader point back to 0 to clean up cache
                del video_reader  # delete the reader to avoid memory leak
                break

        if video_frames is None:
            log.warning(
                f"No valid video frames found, return None. url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                rank0_only=False,
            )
            return None

        video_info["chunk_index"] = chunk_index
        video_info["frame_start"] = start_frame
        video_info["frame_end"] = end_frame
        video_info["num_frames"] = end_frame - start_frame
        video_info["frame_indices"] = frame_indices
        video_info["video"] = video_frames

        # update data_dict, make it back-compatible with old datasets, which video decoding happens in the decoder stage.
        data_dict[self.video_key] = video_info

        return data_dict


class VideoParsingWithImageContext(Augmentor):
    """
    This augmentor is used to parse the video bytes and get the video frames.
    the return dict is back-compatible with old datasets, which video decoding happens in the decoder stage.
    Also extracts an additional frame from a different chunk as image context.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert len(input_keys) == 2, "VideoParsing augmentor only supports two input keys"
        self.meta_key = input_keys[0]
        self.video_key = input_keys[1]

        self.key_for_caption = args["key_for_caption"]
        assert self.key_for_caption in [
            "t2w_windows",
            "i2w_windows_later_frames",
        ], "key_for_caption must be either t2w_windows or i2w_windows_later_frames"
        self.min_duration = args["min_duration"]
        self.min_fps = args["min_fps"]
        self.max_fps = args["max_fps"]
        self.num_frames = args["num_video_frames"]
        self.use_native_fps = args["use_native_fps"]
        if self.use_native_fps:
            assert self.num_frames > 0, "num_frames must be greater than 0 when use_native_fps is True"
        self.video_decode_num_threads = args["video_decode_num_threads"]
        self.resolution = args.get("resolution", "720")

    def __call__(self, data_dict: dict) -> dict:
        try:
            meta_dict = data_dict[self.meta_key]
            video = data_dict[self.video_key]
        except Exception as e:
            log.warning(
                f"Cannot find video. url: {data_dict['__url__']}, key: {data_dict['__key__']}", rank0_only=False
            )
            return None

        if not isinstance(video, bytes):
            return data_dict

        video_info = {
            "fps": meta_dict["framerate"],
            "n_orig_video_frames": meta_dict["nb_frames"],
        }

        if video_info["fps"] < self.min_fps:
            log.warning(f"Video FPS {video_info['fps']} is less than min_fps {self.min_fps}", rank0_only=False)
            return None
        if video_info["fps"] > self.max_fps:
            log.warning(f"Video FPS {video_info['fps']} is greater than max_fps {self.max_fps}", rank0_only=False)
            return None

        options: list = list((i, item) for i, item in enumerate(meta_dict[self.key_for_caption]))

        # Skip the last window if possible.
        # All windows except the last are 5 seconds long. The last window has a duration in the range [2.5s, 7.5), which is less preferred.
        if len(options) > 1:
            options = options[:-1]

        # shuffle options
        random.shuffle(options)
        video_frames = None
        selected_chunk_index = None
        for chunk_index, option in options:
            start_frame = option["start_frame"]
            end_frame = option["end_frame"]
            if (end_frame - start_frame) < self.min_duration * video_info["fps"]:
                continue
            if self.use_native_fps:
                if (end_frame - start_frame) < self.num_frames:
                    continue

            video_buffer = io.BytesIO(video)
            video_reader = decord.VideoReader(video_buffer)
            orig_h, orig_w, _ = video_reader[0].shape  # Get original dimensions
            aspect_ratio = detect_aspect_ratio((orig_w, orig_h))
            new_w, new_h = VIDEO_RES_SIZE_INFO[self.resolution][aspect_ratio]
            scaling_ratio = min((new_w / orig_w), (new_h / orig_h))
            new_w, new_h = int(scaling_ratio * orig_w + 0.5), int(scaling_ratio * orig_h + 0.5)
            # Reload video with resized dimensions
            video_buffer = io.BytesIO(video)
            video_reader = decord.VideoReader(
                video_buffer, num_threads=self.video_decode_num_threads, width=new_w, height=new_h
            )

            if self.use_native_fps:
                if "alpamayo" in data_dict["__url__"].root:
                    start_frame += 5
                if (end_frame - start_frame) < self.num_frames:
                    continue

                # take mid self.num_frames frames from start frame to end frame.
                total_frames = end_frame - start_frame
                # always try lower fps if possible.
                num_multiplier = total_frames // self.num_frames
                expected_length = self.num_frames * num_multiplier
                _start_frame = start_frame + (total_frames - expected_length) // 2
                _end_frame = _start_frame + expected_length
                frame_indices = np.arange(_start_frame, _end_frame, num_multiplier).tolist()
                assert len(frame_indices) == self.num_frames, "frame_indices length is not equal to num_frames"
                try:
                    video_frames = video_reader.get_batch(frame_indices).asnumpy()
                except Exception as e:
                    log.warning(
                        f"Video is not long enough, return None. url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                        rank0_only=False,
                    )
                    return None
                video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
                video_reader.seek(0)  # set video reader point back to 0 to clean up cache
                del video_reader  # delete the reader to avoid memory leak
                selected_chunk_index = chunk_index
                break

            else:
                frame_indices = np.arange(start_frame, end_frame).tolist()

                # online hot-fix for alpamayo data. Skip the first 5 frames as there is chance that the first five frames contain black frames.
                if "alpamayo" in data_dict["__url__"].root:
                    assert len(frame_indices) >= 5, (
                        "Getting less than 5 frames for alpamayo videos. There is no way to skip the first five frames."
                    )
                    frame_indices = frame_indices[5:]
                    start_frame += 5
                video_frames = video_reader.get_batch(frame_indices).asnumpy()

                video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

                # Clean up
                video_reader.seek(0)  # set video reader point back to 0 to clean up cache
                del video_reader  # delete the reader to avoid memory leak
                selected_chunk_index = chunk_index
                break

        if video_frames is None:
            log.warning(
                f"No valid video frames found, return None. url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                rank0_only=False,
            )
            return None

        # Extract a random frame from a different chunk for image context
        image_context = None
        if len(options) > 1:
            # Get all chunks except the selected one
            other_chunks = [opt for i, opt in options if i != selected_chunk_index]
            if other_chunks:
                # Randomly select a chunk
                context_chunk = random.choice(other_chunks)
                context_start = context_chunk["start_frame"]
                context_end = context_chunk["end_frame"]

                # Create a new video reader for the context frame
                video_buffer = io.BytesIO(video)
                video_reader = decord.VideoReader(video_buffer, num_threads=self.video_decode_num_threads)

                # Randomly select a frame from the chunk
                context_frame_idx = random.randint(context_start, context_end - 1)
                context_frame = video_reader.get_batch([context_frame_idx]).asnumpy()
                image_context = torch.from_numpy(context_frame).permute(3, 0, 1, 2)  # (1, H, W, C) -> (C, 1, H, W)

                # Clean up
                video_reader.seek(0)
                del video_reader
        else:
            # If there's only one chunk, randomly select a frame from it
            video_buffer = io.BytesIO(video)
            video_reader = decord.VideoReader(video_buffer, num_threads=self.video_decode_num_threads)
            context_frame_idx = random.randint(0, video_info["n_orig_video_frames"] - 1)
            context_frame = video_reader.get_batch([context_frame_idx]).asnumpy()
            image_context = torch.from_numpy(context_frame).permute(3, 0, 1, 2)  # (1, H, W, C) -> (C, 1, H, W)

            # Clean up
            video_reader.seek(0)
            del video_reader

        video_info["chunk_index"] = selected_chunk_index
        video_info["frame_start"] = start_frame
        video_info["frame_end"] = end_frame
        video_info["num_frames"] = end_frame - start_frame
        video_info["frame_indices"] = frame_indices

        video_info["video"] = video_frames
        video_info["image_context"] = image_context

        # update data_dict, make it back-compatible with old datasets, which video decoding happens in the decoder stage.
        data_dict[self.video_key] = video_info

        return data_dict
