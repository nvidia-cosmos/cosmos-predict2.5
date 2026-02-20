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

import argparse

import imageio.v3 as iio
import numpy as np


def invert_video(input_binary_mask: str, output_video_path: str) -> None:
    frames = list(iio.imiter(input_binary_mask))

    inverted_frames = []

    for frame in frames:
        # Convert to grayscale if needed
        if frame.ndim == 3:
            frame = frame[..., 0]

        # Ensure it's binary (0/255)
        frame = (frame > 127).astype(np.uint8) * 255

        # Invert
        inverted = 255 - frame

        inverted_frames.append(inverted)

    # Save as a new MP4 video
    meta = iio.immeta(input_binary_mask)
    fps = int(meta.get("fps", 30))  # Default to 30
    iio.imwrite(output_video_path, inverted_frames, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invert a binary mask video.")
    parser.add_argument("input_video", help="Path to input binary mask video")
    parser.add_argument("output_video", help="Path to output inverted binary mask video")

    args = parser.parse_args()

    invert_video(args.input_video, args.output_video)


"""
Usage (MP4 output):
  
python cosmos_predict2/_src/transfer2/auxiliary/utils/invert_mask.py \
input_video.mp4 \
inverted_mask.mp4
"""
