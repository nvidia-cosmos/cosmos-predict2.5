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

import cv2


def ranged_float(min_val: float, max_val: float):
    def checker(x: str) -> float:
        x_float = float(x)
        if not (min_val <= x_float <= max_val):
            raise argparse.ArgumentTypeError(f"Value must be between {min_val} and {max_val}")
        return x_float

    return checker


def ranged_int(min_val: int, max_val: int):
    def checker(x: str) -> int:
        x_int = int(x)
        if not (min_val <= x_int <= max_val):
            raise argparse.ArgumentTypeError(f"Value must be between {min_val} and {max_val}")
        return x_int

    return checker


def generate_edges(in_path: str, out_path: str, bright: int = 50, contrast: float = 1.0) -> None:
    cap = cv2.VideoCapture(in_path)
    assert cap.isOpened(), "Could not open input video."
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=False)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=bright)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.4)
        edges = cv2.Canny(blurred, 10, 50)
        out.write(edges)

    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate edge video from input.")

    parser.add_argument("input_video", help="Path to input video")
    parser.add_argument("output_video", help="Path to save generated edge video")

    parser.add_argument(
        "--bright",
        type=ranged_int(-255, 255),
        default=50,
        help="Brightness offset (-255 to 255). Default: 50",
    )
    parser.add_argument(
        "--contrast",
        type=ranged_float(0.0, 5.0),
        default=1.0,
        help="Contrast multiplier (0.0 to 5.0). Default: 1.0",
    )

    args = parser.parse_args()

    generate_edges(args.input_video, args.output_video, bright=args.bright, contrast=args.contrast)


"""
Usage (MP4 output):
  
python cosmos_predict2/_src/transfer2/auxiliary/utils/generate_edges.py \
input_video.mp4 \
edge.mp4 \
--bright 50 \
--contrast 1
"""
