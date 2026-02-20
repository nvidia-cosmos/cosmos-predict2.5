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
import tempfile

import numpy as np

from cosmos_predict2._src.transfer2.auxiliary.sam2.sam2_model import VideoSegmentationModel
from cosmos_predict2._src.transfer2.auxiliary.sam2.sam2_utils import (
    capture_fps,
    generate_tensor_from_images,
    generate_video_from_images,
    video_to_frames,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Video Segmentation using SAM2")
    parser.add_argument("--input_video", type=str, required=True, help="Path to input video file")
    parser.add_argument(
        "--output_video", type=str, default="./outputs/output_video.mp4", help="Path to save the output video"
    )
    parser.add_argument(
        "--output_tensor", type=str, default="./outputs/output_tensor.pt", help="Path to save the output tensor"
    )
    parser.add_argument(
        "--mode", type=str, choices=["points", "box", "prompt"], default="points", help="Segmentation mode"
    )
    parser.add_argument("--prompt", type=str, help="Text prompt for prompt mode")
    parser.add_argument(
        "--grounding_model_path",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="Local directory for GroundingDINO model files",
    )
    parser.add_argument(
        "--points",
        type=str,
        default="200,300",
        help="Comma-separated point coordinates for points mode (e.g., '200,300' or for multiple points use ';' as a separator, e.g., '200,300;100,150').",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="1",
        help="Comma-separated labels for points mode (e.g., '1' or '1,0' for multiple points).",
    )
    parser.add_argument(
        "--box",
        type=str,
        default="300,0,500,400",
        help="Comma-separated box coordinates for box mode (e.g., '300,0,500,400').",
    )
    # New flag to control visualization.
    parser.add_argument("--visualize", action="store_true", help="If set, visualize segmentation frames (save images)")
    return parser.parse_args()


def parse_points(points_str):
    """Parse a string of points into a numpy array.
    Supports a single point ('200,300') or multiple points separated by ';' (e.g., '200,300;100,150').
    """
    points = []
    for point in points_str.split(";"):
        coords = point.split(",")
        if len(coords) != 2:
            continue
        points.append([float(coords[0]), float(coords[1])])
    return np.array(points, dtype=np.float32)


def parse_labels(labels_str):
    """Parse a comma-separated string of labels into a numpy array."""
    return np.array([int(x) for x in labels_str.split(",")], dtype=np.int32)


def parse_box(box_str):
    """Parse a comma-separated string of 4 box coordinates into a numpy array."""
    return np.array([float(x) for x in box_str.split(",")], dtype=np.float32)


def main():
    args = parse_args()

    # Initialize the segmentation model.
    model = VideoSegmentationModel(**vars(args))

    # Prepare input data based on the selected mode.
    if args.mode == "points":
        input_data = {"points": parse_points(args.points), "labels": parse_labels(args.labels)}
    elif args.mode == "box":
        input_data = {"box": parse_box(args.box)}
    elif args.mode == "prompt":
        input_data = {"text": args.prompt}

    with tempfile.TemporaryDirectory() as temp_input_dir:
        fps = capture_fps(args.input_video)
        video_to_frames(args.input_video, temp_input_dir)
        with tempfile.TemporaryDirectory() as temp_output_dir:
            masks = model.sample(
                video_dir=temp_input_dir,
                mode=args.mode,
                input_data=input_data,
                save_dir=str(temp_output_dir),
                visualize=True,
            )
            generate_video_from_images(masks, args.output_video, fps)
            generate_tensor_from_images(temp_output_dir, args.output_tensor, fps, "mask")


if __name__ == "__main__":
    print("Starting video segmentation...")
    main()
