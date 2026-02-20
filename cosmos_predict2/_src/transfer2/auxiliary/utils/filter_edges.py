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
import numpy as np


def ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def make_kernel(px: int) -> np.ndarray:
    """Return an odd-sized elliptical kernel roughly px radius."""
    k = max(1, int(px) * 2 + 1)  # ensure odd
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def grow_and_feather_mask(
    mask_frame: np.ndarray, threshold: int, grow_px: int = 0, close_px: int = 0, feather_px: int = 0
) -> np.ndarray:
    """
    1) Binarize by threshold.
    2) Optional closing (fill small holes/gaps).
    3) Optional dilation (expand outward by ~grow_px).
    4) Optional feather: soft ramp from 0..255 near mask boundary.
    Returns uint8 mask (0..255).
    """
    mask_gray = ensure_gray(mask_frame)

    # Binarize
    _, mask_bin = cv2.threshold(mask_gray, threshold, 255, cv2.THRESH_BINARY)

    # Close (fills pinholes/small gaps inside the mask)
    if close_px and close_px > 0:
        k_close = make_kernel(close_px)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k_close, iterations=1)

    # Dilate (expand outward)
    if grow_px and grow_px > 0:
        k_dilate = make_kernel(grow_px)
        mask_bin = cv2.dilate(mask_bin, k_dilate, iterations=1)

    # Feather: create a soft band around the boundary that passes some alpha
    if feather_px and feather_px > 0:
        # Distance from background to mask (in pixels)
        inv = cv2.bitwise_not(mask_bin)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)  # float32
        # Within feather band, ramp alpha up to 255
        # Pixels already in mask_bin stay at 255.
        feather_band = np.clip((feather_px - dist) / max(1e-6, feather_px), 0.0, 1.0)
        feather_alpha = (feather_band * 255).astype(np.uint8)
        # Combine: max keeps full mask at 255, boundary gets 1..254
        mask_soft = np.maximum(mask_bin, feather_alpha)
        return mask_soft

    return mask_bin


def apply_mask(
    edge_frame_bgr: np.ndarray,
    mask_frame: np.ndarray,
    threshold: int,
    grow_px: int = 0,
    close_px: int = 0,
    feather_px: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a grown/feathered mask to edge_frame_bgr.
    Returns (BGR, RGBA).
    """
    h, w = edge_frame_bgr.shape[:2]
    if mask_frame.shape[:2] != (h, w):
        mask_frame = cv2.resize(mask_frame, (w, h), interpolation=cv2.INTER_NEAREST)

    mask = grow_and_feather_mask(
        mask_frame,
        threshold=threshold,
        grow_px=grow_px,
        close_px=close_px,
        feather_px=feather_px,
    )

    # Bitwise AND uses mask as 0..255 alpha on each channel
    out_bgr = cv2.bitwise_and(edge_frame_bgr, edge_frame_bgr, mask=mask)

    # RGBA with provided alpha
    out_rgba = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2BGRA)
    out_rgba[:, :, 3] = mask  # soft alpha if feathering used

    return out_bgr, out_rgba


def filter_out_edges(
    edges_p: str, mask_p: str, out_p: str, threshold: int = 0, grow_px: int = 0, close_px: int = 0, feather_px: int = 0
) -> None:
    edge_cap = cv2.VideoCapture(str(edges_p))
    mask_cap = cv2.VideoCapture(str(mask_p))

    if not edge_cap.isOpened():
        raise RuntimeError(f"Failed to open edge video: {edges_p}")
    if not mask_cap.isOpened():
        raise RuntimeError(f"Failed to open mask video: {mask_p}")

    fps = edge_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(edge_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(edge_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try a different --fourcc (e.g., avc1, XVID, MJPG).")

    while True:
        _, edge_frame = edge_cap.read()
        _, mask_frame = mask_cap.read()

        if edge_frame is None or mask_frame is None:
            break

        # Ensure edge is BGR
        if edge_frame.ndim == 2:
            edge_frame = cv2.cvtColor(edge_frame, cv2.COLOR_GRAY2BGR)

        out_bgr, out_rgba = apply_mask(
            edge_frame,
            mask_frame,
            threshold=threshold,
            grow_px=grow_px,
            close_px=close_px,
            feather_px=feather_px,
        )
        writer.write(out_bgr)

    edge_cap.release()
    mask_cap.release()
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter edges using a mask video with optional mask processing parameters."
    )

    parser.add_argument("edges_video", help="Path to the edge control modality video")
    parser.add_argument("mask_video", help="Path to the mask video")
    parser.add_argument("output", help="Path to the output filtered video")

    parser.add_argument("--threshold", type=int, default=0, help="Mask threshold (0–255). Default: 0")
    parser.add_argument("--grow_px", type=int, default=0, help="Grow mask radius in pixels. Default: 0")
    parser.add_argument("--close_px", type=int, default=0, help="Morphological closing radius in pixels. Default: 0")
    parser.add_argument("--feather_px", type=int, default=0, help="Feather (blur) radius in pixels. Default: 0")

    args = parser.parse_args()

    filter_out_edges(
        args.edges_video,
        args.mask_video,
        args.output,
        threshold=args.threshold,
        grow_px=args.grow_px,
        close_px=args.close_px,
        feather_px=args.feather_px,
    )


"""
Usage (MP4 output):
  python cosmos_predict2/_src/transfer2/auxiliary/utils/filter_edges.py \
    edge.mp4 \
    mask.mp4 \
    output.mp4 \
    --threshold 0 \
    --grow_px 3 \
    --close_px 3 \
    --feather_px 2
"""
