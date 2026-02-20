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

"""VideoDepthAnything model wrapper for temporally consistent depth estimation."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from cosmos_predict2._src.transfer2.auxiliary.depth_anything.model_utils import get_model_cache_path

logger = logging.getLogger(__name__)

try:
    from video_depth_anything import video_depth
except ImportError as e:
    video_depth = None
    logger.warning(f"video_depth_anything package not available: {e}. Install with: uv sync")

# Model configurations for different encoder variants
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

# Default weight file names
WEIGHTS_NAME = {
    "vits": "video_depth_anything_vits.pth",
    "vitl": "video_depth_anything_vitl.pth",
}

# HuggingFace repository for weights
HF_REPO = {
    "vits": "depth-anything/Video-Depth-Anything-Small",
    "vitl": "depth-anything/Video-Depth-Anything-Large",
}


class VideoDepthAnythingModel:
    def __init__(
        self,
        encoder: str = "vits",
        device: Optional[str] = None,
    ):
        if encoder not in MODEL_CONFIGS:
            raise ValueError(f"Unknown encoder: {encoder}. Choose from {list(MODEL_CONFIGS.keys())}")

        self.encoder = encoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def setup(self, weights_path: Optional[str] = None) -> None:
        """
        Load the model weights.

        Args:
            weights_path: Path to the model weights file. If None, will try to
                         download from HuggingFace or use cached weights.
        """
        if self.model is not None:
            logger.info("Model already loaded, skipping setup")
            return

        logger.info(f"Loading VideoDepthAnything model with {self.encoder} encoder")

        # Determine weights path
        if weights_path is None:
            weights_path = self._get_or_download_weights()

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found at {weights_path}. Please download from HuggingFace: {HF_REPO[self.encoder]}"
            )

        # Load model
        if video_depth is None:
            raise RuntimeError(
                "video_depth_anything package not available. Install with: pip install video-depth-anything"
            )
        self.model = video_depth.VideoDepthAnything(**MODEL_CONFIGS[self.encoder])
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        self.model = self.model.to(self.device).eval()

        logger.info(f"Model loaded successfully from {weights_path} on {self.device}")

    def _get_or_download_weights(self) -> str:
        """
        Get weights from cache or download from HuggingFace.

        Returns:
            Path to the weights file
        """
        cache_dir = get_model_cache_path(f"video_depth_anything_{self.encoder}")
        weights_file = cache_dir / WEIGHTS_NAME[self.encoder]

        if weights_file.exists():
            logger.info(f"Using cached weights: {weights_file}")
            return str(weights_file)

        # Try to download from HuggingFace
        try:
            from huggingface_hub import hf_hub_download

            logger.info(f"Downloading weights from {HF_REPO[self.encoder]}")
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO[self.encoder],
                filename=WEIGHTS_NAME[self.encoder],
                cache_dir=cache_dir,
            )
            logger.info(f"Downloaded weights to {downloaded_path}")
            return downloaded_path
        except Exception as e:
            logger.error(f"Failed to download weights: {e}")
            raise RuntimeError(
                f"Could not find or download weights for {self.encoder}. "
                f"Please download manually from {HF_REPO[self.encoder]} "
                f"and place at {weights_file}"
            ) from e

    def generate(self, video: np.ndarray) -> np.ndarray:
        """Generate depth maps that match input video dimensions exactly."""
        assert video.ndim == 4, "Video tensor should have shape (T, H, W, 3)"
        assert video.dtype == np.uint8, "Video tensor should be uint8"

        original_h, original_w = video.shape[1], video.shape[2]
        depths, _ = self.model.infer_video_depth(video, 30, device=self.device)  # type: ignore

        # Resize depth back to original dimensions if needed
        if depths.shape[1] != original_h or depths.shape[2] != original_w:
            import cv2

            resized_depths = []
            for depth_frame in depths:
                resized = cv2.resize(depth_frame, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
                resized_depths.append(resized)
            depths = np.stack(resized_depths)

        return depths
