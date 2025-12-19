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

"""
Utilities for camera-conditioned predict2/camera models.

Centralizes conversion from (extrinsics, intrinsics) to Plücker ray maps so we
can keep dataloaders lightweight and compute rays inside conditioners.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from einops import rearrange

from cosmos_predict2._src.imaginaire.modules.camera import Camera
from cosmos_predict2._src.predict2.conditioner import ReMapkey


def _normalize_image_size(image_size: torch.Tensor | Sequence[int] | Tuple[int, int]) -> tuple[int, int]:
    """
    Normalize various `image_size` representations into (H, W).

    Accepts:
    - (H, W)
    - tensor/list/tuple with at least 2 elements (H, W, ...)
    - batched tensor of shape (B, 2|4) where all rows must match
    """
    if isinstance(image_size, (tuple, list)):
        if len(image_size) < 2:
            raise ValueError(f"image_size must have at least 2 elements, got {image_size}")
        return int(image_size[0]), int(image_size[1])

    if not isinstance(image_size, torch.Tensor):
        raise TypeError(f"Unsupported image_size type: {type(image_size)}")

    if image_size.ndim == 1:
        if image_size.numel() < 2:
            raise ValueError(f"image_size must have at least 2 elements, got shape {tuple(image_size.shape)}")
        return int(image_size[0].item()), int(image_size[1].item())

    if image_size.ndim == 2:
        # (B, 2|4|...)
        if image_size.shape[1] < 2:
            raise ValueError(f"image_size must have at least 2 columns, got shape {tuple(image_size.shape)}")
        first = image_size[0, :2]
        if not torch.all(image_size[:, :2] == first.unsqueeze(0)):
            raise ValueError(
                "Per-sample image_size differs within a batch; convert_camera_to_plucker_rays requires a single (H, W)."
            )
        return int(first[0].item()), int(first[1].item())

    raise ValueError(f"Unsupported image_size shape: {tuple(image_size.shape)}")


def convert_camera_to_plucker_rays(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: torch.Tensor | Sequence[int] | Tuple[int, int],
    *,
    patch_spatial: int = 16,
    camera_patch_average: bool = False,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> torch.Tensor:
    """
    Convert camera parameters to Plücker ray maps in the patch token format expected by CameraDiT models.

    Args:
        extrinsics: World-to-camera poses. Shape (..., 3, 4) or (..., 4, 4).
        intrinsics: Intrinsics. Shape (..., 4) (fx, fy, cx, cy) or (..., 3, 3).
        image_size: (H, W) or tensor/list with at least 2 entries. Many datasets store (H, W, H, W).
        patch_spatial: Patch size p such that H and W are divisible by p.
        camera_patch_average: If True, average rays within each p×p patch (channels=6).
            If False, flatten p×p rays into channels (channels=6*p*p), default.
        out_dtype: Output dtype (bf16 by default for memory).

    Returns:
        Plücker ray tokens with shape (..., H/p, W/p, C) where C is 6 or 6*p*p.
    """
    if extrinsics.ndim < 2:
        raise ValueError(f"extrinsics must have shape (..., 3, 4) or (..., 4, 4); got {tuple(extrinsics.shape)}")

    if extrinsics.shape[-2:] == (4, 4):
        w2c = extrinsics[..., :3, :]
    elif extrinsics.shape[-2:] == (3, 4):
        w2c = extrinsics
    else:
        raise ValueError(f"Unsupported extrinsics shape: {tuple(extrinsics.shape)}")

    if intrinsics.shape[-2:] == (3, 3):
        K = intrinsics
    elif intrinsics.shape[-1] == 4:
        K = Camera.intrinsic_params_to_matrices(intrinsics)
    else:
        raise ValueError(f"Unsupported intrinsics shape: {tuple(intrinsics.shape)}")

    H, W = _normalize_image_size(image_size)
    if (H % patch_spatial) != 0 or (W % patch_spatial) != 0:
        raise ValueError(f"image_size {(H, W)} must be divisible by patch_spatial={patch_spatial}")

    prefix_shape = w2c.shape[:-2]
    if K.shape[:-2] != prefix_shape:
        # K might be (...,3,3) while w2c is (...,3,4)
        raise ValueError(f"extrinsics prefix {prefix_shape} != intrinsics/K prefix {K.shape[:-2]}")

    # Flatten leading dims so Camera.get_plucker_rays sees [N, ...]
    N = int(torch.tensor(prefix_shape).prod().item()) if len(prefix_shape) > 0 else 1
    w2c_flat = w2c.reshape(N, 3, 4)
    K_flat = K.reshape(N, 3, 3)

    plucker_flat = Camera.get_plucker_rays(w2c_flat, K_flat, (H, W))  # [N, HW, 6]
    if not isinstance(plucker_flat, torch.Tensor):
        plucker_flat = torch.as_tensor(plucker_flat)
    plucker_hw = plucker_flat.reshape(N, H, W, 6)
    plucker_hw = plucker_hw.view(*prefix_shape, H, W, 6)

    if camera_patch_average:
        # (..., H, W, 6) -> (..., H/p, W/p, 6)
        plucker_tokens = rearrange(
            plucker_hw,
            "... (h p1) (w p2) c -> ... h w p1 p2 c",
            p1=patch_spatial,
            p2=patch_spatial,
        ).mean(dim=(-3, -2))
    else:
        # (..., H, W, 6) -> (..., H/p, W/p, 6*p*p)
        plucker_tokens = rearrange(
            plucker_hw,
            "... (h p1) (w p2) c -> ... h w (p1 p2 c)",
            p1=patch_spatial,
            p2=patch_spatial,
        )

    if out_dtype is not None:
        plucker_tokens = plucker_tokens.to(dtype=out_dtype)
    return plucker_tokens


class CameraToPluckerRays(ReMapkey):
    """
    Conditioner embedder that converts (extrinsics, intrinsics, image_size) into the Plücker-ray token map expected by
    CameraDiT.

    This is used by predict2/camera conditioners so dataloaders can return intrinsics/extrinsics only.
    """

    def __init__(
        self,
        extrinsics_key: str = "extrinsics",
        intrinsics_key: str = "intrinsics",
        image_size_key: str = "image_size",
        output_key: str = "camera",
        patch_spatial: int = 16,
        camera_patch_average: bool = False,
        out_dtype: str | None = "bfloat16",
        dropout_rate: float = 0.0,
    ):
        # ReMapkey expects an input_key; we actually consume 3 keys, so we override `_input_key` to a list.
        super().__init__(input_key=extrinsics_key, output_key=output_key, dropout_rate=dropout_rate, dtype=None)
        self._input_key = [extrinsics_key, intrinsics_key, image_size_key]
        self.output_key = output_key
        self.patch_spatial = int(patch_spatial)
        self.camera_patch_average = bool(camera_patch_average)
        self.out_dtype = {
            None: None,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "half": torch.float16,
            "float16": torch.float16,
        }[out_dtype]

    def forward(self, extrinsics: torch.Tensor, intrinsics: torch.Tensor, image_size: torch.Tensor):
        camera = convert_camera_to_plucker_rays(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_size=image_size,
            patch_spatial=self.patch_spatial,
            camera_patch_average=self.camera_patch_average,
            out_dtype=self.out_dtype,
        )
        return {self.output_key: camera}
