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

import random
from typing import Callable, List, Optional

import attrs
import cv2
import numpy as np
import torch

from cosmos_predict2._src.transfer2.datasets.augmentors.fast_blur import BilateralGaussian


@attrs.define
class BilateralFilterConfig:
    """Configuration for Bilateral filter"""

    use_random: bool = False
    # if use_random is False, then optionally define the param values
    d: int = 30
    sigma_color: int = 150
    sigma_space: int = 100
    iter: int = 1

    # if use_random is True, then optionally define the range
    d_min: int = 15
    d_max: int = 50
    sigma_color_min: int = 100
    sigma_color_max: int = 300
    sigma_space_min: int = 50
    sigma_space_max: int = 150
    iter_min: int = 1
    iter_max: int = 2

    # Whether to use GPU kernel (inference only)
    use_cuda: bool = False


@attrs.define
class GaussianBlurConfig:
    """Configuration for Gaussian blur"""

    use_random: bool = False
    # if use_random is False, then optionally define the param values
    ksize: int = 25
    sigmaX: float = 12.5

    # if use_random is True, then optionally define the range
    ksize_min: int = 21
    ksize_max: int = 29
    sigmaX_min: float = 10.5
    sigmaX_max: float = 14.5


def apply_bilateral_filter(
    frames: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
    iter: int = 1,
    bilateral_cuda_module: Optional[Callable] = None,
) -> np.ndarray:
    blurred_image = []
    if bilateral_cuda_module is not None:
        frames = torch.from_numpy(frames).cuda()
        for _image in frames.permute(1, 2, 3, 0):
            # The hyper-parameters to NPP version of Bilateral Gaussian blur need to be adjusted.
            # The chosen values are chosen based on experiments, but it does not guarantee bitwise accuracy.
            blurred_image.append(bilateral_cuda_module(_image, d // 3, (sigma_color // 2) ** 2, sigma_space**2))
        blurred_image = torch.stack(blurred_image).permute(3, 0, 1, 2)
        blurred_image = blurred_image.cpu().numpy()
        return blurred_image

    for _image_np in frames.transpose((1, 2, 3, 0)):
        for _ in range(iter):
            _image_np = cv2.bilateralFilter(_image_np, d, sigma_color, sigma_space)
        blurred_image += [_image_np]

    blurred_image = np.stack(blurred_image).transpose((3, 0, 1, 2))
    return blurred_image


class BilateralFilter:
    def __init__(self, config: BilateralFilterConfig) -> None:
        self.use_random = config.use_random
        self.config = config
        assert not (self.use_random and self.config.use_cuda), "Cannot use GPU kernel for training."
        self.bilateral_cuda_module = BilateralGaussian() if self.config.use_cuda else None

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        config = self.config
        if self.use_random:
            d = np.random.randint(config.d_min, config.d_max)
            sigma_color = np.random.randint(config.sigma_color_min, config.sigma_color_max)
            sigma_space = np.random.randint(config.sigma_space_min, config.sigma_space_max)
            iter = np.random.randint(config.iter_min, config.iter_max)
        else:
            d = config.d
            sigma_color = config.sigma_color
            sigma_space = config.sigma_space
            iter = config.iter
        return apply_bilateral_filter(frames, d, sigma_color, sigma_space, iter, self.bilateral_cuda_module)


# frames CTHW
def apply_gaussian_blur(frames: np.ndarray, ksize: int = 5, sigmaX: float = 1.0) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd
    blurred_image = [
        cv2.GaussianBlur(_image_np, (ksize, ksize), sigmaX=sigmaX) for _image_np in frames.transpose((1, 2, 3, 0))
    ]
    blurred_image = np.stack(blurred_image).transpose((3, 0, 1, 2))
    return blurred_image


class GaussianBlur:
    def __init__(self, config: GaussianBlurConfig) -> None:
        self.use_random = config.use_random
        self.config = config

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if self.use_random:
            ksize = np.random.randint(self.config.ksize_min, self.config.ksize_max + 1)
            sigmaX = np.random.uniform(self.config.sigmaX_min, self.config.sigmaX_max)
        else:
            ksize = self.config.ksize
            sigmaX = self.config.sigmaX
        return apply_gaussian_blur(frames, ksize, sigmaX)


@attrs.define
class BlurCombinationConfig:
    """Configuration for a combination of blurs with associated probability"""

    # list of choices are:  ["gaussian", "bilateral"]
    # the corresponding config must be defined for each item in this blur_types list
    blur_types: List[str]
    probability: float
    gaussian_blur: Optional[GaussianBlurConfig] = None
    bilateral_filter: Optional[BilateralFilterConfig] = None


@attrs.define
class BlurConfig:
    """Configuration for blur augmentation with multiple combinations"""

    # probabilities from the list of combinations should add up to 1.0
    blur_combinations: List[BlurCombinationConfig] = []


# For training
random_blur_config = BlurConfig(
    blur_combinations=[
        BlurCombinationConfig(
            blur_types=["bilateral"],
            probability=0.3,
            bilateral_filter=BilateralFilterConfig(use_random=True),
        ),
        BlurCombinationConfig(
            blur_types=["gaussian"],
            probability=0.5,
            gaussian_blur=GaussianBlurConfig(use_random=True),
        ),
        BlurCombinationConfig(
            blur_types=["bilateral", "gaussian"],
            probability=0.2,
            bilateral_filter=BilateralFilterConfig(use_random=True),
            gaussian_blur=GaussianBlurConfig(use_random=True),
        ),
    ],
)

# For inference
bilateral_blur_config = BlurConfig(
    blur_combinations=[
        BlurCombinationConfig(
            blur_types=["bilateral"],
            probability=1.0,
            bilateral_filter=BilateralFilterConfig(use_random=False),
        ),
    ],
)


class Blur:
    def __init__(self, config: BlurConfig | None = None, use_random: bool = True) -> None:
        if config is None:
            config = random_blur_config if use_random else bilateral_blur_config
        probabilities = [combo.probability for combo in config.blur_combinations]
        total_prob = sum(probabilities)
        assert abs(total_prob - 1.0) < 1e-6, f"Probabilities must sum to 1.0, got {total_prob}"

        self.blur_combinations = config.blur_combinations
        self.probabilities = probabilities
        self._set_blur_instances()

    def _set_blur_instances(self):
        if not self.blur_combinations:
            return
        self.blur_combinations_instances = []

        for blur_combination in self.blur_combinations:
            blur_mapping = {
                "gaussian": (GaussianBlur, blur_combination.gaussian_blur),
                "bilateral": (BilateralFilter, blur_combination.bilateral_filter),
            }

            cur_instances = []
            for blur_type in blur_combination.blur_types:
                assert blur_type in blur_mapping, f"Unknown {blur_type}. Needs to correct blur_type or blur_mapping."

                blur_class, blur_config = blur_mapping[blur_type]
                cur_instances.append(blur_class(blur_config))

            self.blur_combinations_instances.append(cur_instances)

        assert len(self.blur_combinations_instances) == len(self.blur_combinations), (
            "Number of blur_combinations_instances needs to match number of blur_combinations."
        )

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        blur_instances = random.choices(self.blur_combinations_instances, weights=self.probabilities, k=1)[0]
        for ins in blur_instances:
            frames = ins(frames)
        return frames
