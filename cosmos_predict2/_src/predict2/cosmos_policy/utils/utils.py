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
import os
import random

import numpy as np
import torch
from PIL import Image


def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def duplicate_array(arr, total_num_copies):
    """
    Duplicates a NumPy array multiple times along a new first axis.

    Args:
        arr (numpy.ndarray): The input array to duplicate
        total_num_copies (int): Total number of copies to have in the end

    Returns:
        numpy.ndarray: A new array with shape (total_num_copies, *arr.shape)
    """
    # Create a new array by stacking the original array multiple times
    return np.stack([arr] * total_num_copies)


def jpeg_encode_image(image, quality=95):
    """Encode image as JPEG bytes."""
    if image.dtype != np.uint8:
        raise ValueError("Image array must be uint8 for JPEG encoding.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have shape (H, W, 3).")
    img = Image.fromarray(image)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)
