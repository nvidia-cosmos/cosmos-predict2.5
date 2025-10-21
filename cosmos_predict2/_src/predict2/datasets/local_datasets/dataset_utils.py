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

from typing import Any, Optional, Union

import torch
import torchvision.transforms.functional as F
from PIL import Image


def obtain_image_size(data_dict: dict, input_keys: list) -> tuple[int, int]:
    r"""Function for obtaining the image size from the data dict.

    Args:
        data_dict (dict): Input data dict
        input_keys (list): List of input keys
    Returns:
        width (int): Width of the input image
        height (int): Height of the input image
    """

    data1 = data_dict[input_keys[0]]
    if isinstance(data1, Image.Image):
        width, height = data1.size
    elif isinstance(data1, torch.Tensor):
        height, width = data1.size()[-2:]
    else:
        raise ValueError("data to random crop should be PIL Image or tensor")

    return width, height


def obtain_augmentation_size(data_dict: dict, augmentor_cfg: dict) -> Union[int, tuple]:
    r"""Function for obtaining size of the augmentation.
    When dealing with multi-aspect ratio dataloaders, we need to
    find the augmentation size from the aspect ratio of the data.

    Args:
        data_dict (dict): Input data dict
        augmentor_cfg (dict): Augmentor config
    Returns:
        aug_size (int): Size of augmentation
    """
    aspect_ratio = data_dict["aspect_ratio"]
    aug_size = augmentor_cfg["size"][aspect_ratio]
    return aug_size


class Augmentor:
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        r"""Base augmentor class

        Args:
            input_keys (list): List of input keys
            output_keys (list): List of output keys
            args (dict): Arguments associated with the augmentation
        """
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.args = args

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise ValueError("Augmentor not implemented")


class ResizeSmallestSideAspectPreserving(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs aspect-ratio preserving resizing.
        Image is resized to the dimension which has the smaller ratio of (size / target_size).
        First we compute (w_img / w_target) and (h_img / h_target) and resize the image
        to the dimension that has the smaller of these ratios.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        img_w, img_h = self.args["img_w"], self.args["img_h"]

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        scaling_ratio = max((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

        assert target_size[0] >= img_h and target_size[1] >= img_w, (
            f"Resize error. orig {(orig_w, orig_h)} desire {(img_w, img_h)} compute {target_size}"
        )

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            data_dict[out_key] = transforms_F.resize(  # noqa: F821
                data_dict[inp_key],
                size=target_size,  # type: ignore
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),  # noqa: F821
                antialias=True,
            )

            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class CenterCrop(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs center crop.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
            We also save the cropping parameters in the aug_params dict
            so that it will be used by other transforms.
        """
        assert (self.args is not None) and ("img_w" in self.args) and ("img_h" in self.args), (
            "Please specify size in args"
        )

        img_w, img_h = self.args["img_w"], self.args["img_h"]

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        for key in self.input_keys:
            data_dict[key] = transforms_F.center_crop(data_dict[key], [img_h, img_w])  # noqa: F821

        # We also add the aug params we use. This will be useful for other transforms
        crop_x0 = (orig_w - img_w) // 2
        crop_y0 = (orig_h - img_h) // 2
        cropping_params = {
            "resize_w": orig_w,
            "resize_h": orig_h,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_w": img_w,
            "crop_h": img_h,
        }

        if "aug_params" not in data_dict:
            data_dict["aug_params"] = dict()

        data_dict["aug_params"]["cropping"] = cropping_params
        data_dict["padding_mask"] = torch.zeros((1, cropping_params["crop_h"], cropping_params["crop_w"]))
        return data_dict


class Normalize(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs data normalization.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
        """
        assert self.args is not None, "Please specify args"

        mean = self.args["mean"]
        std = self.args["std"]

        for key in self.input_keys:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(dtype=torch.get_default_dtype()).div(255)
            else:
                data_dict[key] = transforms_F.to_tensor(  # noqa: F821
                    data_dict[key]
                )  # division by 255 is applied in to_tensor()  # noqa: F821

            data_dict[key] = transforms_F.normalize(tensor=data_dict[key], mean=mean, std=std)  # noqa: F821
        return data_dict


class ResizePreprocess:
    def __init__(self, size: tuple[int, int]):
        """
        Initialize the preprocessing class with the target size.
        Args:
        size (tuple): The target height and width as a tuple (height, width).
        """
        self.size = size

    def __call__(self, video_frames):
        """
        Apply the transformation to each frame in the video.
        Args:
        video_frames (torch.Tensor): A tensor representing a batch of video frames.
        Returns:
        torch.Tensor: The transformed video frames.
        """
        if video_frames.ndim == 4:
            # Resize each frame in the video
            resized_frames = torch.stack([F.resize(frame, self.size, antialias=True) for frame in video_frames])
        else:
            # Resize a single image
            resized_frames = F.resize(video_frames, self.size, antialias=True)

        return resized_frames


class ToTensorImage:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, image):
        """
        Args:
            image (torch.tensor, dtype=torch.uint8): Size is (C, H, W)
        Return:
            image (torch.tensor, dtype=torch.float): Size is (C, H, W)
        """
        return to_tensor_image(image)

    def __repr__(self) -> str:
        return self.__class__.__name__


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


def to_tensor_image(image):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of image tensor
    Args:
        image (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        image (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_image(image)
    if not image.dtype == torch.uint8:
        raise TypeError("image tensor should have data type uint8. Got %s" % str(image.dtype))
    return image.float() / 255.0


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))  # noqa: UP031
    return clip.float() / 255.0


def _is_tensor_image(image: torch.Tensor) -> bool:
    if not torch.is_tensor(image):
        raise TypeError("image should be Tensor. Got %s" % type(image))

    if not image.ndimension() == 3:
        raise ValueError("image should be 3D. Got %dD" % image.dim())

    return True


def _is_tensor_video_clip(clip) -> bool:
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))  # noqa: UP031

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())  # noqa: UP031

    return True
