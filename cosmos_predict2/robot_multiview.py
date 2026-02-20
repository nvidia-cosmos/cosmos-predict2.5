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

import os
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2
from einops import rearrange
from loguru import logger
from PIL import Image
from torchvision import transforms

from cosmos_predict2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_predict2._src.imaginaire.modules.camera import Camera
from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.inference.video2world import CameraConditionInputs, Video2WorldInference
from cosmos_predict2.config import MODEL_CHECKPOINTS, load_callable
from cosmos_predict2.robot_multiview_config import (
    CameraLoadFn,
    RobotMultiviewInferenceArguments,
    RobotMultiviewSetupArguments,
)


def _finalize_camera_metadata(
    data: dict[str, Any],
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Normalize camera tensors for downstream consumption."""
    inverted_pose = cast(torch.Tensor, Camera.invert_pose(extrinsics[:, :3, :]))
    extrinsics_flat = inverted_pose.reshape(-1, 3, 4)
    intrinsics_flat = intrinsics.reshape(-1, 4)
    image_size = torch.tensor([height, width, height, width], device=extrinsics_flat.device)

    data["extrinsics"] = extrinsics_flat
    data["intrinsics"] = intrinsics_flat
    data["image_size"] = image_size
    return data


def load_agibot_camera_fn() -> CameraLoadFn:
    cam_data_list = ["extrinsic_head", "extrinsic_hand_0", "extrinsic_hand_1"]
    intrinsic_data_list = ["intrinsic_head", "intrinsic_hand_0", "intrinsic_hand_1"]

    def load_fn(
        text: str,
        video: torch.Tensor,
        path: str,
        base_path: str,
        latent_frames: int,
        *,
        width: int,
        height: int,
        input_video_res: str,
        patch_spatial: int,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        # pyrefly: ignore  # missing-attribute
        input_idx = int(re.search(r"input_images/(\d+)", path).group(1))
        data: dict[str, Any] = {"text": text, "video": video, "path": path}
        extrinsics_list = []
        for cam_type in cam_data_list:
            extrinsics_tgt = torch.tensor(
                np.loadtxt(os.path.join(base_path, "cameras", f"{input_idx}_{cam_type}.txt"))
            ).to(torch.bfloat16)
            extrinsics_tgt = extrinsics_tgt[:latent_frames]
            extrinsics_tgt = torch.cat(
                (
                    extrinsics_tgt,
                    torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.bfloat16).unsqueeze(0).expand(latent_frames, -1),
                ),
                dim=1,
            ).reshape(-1, 4, 4)
            extrinsics_list.append(extrinsics_tgt)
        extrinsics = torch.cat(extrinsics_list, dim=0)

        intrinsics_list = []
        for intrinsic_type in intrinsic_data_list:
            intrinsics_tgt = torch.tensor(
                np.loadtxt(os.path.join(base_path, "cameras", f"{input_idx}_{intrinsic_type}.txt"))
            ).to(torch.bfloat16)
            intrinsics_tgt = intrinsics_tgt[:latent_frames]
            intrinsics_list.append(intrinsics_tgt)
        intrinsics = torch.cat(intrinsics_list, dim=0)

        if input_video_res == "720p":
            scale_w = 1280 / 768
            scale_h = 704 / 432
            intrinsics[:, [0, 2]] *= scale_w
            intrinsics[:, [1, 3]] *= scale_h

        result.append(
            _finalize_camera_metadata(
                data=data,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                height=height,
                width=width,
            )
        )
        return result

    return load_fn


class TextImageCameraDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path: str,
        args: RobotMultiviewSetupArguments,
        inference_args: list[RobotMultiviewInferenceArguments],
        num_frames: int,
        max_num_frames: int = 93,
        frame_interval: int = 1,
        patch_spatial: int = 16,
        camera_load_fn: CameraLoadFn | None = None,
    ):
        assert camera_load_fn is not None, "not provided function to load camera metadata"
        self.camera_load_fn = camera_load_fn
        self.base_path = base_path
        self.num_output_video = args.num_output_video
        self.data = inference_args

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.latent_frames = num_frames // 4 + 1
        self.patch_spatial = patch_spatial
        self.input_video_res = args.input_video_res
        if self.input_video_res == "720p":
            self.height, self.width = 704, 1280
        elif self.input_video_res == "480p":
            self.height, self.width = 432, 768
        self.args = args

        # pyrefly: ignore  # implicit-import
        self.frame_process = transforms.v2.Compose(
            [
                # pyrefly: ignore  # implicit-import
                transforms.v2.CenterCrop(size=(self.height, self.width)),
                # pyrefly: ignore  # implicit-import
                transforms.v2.Resize(size=(self.height, self.width), antialias=True),
                # pyrefly: ignore  # implicit-import
                transforms.v2.ToTensor(),
                # pyrefly: ignore  # implicit-import
                transforms.v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        # pyrefly: ignore  # implicit-import
        image = torchvision.transforms.functional.resize(
            image,
            # pyrefly: ignore  # bad-argument-type
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return image

    def load_images(self, input_name: str) -> torch.Tensor:
        images_list = []
        images_name = ["head", "hand_0", "hand_1"]
        for image_name in images_name:
            image_path = os.path.join(self.base_path, "input_images", input_name + "_" + image_name + ".png")
            image = Image.open(image_path)
            image = self.crop_and_resize(image)
            image = self.frame_process(image)
            image = image.unsqueeze(0).expand(self.num_frames, -1, -1, -1)
            images_list.append(image)
        images = torch.cat(images_list, dim=0)
        images = rearrange(images, "T C H W -> C T H W")
        return images

    # pyrefly: ignore [bad-param-name-override]
    def __getitem__(self, data_id: int):
        inference_args = self.data[data_id]
        input_name = inference_args.input_name or ""
        text = inference_args.prompt

        images = self.load_images(input_name)

        assert text is not None
        return self.camera_load_fn(
            text=text,
            video=images,
            path=os.path.join(self.base_path, "input_images", input_name),
            base_path=self.base_path,
            latent_frames=self.latent_frames,
            width=self.width,
            height=self.height,
            input_video_res=self.input_video_res,
            patch_spatial=self.patch_spatial,
        )

    def __len__(self):
        return len(self.data)


def inference(
    setup_args: RobotMultiviewSetupArguments,
    all_inference_args: list[RobotMultiviewInferenceArguments],
):
    """Run robot multiview inference using resolved setup and per-run arguments."""
    assert len(all_inference_args) > 0

    create_camera_load_fn = load_callable(setup_args.camera_load_create_fn)
    dataset = TextImageCameraDataset(
        # pyrefly: ignore  # bad-argument-type
        base_path=setup_args.base_path,
        args=setup_args,
        inference_args=all_inference_args,
        num_frames=setup_args.num_output_frames,
        camera_load_fn=create_camera_load_fn(),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=setup_args.dataloader_num_workers,
    )

    checkpoint = MODEL_CHECKPOINTS[setup_args.model_key]
    experiment = setup_args.experiment or checkpoint.experiment

    checkpoint_path = setup_args.checkpoint_path or checkpoint.s3.uri

    vid2vid_cli = Video2WorldInference(
        # pyrefly: ignore  # bad-argument-type
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        # pyrefly: ignore  # bad-argument-type
        context_parallel_size=setup_args.context_parallel_size,
        config_file=setup_args.config_file,
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory usage after model load: {mem_bytes / (1024**3):.2f} GB")

    # Only process files on rank 0 if using distributed processing
    rank0 = True
    # pyrefly: ignore  # unsupported-operation
    if setup_args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0
    if rank0:
        setup_args.output_dir.mkdir(parents=True, exist_ok=True)
        config_path = setup_args.output_dir / "config.yaml"
        LazyConfig.save_yaml(vid2vid_cli.config, str(config_path))
        logger.info(f"Saved config to {config_path}")

    # Process each file in the input directory
    for batch_idx, (inference_args, batch) in enumerate(zip(all_inference_args, dataloader)):
        for video_idx in range(len(batch)):
            ex = batch[video_idx]
            tgt_text = ex["text"][0]
            input_name = inference_args.input_name or ""
            src_video = ex["video"]
            negative_prompt = inference_args.negative_prompt

            camera_inputs = CameraConditionInputs(
                extrinsics=ex["extrinsics"],
                intrinsics=ex["intrinsics"],
                image_size=ex["image_size"],
            )

            video = vid2vid_cli.generate_vid2world(
                prompt=tgt_text,
                input_path=src_video,
                camera=camera_inputs,
                num_input_video=setup_args.num_input_video,
                num_output_video=setup_args.num_output_video,
                num_latent_conditional_frames=setup_args.num_input_frames,
                num_video_frames=setup_args.num_output_frames,
                seed=inference_args.seed,
                guidance=inference_args.guidance,
                negative_prompt=negative_prompt,
                num_steps=inference_args.num_steps,
            )

            if rank0:
                output_name = f"video_{batch_idx:02d}"
                assert experiment is not None
                save_root = Path(setup_args.output_dir) / experiment / Path(checkpoint_path).name / input_name
                save_root.mkdir(parents=True, exist_ok=True)

                output_path = str(save_root / output_name)
                save_img_or_video((1.0 + video[0]) / 2, output_path, fps=30)
                logger.info(f"Saved video to {output_path}")

    # Synchronize all processes before cleanup
    # pyrefly: ignore  # unsupported-operation
    if setup_args.context_parallel_size > 1:
        torch.distributed.barrier()

    # Clean up distributed resources
    vid2vid_cli.cleanup()
