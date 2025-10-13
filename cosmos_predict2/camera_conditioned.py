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

import imageio
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2
from einops import rearrange
from loguru import logger
from PIL import Image
from torchvision import transforms

from cosmos_predict2._src.imaginaire.modules.camera import Camera
from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2.camera_conditioned_config import (
    CameraConditionedInferenceArguments,
    CameraConditionedSetupArguments,
    CameraLoadFn,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, load_callable


def load_basic_camera_fn():
    data_all_cams = []
    cam_data_lists = [
        # dynamic cameras
        ["rot_right", "arc_left"],
        ["rot_left", "arc_right"],
        ["tilt_up", "translate_down_rot"],
        ["tilt_down", "translate_up_rot"],
        ["zoom_in", "zoom_out"],
        # static cameras
        ["elevation_up_1", "elevation_up_2"],
        ["azimuth_right", "azimuth_left"],
        ["distance_away_1", "distance_away_2"],
    ]

    focal_data_lists = [
        # focal lengths
        "focal24",
        "focal50",
    ]

    def load_fn(
        text: str,
        video: torch.Tensor,
        path: str,
        base_path: str,
        latent_frames: int,
        width: int,
        height: int,
        input_video_res: str,
        patch_spatial: int,
    ):
        result = []
        for cam_data_list in cam_data_lists:
            for focal_data in focal_data_lists:
                data = {"text": text, "video": video, "path": path}
                extrinsics_list = []
                for cam_type in cam_data_list:
                    extrinsics_tgt = torch.tensor(np.loadtxt(os.path.join(base_path, "cameras", cam_type + ".txt"))).to(
                        torch.bfloat16
                    )
                    extrinsics_tgt = extrinsics_tgt[:latent_frames]
                    extrinsics_tgt = torch.cat(
                        (
                            extrinsics_tgt,
                            torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.bfloat16)
                            .unsqueeze(0)
                            .expand(latent_frames, -1),
                        ),
                        dim=1,
                    ).reshape(-1, 4, 4)
                    extrinsics_list.append(extrinsics_tgt)
                extrinsics = torch.cat(extrinsics_list, dim=0)
                # assert input video has static cameras
                extrinsics = torch.cat(
                    (torch.eye(4).unsqueeze(0).expand(latent_frames, -1, -1).to(extrinsics), extrinsics), dim=0
                )
                intrinsics = torch.tensor(
                    np.loadtxt(os.path.join(base_path, "cameras", f"intrinsics_{focal_data}.txt"))
                ).to(torch.bfloat16)
                intrinsics = intrinsics[:latent_frames]
                intrinsics = intrinsics.unsqueeze(0).expand(extrinsics.shape[0], -1).clone()

                if input_video_res == "720p":
                    scale_w = 1280 / 768
                    scale_h = 704 / 432
                    intrinsics[:, [0, 2]] *= scale_w
                    intrinsics[:, [1, 3]] *= scale_h

                K = Camera.intrinsic_params_to_matrices(intrinsics)
                # extrinsics are 4x4 cam-to-world; compute 3x4 world-to-cam via invert on 3x4
                w2c = Camera.invert_pose(extrinsics[:, :3, :])
                plucker_flat = Camera.get_plucker_rays(w2c, K, (height, width))
                plucker_rays = plucker_flat.view(plucker_flat.shape[0], height, width, 6)
                plucker_rays = rearrange(
                    plucker_rays,
                    "T (H p1) (W p2) C -> T H W (p1 p2 C)",
                    p1=patch_spatial,
                    p2=patch_spatial,
                )
                data["camera"] = plucker_rays
                result.append(data)
            return result

    return load_fn


def load_agibot_camera_fn():
    data_all_cams = []
    cam_data_lists = [
        ["camera_tgt_0", "camera_tgt_1"],
    ]
    intrinsic_data_lists = ["intrinsic_head", "intrinsic_hand_0", "intrinsic_hand_1"]

    def load_fn(
        text: str,
        video: torch.Tensor,
        path: str,
        base_path: str,
        latent_frames: int,
        width: int,
        height: int,
        input_video_res: str,
        patch_spatial: int,
    ):
        result = []

        video_idx = int(re.search(r"videos/(\d+).mp4", path).group(1))
        for cam_data_list in cam_data_lists:
            data = {"text": text, "video": video, "path": path}
            extrinsics_list = []
            for cam_type in cam_data_list:
                extrinsics_tgt = torch.tensor(
                    np.loadtxt(
                        os.path.join(
                            base_path,
                            "cameras",
                            f"{video_idx}_{cam_type}.txt",
                        )
                    )
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
            # assert input video has static cameras (head-view)
            extrinsics = torch.cat(
                (torch.eye(4).unsqueeze(0).expand(latent_frames, -1, -1).to(extrinsics), extrinsics), dim=0
            )
            intrinsics_list = []
            for intrinsic_type in intrinsic_data_lists:
                intrinsics_tgt = torch.tensor(
                    np.loadtxt(os.path.join(base_path, "cameras", f"{video_idx}_{intrinsic_type}.txt"))
                ).to(torch.bfloat16)
                intrinsics_tgt = intrinsics_tgt[:latent_frames]
                intrinsics_list.append(intrinsics_tgt)
            intrinsics = torch.cat(intrinsics_list, dim=0)

            if input_video_res == "720p":
                scale_w = 1280 / 768
                scale_h = 704 / 432
                intrinsics[:, [0, 2]] *= scale_w
                intrinsics[:, [1, 3]] *= scale_h

            K = Camera.intrinsic_params_to_matrices(intrinsics)
            w2c = Camera.invert_pose(extrinsics[:, :3, :])

            plucker_flat = Camera.get_plucker_rays(w2c, K, (height, width))
            plucker_rays = plucker_flat.view(plucker_flat.shape[0], height, width, 6)
            plucker_rays = rearrange(
                plucker_rays,
                "T (H p1) (W p2) C -> T H W (p1 p2 C)",
                p1=patch_spatial,
                p2=patch_spatial,
            )
            data["camera"] = plucker_rays
            result.append(data)

        return result

    return load_fn


class TextVideoCameraDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path: str,
        args: CameraConditionedSetupArguments,
        inference_args: list[CameraConditionedInferenceArguments],
        num_frames: int,
        max_num_frames: int = 93,
        frame_interval: int = 1,
        patch_spatial: int = 16,
        height: int = 704,
        width: int = 1280,
        is_i2v: bool = False,
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
        self.is_i2v = is_i2v
        self.args = args

        self.frame_process = transforms.v2.Compose(
            [
                transforms.v2.CenterCrop(size=(self.height, self.width)),
                transforms.v2.Resize(size=(self.height, self.width), antialias=True),
                transforms.v2.ToTensor(),
                transforms.v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return image

    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if (
            reader.count_frames() < max_num_frames
            or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval
        ):
            reader.close()
            return None

        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames

    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(
            file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process
        )
        return frames

    def __getitem__(self, data_id):
        inference_args = self.data[data_id]
        path = str(inference_args.input_path)
        text = inference_args.prompt

        video = self.load_video(path)
        if video is None:
            raise ValueError(f"{path} is not a valid video.")

        result = self.camera_load_fn(
            text=text,
            video=video,
            path=path,
            base_path=self.base_path,
            latent_frames=self.latent_frames,
            width=self.width,
            height=self.height,
            input_video_res=self.input_video_res,
            patch_spatial=self.patch_spatial,
        )
        for x in result:
            x.update(
                {
                    "seed": inference_args.seed,
                    "guidance": inference_args.guidance,
                    "negative_prompt": inference_args.negative_prompt,
                }
            )

        return result

    def __len__(self):
        return len(self.data)


def inference(
    setup_args: CameraConditionedSetupArguments,
    all_inference_args: list[CameraConditionedInferenceArguments],
):
    """Run camera-conditioned inference using resolved setup and per-run arguments."""
    assert len(all_inference_args) > 0

    create_camera_load_fn = load_callable(setup_args.camera_load_create_fn)
    dataset = TextVideoCameraDataset(
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
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        context_parallel_size=setup_args.context_parallel_size,
        config_file=setup_args.config_file,
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory usage after model load: {mem_bytes / (1024**3):.2f} GB")

    # Only process files on rank 0 if using distributed processing
    rank0 = True
    if setup_args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    # Process each file in the input directory
    for batch_idx, batch in enumerate(dataloader):
        save_root = Path(setup_args.output_dir) / experiment / Path(checkpoint_path).name / f"video_{batch_idx + 1}"
        save_root.mkdir(parents=True, exist_ok=True)

        for video_idx in range(len(batch)):
            ex = batch[video_idx]
            tgt_text = ex["text"][0]
            src_video = ex["video"]
            tgt_camera = ex["camera"]

            video = vid2vid_cli.generate_vid2world(
                prompt=tgt_text,
                input_path=src_video,
                camera=tgt_camera,
                num_input_video=setup_args.num_input_video,
                num_output_video=setup_args.num_output_video,
                num_latent_conditional_frames=setup_args.num_input_frames,
                num_video_frames=setup_args.num_output_frames,
                seed=ex["seed"].item(),
                guidance=ex["guidance"].item(),
                negative_prompt=ex["negative_prompt"],
            )

            if rank0:
                output_name = f"cam_{(video_idx // 2 + 1):02d}_focal_{(video_idx % 2 + 1):02d}"
                save_img_or_video((1.0 + video[0]) / 2, str(save_root / output_name), fps=30)
                logger.info(f"Saved video to {save_root / output_name}")

    # Synchronize all processes before cleanup
    if setup_args.context_parallel_size > 1:
        torch.distributed.barrier()

    # Clean up distributed resources
    vid2vid_cli.cleanup()
