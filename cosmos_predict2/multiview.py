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
from pathlib import Path
from typing import Mapping

import decord
import numpy as np
import torch

from cosmos_predict2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_predict2._src.imaginaire.flags import SMOKE
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2_multiview.datasets.local import LocalMultiViewDataset
from cosmos_predict2._src.predict2_multiview.datasets.multiview import AugmentationConfig
from cosmos_predict2._src.predict2_multiview.scripts.inference import NUM_CONDITIONAL_FRAMES_KEY, Vid2VidInference
from cosmos_predict2.multiview_config import (
    MultiviewInferenceArguments,
    MultiviewInferenceArgumentsWithInputPaths,
    MultiviewSetupArguments,
)

RESOLUTIONS: Mapping = {
    "720p": (720, 1280),
}


def setup_config(
    resolution_hw: tuple[int, int],
    num_video_frames_per_view: int,
    fps_downsample_factor: int,
) -> AugmentationConfig:
    camera_keys = ("front_wide", "cross_right", "rear_right", "rear", "rear_left", "cross_left", "front_tele")
    kwargs = dict(
        resolution_hw=resolution_hw,
        fps_downsample_factor=fps_downsample_factor,
        num_video_frames=num_video_frames_per_view,
        camera_keys=camera_keys,
        camera_view_mapping=dict(zip(camera_keys, range(len(camera_keys)))),
        camera_caption_key_mapping={k: f"caption_{k}" for k in camera_keys},
        camera_video_key_mapping={k: f"video_{k}" for k in camera_keys},
        camera_control_key_mapping=None,
        add_view_prefix_to_caption=True,
        camera_prefix_mapping={
            "front_wide": "The video is captured from a camera mounted on a car. The camera is facing forward.",
            "cross_right": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
            "rear_right": "The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
            "rear": "The video is captured from a camera mounted on a car. The camera is facing backwards.",
            "rear_left": "The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
            "cross_left": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
            "front_tele": "The video is captured from a telephoto camera mounted on a car. The camera is facing forward.",
        },
        single_caption_camera_name="front_wide",
    )
    return AugmentationConfig(**kwargs)


class MultiviewInference:
    def __init__(self, args: MultiviewSetupArguments):
        log.debug(f"{args.__class__.__name__}({args})")

        # Enable deterministic inference
        os.environ["NVTE_FUSED_ATTN"] = "0"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.enable_grad(False)  # Disable gradient calculations for inference

        self.rank0 = distributed.is_rank0()
        self.setup_args = args
        self.guardrail_enabled = not args.disable_guardrails

        self.pipe = Vid2VidInference(
            # pyrefly: ignore  # bad-argument-type
            args.experiment,
            # pyrefly: ignore  # bad-argument-type
            args.checkpoint_path,
            # pyrefly: ignore  # bad-argument-type
            context_parallel_size=args.context_parallel_size,
        )
        if self.rank0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            # pyrefly: ignore  # bad-argument-type
            LazyConfig.save_yaml(self.pipe.config, args.output_dir / "config.yaml")

            if self.guardrail_enabled:
                self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                    offload_model_to_cpu=args.offload_guardrail_models
                )
                self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                    offload_model_to_cpu=args.offload_guardrail_models
                )
            else:
                # pyrefly: ignore  # bad-assignment
                self.text_guardrail_runner = None
                # pyrefly: ignore  # bad-assignment
                self.video_guardrail_runner = None

    def generate(
        self, samples: list[MultiviewInferenceArgumentsWithInputPaths] | MultiviewInferenceArguments, output_dir: Path
    ) -> list[str]:
        if self.setup_args.use_config_dataloader:
            assert isinstance(samples, MultiviewInferenceArguments)
            return self._generate_from_config_dataloader(samples, output_dir)
        else:
            assert isinstance(samples, list)
            if SMOKE:
                samples = samples[:1]
            output_paths: list[str] = []
            sample_names = [sample.name for sample in samples]
            log.info(f"Generating {len(samples)} samples: {sample_names}")
            for i_sample, sample in enumerate(samples):
                log.info(f"[{i_sample + 1}/{len(samples)}] Processing sample {sample.name}")
                output_path = self._generate_sample(sample, output_dir)
                if output_path is not None:
                    output_paths.append(output_path)
        return output_paths

    def _generate_sample(self, sample: MultiviewInferenceArgumentsWithInputPaths, output_dir: Path) -> str | None:
        log.debug(f"{sample.__class__.__name__}({sample})")
        output_path = output_dir / sample.name

        if self.rank0:
            output_dir.mkdir(parents=True, exist_ok=True)
            open(f"{output_path}.json", "w").write(sample.model_dump_json())
            log.info(f"Saved arguments to {output_path}.json")

            # Run text guardrail on the prompt
            if self.text_guardrail_runner is not None:
                log.info("Running guardrail check on prompt...")
                if not guardrail_presets.run_text_guardrail(str(sample.prompt), self.text_guardrail_runner):
                    message = f"Guardrail blocked text2world generation. Prompt: {str(sample.prompt)}"
                    log.critical(message)
                    if self.setup_args.keep_going:
                        return None
                    else:
                        raise Exception(message)
                else:
                    log.success("Passed guardrail on prompt")
            elif self.text_guardrail_runner is None:
                log.warning("Guardrail checks on prompt are disabled")

        # setup the input videos dict
        input_video_file_dict = {}
        fps = set()
        for key, value in sample.input_paths.items():
            if value is not None:
                input_video_file_dict[key] = value
                assert value  # make mypy happy
                fps.add(decord.VideoReader(value.as_posix()).get_avg_fps())

        if len(fps) != 1:
            raise ValueError(f"Control and video files have inconsistent FPS: {fps}")
        fps = fps.pop()
        desired_fps = sample.fps
        if fps % desired_fps != 0:
            raise ValueError(f"Video file fps {fps} is not evenly divisible by desired FPS {desired_fps}")
        fps_downsample_factor = int(fps / desired_fps)
        log.info(
            f"Files have FPS of {fps}, and desired FPS is {desired_fps}. Downsampling by factor of {fps_downsample_factor}"
        )

        # Calculate number of video frames to load
        assert self.pipe.config.model.config.state_t >= 1
        num_video_frames_per_view = (
            1 + (self.pipe.config.model.config.state_t - 1) * 4
        )  # tokenizer downsamples by 4x in temporal dimension

        augmentation_config = setup_config(
            resolution_hw=RESOLUTIONS[self.pipe.config.model.config.resolution],
            num_video_frames_per_view=num_video_frames_per_view,
            fps_downsample_factor=fps_downsample_factor,
        )
        if SMOKE:
            log.warning(f"Reducing the number of views to 1 for smoke test. Generated quality will be sub-optimal.")
            augmentation_config.camera_keys = augmentation_config.camera_keys[:1]
        assert sample.prompt is not None  # make mypy happy
        dataset = LocalMultiViewDataset(
            video_file_dicts=[input_video_file_dict],
            prompts=[sample.prompt],
            augmentation_config=augmentation_config,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        assert len(dataloader) == 1

        batch = next(iter(dataloader))
        batch[NUM_CONDITIONAL_FRAMES_KEY] = sample.num_input_frames
        video = self.pipe.generate_from_batch(
            batch,
            guidance=sample.guidance,
            seed=sample.seed,
            stack_mode=sample.stack_mode,
            num_steps=sample.num_steps,
        )
        if self.rank0:
            video = (1.0 + video[0].clamp(-1.0, 1.0)) / 2.0

            # run video guardrail on the video
            if self.video_guardrail_runner is not None:
                log.info("Running guardrail check on video...")
                frames = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)  # (T, H, W, C)
                processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                if processed_frames is None:
                    message = "Guardrail blocked video2world generation."
                    log.critical(message)
                    if self.setup_args.keep_going:
                        return None
                    else:
                        raise Exception(message)
                else:
                    log.success("Passed guardrail on generated video")
                # Convert processed frames back to tensor format
                processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
                video = processed_video.to(video.device, dtype=video.dtype)
            else:
                log.warning("Guardrail checks on video are disabled")

            save_img_or_video(video, str(output_path), fps=sample.fps)
            log.success(f"Saved video to {output_path}.mp4")
        return f"{output_path}.mp4"

    def _generate_from_config_dataloader(self, sample: MultiviewInferenceArguments, output_dir: Path) -> list[str]:
        output_path = output_dir / sample.name

        if self.rank0:
            output_path.mkdir(parents=True, exist_ok=True)
            open(f"{output_path}/config.json", "w").write(sample.model_dump_json())
            log.info(f"Saved arguments to {output_path}/config.json")

        dataloader = instantiate(self.pipe.config.dataloader_val)
        output_paths: list[str] = []
        # pyrefly: ignore  # no-matching-overload
        for batch in iter(dataloader):
            batch[NUM_CONDITIONAL_FRAMES_KEY] = sample.num_input_frames
            video = self.pipe.generate_from_batch(
                batch,
                guidance=sample.guidance,
                seed=sample.seed,
                stack_mode=sample.stack_mode,
                num_steps=sample.num_steps,
            )
            save_path = output_path / f"{batch['__key__'][0]}"
            if self.rank0:
                video = (1.0 + video[0].clamp(-1.0, 1.0)) / 2
                save_img_or_video(video, str(save_path), fps=sample.fps, quality=8)
                log.success(f"Saved video to {save_path}.mp4")
            output_paths.append(f"{save_path}.mp4")
        return output_paths
