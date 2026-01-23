#!/usr/bin/env -S uv run --script
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

# https://docs.astral.sh/uv/guides/scripts/#using-a-shebang-to-create-an-executable-file
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "tyro",
#   "pydantic",
#   "torch==2.10.0",
#   "transformers==4.57.1",
#   "cosmos-guardrail @ git+https://github.com/codeJRV/cosmos-guardrail",
#   "diffusers @ git+https://github.com/huggingface/diffusers.git",
# ]
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cu128", marker = "platform_machine != 'aarch64'" },
#   { index = "pytorch-cu130", marker = "platform_machine == 'aarch64'" },
# ]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import pydantic
import torch
import tyro
from diffusers import Cosmos2_5_PredictBasePipeline
from diffusers.utils import export_to_video, load_image, load_video

DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    output_path: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """
    Where to save the outputs
    """

    model_id: str | None = "nvidia/Cosmos-Predict2.5-2B"
    """
    Which model repository to use. Use "nvidia/Cosmos-Predict2.5-14B" for 14B variant
    """

    revision: str | None = "diffusers/base/post-trained"
    """
    Which variant of the model to use. Defaults to the base post-trained model. Here is a list of valid variants are:
    - diffusers/base/pre-trained
    - diffusers/base/post-trained

    """

    input_path: Annotated[Path | None, tyro.conf.arg(aliases=("-i",))] = None
    """
    Path to the conditioning media (image or video) or to a JSON config file.

    If the path ends with .json the script will load prompt/media values from that file, this JSON file is expected to be the same file used for the example scripts.

    CLI arguments override JSON values when both are provided (overriding the conditioning input image or video must be done with `--override_visual_input <path>`).
    """

    override_visual_input: Path | None = None
    """
    Override media (image/video) path when `--input_path` points to a JSON asset.

    Useful when reusing JSON configs but swapping in a different conditioning file.
    """

    prompt: str | None = None
    """
    A string describing the prompt
    """

    prompt_path: Path | None = None
    """
    A text file describing the prompt, if provided will ignore prompt
    """

    num_output_frames: int = 93
    """
    Number of output frames. Use 1 for "2Image" mode and 93 for "2Video" mode.
    """

    negative_prompt: str | None = DEFAULT_NEGATIVE_PROMPT
    """
    Negative prompt text to use
    """

    negative_prompt_path: Path | None = None
    """
    Negative prompt file to use, if provided negative_prompt will be ignored. 
    """

    seed: int | None = None
    """
    Seed for generation, if not provided no seed will be set
    """

    num_steps: int = 36
    """
    Number of steps to use
    """

    device: str = "cuda"
    """
    device to use
    """

    device_map: str | None = None
    """
    device mapping to use, see: https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    """


def main(args: Args):
    prompt = None
    negative_prompt = None
    resolved_input_path = None
    image = None
    video = None
    if args.input_path and args.input_path.suffix.lower() == ".json":
        config = json.load(args.input_path.open())

        root_dir = args.input_path.parent
        if config.get("input_path") is not None:
            resolved_input_path = (root_dir / config["input_path"]).absolute()

        if config.get("prompt_path") is not None:
            prompt = (root_dir / config["prompt_path"]).open().read()
        elif config.get("prompt") is not None:
            prompt = config["prompt"]

        if config.get("negative_prompt") is not None:
            negative_prompt = config["negative_prompt"]
    else:
        resolved_input_path = args.input_path

    if args.override_visual_input is not None:
        resolved_input_path = args.override_visual_input

    assert args.num_output_frames > 0
    output_path = args.output_path
    if args.num_output_frames == 1:
        if output_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            print(f"WARN: outputting to {str(output_path)}", file=sys.stderr)
            output_path = Path(f"{output_path}.jpg")
    else:
        if output_path.suffix.lower() != ".mp4":
            print(f"WARN: outputting to {str(output_path)}", file=sys.stderr)
            output_path = Path(f"{output_path}.mp4")

    if resolved_input_path is not None:
        suffix = resolved_input_path.suffix.lower()
        if suffix == ".mp4":
            video = load_video(str(resolved_input_path))
        elif suffix in (".jpg", ".jpeg", ".png"):
            image = load_image(str(resolved_input_path))
        else:
            print(
                f"Unsupported input file extension '{resolved_input_path.suffix}'. "
                "Use .mp4 for video or .jpg/.jpeg/.png for images.",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.prompt_path:
        prompt = args.prompt_path.open().read()
    elif args.prompt:
        prompt = args.prompt

    if args.negative_prompt_path:
        negative_prompt = args.negative_prompt_path.open().read()
    elif args.negative_prompt is not None:
        negative_prompt = args.negative_prompt

    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        args.model_id,
        revision=args.revision,
        device_map=args.device_map,
        torch_dtype=torch.bfloat16,
    )
    if not args.device_map:
        pipe = pipe.to(args.device)

    frames = pipe(
        image=image,
        video=video,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=args.num_output_frames,
        num_inference_steps=args.num_steps,
        generator=torch.Generator().manual_seed(args.seed) if args.seed is not None else None,
    ).frames[0]  # NOTE: batch_size == 1

    base_dir = output_path.absolute().parent
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    if args.num_output_frames > 1:
        export_to_video(frames, str(output_path), fps=16)
    else:
        frames[0].save(str(output_path))


if __name__ == "__main__":
    args = tyro.cli(
        Args,
        description=__doc__,
        config=(tyro.conf.OmitArgPrefixes,),
    )
    main(args)
