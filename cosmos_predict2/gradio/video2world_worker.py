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

from pathlib import Path

from cosmos_predict2.config import (
    InferenceArguments,
    SetupArguments,
)
from cosmos_predict2.inference import Inference

setup_args_path = Path("/tmp/video2world_setup_args.json")


def save_setup_args(setup_args: SetupArguments):
    with open(setup_args_path, "w") as f:
        f.write(setup_args.model_dump_json(indent=2))


def create_worker():
    setup_args = SetupArguments.model_validate_json(setup_args_path.read_text())
    pipeline = Video2World_Worker(setup_args=setup_args)
    return pipeline


class Video2World_Worker:
    def __init__(
        self,
        setup_args: SetupArguments,
    ):
        self.pipe = Inference(setup_args)

    def infer(self, args: dict):
        """
        Adjust inputs from gradio to InferenceArguments.
        Adjust output from Inference.generate to gradio.

        Args:
            args (dict): Dictionary containing InferenceArguments attributes

        Returns:
            dict: Dictionary containing:
                - videos: List of generated video paths
        """

        output_dir = args.pop("output_dir", "outputs")

        args_json = args.pop("args_json_file", None)
        if args_json is not None:
            inference_args = InferenceArguments.from_files([Path(args_json)])
        else:
            inference_args = [InferenceArguments(**args)]

        output_videos = self.pipe.generate(inference_args, Path(output_dir))

        return {
            "videos": output_videos,
        }
