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

import json
import os

from cosmos_gradio.deployment_env import DeploymentEnv
from cosmos_gradio.model_ipc.model_worker import ModelWorker
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class InferenceParameters(BaseModel):
    """Parameters for inference requests."""

    prompt: str = Field(..., min_length=1)
    """Text prompt for generation"""
    num_steps: int = Field(..., gt=0)
    """Number of inference steps"""
    input_image: str | None = None
    """Path to the input image"""

    """use_attribute_docstrings=True to use the docstrings as the description of the fields"""
    model_config = ConfigDict(use_attribute_docstrings=True)


class SampleWorker(ModelWorker):
    def __init__(self, num_gpus, model_name):
        pass

    # pyrefly: ignore  # bad-override
    def infer(self, args: dict):
        prompt = args.get("prompt", "")

        img = Image.new("RGB", (256, 256), color="red")
        output_dir = args.get("output_dir", "/mnt/pvc/gradio_output")
        out_file_name = os.path.join(output_dir, "output.png")

        rank = int(os.getenv("RANK", 0))
        if rank == 0:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            img.save(out_file_name)

        # the client will look for either 'videos' or 'images' in the status json
        # if neither is present, the client will look through the output directory for any files and display them
        return {"message": "created a red box", "prompt": prompt, "images": [out_file_name]}

    @staticmethod
    def get_parameters_schema():
        """Return the JSON schema for the inference parameters."""
        return json.dumps(InferenceParameters.model_json_schema(), indent=2)

    @staticmethod
    def validate_parameters(kwargs: dict):
        """Validate the inference parameters."""
        params = InferenceParameters(**kwargs)
        return params.model_dump(mode="json")


def create_worker():
    """Factory function to create sample pipeline."""
    cfg = DeploymentEnv()

    pipeline = SampleWorker(
        num_gpus=cfg.num_gpus,
        model_name=cfg.model_name,
    )

    return pipeline
