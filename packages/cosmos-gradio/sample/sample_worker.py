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

from cosmos_gradio.deployment_env import DeploymentEnv
from cosmos_gradio.model_ipc.model_worker import ModelWorker
from PIL import Image


class SampleWorker(ModelWorker):
    def __init__(self, num_gpus, model_name):
        pass

    def infer(self, args: dict):
        output_dir = args.get("output_dir", "/mnt/pvc/gradio_output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prompt = args.get("prompt", "")

        img = Image.new("RGB", (256, 256), color="red")
        out_file_name = os.path.join(output_dir, "output.png")
        img.save(out_file_name)

        return {"message": "created a red box", "prompt": prompt, "images": [out_file_name]}


def create_worker():
    """Factory function to create sample pipeline."""
    cfg = DeploymentEnv()

    pipeline = SampleWorker(
        num_gpus=cfg.num_gpus,
        model_name=cfg.model_name,
    )

    return pipeline
