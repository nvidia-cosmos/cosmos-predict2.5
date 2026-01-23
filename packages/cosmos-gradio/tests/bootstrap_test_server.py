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
import time

from cosmos_gradio.deployment_env import DeploymentEnv
from cosmos_gradio.gradio_app.gradio_server import launch_gradio_server
from cosmos_gradio.model_ipc.model_worker import ModelWorker
from loguru import logger as log
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class InferenceParameters(BaseModel):
    """Parameters for inference requests."""

    prompt: str = Field(..., min_length=1)

    """use_attribute_docstrings=True to use the docstrings as the description of the fields"""
    model_config = ConfigDict(use_attribute_docstrings=True)


default_request = {
    "prompt": "A blue monkey with a red hat",
}


class TestWorker(ModelWorker):
    def __init__(self):
        pass

    # pyrefly: ignore  # bad-override
    def infer(self, args: dict):
        prompt = args.get("prompt", "")
        log.info(f"TestWorker.infer running inference for prompt: {prompt}")
        rank = int(os.getenv("RANK", 0))
        if prompt == "test_exception" and rank == 0:
            log.info("TestWorker.infer raising a test exception")
            raise ValueError("This is a test exception")
        else:
            if prompt == "timeout":
                timeout = DeploymentEnv.get_instance().worker_timeout + 1
                log.info(f"TestWorker sleeping for longer than the worker timeout. Sleeping for {timeout} seconds")
                time.sleep(timeout)

            img = Image.new("RGB", (256, 256), color="red")
            output_dir = args.get("output_dir", "/mnt/pvc/gradio_output")
            out_file_name = os.path.join(output_dir, "output.png")

            if rank == 0:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                img.save(out_file_name)

            # the client will look for either 'videos' or 'images' in the status json
            # if neither is present, the client will look through the output directory for any files and display them
            return {"message": "created a red box", "prompt": prompt, "images": [out_file_name]}

    @staticmethod
    def validate_parameters(kwargs: dict):
        """Validate the inference parameters."""
        params = InferenceParameters(**kwargs)
        return params.model_dump(mode="json")


def create_worker():
    return TestWorker()


if __name__ == "__main__":
    global_env = DeploymentEnv()
    log.info(f"Starting Gradio app with deployment config: {global_env!s}")

    # based on the model name configuration could be different, strings in UI might be different
    launch_gradio_server(
        factory_module="tests.bootstrap_test_server",
        factory_function="create_worker",
        validator=TestWorker.validate_parameters,
        num_gpus=global_env.num_gpus,
        default_request=default_request,
        header="Cosmos Test Server",
        help_text="This is a test server for the Cosmos Gradio API",
    )
