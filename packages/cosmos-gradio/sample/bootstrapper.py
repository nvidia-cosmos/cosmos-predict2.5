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

from cosmos_gradio.deployment_env import DeploymentEnv
from cosmos_gradio.gradio_app.gradio_app import GradioApp
from cosmos_gradio.gradio_app.gradio_ui import create_gradio_UI
from loguru import logger as log

default_request = json.dumps(
    {
        "prompt": "A blue monkey with a red hat",
    },
    indent=2,
)


def validate(args: dict) -> dict:
    valid_params = ["prompt"]
    for key, value in args.items():
        if key not in valid_params:
            raise ValueError(f"Invalid parameter: {key}")
        if key == "prompt":
            if value.strip() == "":
                raise ValueError("Prompt cannot be empty")
    return args


if __name__ == "__main__":
    global_env = DeploymentEnv()
    log.info(f"Starting Gradio app with deployment config: {global_env!s}")

    # based on the model name configuration could be different, strings in UI might be different
    if global_env.model_name == "sample":
        factory_module = "sample.sample_worker"
        factory_function = "create_worker"
    else:
        raise ValueError(f"Model name {global_env.model_name} not supported")

    # the gradio app needs a validator for parameter validaiton w/o a server round-trip
    # and the factory method so that worker procs can create model instances
    app = GradioApp(
        num_gpus=global_env.num_gpus,
        validator=validate,
        factory_module=factory_module,
        factory_function=factory_function,
        output_dir=global_env.output_dir,
    )

    interface = create_gradio_UI(
        infer_func=app.infer,
        header="Cosmos Sample UI",
        default_request=default_request,
        help_text="This is a sample UI for the Cosmos model. It is used to test the model and the Gradio app.",
        uploads_dir=global_env.uploads_dir,
        output_dir=global_env.output_dir,
        log_file=global_env.log_file,
    )

    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        max_file_size="500MB",
        allowed_paths=[global_env.output_dir, global_env.uploads_dir],
    )
