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


from cosmos_gradio.deployment_env import DeploymentEnv
from cosmos_gradio.gradio_app.gradio_server import launch_gradio_server
from loguru import logger as log
from sample.sample_worker import SampleWorker

default_request = {
    "prompt": "A blue monkey with a red hat",
    "num_steps": 20,
}


if __name__ == "__main__":
    global_env = DeploymentEnv.get_instance()
    log.info(f"Starting Gradio app with deployment config: {global_env!s}")

    # based on the model name configuration could be different, strings in UI might be different
    if global_env.model_name == "sample":
        factory_module = "sample.sample_worker"
        factory_function = "create_worker"
    else:
        raise ValueError(f"Model name {global_env.model_name} not supported")

    launch_gradio_server(
        factory_module=factory_module,
        factory_function=factory_function,
        validator=SampleWorker.validate_parameters,
        num_gpus=global_env.num_gpus,
        output_dir=global_env.output_dir,
        header="Cosmos Sample UI",
        default_request=default_request,
        help_text=f"```json\n{SampleWorker.get_parameters_schema()}\n```",
        uploads_dir=global_env.uploads_dir,
        log_file=global_env.log_file,
        allowed_paths=global_env.allowed_paths,
    )
