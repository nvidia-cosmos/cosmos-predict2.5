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
import gc
from pathlib import Path

import torch
from cosmos_gradio.deployment_env import DeploymentEnv
from cosmos_gradio.gradio_app.gradio_app import GradioApp
from cosmos_gradio.gradio_app.gradio_ui import create_gradio_UI

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2.config import InferenceArguments, SetupArguments
from cosmos_predict2.gradio.model_config import ModelConfig
from cosmos_predict2.gradio.multiview_worker import Multiview_Worker
from cosmos_predict2.gradio.video2world_worker import Video2World_Worker
from cosmos_predict2.multiview_config import MultiviewInferenceArgumentsWithInputPaths


def create_video2world():
    log.info("Creating predict pipeline")
    global_env = DeploymentEnv()
    setup_args = SetupArguments(
        context_parallel_size=global_env.num_gpus,
        output_dir=Path("outputs"),  # dummy parameter, we want to save videos in per inference folders
        model="2B/pre-trained",
        keep_going=True,
        disable_guardrails=global_env.disable_guardrails,
    )

    pipeline = Video2World_Worker(setup_args=setup_args)
    gc.collect()
    torch.cuda.empty_cache()

    return pipeline


def create_multiview():
    log.info("Creating predict multiview pipeline")
    global_env = DeploymentEnv()
    assert global_env.num_gpus == 8, "Multiview currently requires 8 GPUs"
    pipeline = Multiview_Worker(
        num_gpus=global_env.num_gpus,
        disable_guardrails=global_env.disable_guardrails,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return pipeline


def validate_v2w(kwargs):
    inference_args = InferenceArguments(**kwargs)
    return inference_args.model_dump(mode="json")


def validate_multiview(kwargs):
    inference_args = MultiviewInferenceArgumentsWithInputPaths(**kwargs)
    return inference_args.model_dump(mode="json")


if __name__ == "__main__":
    model_cfg = ModelConfig()
    global_env = DeploymentEnv()

    log.info(f"Starting Gradio app with deployment config: {global_env!s}")

    # configure server to use the correct worker in the worker procs
    factory_module = {
        "video2world": "cosmos_predict2.gradio.gradio_bootstrapper",
        "multiview": "cosmos_predict2.gradio.gradio_bootstrapper",
    }

    factory_function = {
        "video2world": "create_video2world",
        "multiview": "create_multiview",
    }

    validators = {
        "video2world": validate_v2w,
        "multiview": validate_multiview,
    }

    app = GradioApp(
        num_gpus=global_env.num_gpus,
        validator=validators[global_env.model_name],
        factory_module=factory_module[global_env.model_name],
        factory_function=factory_function[global_env.model_name],
        output_dir=global_env.output_dir,
    )

    interface = create_gradio_UI(
        app.infer,
        header=model_cfg.header[global_env.model_name],
        default_request=model_cfg.default_request[global_env.model_name],
        help_text=model_cfg.help_text[global_env.model_name],
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
        allowed_paths=global_env.allowed_paths,
    )
