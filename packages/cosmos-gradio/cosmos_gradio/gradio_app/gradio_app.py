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

from loguru import logger as log
from pyparsing import Callable

from cosmos_gradio.gradio_app.util import get_output_folder, get_outputs
from cosmos_gradio.model_ipc.model_server import ModelServer
from cosmos_gradio.model_ipc.model_worker import create_worker_pipeline


class GradioApp:
    """
    The GradioApp is interfacing with the gradio UI:
    * creating the model, distributed or in process model for single GPU inference
    * processing the raw json input before calling the model
    * processing the output files and creating a status message
    """

    def __init__(
        self,
        num_gpus: int,
        validator: Callable[[dict], dict],
        factory_module: str,
        factory_function: str,
        output_dir: str,
    ):
        self.validator = validator
        if num_gpus == 1:
            self.pipeline = create_worker_pipeline(factory_module, factory_function)
        else:
            self.pipeline = ModelServer(num_gpus, factory_module, factory_function)
        self.output_dir = output_dir

    def infer(
        self,
        request_text,
    ):
        output_folder = get_output_folder(self.output_dir)

        try:
            args_dict_unvalidated = json.loads(request_text)
        except json.JSONDecodeError as e:
            return (
                None,
                f"Error parsing request JSON: {e}\nPlease ensure your request is valid JSON.",
            )

        try:
            log.info(f"Model parameters: {json.dumps(args_dict_unvalidated, indent=4)}")

            args_dict = self.validator(args_dict_unvalidated)
            args_dict["output_dir"] = output_folder

            status = self.pipeline.infer(args_dict)

        except Exception as e:
            log.error(f"Error during inference: {e}")
            return None, f"Error: {e}"

        output_file, status_message = get_outputs(output_folder)

        if status:
            status_message += f"\nResult json: {json.dumps(status, indent=4)}"

        return output_file, status_message
