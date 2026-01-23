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

from cosmos_gradio.gradio_app.util import get_output_folder
from cosmos_gradio.model_ipc.command_ipc import StatusData, WorkerException
from cosmos_gradio.model_ipc.model_server import ModelServer


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
        default_request: dict,
    ):
        # Print the full path of this module
        module_path = __file__
        log.info(f"GradioApp module full path: {module_path}")

        self.validator = validator
        self.pipeline = ModelServer.create_server(num_gpus, factory_module, factory_function)
        self.output_dir = output_dir
        self.default_request = default_request

    def generate_video(
        self,
        request_text,
    ) -> tuple[str | None, str]:
        """
        generation function for the gradio UI with input/output matching UI elements.
        Note: at this point we catch exceptions and return a status message to the UI.
        """
        try:
            log.info(f"Model parameters: {request_text}")
            args_dict = json.loads(request_text)
        except json.JSONDecodeError as e:
            return (
                None,
                f"Error parsing request JSON: {e}\nPlease ensure your request is valid JSON.",
            )
        output_folder = get_output_folder(self.output_dir)
        status = self._infer_json(args_dict, output_folder)

        output_file = None
        status_message = f"{status.model_dump_json(indent=4)}"
        output_files = status.result.get("videos", None)
        if output_files is None:
            output_files = status.result.get("images", None)
        if output_files and len(output_files) > 0:
            output_file = output_files[0]

        return output_file, status_message

    def generate(self, request_dict: dict) -> dict:
        output_folder = get_output_folder(self.output_dir)
        status_data = self._infer_json(request_dict, output_folder)
        return status_data.model_dump()

    def generate_default_request(self) -> dict:
        status_data = self._infer_json(self.default_request, get_output_folder(self.output_dir))
        return status_data.model_dump()

    def _infer_json(self, request_dict: dict, output_folder: str) -> StatusData:
        try:
            log.info(f"Model parameters: {request_dict}")

            args_dict = self.validator(request_dict)
            args_dict["output_dir"] = output_folder

            return self.pipeline.infer(args_dict)
        except WorkerException as e:
            log.error(f"Error during inference: {e}")

            if e.status == "timeout":
                # we don't know what happened to the workers
                # out time-out might just be to short
                # or worker died or hangs
                # time to attempt restart the workers
                log.error("Workers timed out. Restarting workers...")
                self.pipeline.stop_workers()
                self.pipeline.start_workers()
                return StatusData(
                    status="Worker not repsonsive. Restarted all workers.", result={}, rank=0, request_id=0
                )
            else:
                return StatusData(status=str(e), result={}, rank=0, request_id=0)
        except Exception as e:
            log.error(f"Error during inference: {e}")
            return StatusData(status=str(e), result={}, rank=0, request_id=0)
