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
import traceback
from abc import ABC, abstractmethod

import torch.distributed as dist
from loguru import logger as log

from cosmos_gradio.model_ipc.command_ipc import WorkerCommand, WorkerStatus


def create_worker_pipeline(factory_module, factory_function):
    log.info(f"initializing model using {factory_module}.{factory_function}")
    module = __import__(factory_module, fromlist=[factory_function])
    factory_function = getattr(module, factory_function)
    return factory_function()


class ModelWorker(ABC):
    """Base class for any model to be run in the server/worker setup.

    Any model we want to run in continuously running worker processes
    needs to implement the following methods:
    - __init__() that loads checkpoints before inference is called
    - infer(args: dict) method that processes inference requests

    """

    def __init__(self):
        pass

    @abstractmethod
    def infer(self, args: dict):
        """
        Perform inference with the given arguments.

        Args:
            args (dict): Dictionary containing inference parameters.

        Returns:
            dict: Dictionary containing inference results.
                - videos: List of generated video paths
                - prompt: Prompt used for inference
                - negative_prompt: Negative prompt used for inference
                - images: List of generated image paths
                - message: Message from the model
        """
        pass


def create_worker():
    """Create a sample worker for testing purposes.

    For any deployed model a factory function needs to be defined as input parameter
    for the GradioApp.
    """
    return ModelWorker()


def worker_main():
    """Main entry point for the worker process.

    A worker process in this context is created and managed by the ModelServer class.
    This function handles:
    - Initializing command and status communication channels
    - Creating the model pipeline using the factory function
    - Receiving and processing commands in a continuous loop
    - Handling errors and sending back status to the server


    Command Processing:
        The worker processes commands received from the model server:
        - 'inference': Calls pipeline.infer() with provided parameters
        - 'shutdown': Breaks from the main loop and performs cleanup
        - Unknown commands: Logs warning and sends error status

    Error Handling:
        All exceptions during command processing are caught, logged with
        full stack traces, and reported back to the server via status updates.

    Cleanup:
        On exit (normal or exception), the worker performs:
        - Command/status channel cleanup
        - Distributed process group cleanup (if initialized)
    """

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    log.info(f"Worker init {rank + 1}/{world_size}")

    worker_cmd = WorkerCommand(world_size)
    worker_status = WorkerStatus(world_size)

    factory_module = os.environ.get("FACTORY_MODULE")
    factory_function = os.environ.get("FACTORY_FUNCTION", "create_worker")

    try:
        pipeline = create_worker_pipeline(factory_module, factory_function)
        worker_status.signal_status(rank, "success", "Worker initialized successfully")

        while True:
            try:
                command_data = worker_cmd.wait_for_command(rank)
                command = command_data.get("command")
                params = command_data.get("params", {})

                log.info(f"Worker {rank} running {command=} with parameters: {params}")

                # Process commands
                if command == "inference":
                    result_json = pipeline.infer(params)
                    worker_status.signal_status(rank, "success", result_json)
                elif command == "shutdown":
                    log.info(f"Worker {rank} shutting down")
                    break
                else:
                    log.warning(f"Worker {rank} received unknown command: {command}")
                    worker_status.signal_status(rank, "error", f"Unknown command: {command}")

            except Exception as e:
                log.error(f"Worker {rank} error processing command: {e}")
                log.error(traceback.format_exc())
                worker_status.signal_status(rank, "error", str(e) + f"\n{traceback.format_exc()}")

    except Exception as e:
        log.error(f"Worker {rank} initialization error processing: {e}")
        log.error(traceback.format_exc())
        worker_status.signal_status(rank, "error", str(e) + f"\n{traceback.format_exc()}")
    finally:
        log.info(f"Worker {rank} shutting down...")

        if pipeline:
            del pipeline

        worker_cmd.cleanup()
        worker_status.cleanup()

        if dist.is_initialized():
            dist.destroy_process_group()
        log.info(f"Worker {rank} shutting down complete.")


if __name__ == "__main__":
    worker_main()
