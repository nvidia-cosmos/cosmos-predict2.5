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
import importlib
import os
import subprocess
import time

from loguru import logger as log

from cosmos_gradio.model_ipc.command_ipc import WorkerCommand, WorkerStatus


class ModelServer:
    """Manages multiple worker processes on a single node for model inference.

    This module provides the ModelServer class which manages multiple worker processes
    using torchrun for model inference. The server coordinates command
    distribution and status collection across workers.

    Key Components:
        - ModelServer: Main server class for managing distributed workers
        - Worker lifecylcle management via torchrun
        - Command IPC: sending inference requests to the workers
        - Status IPC: a compound status for all workers is collected

    The server operates by:
    1. Starting worker processes via torchrun
    2. Broadcasting inference commands to all workers
    3. Collecting status updates from workers

    Args:
        num_workers (int): Number of worker processes to spawn

    Usage:
        with ModelServer(num_gpus, factory_module, factory_function) as server:
            server.infer({"prompt": "A beautiful sunset", "seed": 42})

    """

    def __init__(self, num_gpus, factory_module, factory_function):
        """Initialize the model server and start worker processes.

        Creates IPC channels for worker communication, sets up the environment,
        and launches worker processes via torchrun. Blocks until all workers
        are ready and have signaled successful initialization.

        Args:
            num_gpus (int): Number of worker processes to create (default: 2)
            factory_module (str): Module containing the factory function
            factory_function (str): Function to create the model instance

        Raises:
            Exception: If worker startup fails or workers don't signal readiness
        """

        self.num_workers = num_gpus
        self.process = None
        self.worker_command = WorkerCommand(self.num_workers)
        self.worker_status = WorkerStatus(self.num_workers)
        self.start_workers(num_gpus, factory_module, factory_function)

    def start_workers(self, num_gpus, factory_module, factory_function):
        """Start worker processes using torchrun.

        This method performs the complete worker startup sequence:
        1. Cleans up any existing worker communication channels
        2. Constructs the torchrun command with appropriate parameters
        3. Launches the subprocess with proper environment settings
        . Waits for all workers to signal successful initialization

        Raises:
            Exception: If the subprocess fails to start or workers don't initialize
        """
        # test load worker and fail early on incorrect parameters
        log.info(f"initializing model using {factory_module}.{factory_function}")
        module = __import__(factory_module, fromlist=[factory_function])
        _ = getattr(module, factory_function)
        self.env = os.environ.copy()
        self.env["FACTORY_MODULE"] = factory_module
        self.env["FACTORY_FUNCTION"] = factory_function
        self.env["NUM_GPUS"] = str(num_gpus)

        # don't rely on CWD to create a file path for the worker
        module = importlib.import_module("cosmos_gradio.model_ipc.model_worker")
        module_path = module.__file__
        log.info(f"Module loaded from: {module_path}")

        # clean-up previous runs
        self.worker_command.cleanup()
        self.worker_status.cleanup()

        log.info(f"Starting {self.num_workers} worker processes with torchrun")

        torchrun_cmd = [
            "torchrun",
            f"--nproc_per_node={self.num_workers}",
            "--nnodes=1",
            "--node_rank=0",
            module_path,
        ]

        log.info(f"Running command: {' '.join(torchrun_cmd)}")

        # Launch worker processes
        try:
            self.process = subprocess.Popen(
                torchrun_cmd,
                env=self.env,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            log.info("waiting for workers to start...")
            self.worker_status.wait_for_status()

        except Exception as e:
            log.error(f"Error starting workers: {e}")
            self.stop_workers()
            raise e

    def stop_workers(self):
        """Gracefully shutdown all worker processes.

        Performs orderly shutdown by:
        1. Broadcasting shutdown command to all workers
        2. Terminate torchrun process if still running

        """
        # best effort to gracefully shutdown workers
        # if worker is busy inferencing or hanging then just move on
        # in the end terminating torchrun will signal SIGTERM to all workers
        self.worker_command.broadcast("shutdown", {})

        # Wait a bit for graceful shutdown
        count = 0
        while count < 3:
            count += 1
            log.info(f"Waiting for workers to shutdown... {count}")
            time.sleep(10)

        if self.process is None:
            log.info("torchrun already shut down.")
            return

        # we sent a shutdown command to all workers,
        # now we poll the torchrun process as this the only process we've started from this process
        if self.process.poll() is None:
            log.info("Terminating torchrun process...")
            self.process.terminate()
            self.process.wait(timeout=10)

            if self.process.poll() is None:
                log.warning("torchrun did not terminate")

        log.info("torchrun process terminated successfully")
        self.process = None

    def infer(self, args: dict):
        """Execute inference across all worker processes.

        Broadcasts inference parameters to all workers and waits for completion.
        Handles error propagation.
        Args:
            args (dict): Inference parameters to send to workers.

        Raises:
            Exception: If inference fails on any worker or communication errors occur
        """

        try:
            self.worker_command.broadcast("inference", args)

            log.info("Waiting for tasks to complete...")
            status = self.worker_status.wait_for_status()
            return status

        except Exception as e:
            # only debug here. client need to handle errors
            log.debug(f"Error during inference: {e}")
            raise e
        finally:
            # just in case a worker was hanging and didn't clean up
            self.worker_command.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection."""
        self.cleanup()

    def __enter__(self):
        """Enter the context manager.

        Returns:
            ModelServer: Self reference for context manager usage
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and perform cleanup.
        Ensures proper cleanup regardless of whether an exception occurred.
        """
        log.info("Exiting ModelServer context")
        self.cleanup()

    def cleanup(self):
        """Clean up server resources and shutdown workers."""
        log.info("Cleaning up ModelServer")
        self.stop_workers()
        self.worker_command.cleanup()
        self.worker_status.cleanup()
