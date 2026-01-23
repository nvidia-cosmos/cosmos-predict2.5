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
import time
from typing import Any

import pydantic
from loguru import logger as log


class CommandData(pydantic.BaseModel):
    command: str
    params: dict[str, Any]
    request_id: int

    def serialize(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

    @classmethod
    def deserialize(cls, filepath: str):
        with open(filepath) as f:
            return cls.model_validate_json(f.read())


class WorkerCommand:
    """wrapper around file based IPC command"""

    def __init__(self, num_workers: int):
        self.num_workers = num_workers

    def cleanup(self):
        for rank in range(self.num_workers):
            for file_path in [f"/tmp/worker_{rank}_commands.json"]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    def broadcast(self, task_name: str, task_params: dict[str, Any], request_id: int):
        """Broadcast non-blocking a task to all workers."""
        log.debug(f"Broadcasting task '{task_name}' to all workers...")
        command_data = CommandData(command=task_name, params=task_params, request_id=request_id)

        for rank in range(self.num_workers):
            command_file = f"/tmp/worker_{rank}_commands.json"
            command_data.serialize(command_file)
            log.debug(f"Sent command '{command_data.command}' to worker {rank}")

    def wait_for_command(self, rank: int) -> CommandData:
        """wait blocking for a command from the worker.

        This is an infinite blocking call by design. We want to infinitely wait until typically a user is sending
        a request to the worker.
        """
        command_file = f"/tmp/worker_{rank}_commands.json"
        log.debug(f"worker {rank}: Waiting for command file {command_file}")
        while not os.path.exists(command_file):
            time.sleep(0.5)

        try:
            command_data = CommandData.deserialize(command_file)
            os.remove(command_file)  # Remove command file after reading
            return command_data
        except Exception as e:
            log.error(f"Failed to read command file for worker {rank}: {e}")
            raise e


class WorkerException(Exception):
    def __init__(self, rank, status, result_json: dict[str, Any] = {}):
        super().__init__("worker exception")
        self.rank = rank
        self.status = status
        self.results = result_json

    def __str__(self):
        rank = self.rank
        results = self.results
        return f"{super().__str__()} {rank=}: {self.status}, {results=}"


class StatusData(pydantic.BaseModel):
    rank: int
    status: str
    request_id: int
    result: dict[str, Any]

    def serialize(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

    @classmethod
    def deserialize(cls, filepath: str):
        with open(filepath) as f:
            return cls.model_validate_json(f.read())


class WorkerStatus:
    """wrapper around file based IPC status

    KEY CONCEPT:
    any exception from a worker needs to be serialized and sent to the server.
    In the server process, we deserialize the exception and raise it again as WorkerException.
    This simplifies the flow as try blocks can be used as usual.

    On this protocol level, there are only two possible statuses: success and error.
    Error means model or protocal layer threw exception.
    So for any real error the model needs to throw an exception.
    The model with return its result in the result field.
    Any trivial exceptions can be caught by the model itself and returned with respective result.
    """

    STATUS_SUCCESS = "success"

    def __init__(self, num_workers: int):
        self.num_workers = num_workers

    def cleanup(self):
        for rank in range(self.num_workers):
            for file_path in [f"/tmp/worker_{rank}_status.json"]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    def signal_status(self, rank: int, status: str, request_id: int, result: dict[str, Any] = {}) -> None:
        """signal individual worker status per rank

        Args:
            rank (int): The rank of the worker
            status (str): The status of the worker is either "success" or an error string
            results_json (dict[str, Any]): The result json of the worker/model. Model can place arbitrary data here.
        """
        status_file = f"/tmp/worker_{rank}_status.json"

        status_data = StatusData(rank=rank, status=status, request_id=request_id, result=result)
        log.debug(f"worker {rank} status: {status_data}")
        status_data.serialize(status_file)

    def _get_worker_status(self, rank: int, expected_request_id: int, timeout: int = 1800) -> StatusData:
        status_file = f"/tmp/worker_{rank}_status.json"
        start_time = time.time()

        while not os.path.exists(status_file):
            if time.time() - start_time > timeout:
                # avoid race condition between server/worker during shutdown
                if os.path.exists(status_file):
                    os.remove(status_file)
                log.error(f"Worker {rank} timeout: {timeout} seconds elapsed while waiting for status")
                return StatusData(rank=rank, status="timeout", request_id=expected_request_id, result={})
            time.sleep(0.5)

        try:
            status_data = StatusData.deserialize(status_file)
            if status_data.request_id != expected_request_id:
                log.error(
                    f"Worker {rank} status: {status_data.status}, expected request id: {expected_request_id}, actual request id: {status_data.request_id}"
                )
                status_data.status = "unexpected request id"
                return status_data
            # remove status file so we can do a blocking wait for next status
            log.debug(f"got status worker {rank}. removing status file {status_file}")
            os.remove(status_file)

            assert os.path.exists(status_file) is False, "status file should be removed after processing"
            return status_data

        except Exception:
            log.error(f"Failed to read status file for worker {rank}")
            return StatusData(rank=rank, status="unknown", result={}, request_id=expected_request_id)

    def wait_for_status(self, expected_request_id: int = 0, timeout: int = 1800) -> StatusData:
        """blocking call to wait for completion of all workers

        This functions waits for all workers to signal their status.
        Upon failure of any worker, it raises a WorkerException with a compound status dictionary.
        """
        log.debug(f"Waiting for status from all workers...{expected_request_id=} {timeout=}")
        statuses = [self._get_worker_status(rank, expected_request_id, timeout) for rank in range(self.num_workers)]
        # Collect statuses from all workers, ensure status file is removed after reading

        for status_data in statuses:
            if status_data.status != self.STATUS_SUCCESS:
                log.debug(f"{status_data=}")
                raise WorkerException(status_data.rank, status_data.status, status_data.result)

        log.debug(f"All workers reported success and result json: {statuses[0]}")

        return statuses[0]
