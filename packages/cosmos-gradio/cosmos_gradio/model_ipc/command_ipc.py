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

from loguru import logger as log


class WorkerCommand:
    """wrapper around file based IPC command"""

    def __init__(self, num_workers: int):
        self.num_workers = num_workers

    def cleanup(self):
        for rank in range(self.num_workers):
            for file_path in [f"/tmp/worker_{rank}_commands.json"]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    def _send_command_to_worker(self, rank: int, command: str, params: dict[str, Any] | None = None):
        command_file = f"/tmp/worker_{rank}_commands.json"
        command_data = {"command": command, "params": params or {}}

        with open(command_file, "w") as f:
            json.dump(command_data, f)

        log.debug(f"Sent command '{command}' to worker {rank}")

    def broadcast(self, task_name: str, task_params: dict[str, Any]):
        """Broadcast non-blocking a task to all workers."""
        log.debug(f"Broadcasting task '{task_name}' to all workers...")

        for rank in range(self.num_workers):
            self._send_command_to_worker(rank, task_name, task_params)

    def wait_for_command(self, rank: int) -> dict[str, Any] | None:
        """wait blocking for a command from the worker.

        This is an infinite blocking call by design. We want to infinitely wait until typically a user is sending
        a request to the worker.
        """
        command_file = f"/tmp/worker_{rank}_commands.json"
        log.debug(f"worker {rank}: Waiting for command file {command_file}")
        while not os.path.exists(command_file):
            time.sleep(0.5)

        try:
            with open(command_file) as f:
                command_data = json.load(f)
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


class WorkerStatus:
    """wrapper around file based IPC status"""

    STATUS_SUCCESS = "success"

    def __init__(self, num_workers: int):
        self.num_workers = num_workers

    def cleanup(self):
        for rank in range(self.num_workers):
            for file_path in [f"/tmp/worker_{rank}_status.json"]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    def signal_status(self, rank: int, status: str = STATUS_SUCCESS, results_json: dict[str, Any] = {}) -> None:
        """signal individual worker status per rank

        Args:
            rank (int): The rank of the worker
            status (str): The status of the worker is either "success" or an error string
            results_json (dict[str, Any]): The result json of the worker/model. Model can place arbitrary data here.
        """
        status_file = f"/tmp/worker_{rank}_status.json"

        log.debug(f"worker {rank} status: {status}, result: {results_json}")
        with open(status_file, "w") as f:
            json.dump(
                {"rank": rank, "status": status, "result": results_json},
                f,
            )

    def _get_worker_status(self, rank: int, timeout: int = 1800) -> dict[str, Any]:
        status_file = f"/tmp/worker_{rank}_status.json"
        start_time = time.time()

        while not os.path.exists(status_file):
            if time.time() - start_time > timeout:
                # avoid race condition between server/worker during shutdown
                if os.path.exists(status_file):
                    os.remove(status_file)
                return {"status": "timeout", "rank": rank}
            time.sleep(0.5)

        try:
            with open(status_file) as f:
                status = json.load(f)

            # remove status file so we can do a blocking wait for next status
            log.debug(f"Worker {rank} removing status file {status_file}")
            os.remove(status_file)

            assert os.path.exists(status_file) is False, "status file should be removed after processing"
            return status

        except Exception:
            log.error(f"Failed to read status file for worker {rank}")
            return {"status": "unknown", "rank": rank}

    def wait_for_status(self, timeout: int = 1800) -> bool:
        statuses = {}
        """blocking call to wait for completion of all workers

            This functions waits for all workers to signal their status.
            Upon failure of any worker, it raises a WorkerException with a compound status dictionary.
        """

        # Collect statuses from all workers, ensure status file is removed after reading
        for rank in range(self.num_workers):
            statuses[rank] = self._get_worker_status(rank, timeout)

        for rank, worker_status in statuses.items():
            if worker_status.get("status") != self.STATUS_SUCCESS:
                status = worker_status.get("status")
                res = worker_status.get("result", "no result json")
                log.debug(status, res)
                raise WorkerException(rank, status, res)

        log.debug(f"All workers reported success and result json: {statuses[0]}")

        return statuses[0]
