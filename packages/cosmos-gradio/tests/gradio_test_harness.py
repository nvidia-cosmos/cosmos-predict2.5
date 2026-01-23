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
import json
import os
import subprocess
import time

import gradio_client.client as gradio_client
import requests
from loguru import logger as log


class GradioTestHarness:
    """
    Test harness for launching and testing the Gradio server.
    The test() function is launching a sub-process for the Gradio server and runs the client in the main process.
    This allows to run the muli-process test with client and server from one simple command.
    See test() function for details.
    """

    def __init__(self, server_module, env_vars=None, host="localhost", port=8080, timeout=300, check_interval=10):
        self.timeout = timeout
        self.check_interval = check_interval
        self.base_url = f"http://{host}:{port}"
        self.start_server(server_module, env_vars)

    def start_server(self, server_module, env_vars=None):
        module = importlib.import_module(server_module)
        bootstrapper_path = module.__file__

        # pyrefly: ignore  # bad-argument-type
        if bootstrapper_path is not None and not os.path.exists(bootstrapper_path):
            raise FileNotFoundError(f"gradio_bootstrapper.py not found at {bootstrapper_path}")

        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        try:
            response = requests.get(self.base_url, timeout=5)
            if response.status_code == 200:
                log.error("Server is already running")
                raise RuntimeError(
                    f"Server is already running at {self.base_url}. Failed to create a server with new workders."
                )
        except requests.exceptions.RequestException as e:
            log.debug(f"no existing server at {self.base_url}")

        log.info("-" * 120)
        log.info(f"Starting Gradio server test for {server_module} with {env_vars}")
        log.info("-" * 120)
        log.info(f"launching sub-process for Gradio server with {bootstrapper_path}")
        # pyrefly: ignore  # bad-assignment
        self.process = subprocess.Popen(
            ["python", "-u", str(bootstrapper_path)],
            env=env,
            text=True,
        )

        if not self.wait_for_server_ready():
            raise RuntimeError("Server failed to become ready")

    def wait_for_server_ready(self):
        log.info(f"Waiting for server to become ready at {self.base_url}")
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            # pyrefly: ignore  # missing-attribute
            if self.process.poll() is not None:
                # pyrefly: ignore  # missing-attribute
                stdout, stderr = self.process.communicate()
                log.error("Server process died unexpectedly")
                log.error(f"STDOUT: {stdout}")
                log.error(f"STDERR: {stderr}")
                raise RuntimeError("Server process died before becoming ready")

            try:
                response = requests.get(self.base_url, timeout=5)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    log.info(f"Server is ready! (took {elapsed:.2f} seconds)")
                    return True
            except requests.exceptions.RequestException as e:
                log.debug(f"Server not ready yet: {e}")

            time.sleep(self.check_interval)

        log.error(f"Server did not become ready within {self.timeout} seconds")
        return False

    def send_sample_ui(self, request_data: dict) -> bool:
        client = gradio_client.Client(self.base_url)
        log.info("-" * 120)
        request_text = json.dumps(request_data)
        log.info(f"input request: {json.dumps(request_data, indent=2)}")

        video, result = client.predict(request_text, api_name="/generate_video")

        if video is None:
            log.error(f"Error during inference: {json.dumps(result, indent=2)}")
            return False
        else:
            log.info(f"result: {json.dumps(result, indent=2)}")
            return True

    def send_sample_api(self, request_data: dict) -> bool:
        client = gradio_client.Client(self.base_url)
        log.info("-" * 120)
        log.info(f"input request: {json.dumps(request_data, indent=2)}")

        result = client.predict(request_data, api_name="/generate")

        if result["status"] != "success":
            log.error(f"Error during inference: {json.dumps(result, indent=2)}")
            return False
        else:
            log.info(f"result: {json.dumps(result, indent=2)}")
            return True

    def send_sample_api_default_request(self) -> bool:
        client = gradio_client.Client(self.base_url)
        log.info("-" * 120)
        result = client.predict(api_name="/generate_default_request")
        if result["status"] != "success":
            log.error(f"Error during inference: {json.dumps(result, indent=2)}")
            return False
        else:
            log.info(f"result: {json.dumps(result, indent=2)}")
            return True

    def shutdown_server(self):
        if self.process is None:
            log.warning("No process to shutdown")
            return

        log.info("-" * 120)
        log.info(f"Shutting down Gradio server (PID {self.process.pid})")
        log.info("-" * 120)
        try:
            self.process.terminate()

            try:
                self.process.wait(timeout=10)
                log.info("Server shutdown gracefully")
            except subprocess.TimeoutExpired:
                log.warning("Graceful shutdown timed out, forcing kill")
                self.process.kill()
                self.process.wait()
                log.info("Server killed forcefully")

        except Exception as e:
            log.error(f"Error during shutdown: {e}")
        finally:
            # Kill any remaining model_worker processes
            try:
                # List model_worker processes before killing
                list_result = subprocess.run(
                    ["pgrep", "-af", "model_worker"],
                    capture_output=True,
                    text=True,
                )
                if list_result.stdout.strip():
                    log.info(f"Found model_worker processes:\n{list_result.stdout.strip()}")
                else:
                    log.info("No model_worker processes found")

                log.info("Killing model_worker processes...")
                result = subprocess.run(
                    ["pkill", "-9", "-f", "model_worker"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    log.info("Successfully killed model_worker processes")
                elif result.returncode == 1:
                    log.info("No model_worker processes found")
                else:
                    log.warning(f"pkill returned unexpected code: {result.returncode}")
            except Exception as e:
                log.warning(f"Error while killing model_worker processes: {e}")

            # pyrefly: ignore  # bad-assignment
            self.process = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown_server()

    @staticmethod
    def test(server_module, env_vars, sample_request):
        """
        The main input parameter is the module starting the gradio server.
        Additionally we assume that the server is configured with envrionment variables, so the secondary input is the environment variables.
        The third input is the sample request data to send to the server.
        """

        ret = False
        with GradioTestHarness(server_module, env_vars) as harness:
            try:
                ret = harness.send_sample_api(sample_request)
            except Exception as e:
                log.error(f"Sample request failed: {e}")
                raise

        return ret
