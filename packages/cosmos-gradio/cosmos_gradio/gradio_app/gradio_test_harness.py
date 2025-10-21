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


class TestHarness:
    """
    Test harness for launching and testing the Gradio server.
    The main input parameter is the module starting the gradio server.
    Additionally we assume that the server is configured with envrionment variables, so the secondary input is the environment variables.
    """

    def __init__(self, host="localhost", port=8080, timeout=300, check_interval=10):
        self.timeout = timeout
        self.check_interval = check_interval
        self.base_url = f"http://{host}:{port}"
        self.process = None

    def start_server(self, server_module, env_vars=None):
        module = importlib.import_module(server_module)
        bootstrapper_path = module.__file__

        # pyrefly: ignore  # bad-argument-type
        if not os.path.exists(bootstrapper_path):
            raise FileNotFoundError(f"gradio_bootstrapper.py not found at {bootstrapper_path}")

        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        log.info(f"launching sub-process for Gradio server with {bootstrapper_path}")
        log.info(f"Environment: MODEL_NAME={env.get('MODEL_NAME')}, NUM_GPUS={env.get('NUM_GPUS')}")
        # pyrefly: ignore  # bad-assignment
        self.process = subprocess.Popen(
            ["python", str(bootstrapper_path)],
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

    def send_sample(self, request_data=None):
        client = gradio_client.Client(self.base_url)
        log.info(f"Available APIs: {client.view_api()}")

        request_text = json.dumps(request_data)
        log.info(f"input request: {json.dumps(request_data, indent=2)}")

        video, result = client.predict(request_text, api_name="/generate_video")

        if video is None:
            log.error(f"Error during inference: {result}")
        else:
            log.info(f"video: {json.dumps(video, indent=2)}")

        log.info(f"result: {result}")

    def shutdown_server(self):
        if self.process is None:
            log.warning("No process to shutdown")
            return

        log.info(f"Shutting down Gradio server (PID {self.process.pid})")
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
        log.info(f"Starting Gradio server test for {server_module} with {env_vars}")

        with TestHarness() as harness:
            harness.start_server(server_module=server_module, env_vars=env_vars)

            try:
                harness.send_sample(sample_request)
            except Exception as e:
                log.error(f"Sample request failed: {e}")
                raise

            log.info("Test completed successfully!")

        log.info("Server shutdown complete")
