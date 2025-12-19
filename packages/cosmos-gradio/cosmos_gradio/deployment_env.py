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


from pydantic import Field
from pydantic_settings import BaseSettings

# Module-level singleton storage (outside class to avoid Pydantic treating it as a private attr)
_deployment_env_instance: "DeploymentEnv | None" = None


class DeploymentEnv(BaseSettings):
    """
    Deployment environment settings for Gradio applications.

    This class uses pydantic_settings to load configuration from environment variables.
    Settings can be provided in two ways:

    1. Environment variables:
       MODEL_NAME=predict2 OUTPUT_DIR=outputs/ python sample/bootstrapper.py

    2. .env file in the working directory:
       MODEL_NAME=predict2
       OUTPUT_DIR=outputs/

    The .env file is automatically loaded if present. Environment variables take precedence
    over .env file values.
    """

    model_name: str = Field(default="", description="Name of the model to deploy")
    model_size: str = Field(default="2B/pre-trained", description="Size of the model to deploy")
    output_dir: str = Field(default="outputs/", description="Directory for output files")
    uploads_dir: str = Field(default="uploads/", description="Directory for uploaded files")
    log_file: str = Field(default="output.log", description="Path to log file")
    num_gpus: int = Field(default=1, ge=1, description="Number of GPUs to use")
    disable_guardrails: bool = Field(default=False, description="Whether to disable guardrails")
    worker_timeout: int = Field(default=3600, description="Timeout for worker process in seconds")

    model_config = {"env_file": ".env", "extra": "ignore", "frozen": True}

    @classmethod
    def get_instance(cls) -> "DeploymentEnv":
        """
        Returns a singleton instance of DeploymentEnv.

        The instance is created on first call and cached for subsequent calls.
        This ensures consistent configuration across the application.
        """
        global _deployment_env_instance
        if _deployment_env_instance is None:
            _deployment_env_instance = cls()
        return _deployment_env_instance

    @property
    def allowed_paths(self) -> list[str]:
        """
        Returns list of paths allowed for Gradio file serving.
        Includes output_dir, uploads_dir, and log_file directory.
        """
        import os

        paths = [self.output_dir, self.uploads_dir]
        log_file_dir = os.path.dirname(self.log_file)
        if log_file_dir:
            paths.append(log_file_dir)
        return paths
