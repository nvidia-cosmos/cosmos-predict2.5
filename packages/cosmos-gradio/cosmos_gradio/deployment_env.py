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
from dataclasses import dataclass


@dataclass
class DeploymentEnv:
    model_name: str = os.getenv("MODEL_NAME", "")
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs/")
    uploads_dir: str = os.getenv("UPLOADS_DIR", "uploads/")
    log_file: str = os.getenv("LOG_FILE", "output.log")
    num_gpus: int = int(os.environ.get("NUM_GPUS", 1))
    use_distilled: bool = os.getenv("USE_DISTILLED", "False").lower() in ("true", "1", "t")
    cli_app: str = os.getenv("CLI_APP", "")
