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
from dataclasses import dataclass, field
from typing import Any

from cosmos_predict2.config import InferenceArguments
from cosmos_predict2.gradio.sample_data import (
    sample_request_distilled,
    sample_request_image2world,
    sample_request_multiview,
)
from cosmos_predict2.multiview_config import MultiviewInferenceArguments


@dataclass
class ModelConfig:
    header: dict[str, str] = field(
        default_factory=lambda: {
            "video2world": "Cosmos-Predict2.5 Video2World",
            "distilled": "Cosmos-Predict2.5 Distilled",
            "multiview": "Cosmos-Predict2.5 Multiview",
        }
    )

    help_text: dict[str, str] = field(
        default_factory=lambda: {
            "video2world": f"```json\n{json.dumps(InferenceArguments.model_json_schema(), indent=2)}\n```",
            "distilled": f"```json\n{json.dumps(InferenceArguments.model_json_schema(), indent=2)}\n```",
            "multiview": f"```json\n{json.dumps(MultiviewInferenceArguments.model_json_schema(), indent=2)}\n```",
        }
    )

    default_request: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "video2world": sample_request_image2world,
            "distilled": sample_request_distilled,
            "multiview": sample_request_multiview,
        }
    )
