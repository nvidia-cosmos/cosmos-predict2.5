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

"""
Dataset registration for cosmos datasets with support for different caption types.
"""

from cosmos_predict2._src.imaginaire.utils import log

DATASET_OPTIONS = {}


# embeddings are packed together. Need to clean data to reduce entropy.
_CAPTION_EMBEDDING_KEY_MAPPING_IMAGES = {
    "ai_v3p1": "ai_v3p1",
    "qwen2p5_7b_v4": "qwen2p5_7b_v4",
    "prompts": "qwen2p5_7b_v4",
}


def dataset_register(key):
    log.info(f"registering dataset {key}")

    def decorator(func):
        DATASET_OPTIONS[key] = func
        return func

    return decorator
