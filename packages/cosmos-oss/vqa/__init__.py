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

"""VQA (Video Question Answering) package for Cosmos models."""

__version__ = "1.0.0"

# Make key components available at package level
try:
    from vqa.cosmos_reason_inference import CosmosReasonModel

    __all__ = [
        "CosmosReasonModel",
    ]
except ImportError:
    # Allow package to be imported even if dependencies are not installed
    pass
