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

import gradio_client.client as gradio_client
from loguru import logger

sample_request = {"prompt": "a cat", "num_steps": 10}
url = "http://localhost:8080/"

if __name__ == "__main__":
    client = gradio_client.Client(url)
    logger.info(f"Available APIs: {client.view_api()}")

    request_text = json.dumps(sample_request)
    logger.info(f"input request: {json.dumps(sample_request, indent=2)}")

    video, result = client.predict(request_text, api_name="/generate_video")

    if video is None:
        logger.error(f"Error during inference: {result}")
    else:
        logger.info(f"video: {json.dumps(video, indent=2)}")

    logger.info(f"result: {result}")
