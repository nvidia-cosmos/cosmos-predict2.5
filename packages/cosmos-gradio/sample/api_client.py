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
import gradio_client.utils as gradio_utils
from loguru import logger
from PIL import Image

sample_request = {"prompt": "a cat", "num_steps": 10}
url = "http://localhost:8080/"
asset_0 = "sample_0.png"
asset_1 = "sample_1.png"


def sample_image(fname: str) -> str:
    img = Image.new("RGB", (256, 256), color="red")
    img.save(fname)
    return fname


def create_sample_assets():
    sample_image(asset_0)
    sample_image(asset_1)


def upload_single_file(client, asset: str):
    file_descriptor = gradio_utils.handle_file(asset_0)
    upload_file_result_str = client.predict(file_descriptor, api_name="/upload_file")
    logger.info(f"Upload file result: {upload_file_result_str}")
    upload_file_result_dict = json.loads(upload_file_result_str)
    return upload_file_result_dict["path"]


def upload_file_list(client, assets: list[str]):
    file_descriptors = [gradio_utils.handle_file(asset) for asset in assets]
    upload_file_result_str = client.predict(file_descriptors, api_name="/upload_file_list")
    logger.info(f"Upload file list result: {upload_file_result_str}")


if __name__ == "__main__":
    client = gradio_client.Client(url)
    logger.info(f"Available APIs: {client.view_api()}")

    create_sample_assets()

    server_asset_0 = upload_single_file(client, asset_0)

    # if more than one file is required then files can also be uploaded as a list
    upload_file_list(client, [asset_0, asset_1])

    sample_request["input_image"] = server_asset_0
    request_text = json.dumps(sample_request)
    logger.info(f"input request: {json.dumps(sample_request, indent=2)}")

    video, result = client.predict(request_text, api_name="/generate_video")

    if video is None:
        logger.error(f"Error during inference: {result}")
    else:
        logger.info(f"video: {json.dumps(video, indent=2)}")

    logger.info(f"result: {result}")
