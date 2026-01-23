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

from cosmos_gradio.gradio_app.gradio_app import GradioApp
from cosmos_gradio.gradio_app.gradio_ui import create_gradio_UI


def launch_gradio_server(
    factory_module: str,
    factory_function: str,
    validator: callable,
    num_gpus: int,
    header: str,
    default_request: dict,
    help_text: str,
    output_dir="/tmp/gradio/outputs",
    uploads_dir="/tmp/gradio/uploads",
    log_file="/tmp/gradio/log.txt",
    server_name="0.0.0.0",
    server_port=8080,
    allowed_paths=None,
):
    app = GradioApp(
        num_gpus=num_gpus,
        validator=validator,
        factory_module=factory_module,
        factory_function=factory_function,
        output_dir=output_dir,
        default_request=default_request,
    )

    interface = create_gradio_UI(
        app,
        header=header,
        default_request=default_request,
        help_text=help_text,
        uploads_dir=uploads_dir,
        output_dir=output_dir,
        log_file=log_file,
    )

    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        debug=True,
        max_file_size="500MB",
        allowed_paths=allowed_paths,
    )
