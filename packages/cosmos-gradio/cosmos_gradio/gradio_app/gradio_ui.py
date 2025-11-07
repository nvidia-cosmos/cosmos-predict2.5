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

from datetime import datetime

import gradio as gr

from cosmos_gradio.gradio_app.gradio_file_server import file_server_components
from cosmos_gradio.gradio_app.gradio_log_file_viewer import log_file_viewer
from cosmos_gradio.gradio_app.util import get_git_info


def create_gradio_UI(infer_func, header, default_request, help_text, uploads_dir, output_dir, log_file):
    with gr.Blocks(title=header, theme=gr.themes.Soft()) as interface:
        gr.Markdown(f"# {header}")
        gr.Markdown(f"instance created {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {get_git_info()}")
        gr.Markdown("Upload a media file. Use the resulting server file path as input media in the json request.")

        with gr.Row():
            file_server_components(uploads_dir, open=False)

        gr.Markdown("---")
        gr.Markdown(f"**Output Directory**: {output_dir}")

        with gr.Row():
            with gr.Column(scale=1):
                # Single request input field
                request_input = gr.Textbox(
                    label="Request (JSON)",
                    value=default_request,
                    lines=20,
                    interactive=True,
                )

                # Help section
                with gr.Accordion("Request Format Help", open=False):
                    gr.Markdown(help_text)

            with gr.Column(scale=1):
                # Output
                output_video = gr.Video(label="Generated Video", height=400)
                status_text = gr.Textbox(label="Status", lines=5, interactive=False)
                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

        log_file_viewer(log_file=log_file, num_lines=100, update_interval=1)

        generate_btn.click(
            fn=infer_func,
            inputs=[request_input],
            outputs=[output_video, status_text],
            api_name="generate_video",
        )

    return interface
