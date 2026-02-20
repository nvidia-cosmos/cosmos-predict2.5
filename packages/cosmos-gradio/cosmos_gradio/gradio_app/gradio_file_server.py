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

import datetime
import json
import os
import shutil
import time
import typing
import zipfile
from typing import Any

import gradio as gr
from loguru import logger

VIDEO_EXTENSION = typing.Literal[".mp4", ".avi", ".mov", ".mkv", ".webm"]
IMAGE_EXTENSION = typing.Literal[".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
JSON_EXTENSION = typing.Literal[".json"]
TEXT_EXTENSION = typing.Literal[".txt", ".md"]

FILE_EXTENSION = typing.Literal[VIDEO_EXTENSION, IMAGE_EXTENSION, JSON_EXTENSION, TEXT_EXTENSION]

FILE_TYPE = typing.Literal["video", "image", "json", "text", "other"]


def _get_file_type(file_path: str) -> FILE_TYPE:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        return "video"
    if ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
        return "image"
    if ext in [".json"]:
        return "json"
    if ext in [".txt", ".md"]:
        return "text"
    return "other"


def _handle_api_file_upload_event(file: str, upload_dir: str) -> str:
    """
    Event handler for the hidden file upload component.

    Used to upload files to the server without showing them in the UI (i.e. via the Python client).

    Args:
        file (str): The path to the temporary file created by Gradio
        upload_dir (str): The directory to save the uploaded files

    Returns:
        str: A JSON string with either of the following keys:
            - "path": (optional) The path to the uploaded file
            - "error": (optional) A message describing the error that occurred
    """
    response = _handle_api_file_upload_event_list([file], upload_dir)
    response_dict = json.loads(response)
    if "error" in response_dict:
        return response

    single_response = response_dict["files"][0]
    return json.dumps(single_response)


def _handle_api_file_upload_event_list(files: list[Any], upload_dir: str) -> str:
    try:
        responses = []
        # Create timestamped subfolder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_folder = os.path.join(upload_dir, f"upload_{timestamp}")
        os.makedirs(upload_folder, exist_ok=True)

        for file in files:
            if file is None:
                continue
            logger.info(f"Uploading file: {file=} {upload_dir=}")

            filename = os.path.basename(file.name)
            dest_path = os.path.join(upload_folder, filename)
            shutil.copy2(file.name, dest_path)
            logger.info(f"File uploaded to: {dest_path}")

            response = {"path": dest_path}
            logger.info(f"{response=}")
            responses.append(response)

        response = {"files": responses}
        return json.dumps(response)

    except Exception as e:
        message = f"Upload error: {e}"
        logger.error(message)
        return json.dumps({"error": message})


def _handle_file_upload_event(temp_files, output_dir: str) -> tuple[str, dict]:
    """Handle file uploads by copying to output directory"""
    refresh_update = _refresh_file_explorer_update(output_dir)
    if not temp_files:
        return "No files selected.", refresh_update

    try:
        # Create timestamped subfolder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_folder = os.path.join(output_dir, f"upload_{timestamp}")
        os.makedirs(upload_folder, exist_ok=True)

        uploaded_paths = []
        for temp_file in temp_files:
            if temp_file and hasattr(temp_file, "name"):
                filename = os.path.basename(temp_file.name)
                dest_path = os.path.join(upload_folder, filename)

                # Handle duplicates
                counter = 1
                original_name, ext = os.path.splitext(filename)
                while os.path.exists(dest_path):
                    filename = f"{original_name}_{counter}{ext}"
                    dest_path = os.path.join(upload_folder, filename)
                    counter += 1

                shutil.copy2(temp_file.name, dest_path)
                uploaded_paths.append(dest_path)

        # Format status message with full paths
        if uploaded_paths:
            status_lines = [f"✅ Uploaded {len(uploaded_paths)} files to {upload_folder}"]
            status_lines.extend(uploaded_paths)
            status_message = "\n".join(status_lines)
        else:
            status_message = "No files were uploaded."

        return status_message, refresh_update

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return f"❌ Upload failed: {e!s}", refresh_update


def _refresh_file_explorer_update(upload_dir: str) -> dict:
    """Return gr.update() to force FileExplorer to re-read the directory (refresh folder list).
    Gradio FileExplorer only re-fetches when its config changes; passing a unique ignore_glob
    (that matches no files) forces a refresh without hiding any files.
    """
    return gr.update(
        root_dir=upload_dir,
        ignore_glob=f"*__refresh_{int(time.time() * 1000)}__*",
    )


def _zip_folder(folder_path: str) -> str | None:
    """Create a zip of the given folder; return path to the zip, or None on error."""
    if not os.path.isdir(folder_path):
        return None
    zip_path = folder_path.rstrip(os.sep) + ".zip"
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, filenames in os.walk(folder_path):
                for name in filenames:
                    path = os.path.join(root, name)
                    arcname = os.path.relpath(path, os.path.dirname(folder_path))
                    zf.write(path, arcname)
        logger.info(f"Created folder zip: {zip_path}")
        return zip_path
    except Exception as e:
        logger.error(f"Failed to zip folder {folder_path}: {e}")
        return None


def _handle_file_explorer_select_event(
    selection: list[str] | str | None,
) -> tuple[gr.Video, gr.Image, gr.JSON, gr.Textbox, dict]:
    """
    Callback when the user selects a file or folder in the FileExplorer.
    Returns (video, image, json, text, download_btn_update).
    For a file: download points to the file; for a folder: download points to a .zip of the folder.
    """
    output_video = gr.Video(visible=False, height=400)
    output_image = gr.Image(visible=False, height=400)
    output_json = gr.JSON(visible=False)
    output_text = gr.Textbox(visible=False)
    download_update = gr.update(visible=False)

    try:
        if isinstance(selection, list):
            if not selection:
                raise ValueError("No file selected")
            path = selection[0]
        elif isinstance(selection, str):
            path = selection
        else:
            raise ValueError("No file selected")

        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")

        logger.info(f"FileExplorer selected: {path}")

        if os.path.isdir(path):
            output_text = gr.Textbox(
                value="Selected a folder. Use the download button below to download it as .zip",
                visible=True,
            )
            zip_path = _zip_folder(path)
            if zip_path:
                download_update = gr.update(value=zip_path, visible=True)
            return output_video, output_image, output_json, output_text, download_update

        # File: show preview and offer download
        file_type = _get_file_type(path)
        if file_type == "video":
            output_video = gr.Video(value=path, visible=True, height=400)
        elif file_type == "image":
            output_image = gr.Image(value=path, visible=True, height=400)
        elif file_type == "json":
            with open(path, encoding="utf-8") as f:
                output_json = gr.JSON(value=json.load(f), visible=True)
        elif file_type == "text":
            with open(path, encoding="utf-8") as f:
                output_text = gr.Textbox(value=f.read(), visible=True)
        else:
            output_text = gr.Textbox(value=f"Unsupported file type: {path}", visible=True)

        download_update = gr.update(value=path, visible=True)

    except Exception as e:
        logger.error(f"Error viewing selection: {e!s}")
        output_text = gr.Textbox(value=f"Error: {e!s}", visible=True)

    return output_video, output_image, output_json, output_text, download_update


def _instructions():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
            **Upload Files:**
            1. Upload files to the server by clicking/dragging files into the Upload Files section
            2. One or more files can be uploaded at once
            3. Supported file types:
            - Videos: .mp4, .avi, .mov, .mkv, .webm
            - Images: .jpg, .jpeg, .png, .gif, .bmp, .webp
            - JSON: .json
            - Text: .txt, .md
            4. The upload status will show if files were uploaded successfully
        """
            )

        with gr.Column(scale=1):
            gr.Markdown(
                """
            **Browse & View Files:**
            1. Use the File Browser to navigate folders (click to expand/collapse)
            2. Click a file to preview it, or a folder to download it as .zip
            3. Use **Download selected file or folder (.zip)** to download the current selection
            4. The file preview appears below the browser
            5. Click **Refresh Folder List** to update the folder list
        """
            )


def file_server_components(upload_dir: str, open: bool = True) -> gr.Accordion:
    """
    Gradio component that allows users to upload files, browse uploads, and view file contents.

    Args:
        upload_dir (str): The directory to store the uploaded files
        open (bool): Whether to open the top-level accordion by default

    Returns:
        gr.Accordion: The top-level accordion component
    """
    os.makedirs(upload_dir, exist_ok=True)

    with gr.Accordion("File Upload and Viewer", open=open) as top_level_accordion:
        with top_level_accordion:
            gr.Markdown(f"**Directory**: `{upload_dir}`")
            # Hidden components to support API file uploads (i.e. via the Python client)
            with gr.Row(visible=False):
                api_upload_file_input = gr.File(visible=False)
                api_upload_file_response = gr.Textbox(visible=False)
                api_upload_file_input_list = gr.File(visible=False, file_count="multiple")
                api_upload_file_response_list = gr.Textbox(visible=False)

            # UI components for file upload/browsing
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Upload Files")
                    file_upload = gr.File(
                        label="Select Files",
                        file_count="multiple",
                        file_types=[
                            ".mp4",
                            ".avi",
                            ".mov",
                            ".mkv",
                            ".webm",
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".gif",
                            ".bmp",
                            ".webp",
                            ".json",
                            ".txt",
                            ".md",
                        ],
                    )
                    upload_status = gr.Textbox(label="Status", lines=2, interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("## View Files")
                    file_explorer = gr.FileExplorer(
                        root_dir=upload_dir,
                        glob="**/*",
                        file_count="single",
                        label="Browse Files",
                        height=300,
                    )
                    refresh_file_list_btn = gr.Button(
                        value="🔄 Refresh Folder List",
                        variant="secondary",
                    )

                    # Output components
                    with gr.Group(elem_classes=["view-file-content"]):
                        output_video = gr.Video(label="Video", visible=False, height=400)
                        output_image = gr.Image(label="Image", visible=False, height=400)
                        output_json = gr.JSON(label="JSON", visible=False)
                        output_text = gr.Textbox(
                            label="Text",
                            value="Select a file to view its content",
                            lines=10,
                            visible=True,
                            interactive=False,
                        )
                        download_selected_btn = gr.DownloadButton(
                            label="Download selected file or folder (.zip)",
                            visible=False,
                        )

            with gr.Accordion("Instructions", open=False) as instr_accordion:
                with instr_accordion:
                    _instructions()

    # Set up event handlers
    api_upload_file_input.upload(
        fn=lambda file: _handle_api_file_upload_event(file, upload_dir),
        inputs=[api_upload_file_input],
        outputs=[api_upload_file_response],
        api_name="upload_file",
    )
    api_upload_file_input_list.upload(
        fn=lambda files: _handle_api_file_upload_event_list(files, upload_dir),
        inputs=[api_upload_file_input_list],
        outputs=[api_upload_file_response_list],
        api_name="upload_file_list",
    )
    file_upload.upload(
        fn=lambda temp_files: _handle_file_upload_event(temp_files, upload_dir),
        inputs=[file_upload],
        outputs=[upload_status, file_explorer],
        api_name=False,  # UI only component.
    )
    refresh_file_list_btn.click(
        fn=lambda: _refresh_file_explorer_update(upload_dir),
        inputs=[],
        outputs=[file_explorer],
        api_name=False,
    )
    # FileExplorer file/folder selection: preview + generic download for file or folder
    file_explorer.change(
        fn=_handle_file_explorer_select_event,
        inputs=[file_explorer],
        outputs=[output_video, output_image, output_json, output_text, download_selected_btn],
        api_name=False,  # UI only component.
    )

    return top_level_accordion


def create_gradio_blocks(output_dir: str) -> gr.Blocks:
    with gr.Blocks(title="File Upload and Viewer", theme=gr.themes.Soft()) as blocks:
        file_server_components(output_dir, open=True)

    return blocks


if __name__ == "__main__":
    save_dir = os.environ.get("GRADIO_SAVE_DIR", "/mnt/pvc/gradio/uploads")
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", 8080))

    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Starting app - {server_name}:{server_port} -> {save_dir}")

    blocks = create_gradio_blocks(output_dir=save_dir)
    blocks.launch(server_name=server_name, server_port=server_port, allowed_paths=[save_dir], share=False)
