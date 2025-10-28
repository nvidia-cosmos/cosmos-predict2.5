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
import typing

import gradio as gr
from loguru import logger

VIDEO_EXTENSION = typing.Literal[".mp4", ".avi", ".mov", ".mkv", ".webm"]
IMAGE_EXTENSION = typing.Literal[".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
JSON_EXTENSION = typing.Literal[".json"]
TEXT_EXTENSION = typing.Literal[".txt", ".md"]

FILE_EXTENSION = typing.Literal[VIDEO_EXTENSION, IMAGE_EXTENSION, JSON_EXTENSION, TEXT_EXTENSION]

FILE_TYPE = typing.Literal["video", "image", "json", "text", "other"]


def _get_files_in_output_dir(output_dir: str):
    """Scan output directory and return list of all files with their info"""
    if not os.path.exists(output_dir):
        return []

    files = []
    for root, _, filenames in os.walk(output_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            files.append(
                {
                    "path": filepath,
                    "name": filename,
                    "type": _get_file_type(filepath),
                    "relative_path": os.path.relpath(filepath, output_dir),
                }
            )
    return sorted(files, key=lambda x: x["path"])


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


def _get_file_icon(file_path: str) -> str:
    file_type = _get_file_type(file_path)
    if file_type == "video":
        return "🎥"
    if file_type == "image":
        return "🖼️"
    if file_type == "json":
        return "📋"
    if file_type == "text":
        return "📄"
    return "📄"


def _format_file_path_with_icon(file_path: str) -> str:
    icon = _get_file_icon(file_path)
    return f"{icon} {file_path}"


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
    dest_path = None
    try:
        logger.info(f"Uploading file: {file=} {upload_dir=}")

        # Create timestamped subfolder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_folder = os.path.join(upload_dir, f"upload_{timestamp}")
        os.makedirs(upload_folder, exist_ok=True)

        filename = os.path.basename(file)
        dest_path = os.path.join(upload_folder, filename)
        shutil.copy2(file, dest_path)
        logger.info(f"File uploaded to: {dest_path}")

        response = {"path": dest_path}
        logger.info(f"{response=}")
        # pyrefly: ignore  # bad-return
        return json.dumps(response)

    except Exception as e:
        message = f"Upload error: {e}"
        logger.error(message)
        return json.dumps({"error": message})


def _handle_file_upload_event(temp_files, output_dir: str):
    """Handle file uploads by copying to output directory"""
    if not temp_files:
        return "", "No files selected.", gr.Dropdown()

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

        # Get updated list of choices for the file browser dropdown
        choices = _format_files_list(output_dir=output_dir)

        # Format status message with full paths
        if uploaded_paths:
            status_lines = [f"✅ Uploaded {len(uploaded_paths)} files to {upload_folder}"]
            status_lines.extend(uploaded_paths)
            status_message = "\n".join(status_lines)
        else:
            status_message = "No files were uploaded."

        return (status_message, gr.Dropdown(choices=choices, value=None))

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return "", f"❌ Upload failed: {e!s}", gr.Dropdown()


def _format_files_list(files: list[dict] | None = None, output_dir: str | None = None) -> list[str]:
    # pyrefly: ignore  # bad-argument-type
    files = files or _get_files_in_output_dir(output_dir)

    if not files:
        logger.warning("No files in directory.")
        return []

    file_paths = [file["path"] for file in files]
    file_paths = sorted(file_paths)
    file_paths = [_format_file_path_with_icon(file_path) for file_path in file_paths]

    return file_paths


def _handle_refresh_button_click_event(
    dropdown_value: str | None | list[str | int | float] = None,
    output_dir: str | None = None,
) -> gr.Dropdown:
    logger.info(f"Refreshing file list: {dropdown_value=}")
    return _view_file_dropdown(value=dropdown_value or None, output_dir=output_dir)


def _view_file_dropdown(
    value: str | None | list[str | int | float] = None, output_dir: str | None = None
) -> gr.Dropdown:
    file_paths_with_icons = _format_files_list(output_dir=output_dir)
    return gr.Dropdown(
        label="Select a File to View",
        interactive=True,
        choices=file_paths_with_icons,  # type: ignore (gradio mistake)
        value=value,
    )


def _handle_view_file_dropdown_select_event(selection: str) -> tuple[gr.Video, gr.Image, gr.JSON, gr.Textbox]:
    """
    Callback executed when the user selects a file from the dropdown

    Args:
        selection (str): The value of the dropdown that was selected (an icon and file path)

    Returns:
        A tuple containing 4 output components: (video, image, json, text). Only one component will be visible,
        depending on the selected file's type.
    """
    logger.info(f"Loading file: {selection}")

    # Output components
    output_video: gr.Video = gr.Video(visible=False)
    output_image: gr.Image = gr.Image(visible=False)
    output_json: gr.JSON = gr.JSON(visible=False)
    output_text: gr.Textbox = gr.Textbox(visible=False)

    try:
        # Strip the leading icon from the selected path
        index_leading_slash = selection.find("/")
        file_path = selection[index_leading_slash:]

        if not file_path:
            raise ValueError("No file selected")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Construct the appropriate output component based on the file type
        file_type = _get_file_type(file_path)
        if file_type == "video":
            output_video = gr.Video(value=file_path, visible=True)
        elif file_type == "image":
            output_image = gr.Image(value=file_path, visible=True)
        elif file_type == "json":
            with open(file_path, encoding="utf-8") as file:
                output_json = gr.JSON(value=json.load(file), visible=True)
        elif file_type == "text":
            with open(file_path, encoding="utf-8") as file:
                output_text = gr.Textbox(value=file.read(), visible=True)
        else:
            message = f"Unable to display unsupported file type: {file_path}"
            logger.warning(message)
            output_text = gr.Textbox(value=message, visible=True)

    # Handle errors by displaying the message in the textbox
    except Exception as e:
        message = f"Error viewing {selection}: {e!s}"
        logger.error(message)
        output_text = gr.Textbox(value=message, visible=True)

    # Return all 4 output components, only one of which will be visible
    return output_video, output_image, output_json, output_text


def _instructions():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
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
            1. Use the dropdown menu to select a file to view
            2. The file contents will be displayed under the dropdown menu
            3. Click "Refresh File List" to update the dropdown's choices
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

    with gr.Accordion("File Upload and Viewer", open=open) as top_level_accordion:
        with top_level_accordion:
            gr.Markdown(f"**Directory**: `{upload_dir}`")
            # Hidden components to support API file uploads (i.e. via the Python client)
            with gr.Row(visible=False):
                api_upload_file_input = gr.File(visible=False)
                api_upload_file_response = gr.Textbox(visible=False)

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
                    view_file_dropdown = _view_file_dropdown(output_dir=upload_dir)
                    refresh_btn = gr.Button("🔄 Refresh File List", variant="secondary")

                    # Output components
                    with gr.Group(elem_classes=["view-file-content"]):
                        output_video = gr.Video(label="Video", visible=False)
                        output_image = gr.Image(label="Image", visible=False)
                        output_json = gr.JSON(label="JSON", visible=False)
                        output_text = gr.Textbox(
                            label="Text",
                            value="Select a file to view its content",
                            lines=10,
                            visible=True,
                            interactive=False,
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
    file_upload.upload(
        fn=lambda temp_files: _handle_file_upload_event(temp_files, upload_dir),
        inputs=[file_upload],
        outputs=[upload_status, view_file_dropdown],
        api_name=False,  # UI only component.
    )
    refresh_btn.click(
        fn=lambda dropdown_value: _handle_refresh_button_click_event(dropdown_value, upload_dir),
        inputs=[view_file_dropdown],
        outputs=[view_file_dropdown],
        api_name=False,  # UI only component.
    )
    view_file_dropdown.select(
        fn=_handle_view_file_dropdown_select_event,
        inputs=[view_file_dropdown],
        outputs=[output_video, output_image, output_json, output_text],
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
