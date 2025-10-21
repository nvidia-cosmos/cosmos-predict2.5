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


import gradio as gr

from cosmos_gradio.deployment_env import DeploymentEnv

cfg = DeploymentEnv()


def _tail_file(file_path: str, num_lines: int) -> str:
    """
    Reads the last `num_lines` lines from a file and returns them as a string.
    - Uses a seek-based approach to read only the last `num_lines` lines of the file.
    - So, performance shouldn't be affected by file size like it would be with readlines().
    """
    try:
        with open(file_path, "rb") as f:
            total_lines_wanted = num_lines

            BLOCK_SIZE = 1024
            f.seek(0, 2)
            block_end_byte = f.tell()
            lines_to_go = total_lines_wanted
            block_number = -1
            blocks = []
            while lines_to_go > 0 and block_end_byte > 0:
                if block_end_byte - BLOCK_SIZE > 0:
                    f.seek(block_number * BLOCK_SIZE, 2)
                    blocks.append(f.read(BLOCK_SIZE))
                else:
                    f.seek(0, 0)
                    blocks.append(f.read(block_end_byte))
                lines_found = blocks[-1].count(b"\n")
                lines_to_go -= lines_found
                block_end_byte -= BLOCK_SIZE
                block_number -= 1
            all_read_text = b"".join(reversed(blocks))
            # Decode bytes to string and return
            text = all_read_text.decode("utf-8", errors="replace")
            return "\n".join(text.splitlines()[-total_lines_wanted:])
    except FileNotFoundError:
        return f"Log file not found: {file_path}"
    except Exception as e:
        return f"Error reading log file: {e}"


def log_file_viewer(
    log_file: str = cfg.log_file,
    num_lines: int = 100,
    update_interval: float = 1,
) -> str:
    """
    Gradio component that renders the final `num_lines` lines of a log file, updating periodically.

    Args:
        log_file (str): The path to the log file.
        num_lines (int): The number of lines to read from the log file.
        update_interval (float): The interval in seconds at which to update the log file.

    Returns:
        gr.Textbox: Textbox component that displays the tail of the log file.
    """

    def _tail_logs() -> str:
        return _tail_file(log_file, num_lines)

    # Use timer.tick() to update the log file, as gr.Textbox(every=...) reveals the API endpoint.
    timer = gr.Timer(value=update_interval, active=True)
    logs = gr.Textbox(label="Logs", interactive=False, lines=30, autoscroll=True, value=_tail_logs())
    timer.tick(fn=_tail_logs, outputs=logs, api_name=False)

    # pyrefly: ignore  # bad-return
    return logs
