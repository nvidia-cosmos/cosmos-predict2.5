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

import os
import subprocess
from datetime import datetime


def get_output_folder(output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_dir, f"generation_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def get_outputs(output_folder):
    # Check if any mp4 file was generated
    mp4_files = [f for f in os.listdir(output_folder) if f.endswith(".mp4")]
    img_files = [f for f in os.listdir(output_folder) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG", ".png"))]
    txt_files = [f for f in os.listdir(output_folder) if f.endswith(".txt")]

    # Read the generated prompt
    final_prompt = ""
    if txt_files:
        with open(os.path.join(output_folder, txt_files[0]), encoding="utf-8") as f:
            final_prompt = f.read().strip()

    output_path = None

    if img_files:
        # Use the first image file found
        output_path = os.path.join(output_folder, img_files[0])

    if mp4_files:
        if "output.mp4" in mp4_files:
            output_path = os.path.join(output_folder, "output.mp4")
        else:
            # Use the first mp4 file found
            output_path = os.path.join(output_folder, mp4_files[0])

    if output_path:
        msg = f"Found output: {output_path}"
        if final_prompt:
            msg += f"\nFinal prompt: {final_prompt}"

        return (
            output_path,
            msg,
        )
    else:
        return (
            None,
            f"Generation failed - no output found in output folder: {output_folder}",
        )


def get_git_info() -> str:
    """Get current git branch and SHA as a string."""

    try:
        # Get repository name
        repo_result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"], capture_output=True, text=True, check=True
        )
        repo_url = repo_result.stdout.strip()
        repo_name = repo_url.split("/")[-1].replace(".git", "")

        # Get current branch
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
        )
        branch = branch_result.stdout.strip()

        # Get current commit SHA
        sha_result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        sha = sha_result.stdout.strip()[:8]  # Short SHA

        # Get last commit date
        date_result = subprocess.run(
            ["git", "log", "-1", "--format=%cd", "--date=short"], capture_output=True, text=True, check=True
        )
        date = date_result.stdout.strip()

        return f"{repo_name}/{branch}@{sha} last change {date}"

    except subprocess.CalledProcessError:
        return "unknown@unknown"
