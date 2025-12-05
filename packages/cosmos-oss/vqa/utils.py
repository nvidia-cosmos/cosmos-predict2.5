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
from pathlib import Path

__all__ = ["is_verbose", "log_verbose", "log_info", "find_mp4_files"]


def is_verbose() -> bool:
    """Check if verbose mode is enabled via VERBOSE environment variable."""
    return os.environ.get("VERBOSE", "0") == "1"


def log_verbose(message: str) -> None:
    """
    Print a message only if VERBOSE environment variable is set to 1.

    Args:
        message: The message to print
    """
    if is_verbose():
        print(message)


def log_info(message: str) -> None:
    """
    Print a message regardless of VERBOSE setting.
    Use for important information, summaries, and critical messages.

    Args:
        message: The message to print
    """
    print(message)


def find_mp4_files(directory: str | Path) -> list[Path]:
    """
    Recursively find all MP4 files in a directory.

    Args:
        directory: Path to the directory to search

    Returns:
        List of Path objects for all found MP4 files (sorted)

    Raises:
        FileNotFoundError: If the directory doesn't exist
        ValueError: If the path is not a directory
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Find all .mp4 files recursively
    mp4_files = sorted(directory.rglob("*.mp4"))

    return mp4_files
