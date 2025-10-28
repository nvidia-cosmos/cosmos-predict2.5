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

import atexit
import fnmatch
import logging
import os
import sys
import warnings
from pathlib import Path

import loguru
from cosmos_predict2._src.imaginaire.flags import FLAGS, VERBOSE
from cosmos_predict2._src.imaginaire.utils import log

"""Package initialization."""


def enable_distributed() -> bool:
    return "RANK" in os.environ


def is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


_LOGGER_FORMAT = f"{log.get_datetime_format()}{log.get_machine_format()}{log.get_message_format()}"
_LOGGER_INCLUDE = [
    "cosmos_predict2._src.imaginaire.utils.checkpoint_db",
    "cosmos_predict2._src.imaginaire.trainer",
    "*.callbacks.*",
]
_LOGGER_EXCLUDE = [
    "*._*",
    "projects.*",
    "cosmos_predict2._src.imaginaire.*",
]


def _console_filter(record: dict) -> bool:
    # Not sure why but critical messages need a special case to be filtered
    if record["level"].name == "CRITICAL":
        module_name: str = record["name"]
        for pat in _LOGGER_INCLUDE:
            if fnmatch.fnmatch(module_name, pat):
                return True
        for pat in _LOGGER_EXCLUDE:
            if fnmatch.fnmatch(module_name, pat):
                return False
        return True

    if not log._rank0_only_filter(record):
        return False
    module_name: str = record["name"]
    for pat in _LOGGER_INCLUDE:
        if fnmatch.fnmatch(module_name, pat):
            return True
    for pat in _LOGGER_EXCLUDE:
        if fnmatch.fnmatch(module_name, pat):
            return False
    return True


def _init_log_console():
    log.logger.remove()
    log.logger.add(
        sys.stdout,
        level="DEBUG" if VERBOSE else "INFO",
        format=_LOGGER_FORMAT,
        filter=log._rank0_only_filter if VERBOSE else _console_filter,
        catch=False,
    )
    if not VERBOSE:
        logging.basicConfig(
            level=logging.ERROR,
        )
        loguru.logger.remove()
        warnings.filterwarnings("ignore")


def _init_log_files(output_dir: Path):
    console_path = output_dir / "console.log"
    debug_path = output_dir / "debug.log"
    log.info(f"Log saved to {console_path}")
    log.logger.add(
        console_path,
        mode="w",
        level="INFO",
        format=_LOGGER_FORMAT,
        filter=_console_filter,
        enqueue=True,
        catch=False,
    )
    log.logger.add(
        debug_path,
        mode="w",
        level="DEBUG",
        format=_LOGGER_FORMAT,
        filter=log._rank0_only_filter,
        enqueue=True,
        catch=False,
    )


def _init_distributed():
    from cosmos_predict2._src.imaginaire.utils import distributed

    distributed.init()


def _cleanup_distributed():
    import torch.distributed as dist
    from megatron.core import parallel_state

    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def _init_profiler(output_dir: Path):
    import pyinstrument
    import pyinstrument.renderers

    profiler = pyinstrument.Profiler()
    profiler.start()

    def stop_profiler():
        log.info("Stopping profiler")
        profiler.stop()
        renderers: list[pyinstrument.renderers.Renderer] = [
            pyinstrument.renderers.SessionRenderer(),
        ]
        for renderer in renderers:
            output_path = output_dir / f"profile.{renderer.output_file_extension}"
            output_path.write_text(profiler.output(renderer))
            log.info(f"Profile saved to {output_path}")

    atexit.register(stop_profiler)


def init_environment():
    """Initialize environment."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _init_log_console()
    if enable_distributed():
        _init_distributed()


def cleanup_environment():
    """Clean up environment."""
    if enable_distributed():
        _cleanup_distributed()


def init_output_dir(output_dir: Path, *, profile: bool = False):
    """Initialize output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not is_rank0():
        return

    _init_log_files(output_dir)
    log.debug(f"{FLAGS}")
    if profile:
        _init_profiler(output_dir)
