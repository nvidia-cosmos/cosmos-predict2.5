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

"""
Distributed checkpoint (DCP) directory structure and storage backends.

The checkpointer saves model state in a sharded format across multiple processes:

self.save_dirname/
├── iter_000000005/                    # Checkpoint at iteration 5
│   ├── model/                         # Model state shards
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   ├── optim/                        # Optimizer state shards
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   ├── scheduler/                    # Learning rate scheduler state
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   └── trainer/                      # Additional training state
│       ├── __0_0.distcp              # Shard 0 from rank 0
│       └── __1_0.distcp              # Shard 1 from rank 1
└── latest_checkpoint.txt             # Points to most recent checkpoint folder, e.g. iter_000000005

Storage backends:
- Local filesystem:
  self.save_dirname = "{config_job.path_local}/checkpoints"

- S3 object store:
  self.save_dirname = "s3://{bucket}/{config_job.path}/checkpoints"
  where bucket = self.config_checkpoint.save_to_object_store.bucket

The sharded format enables efficient distributed saving/loading by:
1. Parallelizing I/O across processes
2. Reducing memory usage per process
3. Supporting both local and cloud storage backends
"""

import enum
import functools
import multiprocessing
import os
import queue
import re
import time
from collections import namedtuple
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from cosmos_predict2._src.imaginaire.checkpointer.base import AbstractCheckpointer
from cosmos_predict2._src.imaginaire.checkpointer.s3_filesystem import S3StorageReader, S3StorageWriter
from cosmos_predict2._src.imaginaire.config import CheckpointConfig, JobConfig
from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import callback, distributed, log, misc
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io

try:
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner as _DefaultLoadPlanner
    from torch.distributed.checkpoint.default_planner import (
        DTensor,
        LoadPlan,
        _create_read_items,
        _version,
        flatten_state_dict,
    )
    from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

    def create_default_local_load_plan(
        state_dict: dict[str, Any], metadata: Metadata, strict: bool = True, dcp_allow_mismatched_size: bool = False
    ) -> LoadPlan:
        requests = []
        """
        Create the ``LoadPlan`` used by DefaultLoadPlanner.

        It produces one read item per value in ``state_dict`` using the metadata in ``metadata``.

        The default behavior is to match key exactly between state_dict and metadata.
        It handles resharding by issuing multiple read requests against storage in order to match
        load requirements.
        """

        for fqn, obj in state_dict.items():
            if fqn.endswith("._extra_state"):  # dirty TE attention package!
                continue
            # ignore state_dict keys which do not exist in `state_dict` if strict=False
            if fqn not in metadata.state_dict_metadata:
                if strict:
                    raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")
                else:
                    continue

            md = metadata.state_dict_metadata[fqn]

            if not dcp_allow_mismatched_size:
                if (
                    isinstance(md, TensorStorageMetadata)
                    and getattr(obj, "size", None) is not None
                    and md.size != obj.size()
                ):
                    if not strict:
                        log.critical(f"Size mismatch between saved {md.size} and current: {obj.size()} for {fqn}")
                        continue
                    else:
                        raise ValueError(
                            f"Size mismatch between saved {md.size} and current: {obj.size()} for {fqn}",
                        )
            # Since DTensor supports submesh, adding extra check to ensure _create_read_items()
            # gets called only when the current rank is part of the mesh for the corresponding DTensor.
            if isinstance(obj, DTensor):
                if obj.device_mesh.get_coordinate() is not None:
                    requests += _create_read_items(fqn, md, obj)
            else:
                requests += _create_read_items(fqn, md, obj)

        return LoadPlan(requests)

    class DefaultLoadPlanner(_DefaultLoadPlanner):
        def set_partial_channel_weight(self, dcp_allow_mismatched_size: bool):
            self.dcp_allow_mismatched_size = dcp_allow_mismatched_size

        def create_local_plan(self) -> LoadPlan:
            assert self.metadata is not None
            if self.flatten_state_dict:
                # To support checkpoints that are saved before v2.4, we have to
                # differentiate if the missing keys are due to old checkpoints.
                # The contracts are:
                # 1. There are 3 cases when we found a missing key.
                #    1.1 Actual missing key, but allow_partial_load is False
                #    1.2 Actual missing key, but allow_partial load is True
                #    1.3 Old checkpoint, but allow_partial_load is False
                #    1.4 Old checkpoint, but allow_partial_load is True
                # 2. If we found a missing key, we first convert the keys back to
                #    the key format of v2.3
                # 3. If the previous missing keys are in the v2.3 keys, we assume
                #    this is a old checkpoint.
                # 4. Pass the state_dict to `create_default_local_load_plan()`,
                #    which has the logic to check missing for allow_partial_load.
                # So for 1.2 and 1.4 cases, we delegate allow_partial_load check to
                # `create_default_local_load_plan()`. The logic here is to determine
                # whether the checkpoint belong to 2.3 (or before) or 2.4 (or after).
                current_keys = set(self.state_dict.keys())
                load_keys = set(self.metadata.state_dict_metadata.keys())
                missing_keys = load_keys - current_keys
                if missing_keys:
                    _version._derived_version = "2_3"
                    old_state_dict, old_mappings = flatten_state_dict(self.original_state_dict)
                    old_keys = set(old_state_dict.keys())
                    if old_keys & missing_keys:
                        self.state_dict, self.mappings = old_state_dict, old_mappings
                    # _derived_version is only used by flatten_state_dict now.
                    # Set it back to None so that later we can save to a new version.
                    _version._derived_version = None

            return create_default_local_load_plan(
                self.state_dict,
                self.metadata,
                not self.allow_partial_load,
                getattr(self, "dcp_allow_mismatched_size", False),
            )

    log.critical("for the back comptiable pytorch! New DefaultLoadPlanner class is created.")
except ImportError as e:
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    log.critical(f"{e}, using default planner")


StateDictItemPath = namedtuple("StateDictItemPath", ["state_dict", "save_path"])

# (qsh 2025-01-01) the design is from https://github.com/pytorch/torchtitan/blob/1060feacc1b51cb6b339a04e53a5243b8466552b/torchtitan/checkpoint.py
# we recreate wrapper when needed instead of creating one from the beginning.
# to people who find it difficult to digest the code, official tutorial for torch dcp may be helpful


class ModelWrapper(Stateful):
    """Wrapper for model state dict handling"""

    def __init__(self, model: Union[nn.Module, List[nn.Module]], load_ema_to_reg: bool = False):
        self.model = [model] if isinstance(model, nn.Module) else model
        self.load_ema_to_reg = load_ema_to_reg
        if self.load_ema_to_reg:
            supported_model_types = []
            from cosmos_predict2._src.predict2.models.text2world_model import DiffusionModel as predict2_DiffusionModel

            supported_model_types.append(predict2_DiffusionModel)
            from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
                Text2WorldModelRectifiedFlow as predict2_DiffusionModel_rectified_flow,
            )

            supported_model_types.append(predict2_DiffusionModel_rectified_flow)
            from cosmos_predict2._src.predict2.models.text2world_wan2pt1_model import (
                WANDiffusionModel as wan2pt1_DiffusionModel,
            )

            supported_model_types.append(wan2pt1_DiffusionModel)

            assert any(isinstance(model, cls) for cls in supported_model_types), (
                f"ModelWrapper only supports DiffusionModel when load_ema_to_reg is True, but got {type(model)}"
            )

    def state_dict(self) -> Dict[str, Any]:
        _state_dict = {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}
        if self.load_ema_to_reg:
            assert not self.model[0].config.ema.enabled, (
                "EMA is enabled, can not load EMA weights to regular model weights"
            )
            all_keys = list(_state_dict.keys())
            assert all(k.startswith("net.") for k in all_keys), "All keys must start with net."
            for k in all_keys:
                _state_dict[k.replace("net.", "net_ema.")] = _state_dict.pop(k)

        if hasattr(self.model[0].config, "use_lora") and self.model[0].config.use_lora:
            """
            When using LoRA, `inject_adapter_in_model` modifies the target modules in place.
            For example, `blocks[0].attn.q_proj.weight` will be modified to `blocks[0].attn.q_proj.base_layer.weight`.
            This means that the model will have the key `blocks[0].attn.q_proj.base_layer.weight`,
            but the checkpoint will have the key `blocks[0].attn.q_proj.weight`.
            We need to map the model key to the checkpoint key.
            """
            self.checkpoint_to_model_key = {}
            keys_to_update = []
            for k in _state_dict.keys():
                if "base_layer." in k:
                    keys_to_update.append(k)
                    self.checkpoint_to_model_key[k.replace("base_layer.", "")] = k
            for k in keys_to_update:
                _state_dict[k.replace("base_layer.", "")] = _state_dict.pop(k)

        return _state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if hasattr(self.model[0].config, "use_lora") and self.model[0].config.use_lora:
            if hasattr(self, "checkpoint_to_model_key"):
                for checkpoint_key, model_key in self.checkpoint_to_model_key.items():
                    state_dict[model_key] = state_dict.pop(checkpoint_key)
            else:
                raise ValueError("checkpoint_to_model_key is not set by `state_dict`")
        if self.load_ema_to_reg:
            assert not self.model[0].config.ema.enabled, (
                "EMA is enabled, can not load EMA weights to regular model weights"
            )
            all_keys = list(state_dict.keys())
            assert all(k.startswith("net_ema.") for k in all_keys), "All keys must start with net_ema."
            for k in all_keys:
                state_dict[k.replace("net_ema.", "net.")] = state_dict.pop(k)

        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))


class OptimizerWrapper(Stateful):
    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        optim: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
    ) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.optim = [optim] if isinstance(optim, torch.optim.Optimizer) else optim

    def state_dict(self) -> Dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {k: v for sd in map(func, self.model, self.optim) for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model, self.optim))


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class Terminate:
    pass


class SaveDone:
    def __init__(self, iteration: int, elapsed_time: float, succeeded: bool):
        self.iteration = iteration
        self.elapsed_time = elapsed_time
        self.succeeded = succeeded

    def __str__(self):
        return f"SaveDone(iteration={self.iteration}, elapsed_time={self.elapsed_time}, succeeded={self.succeeded})"


def save_checkpoint_in_background(
    receiver_queue: multiprocessing.Queue,
    sender_queue: multiprocessing.Queue,
    checkpoint_config: CheckpointConfig,
    job_config: JobConfig,
) -> None:
    """
    Handles model checkpoint saving in a separate background process using PyTorch's distributed functionality.
    This function runs in a dedicated process to avoid blocking the main training loop.

    Args:
        receiver_queue: Queue to receive state dictionaries and commands from the main process
        sender_queue: Queue to send completion signals back to the main process
        checkpoint_config: Configuration settings for checkpoint saving behavior
        job_config: Configuration settings for the training job

    Flow:
        1. Initializes distributed processing environment
        2. Continuously waits for state dictionaries to save
        3. Saves checkpoints asynchronously
        4. Signals completion back to main process
        5. Terminates when receiving a Terminate signal

    Raises:
        AssertionError: If received object is neither Terminate signal nor valid state dict tuple

    Note:
        - Uses a different port than the main process to avoid conflicts
        - Disables TorchElastic agent store for checkpoint operations
        - Automatically cleans up distributed process group on exit
    """
    # Configure distributed environment
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"

    # Set up GPU device and distributed processing
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init()

    # Initialize checkpointing mechanism
    checkpoint_handler = DistributedCheckpointer(checkpoint_config, job_config, None, disable_async=True)

    try:
        while True:
            log.debug("Checkpoint background process is ready for next task, waiting for new state_dict")
            received_data = receiver_queue.get()
            log.debug("Received new state_dict")

            if isinstance(received_data, Terminate):
                log.info("Received termination signal in checkpoint background process, closing sender queue")
                sender_queue.put(Terminate())
                sender_queue.close()
                return

            assert isinstance(received_data, tuple), "Received data must be a tuple of (state_dict, checkpoint_path)"
            state_dict, checkpoint_path = received_data

            # Save checkpoint and measure time taken
            start_time = time.monotonic()
            iteration = state_dict["trainer"][0]["iteration"]
            elapsed_time = 0
            succeeded = False
            try:
                checkpoint_handler.save_state_dict_worker(state_dict, checkpoint_path)
                elapsed_time = time.monotonic() - start_time
                log.info(
                    f"Checkpoint saved successfully in background process. Time taken: {elapsed_time:.2f} seconds, iteration: {iteration}"
                )
                succeeded = True
            except Exception as e:
                log.error(f"Error saving checkpoint to {checkpoint_path}: {e}")
                # continue because if the thread exits, the main thread keeps on adding to the queue
            finally:
                if elapsed_time == 0:
                    elapsed_time = time.monotonic() - start_time
                sender_queue.put(SaveDone(iteration, elapsed_time, succeeded))

    finally:
        log.info("Cleaning up: destroying distributed process group")
        torch.distributed.destroy_process_group()


class DistributedCheckpointer(AbstractCheckpointer):
    KEYS_TO_SAVE = ["model", "optim", "scheduler", "trainer"]

    def __init__(
        self,
        config_checkpoint: CheckpointConfig,
        config_job: JobConfig,
        callbacks: Optional[callback.CallBackGroup] = None,
        disable_async: bool = False,
    ):
        super().__init__(config_checkpoint, config_job, callbacks)
        self.config_checkpoint = config_checkpoint
        if config_checkpoint.dcp_async_mode_enabled:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            self.async_mode = AsyncMode.DISABLED

        if disable_async:
            self.async_mode = AsyncMode.DISABLED

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ctx = get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=save_checkpoint_in_background,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                    config_checkpoint,
                    config_job,
                ),
                daemon=True,
            )
            self.mp.start()
            self.cpu_offload_state_dict = None
            self.staging = False
            self.staging_ckpt_file = None
            self.staging_stream = torch.cuda.Stream()

    def keys_to_resume_during_load(self) -> Tuple[Set, Union[str, None]]:
        latest_checkpoint_file = self._read_latest_checkpoint_file()

        resume_keys = []

        if latest_checkpoint_file is not None:
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_path = os.path.join(self.load_dirname, latest_checkpoint_file)
            resume_keys.extend(self.KEYS_TO_SAVE)
        else:
            if self.load_path and not str(self.load_path).endswith(".pt"):
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                if self.load_s3_backend_key:
                    checkpoint_path = f"s3://{self.config_checkpoint.load_from_object_store.bucket}/{checkpoint_path}"
                    if not re.search(r"/checkpoints/iter_\d{9}/?$", checkpoint_path):
                        old_ckpt_path = checkpoint_path
                        # If path doesn't end with specific checkpoint, read latest checkpoint file
                        latest_ckpt_path = os.path.join(checkpoint_path, "checkpoints/latest_checkpoint.txt")
                        if easy_io.exists(latest_ckpt_path, backend_key=self.load_s3_backend_key):
                            checkpoint_file = easy_io.load(
                                latest_ckpt_path, backend_key=self.load_s3_backend_key
                            ).strip()
                            checkpoint_path = f"{checkpoint_path}/checkpoints/{checkpoint_file}"
                        else:
                            log.warning(
                                f"Latest checkpoint file {latest_ckpt_path} not found, load from {old_ckpt_path}"
                            )
                            checkpoint_path = old_ckpt_path

                if self.load_training_state:
                    resume_keys.extend(self.KEYS_TO_SAVE)
                else:
                    resume_keys.append("model")
                    if self.only_load_scheduler_state:
                        resume_keys.append("scheduler")
            else:
                checkpoint_path = None
        if len(self.keys_not_to_resume) > 0:
            for key in self.keys_not_to_resume:
                assert key in self.KEYS_TO_SAVE, f"Invalid key to resume: {key} not in {self.KEYS_TO_SAVE}"
            resume_keys = [key for key in resume_keys if key not in self.keys_not_to_resume]
        return set(resume_keys), checkpoint_path

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_start(model)

        resume_keys, checkpoint_path = self.keys_to_resume_during_load()
        resume_keys = sorted(resume_keys)
        log.critical(f"Resuming ckpt {checkpoint_path} with keys: {resume_keys}")

        iteration = 0

        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            for key in resume_keys:
                load_planner = DefaultLoadPlanner(allow_partial_load=True)
                if hasattr(load_planner, "set_partial_channel_weight"):
                    log.critical(f"set_partial_channel_weight: {self.config_checkpoint.dcp_allow_mismatched_size}")
                    load_planner.set_partial_channel_weight(self.config_checkpoint.dcp_allow_mismatched_size)
                cur_key_ckpt_full_path = os.path.join(checkpoint_path, key)
                log.critical(f"Start loading checkpoint from {checkpoint_path}")
                storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                torch.distributed.barrier()
                log.critical(f"starting {cur_key_ckpt_full_path}", rank0_only=False)
                if key == "model":
                    log.info("- Loading the model...")
                    _model_wrapper = ModelWrapper(model)
                    _state_dict = _model_wrapper.state_dict()

                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    _model_wrapper.load_state_dict(_state_dict)
                elif key == "optim":
                    log.info("- Loading the optimizer...")
                    _optim_wrapper = OptimizerWrapper(model, optimizer)
                    _state_dict = _optim_wrapper.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    _optim_wrapper.load_state_dict(_state_dict)
                elif key == "scheduler":
                    log.info("- Loading the scheduler...")
                    _state_dict = scheduler.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    scheduler.load_state_dict(_state_dict)
                elif key == "trainer":
                    log.info("- Loading the trainer...")
                    _state_dict = {
                        "grad_scaler": grad_scaler.state_dict(),
                        "iteration": iteration,
                    }
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    grad_scaler.load_state_dict(_state_dict["grad_scaler"])
                    iteration = _state_dict["iteration"]
                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")
            if self.callbacks is not None:
                self.callbacks.on_load_checkpoint(model, state_dict=_state_dict)
            log.critical(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")
        else:
            log.info("Training from scratch.")
        torch.cuda.empty_cache()

        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)
        return iteration

    def _async_with_pinned_memory(self, checkpoint_file: str, state_dict: Dict[str, Tuple[Any, str]]) -> None:
        try:
            from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict
        except ImportError as e:
            raise ImportError(
                "Please install the latest PyTorch nightly to use async checkpointing with pinned memory."
            ) from e
        if self.cpu_offload_state_dict is None:
            log.debug(f"Preparing the CPU memory, {time.monotonic()=}.:.2f")
            self.cpu_offload_state_dict = _create_cpu_state_dict(state_dict, pin_memory=True, share_memory=True)

        log.debug(f"Staging the state_dict, {time.monotonic()=}.:.2f")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging = True
            self.staging_ckpt_file = checkpoint_file

        self.maybe_wait_for_staging()

    def maybe_wait_for_staging(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM and self.staging:
            if not self.staging_stream.query():
                self.staging_stream.synchronize()

            def sync_func():
                self.mp_queue_send.put_nowait((self.cpu_offload_state_dict, self.staging_ckpt_file))

            sync_func()
            self.staging = False

    def get_previous_checkpoint_results(self, wait_for: int = 0) -> None:
        """Get the results of previously submitted checkpoints and pass them to callbacks if checkpoint succeeded"""
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            try:
                start_time = time.monotonic()
                while not self.mp_queue_recv.empty() or wait_for > 0:
                    try:
                        ret = self.mp_queue_recv.get(timeout=1)
                        if isinstance(ret, Terminate):
                            log.info("Received termination event from checkpoint background process")
                            break
                        save_done: SaveDone = ret
                        log.logger.info(f"Received checkpoint save result: {save_done}")
                        if self.callbacks is not None and save_done.succeeded:
                            self.callbacks.on_save_checkpoint_success(
                                iteration=save_done.iteration, elapsed_time=save_done.elapsed_time
                            )
                    except queue.Empty:
                        elapsed_time = time.monotonic() - start_time
                        if elapsed_time > wait_for:
                            break
            except (EOFError, BrokenPipeError):
                log.info("Queue was closed by checkpoint background process")

    def get_storage_writer(self, checkpoint_path: str) -> Union[S3StorageWriter, FileSystemWriter]:
        if self.save_to_object_store:
            return S3StorageWriter(
                credential_path=self.config_checkpoint.save_to_object_store.credentials,
                path=checkpoint_path,
            )
        return FileSystemWriter(path=checkpoint_path)

    def get_storage_reader(self, checkpoint_path: str) -> Union[S3StorageReader, FileSystemReader]:
        if self.load_from_object_store:
            return S3StorageReader(
                credential_path=self.config_checkpoint.load_from_object_store.credentials,
                path=checkpoint_path,
            )
        return FileSystemReader(checkpoint_path)

    def save_state_dict_worker(self, to_save_dict: Dict[str, Tuple[Any, str]], checkpoint_file: str) -> None:
        for k, (v, full_checkpoint_path) in to_save_dict.items():
            storage_writer = self.get_storage_writer(full_checkpoint_path)
            dcp.save(
                v,
                storage_writer=storage_writer,
                planner=DefaultSavePlanner(dedup_save_to_lowest_rank=True),
            )

        if distributed.is_rank0():
            print(f"Saving last checkpoint file {checkpoint_file}")
            self._write_latest_checkpoint_file(checkpoint_file)

        log.critical(f"Saved checkpoint to {os.path.join(self.save_dirname, checkpoint_file)}", rank0_only=True)

    def save(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.get_previous_checkpoint_results(wait_for=0)

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_start(model, iteration)

        checkpoint_file = f"iter_{iteration:09}"
        to_save_dict = {
            "model": ModelWrapper(model).state_dict(),
            "optim": OptimizerWrapper(model, optimizer).state_dict(),
            "scheduler": scheduler.state_dict(),
            "trainer": {
                "grad_scaler": grad_scaler.state_dict(),
                "iteration": iteration,
            },
        }
        for k in to_save_dict.keys():
            output_dirname = os.path.join(self.save_dirname, f"iter_{iteration:09}/{k}")
            to_save_dict[k] = (to_save_dict[k], output_dirname)

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint(model, state_dict=to_save_dict)

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._async_with_pinned_memory(checkpoint_file, to_save_dict)
        else:
            start_time = time.monotonic()
            try:
                self.save_state_dict_worker(to_save_dict, checkpoint_file)
            finally:
                if self.callbacks is not None:
                    self.callbacks.on_save_checkpoint_success(
                        iteration=iteration, elapsed_time=time.monotonic() - start_time
                    )

        # This measures exposed (synchronous) checkpoint time, on_save_checkpoint_success()
        # is instead called to measure the entire duration for asynchronous checkpoint for the async case too.
        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    def finalize(self) -> None:
        super().finalize()
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.mp and self.mp.is_alive():
                self.mp_queue_send.put(Terminate())
                self.get_previous_checkpoint_results(wait_for=60)
                self.mp.join()
