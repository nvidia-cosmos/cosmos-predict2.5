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

import importlib
import os

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemReader

from cosmos_predict2._src.imaginaire.checkpointer.s3_filesystem import S3StorageReader
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import distributed, log, misc
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.interactive.checkpointer.dcp import (
    DefaultLoadPlanner,
    DistributedCheckpointer,
    ModelWrapper,
)


def load_model_from_checkpoint(
    experiment_name,
    s3_checkpoint_dir,
    config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
    enable_fsdp=False,
    load_ema_to_reg=False,
    instantiate_ema=True,
    seed=0,
    local_cache_dir=None,
    override_cache: bool = False,
    experiment_opts: list[str] = [],
    skip_teacher_init: bool = True,
):
    """
    experiment_name: experiment name
    s3_checkpoint_dir: s3 path to iteration_model
    s3_credential_path: s3 credential path, if None, use credential from config
    config_file: config file path
    enable_fsdp: enable fsdp
    load_ema_to_reg: load ema as regular model
    seed: random seed
    local_cache_dir: local cache directory, if None, do not cache
    override_cache: override cache, if True, override cache if local cache exists
    skip_teacher_init: if True, skip loading teacher checkpoint during inference (faster)
    """
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={experiment_name}"] + experiment_opts)

    if load_ema_to_reg:
        config.model.config.ema.enabled = False

    if instantiate_ema is False and config.model.config.ema.enabled:
        config.model.config.ema.enabled = False

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    misc.set_random_seed(seed=seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True

    log.info(f"Loading model from {s3_checkpoint_dir}")

    # Cache text encoder checkpoint to avoid re-downloading every time
    if hasattr(config.model, "config") and hasattr(config.model.config, "text_encoder_config"):
        if config.model.config.text_encoder_config is not None:
            from cosmos_predict2._src.interactive.utils.text_encoder_cache import cache_text_encoder_checkpoint

            original_ckpt_path = config.model.config.text_encoder_config.ckpt_path
            text_encoder_cache_dir = (
                os.path.join(local_cache_dir, "text_encoder")
                if local_cache_dir
                else "./cosmos3_interactive_cache_ckpts/text_encoder"
            )

            cached_ckpt_path = cache_text_encoder_checkpoint(
                s3_ckpt_path=original_ckpt_path,
                cache_dir=text_encoder_cache_dir,
                s3_credential_path=config.model.config.text_encoder_config.s3_credential_path,
            )

            if cached_ckpt_path != original_ckpt_path:
                log.info(f"Using cached text encoder checkpoint: {cached_ckpt_path}")
                config.model.config.text_encoder_config.ckpt_path = cached_ckpt_path

    # Optionally skip teacher checkpoint loading during inference to avoid unnecessary S3 downloads
    # The teacher weights will be overwritten by the trained checkpoint anyway
    if skip_teacher_init:
        if hasattr(config.model, "config") and hasattr(config.model.config, "load_teacher_weights"):
            log.info("Setting load_teacher_weights=False for inference to skip teacher checkpoint download")
            config.model.config.load_teacher_weights = False

    if not enable_fsdp:
        # disable fsdp
        config.model.config.fsdp_shard_size = 1
    with misc.timer("instantiate model"):
        model = instantiate(config.model).cuda()
        # Convert the model parameters to bf16
        model.on_train_start()

    print(f"Loading checkpoint from {s3_checkpoint_dir}")
    model = load_model_state_dict_from_checkpoint(
        model, config, s3_checkpoint_dir, load_ema_to_reg, local_cache_dir, override_cache
    )

    return model, config


def _process_pt_state_dict_for_ema(state_dict, model, load_ema_to_reg):
    """
    Process .pt checkpoint state dict to handle EMA and non-EMA variants.

    Consolidated .pt checkpoints can be in different formats:
    1. Non-EMA checkpoint: Keys like 'net.xxx' - load directly
    2. EMA-only checkpoint: Keys like 'net.xxx' (exported from EMA weights) - load directly
    3. Full checkpoint with EMA: Has both 'net.xxx' and 'net_ema.xxx' keys

    When load_ema_to_reg=True:
    - If checkpoint has 'net_ema.xxx' keys, map them to 'net.xxx' for loading into regular model
    - If checkpoint only has 'net.xxx' keys (common for exported EMA checkpoints), load directly

    Args:
        state_dict: The loaded state dict from .pt file
        model: The model instance (used to get expected keys)
        load_ema_to_reg: Whether to load EMA weights into regular model

    Returns:
        Processed state dict ready for loading
    """
    if not load_ema_to_reg:
        return state_dict

    # Check if checkpoint has net_ema keys
    has_net_ema_keys = any(k.startswith("net_ema.") for k in state_dict.keys())
    has_net_keys = any(k.startswith("net.") for k in state_dict.keys())

    if has_net_ema_keys:
        # Checkpoint has net_ema keys - map them to net keys for loading into regular model
        log.info("Processing EMA checkpoint: mapping net_ema.* keys to net.* for load_ema_to_reg=True")
        mapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("net_ema."):
                # Map net_ema.xxx -> net.xxx
                new_key = key.replace("net_ema.", "net.", 1)
                mapped_state_dict[new_key] = value
            elif not key.startswith("net."):
                # Keep non-net keys (e.g., other components)
                mapped_state_dict[key] = value
            # Skip net.* keys when we have net_ema.* keys and load_ema_to_reg=True
        return mapped_state_dict
    elif has_net_keys:
        # Checkpoint only has net keys (common for exported EMA checkpoints)
        # Load directly - the checkpoint was already exported from EMA weights
        log.info("Loading EMA-exported checkpoint with net.* keys directly")
        return state_dict
    else:
        # Checkpoint has neither net. nor net_ema. prefix - load as-is
        log.info("Loading checkpoint without net./net_ema. prefix")
        return state_dict


def load_model_state_dict_from_checkpoint(
    model,
    config,
    s3_checkpoint_dir,
    load_ema_to_reg=False,
    local_cache_dir=None,
    override_cache: bool = False,
):
    if s3_checkpoint_dir is not None:
        s3_checkpoint_dir = str(s3_checkpoint_dir)

    # Detect checkpoint format based on file extension
    checkpoint_format = "pt" if s3_checkpoint_dir.endswith(".pt") else "dcp"

    # Build the full checkpoint path
    if s3_checkpoint_dir.startswith("s3:"):
        if checkpoint_format == "pt":
            cur_key_ckpt_full_path = s3_checkpoint_dir
        elif s3_checkpoint_dir.rstrip("/").endswith("/model"):
            cur_key_ckpt_full_path = s3_checkpoint_dir
        else:
            cur_key_ckpt_full_path = os.path.join(s3_checkpoint_dir, "model")
    else:
        cur_key_ckpt_full_path = s3_checkpoint_dir

    from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path

    load_from_local = True
    local_s3_ckpt_fp = get_checkpoint_path(cur_key_ckpt_full_path)

    if load_from_local:
        # Load on rank0 only and broadcast for efficiency
        if distributed.is_rank0():
            log.info(f"Loading model cached locally from {local_s3_ckpt_fp}")
            # `strict=False` is needed to avoid errors: `Skipping key ... introduced by TransformerEngine for FP8 in the checkpoint.`
            # Use direct torch.load instead of easy_io.load for better performance with large checkpoints
            state_dict = torch.load(local_s3_ckpt_fp, map_location="cpu", weights_only=False)

            # Handle EMA checkpoint loading
            state_dict = _process_pt_state_dict_for_ema(state_dict, model, load_ema_to_reg)

            model.load_state_dict(state_dict, strict=False)
        # Synchronize model states from rank 0 to all other ranks
        distributed.sync_model_states(model, src=0)
    elif checkpoint_format == "pt":
        # .pt format - load on rank0 only and broadcast
        log.info(f"Loading .pt checkpoint from {s3_checkpoint_dir}")
        if distributed.is_rank0():
            if "s3://" in s3_checkpoint_dir:
                pt_state_dict = easy_io.load(
                    s3_checkpoint_dir,
                    backend_args={
                        "backend": "s3",
                        "s3_credential_path": "credentials/s3_training.secret",
                    },
                )
            else:
                pt_state_dict = torch.load(s3_checkpoint_dir, map_location="cpu", weights_only=False)

            # Handle different .pt checkpoint formats
            if "model" in pt_state_dict:
                # Checkpoint contains multiple components (model, optimizer, etc.)
                model_state = pt_state_dict["model"]
            elif "state_dict" in pt_state_dict:
                # Alternative format
                model_state = pt_state_dict["state_dict"]
            else:
                # Assume the checkpoint is the state dict itself
                model_state = pt_state_dict

            # Handle EMA checkpoint loading
            model_state = _process_pt_state_dict_for_ema(model_state, model, load_ema_to_reg)

            # Load state dict with strict=False for compatibility
            model.load_state_dict(model_state, strict=False)

        # Synchronize model states from rank 0 to all other ranks
        distributed.sync_model_states(model, src=0)

        # Optionally cache for future runs
        if local_cache_dir is not None and distributed.is_rank0():
            log.info(f"Caching model state dict to {local_s3_ckpt_fp}")
            easy_io.dump(model.state_dict(), local_s3_ckpt_fp)
    else:
        # DCP format
        log.info(f"Loading DCP checkpoint from s3 {s3_checkpoint_dir}")

        checkpointer = DistributedCheckpointer(config.checkpoint, config.job, callbacks=None, disable_async=True)
        storage_reader = checkpointer.get_storage_reader(cur_key_ckpt_full_path)
        _model_wrapper = ModelWrapper(model, load_ema_to_reg=load_ema_to_reg)

        _state_dict = _model_wrapper.state_dict()

        dcp.load(
            _state_dict,
            storage_reader=storage_reader,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        _model_wrapper.load_state_dict(_state_dict)
        if local_cache_dir is not None:
            log.info(f"Caching model state dict to {local_s3_ckpt_fp}")
            easy_io.dump(model.state_dict(), local_s3_ckpt_fp)

    # Clear unused reserved memory from fp32
    torch.cuda.empty_cache()
    return model


def get_storage_reader(checkpoint_path: str, credential_path: str | None = None):
    if "s3://" in checkpoint_path and credential_path:
        storage_reader = S3StorageReader(
            credential_path=credential_path,
            path=checkpoint_path,
        )
    else:
        storage_reader = FileSystemReader(checkpoint_path)
    return storage_reader
