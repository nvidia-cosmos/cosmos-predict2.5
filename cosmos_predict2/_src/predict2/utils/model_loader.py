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

from cosmos_predict2._src.common.utils.fsdp_helper import hsdp_device_mesh
from cosmos_predict2._src.imaginaire.config import Config
from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import log, misc
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.checkpointer.dcp import DefaultLoadPlanner, DistributedCheckpointer, ModelWrapper


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
    """
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={experiment_name}"] + experiment_opts)

    # Override checkpoint path if provided
    if s3_checkpoint_dir:
        log.info(f"Overriding config checkpoint path with: {s3_checkpoint_dir}")
        config.checkpoint.load_path = str(s3_checkpoint_dir)

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

    if not enable_fsdp:
        # disable fsdp
        config.model.config.fsdp_shard_size = 1
    with misc.timer("instantiate model"):
        model = instantiate(config.model).cuda()
        # Convert the model parameters to bf16
        model.on_train_start()

    model = load_model_state_dict_from_checkpoint(
        model, config, s3_checkpoint_dir, load_ema_to_reg, local_cache_dir, override_cache
    )

    return model, config


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
    checkpoint_format = "pt" if s3_checkpoint_dir.endswith(".pt") else "dcp"
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
        log.info(f"Loading model cached locally from {local_s3_ckpt_fp}")
        local_state_dict = easy_io.load(local_s3_ckpt_fp, weights_only=INTERNAL)

        # Handle LoRA key mapping if the model uses LoRA and checkpoint is in .pt format
        if (
            hasattr(model, "config")
            and hasattr(model.config, "use_lora")
            and model.config.use_lora
            and checkpoint_format == "pt"
        ):
            log.info("Model uses LoRA, mapping checkpoint keys to model keys with base_layer...")
            mapped_state_dict = {}
            mapped_keys = []
            missing_keys = []

            # Get current model state dict to understand what keys are expected
            model_state_dict = model.state_dict()

            for model_key in model_state_dict.keys():
                if "base_layer." in model_key:
                    # This is a LoRA layer - map from checkpoint key (without base_layer)
                    checkpoint_key = model_key.replace("base_layer.", "")
                    if checkpoint_key in local_state_dict:
                        mapped_state_dict[model_key] = local_state_dict[checkpoint_key]
                        mapped_keys.append(f"{checkpoint_key} -> {model_key}")
                    else:
                        missing_keys.append(model_key)
                elif model_key in local_state_dict:
                    # Direct mapping for non-LoRA keys
                    mapped_state_dict[model_key] = local_state_dict[model_key]
                else:
                    missing_keys.append(model_key)

            if mapped_keys:
                log.info(f"Mapped {len(mapped_keys)} LoRA keys from checkpoint to model (showing first 5):")
                for mapped_key in mapped_keys[:5]:
                    log.info(f"  {mapped_key}")
            if missing_keys:
                log.warning(f"Missing keys in checkpoint: {missing_keys[:10]}... (showing first 10)")

            local_state_dict = mapped_state_dict

        # `strict=False` is needed to avoid errors: `Skipping key ... introduced by TransformerEngine for FP8 in the checkpoint.`
        model.load_state_dict(local_state_dict, strict=False)
    else:
        log.info(f"Loading model from s3 {s3_checkpoint_dir}")

        checkpointer = DistributedCheckpointer(config.checkpoint, config.job, callbacks=None, disable_async=True)

        _model_wrapper = ModelWrapper(model, load_ema_to_reg=load_ema_to_reg if checkpoint_format == "dcp" else False)
        _state_dict = _model_wrapper.state_dict()
        if checkpoint_format == "dcp":
            storage_reader = checkpointer.get_storage_reader(cur_key_ckpt_full_path)
            dcp.load(
                _state_dict,
                storage_reader=storage_reader,
                planner=DefaultLoadPlanner(allow_partial_load=True),
            )
        else:  # pt format
            if "s3://" in s3_checkpoint_dir:
                pt_state_dict = easy_io.load(
                    s3_checkpoint_dir,
                    backend_args={
                        "backend": "s3",
                        "s3_credential_path": "credentials/s3_training.secret",
                    },
                )
            else:
                pt_state_dict = easy_io.load(s3_checkpoint_dir)
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
            # Update the state dict with loaded weights
            # Handle potential key mismatches
            missing_keys = []
            unexpected_keys = []
            for key in _state_dict.keys():
                if key in model_state:
                    _state_dict[key] = model_state[key]
                else:
                    missing_keys.append(key)

            for key in model_state.keys():
                if key not in _state_dict:
                    unexpected_keys.append(key)

            if missing_keys:
                log.warning(f"Missing keys in checkpoint: {missing_keys[:10]}... (showing first 10)")
            if unexpected_keys:
                log.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:10]}... (showing first 10)")
        _model_wrapper.load_state_dict(_state_dict)
        if local_cache_dir is not None:
            log.info(f"Caching model state dict to {local_s3_ckpt_fp}")
            easy_io.dump(model.state_dict(), local_s3_ckpt_fp)

    # Clear unused reserved memory from fp32
    torch.cuda.empty_cache()
    return model


def create_model_from_consolidated_checkpoint_with_fsdp(config: Config) -> ImaginaireModel:
    """
    Instantiate a model, load weights from a consolidated checkpoint, and initialize FSDP if required.

    Args:
        config: The configuration object for the experiment.

    Returns:
        model: The loaded and (optionally) FSDP-wrapped model.
    """
    # To avoid DTensor issues, load the model from a consolidated checkpoint in Tensor format before applying FSDP.
    fsdp_shard_size = config.model.config.fsdp_shard_size
    config.model.config.fsdp_shard_size = 1  # Set to 1 to disable FSDP during model instantiation.
    model = instantiate(config.model)
    # DCP checkpointer does not support loading from a consolidated checkpoint, so we support it here.
    model = load_model_state_dict_from_checkpoint(
        model=model,
        config=config,
        s3_checkpoint_dir=config.checkpoint.load_path,
        load_ema_to_reg=config.checkpoint.load_ema_to_reg,
    )
    # If FSDP is enabled, apply FSDP to the model.
    if fsdp_shard_size > 1:
        config.model.config.fsdp_shard_size = fsdp_shard_size
        fsdp_device_mesh = hsdp_device_mesh(
            sharding_group_size=fsdp_shard_size,
        )
        if hasattr(model, "apply_fsdp") and callable(model.apply_fsdp):
            model.apply_fsdp(fsdp_device_mesh)
        else:
            raise AttributeError(
                "Model does not implement 'apply_fsdp'. Please implement this method to enable FSDP after consolidated checkpoint loading."
            )

    return model
