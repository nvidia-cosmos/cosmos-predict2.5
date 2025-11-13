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

from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import log, misc
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.distill.checkpointer.dcp import (
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
            from cosmos_predict2._src.predict2.distill.utils.text_encoder_cache import cache_text_encoder_checkpoint

            original_ckpt_path = config.model.config.text_encoder_config.ckpt_path
            text_encoder_cache_dir = (
                os.path.join(local_cache_dir, "text_encoder")
                if local_cache_dir
                else "./predict2_distill_cache_ckpts/text_encoder"
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
        if hasattr(config.model, "config") and hasattr(config.model.config, "init_student_with_teacher"):
            log.info("Setting init_student_with_teacher=False for inference to skip teacher checkpoint download")
            config.model.config.init_student_with_teacher = False

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


def load_model_state_dict_from_checkpoint(
    model,
    config,
    s3_checkpoint_dir,
    load_ema_to_reg=False,
    local_cache_dir=None,
    override_cache: bool = False,
):
    load_from_local = "s3://" not in s3_checkpoint_dir
    local_s3_ckpt_fp = s3_checkpoint_dir

    if load_from_local:
        log.info(f"Loading model cached locally from {local_s3_ckpt_fp}")
        # `strict=False` is needed to avoid errors: `Skipping key ... introduced by TransformerEngine for FP8 in the checkpoint.`
        # Use direct torch.load instead of easy_io.load for better performance with large checkpoints
        state_dict = torch.load(local_s3_ckpt_fp, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)
    else:
        log.info(f"Loading model from s3 {s3_checkpoint_dir}")

        checkpointer = DistributedCheckpointer(config.checkpoint, config.job, callbacks=None, disable_async=True)
        cur_key_ckpt_full_path = os.path.join(s3_checkpoint_dir, "model")
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
