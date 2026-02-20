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
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path


def build_no_s3_run(job: dict, local_path: bool = False) -> dict:
    """
    Make a copy of the input config that doesn't require S3 for checkpointing
    and I/O in the callbacks.
    """
    # If local_path is True, use the local path as the load path
    if local_path:
        load_path = job["checkpoint"]["load_path"]
    else:
        model_url = f"s3://bucket/{job['checkpoint']['load_path']}/model"
        load_path = get_checkpoint_path(model_url)
    defaults = job.get("defaults", [])
    no_s3_run = dict(
        defaults=defaults + ["_self_"] if "_self_" not in defaults else defaults,
        job=dict(
            name=f"{job['job']['name']}_no_s3" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
            wandb_mode="offline",
        ),
        checkpoint=dict(
            save_to_object_store=dict(enabled=False, credentials=""),
            load_from_object_store=dict(enabled=False),
            load_path=load_path,
        ),
        trainer=dict(
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                heart_beat=dict(save_s3=False),
                iter_speed=dict(save_s3=False),
                device_monitor=dict(save_s3=False),
                every_n_sample_reg=dict(save_s3=False),
                every_n_sample_ema=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
    )
    return no_s3_run


def deep_update_config_dict(dst: dict, src: dict) -> dict:
    """
    Updates nested dictionaries in the config dictionary (dst) with the values in src dictionary.
    Standard update in hydra only goes one level deep. This function goes arbitrarily deep.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update_config_dict(dst[k], v)
        else:
            dst[k] = v
    return dst
