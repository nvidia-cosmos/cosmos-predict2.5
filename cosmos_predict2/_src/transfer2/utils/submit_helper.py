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

import json
import os
import pwd
import time
from typing import Optional

from cosmos_predict2._src.imaginaire.utils import log


def get_executor(
    nnode: int,
    job_group: str,
    job_name: str,
    cluster: str,
    partition: str,
    node_group: str,
    stage_code: bool = True,
    docker_image: str = "/project/cosmos/snah/dpdata/sqsh/imaginaire4_mcore_v0.0.7_efa.sqsh",
    enable_aps: bool = False,
    user: Optional[str] = None,
    extra_env_vars: Optional[dict] = None,
    user_fp: Optional[str] = None,
):
    import launcher

    if "WANDB_API_KEY" not in os.environ:
        log.critical("Please set WANDB_API_KEY in the environment variables.")
        exit(1)
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

    TIME_TAG = time.strftime("%Y%m%d-%H%M%S")

    if user_fp is None:
        if user is None:
            user = pwd.getpwuid(os.getuid()).pw_name
        assert user is not None, "Cannot get user name."
        if cluster.lower() == "aws":
            user_fp = f"/project/cosmos/{user}"
        elif cluster.lower() == "lepton":
            user_fp = f"/workspace/log/{user}"
    else:
        print(f"Use given user_fp {user_fp} to set slurm_workdir, slurm_logdir, slurm_cachedir")

    extra_env_vars = extra_env_vars or {}
    env_vars = dict(
        WANDB_API_KEY=WANDB_API_KEY,
        WANDB_ENTITY="nvidia-dir",
        TORCH_NCCL_ENABLE_MONITORING="0",
        TORCH_NCCL_AVOID_RECORD_STREAMS="1",
        TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="1800",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        IMAGINAIRE_OUTPUT_ROOT=os.path.join(user_fp, "imaginaire4-output"),
        IMAGINAIRE_CACHE_DIR=os.path.join(user_fp, "imaginaire4-cache"),
        TORCH_HOME=os.path.join(user_fp, "imaginaire4-cache"),
        ENABLE_ONELOGGER="True" if cluster == "aws" else "False",
        **extra_env_vars,
    )

    if cluster.lower() == "aws":
        executor = launcher.SlurmExecutor(
            env_vars=env_vars,
            local_root=os.getcwd(),
            docker_image=docker_image,
            cluster="aws-iad-cs-002",
            partition=partition,
            account="dir_cosmos_base",
            num_gpus=8,
            num_nodes=nnode,
            exclude_nodes=[],
            slurm_workdir=os.path.join(user_fp, "cosmos_predict2/_src/transfer2", job_group, job_name, TIME_TAG),
            slurm_logdir=os.path.join(user_fp, "logs", "cosmos_transfer2", job_group, job_name),
            slurm_cachedir=user_fp,
            enable_aps=enable_aps,
        )
    elif cluster.lower() == "lepton":
        if "aws" in node_group:
            code_credential_path = "credentials/s3_training.secret"
        else:
            code_credential_path = "credentials/pbss_dir.secret"
        with open(code_credential_path, "r") as f:
            secret = json.load(f)
        os.environ["AWS_ACCESS_KEY_ID"] = secret["aws_access_key_id"]  # team-dir
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret["aws_secret_access_key"]  # <secret>
        os.environ["AWS_ENDPOINT_URL"] = secret["endpoint_url"]  # https://pbss.s8k.io
        os.environ["AWS_REGION"] = "us-east-1"

        executor = launcher.LeptonExecutor(
            resource_shape="gpu.h100-sxm",
            node_group=node_group,
            num_gpus=8,
            num_nodes=nnode,
            workdir=os.path.join(user_fp, "cosmos_predict2/_src/transfer2", job_group, job_name, TIME_TAG),
            env_vars=env_vars,
            docker_image=docker_image,
            local_root=os.getcwd(),
            image_pull_secrets=[f"lepton-nvidia-{user}"],  # image registry secret
            container_port=["8000:tcp", "7777:tcp", "29500:tcp"] + [str(i) + ":tcp" for i in range(8501, 8510)],
        )

    if stage_code:
        executor.stage_code()

    return executor
