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

CLUSTERS = {
    "aws-iad-cs-002": {
        "cluster_name": "aws-iad-cs-002",
        "exclude_nodes": [
            "pool0-0023",
            "pool0-0006",
            "pool0-0028",
            "pool0-0002",
            "pool0-0081",
            "pool0-0159",
            "pool0-0180",
            "pool0-0028",
            "pool0-0183",
            "pool0-1424",
            "pool0-1456",
            "pool0-1688",
            "pool0-0154",
            "pool0-0294",
        ],
        "docker_image": "/lustre/fsw/portfolios/dir/users/snah/dpdata/sqsh/imaginaire4_v10.1.3.sqsh",
        "account": "dir_cosmos_base",
        "adjust_variables": [
            "checkpoint.load_from_object_store.bucket=bucket",
            "checkpoint.load_from_object_store.credentials=credentials/s3_checkpoint.secret",
            "checkpoint.save_to_object_store.bucket=bucket",
            "checkpoint.save_to_object_store.credentials=credentials/s3_checkpoint.secret",
            "dataloader_train.dataloaders.image_data.dataloader.dataset.object_store=s3",
            "dataloader_train.dataloaders.video_data.dataloader.dataset.object_store=s3",
            "dataloader_val.dataloaders.image_data.dataloader.dataset.object_store=s3",
            "dataloader_val.dataloaders.video_data.dataloader.dataset.object_store=s3",
        ],
    },
    "aws": {
        "cluster_name": "aws-iad-cs-002",
        "exclude_nodes": [
            "pool0-0023",
            "pool0-0006",
            "pool0-0028",
            "pool0-0002",
            "pool0-0081",
            "pool0-0159",
            "pool0-0180",
            "pool0-0028",
            "pool0-0183",
            "pool0-1424",
            "pool0-1456",
            "pool0-1688",
            "pool0-0154",
            "pool0-0294",
        ],
        "docker_image": "/lustre/fsw/portfolios/dir/users/snah/dpdata/sqsh/imaginaire4_v10.1.3.sqsh",
        "account": "dir_cosmos_base",
        "adjust_variables": [
            "checkpoint.load_from_object_store.bucket=bucket",
            "checkpoint.load_from_object_store.credentials=credentials/s3_checkpoint.secret",
            "checkpoint.save_to_object_store.bucket=bucket",
            "checkpoint.save_to_object_store.credentials=credentials/s3_checkpoint.secret",
            "dataloader_train.dataloaders.image_data.dataloader.dataset.object_store=s3",
            "dataloader_train.dataloaders.video_data.dataloader.dataset.object_store=s3",
            "dataloader_val.dataloaders.image_data.dataloader.dataset.object_store=s3",
            "dataloader_val.dataloaders.video_data.dataloader.dataset.object_store=s3",
        ],
    },
    "cw-pdx-cs-001": {
        "cluster_name": "cw-pdx-cs-001",
        "partitions": ["batch", "interactive"],
        "remote_path": "/lustre/fsw/portfolios/dir/users/",
        "exclude_nodes": [],
        "docker_image": "nvcr.io/nvidian/imaginaire4:v10.1.3",
        "account": "dir_cosmos_misc",
        "adjust_variables": [
            "checkpoint.load_from_object_store.bucket=checkpoints",
            "checkpoint.load_from_object_store.credentials=credentials/pdx_vfm_checkpoint.secret",
            "checkpoint.save_to_object_store.bucket=checkpoints",
            "checkpoint.save_to_object_store.credentials=credentials/pdx_vfm_checkpoint.secret",
            "dataloader_train.dataloaders.image_data.dataloader.dataset.object_store=pdx",
            "dataloader_train.dataloaders.video_data.dataloader.dataset.object_store=pdx",
            "dataloader_val.dataloaders.image_data.dataloader.dataset.object_store=pdx",
            "dataloader_val.dataloaders.video_data.dataloader.dataset.object_store=pdx",
        ],
    },
    "pdx": {
        "cluster_name": "cw-pdx-cs-001",
        "partitions": ["batch", "interactive"],
        "remote_path": "/lustre/fsw/portfolios/dir/users/",
        "exclude_nodes": [],
        "docker_image": "nvcr.io/nvidian/imaginaire4:v10.1.3",
        "account": "dir_cosmos_misc",
        "adjust_variables": [
            "checkpoint.load_from_object_store.bucket=checkpoints",
            "checkpoint.load_from_object_store.credentials=credentials/pdx_vfm_checkpoint.secret",
            "checkpoint.save_to_object_store.bucket=checkpoints",
            "checkpoint.save_to_object_store.credentials=credentials/pdx_vfm_checkpoint.secret",
            "dataloader_train.dataloaders.image_data.dataloader.dataset.object_store=pdx",
            "dataloader_train.dataloaders.video_data.dataloader.dataset.object_store=pdx",
            "dataloader_val.dataloaders.image_data.dataloader.dataset.object_store=pdx",
            "dataloader_val.dataloaders.video_data.dataloader.dataset.object_store=pdx",
        ],
    },
    "lepton": {
        "cluster_name": "lepton",
        "resource_shape": "gpu.h100-sxm",
        "node_group": "cosmos-aws-h100-02",
        "docker_image": "nvcr.io/nvidian/imaginaire4:v10.1.3",
        "remote_path": "/workspace/log/",
    },
}


def get_executor(
    nnode: int,
    job_group: str,
    job_name: str,
    cluster: str,
    partition: str,
    node_group: str,
    stage_code: bool = True,
    docker_image: str = "/lustre/fsw/portfolios/dir/users/snah/dpdata/sqsh/imaginaire4_v10.1.3.sqsh",
    enable_aps: bool = False,
    user: Optional[str] = None,
    extra_env_vars: Optional[dict] = None,
    user_fp: Optional[str] = None,
    account: Optional[str] = None,
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
            user_fp = f"/lustre/fsw/portfolios/dir/users/{user}"
        elif cluster.lower() == "pdx":
            user_fp = f"/lustre/fsw/portfolios/dir/users/{user}"
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
        ENABLE_ONELOGGER=os.environ.get("ENABLE_ONELOGGER", "True" if cluster == "aws" else "False"),
        **extra_env_vars,
    )

    if cluster.lower() in ["aws", "aws-iad-cs-002", "pdx", "cw-pdx-cs-001"]:
        executor = launcher.SlurmExecutor(
            env_vars=env_vars,
            local_root=os.getcwd(),
            docker_image=docker_image,
            cluster=CLUSTERS[cluster.lower()]["cluster_name"],
            partition=partition,
            account=account if account else CLUSTERS[cluster.lower()]["account"],
            num_gpus=8,
            num_nodes=nnode,
            exclude_nodes=CLUSTERS[cluster.lower()]["exclude_nodes"],
            slurm_workdir=os.path.join(
                user_fp, "cosmos_predict2/_src/predict2/interactive", job_group, job_name, TIME_TAG
            ),
            slurm_logdir=os.path.join(user_fp, "logs", "cosmos_interactive", job_group, job_name),
            autoresubmit_if_failed=True,
            max_auto_resubmit_when_failed=6,
        )
    elif cluster.lower() == "lepton":
        # prepare team-dir (PBSS) secret
        with open("credentials/pbss_dir.secret", "r") as f:
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
            workdir=os.path.join(user_fp, "cosmos_predict2/_src/predict2/interactive", job_group, job_name, TIME_TAG),
            env_vars=env_vars,
            docker_image=docker_image,
            local_root=os.getcwd(),
            image_pull_secrets=["lepton-nvidia-xzeng"],  # image registry secret
            container_port=["8000:tcp", "7777:tcp", "29500:tcp"] + [str(i) + ":tcp" for i in range(8501, 8510)],
        )

    if stage_code:
        executor.stage_code()

    return executor
