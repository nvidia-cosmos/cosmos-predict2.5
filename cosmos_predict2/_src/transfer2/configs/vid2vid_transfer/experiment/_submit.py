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
AWS:
PYTHONPATH=. python cosmos_predict2/_src/transfer2/configs/vid2vid_transfer/experiment/_submit.py --exp_name=example_experiment_control_layer14 --run_tag=p0 --nnode=2 --partition=pool0_datahall_a
Lepton:
PYTHONPATH=. python cosmos_predict2/_src/transfer2/configs/vid2vid_transfer/experiment/_submit.py --exp_name=example_experiment_control_layer14 --run_tag=p0 --nnode=2 --cluster lepton --node_group cosmos-aws-h100-02
"""

import hashlib
import os

from cosmos_predict2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list import EXPERIMENTS
from cosmos_predict2._src.transfer2.utils.submit_helper import get_executor

# DOCKER_IMAGE = "/project/cosmos/snah/dpdata/sqsh/imaginaire4_mcore_v0.0.7_efa.sqsh"
AWS_DOCKER_IMAGE = "/project/cosmos/snah/dpdata/sqsh/imaginaire4_v10.1.3.sqsh"
LEPTON_DOCKER_IMAGE = "nvcr.io/nvidian/imaginaire4:v10.1.3"


def run_experiment(
    exp_name: str,
    nnode: int | None,
    run_tag: str,
    partition: str,
    node_group: str,
    exp_name_tag: str = "",
    cluster: str = "aws",
):
    """
    exp_name_tag: in case one wants to run several different training runs with the exact same config in experiment_list.py,
                this tag will update the job.name in the config, so that each run can have their own log/wandb/ckpt folder name.
    """
    exp = EXPERIMENTS[exp_name]

    job_name_for_ckpt = f"{exp.job_name_for_ckpt}_{exp_name_tag}" if exp_name_tag != "" else exp.job_name_for_ckpt

    if nnode:
        exp.nnode = nnode

    user = os.environ.get("USER")
    assert user is not None, "Cannot get user name."
    is_debug_job = job_name_for_ckpt.startswith("eval_") or "debug" in job_name_for_ckpt.lower()
    if cluster.lower() == "lepton":
        docker_image = LEPTON_DOCKER_IMAGE
    else:
        docker_image = AWS_DOCKER_IMAGE

    job_executor = get_executor(
        nnode=exp.nnode,
        job_group=exp.job_group,
        job_name=job_name_for_ckpt,
        cluster=cluster,
        partition=partition,
        node_group=node_group,
        stage_code=True,
        enable_aps=not is_debug_job,
        user=user,
        docker_image=docker_image,
        extra_env_vars={"NVTE_FUSED_ATTN": "0"},
    )

    if cluster.lower() == "aws":
        command_prefix = "python -m "
        cluster_job_name = f"{exp.nnode}N@{exp.job_group}@{job_name_for_ckpt}@{run_tag}"
    elif cluster.lower() == "lepton":
        command_prefix = f"torchrun --nproc_per_node=8 --nnodes={exp.nnode} --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT -m "

        # Lepton job has a limit of 41 characters for job name
        # hacky way to reduce job name length...
        cluster_job_name = exp.job_name_for_ckpt.replace("multicontrol_", "").replace("720p_", "").replace("t24_", "")
        cluster_job_name = (
            cluster_job_name.replace("spaced_layer", "sl").replace("maskprob0.5", "mp").replace("oldsde", "os")
        )
        if exp_name_tag:
            cluster_job_name += f"-{exp_name_tag}"
        if run_tag:
            cluster_job_name += f"-{run_tag}"

        # Create hash of job_group and job_name to keep it concise
        combined_name = f"{exp.job_group}-{cluster_job_name}"
        name_hash = hashlib.md5(combined_name.encode()).hexdigest()
        cluster_job_name = f"{cluster_job_name[:30].lower().replace('_', '-')}-{name_hash[:4]}"

    command = (
        f"{command_prefix} scripts.train "
        f"--config=cosmos_predict2/_src/transfer2/configs/vid2vid_transfer/config.py "
        f"-- experiment={exp.registered_exp_name} job.group={exp.job_group} "
        f"job.name={exp.job_name_for_ckpt} "  # this will distinguish the experiment name in wandb and ckpt path
        f"{' '.join(exp.command_args)}"
    )
    print(f"command: {command}")

    if cluster.lower() == "lepton":
        job_executor.submit_job(
            command=command,
            job_name=cluster_job_name,
            queue_priority=9,
            can_preempt=True,
            can_be_preempted=False,
            visibility="public",
            shared_memory_size=int(1.7 * 1024 * 1024),  # in MB
        )
    else:
        job_executor.submit_job(
            command=command,
            job_name=cluster_job_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="vid2vid_2B_control_edge_480p_random_init")
    parser.add_argument(
        "--run_tag", type=str, default="P0", help="Tag in cluster job name to provide more info to other cluster users."
    )
    parser.add_argument(
        "--exp_name_tag",
        type=str,
        default="",
        help="Tag to distinguish different runs of same config. in case one wants to run several different training runs with the exact same config in experiment_list.py,\
        this tag will update the job.name in the config, so that each run can have their own log/wandb/ckpt folder name.",
    )
    parser.add_argument("--nnode", type=int, default=0)
    parser.add_argument("--cluster", default="aws", choices=["aws", "lepton"], help="Cluster to use")
    parser.add_argument("--partition", type=str, default="pool0_datahall_a", help="Partition to use for aws")
    parser.add_argument(
        "--node_group",
        type=str,
        default="cosmos-aws-h100-02",
        choices=["cosmos-aws-h100-02", "cosmos-azure-a100-01"],
        help="Node group to use for lepton",
    )

    args = parser.parse_args()
    run_experiment(
        args.exp_name, args.nnode, args.run_tag, args.partition, args.node_group, args.exp_name_tag, args.cluster
    )
