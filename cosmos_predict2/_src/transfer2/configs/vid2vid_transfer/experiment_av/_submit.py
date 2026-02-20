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
PYTHONPATH=. python cosmos_predict2/_src/transfer2/configs/vid2vid_transfer/experiment_av/_submit.py --exp_name=vid2vid_2B_control_480p_control_layer14_av --run_tag=node1 --nnode=1 --partition=pool0_datahall_a

PYTHONPATH=. python cosmos_predict2/_src/transfer2/configs/vid2vid_transfer/experiment_av/_submit.py --exp_name=vid2vid_2B_control_true_720p_control_layer14_av_full_65k_low_sigma --run_tag=p0 --nnode=16 --partition=pool0_datahall_a
"""

import os

from cosmos_predict2._src.transfer2.configs.vid2vid_transfer.experiment_av.experiment_list import EXPERIMENTS
from cosmos_predict2._src.transfer2.utils.submit_helper import get_executor

DOCKER_IMAGE = "/project/cosmos/snah/dpdata/sqsh/imaginaire4_mcore_v0.0.7_efa.sqsh"


def run_experiment(exp_name: str, nnode: int | None, run_tag: str, partition: str, exp_name_tag: str = ""):
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

    slurm_executor = get_executor(
        nnode=exp.nnode,
        job_group=exp.job_group,
        job_name=job_name_for_ckpt,
        partition=partition,
        stage_code=True,
        enable_aps=not is_debug_job,
        user=user,
        docker_image=DOCKER_IMAGE,
        extra_env_vars={"NVTE_FUSED_ATTN": "0"},
        user_fp="/project/cosmos/huling/projects/yfl_dir/",
    )

    command = (
        f"python -m scripts.train "
        f"--config=cosmos_predict2/_src/transfer2/configs/vid2vid_transfer/config.py "
        f"-- experiment={exp.registered_exp_name} job.group={exp.job_group} "
        f"job.name={exp.job_name_for_ckpt} "  # this will distinguish the experiment name in wandb and ckpt path
        f"{' '.join(exp.command_args)}"
    )
    print(f"command: {command}")

    # cp_cmd = f"/bin/cp /project/cosmos/{user}/projects/nvrun/imaginaire4/projects/edify_video/v4/patches/parallel_state_wo_gloo_TE_1_10_0.py /usr/local/lib/python3.10/dist-packages/megatron/core/parallel_state.py"
    # second_cp_cmd = f"/bin/cp /project/cosmos/{user}/projects/nvrun/imaginaire4/projects/edify_video/v4/patches/pytorch_constants.py /usr/local/lib/python3.10/dist-packages/torch/distributed/constants.py"
    # command = f"{cp_cmd} && {second_cp_cmd} && {command}"

    cluster_job_name = f"{exp.nnode}N@{exp.job_group}@{job_name_for_ckpt}@{run_tag}"
    slurm_executor.submit_job(
        command=command,
        job_name=cluster_job_name,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="vid2vid_2B_control_480p_control_layer14_av")
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
    parser.add_argument("--partition", type=str, default="pool0_datahall_a")

    args = parser.parse_args()
    run_experiment(args.exp_name, args.nnode, args.run_tag, args.partition, args.exp_name_tag)
