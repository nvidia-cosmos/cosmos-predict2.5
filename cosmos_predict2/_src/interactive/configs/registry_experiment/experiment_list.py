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
List of experiments for distillation. Steps to run new experimients:

- Find the registered experiment config in either experiments_dmd2_*.py or experiments_rcm.py that's closest to the experiment to be run.
- Add a new experiment entry here in the dict of the appropriate category
- Submit the job with _submit.py
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Experiment:
    registered_exp_name: str
    job_name_for_ckpt: str
    job_group: str
    nnode: int
    command_args: List[str]


dmd2_predict2p5_experiments = {
    # An example experiment for tutorial/README
    "cosmos_interactive_dmd2_trigflow_distill_cosmos_predict2_2B_example": Experiment(
        registered_exp_name="dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V",
        job_name_for_ckpt="cosmos_interactive_dmd2_trigflow_distill_cosmos_predict2_2B_example",
        job_group="cosmos2_interactive",
        nnode=2,
        command_args=[
            "model_parallel.context_parallel_size=4",
            "trainer.max_iter=120",
        ],
    ),
    # the most basic: lr=1e-6, lr_crit=2e-7, lr_disc=2e-7, warmup_steps=1,
    # adam optimizer betas=(0.9, 0.999), use grad clip with norm 1
    # no discriminator
    "cosmos_interactive_dmd2_trigflow_distill_predict2p5_2B_TnI2V_basic": Experiment(
        registered_exp_name="dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V",
        job_name_for_ckpt="cosmos_interactive_dmd2_trigflow_distill_predict2p5_2B_TnI2V_basic",
        job_group="cosmos2_interactive",
        nnode=32,
        command_args=[
            "model_parallel.context_parallel_size=4",
        ],
    ),
    # 14B
    "cosmos_interactive_dmd2_trigflow_distill_predict2p5_14B_TnI2V_basic": Experiment(
        registered_exp_name="dmd2_trigflow_distill_cosmos_predict2_14B_bidirectional_TnI2V",
        job_name_for_ckpt="cosmos_interactive_dmd2_trigflow_distill_predict2p5_14B_TnI2V_basic",
        job_group="cosmos2_interactive",
        nnode=32,
        command_args=[
            "model_parallel.context_parallel_size=8",
        ],
    ),
}


dmd2_transfer2p5_experiments = {
    "cosmos_interactive_dmd2_trigflow_distill_transfer2p5_2B_edge_example": Experiment(
        registered_exp_name="dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_edge",
        job_name_for_ckpt="cosmos_interactive_dmd2_trigflow_distill_transfer2p5_2B_edge_example",
        job_group="cosmos2_interactive",
        nnode=2,
        command_args=[],
    ),
    # old naming convention, kept for verification of old checkpoint. To be removed after verification.
    "cosmos_fastgen_dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_edge_bugfix_v2": Experiment(
        registered_exp_name="dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_edge",
        job_name_for_ckpt="cosmos_fastgen_dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_edge_bugfix_v2",
        job_group="cosmos2_interactive",
        nnode=32,
        command_args=[],
    ),
    "cosmos_fastgen_dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_edge_v1g": Experiment(
        registered_exp_name="dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_edge",
        job_name_for_ckpt="cosmos_fastgen_dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_edge_v1g",
        job_group="cosmos2_interactive",
        nnode=32,
        command_args=[
            "model.config.use_neg_prompt_str=True",
        ],
    ),
}

EXPERIMENTS = {}
EXPERIMENTS_LIST = [
    dmd2_predict2p5_experiments,
    dmd2_transfer2p5_experiments,
]
for experiments in EXPERIMENTS_LIST:
    for exp_name, _ in experiments.items():
        assert exp_name not in EXPERIMENTS, f"Experiment {exp_name} already exists"
    EXPERIMENTS.update(experiments)


# Action-conditioned DMD2 distillation experiments
dmd2_action_conditioned_experiments = {
    # Bridge dataset - 13 frame action-conditioned prediction at 256x320
    "cosmos_interactive_dmd2_trigflow_distill_predict2p5_2B_action_conditioned_bridge_13frame_256x320": Experiment(
        registered_exp_name="dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_256x320",
        job_name_for_ckpt="cosmos_interactive_dmd2_trigflow_distill_predict2p5_2B_action_conditioned_bridge_13frame_256x320",
        job_group="cosmos3_interactive_action_conditioned",
        nnode=4,
        command_args=[
            "model_parallel.context_parallel_size=1",
        ],
    ),
    # Bridge dataset - 13 frame action-conditioned prediction at 480x640
    "cosmos_interactive_dmd2_trigflow_distill_predict2p5_2B_action_conditioned_bridge_13frame_480x640": Experiment(
        registered_exp_name="dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_480x640",
        job_name_for_ckpt="cosmos_interactive_dmd2_trigflow_distill_predict2p5_2B_action_conditioned_bridge_13frame_480x640",
        job_group="cosmos3_interactive_action_conditioned",
        nnode=4,
        command_args=[
            "model_parallel.context_parallel_size=1",
        ],
    ),
}

EXPERIMENTS_LIST.append(dmd2_action_conditioned_experiments)
for exp_name, _ in dmd2_action_conditioned_experiments.items():
    assert exp_name not in EXPERIMENTS, f"Experiment {exp_name} already exists"
EXPERIMENTS.update(dmd2_action_conditioned_experiments)
