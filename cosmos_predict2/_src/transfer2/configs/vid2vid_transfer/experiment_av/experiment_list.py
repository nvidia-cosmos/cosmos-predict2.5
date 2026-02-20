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
List of experiments for vid2vid_transfer. Steps to run new experimients:

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


EXPERIMENTS = {}
EXPERIMENTS_LIST = []
for experiments in EXPERIMENTS_LIST:
    for exp_name, _ in experiments.items():
        assert exp_name not in EXPERIMENTS, f"Experiment {exp_name} already exists"
    EXPERIMENTS.update(experiments)
