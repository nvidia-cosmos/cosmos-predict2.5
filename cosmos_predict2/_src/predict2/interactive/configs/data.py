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

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.interactive.datasets.dataset_action_warmup import ActionDatasetSFWarmup


def get_sampler(dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=1234,
    )


dataset_gr00t_gr1_warmup = L(ActionDatasetSFWarmup)(
    data_path="datasets/gr1_warmup_regenerated_4step",
    cr1_embeddings_path="cr1_empty_string_text_embeddings.pt",
)

dataset_gr00t_g1_warmup = L(ActionDatasetSFWarmup)(
    data_path="datasets/g1_warmup_regenerated_4step",
    cr1_embeddings_path="cr1_empty_string_text_embeddings.pt",
)

# ----------- Dataloaders -----------

DATALOADER_DEFAULTS = dict(
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)


def make_dataloader(dataset):
    return L(DataLoader)(
        dataset=dataset,
        sampler=L(get_sampler)(dataset=dataset),
        **DATALOADER_DEFAULTS,
    )


def register_interactive_data():
    cs = ConfigStore.instance()

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="gr00t_gr1_warmup",
        node=make_dataloader(dataset_gr00t_gr1_warmup),
    )
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="gr00t_g1_warmup",
        node=make_dataloader(dataset_gr00t_g1_warmup),
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="gr00t_gr1_warmup",
        node=make_dataloader(dataset_gr00t_gr1_warmup),
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="gr00t_g1_warmup",
        node=make_dataloader(dataset_gr00t_g1_warmup),
    )
