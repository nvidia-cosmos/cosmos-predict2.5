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

# This script contains the code for multi-aspect ratio shard iterator

import os
import random
import time
from copy import deepcopy

import torch

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False
from webdataset.utils import pytorch_worker_info

from cosmos_predict2._src.imaginaire.datasets.webdataset.config.schema import TarSample
from cosmos_predict2._src.imaginaire.datasets.webdataset.distributors.multi_aspect_ratio import (
    ShardlistMultiAspectRatio,
)
from cosmos_predict2._src.imaginaire.utils import log


class ShardlistMultiAspectRatioParallelSync(ShardlistMultiAspectRatio):
    r"""
    An iterable dataset that parses and yields tar files.
    This distributor is based on ShardlistMultiAspectRatio.
    Additionally, it allows users to synchronize inputs for context/tensor parallelism. This is achieved by specifying the context/tensor parallel group size during initialization.
    """

    def __init__(self, **kwargs):
        r"""Create a multi-aspect ratio ShardList."""
        super().__init__(**kwargs)
        self.enable_parallel()

    def _obtain_node_worker_url_mapping(
        self,
        url_aspect_split: dict[str, list[TarSample]],
        num_urls_per_worker: int,
        group_id: int,
        group_num: int,
        worker_id: int,
        num_workers: int,
    ):
        r"""This function obtains the worker-URL mapping. It assigns the tar list seen by
        each workers.

        Args:
            url_aspect_split (dict[list[TarSample]]: TarSample split by aspect ratio
            num_urls_per_worker (int): Number of tar files seen by each worker
            group_id (int): Rank of the current GPU
            group_num (int): Total number of groups
            worker_id (int): ID for the current worker in the dataloader
            num_workers (int): Total number of workers in the dataloader

        Returns:
            URL list for the current worker
        """
        assert self.split_by_node is True and self.split_by_worker is True

        # First chunk the tars
        chunk_mappings = []
        for aspect_ratio in url_aspect_split:
            samples_asp = url_aspect_split[aspect_ratio]
            nchunks_asp = int(len(samples_asp) / num_urls_per_worker)
            for chunk_id in range(nchunks_asp):
                chunk_mappings.append((aspect_ratio, samples_asp[chunk_id::nchunks_asp]))
        # Split by rank and workers
        chunk_mappings = chunk_mappings[group_id::group_num]
        chunk_mappings = chunk_mappings[worker_id::num_workers]

        assert len(chunk_mappings) == 1, f"Length of chunk_mappings {len(chunk_mappings)} != 1"
        return chunk_mappings[0][1]

    def enable_parallel(self):
        # Ranks of the same pp/tp/cp group will have the same dp rank and thus share the same group id.
        self.group_id = parallel_state.get_data_parallel_rank()
        # The size of the group is how many GPUs we use to process one batch of data.
        self.group_size = torch.distributed.get_world_size() // parallel_state.get_data_parallel_world_size()

    def obtain_url_list(self):
        r"""Return an iterator over the shards."""

        rank, world_size, worker_id, num_workers = pytorch_worker_info()

        num_groups = world_size // self.group_size
        # Setting epoch and start index
        if self.resume_flag:
            self.epoch = int(os.environ.get("WDS_EPOCH_NUM", 0))

            # This tells us number of chunks that have been seen by one GPU
            self.start_index = int(os.environ.get("WDS_START_INDEX", 0)) // self.chunk_size

        url_aspect_split = deepcopy(self.url_aspect_split)

        # nworkers_all is no longer world_size * num_workers, since self.group_size workers duplicate
        nworkers_all = num_groups * num_workers

        if self.verbose:
            log.info(f"Total {nworkers_all} workers are in effect")

        # Perform DDP equalization
        url_aspect_split, num_urls_per_worker = self._ddp_equalize(url_aspect_split, nworkers_all)

        # Form a mapping of url_aspect_split to node and workers
        urls = self._obtain_node_worker_url_mapping(
            url_aspect_split, num_urls_per_worker, self.group_id, num_groups, worker_id, num_workers
        )

        if self.shuffle:
            random.Random(self.group_id).shuffle(urls)

        # This tells us the number of chunks seen by one worker.
        # Do not iterate over the seen chunks.
        start_index_per_worker = self.start_index // num_workers
        if not self.is_infinite_loader:
            urls = urls[start_index_per_worker:]

        if self.verbose:
            log.info(
                f"Rank {rank}, group {self.group_id}, worker {worker_id} of {num_workers}, group_size {self.group_size} got {len(urls)} urls, first five are {urls[:5]}"
            )

        return urls

    def __iter__(self):
        url_list = self.obtain_url_list()

        if self.is_infinite_loader:
            while True:
                cur_time = time.time_ns()
                random.Random(cur_time).shuffle(url_list)
                for url in url_list:
                    yield dict(url=url)
        else:
            for url in url_list:
                yield dict(url=url)
