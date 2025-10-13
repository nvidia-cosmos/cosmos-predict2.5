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
Usage:
    pytest --L1 -s cosmos_predict2/_src/imaginaire/datasets/webdataset/distributors/multi_aspect_ratio_v2_test.py
"""

import os

import pytest

from cosmos_predict2._src.imaginaire.config import ObjectStoreConfig
from cosmos_predict2._src.imaginaire.datasets.webdataset.config.schema import DatasetInfo, TarSample
from cosmos_predict2._src.imaginaire.datasets.webdataset.distributors.multi_aspect_ratio_v2 import (
    ShardlistMultiAspectRatioInfinite,
)
from cosmos_predict2._src.imaginaire.utils import log, misc


@pytest.mark.skip(reason="not a test, it prepare test data")
def generate_data(counts):
    urls = []
    for aspect_key, num_urls in zip(["1:1", "4:3", "3:4", "16:9", "9:16"], counts):
        dataset_info = DatasetInfo(
            object_store_config=ObjectStoreConfig(), wdinfo=[], opts={"aspect_ratio": aspect_key}
        )
        for i in range(num_urls):
            urls.append(
                TarSample(
                    path=f"this_is_a_url_to_a_tar_file_{i:09d}",
                    root="root/",
                    keys=[],
                    meta=dataset_info,
                    dset_id="mock",
                )
            )
    log.info(f"Generated a total of {len(urls)} urls")
    return urls


@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmpdir):
    # Setup: run before the test
    rank = os.environ.get("RANK", None)
    world_size = os.environ.get("WORLD_SIZE", None)
    worker = os.environ.get("WORKER", None)
    num_workers = os.environ.get("NUM_WORKERS", None)

    yield  # this is where the testing happens

    # Teardown: run after the test
    def restore_env(name, value):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ.set(name, value)

    restore_env("RANK", rank)
    restore_env("WORLD_SIZE", world_size)
    restore_env("WORKER", worker)
    restore_env("NUM_WORKERS", num_workers)


@misc.timer("test_shardlist_multi_aspect_ratio_infinite_mini")
@pytest.mark.L1
def test_shardlist_multi_aspect_ratio_infinite_mini():
    urls = generate_data([100, 100, 100, 100, 100])

    aspect_ratios = set()
    for worker_id in range(16):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["WORKER"] = str(worker_id)
        os.environ["NUM_WORKERS"] = "16"

        distributor = ShardlistMultiAspectRatioInfinite(verbose=True, shuffle=False)
        distributor.set_urls(urls)

        distributor_iter = iter(distributor)

        # Print first 10 URLs produced by the distributor
        for i in range(2):
            url = next(distributor_iter)
            aspect_ratios.add(url["url"].meta.opts["aspect_ratio"])

    assert len(aspect_ratios) == 5


# Test on a large dataset. Takes 1 minute
@misc.timer("test_shardlist_multi_aspect_ratio_infinite_large")
@pytest.mark.L1
def test_shardlist_multi_aspect_ratio_infinite_large():
    urls = generate_data([123456, 234567, 10000, 500000, 500000])

    aspect_ratios = set()
    for worker_id in range(7):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["WORKER"] = str(worker_id)
        os.environ["NUM_WORKERS"] = "7"

        distributor = ShardlistMultiAspectRatioInfinite(verbose=True, shuffle=False)
        distributor.set_urls(urls)

        distributor_iter = iter(distributor)

        # Print first 10 URLs produced by the distributor
        for i in range(2):
            url = next(distributor_iter)
            aspect_ratios.add(url["url"].meta.opts["aspect_ratio"])

    assert len(aspect_ratios) == 5
