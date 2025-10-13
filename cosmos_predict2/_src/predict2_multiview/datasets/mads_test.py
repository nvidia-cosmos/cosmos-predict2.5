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

import os

import pytest

from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.config_helper import override
from cosmos_predict2._src.predict2_multiview.configs.vid2vid.config import make_config

"""
    LOGURU_LEVEL=DEBUG RUN_LOCAL_TESTS=1 pytest -s cosmos_predict2/_src/predict2_multiview/datasets/mads_test.py::test_mads_dataset
"""

SKIP_LOCAL_TESTS = os.environ.get("RUN_LOCAL_TESTS", "0") != "1"


@pytest.mark.skipif(SKIP_LOCAL_TESTS, reason="Local test")
@pytest.mark.L0
@pytest.mark.parametrize(
    "data_train",
    [
        "video_only_cosmos_transfer2_av_mads_mv_20250710_720p_s3",
        "video_only_cosmos_transfer2_av_mads_mv_20250710_720p_29frames_s3",
    ],
)
def test_mads_dataset(data_train: str):
    os.environ["CAM_T5_EMBEDDINGS_CACHE_DIR"] = "s3://bucket/cosmos_predict2_multiview/cam_t5_embeddings_cache/"
    config = make_config()
    config = override(
        config,
        ["--", f"data_train={data_train}"],
    )
    config.dataloader_train.num_workers = 0
    config.dataloader_train.prefetch_factor = None
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_train_iter = iter(dataloader_train)
    for _ in range(10):
        data_batch = next(dataloader_train_iter)
        assert data_batch["video"].shape == (
            1,
            3,
            203,
            720,
            1280,
        ), f"Expected video shape (1, 3, 203, 720, 1280), got {data_batch['video'].shape}"
        assert data_batch["view_indices"].shape == (
            1,
            203,
        ), f"Expected view_indices shape (1, 203), got {data_batch['view_indices'].shape}"
        assert 1 in data_batch["view_indices"].squeeze().tolist(), (
            f"Expected view_indices to contain view_id 1, got {data_batch['view_indices'].squeeze().tolist()}"
        )
        assert data_batch["sample_n_views"].item() == 7, (
            f"Expected sample_n_views 3, got {data_batch['sample_n_views'].item()}"
        )
        assert data_batch["fps"].shape == (1,), f"Expected fps shape (1,), got {data_batch['fps'].shape}"
        assert data_batch["fps"].item() == 30, f"Expected fps 30, got {data_batch['fps'].item()}"
        assert data_batch["t5_text_embeddings"].shape == (
            1,
            3584,
            1024,
        ), f"Expected t5_text_embeddings shape (1, 3584, 1024), got {data_batch['t5_text_embeddings'].shape}"
        assert data_batch["t5_text_mask"].shape == (
            1,
            3584,
        ), f"Expected t5_text_mask shape (1, 3584), got {data_batch['t5_text_mask'].shape}"
        assert data_batch["t5_text_mask"].sum().item() == 3584, (
            f"Expected t5_text_mask to be all ones, got {data_batch['t5_text_mask'].sum().item()}"
        )
        assert len(data_batch["ai_caption"]) == 1, (
            f"Expected ai_caption to be a single string, got {data_batch['ai_caption']}"
        )
        log.info(f"data_batch ai_caption: {data_batch['ai_caption']}")
        first_view_caption = data_batch["ai_caption"][0].split(" -- ")[0]
        log.info(f"first_view_caption: {first_view_caption}")
