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
    LOGURU_LEVEL=DEBUG RUN_LOCAL_TESTS=1 pytest -s cosmos_predict2/_src/predict2_multiview/datasets/buttercup_joint_alpamayov2mads_test.py::test_buttercup_joint_alpamayov2mads_dataset
"""

SKIP_LOCAL_TESTS = os.environ.get("RUN_LOCAL_TESTS", "0") != "1"


@pytest.mark.skipif(SKIP_LOCAL_TESTS, reason="Local test")
@pytest.mark.L0
def test_buttercup_joint_alpamayov2mads_dataset():
    os.environ["CAM_T5_EMBEDDINGS_CACHE_DIR"] = "s3://bucket/cosmos_predict2_multiview/cam_t5_embeddings_cache/"
    config = make_config()
    config = override(
        config,
        [
            "--",
            "experiment=buttercup_predict2p1_2b_mv_7views_res720p_fps10_t8_frombase2p1iter45k_jointalpamayov2mads720p",
        ],
    )
    print(config.dataloader_train.dataloaders)
    config.dataloader_train.dataloaders.alpamayo.dataloader.num_workers = 0
    config.dataloader_train.dataloaders.mads.dataloader.num_workers = 0
    config.dataloader_train.dataloaders.alpamayo.dataloader.prefetch_factor = None
    config.dataloader_train.dataloaders.mads.dataloader.prefetch_factor = None
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_train_iter = iter(dataloader_train)
    mads_samples = 0
    alpamayo_samples = 0
    for _ in range(22):
        data_batch = next(dataloader_train_iter)
        url = data_batch["__url__"][0]
        if "AV-V2.2" in url:
            alpamayo_samples += 1
        elif "mads" in url:
            mads_samples += 1
        else:
            raise ValueError(f"Unknown dataset: {url}")
        log.info(f"__url__: {url}")
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
        assert "t5_text_embeddings" not in data_batch, "Found unexpected t5_text_embeddings key in data_batch"
        assert "t5_text_mask" not in data_batch, "Found unexpected t5_text_mask key in data_batch"
        assert len(data_batch["ai_caption"]) == 1, (
            f"Expected ai_caption to be a single string, got {data_batch['ai_caption']}"
        )
        log.info(f"data_batch ai_caption: {data_batch['ai_caption']}")
        first_view_caption = data_batch["ai_caption"][0].split(" -- ")[0]
        log.info(f"first_view_caption: {first_view_caption}")
    assert alpamayo_samples == 20, f"Expected 20 alpamayo samples, got {alpamayo_samples}"
    assert mads_samples == 2, f"Expected 2 mads samples, got {mads_samples}"
    log.info(f"alpamayo_samples: {alpamayo_samples} out of 22")
    log.info(f"mads_samples: {mads_samples} out of 22")
