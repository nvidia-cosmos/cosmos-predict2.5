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
    LOGURU_LEVEL=DEBUG RUN_LOCAL_TESTS=1 pytest -s cosmos_predict2/_src/predict2_multiview/datasets/alpamayo_test.py::test_alpamayo_dataset
"""

SKIP_LOCAL_TESTS = os.environ.get("RUN_LOCAL_TESTS", "0") != "1"


@pytest.mark.skipif(SKIP_LOCAL_TESTS, reason="Local test")
@pytest.mark.L0
@pytest.mark.parametrize(
    "data_train",
    [
        "alpamayo_v2_7cameras_tar_sample7views_29frames_res720p_norepeat_hybrid_captions",
        "alpamayo_v2_7cameras_tar_sample7views_85framesto29_res720p_1cap_norepeat",
        "alpamayo_v2_7cameras_tar_sample7views_85framesto29_res720p_noviewprefix_1cap_norepeat",
    ],
)
def test_alpamayo_dataset(data_train: str):
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
    for _ in range(2):
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
        captions = data_batch["ai_caption"][0].split(" -- ")
        if "1cap" in data_train:
            assert len(captions) == 1, f"Expected 1 caption, got {len(captions)}"
        else:
            assert len(captions) == 7, f"Expected 7 captions, got {len(captions)}"
        for caption in captions:
            log.info(f"caption: {caption}")
            if "noviewprefix" not in data_train:
                prefix_tele = "The video is captured from a telephoto camera mounted on a car. The camera is facing"
                prefix_wide = "The video is captured from a camera mounted on a car. The camera is facing"
                assert caption.startswith(prefix_tele) or caption.startswith(prefix_wide)


"""
    LOGURU_LEVEL=DEBUG RUN_LOCAL_TESTS=1 pytest -s cosmos_predict2/_src/predict2_multiview/datasets/alpamayo_test.py::test_alpamayo_dataset_joint_alpamayo1capnoviewprefix_allcapsviewprefix_720p_29frames_hybrid_captions
"""


@pytest.mark.skipif(SKIP_LOCAL_TESTS, reason="Local test")
@pytest.mark.L0
def test_alpamayo_dataset_joint_alpamayo1capnoviewprefix_allcapsviewprefix_720p_29frames_hybrid_captions():
    os.environ["CAM_T5_EMBEDDINGS_CACHE_DIR"] = "s3://bucket/cosmos_predict2_multiview/cam_t5_embeddings_cache/"
    config = make_config()
    config = override(
        config,
        [
            "--",
            "experiment=buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames",
        ],
    )
    config.dataloader_train.dataloaders.alpamayo_1cap.dataloader.num_workers = 0
    config.dataloader_train.dataloaders.alpamayo_allcaps.dataloader.num_workers = 0
    config.dataloader_train.dataloaders.alpamayo_1cap.dataloader.prefetch_factor = None
    config.dataloader_train.dataloaders.alpamayo_allcaps.dataloader.prefetch_factor = None
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_train_iter = iter(dataloader_train)
    for i in range(4):
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
        assert "t5_text_embeddings" not in data_batch, (
            f"Expected t5_text_embeddings not in data_batch, got {data_batch.keys()}"
        )
        assert "t5_text_mask" not in data_batch, f"Expected t5_text_mask not in data_batch, got {data_batch.keys()}"
        assert len(data_batch["ai_caption"]) == 1, (
            f"Expected ai_caption to be a single string, got {data_batch['ai_caption']}"
        )
        log.info(f"data_batch ai_caption: {data_batch['ai_caption']}")
        captions = data_batch["ai_caption"][0].split(" -- ")
        if i % 2 == 0:
            assert len(captions) == 1, f"Expected 1 caption, got {len(captions)}"
        else:
            assert len(captions) == 7, f"Expected 7 captions, got {len(captions)}"
        prefix_tele = "The video is captured from a telephoto camera mounted on a car. The camera is facing"
        prefix_wide = "The video is captured from a camera mounted on a car. The camera is facing"
        for caption in captions:
            log.info(f"caption: {caption}")
            if i % 2 == 1:
                assert caption.startswith(prefix_tele) or caption.startswith(prefix_wide)
            else:
                assert not caption.startswith(prefix_tele) and not caption.startswith(prefix_wide)


@pytest.mark.skipif(SKIP_LOCAL_TESTS, reason="Local test")
@pytest.mark.L0
def test_alpamayo_dataset_joint_alpamayo1capnoviewprefix_allcapsviewprefix_480p_61frames_hybrid_captions_4views():
    os.environ["CAM_T5_EMBEDDINGS_CACHE_DIR"] = "s3://bucket/cosmos_predict2_multiview/cam_t5_embeddings_cache/"
    config = make_config()
    config = override(
        config,
        [
            "--",
            "experiment=buttercup_predict2p5_2b_mv_4views_res720p_fps30_t16_base2p5_alpamayo1capviewprefix_allcapsviewprefix_61frames_nofps_uniform",
        ],
    )
    config.dataloader_train.dataloaders.alpamayo_1cap.dataloader.num_workers = 0
    config.dataloader_train.dataloaders.alpamayo_allcaps.dataloader.num_workers = 0
    config.dataloader_train.dataloaders.alpamayo_1cap.dataloader.prefetch_factor = None
    config.dataloader_train.dataloaders.alpamayo_allcaps.dataloader.prefetch_factor = None
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_train_iter = iter(dataloader_train)
    num_views = 4
    num_frames = num_views * 61
    import matplotlib.pyplot as plt

    for i in range(4):
        data_batch = next(dataloader_train_iter)
        assert data_batch["video"].shape == (
            1,
            3,
            num_frames,
            480,
            832,
        ), f"Expected video shape (1, 3, {num_frames}, 480, 832), got {data_batch['video'].shape}"
        assert data_batch["view_indices"].shape == (
            1,
            num_frames,
        ), f"Expected view_indices shape (1, {num_frames}), got {data_batch['view_indices'].shape}"
        assert 1 in data_batch["view_indices"].squeeze().tolist(), (
            f"Expected view_indices to contain view_id 1, got {data_batch['view_indices'].squeeze().tolist()}"
        )
        assert data_batch["sample_n_views"].item() == num_views, (
            f"Expected sample_n_views {num_views}, got {data_batch['sample_n_views'].item()}"
        )
        assert data_batch["fps"].shape == (1,), f"Expected fps shape (1,), got {data_batch['fps'].shape}"
        assert data_batch["fps"].item() == 30, f"Expected fps 30, got {data_batch['fps'].item()}"
        assert "t5_text_embeddings" not in data_batch, (
            f"Expected t5_text_embeddings not in data_batch, got {data_batch.keys()}"
        )
        assert "t5_text_mask" not in data_batch, f"Expected t5_text_mask not in data_batch, got {data_batch.keys()}"
        assert len(data_batch["ai_caption"]) == 1, (
            f"Expected ai_caption to be a single string, got {data_batch['ai_caption']}"
        )
        log.info(f"data_batch ai_caption: {data_batch['ai_caption']}")
        captions = data_batch["ai_caption"][0].split(" -- ")

        assert len(captions) == 4, f"Expected 4 caption, got {len(captions)}"
        prefix_tele = "The video is captured from a telephoto camera mounted on a car. The camera is facing"
        prefix_wide = "The video is captured from a camera mounted on a car. The camera is facing"
        for caption in captions:
            log.info(f"caption: {caption}")
            assert caption.startswith(prefix_tele) or caption.startswith(prefix_wide)

        import matplotlib.pyplot as plt
        import numpy as np

        # (3, {num_frames}, 480, 832)
        video = data_batch["video"][0].permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
        for j in range(num_views):
            frame = video[j * 61]
            index = data_batch["view_indices"][0][j * 61]
            plt.imshow(frame)
            cap = ("" if index > 0 else captions[0]) if len(captions) == 1 else captions[j]
            plt.title(f"View {index}, Caption: {cap}", fontsize=10, wrap=True)
            plt.savefig(f"alpamayo_test_{i}_{index}.png")
            plt.close()
