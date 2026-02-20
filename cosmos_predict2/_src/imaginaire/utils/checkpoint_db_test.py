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

from pathlib import Path
from uuid import uuid4

import pytest

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import (
    CheckpointConfig,
    CheckpointDirHf,
    CheckpointDirS3,
    CheckpointFileHf,
    CheckpointFileS3,
    download_checkpoint,
    get_checkpoint_uri,
)

EXPERIMENT = "experiment"
CHECKPOINT_ITER = "000023000"

CHECKPOINT_HF_REPOSITORY = "nvidia/Cosmos-Predict2.5-2B"
CHECKPOINT_HF_REVISION = "e26f8a125a2235c5a00245a65207402dd0cdcb89"
CHECKPOINT_HF_SUBDIRECTORY = "base/post-trained"
CHECKPOINT_HF_FILENAME = "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"

CHECKPOINT_DIR_UUID = str(uuid4())
CHECKPOINT_DIR_S3_URI = f"s3://{__name__}/{EXPERIMENT}/checkpoints/iter_{CHECKPOINT_ITER}/model"

CHECKPOINT_FILE_UUID = str(uuid4())
CHECKPOINT_FILE_S3_URI = f"s3://{__name__}/model.pt"
CHECKPOINT_FILE_HF_URI = f"hf://{CHECKPOINT_HF_REPOSITORY}/{CHECKPOINT_HF_SUBDIRECTORY}/{CHECKPOINT_HF_FILENAME}"


@pytest.fixture(scope="session", autouse=True)
def register_checkpoints():
    CheckpointConfig(
        uuid=CHECKPOINT_DIR_UUID,
        name=f"{__name__}/dir",
        s3=CheckpointDirS3(
            uri=CHECKPOINT_DIR_S3_URI,
        ),
        hf=CheckpointDirHf(
            repository=CHECKPOINT_HF_REPOSITORY,
            revision=CHECKPOINT_HF_REVISION,
            subdirectory=CHECKPOINT_HF_SUBDIRECTORY,
        ),
    ).register()

    CheckpointConfig(
        uuid=CHECKPOINT_FILE_UUID,
        name=f"{__name__}/file",
        s3=CheckpointFileS3(
            uri=CHECKPOINT_FILE_S3_URI,
        ),
        hf=CheckpointFileHf(
            repository=CHECKPOINT_HF_REPOSITORY,
            revision=CHECKPOINT_HF_REVISION,
            filename=f"{CHECKPOINT_HF_SUBDIRECTORY}/{CHECKPOINT_HF_FILENAME}",
        ),
    ).register()


@pytest.mark.L0
def test_get_checkpoint_uri():
    assert get_checkpoint_uri("/path/to/checkpoint") == "/path/to/checkpoint"
    assert get_checkpoint_uri(CHECKPOINT_FILE_UUID) == CheckpointConfig.from_uri(CHECKPOINT_FILE_UUID).s3.uri
    assert get_checkpoint_uri(CHECKPOINT_FILE_HF_URI) == CHECKPOINT_FILE_HF_URI
    with pytest.raises(ValueError):
        get_checkpoint_uri("/invalid/path", check_exists=True)
    if INTERNAL:
        assert get_checkpoint_uri("s3://invalid/uri") == "s3://invalid/uri"
    else:
        with pytest.raises(ValueError):
            get_checkpoint_uri("s3://invalid/uri")


@pytest.mark.L0
def test_download_checkpoint():
    assert download_checkpoint("/path/to/checkpoint", check_exists=False) == "/path/to/checkpoint"
    if INTERNAL:
        assert download_checkpoint(CHECKPOINT_FILE_S3_URI) == CHECKPOINT_FILE_S3_URI
        assert get_checkpoint_uri("s3://invalid/uri") == "s3://invalid/uri"
        assert download_checkpoint(CHECKPOINT_FILE_HF_URI) == CHECKPOINT_FILE_HF_URI
    else:
        with pytest.raises(ValueError):
            download_checkpoint("/invalid/path", check_exists=True)
        with pytest.raises(ValueError):
            download_checkpoint("s3://invalid/uri")

        hf_path = Path(download_checkpoint(CHECKPOINT_FILE_HF_URI))
        assert hf_path.is_file()
        assert hf_path.name == CHECKPOINT_HF_FILENAME


@pytest.mark.L0
def test_get_checkpoint_file():
    config = CheckpointConfig.from_uri(CHECKPOINT_FILE_UUID)
    assert CheckpointConfig.from_uri(CHECKPOINT_FILE_S3_URI) is config

    if not INTERNAL:
        path = config.download()
        assert download_checkpoint(CHECKPOINT_FILE_S3_URI) == path
        assert download_checkpoint(config.s3.uri) == path
        assert download_checkpoint(path) == path

        path = Path(path)
        assert path.is_file()
        assert path.name == CHECKPOINT_HF_FILENAME


@pytest.mark.L0
def test_get_checkpoint_dir():
    config = CheckpointConfig.from_uri(CHECKPOINT_DIR_UUID)
    assert CheckpointConfig.from_uri(CHECKPOINT_DIR_S3_URI) is config

    if not INTERNAL:
        path = config.download()
        assert download_checkpoint(CHECKPOINT_DIR_S3_URI) == path
        assert download_checkpoint(config.s3.uri) == path
        assert download_checkpoint(path) == path

        path = Path(path)
        assert path.is_dir()
        assert path.joinpath(CHECKPOINT_HF_FILENAME).is_file()
