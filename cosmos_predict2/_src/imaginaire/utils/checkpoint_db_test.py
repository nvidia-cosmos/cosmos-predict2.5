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

import pytest

from cosmos_predict2._src.imaginaire.utils.checkpoint_db import (
    CheckpointConfig,
    CheckpointDirHf,
    CheckpointDirS3,
    CheckpointFileHf,
    CheckpointFileS3,
    get_checkpoint_by_s3,
    get_checkpoint_by_uuid,
    get_checkpoint_path,
    register_checkpoint,
)

CHECKPOINT_DIR_UUID = "19bb41fd-298d-42d8-8ec1-2ac1cb3ff204"
CHECKPOINT_DIR_S3_URI = "s3://test/model"
CHECKPOINT_FILE_UUID = "9854561e-7f45-4200-a1c1-6f99baf79ecb"
CHECKPOINT_FILE_S3_URI = "s3://test/model.pth"


@pytest.fixture(scope="session", autouse=True)
def register_checkpoints():
    register_checkpoint(
        CheckpointConfig(
            uuid=CHECKPOINT_DIR_UUID,
            name="test/dir",
            s3=CheckpointDirS3(
                uri=CHECKPOINT_DIR_S3_URI,
            ),
            hf=CheckpointDirHf(
                repository="nvidia/Cosmos-Reason1-7B",
                revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid=CHECKPOINT_FILE_UUID,
            name="test/file",
            s3=CheckpointFileS3(
                uri=CHECKPOINT_FILE_S3_URI,
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Predict2.5-2B",
                revision="6787e176dce74a101d922174a95dba29fa5f0c55",
                filename="tokenizer.pth",
            ),
        ),
    )


@pytest.mark.L0
def test_get_checkpoint_file():
    uuid = CHECKPOINT_FILE_UUID
    s3_uri = CHECKPOINT_FILE_S3_URI
    config = get_checkpoint_by_uuid(uuid)
    assert config.s3 is not None
    assert config.hf is not None
    assert get_checkpoint_by_s3(s3_uri) is config
    assert get_checkpoint_path(s3_uri) == config.path
    assert get_checkpoint_path(config.path) == config.path


@pytest.mark.L1
def test_get_checkpoint_hf_file():
    uuid = CHECKPOINT_FILE_UUID
    config = get_checkpoint_by_uuid(uuid)
    hf_path = Path(config.hf.path)
    assert hf_path.is_file()
    assert hf_path.suffix == ".pth"


@pytest.mark.L0
def test_get_checkpoint_dir():
    uuid = CHECKPOINT_DIR_UUID
    s3_uri = CHECKPOINT_DIR_S3_URI
    config = get_checkpoint_by_uuid(uuid)
    assert config.s3 is not None
    assert config.hf is not None
    assert get_checkpoint_by_s3(s3_uri) is config
    assert get_checkpoint_path(s3_uri) == config.path
    assert get_checkpoint_path(config.path) == config.path


@pytest.mark.L1
def test_get_checkpoint_hf_dir():
    uuid = CHECKPOINT_DIR_UUID
    config = get_checkpoint_by_uuid(uuid)
    hf_path = Path(config.hf.path)
    assert hf_path.is_dir()
    assert hf_path.joinpath("tokenizer.json").is_file()
