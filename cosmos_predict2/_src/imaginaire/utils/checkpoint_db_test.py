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
    _CHECKPOINTS_BY_UUID,
    get_checkpoint_by_s3,
    get_checkpoint_by_uuid,
    get_checkpoint_path,
)


@pytest.mark.L0
def test_get_checkpoint_file():
    uuid = "685afcaa-4de2-42fe-b7b9-69f7a2dee4d8"
    s3_uri = "s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/Wan2.1_VAE.pth"
    config = get_checkpoint_by_uuid(uuid)
    assert config.s3 is not None
    assert config.hf is not None
    assert get_checkpoint_by_s3(s3_uri) is config
    assert get_checkpoint_path(s3_uri) == config.path
    assert get_checkpoint_path(config.path) == config.path


@pytest.mark.L1
def test_get_checkpoint_hf_file():
    uuid = "685afcaa-4de2-42fe-b7b9-69f7a2dee4d8"
    config = get_checkpoint_by_uuid(uuid)
    hf_path = Path(config.hf.path)
    assert hf_path.is_file()
    assert hf_path.suffix == ".pth"


@pytest.mark.L0
def test_get_checkpoint_dir():
    uuid = "7219c6c7-f878-4137-bbdb-76842ea85e70"
    s3_uri = "s3://bucket/cosmos_reasoning1/pretrained/Qwen_tokenizer/Qwen/Qwen2.5-VL-7B-Instruct"
    config = get_checkpoint_by_uuid(uuid)
    assert config.s3 is not None
    assert config.hf is not None
    assert get_checkpoint_by_s3(s3_uri) is config
    assert get_checkpoint_path(s3_uri) == config.path
    assert get_checkpoint_path(config.path) == config.path


@pytest.mark.L1
def test_get_checkpoint_hf_dir():
    uuid = "7219c6c7-f878-4137-bbdb-76842ea85e70"
    config = get_checkpoint_by_uuid(uuid)
    hf_path = Path(config.hf.path)
    assert hf_path.is_dir()
    assert hf_path.joinpath("tokenizer.json").is_file()


@pytest.mark.L1
def test_all_checkpoints():
    for config in _CHECKPOINTS_BY_UUID.values():
        # Check Hugging Face checkpoint
        if config.hf is not None:
            hf_path = Path(config.hf.path)
            assert hf_path.exists()
