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

import pytest

from cosmos_predict2._src.imaginaire.utils import misc
from cosmos_predict2._src.predict2.inference.get_umt5_emb import UMT5EncoderModel


@pytest.mark.L2
def test_encoder():
    with misc.timer("load model"):
        model = UMT5EncoderModel(
            checkpoint_path="s3://bucket/cosmos_diffusion_v2/pretrain_weights/models_t5_umt5-xxl-enc-bf16.pth"
        )
    emb = model(texts=["hello world", "hello", "world"])
    assert len(emb) == 3
    assert emb[0].shape == (512, 4096)
    assert emb[1].shape == (512, 4096)
    assert emb[2].shape == (512, 4096)
