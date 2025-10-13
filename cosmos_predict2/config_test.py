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

from cosmos_predict2.config import (
    DEFAULT_NEGATIVE_PROMPT,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ModelKey,
    ModelSize,
    ModelVariant,
)


@pytest.mark.L0
def test_constants():
    import cosmos_predict2._src.predict2.inference.video2world as src

    assert IMAGE_EXTENSIONS == src._IMAGE_EXTENSIONS
    assert VIDEO_EXTENSIONS == src._VIDEO_EXTENSIONS
    assert DEFAULT_NEGATIVE_PROMPT == src._DEFAULT_NEGATIVE_PROMPT


@pytest.mark.L0
def test_model_key():
    assert ModelKey().name == "2B/post-trained"
    assert ModelKey(size=ModelSize._14B).name == "14B/post-trained"
    assert ModelKey(variant=ModelVariant.AUTO_MULTIVIEW).name == "2B/auto/multiview"
    assert ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW).name == "2B/robot/multiview"
    assert ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_AGIBOT).name == "2B/robot/multiview-agibot"
