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

from cosmos_predict2.config import (
    ModelKey,
    ModelSize,
    ModelVariant,
)


def test_model_key():
    assert ModelKey().name == "2B/post-trained"
    assert ModelKey(size=ModelSize._14B).name == "14B/post-trained"
    assert ModelKey(variant=ModelVariant.AUTO_MULTIVIEW).name == "2B/auto/multiview"
