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

from cosmos_predict2.config import CommonInferenceArguments, InferenceArguments
from cosmos_predict2.multiview_config import MultiviewInferenceArgumentsWithInputPaths

_INFERENCE_ASSETS: list[tuple[str, type[CommonInferenceArguments]]] = [
    ("base", InferenceArguments),
    ("multiview", MultiviewInferenceArgumentsWithInputPaths),
    ("video2world_cosmos_nemo_assets", InferenceArguments),
    ("sample_gr00t_dreams_gr1", InferenceArguments),
]


@pytest.mark.L0
@pytest.mark.parametrize("name,args_cls", _INFERENCE_ASSETS, ids=[asset[0] for asset in _INFERENCE_ASSETS])
def test_inference_assets(name: str, args_cls: type[CommonInferenceArguments]):
    input_dir = Path("assets") / name
    # Sample names should be unique accross json files
    assert args_cls.from_files(list(input_dir.glob("**/*.json")))
    # Sample names are not unique accross jsonl files
    for input_file in list(input_dir.glob("**/*.json*")):
        assert args_cls.from_files([input_file])
