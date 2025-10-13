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

from cosmos_predict2._src.imaginaire.utils.validator import Bool, Dict, Float, Int, OneOf, String
from cosmos_predict2._src.imaginaire.utils.validator_params import ValidatorParams

param_dict = {
    "prompt": "a cat",
    "num_samples": 2,
    "guidance": 6.5,
    "media_type": "illustration",
}


class SampleParams(ValidatorParams):
    """All the required values to generate image from text at a given resolution."""

    # no default, so it is mandatory
    prompt = String()

    negative_prompt = String(
        "ugly, blurry, childish, flat, malformed, poorly drawn, old, dated, 80s, 90s, photoshop, post process, "
        "collage, fake"
    )
    num_samples = Int(4, min=1, max=4)
    media_type = OneOf("photography", ["photography", "illustration", "film"])
    guidance = Float(7.5, min=5, max=10)


class ChildParams(SampleParams):
    """Child class might want to remove some parameters if certain model paramters shouldn't be exposed to the user."""

    num_samples = Int(1, hidden=True)
    nsfw_flag = Bool(False)


nested_dict_param = {
    "dict_param": {
        "path": "some/file/path",
    },
}


class ParamsWithNesting(ValidatorParams):
    """Test dict parameter."""

    seed = Int(0)
    dict_param = Dict(default={})


@pytest.mark.L0
def test_from_kwargs():
    params = SampleParams.create(param_dict)
    assert params.prompt == "a cat"
    assert params.num_samples == 2
    assert params.guidance == 6.5
    assert params.media_type == "illustration"
    print("✅ test_from_kwargs: PASSED")


@pytest.mark.L0
def test_from_kwargs_dict():
    params = ParamsWithNesting.create(nested_dict_param)
    assert params.dict_param == nested_dict_param["dict_param"]
    assert params.dict_param["path"] == nested_dict_param["dict_param"]["path"]
    params.debug_print()
    print(params)
    print("✅ test_from_kwargs_dict: PASSED")


@pytest.mark.L0
def test_from_cmd_line():
    # input is the legacy command line format
    cmd = "--prompt='cat' --num_samples=1 --guidance=5.5 --media_type='illustration'"
    params = SampleParams.createFromCmd(cmd)
    params.debug_print()
    # access descriptors same as regular variables
    assert params.num_samples == 1
    assert params.guidance == 5.5
    print("✅ test_from_cmd_line: PASSED")


@pytest.mark.L0
def test_unknown_parameter():
    cmd = "--prompt='cat' --unkown_param=1"
    with pytest.raises(ValueError):
        params_config = ValidatorParams.createFromCmd(cmd)
    print("✅ test_unknown_parameter: PASSED")


@pytest.mark.L0
def test_no_default():
    with pytest.raises(ValueError):
        params = SampleParams.create({})
    print("✅ test_no_default: PASSED")


@pytest.mark.L0
def test_freeze():
    test_config = SampleParams()
    # todo the class dict isn't frozen
    # so unfortunately following code is still allowed
    test_config.nsamplesss = "some_param"
    print("⚠️ test_freeze: PASSED (freezing not fully implemented yet)")


@pytest.mark.L0
def test_out_of_range():
    cmd = "--prompt='cat' --human_attributes='out of range param'"
    with pytest.raises(ValueError):
        params_config = ValidatorParams.createFromCmd(cmd)
    print("✅ test_out_of_range: PASSED")


@pytest.mark.L0
def test_range_iter():
    range_config = SampleParams()
    range_config.prompt = "a fat cat"
    val_dict = range_config.get_val_dict()
    descriptors = list(val_dict.values())

    # simple test over parameter ranges (not permutation of all parameters)
    # we interate over the value range of each parameter while keeping the rest at default
    iteration_count = 0
    for desc in descriptors:
        param_iterations = 0
        for i in desc.get_range_iterator():
            setattr(range_config, desc.private_name, i)
            param_iterations += 1
            iteration_count += 1
            # Limit iterations to prevent excessive output
            if param_iterations > 5:
                break
            # todo run test with upated values
            # t2i(range_config)
        setattr(range_config, desc.private_name, desc.default)

    print(f"✅ test_range_iter: PASSED ({iteration_count} iterations)")


if __name__ == "__main__":
    print("🚀 Running validator tests...\n")
    test_no_default()
    test_from_kwargs_dict()

    # test_from_kwargs()
    # test_from_cmd_line()
    # test_probe()
    # test_unknown_parameter()
    # test_freeze()
    # test_out_of_range()
    # test_range_iter()
