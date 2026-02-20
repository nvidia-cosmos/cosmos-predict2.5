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

from cosmos_predict2._src.imaginaire.lazy_config.registry import convert_target_to_string


def function(): ...


class Base:
    @classmethod
    def class_method(cls): ...


class Derived(Base):
    @staticmethod
    def static_method(): ...

    def instance_method(self): ...


@pytest.mark.L0
def test_convert_target_to_string():
    assert convert_target_to_string(int) == "builtins.int"
    assert convert_target_to_string(print) == "builtins.print"
    assert convert_target_to_string(Derived().instance_method) == f"{__name__}.{Derived.__qualname__}.instance_method"
    assert convert_target_to_string(Derived.static_method) == f"{__name__}.{Derived.__qualname__}.static_method"
    assert convert_target_to_string(Derived.class_method) == f"{__name__}.{Derived.__qualname__}.class_method"
    assert convert_target_to_string(function) == f"{__name__}.{function.__qualname__}"
