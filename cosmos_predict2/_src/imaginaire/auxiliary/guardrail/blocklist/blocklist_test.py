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

from cosmos_predict2._src.imaginaire.auxiliary.guardrail.blocklist.blocklist import Blocklist


@pytest.mark.L1
def test_exact_match():
    """Test exact word matching."""
    prompt = "this contains badword in the middle"
    word = "badword"

    match, message = Blocklist.check_partial_match(prompt, word, 0.4)

    assert match is True
    assert "badword" in message


@pytest.mark.L1
def test_no_match():
    """Test when there's no match."""
    prompt = "this is a clean prompt"
    word = "badword"

    match, message = Blocklist.check_partial_match(prompt, word, 0.4)

    assert match is False
    assert message == ""


@pytest.mark.L1
def test_partial_match_with_threshold():
    """Test partial matching with character difference threshold."""
    prompt = "this contains badw0rd here"  # One character different
    word = "badword"

    # With threshold of 1.0 character difference, should match
    match, _ = Blocklist.check_partial_match(prompt, word, 1.0)
    assert match is True

    # With threshold of 0.5 character difference, should not match
    match, _ = Blocklist.check_partial_match(prompt, word, 0.5)
    assert match is False
