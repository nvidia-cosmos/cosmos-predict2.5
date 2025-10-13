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

import time

import torch

from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.text2world.defaults.net import (
    COSMOS_V1_2B_NET_MININET,
    COSMOS_V1_14B_NET_MININET,
)
from cosmos_predict2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT, replace_selfattn_op_with_sparse_attn_op

natten_parameters_90pct = {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)}

natten_parameters_2b_comb01 = [
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 0, 90%
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 1, 50%
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 2, 90%
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 3, 90%
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 4, 90%
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 5, 90%
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 6, 90%
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 7, 90%
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 8, 90%
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 9, 50%
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 10, 90%
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 11, 50%
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 12, 50%
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 13, 50%
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 14, 50%
    None,  # blk 15, SA
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 16, 50%
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 17, 50%
    None,  # blk 18, SA
    None,  # blk 19, SA
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 20, 50%
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 21, 50%
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 22, 50%
    None,  # blk 23, SA
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 24, 50%
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 25, 50%
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 26, 50%
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 27, 50%
]

natten_parameters_2b_comb02 = [
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 0
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 1
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # blk 2
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 3
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 4
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # blk 5
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 6
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 7
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 8
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 9
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # blk 10
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 11
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 12
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 13
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 14
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 15
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 16
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 17
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 18
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 19
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 20
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 21
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 22
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 23
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 24
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 25
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 26
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 27
]

natten_parameters_2b_comb03 = [
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 0
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 1
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # blk 2
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 3
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 4
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # blk 5
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 6
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 7
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 8
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 9
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 10
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 11
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 12
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 13
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 14
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 15
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 16
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 17
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 18
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 19
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 20
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 21
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 22
    None,  # blk 23
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 24
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 25
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 26
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 27
]

natten_parameters_2b_comb04 = [
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 0
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 1
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 2
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 3
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 4
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 5
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 6
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 7
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 8
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 9
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 10
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 11
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 12
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 13
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 14
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 15
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 16
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 17
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 18
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 19
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 20
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 21
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 22
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 23
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 24
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 25
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 26
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 27
]

natten_parameters_2b_comb05 = [
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 0
    {"window_size": (-1, 4, 24), "stride": (1, 1, 8), "dilation": (1, 11, 1), "base_size": (-1, 44, 80)},  # blk 1
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # blk 2
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 3
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 4
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # blk 5
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 6
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 7
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 8
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # blk 9
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # blk 10
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 11
    None,  # blk 12
    None,  # blk 13
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 14
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 15
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 16
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 17
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 18
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 19
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 20
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 21
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 22
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 23
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 24
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 25
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 26
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 27
]

natten_parameters_14b_comb01 = [
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 0
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 1
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 2
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 3
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 4
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 5
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 6
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 7
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 8
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 9
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 10
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # blk 11
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # blk 12
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 13
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 14
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 15
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 16
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 17
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 18
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 19
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 20
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 21
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 22
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 23
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 24
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 25
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 26
    None,  # blk 27
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 28
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 29
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 30
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 31
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 32
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 33
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 34
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 35
]

natten_parameters_14b_comb02 = [
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 0
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 1
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 2
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 3
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 4
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 5
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 6
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 7
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 8
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 9
    {"window_size": (-1, 4, 16), "stride": (1, 1, 1), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # blk 10
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # blk 11
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # blk 12
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 13
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 14
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 15
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 16
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 17
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 18
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 19
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 20
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 21
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 22
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 23
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 24
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 25
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 26
    None,  # blk 27
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 28
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 29
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 30
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 31
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 32
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 33
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 34
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # blk 35
]

"""
Forward pass only
"""


def measure_e2e_perf(model, inputs, warmup_iters, iters):
    torch.cuda.synchronize()
    for _ in range(warmup_iters):
        output = model(**inputs)
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    for _ in range(iters):
        output = model(**inputs)
    torch.cuda.synchronize()

    end_time = time.perf_counter()

    e2e_time_total = end_time - start_time

    avg_e2e_time = e2e_time_total / iters

    print(f"E2E time avg: {avg_e2e_time:.2f} s")


@torch.no_grad()
def test_inference_e2e():
    try:
        import natten  # noqa: F401
    except ImportError:
        print("NATTEN is not installed, skipping...")
        return

    batch_size = 1
    warmup_iters = 5
    iters = 5

    res = "720"
    video_size_options = VIDEO_RES_SIZE_INFO[res]
    H, W = 704, 1280
    T_list = [
        24,
    ]
    dH, dW = 8, 8
    pH, pW = 2, 2

    COSMOS_V1_2B_NET_MININET.atten_backend = "minimal_a2a"
    COSMOS_V1_14B_NET_MININET.atten_backend = "minimal_a2a"

    for net_str, net_cfg, n_dense_blocks, natten_parameters in [
        # ("2B", COSMOS_V1_2B_NET_MININET, -1, {}),
        # ("2B", COSMOS_V1_2B_NET_MININET, 4, natten_parameters_90pct),
        # ("2B", COSMOS_V1_2B_NET_MININET, 7, natten_parameters_90pct),
        # ("2B", COSMOS_V1_2B_NET_MININET, 9, natten_parameters_90pct),
        # ("2B", COSMOS_V1_2B_NET_MININET, 0, natten_parameters_2b_comb01),
        # ("2B", COSMOS_V1_2B_NET_MININET, 0, natten_parameters_2b_comb02),
        # ("2B", COSMOS_V1_2B_NET_MININET, 0, natten_parameters_2b_comb03),
        # ("2B", COSMOS_V1_2B_NET_MININET, 0, natten_parameters_2b_comb04),
        # ("2B", COSMOS_V1_2B_NET_MININET, 0, natten_parameters_2b_comb05),
        ("14B", COSMOS_V1_14B_NET_MININET, -1, {}),
        # ("14B", COSMOS_V1_14B_NET_MININET, 5, natten_parameters_90pct),
        # ("14B", COSMOS_V1_14B_NET_MININET, 7, natten_parameters_90pct),
        # ("14B", COSMOS_V1_14B_NET_MININET, 9, natten_parameters_90pct),
        # ("14B", COSMOS_V1_14B_NET_MININET, 12, natten_parameters_90pct),
        # ("14B", COSMOS_V1_14B_NET_MININET, 0, natten_parameters_14b_comb01),
        ("14B", COSMOS_V1_14B_NET_MININET, 0, natten_parameters_14b_comb02),
    ]:
        torch.cuda.empty_cache()
        print()
        print()
        print(f"Model: Minimal v4 DiT - {net_str}")

        replace_self_attn = n_dense_blocks >= 0
        model: MiniTrainDIT = instantiate(net_cfg).cuda().bfloat16()
        model.eval()
        if replace_self_attn:
            model = replace_selfattn_op_with_sparse_attn_op(
                model, n_dense_blocks=n_dense_blocks, natten_parameters=natten_parameters
            )
            print()
            print(f"Replaced Self Attention with NATTEN: {n_dense_blocks=}, {natten_parameters=}")
        else:
            print()
            print("Self Attention case (unmodified)")

        for T in T_list:
            T_, H_, W_ = T, H // dH, W // dW
            print(
                f"Res: {T=}, {H=}, {W=}; DiT input: ({T_}, {H_}, {W_}); feature map size: ({T}, {H_ // pH}, {W_ // pW})"
            )
            video_example_input = {
                "x_B_C_T_H_W": torch.randn(batch_size, 16, T_, H_, W_, dtype=torch.bfloat16, device="cuda"),
                "timesteps_B_T": torch.randn(batch_size, dtype=torch.bfloat16, device="cuda"),
                "crossattn_emb": torch.randn(batch_size, 512, 1024, dtype=torch.bfloat16, device="cuda"),
                "fps": torch.randint(size=(batch_size,), low=2, high=30, device="cuda"),
                "padding_mask": torch.randn(batch_size, 1, H_, W_, dtype=torch.bfloat16, device="cuda"),
                "data_type": DataType.VIDEO,
            }

            output = model(**video_example_input)
            assert output.shape == (batch_size, 16, T_, H_, W_)

            print("Video model")
            measure_e2e_perf(model, video_example_input, warmup_iters=warmup_iters, iters=iters)
            print()

        del model


if __name__ == "__main__":
    test_inference_e2e()
