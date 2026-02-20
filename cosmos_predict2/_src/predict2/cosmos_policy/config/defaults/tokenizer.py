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

"""
Cosmos Policy tokenizer registration with deterministic seeding support.
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.cosmos_policy.tokenizers.wan2pt1 import Wan2pt1VAEInterface

# Policy-specific wan2pt1 tokenizer with deterministic seeding
PolicyWan2pt1VAEConfig = L(Wan2pt1VAEInterface)(
    vae_pth="hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
    s3_credential_path="credentials/s3_training.secret",
    load_mean_std=False,
    temporal_window=4,
    is_parallel=False,
    cp_grid_shape=None,
)


def register_policy_tokenizer():
    """
    Register Cosmos Policy tokenizer configurations.

    This registers the wan2pt1 tokenizer with deterministic seeding support.
    To enable deterministic encoding, set: DETERMINISTIC=true
    """
    cs = ConfigStore.instance()
    # Also register with explicit policy prefix
    cs.store(
        group="tokenizer",
        package="model.config.tokenizer",
        name="policy_wan2pt1_tokenizer",
        node=PolicyWan2pt1VAEConfig,
    )
