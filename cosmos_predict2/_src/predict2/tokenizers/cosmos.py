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


from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from cosmos_predict2._src.predict2.tokenizers.wan2pt2 import Wan2pt2VAEInterface

Wan2pt1VAEConfig: LazyDict = L(Wan2pt1VAEInterface)(name="wan2pt1_tokenizer", compile_encode=False)
Wan2pt1VAEConfig_GCP: LazyDict = L(Wan2pt1VAEInterface)(
    name="wan2pt1_tokenizer_gcp",
    compile_encode=False,
    s3_credential_path="credentials/gcp_training.secret",
    vae_pth="s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/Wan2.1_VAE.pth",
)
Wan2pt2VAEConfig: LazyDict = L(Wan2pt2VAEInterface)(name="wan2pt2_tokenizer")
