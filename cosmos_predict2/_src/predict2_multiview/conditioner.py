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

from __future__ import annotations

from typing import List, Optional

import torch

from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.conditioner import AbstractEmbModel


class MVTextAttr(AbstractEmbModel):
    def __init__(
        self,
        input_key: List[str],
        dropout_rate: Optional[float] = 0.0,
        use_empty_string: bool = False,
        empty_string_embeddings_path: str = "s3://bucket/predict2_assets/reason1_empty_string_embeddings.pt",
        credential_path: str = "credentials/s3_training.secret",
        single_caption_length: int = 512,
    ):
        super().__init__()
        self._input_key = input_key
        self._dropout_rate = dropout_rate
        # if True, will use empty string embeddings
        # otherwise use zero tensor embeddings
        self.use_empty_string = use_empty_string
        self._empty_string_embeddings_cache = None
        self.empty_string_embeddings_path = empty_string_embeddings_path
        self.credential_path = credential_path
        self.single_caption_length = single_caption_length

    def forward(self, token: torch.Tensor):
        return {"crossattn_emb": token}

    def _get_empty_string_embeddings(self) -> torch.Tensor:
        """Lazy load and cache empty string embeddings."""
        if self._empty_string_embeddings_cache is None:
            self._empty_string_embeddings_cache = easy_io.load(
                self.empty_string_embeddings_path,
                backend_args={"backend": "s3", "s3_credential_path": self.credential_path},
            )
        return self._empty_string_embeddings_cache

    def random_dropout_input(
        self, in_tensor: torch.Tensor, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        if key is not None and "mask" in key:
            return in_tensor

        dropout_rate = dropout_rate if dropout_rate is not None else self._dropout_rate

        B = in_tensor.shape[0]  # batch size
        S_per_view = self.single_caption_length  # sequence length per view
        if in_tensor.shape[1] % S_per_view != 0:
            raise ValueError(
                f"in_tensor sequence length {in_tensor.shape[1]} is not divisible by single_caption_length {S_per_view}"
            )
        V = in_tensor.shape[1] // S_per_view  # number of views

        # reshape input tensor to [B, V, S, C]
        in_tensor_reshaped = in_tensor.view(B, V, S_per_view, -1)

        # create independent dropout mask for each view: [B, V]
        dropout_rates = torch.ones(B, V, device=in_tensor.device) * dropout_rate
        keep_mask = torch.bernoulli(1.0 - dropout_rates).type_as(in_tensor)
        keep_mask = keep_mask.view(B, V, 1, 1)  # broadcastable shape

        # prepare empty prompt data
        if not self.use_empty_string:
            empty_string_embeddings = torch.zeros(
                1, 1, S_per_view, in_tensor_reshaped.shape[-1], dtype=in_tensor.dtype, device=in_tensor.device
            )
        else:
            empty_string_embeddings = (
                self._get_empty_string_embeddings().to(dtype=in_tensor.dtype, device=in_tensor.device).unsqueeze(1)
            )  # [1, 1, 512, C]

        # expand empty prompt data to all views: [B, V, S, C]
        empty_string_embeddings = empty_string_embeddings.expand(B, V, S_per_view, -1)

        # apply dropout independently for each view
        output_reshaped = keep_mask * in_tensor_reshaped + (1.0 - keep_mask) * empty_string_embeddings

        # reshape back to original shape: [B, V * S, C]
        return output_reshaped.view(B, V * S_per_view, -1)

    def details(self) -> str:
        return "Output key: [crossattn_emb]"
