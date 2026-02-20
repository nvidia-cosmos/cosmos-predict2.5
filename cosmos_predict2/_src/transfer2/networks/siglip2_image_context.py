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

from typing import List, Optional

import torch

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import AbstractEmbModel
from cosmos_predict2._src.transfer2.networks.siglip2 import get_siglip2_latents, get_siglip2_model_processor


class SigLip2EmbImgContext(AbstractEmbModel):
    def __init__(
        self,
        input_key: List[str],
        output_key: Optional[str] = None,
        dropout_rate: Optional[float] = 0.0,
        num_token: int = 256,
    ):
        super().__init__()
        self.num_token = num_token
        self.model_dim = 1152
        self.model, self.processor = get_siglip2_model_processor("google/siglip2-so400m-patch16-naflex")

        self._input_key = input_key
        self._output_key = output_key
        self._dropout_rate = dropout_rate

    def random_dropout_input(
        self, in_tensor: Optional[torch.Tensor] = None, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        if in_tensor is None:
            return None

        # Handle dropout differently based on the input key
        if key is not None:
            # Don't apply dropout to video tensor itself, only to image_context
            if key == "video":
                return in_tensor
            # Don't apply dropout to plain images
            elif key == "images":
                return in_tensor
            # Apply dropout to image_context only
            elif key == "image_context":
                return super().random_dropout_input(in_tensor, dropout_rate, key)

        # Fallback for when key is not specified
        return super().random_dropout_input(in_tensor, dropout_rate, key)

    def forward(
        self,
        input_tensor: Optional[torch.Tensor] = None,
        image_context: Optional[torch.Tensor] = None,
    ) -> dict:
        if image_context is not None:
            assert image_context.shape[0] == input_tensor.shape[0], (
                "image_context and input_tensor must have the same batch size"
            )
            batch_size = image_context.shape[0]
            # Handle image_context with shape [B, C, T, H, W] -> [B, C, H, W]
            if image_context.ndim == 5:  # [B, C, T, H, W]
                # Take the first (and typically only) frame
                image_context_B_C_H_W = image_context[:, :, 0, :, :]
            else:  # [B, C, H, W]
                image_context_B_C_H_W = image_context

            latents = get_siglip2_latents(self.model, self.processor, image_context_B_C_H_W)
        else:
            # If no image_context provided, return zero latents
            log.warning("No image_context provided, using zero latents")
            batch_size = input_tensor.shape[0]
            latents = torch.zeros(
                batch_size, self.num_token, self.model_dim, device=input_tensor.device, dtype=torch.bfloat16
            )

        return_dict = {
            "img_context_emb": latents,
        }
        return return_dict

    def details(self) -> str:
        output_key = ["img_context_emb"]
        return f"Input key: {self.input_key} \n\tOutput key: {output_key}"
