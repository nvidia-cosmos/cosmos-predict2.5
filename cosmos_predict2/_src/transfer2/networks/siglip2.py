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
from transformers import AutoConfig, AutoImageProcessor

try:
    from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel
except ImportError:
    Siglip2VisionModel = None

from cosmos_predict2._src.imaginaire.models.abstract_emb_model import AbstractEmbModel
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io

S3_KEY = "_s3_predict2_siglip"
S3_PATH_FORMAT = "s3://bucket/cosmos_diffusion_v2/pretrain_weights/siglip2/{model_name}.pth"


def upload_siglip2_weights(model_name: str):
    easy_io.set_s3_backend(
        key=S3_KEY, backend_args={"backend": "s3", "s3_credential_path": "credentials/s3_training.secret"}
    )
    config = AutoConfig.from_pretrained(model_name)
    config.vision_config.vision_use_head = False
    # model = Siglip2VisionModel(config.vision_config)
    # model.to("cuda", dtype=torch.bfloat16)
    model = Siglip2VisionModel.from_pretrained(
        model_name, config=config.vision_config, device_map="cuda", torch_dtype=torch.bfloat16
    ).eval()

    easy_io.dump(
        model.state_dict(),
        S3_PATH_FORMAT.format(model_name=model_name),
        backend_key=S3_KEY,
    )


def get_siglip2_model_processor(model_name: str):
    config = AutoConfig.from_pretrained(model_name)
    config.vision_config.vision_use_head = False
    model = Siglip2VisionModel(config.vision_config)
    model.eval().to("cuda", dtype=torch.bfloat16)
    s3_path = S3_PATH_FORMAT.format(model_name=model_name)

    if distributed.is_rank0():
        model_hf = Siglip2VisionModel.from_pretrained(
            model_name, config=config.vision_config, torch_dtype=torch.bfloat16
        )
        state_dict = model_hf.state_dict()
        log.info(model.load_state_dict(state_dict))

    distributed.sync_model_states(model)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor


def get_siglip2_latents(
    model: Siglip2VisionModel, processor: AutoImageProcessor, image_tensor_B_C_H_W_in_n1_to_p1: torch.Tensor
):
    in_dtype = image_tensor_B_C_H_W_in_n1_to_p1.dtype
    inputs = processor(images=(1.0 + image_tensor_B_C_H_W_in_n1_to_p1) / 2.0 * 255.0, return_tensors="pt").to(
        "cuda", dtype=torch.bfloat16
    )
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
    return last_hidden_state.to(in_dtype)


class SigLip2Emb(AbstractEmbModel):
    def __init__(
        self,
        input_key: List[str],
        output_key: Optional[str] = None,
        dropout_rate: Optional[float] = 0.0,
        num_token: int = 256,
        add_use_video_condition: bool = False,
    ):
        super().__init__()
        self.num_token = num_token
        self.model_dim = 1152
        self.add_use_video_condition = add_use_video_condition
        self.model, self.processor = get_siglip2_model_processor("google/siglip2-so400m-patch16-naflex")

        self._input_key = input_key
        self._output_key = output_key
        self._dropout_rate = dropout_rate

    def random_dropout_input(
        self, in_tensor: Optional[torch.Tensor] = None, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        if in_tensor is None:
            return None
        return super().random_dropout_input(in_tensor[:, :, :1], dropout_rate, key)

    def forward(
        self, image_tensor: Optional[torch.Tensor] = None, video_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if image_tensor is not None:
            batch_size = image_tensor.shape[0]
            latents = torch.zeros(
                batch_size, self.num_token, self.model_dim, device=image_tensor.device, dtype=image_tensor.dtype
            )
            use_video_condition = torch.zeros(batch_size, device=image_tensor.device, dtype=torch.bool)
        else:
            first_frame_B_C_H_W = video_tensor[:, :, 0, :, :]
            batch_size = first_frame_B_C_H_W.shape[0]
            latents = get_siglip2_latents(self.model, self.processor, first_frame_B_C_H_W)
            if abs(first_frame_B_C_H_W.abs().sum()) > 1e-2:
                use_video_condition = torch.ones(batch_size, device=video_tensor.device, dtype=torch.bool)
            else:
                use_video_condition = torch.zeros(batch_size, device=video_tensor.device, dtype=torch.bool)
        return_dict = {
            "img_context_emb": latents,
        }
        if self.add_use_video_condition:
            return_dict["use_video_condition"] = use_video_condition
        return return_dict

    def details(self) -> str:
        output_key = ["img_context_emb"]
        if self.add_use_video_condition:
            output_key.append("use_video_condition")
        return f"Input key: {self.input_key} \n\tOutput key: {output_key}"


if __name__ == "__main__":
    # upload_siglip2_weights("google/siglip2-so400m-patch16-naflex")
    model, processor = get_siglip2_model_processor("google/siglip2-so400m-patch16-naflex")
