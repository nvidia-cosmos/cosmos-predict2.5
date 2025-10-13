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
PYTHONPATH=. streamlit run cosmos_predict2/_src/predict2/inference/text2image.py --server.port 2222
"""

import torch

from cosmos_predict2._src.predict2.datasets.utils import IMAGE_RES_SIZE_INFO
from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

torch.enable_grad(False)


def get_sample_batch(
    resolution: str = "1024",
    aspect_ratio: str = "16,9",
    batch_size: int = 1,
) -> torch.Tensor:
    w, h = IMAGE_RES_SIZE_INFO[resolution][aspect_ratio]
    data_batch = {
        "dataset_name": "image_data",
        "images": torch.randn(batch_size, 3, h, w).cuda(),
        "t5_text_embeddings": torch.randn(batch_size, 512, 1024).cuda(),
        "fps": torch.randint(16, 32, (batch_size,)).cuda(),
        "padding_mask": torch.zeros(batch_size, 1, h, w).cuda(),
    }

    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
            data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

    return data_batch


class Text2ImageInference:
    def __init__(self, experiment_name: str, ckpt_path: str, s3_credential_path: str):
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.s3_credential_path = s3_credential_path

        model, config = load_model_from_checkpoint(
            experiment_name=experiment_name,
            config_file="cosmos_predict2/_src/predict2/configs/text2world/config.py",
            s3_checkpoint_dir=ckpt_path,
            enable_fsdp=False,
            load_ema_to_reg=True,
        )
        self.model = model
        self.config = config
        self.resolution = str(self.model.config.resolution)  # Store resolution from loaded model

    def generate_image(
        self, prompt: str, neg_prompt: str, guidance: int = 7, aspect_ratio: str = "16,9", num_samples: int = 1
    ):
        data_batch = get_sample_batch(
            resolution=self.resolution,  # Use resolution from loaded model
            aspect_ratio=aspect_ratio,
            batch_size=num_samples,
        )

        # modify the batch if prompt is provided
        if self.model.text_encoder is not None:
            # Text encoder is defined in the model class. Use it
            if prompt:
                data_batch["ai_caption"] = [prompt]
                data_batch["t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                    data_batch={"ai_caption": [prompt], "images": None},
                    input_caption_key="ai_caption",
                )
            if neg_prompt:
                data_batch["neg_t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                    data_batch={"ai_caption": [neg_prompt], "images": None},
                    input_caption_key="ai_caption",
                )
        else:
            if prompt:
                text_emb = get_text_embedding(prompt)
                data_batch["t5_text_embeddings"] = text_emb.to(dtype=torch.bfloat16).cuda()
            if neg_prompt:
                text_emb = get_text_embedding(neg_prompt)
                data_batch["neg_t5_text_embeddings"] = text_emb.to(dtype=torch.bfloat16).cuda()

        # generate samples
        sample = self.model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            seed=torch.randint(0, 10000, (1,)).item(),  # Use random seed for variation
            is_negative_prompt=bool(neg_prompt),  # Only set true if neg_prompt provided
        )
        out_samples = self.model.decode(sample)
        out_samples = (1.0 + out_samples) / 2  # Convert from [-1, 1] to [0, 1]
        out_samples = out_samples.clamp(0, 1)  # Clamp values
        out_samples = out_samples.squeeze(2)  # Convert the video tensor to image tensor

        # Now reshape
        return out_samples
