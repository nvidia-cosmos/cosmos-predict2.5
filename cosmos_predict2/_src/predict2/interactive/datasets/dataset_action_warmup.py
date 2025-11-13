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

import glob
import json
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.interactive.datasets.utils import extract_cr1_embedding


class ActionDatasetSFWarmup(Dataset):
    def __init__(self, data_path: str, cr1_embeddings_path: Optional[str] = None):
        super().__init__()
        self.data_path = data_path
        log.info(f"Loading dataset from {self.data_path}")

        # define subdirectories
        self.actions_subdir = "actions"
        self.images_subdir = "images"
        self.latents_subdir = "latents"

        # check for .pt files under latents_subdir
        latents_paths = sorted(glob.glob(f"{self.data_path}/{self.latents_subdir}/*.pt"))
        latents_filenames = [os.path.splitext(os.path.basename(path))[0] for path in latents_paths]
        self.samples = [
            filename
            for filename in latents_filenames
            if os.path.exists(f"{self.data_path}/{self.actions_subdir}/{filename}.json")
            and os.path.exists(f"{self.data_path}/{self.images_subdir}/{filename}.png")
        ]
        log.info(f"Found {len(self.samples)} data samples")

        # load cr1 empty string text embeddings
        if cr1_embeddings_path is not None:
            self.t5_text_embeddings = torch.load(extract_cr1_embedding(cr1_embeddings_path), map_location="cpu")[0]
            self.t5_text_mask = torch.ones(self.t5_text_embeddings.shape[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename = self.samples[index]
        action = torch.tensor(json.load(open(f"{self.data_path}/{self.actions_subdir}/{filename}.json")))  # (12, 29)
        image = Image.open(f"{self.data_path}/{self.images_subdir}/{filename}.png")
        image = torch.tensor(np.array(image)).permute(2, 0, 1)  # (3, 480, 832)
        vid_input = image.unsqueeze(1)
        vid_input = torch.cat([vid_input, torch.zeros_like(vid_input).repeat(1, action.shape[0], 1, 1)], dim=1)
        latent = torch.load(f"{self.data_path}/{self.latents_subdir}/{filename}.pt")
        latent_array = torch.stack([latent[0], latent[9], latent[18], latent[27], latent[34]], dim=0)
        out = dict(
            action=action,
            padding_mask=torch.zeros(image.shape[1], image.shape[2]),
            input_image=image,
            ode_latents=latent_array,
            data_type=DataType.VIDEO,
            ai_caption="",
            # Warmup uses no conditional frames
            num_conditional_frames=0,
            video=vid_input,
            fps=4,
        )
        if hasattr(self, "t5_text_embeddings"):
            out["t5_text_embeddings"] = self.t5_text_embeddings
            out["t5_text_mask"] = self.t5_text_mask
        return out
