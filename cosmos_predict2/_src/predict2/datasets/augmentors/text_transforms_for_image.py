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

import random
from typing import Optional

from cosmos_predict2._src.common.datasets.augmentors.v3_text_transforms import pad_and_resize
from cosmos_predict2._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.data_sources.data_registration import _CAPTION_EMBEDDING_KEY_MAPPING_IMAGES

# For the qwen captions, we have 3 variants: short, medium, long
# In addition, for synthetic data, we create prompt embeddings as well.
# There is quite a bit of entropy in the way prompt data is saved.
# Captions are saved as "prompts", while the corresponding embeddings are saved as "original_prompt"
# This part will be cleaned after synthetic data is cleaned to be in the same format as real data.
_AVAILABLE_QWEN_CAPTIONS = ["qwen2p5_7b_short", "qwen2p5_7b_medium", "qwen2p5_7b_long"]
_CAPTION_EMBEDDING_MAPPING = {
    "qwen2p5_7b_short": "qwen2p5_7b_short",
    "qwen2p5_7b_medium": "qwen2p5_7b_medium",
    "qwen2p5_7b_long": "qwen2p5_7b_long",
    "prompts": "original_prompt",
}


class TextTransformForImage(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs camera transformation.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with camera attributes added
        """

        caption_type = self.args["caption_type"]
        embedding_key_in_dict = _CAPTION_EMBEDDING_KEY_MAPPING_IMAGES[caption_type]
        embedding_type = self.args["embedding_type"]
        embedding_input_key_prefix = "" if embedding_type == "t5_xxl" else "umt5_"

        captions_key, embeddings_key = (
            f"captions_{caption_type}",
            f"{embedding_input_key_prefix}embeddings_captions_{embedding_key_in_dict}",
        )
        decoded_captions_ai = data_dict[captions_key]
        decoded_embeddings_ai = data_dict[embeddings_key]

        try:
            # Hotfix: Some captions are labeled as "captions" and some are labeled as "caption"
            # This issue needs to be fixed in the synthetic data. This is a hack and will be removed
            # once the data is cleaned.
            caption_key = "captions" if "captions" in decoded_captions_ai else "caption"
            embedding_key = "t5_xxl_fp8" if embedding_type == "t5_xxl" else "umt5_xxl"
            if caption_type == "qwen2p5_7b_v4":
                selected_caption_type = random.choice(_AVAILABLE_QWEN_CAPTIONS)
                data_dict["ai_caption"] = decoded_captions_ai[caption_key][selected_caption_type]
                t5_embedding = decoded_embeddings_ai[selected_caption_type]["embeddings"][embedding_key]
                data_dict["selected_caption_type"] = selected_caption_type
            elif caption_type == "prompts":
                data_dict["ai_caption"] = decoded_captions_ai["caption"]["prompt"]
                t5_embedding = decoded_embeddings_ai[_CAPTION_EMBEDDING_MAPPING[caption_type]]["embeddings"][
                    embedding_key
                ]
                data_dict["selected_caption_type"] = caption_type
            else:
                assert caption_type == "ai_v3p1", f"Caption type {caption_type} not supported"
                if decoded_captions_ai["had_parse_issue"]:
                    data_dict["ai_caption"] = decoded_captions_ai["captions"]["kosmos_2"]
                    t5_embedding = decoded_embeddings_ai["kosmos2"]["embeddings"][embedding_key]
                else:
                    data_dict["ai_caption"] = decoded_captions_ai["captions"]["vfc"]
                    t5_embedding = decoded_embeddings_ai["vfc_fidelity"]["embeddings"][embedding_key]

            out_t5, out_t5_mask = pad_and_resize(
                t5_embedding,
                self.args["t5_tokens"]["num"],
                is_mask_all_ones=self.args["is_mask_all_ones"],
            )
            data_dict["t5_text_embeddings"] = out_t5
            data_dict["t5_text_mask"] = out_t5_mask
        except Exception as e:
            log.warning(
                f"TextTransform dataloader error: {data_dict['__url__']}, {data_dict['__key__']}\n error {e}",
                rank0_only=False,
            )
            return None

        del data_dict[captions_key]
        del data_dict[embeddings_key]

        return data_dict


class TextTransformForImageWithoutEmbeddings(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs text transform without any embedding loading.
        This is useful for online computation.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with camera attributes added
        """

        caption_type = self.args["caption_type"]
        captions_key = f"captions_{caption_type}"
        decoded_captions_ai = data_dict[captions_key]

        try:
            # Hotfix: Some captions are labeled as "captions" and some are labeled as "caption"
            # This issue needs to be fixed in the synthetic data. This is a hack and will be removed
            # once the data is cleaned.
            caption_key = "captions" if "captions" in decoded_captions_ai else "caption"
            if caption_type == "qwen2p5_7b_v4":
                selected_caption_type = random.choice(_AVAILABLE_QWEN_CAPTIONS)
                data_dict["ai_caption"] = decoded_captions_ai[caption_key][selected_caption_type]
                data_dict["selected_caption_type"] = selected_caption_type
            elif caption_type == "prompts":
                data_dict["ai_caption"] = decoded_captions_ai["caption"]["prompt"]
                data_dict["selected_caption_type"] = caption_type
            else:
                assert caption_type == "ai_v3p1", f"Caption type {caption_type} not supported"
                if decoded_captions_ai["had_parse_issue"]:
                    data_dict["ai_caption"] = decoded_captions_ai["captions"]["kosmos_2"]
                else:
                    data_dict["ai_caption"] = decoded_captions_ai["captions"]["vfc"]

        except Exception as e:
            log.warning(
                f"TextTransform dataloader error: {data_dict['__url__']}, {data_dict['__key__']}\n error {e}",
                rank0_only=False,
            )
            return None

        del data_dict[captions_key]

        return data_dict
