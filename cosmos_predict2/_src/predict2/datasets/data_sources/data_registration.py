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
Dataset registration for cosmos datasets with support for different caption types.
"""

from typing import Dict, List, Optional, Tuple

from cosmos_predict2._src.imaginaire import config
from cosmos_predict2._src.imaginaire.datasets.webdataset.config.schema import DatasetInfo
from cosmos_predict2._src.imaginaire.utils import log

DATASET_OPTIONS = {}


# embeddings are packed together. Need to clean data to reduce entropy.
_CAPTION_EMBEDDING_KEY_MAPPING_IMAGES = {
    "ai_v3p1": "ai_v3p1",
    "qwen2p5_7b_v4": "qwen2p5_7b_v4",
    "prompts": "qwen2p5_7b_v4",
}


def dataset_register(key):
    log.info(f"registering dataset {key}")

    def decorator(func):
        DATASET_OPTIONS[key] = func
        return func

    return decorator


def create_dataset_infos(
    wdinfos_w_aspect_ratio_by_sensitive_type: Dict[str, List[Tuple[str, str]]],
    data_source_name: str,
    data_type: str,
    object_store: str,
    caption_type: str = "ai_v3p1",
    embedding_type: Optional[str] = "t5_xxl",
) -> Tuple[List[DatasetInfo], Optional[Dict[str, Dict[str, str]]]]:
    """
    Create dataset infos with support for different embedding types.

    Args:
        wdinfos_w_aspect_ratio_by_sensitive_type: Dictionary mapping sensitive types to lists of (wdinfo, aspect_ratio) tuples
        data_source_name: Name of the data source
        data_type: Type of data ("image" or "video")
        object_store: Object store configuration
        caption_type: Type of embedding to use (e.g., "ai_v3p1", "qwen2p5_7b_v4")
    """
    if data_type == "video":
        if embedding_type is None:
            # In this case, do not load any embeddings
            input_keys = ["video", "metas"]
        else:
            assert embedding_type in [
                "t5_xxl",
                "umt5_xxl",
            ], f"Unsupported embedding type ({embedding_type}) for video data"
            assert caption_type in [
                "vila_caption",
                "i2w_qwen2p5_7b_later_frames",
                "t2w_qwen2p5_7b",
            ], f"Unsupported caption type {caption_type} for video data"
            input_keys = [
                "video",
                "metas",
                embedding_type,
            ]  # the modality names to fetch video, caption and t5 embedding
    elif data_type == "image":
        # Support different embedding types for images

        if embedding_type is None:
            # In this case, do not load any embeddings
            input_keys = ["images", f"captions_{caption_type}"]
        else:
            assert embedding_type in [
                "t5_xxl",
                "umt5_xxl",
            ], f"Unsupported embedding type ({embedding_type}) for image data"
            embedding_type_key = _CAPTION_EMBEDDING_KEY_MAPPING_IMAGES[caption_type]

            if embedding_type == "t5_xxl":
                embedding_input_key_prefix = ""
            elif embedding_type == "umt5_xxl":
                embedding_input_key_prefix = "umT5_"
            else:
                f"Unsupported embedding type ({embedding_type}) for image data"

            input_keys = [
                "images",
                f"captions_{caption_type}",
                f"{embedding_input_key_prefix}embeddings_captions_{embedding_type_key}",
            ]

    dataset_infos = []
    for sensitive_type, wdinfos_w_aspect_ratio in wdinfos_w_aspect_ratio_by_sensitive_type.items():
        credential_path, bucket_name = get_credential_path_and_bucket(
            data_source_name, object_store, data_type, sensitive_type
        )
        dataset_infos.extend(
            [
                DatasetInfo(
                    object_store_config=config.ObjectStoreConfig(
                        enabled=True,
                        credentials=credential_path,
                        bucket=bucket_name,
                    ),
                    wdinfo=[wdinfo],
                    opts={
                        "aspect_ratio": aspect_ratio,
                    },
                    per_dataset_keys=input_keys,
                    source=data_source_name,
                )
                for wdinfo, aspect_ratio in wdinfos_w_aspect_ratio
            ]
        )

    return dataset_infos
