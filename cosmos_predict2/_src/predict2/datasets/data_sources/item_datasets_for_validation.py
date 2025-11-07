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

import os

from cosmos_predict2._src.predict2.datasets.item_dataset import ItemDatasetConfig


def get_itemdataset_option(name: str, text_embedding_type: str = "t5_xxl") -> ItemDatasetConfig:
    item_dataset_config = ITEMDATASET_OPTIONS[name]

    if text_embedding_type != "t5_xxl":
        # For all datasets other than T5_XXL, we save the dataset in the following path
        # {data_root}/ablation_text_embeddings/{text_embedding_type}/{dataset_name}
        dataset_path = item_dataset_config.path
        dataset_path_split = dataset_path.split("/")
        is_file = os.path.splitext(dataset_path)[1] != ""

        if is_file:
            # In case of a file, we have
            # {data_root}/ablation_text_embeddings/{text_embedding_type}/{dataset_name}/{filename.ext}
            new_dataset_path = (
                dataset_path_split[0:-2]
                + ["ablation_text_embeddings", f"{text_embedding_type}"]
                + dataset_path_split[-2:]
            )
        else:
            new_dataset_path = (
                dataset_path_split[0:-1]
                + ["ablation_text_embeddings", f"{text_embedding_type}"]
                + dataset_path_split[-1:]
            )

        new_dataset_path = "/".join(new_dataset_path)

        return ItemDatasetConfig(path=new_dataset_path, length=item_dataset_config.length)
    return item_dataset_config


# length must % 8 =0 to avoid mysterious hang bug of fsdp+CP!It is tested with cp4.
ITEMDATASET_OPTIONS = {}
