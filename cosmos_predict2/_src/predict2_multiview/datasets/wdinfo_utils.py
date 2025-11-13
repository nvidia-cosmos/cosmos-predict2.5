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

"""Utility functions for handling wdinfo files."""

from typing import Literal, Mapping

from cosmos_predict2._src.imaginaire import config
from cosmos_predict2._src.imaginaire.datasets.webdataset.config.schema import DatasetInfo

DEFAULT_CATALOG: Mapping = {
    "alpamayo_dec2024": {
        "sensitive": [
            "wdinfo/alpamayo_dec2024/v0/resolution_1080/aspect_ratio_16_9/duration_10_30/wdinfo_test.json",
        ],
    },
    "mads_multiview_0823": {
        "sensitive": [
            "wdinfo/mads/cosmos-mads-dataset-transfer2-multiview-0823/v0/driving/resolution_720/aspect_ratio_16_9/duration_5_10/wdinfo_08232025.json",
        ],
    },
}


def get_video_dataset_info(
    source_name: str,
    *,
    dataset_keys: list[str] | None = None,
    object_store: Literal["gcs", "s3"] = "gcs",
    dataset_catalog: dict = DEFAULT_CATALOG,
) -> list[DatasetInfo]:
    if source_name not in dataset_catalog:
        raise KeyError(
            f"Source {source_name} not found in dataset catalog. Available keys are {dataset_catalog.keys()}"
        )

    # Create the wdinfo files here
    dataset_infos = []
    for sensitive_type, wdinfos in dataset_catalog[source_name].items():
        if object_store == "gcs":
            bucket = "bucket" if sensitive_type == "nonsensitive" else "bucket-s"
        elif object_store == "s3":
            bucket = "bucket" if sensitive_type == "nonsensitive" else "bucket-sensitive"
        else:
            raise ValueError("Cosmos data: only support gcs or s3 for object store")

        if not wdinfos:
            continue

        dataset_infos.append(
            DatasetInfo(
                object_store_config=config.ObjectStoreConfig(
                    enabled=True,
                    credentials=(
                        "credentials/s3_training.secret" if object_store == "s3" else "credentials/gcs_training.secret"
                    ),
                    bucket=bucket,
                ),
                wdinfo=wdinfos,
                per_dataset_keys=dataset_keys,
                source=source_name,
                opts={
                    "aspect_ratio": "16,9",
                },
            )
        )
    return dataset_infos
