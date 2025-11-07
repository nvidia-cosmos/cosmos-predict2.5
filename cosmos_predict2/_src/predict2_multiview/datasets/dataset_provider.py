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

from typing import Optional

import omegaconf
from webdataset.handlers import warn_and_continue

import cosmos_predict2._src.imaginaire.datasets.webdataset.decoders.pickle as pickle_decoders
import cosmos_predict2._src.imaginaire.datasets.webdataset.distributors as distributors
import cosmos_predict2._src.predict2.datasets.decoders.video_decoder as video_decoder
import cosmos_predict2._src.predict2_multiview.datasets.augmentor_provider  # This is needed to register multiview augmentors before AUGMENTOR_OPTIONS is used
from cosmos_predict2._src.imaginaire.datasets.webdataset.config.schema import DatasetConfig
from cosmos_predict2._src.predict2.datasets.augmentor_provider import AUGMENTOR_OPTIONS
from cosmos_predict2._src.predict2_multiview.configs.vid2vid.defaults.driving import DrivingVideoDataloaderConfig
from cosmos_predict2._src.predict2_multiview.datasets.data_sources.data_registration import (
    create_dataset_info_fn,
    create_dataset_info_fn_simple,
)
from cosmos_predict2._src.predict2_multiview.webdataset.decoders import bin_decoder, json_decoder

_ = cosmos_predict2._src.predict2_multiview.datasets.augmentor_provider


def get_multiview_raw_webdataset(
    dataset_class: type,
    dataset_name: str,
    video_decoder_name: str,
    resolution: str,
    driving_dataloader_config: Optional[DrivingVideoDataloaderConfig] = None,
    is_train: bool = True,
    chunk_size: int = 0,
    min_fps_thres: int = 10,
    max_fps_thres: int = 60,
    augmentor_name: str = "video_basic_augmentor_v1",
    object_store: str = "s3",
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: str = "t5_xxl",
    detshuffle: bool = False,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    select_views: list[str] | None = None,
    simple_alpamayo: bool = False,
    **args,
) -> omegaconf.dictconfig.DictConfig:
    if simple_alpamayo:
        dataset_info_fn = create_dataset_info_fn_simple(
            dataset_name,
            object_store,
            driving_dataloader_config,
        )
    else:
        dataset_info_fn = create_dataset_info_fn(
            dataset_name,
            object_store,
            driving_dataloader_config,
        )
    distributor = distributors.ShardlistBasic(
        shuffle=True,
        split_by_node=True,
        split_by_worker=True,
        resume_flag=True,
        verbose=True,
        is_infinite_loader=is_train,
    )

    augmentor = AUGMENTOR_OPTIONS[augmentor_name](
        resolution=resolution,
        driving_dataloader_config=driving_dataloader_config,
        caption_type=caption_type,
        embedding_type=embedding_type,
        min_fps=min_fps_thres,
        max_fps=max_fps_thres,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
        select_views=select_views,
    )

    video_data_config = DatasetConfig(
        keys=[],  # use the per_dataset_keys in DatasetInfo instead
        buffer_size=1,
        streaming_download=True,
        dataset_info=dataset_info_fn,
        distributor=distributor,
        decoders=[
            video_decoder.construct_video_decoder(
                video_decoder_name=video_decoder_name,
                sequence_length=-1,
                chunk_size=chunk_size,
                min_fps_thres=min_fps_thres,
                max_fps_thres=max_fps_thres,
            ),
            pickle_decoders.pkl_decoder,
            bin_decoder,
            json_decoder,
        ],
        # augmentation=augmentor,
        augmentation=augmentor,
        remove_extension_from_keys=False,
        sample_keys_full_list_path=None,
    )
    return dataset_class(config=video_data_config, decoder_handler=warn_and_continue, detshuffle=detshuffle)
