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

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False
from webdataset.handlers import warn_and_continue

import cosmos_predict2._src.imaginaire.datasets.webdataset.decoders.image as image_decoders
import cosmos_predict2._src.imaginaire.datasets.webdataset.decoders.pickle as pickle_decoders
import cosmos_predict2._src.imaginaire.datasets.webdataset.distributors as distributors
import cosmos_predict2._src.predict2.datasets.decoders.video_decoder as video_decoder
import cosmos_predict2._src.predict2.datasets.distributor.parallel_sync_multi_aspect_ratio as parallel_sync_multi_aspect_ratio
import cosmos_predict2._src.predict2.datasets.webdataset as webdataset
from cosmos_predict2._src.imaginaire.datasets.webdataset.config.schema import DatasetConfig
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO
from cosmos_predict2._src.transfer2.datasets.augmentor_provider import AUGMENTOR_OPTIONS
from cosmos_predict2._src.transfer2.datasets.data_sources.data_registration import DATASET_OPTIONS


def get_video_dataset(
    dataset_name: str,
    video_decoder_name: str,
    resolution: str,
    is_train: bool = True,
    num_video_frames: int = 121,
    chunk_size: int = 0,
    min_fps_thres: int = 10,
    max_fps_thres: int = 60,
    dataset_resolution_type: str = "all",
    augmentor_name: str = "video_basic_augmentor_v1",
    object_store: str = "s3",
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: str = "t5_xxl",
    detshuffle: bool = False,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    use_native_fps: bool = False,
    control_input_type: str = "edge_vis_depth_seg",
    dataset_loading_keys: list = [],
    use_control_mask_prob: float = 0.0,
    num_control_inputs_prob: list[float] = [1.0, 0.0, 0.0, 0.0],
    edge_t_lower: Optional[int] = None,
    edge_t_upper: Optional[int] = None,
) -> omegaconf.dictconfig.DictConfig:
    assert resolution in VIDEO_RES_SIZE_INFO.keys(), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."
    assert object_store in ["s3", "swiftstack"], "We support s3 and swiftstack only."
    basic_augmentor_names = [
        "video_basic_augmentor_v2",
        "video_basic_augmentor_v2_with_control",
        # This is the augmentor which load pre-rendered control video. How only AV experiment uses it.
        "video_basic_augmentor_with_control_input",
        "video_basic_augmentor_v2_with_control_and_image_context",
    ]
    if video_decoder_name == "video_naive_bytes":
        assert augmentor_name in basic_augmentor_names, (
            "We can only use video_basic_augmentor_v2 with video_naive_bytes decoder."
        )
    if augmentor_name in basic_augmentor_names:
        assert video_decoder_name == "video_naive_bytes", (
            "We can only use video_naive_bytes decoder with video_basic_augmentor_v2."
        )

    assert dataset_resolution_type in [
        "all",
        "gt720p",
        "gt1080p",
    ], f"The provided dataset resolution type {dataset_resolution_type} is not supported."
    # dataset_resolution_type
    # -- all - uses all dataset resolutions
    # -- gt720p - Uses only resolutions >= 720p
    # -- gt1080p - Uses only resolutions >= 1080p
    dataset_info_fn = DATASET_OPTIONS[dataset_name]
    dataset_info = dataset_info_fn(
        object_store, caption_type, embedding_type, dataset_resolution_type, control_input_type
    )
    if dataset_loading_keys != []:
        dataset_info[0].per_dataset_keys = dataset_loading_keys  # ["video", "metas", "t5_xxl", ...]

    augmentor = AUGMENTOR_OPTIONS[augmentor_name](
        resolution=resolution,
        caption_type=caption_type,
        embedding_type=embedding_type,
        min_fps=min_fps_thres,
        max_fps=max_fps_thres,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
        num_video_frames=num_video_frames,
        use_native_fps=use_native_fps,
        control_input_type=control_input_type,
        use_control_mask_prob=use_control_mask_prob,
        num_control_inputs_prob=num_control_inputs_prob,
        edge_t_lower=edge_t_lower,
        edge_t_upper=edge_t_upper,
    )

    if parallel_state.is_initialized() and (
        parallel_state.get_context_parallel_world_size() > 1
        or parallel_state.get_tensor_model_parallel_world_size() > 1
    ):
        log.critical(
            f"Using parallelism size CP :{parallel_state.get_context_parallel_world_size()}, TP :{parallel_state.get_tensor_model_parallel_world_size()} for video dataset, switch to ShardlistMultiAspectRatioParallelSync distributor"
        )
        distributor = parallel_sync_multi_aspect_ratio.ShardlistMultiAspectRatioParallelSync(
            shuffle=True,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=True,
            is_infinite_loader=is_train,
        )
        detshuffle = True  # overwrite detshuffle.
    else:
        distributor = distributors.ShardlistMultiAspectRatio(
            shuffle=True,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=False,
            is_infinite_loader=is_train,
        )

    video_data_config = DatasetConfig(
        keys=[],  # use the per_dataset_keys in DatasetInfo instead
        buffer_size=100,
        streaming_download=True,
        dataset_info=dataset_info,
        distributor=distributor,
        decoders=[
            video_decoder.construct_video_decoder(
                video_decoder_name=video_decoder_name,
                sequence_length=num_video_frames,
                chunk_size=chunk_size,
                min_fps_thres=min_fps_thres,
                max_fps_thres=max_fps_thres,
            ),
            pickle_decoders.pkl_decoder,
        ],
        augmentation=augmentor,
        remove_extension_from_keys=True,
        sample_keys_full_list_path=None,
    )

    return webdataset.Dataset(config=video_data_config, decoder_handler=warn_and_continue, detshuffle=detshuffle)


def get_image_dataset(
    dataset_name: str,
    resolution: str,
    dataset_resolution_type: str = "all",
    is_train: bool = True,
    augmentor_name: str = "image_basic_augmentor",
    object_store: str = "s3",
    detshuffle: bool = False,
    caption_type: str = "ai_v3p1",
    embedding_type: str = "t5_xxl",
) -> omegaconf.dictconfig.DictConfig:
    assert resolution in IMAGE_RES_SIZE_INFO.keys(), "The provided resolution cannot be found in IMAGE_RES_SIZE_INFO."
    assert object_store in ["s3", "swiftstack"], "We support s3 and swiftstack only."
    assert dataset_resolution_type in [
        "all",
        "gt720p",
        "gt1080p",
    ], f"The provided dataset resolution type {dataset_resolution_type} is not supported."
    # dataset_resolution_type
    # -- all - uses all dataset resolutions
    # -- gt720p - Uses only resolutions >= 720p
    # -- gt1080p - Uses only resolutions >= 1080p
    dataset_info_fn = DATASET_OPTIONS[dataset_name]
    dataset_info = dataset_info_fn(object_store, caption_type, embedding_type, dataset_resolution_type)
    augmentation = AUGMENTOR_OPTIONS[augmentor_name](
        resolution=resolution,
        caption_type=caption_type,
        embedding_type=embedding_type,
    )

    if parallel_state.is_initialized() and (
        parallel_state.get_context_parallel_world_size() > 1
        or parallel_state.get_tensor_model_parallel_world_size() > 1
    ):
        log.critical(
            f"Using parallelism size CP :{parallel_state.get_context_parallel_world_size()}, TP :{parallel_state.get_tensor_model_parallel_world_size()} for image dataset, switch to ShardlistMultiAspectRatioParallelSync distributor"
        )
        distributor = parallel_sync_multi_aspect_ratio.ShardlistMultiAspectRatioParallelSync(
            shuffle=True,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=True,
            is_infinite_loader=is_train,
        )
        detshuffle = True  # overwrite detshuffle.
    else:
        distributor = distributors.ShardlistMultiAspectRatio(
            shuffle=True,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=False,
            is_infinite_loader=is_train,
        )

    image_data_config = DatasetConfig(
        keys=[],
        # https://gitlab-master.nvidia.com/dir/imaginaire4/-/issues/119
        buffer_size=25,
        streaming_download=True,
        dataset_info=dataset_info,
        distributor=distributor,
        decoders=[
            image_decoders.pil_loader,
            pickle_decoders.pkl_decoder,
        ],
        augmentation=augmentation,
    )

    return webdataset.Dataset(config=image_data_config, detshuffle=detshuffle)
