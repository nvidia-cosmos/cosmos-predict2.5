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

from itertools import combinations

from cosmos_predict2._src.imaginaire import config
from cosmos_predict2._src.imaginaire.datasets.webdataset.config.schema import DatasetInfo
from cosmos_predict2._src.predict2_multiview.configs.vid2vid.defaults.driving import DrivingVideoDataloaderConfig


def _get_contiguous_view_indices_options(sample_n_views: int, ref_cam_view_idx: int, n_cameras: int):
    view_indices_options = []
    for i in range(n_cameras):
        view_indices_option_i = [i]
        for j in range(1, sample_n_views):
            view_indices_option_i.append((i + j) % n_cameras)
        if ref_cam_view_idx < 0 or ref_cam_view_idx in view_indices_option_i:
            view_indices_options.append(view_indices_option_i)
    return view_indices_options


def _get_noncontiguous_view_indices_options(sample_n_views: int, ref_cam_view_idx: int, n_cameras: int):
    """
    Return every possible set of `sample_n_views` distinct camera indices
    chosen from `n_cameras` cameras.  If `ref_cam_view_idx` ≥ 0, only keep
    those sets that include that reference camera.  Set `shuffle=True` to
    randomise the order of the returned list.
    """
    camera_indices = list(range(n_cameras))
    view_indices_options = [
        list(c) for c in combinations(camera_indices, sample_n_views) if (ref_cam_view_idx < 0 or ref_cam_view_idx in c)
    ]
    return view_indices_options


def create_dataset_info_fn(
    dataset_source_name: str,
    object_store: str,
    driving_dataloader_config: DrivingVideoDataloaderConfig,
):
    sample_n_views = driving_dataloader_config.sample_n_views
    ref_cam_view_idx = driving_dataloader_config.ref_cam_view_idx
    camera_to_view_id = driving_dataloader_config.camera_to_view_id
    overfit_firstn = driving_dataloader_config.overfit_firstn
    sample_noncontiguous_views = driving_dataloader_config.sample_noncontiguous_views
    single_caption_only = driving_dataloader_config.single_caption_only
    front_cam_key = driving_dataloader_config.front_tele_and_front_cam_keys[1]
    download_t5_tar = driving_dataloader_config.download_t5_tar
    t5_store_prefix = driving_dataloader_config.t5_store_prefix

    n_cameras = len(camera_to_view_id)
    view_id_to_camera_key = {v: k for k, v in camera_to_view_id.items()}

    if sample_noncontiguous_views:
        view_indices_options = _get_noncontiguous_view_indices_options(sample_n_views, ref_cam_view_idx, n_cameras)
    else:
        view_indices_options = _get_contiguous_view_indices_options(sample_n_views, ref_cam_view_idx, n_cameras)

    if single_caption_only:
        front_cam_view_idx = camera_to_view_id[front_cam_key]
        view_indices_options = [c for c in view_indices_options if front_cam_view_idx in c]

    train_urls = "s3_alpamayo2_train_urls.pkl.gz"
    if sample_n_views >= 6 and not single_caption_only:
        train_urls = "s3_alpamayo2_6views_urls.pkl.gz"
    data_list = train_urls if overfit_firstn <= 0 else f"s3_alpamayo2_overfit{overfit_firstn}_urls.pkl.gz"
    caption_list = "qwen_t5_tars_6cameras_mappings.pkl.gz"

    return [
        DatasetInfo(
            object_store_config=config.ObjectStoreConfig(
                enabled=True,
                credentials="credentials/s3_training.secret",
                bucket="bucket-sensitive",
            ),
            wdinfo=[
                (data_list, caption_list),
            ],
            opts={
                "download_t5_tar": download_t5_tar,
                "t5_store_config": {
                    "object_store": config.ObjectStoreConfig(
                        credentials="credentials/s3_training.secret",
                        bucket="bucket-sensitive",
                    ),
                    "skip_files_without_t5": True,
                    "prefix": t5_store_prefix,
                    "video_prefix": "predict2_multiview/mvd/AV-V2.2/videos",
                },
                "sample_n_views": sample_n_views,
                "ref_cam_view_idx": ref_cam_view_idx,
                "overfit_firstn": overfit_firstn,
                "view_indices_options": view_indices_options,
                "view_id_to_camera_key": view_id_to_camera_key,
            },
            per_dataset_keys=["dummykey"],
            source=dataset_source_name,
        )
    ]


def create_dataset_info_fn_simple(
    dataset_source_name: str,
    object_store: str,
    driving_dataloader_config: DrivingVideoDataloaderConfig,
):
    sample_n_views = driving_dataloader_config.sample_n_views
    ref_cam_view_idx = driving_dataloader_config.ref_cam_view_idx
    camera_to_view_id = driving_dataloader_config.camera_to_view_id
    overfit_firstn = driving_dataloader_config.overfit_firstn
    sample_noncontiguous_views = driving_dataloader_config.sample_noncontiguous_views
    single_caption_only = driving_dataloader_config.single_caption_only
    front_cam_key = driving_dataloader_config.front_tele_and_front_cam_keys[1]
    download_t5_tar = driving_dataloader_config.download_t5_tar
    t5_store_prefix = driving_dataloader_config.t5_store_prefix

    n_cameras = len(camera_to_view_id)
    view_id_to_camera_key = {v: k for k, v in camera_to_view_id.items()}

    if sample_noncontiguous_views:
        view_indices_options = _get_noncontiguous_view_indices_options(sample_n_views, ref_cam_view_idx, n_cameras)
    else:
        view_indices_options = _get_contiguous_view_indices_options(sample_n_views, ref_cam_view_idx, n_cameras)

    # filter duplicated view_indices_options
    view_indices_options = list(set(map(tuple, view_indices_options)))
    view_indices_options = [list(c) for c in view_indices_options]

    if single_caption_only:
        front_cam_view_idx = camera_to_view_id[front_cam_key]
        view_indices_options = [c for c in view_indices_options if front_cam_view_idx in c]

    train_chunk_id_clip_id_pkl_gz = "alpamayo_v2.2_mv_training_3000k.pkl.gz"

    return [
        DatasetInfo(
            object_store_config=config.ObjectStoreConfig(
                enabled=True,
                credentials="credentials/s3_training.secret",
                bucket="bucket-sensitive",
            ),
            wdinfo=[train_chunk_id_clip_id_pkl_gz],
            opts={
                "download_t5_tar": download_t5_tar,
                "t5_store_config": {
                    "object_store": config.ObjectStoreConfig(
                        credentials="credentials/team-sil-videogen.secret",
                        bucket="alpamayo_v2.2",
                    ),
                    "skip_files_without_t5": True,
                    "prefix": t5_store_prefix,
                    "video_prefix": "predict2_multiview/mvd/AV-V2.2/videos",
                },
                "sample_n_views": sample_n_views,
                "ref_cam_view_idx": ref_cam_view_idx,
                "overfit_firstn": overfit_firstn,
                "view_indices_options": view_indices_options,
                "view_id_to_camera_key": view_id_to_camera_key,
            },
            per_dataset_keys=["dummykey"],
            source=dataset_source_name,
        )
    ]
