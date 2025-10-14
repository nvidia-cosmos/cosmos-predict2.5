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

import copy
from typing import Optional

import omegaconf
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from projects.cosmos.transfer2_multiview.datasets.dataset_provider import get_transfer2_multiview_dataset

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.cached_replay_dataloader import get_cached_replay_dataloader
from cosmos_predict2._src.predict2.datasets.joint_dataloader import IterativeJointDataLoader  # or RandomJointDataLoader
from cosmos_predict2._src.predict2_multiview.configs.vid2vid.defaults.driving import (
    MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION,
    DrivingVideoDataloaderConfig,
    MADSDrivingVideoDataloaderConfig,
)
from cosmos_predict2._src.predict2_multiview.datasets.alpamayo_raw_webdataset import AlpamayoRawWebdataset
from cosmos_predict2._src.predict2_multiview.datasets.alpamayo_tar_webdataset import AlpamayoTarWebdataset
from cosmos_predict2._src.predict2_multiview.datasets.dataset_provider import get_multiview_raw_webdataset


def get_cached_replay_dataloader_video_only(
    **kwargs,
):
    if "dataloaders" in kwargs:
        log.info(f"kwargs dataloaders: {kwargs['dataloaders']}")
        del kwargs["dataloaders"]
    return get_cached_replay_dataloader(
        **kwargs,
    )


def get_video_dataloader_multiview(
    dataset_name: str,
    object_store: str,
    driving_dataloader_config: MADSDrivingVideoDataloaderConfig,
    resolution: str = "720",
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> omegaconf.dictconfig.DictConfig:
    return L(get_cached_replay_dataloader_video_only)(
        dataset=L(get_transfer2_multiview_dataset)(
            dataset_name=dataset_name,
            video_decoder_name="video_naive_bytes",
            augmentor_name="video_basic_augmentor_v2_multiview_with_control",
            driving_dataloader_config=driving_dataloader_config,
            resolution=resolution,
            is_train=True,
            object_store=object_store,
            chunk_size=256,
        ),
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
        cache_replay_name="video_dataloader",
    )


def get_driving_video_dataloader(
    dataset_class: type,
    dataset_name: str,
    is_train: bool = True,
    resolution: str = "",
    driving_dataloader_config: Optional[DrivingVideoDataloaderConfig] = None,
    num_workers: int = 8,
    prefetch_factor: int = 2,
) -> omegaconf.dictconfig.DictConfig:
    return L(get_cached_replay_dataloader_video_only)(
        dataset=L(get_multiview_raw_webdataset)(
            dataset_class=dataset_class,
            dataset_name=dataset_name,
            video_decoder_name="video_naive_bytes",
            resolution=resolution,
            driving_dataloader_config=driving_dataloader_config,
            chunk_size=256,
            is_train=is_train,
            min_fps_thres=10,
            max_fps_thres=60,
            augmentor_name="video_basic_augmentor_v2_multiview",
            object_store="s3",
            caption_type="t2w_qwen2p5_7b",
            embedding_type="t5_xxl",
            detshuffle=False,
            long_caption_ratio=7,
            medium_caption_ratio=2,
            short_caption_ratio=1,
            user_caption_ratio=90,
        ),
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
        cache_replay_name="video_dataloader",
    )


alpamayo_front_cam_key = "camera_front_wide_120fov"
alpamayo_front_tele_cam_key = "camera_front_tele_30fov"
alpamayo_front_tele_and_front_cam_keys = (alpamayo_front_tele_cam_key, alpamayo_front_cam_key)

#### 3 views ####
alpamayo_camera_to_view_id_3views = {
    "camera_front_wide_120fov": 0,
    "camera_cross_left_120fov": 1,
    "camera_cross_right_120fov": 2,
}

alpamayo_view_id_to_caption_id_3views = {
    0: 0,
    1: 1,
    2: 2,
}

alpamayo_camera_to_caption_prefix_3views = {
    "camera_front_wide_120fov": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "camera_cross_left_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
    "camera_cross_right_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
}

#### 4 views ####
alpamayo_camera_to_view_id_4views = {
    "camera_front_wide_120fov": 0,
    "camera_cross_left_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_rear_tele_30fov": 3,
}

alpamayo_view_id_to_caption_id_4views = {
    0: 0,
    1: 1,
    2: 2,
    3: 5,
}

alpamayo_camera_to_caption_prefix_4views = {
    "camera_front_wide_120fov": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "camera_cross_left_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
    "camera_cross_right_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
    "camera_rear_tele_30fov": "The video is captured from a camera mounted on a car. The camera is facing backwards.",
}

#### 6 views ####
alpamayo_camera_to_view_id_6views = {
    "camera_front_wide_120fov": 0,
    "camera_cross_left_120fov": 5,
    "camera_cross_right_120fov": 1,
    "camera_rear_left_70fov": 4,
    "camera_rear_right_70fov": 2,
    "camera_rear_tele_30fov": 3,
}
alpamayo_view_id_to_caption_id_6views = {
    0: 0,
    5: 1,
    1: 2,
    4: 3,
    2: 4,
    3: 5,
}
alpamayo_camera_to_caption_prefix_6views = {
    "camera_front_wide_120fov": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "camera_rear_tele_30fov": "The video is captured from a camera mounted on a car. The camera is facing backwards.",
    "camera_cross_left_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
    "camera_cross_right_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
    "camera_rear_right_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
    "camera_rear_left_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
}

alpamayo_camera_to_view_id_7views, alpamayo_view_id_to_caption_id_7views, alpamayo_camera_to_caption_prefix_7views = (
    alpamayo_camera_to_view_id_6views.copy(),
    alpamayo_view_id_to_caption_id_6views.copy(),
    alpamayo_camera_to_caption_prefix_6views.copy(),
)
alpamayo_camera_to_view_id_7views[alpamayo_front_tele_and_front_cam_keys[0]] = 6
alpamayo_view_id_to_caption_id_7views[6] = 0
alpamayo_camera_to_caption_prefix_7views[alpamayo_front_tele_and_front_cam_keys[0]] = (
    "The video is captured from a telephoto camera mounted on a car. The camera is facing forward."
)

H_W = {
    "720p": (720, 1280),
    "480p": (480, 832),
    "480": (432, 768),
    "720": (704, 1280),
}


def register_alpamayo_jointsinglemulticaption_dataloaders():
    for num_views in [3, 4, 6, 7]:
        for num_video_frames_loaded_per_view in [29, 61, 93]:
            for num_video_frames_per_view in [29, 61, 93]:
                register_alpamayo_jointsinglemulticaption_dataloader(
                    num_video_frames_loaded_per_view, num_video_frames_per_view, num_views
                )


def register_alpamayo_jointsinglemulticaption_dataloader(
    num_video_frames_loaded_per_view: int,
    num_video_frames_per_view: int,
    num_views: int,
):
    camera_to_view_id_config = {
        7: alpamayo_camera_to_view_id_7views,
        6: alpamayo_camera_to_view_id_6views,
        4: alpamayo_camera_to_view_id_4views,
        3: alpamayo_camera_to_view_id_3views,
    }
    view_id_to_caption_id_config = {
        7: alpamayo_view_id_to_caption_id_7views,
        6: alpamayo_view_id_to_caption_id_6views,
        4: alpamayo_view_id_to_caption_id_4views,
        3: alpamayo_view_id_to_caption_id_3views,
    }
    camera_to_caption_prefix_config = {
        7: alpamayo_camera_to_caption_prefix_7views,
        6: alpamayo_camera_to_caption_prefix_6views,
        4: alpamayo_camera_to_caption_prefix_4views,
        3: alpamayo_camera_to_caption_prefix_3views,
    }

    for resolution in ["720p", "480p"]:
        resolution_str = "" if resolution == "720" else f"_{resolution}"
        H, W = H_W[resolution]
        for hybrid in [True]:
            hybrid_str = "_hybrid_captions" if hybrid else ""
            alpamayo_config_single_caption_no_view_prefix = DrivingVideoDataloaderConfig(
                sample_n_views=num_views,
                num_video_frames_per_view=num_video_frames_per_view,
                num_video_frames_loaded_per_view=num_video_frames_loaded_per_view,
                sample_noncontiguous_views=False,
                ref_cam_view_idx=-1,
                overfit_firstn=-1,
                camera_to_view_id=camera_to_view_id_config[num_views],
                view_id_to_caption_id=view_id_to_caption_id_config[num_views],
                camera_to_caption_prefix=camera_to_caption_prefix_config[num_views],
                front_tele_and_front_cam_keys=alpamayo_front_tele_and_front_cam_keys,
                concat_viewt5=False,
                no_view_prefix=True,
                single_caption_only=True,
                H=H,
                W=W,
                hint_keys="",
                download_t5_tar=False if hybrid else True,
                t5_store_prefix=(
                    "predict2_multiview/mvd/AV-V2.2/alpamayo_caption_hybrid_dict/qwen_t5_tars_6cameras"
                    if hybrid
                    else "predict2_multiview/mvd/AV-V2.2/alpamayo_caption_t5/qwen_t5_tars_6cameras"
                ),
            )
            alpamayo_config_multiple_captions_view_prefix = copy.deepcopy(alpamayo_config_single_caption_no_view_prefix)
            alpamayo_config_multiple_captions_view_prefix.no_view_prefix = False
            alpamayo_config_multiple_captions_view_prefix.single_caption_only = False

            alpamayo_config_single_caption_view_prefix = copy.deepcopy(alpamayo_config_single_caption_no_view_prefix)
            alpamayo_config_single_caption_view_prefix.no_view_prefix = False

            alpamayo_loader_single_caption_no_view_prefix = L(get_driving_video_dataloader)(
                dataset_class=AlpamayoTarWebdataset,
                dataset_name=f"alpamayo_v2_7cameras_tar_sample7views_{num_video_frames_loaded_per_view}framesto{num_video_frames_per_view}_res{resolution}_noviewprefix_1cap_norepeat{hybrid_str}",
                is_train=True,
                resolution=resolution,
                driving_dataloader_config=alpamayo_config_single_caption_no_view_prefix,
                num_workers=4,
                prefetch_factor=4,
            )

            alpamayo_loader_single_caption_view_prefix = L(get_driving_video_dataloader)(
                dataset_class=AlpamayoTarWebdataset,
                dataset_name=f"alpamayo_v2_7cameras_tar_sample7views_{num_video_frames_loaded_per_view}framesto{num_video_frames_per_view}_res{resolution}_viewprefix_1cap_norepeat{hybrid_str}",
                is_train=True,
                resolution=resolution,
                driving_dataloader_config=alpamayo_config_single_caption_view_prefix,
                num_workers=4,
                prefetch_factor=4,
            )

            alpamayo_loader_multiple_captions_view_prefix = L(get_driving_video_dataloader)(
                dataset_class=AlpamayoTarWebdataset,
                dataset_name=f"alpamayo_v2_7cameras_tar_sample7views_{num_video_frames_loaded_per_view}framesto{num_video_frames_per_view}_res{resolution}_norepeat{hybrid_str}",
                is_train=True,
                resolution=resolution,
                driving_dataloader_config=alpamayo_config_multiple_captions_view_prefix,
                num_workers=4,
                prefetch_factor=4,
            )

            if num_video_frames_loaded_per_view == num_video_frames_per_view:
                frames_str = f"_{num_video_frames_per_view}frames"
            else:
                frames_str = f"_{num_video_frames_loaded_per_view}framesto{num_video_frames_per_view}"
            cs = ConfigStore.instance()

            # the following fake mads loader is added with ratio 0, so it will not be used
            # this is a hack to avoid error when running generation pipeline without mads dataset
            mads_driving_dataloader_config = copy.deepcopy(MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION[resolution])
            mads_driving_dataloader_config.num_video_frames_loaded_per_view = num_video_frames_loaded_per_view
            mads_driving_dataloader_config.num_video_frames_per_view = num_video_frames_per_view

            mads_loader = L(get_video_dataloader_multiview)(
                dataset_name="cosmos_transfer2_av_mads_mv_20250710_video_whole",
                object_store="s3",
                resolution=resolution,
                driving_dataloader_config=mads_driving_dataloader_config,
                num_workers=1,
                prefetch_factor=0,
            )

            view_str = f"_{num_views}views" if num_views != 7 else ""
            for no_view_prefix_1cap in [True, False]:
                if no_view_prefix_1cap:
                    prefix_str = "noviewprefix"
                    alpamayo_1cap = alpamayo_loader_single_caption_no_view_prefix
                else:
                    prefix_str = "viewprefix"
                    alpamayo_1cap = alpamayo_loader_single_caption_view_prefix
                cs.store(
                    group="data_train",
                    package="dataloader_train",
                    name=f"video_joint_alpamayo1cap{prefix_str}_allcapsviewprefix{resolution_str}{frames_str}{hybrid_str}{view_str}",
                    node=L(IterativeJointDataLoader)(
                        dataloaders={
                            "alpamayo_1cap": {
                                "dataloader": instantiate(alpamayo_1cap),
                                "ratio": 1,
                            },
                            "alpamayo_allcaps": {
                                "dataloader": instantiate(alpamayo_loader_multiple_captions_view_prefix),
                                "ratio": 1,
                            },
                            "mads": {
                                "dataloader": instantiate(mads_loader),
                                "ratio": 0,
                            },
                        }
                    ),
                )


def register_alpamayo_dataloader():
    cs = ConfigStore.instance()
    for dataset_type, dataset_class in [("", AlpamayoRawWebdataset), ("_tar", AlpamayoTarWebdataset)]:
        # for sample_n_views in range(1, 8):
        for sample_n_views in [1, 4, 6, 7]:
            include_front_tele = sample_n_views == 7
            camera_to_view_id = (
                alpamayo_camera_to_view_id_7views if include_front_tele else alpamayo_camera_to_view_id_6views
            )
            view_id_to_caption_id = (
                alpamayo_view_id_to_caption_id_7views if include_front_tele else alpamayo_view_id_to_caption_id_6views
            )
            camera_to_caption_prefix = (
                alpamayo_camera_to_caption_prefix_7views
                if include_front_tele
                else alpamayo_camera_to_caption_prefix_6views
            )

            # for num_video_frames_per_view in [75, 85, 150]:
            # for resolution in ["480p", "720p", "480", "720"]:
            for num_video_frames_loaded_per_view in [
                85,
                29,
            ]:
                for num_video_frames_per_view in [85, 29]:
                    for resolution in ["720", "720p"]:
                        for cam_idx in [-1]:
                            cam_str = f"_cam{cam_idx}" if cam_idx != -1 else ""
                            for concat_viewt5 in [
                                False,
                            ]:
                                for overfit_firstn in [-1]:  # , 1, 10, 100]:
                                    for sample_noncontiguous_views in [
                                        False,
                                    ]:
                                        for no_view_prefix in [True, False]:
                                            for single_caption_only in [False, True]:
                                                for hint_keys in ["", "edge2x"]:
                                                    single_caption_only_str = "_1cap" if single_caption_only else ""
                                                    no_view_prefix_str = "_noviewprefix" if no_view_prefix else ""
                                                    noncontiguous_str = "_noncont" if sample_noncontiguous_views else ""
                                                    overfit_str = (
                                                        f"_overfitfirst{overfit_firstn}" if overfit_firstn != -1 else ""
                                                    )
                                                    concat_viewt5_str = "_concatviewt5" if concat_viewt5 else ""
                                                    out_frames_str = (
                                                        f"to{num_video_frames_per_view}"
                                                        if num_video_frames_loaded_per_view != num_video_frames_per_view
                                                        else ""
                                                    )
                                                    hint_keys_str = f"_hint{hint_keys}" if hint_keys else ""
                                                    for hybrid in [True, False]:
                                                        hybrid_str = "_hybrid_captions" if hybrid else ""
                                                        dataset_name = f"alpamayo_v2_7cameras{dataset_type}_sample{sample_n_views}views{noncontiguous_str}_{num_video_frames_loaded_per_view}frames{out_frames_str}_res{resolution}{cam_str}{concat_viewt5_str}{overfit_str}{no_view_prefix_str}{single_caption_only_str}{hint_keys_str}_norepeat{hybrid_str}"
                                                        log.info(f"Registering dataset: {dataset_name}")
                                                        config = DrivingVideoDataloaderConfig(
                                                            sample_n_views=sample_n_views,
                                                            num_video_frames_per_view=num_video_frames_per_view,
                                                            num_video_frames_loaded_per_view=num_video_frames_loaded_per_view,
                                                            sample_noncontiguous_views=sample_noncontiguous_views,
                                                            ref_cam_view_idx=cam_idx,
                                                            overfit_firstn=overfit_firstn,
                                                            camera_to_view_id=camera_to_view_id,
                                                            view_id_to_caption_id=view_id_to_caption_id,
                                                            camera_to_caption_prefix=camera_to_caption_prefix,
                                                            front_tele_and_front_cam_keys=alpamayo_front_tele_and_front_cam_keys,
                                                            concat_viewt5=concat_viewt5,
                                                            no_view_prefix=no_view_prefix,
                                                            single_caption_only=single_caption_only,
                                                            H=H_W[resolution][0],
                                                            W=H_W[resolution][1],
                                                            hint_keys=hint_keys,
                                                            download_t5_tar=False if hybrid else True,
                                                            t5_store_prefix=(
                                                                "predict2_multiview/mvd/AV-V2.2/alpamayo_caption_hybrid_dict/qwen_t5_tars_6cameras"
                                                                if hybrid
                                                                else "predict2_multiview/mvd/AV-V2.2/alpamayo_caption_t5/qwen_t5_tars_6cameras"
                                                            ),
                                                        )
                                                        cs.store(
                                                            group="data_train",
                                                            package="dataloader_train",
                                                            name=dataset_name,
                                                            node=get_driving_video_dataloader(
                                                                dataset_class=dataset_class,
                                                                dataset_name=dataset_name,
                                                                is_train=True,
                                                                resolution=resolution,
                                                                driving_dataloader_config=config,
                                                                num_workers=8,
                                                                prefetch_factor=4,
                                                            ),
                                                        )


def register_alpamayo_mads_joint_dataloaders():
    for num_video_frames_loaded_per_view in [85, 57, 29]:
        register_alpamayo_mads_joint_dataloader(
            num_video_frames_loaded_per_view=num_video_frames_loaded_per_view, num_video_frames_per_view=29
        )
    register_alpamayo_mads_joint_dataloader(num_video_frames_loaded_per_view=61, num_video_frames_per_view=61)


def register_alpamayo_mads_joint_dataloader(num_video_frames_loaded_per_view: int, num_video_frames_per_view: int):
    for resolution in ["720", "720p", "480p"]:
        resolution_str = "" if resolution == "720" else f"_{resolution}"
        H, W = H_W[resolution]
        mads_driving_dataloader_config = copy.deepcopy(MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION[resolution])
        mads_driving_dataloader_config.num_video_frames_loaded_per_view = num_video_frames_loaded_per_view
        mads_driving_dataloader_config.num_video_frames_per_view = num_video_frames_per_view
        for hybrid in [True, False]:
            hybrid_str = "_hybrid_captions" if hybrid else ""
            alpamayo_config = DrivingVideoDataloaderConfig(
                sample_n_views=7,
                num_video_frames_per_view=num_video_frames_per_view,
                num_video_frames_loaded_per_view=num_video_frames_loaded_per_view,
                sample_noncontiguous_views=False,
                ref_cam_view_idx=-1,
                overfit_firstn=-1,
                camera_to_view_id=alpamayo_camera_to_view_id_7views,
                view_id_to_caption_id=alpamayo_view_id_to_caption_id_7views,
                camera_to_caption_prefix=alpamayo_camera_to_caption_prefix_7views,
                front_tele_and_front_cam_keys=alpamayo_front_tele_and_front_cam_keys,
                concat_viewt5=False,
                no_view_prefix=True,
                single_caption_only=True,
                H=H,
                W=W,
                hint_keys="",
                download_t5_tar=False if hybrid else True,
                t5_store_prefix=(
                    "predict2_multiview/mvd/AV-V2.2/alpamayo_caption_hybrid_dict/qwen_t5_tars_6cameras"
                    if hybrid
                    else "predict2_multiview/mvd/AV-V2.2/alpamayo_caption_t5/qwen_t5_tars_6cameras"
                ),
            )

            alpamayo_loader = L(get_driving_video_dataloader)(
                dataset_class=AlpamayoTarWebdataset,
                dataset_name=f"alpamayo_v2_7cameras_tar_sample7views_{num_video_frames_loaded_per_view}framesto{num_video_frames_per_view}_res{resolution}_noviewprefix_1cap_norepeat{hybrid_str}",
                is_train=True,
                resolution=resolution,
                driving_dataloader_config=alpamayo_config,
                num_workers=4,
                prefetch_factor=4,
            )

            mads_loader = L(get_video_dataloader_multiview)(
                dataset_name="cosmos_transfer2_av_mads_mv_20250710_video_whole",
                object_store="s3",
                resolution=resolution,
                driving_dataloader_config=mads_driving_dataloader_config,
                num_workers=4,
                prefetch_factor=4,
            )
            if num_video_frames_loaded_per_view == 85 and num_video_frames_per_view == 29:
                frames_str = ""
            elif num_video_frames_loaded_per_view == num_video_frames_per_view:
                frames_str = f"_{num_video_frames_per_view}frames"
            else:
                frames_str = f"_{num_video_frames_loaded_per_view}framesto{num_video_frames_per_view}"
            cs = ConfigStore.instance()
            cs.store(
                group="data_train",
                package="dataloader_train",
                name=f"video_joint_alpamayov2_mads{resolution_str}{frames_str}{hybrid_str}",
                node=L(IterativeJointDataLoader)(
                    dataloaders={
                        "alpamayo": {"dataloader": instantiate(alpamayo_loader), "ratio": 10},
                        "mads": {"dataloader": instantiate(mads_loader), "ratio": 1},
                    }
                ),
            )
