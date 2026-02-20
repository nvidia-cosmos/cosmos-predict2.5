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
Configs for submitting large scale job using ./_submit.py from local workstation.

Recommended usage: the config here serve as a base config for large scale job, don't modify this script; instead, override specific fields in the config
by creating a new experiment in ./experiment_list.py. See examples there for how to add new experiments and how to submit a job.
"""

import functools

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.datasets.cached_replay_dataloader import duplicate_batches_random
from cosmos_predict2._src.predict2.models.video2world_model import HighSigmaStrategy

# new load path for 720p
BASE_CKPT_720 = dict(
    # load_path="bucket/cosmos_diffusion_v2/vid2vid_control_base/Stage-c_pt_4-Index-11-Size-2B-Res-720-Fps-16-Note-06_02_TnI2V_data_all_hq/checkpoints/iter_000050000",
    load_path="bucket/cosmos_diffusion_v2/vid2vid_control_base/Stage-c_pt_4-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_TnI2V/checkpoints/iter_000040000",
    credentials="credentials/s3_checkpoint.secret",
)

_TRAINER_DEBUG_CONFIG = dict(
    max_iter=100,
    logging_iter=5,
    callbacks=dict(
        every_n_sample_reg=dict(
            every_n=1,
        ),
        every_n_sample_ema=dict(
            every_n=1,
        ),
        reg_model_image2video_sora_val_sampling=dict(
            every_n=1000000,
            is_debug=True,
            latent_video_length="${model.config.state_t}",
        ),
        ema_model_image2video_sora_val_sampling=dict(
            every_n=1000000,
            is_debug=True,
            latent_video_length="${model.config.state_t}",
        ),
        reg_model_image2video_vbench_val_sampling=dict(
            every_n=1000000,
            is_debug=True,
            latent_video_length="${model.config.state_t}",
        ),
        ema_model_image2video_vbench_val_sampling=dict(
            every_n=1000000,
            is_debug=True,
            latent_video_length="${model.config.state_t}",
        ),
    ),
)

_TRAINER_LARGE_SCALE_CONFIG = dict(
    max_iter=100000,
    logging_iter=100,
    callbacks=dict(
        every_n_sample_reg=dict(
            every_n=1000,
        ),
        every_n_sample_ema=dict(
            every_n=1000,
        ),
        reg_model_image2video_sora_val_sampling=dict(
            every_n=1000000,
            latent_video_length="${model.config.state_t}",
        ),
        ema_model_image2video_sora_val_sampling=dict(
            every_n=1000000,
            latent_video_length="${model.config.state_t}",
        ),
        reg_model_image2video_vbench_val_sampling=dict(
            every_n=1000000,
            latent_video_length="${model.config.state_t}",
        ),
        ema_model_image2video_vbench_val_sampling=dict(
            every_n=1000000,
            latent_video_length="${model.config.state_t}",
        ),
    ),
)


_CKPT_DEBUG_CONFIG = dict(
    save_iter=50,
    load_training_state=False,
    strict_resume=False,
)


vid2vid_2B_control_480p_control_layer14_av = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_480p_control_layer7",
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_202506_swiftstack"},
            {"override /conditioner": "video_prediction_control_conditioner_av"},
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_480p_control_layer14_av",
        ),
        model=dict(
            config=dict(
                state_t=24,  # 93 frames
                net=dict(
                    vace_block_every_n=2,
                ),
            ),
        ),
        trainer=_TRAINER_LARGE_SCALE_CONFIG,
        dataloader_train=dict(
            batch_size=1,
            use_cache=True,
            cache_size=16,
            concat_size=1,
            cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
            dataset=dict(
                resolution="${model.config.resolution}",
                augmentor_name="video_basic_augmentor_with_control_input",
                video_decoder_name="video_naive_bytes",
                caption_type="t2w_qwen2p5_7b",
                embedding_type="t5_xxl",
                min_fps_thres=10,
                max_fps_thres=60,
                num_video_frames=93,
                use_native_fps=True,
                dataset_loading_keys=["video", "metas", "t5_xxl", "hdmap_bbox"],
            ),
        ),
    ),
    flags={"allow_objects": True},
)


vid2vid_2B_control_480p_control_layer14_av_full_65k = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_480p_control_layer14_av",
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_20250701_swiftstack"},
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_480p_control_layer14_av_full_65k",
        ),
        model=dict(
            config=dict(
                preset_hint_keys=[
                    "control_input_edge",
                    "control_input_blur",
                    "control_input_hdmap_bbox",
                ]  # set this for ckpt compatibility, need to change to hdmap only later
            )
        ),
    ),
)

vid2vid_2B_control_480p_control_layer14_av_full_65k_reset_rescheduler = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_480p_control_layer14_av_full_65k",
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_20250701_swiftstack"},
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_480p_control_layer14_av_full_65k_reset_rescheduler",
        ),
        model=dict(
            config=dict(
                high_sigma_strategy=str(HighSigmaStrategy.UNIFORM80_2000),
                high_sigma_ratio=0.0,
                sde=dict(
                    p_mean=0.0,
                    p_std=1.0,
                    sigma_max=80,
                    sigma_min=0.0002,
                ),
            ),
        ),
        checkpoint=dict(
            load_path="cosmos_transfer2/vid2vid_2B_control_av/vid2vid_2B_control_480p_control_layer14_av_full_65k/checkpoints/iter_000010000/",
        ),
    ),
)


vid2vid_2B_control_480p_control_layer14_av_full_65k_online_testing = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_480p_control_layer14_av_full_65k",
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_20250701_swiftstack"},
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_480p_control_layer14_av_full_65k_online_testing",
        ),
        trainer=dict(
            max_iter=2,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1,
                ),
                every_n_sample_ema=dict(
                    every_n=1,
                ),
            ),
        ),
        checkpoint=dict(
            load_path="cosmos_transfer2/vid2vid_2B_control_av/vid2vid_2B_control_480p_control_layer14_av_full_65k/checkpoints/iter_000008000/",
        ),
    ),
)


"""
720p models
"""
vid2vid_2B_control_true_720p_control_layer14_av_full_65k = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_720p_control_layer14",
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_20250701_swiftstack"},
            {"override /conditioner": "video_prediction_control_conditioner_av"},
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_true_720p_control_layer14_av_full_65k",
        ),
        model=dict(
            config=dict(
                state_t=24,  # 93 frames
                base_load_from=BASE_CKPT_720,
                fsdp_shard_size=32,
                resolution="720p",
            ),
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        trainer=_TRAINER_LARGE_SCALE_CONFIG,
        dataloader_train=dict(
            batch_size=1,
            use_cache=True,
            cache_size=32,
            concat_size=1,
            cache_augment_fn=functools.partial(duplicate_batches_random, n=1.8),
            dataset=dict(
                resolution="${model.config.resolution}",
                augmentor_name="video_basic_augmentor_with_control_input",
                video_decoder_name="video_naive_bytes",
                caption_type="t2w_qwen2p5_7b",
                embedding_type="t5_xxl",
                min_fps_thres=10,
                max_fps_thres=60,
                num_video_frames=93,
                use_native_fps=True,
                dataset_loading_keys=["video", "metas", "t5_xxl", "hdmap_bbox"],
            ),
        ),
        checkpoint=dict(
            # resume from bad 720 model
            load_path="cosmos_transfer2/vid2vid_2B_control_av/vid2vid_2B_control_720p_control_layer14_av_full_65k/checkpoints/iter_000004000/",
        ),
    ),
    flags={"allow_objects": True},
)


vid2vid_2B_control_true_720p_control_layer14_av_full_65k_low_sigma = LazyDict(
    dict(
        defaults=[
            "vid2vid_2B_control_true_720p_control_layer14_av_full_65k",
            "_self_",
        ],
        job=dict(
            group="vid2vid_2B_control",
            name="vid2vid_2B_control_true_720p_control_layer14_av_full_65k_low_sigma",
        ),
        checkpoint=dict(
            # resume from true 720p model!
            load_path="cosmos_transfer2/vid2vid_2B_control_av/vid2vid_2B_control_true_720p_control_layer14_av_full_65k/checkpoints/iter_000029000/",
        ),
        model=dict(
            config=dict(
                high_sigma_strategy=str(HighSigmaStrategy.UNIFORM80_2000),
                high_sigma_ratio=0.0,
                sde=dict(
                    p_mean=0.0,
                    p_std=1.0,
                    sigma_max=80,
                    sigma_min=0.0002,
                ),
            ),
        ),
    ),
)

"""
torchrun --nproc_per_node=8 --master_port=12340 -m scripts.train --config=cosmos_predict2/_src/transfer2/configs/vid2vid_transfer/config.py -- experiment=vid2vid_2B_control_480p_control_layer14_av_full_65k_reset_rescheduler
"""


def build_debug_runs(job):
    """
    Will take a config dict and create another config with "_WO_RESUME" suffix
    """
    debug = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_DEBUG" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=_TRAINER_DEBUG_CONFIG,
        checkpoint=_CKPT_DEBUG_CONFIG,
    )

    wo_resume = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_WO_RESUME" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=_TRAINER_DEBUG_CONFIG,
        checkpoint=_CKPT_DEBUG_CONFIG,
        model=dict(
            config=dict(
                base_load_from=None,
            ),
        ),
    )

    mock_wo_resume = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_MOCK_WO_RESUME" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=_TRAINER_DEBUG_CONFIG,
        checkpoint=_CKPT_DEBUG_CONFIG,
        model=dict(
            config=dict(
                base_load_from=None,
            ),
        ),
    )

    return [debug, wo_resume, mock_wo_resume]


cs = ConfigStore.instance()
for _item, _item_debug, _item_wo_resume, _item_mock_wo_resume in [
    [
        vid2vid_2B_control_480p_control_layer14_av,
        *build_debug_runs(vid2vid_2B_control_480p_control_layer14_av),
    ],
    [
        vid2vid_2B_control_480p_control_layer14_av_full_65k,
        *build_debug_runs(vid2vid_2B_control_480p_control_layer14_av_full_65k),
    ],
    [
        vid2vid_2B_control_480p_control_layer14_av_full_65k_online_testing,
        *build_debug_runs(vid2vid_2B_control_480p_control_layer14_av_full_65k_online_testing),
    ],
    [
        vid2vid_2B_control_480p_control_layer14_av_full_65k_reset_rescheduler,
        *build_debug_runs(vid2vid_2B_control_480p_control_layer14_av_full_65k_reset_rescheduler),
    ],
    [
        vid2vid_2B_control_true_720p_control_layer14_av_full_65k,
        *build_debug_runs(vid2vid_2B_control_true_720p_control_layer14_av_full_65k),
    ],
    [
        vid2vid_2B_control_true_720p_control_layer14_av_full_65k_low_sigma,
        *build_debug_runs(vid2vid_2B_control_true_720p_control_layer14_av_full_65k_low_sigma),
    ],
]:
    cs.store(group="experiment", package="_global_", name=f"{_item['job']['name']}", node=_item)  # register the config
    if _item_debug is not None:
        cs.store(
            group="experiment",
            package="_global_",
            name=f"{_item['job']['name']}_debug",
            node=_item_debug,
        )
    if _item_wo_resume is not None:
        cs.store(
            group="experiment",
            package="_global_",
            name=f"{_item['job']['name']}_wo_resume",
            node=_item_wo_resume,
        )
    if _item_mock_wo_resume is not None:
        cs.store(
            group="experiment",
            package="_global_",
            name=f"{_item['job']['name']}_mock_wo_resume",
            node=_item_mock_wo_resume,
        )
