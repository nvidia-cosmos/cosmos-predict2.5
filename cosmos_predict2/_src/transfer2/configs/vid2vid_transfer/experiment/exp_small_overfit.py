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

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict

"""
To be run on an interactive node.
dryrun:
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --dryrun --config=cosmos_predict2/_src/transfer2/configs/vid2vid_transfer/config.py -- experiment=vid2vid_control_vace_overfit_fsdp_video-only

run:
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12343 -m scripts.train --config=cosmos_predict2/_src/transfer2/configs/vid2vid_transfer/config.py -- experiment=vid2vid_control_vace_overfit_fsdp_video-only job.wandb_mode=disabled

"""


OVERFIT_FSDP_VIDEO_ONLY_EXP: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "video_only_cosmos_transfer2_high_quality_v2_202505_s3"},
            {"override /model": "fsdp_control_vace"},
            {"override /net": "transfer2_control2world_net_2B"},
            {"override /conditioner": "video_prediction_control_conditioner"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            {"override /ema": "power"},
            {"override /optimizer": "fusedadamw"},
            {"override /ckpt_type": "dcp"},
            {"override /callbacks": ["basic", "viz_online_sampling", "cluster_speed", "wandb", "long"]},
            "_self_",
        ],
        job=dict(
            group="vid2vid_control_vace_edge_overfit",
            name="vid2vid_control_vace_overfit_fsdp_video-only_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        model=dict(
            config=dict(
                net=dict(
                    num_blocks=8,  # smaller model to load in a single GPU
                    vace_has_mask=False,
                    condition_strategy="spaced",
                ),
                adjust_video_noise=True,
                # sigma_data=1.0,
                resize_online=True,
                state_t=20,
                fsdp_shard_size=8,
                resolution="480",
                base_load_from=dict(
                    load_path="bucket/cosmos_diffusion_v2/vid2vid_control_base/Stage-c_pt_4-Index-10-Size-2B-Res-480-Fps-16-Note-06_02_TnI2V_data_all_hq/checkpoints/iter_000022500",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            save_iter=10,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            # load_path="cosmos_diffusion_v2/vid2vid_control_base/Stage-c_pt_4-Index-10-Size-2B-Res-480-Fps-16-Note-06_02_TnI2V_data_all_hq/checkpoints/iter_000022500",
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            max_iter=5,
            logging_iter=200,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=5_000,
                    is_x0=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
                every_n_sample_ema=dict(
                    every_n=5_000,
                    is_x0=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
            ),
        ),
        dataloader_train=dict(
            dataset=dict(
                num_video_frames=57,
                augmentor_name="video_basic_augmentor_v2_with_control",
                video_decoder_name="video_naive_bytes",
                resolution="480",
                # caption_type="t2w_qwen2p5_7b",
                # embedding_type="umt5_xxl",
                min_fps_thres=3,
                # max_fps_thres=60,
            ),
            num_workers=0,
            prefetch_factor=None,
        ),
        upload_reproducible_setup=True,
    )
)


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="vid2vid_control_vace_overfit_fsdp_video-only",
    node=OVERFIT_FSDP_VIDEO_ONLY_EXP,
)
