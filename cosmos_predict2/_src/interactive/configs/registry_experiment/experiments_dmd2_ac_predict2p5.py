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

from copy import deepcopy

from hydra.core.config_store import ConfigStore  # type: ignore[import]

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import (
    get_checkpoint_path,
)
from cosmos_predict2._src.interactive.configs.registry_defaults.teacher_model_paths import (
    ACTION_CONDITIONED_TEACHER_CKPT_2B_256X320,
)
from cosmos_predict2._src.interactive.configs.registry_experiment.experiments_dmd2_predict2p5 import (
    make_experiment,
)

# Bridge dataset - 13 frame prediction at 256x320 resolution
dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_256x320 = make_experiment(
    name="dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_256x320",
    data_train="bridge_13frame_480_640_train",
    net="cosmos_v1_2B_action_chunk_conditioned_student",
    net_teacher="cosmos_v1_2B_action_chunk_conditioned_teacher",
    net_fake_score="cosmos_v1_2B_action_chunk_conditioned_fake_score",
    conditioner="action_conditioned_video_conditioner",
    resolution="256",
    cp_size=1,
    overrides=dict(
        model=dict(
            config=dict(
                state_t=4,
                use_clean_cond_timesteps=False,
                conditional_frames_probs={0: 0.0, 1: 1.0, 2: 0.0},
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                net=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                    temporal_compression_ratio=4,
                    crossattn_emb_channels=1024,
                ),
                net_fake_score=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                    temporal_compression_ratio=4,
                    crossattn_emb_channels=1024,
                ),
                net_teacher=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                    temporal_compression_ratio=4,
                    crossattn_emb_channels=1024,
                ),
                teacher_load_from=ACTION_CONDITIONED_TEACHER_CKPT_2B_256X320,
                teacher_guidance=0,
                student_update_freq=10,
            ),
        ),
        dataloader_train=dict(
            batch_size=40,
            sampler=dict(
                dataset=dict(
                    video_size=[256, 320],
                    num_action_per_chunk=12,
                    fps_downsample_ratio=1,
                    gripper_rescale_factor=1,
                ),
            ),
            dataset=dict(
                video_size=[256, 320],
                num_action_per_chunk=12,
                fps_downsample_ratio=1,
                gripper_rescale_factor=1,
            ),
        ),
    ),
)
# Remove the nested dataloaders structure inherited from base make_experiment
del dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_256x320["dataloader_train"][
    "dataloaders"
]

# Bridge dataset - 13 frame prediction at 480x640 resolution (if you have a 480p teacher)
dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_480x640 = make_experiment(
    name="dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_480x640",
    data_train="bridge_13frame_480_640_train",
    net="cosmos_v1_2B_action_chunk_conditioned_student",
    net_teacher="cosmos_v1_2B_action_chunk_conditioned_teacher",
    net_fake_score="cosmos_v1_2B_action_chunk_conditioned_fake_score",
    conditioner="action_conditioned_video_conditioner",
    resolution="480",
    cp_size=1,
    # NOTE: Update this to your 480p teacher checkpoint if you have one
    overrides=dict(
        model=dict(
            config=dict(
                state_t=4,
                use_clean_cond_timesteps=False,
                conditional_frames_probs={0: 0.0, 1: 1.0, 2: 0.0},
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                net=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                    temporal_compression_ratio=4,
                    crossattn_emb_channels=1024,
                ),
                net_fake_score=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                    temporal_compression_ratio=4,
                    crossattn_emb_channels=1024,
                ),
                net_teacher=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                    temporal_compression_ratio=4,
                    crossattn_emb_channels=1024,
                ),
                teacher_load_from=ACTION_CONDITIONED_TEACHER_CKPT_2B_256X320,
                teacher_guidance=0,
                student_update_freq=10,
            ),
        ),
        dataloader_train=dict(
            batch_size=10,
            sampler=dict(
                dataset=dict(
                    video_size=[480, 640],
                    num_action_per_chunk=12,
                    fps_downsample_ratio=1,
                    gripper_rescale_factor=1,
                ),
            ),
            dataset=dict(
                video_size=[480, 640],
                num_action_per_chunk=12,
                fps_downsample_ratio=1,
                gripper_rescale_factor=1,
            ),
        ),
    ),
)
# Remove the nested dataloaders structure inherited from base make_experiment
del dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_480x640["dataloader_train"][
    "dataloaders"
]


def _build_no_s3_run(job: LazyDict) -> LazyDict:
    """
    Build a no S3 run of the given job.
    """
    no_s3_job = deepcopy(job)

    teacher_load_path = no_s3_job["model"]["config"]["teacher_load_from"]["load_path"]
    no_s3_job["model"]["config"]["teacher_load_from"] = {
        "load_path": get_checkpoint_path(teacher_load_path),
        "credentials": None,
    }

    no_s3_job["job"]["name"] = f"{job['job']['name']}_no_s3" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"
    no_s3_job["upload_reproducible_setup"] = False

    no_s3_job["checkpoint"]["save_to_object_store"]["enabled"] = False
    no_s3_job["checkpoint"]["load_from_object_store"]["enabled"] = False

    no_s3_job["trainer"]["straggler_detection"] = {"enabled": False}
    no_s3_job["trainer"]["callbacks"] = {
        "heart_beat": {"save_s3": False},
        "iter_speed": {"save_s3": False},
        "device_monitor": {"save_s3": False},
        "every_n_sample_reg": {"save_s3": False, "every_n": 500},
        "every_n_sample_ema": {"save_s3": False, "every_n": 500},
        "wandb": {"save_s3": False},
        "wandb_10x": {"save_s3": False},
        "dataloader_speed": {"save_s3": False},
    }

    return no_s3_job


cs = ConfigStore.instance()
"""
2B:
torchrun --nproc_per_node=4 --master_port=12340 -m scripts.train --config=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py -- experiment=dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V

14B: requires fsdp_size=32, cannot be run on single node. Please try submitting a >=4 node job to verify.
"""
for _item in [
    dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_256x320,
    dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_480x640,
]:
    cs.store(
        group="experiment",
        package="_global_",
        name=f"{_item['job']['name']}",
        node=_item,
    )

    cs.store(
        group="experiment",
        package="_global_",
        name=f"{_item['job']['name']}_no_s3",
        node=_build_no_s3_run(_item),
    )
