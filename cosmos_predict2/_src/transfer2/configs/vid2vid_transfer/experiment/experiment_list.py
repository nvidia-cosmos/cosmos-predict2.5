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
List of experiments for vid2vid_transfer. Steps to run new experimients:

- Add a new experiment entry here in the dict of the appropriate category
- Submit the job with _submit.py
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Experiment:
    registered_exp_name: str
    job_name_for_ckpt: str
    job_group: str
    nnode: int
    command_args: List[str]


archi_ablation_experiments_720p = {
    "edge_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow": Experiment(  # edge, hqv3p1 + spaced_4
        registered_exp_name="vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow",
        job_name_for_ckpt="edge_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow",
        job_group="vid2vid_2B_control",
        nnode=2,
        command_args=[
            "model.config.hint_keys=edge",
            "dataloader_train.dataset.control_input_type=edge",
            "data_train=mock_video",
            "checkpoint.load_path=cosmos_transfer2/vid2vid_2B_control/edge_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N/checkpoints/iter_000060000",
        ],
    ),
    # low sigma 0.05 + nonuniform loss weight, hqv3p1_20250714 data, 2x lr
    "vis_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow": Experiment(  # vis, hqv3p1 + spaced_4
        registered_exp_name="vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow",
        job_name_for_ckpt="vis_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow",
        job_group="vid2vid_2B_control",
        nnode=64,
        command_args=[
            "model.config.hint_keys=vis",
            "dataloader_train.dataset.control_input_type=vis",
            "data_train=mock_video",
            "checkpoint.load_path=cosmos_transfer2/vid2vid_2B_control/vis_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N/checkpoints/iter_000022000",
        ],
    ),
    # with v4p1_20250823 data (+ alpamayo, highquality, physics-cosmos-db, robomind), 2x lr, low sigma 0.05 + nonuniform loss weight
    "depth_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p1_20250823_64N_rectified_flow": Experiment(  # depth, hqv1p1 + spaced_4
        registered_exp_name="vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow",
        job_name_for_ckpt="depth_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p1_20250823_64N_rectified_flow",
        job_group="vid2vid_2B_control",
        nnode=64,
        command_args=[
            "model.config.hint_keys=depth",
            "dataloader_train.dataset.control_input_type=depth",
            "checkpoint.load_path=cosmos_transfer2/vid2vid_2B_control/depth_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p1_20250823_64N/checkpoints/iter_000050000",
            "data_train=mock_video",
        ],
    ),
    # with v4p2_20250823 data (+ alpamayo ), 2x lr, low sigma 0.05 + nonuniform loss weight
    "seg_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p2_20250823_64N_rectified_flow": Experiment(
        registered_exp_name="vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow",
        job_name_for_ckpt="seg_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p2_20250823_64N_rectified_flow",
        job_group="vid2vid_2B_control",
        nnode=64,
        command_args=[
            "model.config.hint_keys=seg",
            "dataloader_train.dataset.control_input_type=segcolor",
            "checkpoint.load_path=cosmos_transfer2/vid2vid_2B_control/seg_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p2_20250823_64N/checkpoints/iter_000030000",
            "data_train=mock_video",
            # "optimizer.lr=0.0000863",  # 2**(-13.5)
            "optimizer.lr=0.0000432",  # 2**(-14.5)
        ],
    ),
    "vis_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv3p1_20250714_64N_rectified_flow_refimdrop0pt5_filterb3g5m2": Experiment(  # vis, hqv3p1
        registered_exp_name="vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow_with_image_context_with_image_data",
        job_name_for_ckpt="vis_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv3p1_20250714_64N_rectified_flow_refimdrop0pt5_filterb3g5m2",
        job_group="vid2vid_2B_control",
        nnode=64,
        command_args=[
            "model.config.hint_keys=vis",
            "model.config.train_time_weight=uniform",
            "model.config.conditioner.reference_image_context.dropout_rate=0.5",
            "dataloader_train.dataloaders.video_data.dataloader.dataset.control_input_type=vis",
            "dataloader_train.dataloaders.video_data_1.dataloader.dataset.control_input_type=vis",
            "data_train=image_cosmos_pretrain_20241108_two_video_cosmos_transfer2_high_quality_v3p1_20250714_s3",
            "checkpoint.load_path=cosmos_transfer2/vid2vid_2B_control/vis_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow_refimdrop0pt3/checkpoints/iter_000021000/",
        ],
    ),
    # with image context, image data, cr1pt1 embedding, rectified flow
    "edge_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv3p1_20250714_64N_rectified_flow_refimdrop0pt5": Experiment(  # edge, hqv3p1
        registered_exp_name="vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow_with_image_context_with_image_data",
        job_name_for_ckpt="edge_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv3p1_20250714_64N_rectified_flow_refimdrop0pt5",
        job_group="vid2vid_2B_control",
        nnode=64,
        command_args=[
            "model.config.hint_keys=edge",
            "model.config.train_time_weight=uniform",
            "model.config.conditioner.reference_image_context.dropout_rate=0.5",
            "dataloader_train.dataloaders.video_data.dataloader.dataset.control_input_type=edge",
            "dataloader_train.dataloaders.video_data_1.dataloader.dataset.control_input_type=edge",
            "data_train=image_cosmos_pretrain_20241108_two_video_cosmos_transfer2_high_quality_v3p1_20250714_s3",
            "checkpoint.load_path=cosmos_transfer2/vid2vid_2B_control/edge_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow_refimdrop0pt3/checkpoints/iter_000022000/",
        ],
    ),
    "depth_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv4p1_20250823_64N_rectified_flow_refimdrop0pt5": Experiment(  # depth, hqv4p1
        registered_exp_name="vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow_with_image_context_with_image_data",
        job_name_for_ckpt="depth_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv4p1_20250823_64N_rectified_flow_refimdrop0pt5",
        job_group="vid2vid_2B_control",
        nnode=64,
        command_args=[
            "model.config.hint_keys=depth",
            "model.config.train_time_weight=uniform",
            "model.config.conditioner.reference_image_context.dropout_rate=0.5",
            "dataloader_train.dataloaders.video_data.dataloader.dataset.control_input_type=depth",
            "dataloader_train.dataloaders.video_data_1.dataloader.dataset.control_input_type=depth",
            "data_train=image_cosmos_pretrain_20241108_two_video_cosmos_transfer2_high_quality_v4p1_20250823_s3",
            "checkpoint.load_path=cosmos_transfer2/vid2vid_2B_control/depth_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p1_20250823_64N_rectified_flow_refimdrop0pt3/checkpoints/iter_000047000/",
        ],
    ),
    "seg_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv4p2_20250823_64N_rectified_flow_refimdrop0pt5": Experiment(  # seg, hqv4p2
        registered_exp_name="vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow_with_image_context_with_image_data",
        job_name_for_ckpt="seg_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv4p2_20250823_64N_rectified_flow_refimdrop0pt5",
        job_group="vid2vid_2B_control",
        nnode=64,
        command_args=[
            "model.config.hint_keys=seg",
            "model.config.train_time_weight=uniform",
            "model.config.conditioner.reference_image_context.dropout_rate=0.5",
            "dataloader_train.dataloaders.video_data.dataloader.dataset.control_input_type=segcolor",
            "dataloader_train.dataloaders.video_data_1.dataloader.dataset.control_input_type=segcolor",
            "data_train=image_cosmos_pretrain_20241108_two_video_cosmos_transfer2_high_quality_v4p2_20250823_s3",
            "checkpoint.load_path=cosmos_transfer2/vid2vid_2B_control/seg_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p2_20250823_64N_rectified_flow_refimdrop0pt3/checkpoints/iter_000036000/",
        ],
    ),
    "multibranch_720p_t24_spaced_layer4_cr1pt1_rectified_flow_inference": Experiment(  # spaced_4, multi-branch, rectified flow
        registered_exp_name="vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow",
        job_name_for_ckpt="multibranch_720p_t24_spaced_layer4_cr1pt1_rectified_flow_inference",
        job_group="vid2vid_2B_control",
        nnode=64,
        command_args=[
            "model.config.hint_keys=edge_vis_depth_seg",
            "dataloader_train.dataset.control_input_type=edge_vis_depth_segcolor",
            "model.config.net.num_control_branches=4",
            "model.config.net.use_after_proj_for_multi_branch=False",
        ],
    ),
}

release_experiments = {
    "edge_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow_mock_data": Experiment(
        registered_exp_name="vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow",
        job_name_for_ckpt="edge_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow",
        job_group="vid2vid_2B_control",
        nnode=2,
        command_args=[
            "model.config.hint_keys=edge",
            "dataloader_train.dataset.control_input_type=edge",
            "data_train=mock_video",
            "checkpoint.load_path=cosmos_transfer2/vid2vid_2B_control/edge_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N/checkpoints/iter_000060000",
        ],
    ),
}


EXPERIMENTS = {}
EXPERIMENTS_LIST = [
    archi_ablation_experiments_720p,
    release_experiments,
]
for experiments in EXPERIMENTS_LIST:
    for exp_name, _ in experiments.items():
        assert exp_name not in EXPERIMENTS, f"Experiment {exp_name} already exists"
    EXPERIMENTS.update(experiments)
