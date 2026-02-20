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

import functools

from cosmos_predict2._src.imaginaire.utils.checkpoint_db import (
    CheckpointConfig,
    CheckpointDirS3,
    CheckpointFileHf,
    register_checkpoint,
)


@functools.cache
def register_checkpoints():
    from cosmos_oss.checkpoints_predict2 import register_checkpoints as _register_checkpoints

    _register_checkpoints()

    register_checkpoint(
        CheckpointConfig(
            uuid="61f5694b-0ad5-4ecd-8ad7-c8545627d125",
            name="nvidia/Cosmos-Transfer2.5-2B/general/edge",
            experiment="edge_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv3p1_20250714_64N_rectified_flow_refimdrop0pt5",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_transfer2/vid2vid_2B_control/edge_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv3p1_20250714_64N_rectified_flow_refimdrop0pt5/checkpoints/iter_000032000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Transfer2.5-2B",
                revision="b67b64abda3801a9aceddbff2bdb86126c06db74",
                filename="general/edge/61f5694b-0ad5-4ecd-8ad7-c8545627d125_ema_bf16.pt",
            ),
        )
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="626e6618-bfcd-4d9a-a077-1409e2ce353f",
            name="nvidia/Cosmos-Transfer2.5-2B/general/depth",
            experiment="depth_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv4p1_20250823_64N_rectified_flow_refimdrop0pt5",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_transfer2/vid2vid_2B_control/depth_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv4p1_20250823_64N_rectified_flow_refimdrop0pt5/checkpoints/iter_000044000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Transfer2.5-2B",
                revision="dea7737ca29dd8d9086413c6dc5724b8250a0bb4",
                filename="general/depth/626e6618-bfcd-4d9a-a077-1409e2ce353f_ema_bf16.pt",
            ),
        )
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="ba2f44f2-c726-4fe7-949f-597069d9b91c",
            name="nvidia/Cosmos-Transfer2.5-2B/general/blur",
            experiment="vis_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv3p1_20250714_64N_rectified_flow_refimdrop0pt5_filterb3g5m2",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_transfer2/vid2vid_2B_control/vis_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv3p1_20250714_64N_rectified_flow_refimdrop0pt5_filterb3g5m2/checkpoints/iter_000036000/",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Transfer2.5-2B",
                revision="eb5325b77d358944da58a690157dd2b8071bbf85",
                filename="general/blur/ba2f44f2-c726-4fe7-949f-597069d9b91c_ema_bf16.pt",
            ),
        )
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="5136ef49-6d8d-42e8-8abf-7dac722a304a",
            name="nvidia/Cosmos-Transfer2.5-2B/general/seg",
            experiment="seg_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv4p2_20250823_64N_rectified_flow_refimdrop0pt5",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_transfer2/vid2vid_2B_control/seg_720p_t24or1_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_uniform_hqv4p2_20250823_64N_rectified_flow_refimdrop0pt5/checkpoints/iter_000043000/",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Transfer2.5-2B",
                revision="23057a4167b89de89a4a397fdbf3887994d115eb",
                filename="general/seg/5136ef49-6d8d-42e8-8abf-7dac722a304a_ema_bf16.pt",
            ),
        )
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="ecd0ba00-d598-4f94-aa09-e8627899c431",
            name="nvidia/Cosmos-Transfer2.5-2B/general/edge",
            experiment="edge_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow_mock_data",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_transfer2/vid2vid_2B_control/edge_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow/checkpoints/iter_000029000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Transfer2.5-2B",
                revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
                filename="general/edge/ecd0ba00-d598-4f94-aa09-e8627899c431_ema_bf16.pt",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab",
            name="nvidia/Cosmos-Transfer2.5-2B/general/seg",
            experiment="seg_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p2_20250823_64N_rectified_flow",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_transfer2/vid2vid_2B_control/seg_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p2_20250823_64N_rectified_flow/checkpoints/iter_000031000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Transfer2.5-2B",
                revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
                filename="general/seg/fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab_ema_bf16.pt",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1",
            name="nvidia/Cosmos-Transfer2.5-2B/general/blur",
            experiment="vis_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_transfer2/vid2vid_2B_control/vis_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow/checkpoints/iter_000043000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Transfer2.5-2B",
                revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
                filename="general/blur/20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1_ema_bf16.pt",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="0f214f66-ae98-43cf-ab25-d65d09a7e68f",
            name="nvidia/Cosmos-Transfer2.5-2B/general/depth",
            experiment="depth_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p1_20250823_64N_rectified_flow",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_transfer2/vid2vid_2B_control/depth_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p1_20250823_64N_rectified_flow/checkpoints/iter_000028000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Transfer2.5-2B",
                revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
                filename="general/depth/0f214f66-ae98-43cf-ab25-d65d09a7e68f_ema_bf16.pt",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="4ecc66e9-df19-4aed-9802-0d11e057287a",
            name="nvidia/Cosmos-Transfer2.5-2B/auto/multiview",
            experiment="buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_fromfinetuned12knofpsuniform_mads720pmulticaps29frames_world_scenario_nofps_uniform",
            metadata={
                "resolution": "720p",
                "fps": 10,
                "views": 7,
                "frames": 29,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_transfer2_multiview/cosmos2_mv/buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_fromfinetuned12knofpsuniform_mads720pmulticaps29frames_world_scenario_nofps_uniform-0/checkpoints/iter_000006500/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Transfer2.5-2B",
                revision="00c591edab119e8a6ca06e6e091351a04ce0ecc9",
                filename="auto/multiview/4ecc66e9-df19-4aed-9802-0d11e057287a_ema_bf16.pt",
            ),
        )
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="b5ab002d-a120-4fbf-a7f9-04af8615710b",
            name="nvidia/Cosmos-Transfer2.5-2B/auto/multiview",
            experiment="buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase5knofps_mads720pmulticaps29frames_world_scenario_resumefrom21k",
            metadata={
                "resolution": "720p",
                "fps": 16,
                "views": 7,
                "frames": 29,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_transfer2_multiview/cosmos2_mv/buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase5knofps_mads720pmulticaps29frames_world_scenario_resumefrom21k-0/checkpoints/iter_000010000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Transfer2.5-2B",
                revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
                filename="auto/multiview/b5ab002d-a120-4fbf-a7f9-04af8615710b_ema_bf16.pt",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="0e8177cc-0db5-4cfd-a8a4-b820c772f4fc",
            name="nvidia/Cosmos-Transfer2.5-2B/robot/multiview",
            experiment="multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_multicam_syncam",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_multicam_syncam/checkpoints/iter_000002000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Experimental",
                revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
                filename="0e8177cc-0db5-4cfd-a8a4-b820c772f4fc/model_ema_bf16.pt",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="7f6b99b7-7fac-4e74-8dbe-a394cb56ef99",
            name="nvidia/Cosmos-Transfer2.5-2B/robot/multiview-agibot",
            experiment="multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_agibot",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_agibot/checkpoints/iter_000003000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Experimental",
                revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
                filename="7f6b99b7-7fac-4e74-8dbe-a394cb56ef99/model_ema_bf16.pt",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="41f07f13-f2e4-4e34-ba4c-86f595acbc20",
            name="nvidia/Cosmos-Transfer2.5-2B/distilled/edge",
            experiment="dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_edge",
            metadata={
                "resolution": "720p",
                "fps": 16,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_interactive_fastgen/cosmos_interactive/cosmos_fastgen_dmd2_trigflow_distill_cosmos_transfer2p5_2B_bidirectional_edge_bugfix_v2/checkpoints/iter_000010000",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Transfer2.5-2B",
                revision="bbaeedb2b57cc8b14a44653099e2551adb69dcc7",
                filename="distilled/general/edge/41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt",
            ),
        ),
    )

    # Plenoptic Multiview Camera Model
    register_checkpoint(
        CheckpointConfig(
            uuid="a8794d70-842c-44a5-95bb-9010d5ace7be",
            name="nvidia/Cosmos-Experimental/robot/multiview-many-camera",
            experiment="multicamera_ar_video2video_rectified_flow_2b_res_480_fps16_s3_multicam_syncam_in4out1",
            metadata={
                "resolution": "480p",
                "fps": 16,
                "num_input_views": 4,
                "num_output_views": 1,
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/multicamera_ar_video2video_rectified_flow_2b_res_480_fps16_s3_multicam_syncam_in4out1_partialft/checkpoints/iter_000007600",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Experimental",
                revision="main",
                filename="a8794d70-842c-44a5-95bb-9010d5ace7be/model_ema_bf16.pt",
            ),
        )
    )

    # Transfer2.5 Agibot Control-Conditioned Multiview Models
    # Depth Control
    register_checkpoint(
        CheckpointConfig(
            uuid="32514ba1-6d05-4ce5-997d-a3b5bf894cab",
            name="nvidia/Cosmos-Transfer2.5-2B/robot/multiview-agibot-depth",
            experiment="transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_depth",
            metadata={
                "resolution": "720p",
                "fps": 16,
                "views": 3,
                "control_type": "depth",
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_transfer2_multiview/cosmos2p5_mv/transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_depth-0/checkpoints/iter_000038000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Experimental",
                revision="main",
                filename="32514ba1-6d05-4ce5-997d-a3b5bf894cab/model_ema_bf16.pt",
            ),
        ),
    )

    # Edge Control
    register_checkpoint(
        CheckpointConfig(
            uuid="fffbd388-89c9-4604-ad4f-6c6b36272c48",
            name="nvidia/Cosmos-Transfer2.5-2B/robot/multiview-agibot-edge",
            experiment="transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_edge",
            metadata={
                "resolution": "720p",
                "fps": 16,
                "views": 3,
                "control_type": "edge",
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_transfer2_multiview/cosmos2p5_mv/transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_edge-0/checkpoints/iter_000039000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Experimental",
                revision="main",
                filename="fffbd388-89c9-4604-ad4f-6c6b36272c48/model_ema_bf16.pt",
            ),
        ),
    )

    # Vis Control
    register_checkpoint(
        CheckpointConfig(
            uuid="2eca9f80-bf8f-4257-b05f-278065d21500",
            name="nvidia/Cosmos-Transfer2.5-2B/robot/multiview-agibot-vis",
            experiment="transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_vis",
            metadata={
                "resolution": "720p",
                "fps": 16,
                "views": 3,
                "control_type": "vis",
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_transfer2_multiview/cosmos2p5_mv/transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_vis-0/checkpoints/iter_000029000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Experimental",
                revision="main",
                filename="2eca9f80-bf8f-4257-b05f-278065d21500/model_ema_bf16.pt",
            ),
        ),
    )

    # Seg Control
    register_checkpoint(
        CheckpointConfig(
            uuid="c5a9a58b-7f3e-4b45-9e5d-8f7b3d4e5a6c",
            name="nvidia/Cosmos-Transfer2.5-2B/robot/multiview-agibot-seg",
            experiment="transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_seg",
            metadata={
                "resolution": "720p",
                "fps": 16,
                "views": 3,
                "control_type": "seg",
            },
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_transfer2_multiview/cosmos2p5_mv/transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_seg-0/checkpoints/iter_000026000/model",
            ),
            hf=CheckpointFileHf(
                repository="nvidia/Cosmos-Experimental",
                revision="main",
                filename="c5a9a58b-7f3e-4b45-9e5d-8f7b3d4e5a6c/model_ema_bf16.pt",
            ),
        ),
    )
