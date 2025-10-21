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

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import ModalityConfig

# from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform import (
#     VideoCrop,
#     VideoResize,
#     VideoToTensor,
# )
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.base import ComposedModalityTransform
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.concat import ConcatTransform
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.video import (
    VideoCrop,
    VideoResize,
    VideoToTensor,
)


def construct_modality_config_and_transforms(num_frames, embodiment, downscaled_res=False):
    if embodiment == "gr1":
        timestep_interval = 2
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        video_key = "video.ego_view_freq20" if not downscaled_res else "video.ego_view_bg_crop_pad_res256_freq20"
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[video_key],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_arm",
                    "state.right_arm",
                    "state.left_hand",
                    "state.right_hand",
                    "state.waist",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_arm",
                    "action.right_arm",
                    "action.left_hand",
                    "action.right_hand",
                    "action.waist",
                ],
            ),
        }
    elif embodiment == "gr1_video_only":
        timestep_interval = 1
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=["video.ego_view_bg_crop_pad_res256_freq20"],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_arm",
                    "state.right_arm",
                    "state.left_hand",
                    "state.right_hand",
                    "state.waist",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_arm",
                    "action.right_arm",
                    "action.left_hand",
                    "action.right_hand",
                    "action.waist",
                ],
            ),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.coarse_action"]),
        }
    elif embodiment == "agibot":
        timestep_interval = 4
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        video_key = "video.top_head" if not downscaled_res else "video.top_head_pad_res256_freq10"
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[video_key],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_arm_joint_position",
                    "state.right_arm_joint_position",
                    "state.left_effector_position",
                    "state.right_effector_position",
                    "state.head_position",
                    "state.waist_position",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_arm_joint_position",
                    "action.right_arm_joint_position",
                    "action.left_effector_position",
                    "action.right_effector_position",
                    "action.head_position",
                    "action.waist_position",
                    "action.robot_velocity",
                ],
            ),
        }

    video_modality, state_modality, action_modality = config["video"], config["state"], config["action"]
    if embodiment == "gr1" or embodiment == "gr1_video_only":
        width = 832 if not downscaled_res else 256
        height = 480 if not downscaled_res else 256
    elif embodiment == "agibot":
        width = 640 if not downscaled_res else 256
        height = 480 if not downscaled_res else 256

    train_transform = ComposedModalityTransform(
        transforms=[
            VideoToTensor(apply_to=video_modality.modality_keys),
            VideoCrop(apply_to=video_modality.modality_keys, scale=0.95),
            VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
            # VideoColorJitter(apply_to=video_modality.modality_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            StateActionToTensor(apply_to=state_modality.modality_keys),
            StateActionTransform(
                apply_to=state_modality.modality_keys,
                normalization_modes={key: "min_max" for key in state_modality.modality_keys},
            ),
            StateActionToTensor(apply_to=action_modality.modality_keys),
            StateActionTransform(
                apply_to=action_modality.modality_keys,
                normalization_modes={key: "min_max" for key in action_modality.modality_keys},
            ),
            ConcatTransform(
                video_concat_order=video_modality.modality_keys,
                state_concat_order=state_modality.modality_keys,
                action_concat_order=action_modality.modality_keys,
            ),
        ]
    )
    test_transform = ComposedModalityTransform(
        transforms=[
            VideoToTensor(apply_to=video_modality.modality_keys),
            VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
            StateActionToTensor(apply_to=state_modality.modality_keys),
            StateActionTransform(
                apply_to=state_modality.modality_keys,
                normalization_modes={key: "min_max" for key in state_modality.modality_keys},
            ),
            StateActionToTensor(apply_to=action_modality.modality_keys),
            StateActionTransform(
                apply_to=action_modality.modality_keys,
                normalization_modes={key: "min_max" for key in action_modality.modality_keys},
            ),
            ConcatTransform(
                video_concat_order=video_modality.modality_keys,
                state_concat_order=state_modality.modality_keys,
                action_concat_order=action_modality.modality_keys,
            ),
        ]
    )

    return config, train_transform, test_transform
