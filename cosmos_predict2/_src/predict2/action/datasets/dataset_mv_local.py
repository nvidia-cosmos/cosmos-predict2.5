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
Run this command to interactively debug:
PYTHONPATH=. python cosmos_predict2/_src/predict2/action/datasets/dataset_mv_local.py

Adapted from:
https://github.com/bytedance/IRASim/blob/main/dataset/dataset_3D.py
"""

import json
import random
import time

import torch

from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D


class ActionConditionedMultiViewDataset(Dataset_3D):
    def _get_obs(self, label, frame_ids, cam_id, pre_encode):
        if cam_id is None:
            temp_cam_id_0 = random.choice(self.cam_ids[0])
            temp_cam_id_1 = self.cam_ids[1]
        else:
            temp_cam_id_0 = cam_id[0]
            temp_cam_id_1 = cam_id[1]
        frames_0 = self._get_frames(label, frame_ids, cam_id=temp_cam_id_0, pre_encode=pre_encode)
        frames_1 = self._get_frames(label, frame_ids, cam_id=temp_cam_id_1, pre_encode=pre_encode)
        frames = torch.cat([frames_0, frames_1], dim=3)
        return frames, [temp_cam_id_0, temp_cam_id_1]

    def _load_and_process_ann_file(self, ann_file):
        samples = []
        with open(ann_file, "r") as f:
            ann = json.load(f)

        n_frames = len(ann[self._state_key])

        if isinstance(self.fps_downsample_ratio, int):
            fps_downsample_ratio_list = [self.fps_downsample_ratio]
        else:
            fps_downsample_ratio_list = self.fps_downsample_ratio

        for fps_downsample_ratio in fps_downsample_ratio_list:
            for frame_i in range(0, n_frames, self.start_frame_interval):
                sample = dict()
                sample["ann_file"] = ann_file
                sample["frame_ids"] = []
                curr_frame_i = frame_i
                while True:
                    if curr_frame_i > (n_frames - 1):
                        break
                    sample["frame_ids"].append(curr_frame_i)
                    if len(sample["frame_ids"]) == self.sequence_length:
                        break
                    # curr_frame_i += self.fps_downsample_ratio
                    curr_frame_i += fps_downsample_ratio
                # make sure there are sequence_length number of frames
                if len(sample["frame_ids"]) == self.sequence_length:
                    samples.append(sample)
        return samples


if __name__ == "__main__":
    dataset = ActionConditionedMultiViewDataset(
        train_annotation_path="/project/cosmos/user/nvidia-cosmos-raw-data/ur5-data/video-evals-raw-data/datasets/action_dataset/single_chunk/annotation/train",
        val_annotation_path="/project/cosmos/user/nvidia-cosmos-raw-data/ur5-data/video-evals-raw-data/datasets/action_dataset/single_chunk/annotation/val",
        test_annotation_path="/project/cosmos/user/nvidia-cosmos-raw-data/ur5-data/video-evals-raw-data/datasets/action_dataset/single_chunk/annotation/test",
        video_path="/project/cosmos/user/nvidia-cosmos-raw-data/ur5-data/video-evals-raw-data/datasets/action_dataset/single_chunk/",
        fps_downsample_ratio=1,
        num_action_per_chunk=1,
        cam_ids=[["base_0", "base_1"], "wrist"],
        accumulate_action=False,
        video_size=[480, 640],
        val_start_frame_interval=1,
        mode="train",
        load_t5_embeddings=False,
        state_key="ee_pose",
    )

    indices = [0, 13, 200, -1]
    for idx in indices:
        start_time = time.time()
        print(
            (
                f"{idx=} "
                f"{dataset[idx]['video'].sum()=}\n"
                f"{dataset[idx]['video'].shape=}\n"
                # f"{dataset[idx]['video_name']=}\n"
                f"{dataset[idx]['action'].sum()=}\n"
                "---"
            )
        )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
    from IPython import embed

    embed()
