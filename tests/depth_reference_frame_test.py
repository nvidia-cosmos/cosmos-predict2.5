# SPDX-FileCopyrightText: Copyright (c) 2026 Afloat16
# SPDX-License-Identifier: Apache-2.0

"""A static landmark projects to one reference-frame point from every view."""

import numpy as np
import pytest
import torch

from cosmos_predict2._src.imaginaire.datasets.webdataset.augmentors.geometry.pointcloud import DepthToPointcloud


@pytest.mark.level(0)
@pytest.mark.gpus(0)
@pytest.mark.parametrize("landmark", [[0.2, -0.3, 2], [7, -2, 5], [-3, 4, 1]])
def test_video_depth_backprojection_expresses_points_in_first_camera(landmark):
    rotations = np.array(
        [
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        ],
        dtype=np.float32,
    )
    depths = np.array([2, 3, 4], dtype=np.float32)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    poses[:, :3, :3] = rotations
    # Every camera sees the same static world landmark on its optical axis
    # at a positive, known depth.
    for frame in range(3):
        poses[frame, :3, 3] = [0, 0, depths[frame]] - rotations[frame] @ np.asarray(landmark)

    data = {
        "depth": torch.from_numpy(depths[:, None, None]),
        "intrinsics": torch.eye(3).repeat(3, 1, 1),
        "world_to_cam": torch.from_numpy(poses),
    }
    augmentor = DepthToPointcloud(
        input_keys=["depth", "intrinsics", "world_to_cam"],
        output_keys=["points"],
        args={"to_world_coords": True},
    )
    result = augmentor(data)["points"][:, :, 0, 0].T.numpy()
    expected = np.tile([0, 0, depths[0]], (3, 1))
    np.testing.assert_allclose(result, expected, atol=3e-6, rtol=0)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_video_depth_backprojection_keeps_camera_coordinates_when_disabled():
    augmentor = DepthToPointcloud(
        input_keys=["depth", "intrinsics", "world_to_cam"],
        output_keys=["points"],
        args={"to_world_coords": False},
    )
    data = {"depth": torch.tensor([[[2.0]], [[3.0]]]), "intrinsics": torch.eye(3).repeat(2, 1, 1)}
    result = augmentor(data)["points"][:, :, 0, 0].T
    torch.testing.assert_close(result, torch.tensor([[0, 0, 2.0], [0, 0, 3.0]]))
