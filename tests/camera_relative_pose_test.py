# SPDX-FileCopyrightText: Copyright (c) 2026 Afloat16
# SPDX-License-Identifier: Apache-2.0

"""Physical consistency of batched relative camera coordinate systems."""

import numpy as np
import pytest
import torch

from cosmos_predict2._src.imaginaire.modules.camera import Camera


def _batched_world_to_camera(batch_shape, views):
    rotations = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        ],
        dtype=np.float32,
    )
    poses = np.empty((*batch_shape, views, 3, 4), dtype=np.float32)
    for batch_number, batch_index in enumerate(np.ndindex(batch_shape)):
        for view in range(views):
            poses[(*batch_index, view, slice(None), slice(0, 3))] = rotations[(batch_number + view) % 4]
            poses[(*batch_index, view, slice(None), 3)] = [batch_number + 0.3, view - 0.4, batch_number - view]
    return poses


@pytest.mark.level(0)
@pytest.mark.gpus(0)
@pytest.mark.parametrize("batch_shape,views", [((1,), 3), ((2,), 3), ((2,), 2), ((2,), 1), ((2, 2), 3)])
@pytest.mark.parametrize("numpy_input", [False, True])
def test_relative_poses_preserve_each_batch_reference(batch_shape, views, numpy_input):
    poses = _batched_world_to_camera(batch_shape, views)
    value = poses if numpy_input else torch.from_numpy(poses)
    relative = Camera.get_relative_poses_wrt_frame0(value)
    result = relative if numpy_input else relative.numpy()

    assert result.shape == poses.shape
    assert isinstance(relative, np.ndarray if numpy_input else torch.Tensor)
    reference_identity = np.zeros((*batch_shape, 3, 4), dtype=np.float32)
    reference_identity[..., :3] = np.eye(3, dtype=np.float32)
    np.testing.assert_allclose(result[..., 0, :, :], reference_identity, atol=2e-6, rtol=0)

    # The same world landmark must have the same coordinates in every original
    # camera, whether mapped directly or through that sample's first camera.
    world_landmarks = np.array([[0.2, -0.7, 2.0, 1.0], [-1.0, 0.3, 4.0, 1.0]], dtype=np.float32)
    original_coordinates = np.einsum("...ij,nj->...ni", poses, world_landmarks)
    first_coordinates = original_coordinates[..., 0, :, :]
    first_homogeneous = np.concatenate([first_coordinates, np.ones_like(first_coordinates[..., :1])], axis=-1)
    relative_coordinates = np.einsum("...vij,...nj->...vni", result, first_homogeneous)
    np.testing.assert_allclose(relative_coordinates, original_coordinates, atol=3e-6, rtol=0)
