# SPDX-FileCopyrightText: Copyright (c) 2026 Afloat16
# SPDX-License-Identifier: Apache-2.0

"""Ray directions must be independent of the choice of world origin."""

import numpy as np
import pytest
import torch

from cosmos_predict2._src.imaginaire.modules.camera import Camera


@pytest.mark.level(0)
@pytest.mark.gpus(0)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("quarter_turn", [False, True])
@pytest.mark.parametrize("numpy_input", [False, True])
def test_camera_ray_direction_is_invariant_to_large_world_translation(dtype, quarter_turn, numpy_input):
    rotation = torch.tensor(
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]] if quarter_turn else [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=dtype,
    )
    pose = torch.cat([rotation, torch.zeros((3, 1), dtype=dtype)], dim=-1)
    shifted_pose = pose.clone()
    shifted_pose[:, 3] = 100_000_000
    intrinsics = torch.tensor([[2, 0, 0], [0, 4, 0], [0, 0, 1]], dtype=dtype)
    inputs = [pose, shifted_pose, intrinsics]
    if numpy_input:
        inputs = [value.numpy() for value in inputs]
    unshifted = Camera.get_camera_rays(inputs[0], inputs[2], (2, 3))
    shifted = Camera.get_camera_rays(inputs[1], inputs[2], (2, 3))
    unshifted = np.asarray(unshifted)
    shifted = np.asarray(shifted)

    np.testing.assert_allclose(np.linalg.norm(shifted, axis=-1), 1, atol=2e-6, rtol=0)
    np.testing.assert_allclose(shifted, unshifted, atol=2e-6, rtol=0)

    # An independently constructed calibrated pixel-center direction.
    direction_camera = np.array([0.5 / 2, 0.5 / 4, 1])
    direction_world = rotation.numpy().T @ direction_camera
    direction_world /= np.linalg.norm(direction_world)
    np.testing.assert_allclose(shifted[0], direction_world, atol=2e-6, rtol=0)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_camera_rays_continue_to_reject_invalid_pose_rotation():
    pose = torch.tensor([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32)
    with pytest.raises(AssertionError, match="Rotation component"):
        Camera.get_camera_rays(pose, torch.eye(3), (1, 1))
