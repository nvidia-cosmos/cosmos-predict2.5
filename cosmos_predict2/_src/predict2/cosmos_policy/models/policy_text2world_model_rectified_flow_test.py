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
Tests for CosmosPolicyDiffusionModelRectifiedFlow using closed-form test cases.

These tests instantiate the ACTUAL CosmosPolicyDiffusionModelRectifiedFlow class
and test its compute_loss_rectified_flow method. Heavy dependencies (tokenizer,
large networks) are mocked, but the class logic itself is NOT mocked.

Test cases:
1. Identity (zero velocity): When noise equals clean data, target velocity is zero
2. Linear drift (constant velocity): When noise - clean is constant, velocity is constant
3. Perfect prediction: When network predicts target velocity exactly, loss is zero

Run with:
    pytest -s cosmos_predict2/_src/predict2/cosmos_policy/models/policy_text2world_model_rectified_flow_test.py -v --L0
    pytest -s cosmos_predict2/_src/predict2/cosmos_policy/models/policy_text2world_model_rectified_flow_test.py -v --L1
"""

from __future__ import annotations

import sys

# =============================================================================
# Fix Python path to avoid tokenizers shadowing issue
# =============================================================================
# The cosmos_policy/tokenizers/ directory shadows the 'tokenizers' pip package.
# We need to ensure site-packages is searched first for 'tokenizers'.

# Remove any paths containing cosmos_policy/tokenizers from consideration for tokenizers package
_problematic_paths = [p for p in sys.path if "cosmos_policy" in p]
for _p in _problematic_paths:
    if _p in sys.path:
        sys.path.remove(_p)
        sys.path.append(_p)  # Move to end so site-packages is searched first

# Pre-import tokenizers from site-packages before anything else imports it

from unittest.mock import MagicMock, patch

import pytest
import torch

# =============================================================================
# Mock Heavy Dependencies BEFORE importing the actual class
# =============================================================================

# Mock megatron.core.parallel_state before any imports that need it
mock_parallel_state = MagicMock()
mock_parallel_state.is_initialized.return_value = False
mock_parallel_state.get_data_parallel_world_size.return_value = 1
mock_parallel_state.get_context_parallel_group.return_value = None
# Now we can import the actual class
try:
    from cosmos_predict2._src.predict2.cosmos_policy.conditioner import DataType, Text2WorldCondition
    from cosmos_predict2._src.predict2.cosmos_policy.models.policy_text2world_model import (
        replace_latent_with_action_chunk,
        replace_latent_with_proprio,
    )
    from cosmos_predict2._src.predict2.cosmos_policy.models.policy_text2world_model_rectified_flow import (
        CosmosPolicyDiffusionModelRectifiedFlow,
        CosmosPolicyModelConfigRectifiedFlow,
    )
    from cosmos_predict2._src.predict2.schedulers.rectified_flow import RectifiedFlow
except AssertionError as e:
    if "flash_attn" in str(e):
        pytest.skip(reason="OWNER_TO_CHECK_LOGIC: flash_attn_2 required", allow_module_level=True)
    raise

# =============================================================================
# Lightweight Mock Components (for dependencies only, NOT the class logic)
# =============================================================================


class MockTokenizer:
    """Mock tokenizer - only mocks the VAE, not the model logic."""

    latent_ch: int = 16
    spatial_compression_factor: int = 8
    temporal_compression_factor: int = 4

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Identity encoding for testing."""
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Identity decoding for testing."""
        return x

    def get_latent_num_frames(self, num_frames: int) -> int:
        return max(1, num_frames // self.temporal_compression_factor)


class MockNet(torch.nn.Module):
    """
    Mock neural network that can be configured to return specific velocities.
    This mocks the DiT weights, NOT the model's compute_loss logic.
    """

    def __init__(self):
        super().__init__()
        self.is_context_parallel_enabled = False
        self._target_velocity: torch.Tensor | None = None
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Return configured velocity prediction."""
        if self._target_velocity is not None:
            return self._target_velocity.to(x_B_C_T_H_W.device, x_B_C_T_H_W.dtype)
        return torch.zeros_like(x_B_C_T_H_W)

    def set_target_velocity(self, velocity: torch.Tensor):
        """Configure what velocity this network should predict."""
        self._target_velocity = velocity.clone()

    def enable_context_parallel(self, cp_group):
        pass

    def disable_context_parallel(self):
        pass


class MockConditioner:
    """Mock conditioner - returns simple conditions."""

    def __call__(self, data_batch: dict) -> Text2WorldCondition:
        batch_size = data_batch.get("video", torch.zeros(1)).shape[0]
        device = data_batch.get("video", torch.zeros(1)).device

        return Text2WorldCondition(
            crossattn_emb=torch.randn(batch_size, 512, 4096, device=device),
            data_type=DataType.VIDEO,
            padding_mask=torch.ones(batch_size, 512, device=device),
            fps=torch.tensor([30.0] * batch_size, device=device),
        )


# =============================================================================
# Helper to Create ACTUAL Model with Mocked Dependencies
# =============================================================================


def create_model_with_mocked_deps(device: torch.device) -> CosmosPolicyDiffusionModelRectifiedFlow:
    """
    Create an ACTUAL CosmosPolicyDiffusionModelRectifiedFlow instance,
    but with heavy dependencies (tokenizer, network) replaced by lightweight mocks.

    The model's actual methods (compute_loss_rectified_flow, etc.) are NOT mocked.
    """
    # Bypass the class __init__ to avoid heavy initialization, but still call nn.Module.__init__
    with patch.object(CosmosPolicyDiffusionModelRectifiedFlow, "__init__", lambda self, cfg: None):
        model = object.__new__(CosmosPolicyDiffusionModelRectifiedFlow)

    # Initialize nn.Module base class (required for submodule assignment)
    torch.nn.Module.__init__(model)

    # Set up config (actual config class)
    config = CosmosPolicyModelConfigRectifiedFlow()
    config.state_ch = 16
    config.state_t = 4
    config.mask_loss_for_action_future_state_prediction = False
    config.mask_value_prediction_loss_for_policy_prediction = False
    config.mask_current_state_action_for_value_prediction = False
    config.mask_future_state_for_qvalue_prediction = False
    config.action_loss_multiplier = 1

    model.config = config
    model.precision = torch.float32
    model.tensor_kwargs = {"device": device, "dtype": torch.float32}
    model.tensor_kwargs_fp32 = {"device": device, "dtype": torch.float32}

    # Mock only the DEPENDENCIES, not the model logic
    model.tokenizer = MockTokenizer()
    model.net = MockNet().to(device)
    model.conditioner = MockConditioner()
    model.sde = MagicMock()
    model.sampler = MagicMock()

    # Use ACTUAL RectifiedFlow scheduler (this is what we want to test!)
    model.rectified_flow = RectifiedFlow(
        velocity_field=model.net,
        train_time_distribution="uniform",
        device=device,
        dtype=torch.float32,
    )

    # Data keys
    model.input_data_key = "video"
    model.input_image_key = "images"
    model.input_caption_key = "ai_caption"
    model.data_parallel_size = 1

    return model


def create_condition(batch_size: int, device: torch.device) -> Text2WorldCondition:
    """Create a Text2WorldCondition for testing."""
    return Text2WorldCondition(
        crossattn_emb=torch.randn(batch_size, 512, 4096, device=device),
        data_type=DataType.VIDEO,
        padding_mask=torch.ones(batch_size, 512, device=device),
        fps=torch.tensor([30.0] * batch_size, device=device),
    )


def create_policy_inputs(
    batch_size: int, num_frames: int, latent_h: int, latent_w: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Create policy-specific inputs for compute_loss_rectified_flow."""
    return {
        "action_chunk": torch.randn(batch_size, 16, 7, device=device),
        "action_indices": torch.zeros(batch_size, dtype=torch.long, device=device),
        "proprio": torch.randn(batch_size, 7, device=device),
        "current_proprio_indices": torch.full((batch_size,), -1, dtype=torch.long, device=device),
        "future_proprio": torch.randn(batch_size, 7, device=device),
        "future_proprio_indices": torch.full((batch_size,), -1, dtype=torch.long, device=device),
        "future_wrist_image_indices": torch.full((batch_size,), -1, dtype=torch.long, device=device),
        "future_wrist_image2_indices": None,
        "future_image_indices": torch.ones(batch_size, dtype=torch.long, device=device),
        "future_image2_indices": None,
        "rollout_data_mask": torch.zeros(batch_size, dtype=torch.long, device=device),
        "world_model_sample_mask": torch.zeros(batch_size, dtype=torch.long, device=device),
        "value_function_sample_mask": torch.zeros(batch_size, dtype=torch.long, device=device),
        "value_function_return": torch.randn(batch_size, device=device),
        "value_indices": torch.full((batch_size,), 2, dtype=torch.long, device=device),
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def latent_shape(batch_size):
    """Return shape (B, C, T, H, W) for latent tensors."""
    return (batch_size, 16, 4, 14, 14)


@pytest.fixture
def model(device) -> CosmosPolicyDiffusionModelRectifiedFlow:
    """Create actual CosmosPolicyDiffusionModelRectifiedFlow with mocked dependencies."""
    return create_model_with_mocked_deps(device)


# =============================================================================
# Test Classes - Testing ACTUAL CosmosPolicyDiffusionModelRectifiedFlow
# =============================================================================


class TestZeroVelocityCase:
    """
    Test identity case: when noise equals clean data, target velocity should be zero
    ON FRAMES NOT MODIFIED BY POLICY INJECTION.

    Note: compute_loss_rectified_flow modifies x0 by injecting:
    - action_chunk at action_indices (frame 0 by default)
    - value_function_return at value_indices (frame 2 by default)

    So we test velocity = 0 only on unmodified frames (1 and 3).
    """

    @pytest.mark.L0
    def test_zero_velocity_on_unmodified_frames_when_noise_equals_clean(
        self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape
    ):
        """
        When epsilon = x0 (noise equals clean data), the target velocity v = epsilon - x0 = 0
        on frames that are NOT modified by policy injection (frames 1 and 3).

        Frames 0 and 2 are modified by action_chunk and value_function_return injection,
        so they will have non-zero velocity.
        """
        B, C, T, H, W = latent_shape

        # Create identical noise and clean data
        shared_data = torch.randn(B, C, T, H, W, device=device)
        x0 = shared_data.clone()
        epsilon = shared_data.clone()

        condition = create_condition(B, device)
        policy_inputs = create_policy_inputs(B, T, H, W, device)

        timesteps = torch.full((B, 1), 500.0, device=device)
        sigmas = timesteps / 1000.0

        # Set network to predict zero velocity
        model.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)

        # Call the ACTUAL compute_loss_rectified_flow method
        output_batch, loss_tensor = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0,
            condition=condition,
            epsilon_B_C_T_H_W=epsilon,
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        # Verify target velocity is zero ONLY on unmodified frames (1 and 3)
        # Frame 0 is modified by action_chunk, frame 2 is modified by value_function_return
        target_velocity = output_batch["vt_target"]
        unmodified_frames = [1, 3]  # Frames not touched by policy injection
        for frame_idx in unmodified_frames:
            frame_velocity = target_velocity[:, :, frame_idx, :, :]
            assert torch.allclose(frame_velocity, torch.zeros_like(frame_velocity), atol=1e-5), (
                f"Target velocity on unmodified frame {frame_idx} should be zero. Max abs: {frame_velocity.abs().max()}"
            )

    @pytest.mark.L0
    def test_modified_frames_have_nonzero_velocity(
        self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape
    ):
        """
        Verify that frames modified by policy injection (action at 0, value at 2)
        have non-zero target velocity even when epsilon = x0.

        This confirms the policy injection is working correctly.
        """
        B, C, T, H, W = latent_shape

        # Create identical noise and clean data
        shared_data = torch.randn(B, C, T, H, W, device=device)
        x0 = shared_data.clone()
        epsilon = shared_data.clone()

        condition = create_condition(B, device)
        policy_inputs = create_policy_inputs(B, T, H, W, device)

        timesteps = torch.full((B, 1), 500.0, device=device)
        sigmas = timesteps / 1000.0

        model.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)

        output_batch, _ = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0,
            condition=condition,
            epsilon_B_C_T_H_W=epsilon,
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        target_velocity = output_batch["vt_target"]

        # Frame 0 (action injection) should have non-zero velocity
        action_frame_velocity = target_velocity[:, :, 0, :, :]
        assert action_frame_velocity.abs().max() > 0.1, (
            f"Action frame (0) should have non-zero velocity due to injection. "
            f"Max abs: {action_frame_velocity.abs().max()}"
        )

        # Frame 2 (value injection) should have non-zero velocity
        value_frame_velocity = target_velocity[:, :, 2, :, :]
        assert value_frame_velocity.abs().max() > 0.1, (
            f"Value frame (2) should have non-zero velocity due to injection. "
            f"Max abs: {value_frame_velocity.abs().max()}"
        )


class TestConstantVelocityCase:
    """
    Test linear drift case: when noise - clean is constant, velocity is that constant
    ON FRAMES NOT MODIFIED BY POLICY INJECTION.

    Note: compute_loss_rectified_flow modifies x0 by injecting:
    - action_chunk at action_indices (frame 0 by default)
    - value_function_return at value_indices (frame 2 by default)

    So we test constant velocity only on unmodified frames (1 and 3).
    """

    @pytest.mark.L0
    def test_constant_velocity_on_unmodified_frames(
        self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape
    ):
        """
        When epsilon = x0 + constant, the target velocity v = epsilon - x0 = constant
        on frames that are NOT modified by policy injection (frames 1 and 3).
        """
        B, C, T, H, W = latent_shape

        x0 = torch.zeros(B, C, T, H, W, device=device)
        constant_offset = 2.5
        epsilon = torch.full((B, C, T, H, W), constant_offset, device=device)

        condition = create_condition(B, device)
        policy_inputs = create_policy_inputs(B, T, H, W, device)

        timesteps = torch.full((B, 1), 500.0, device=device)
        sigmas = torch.full((B, 1), 0.5, device=device)

        model.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)

        output_batch, _ = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0,
            condition=condition,
            epsilon_B_C_T_H_W=epsilon,
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        # Verify constant velocity ONLY on unmodified frames (1 and 3)
        target_velocity = output_batch["vt_target"]
        expected_constant = torch.full((B, C, H, W), constant_offset, device=device)

        unmodified_frames = [1, 3]
        for frame_idx in unmodified_frames:
            frame_velocity = target_velocity[:, :, frame_idx, :, :]
            assert torch.allclose(frame_velocity, expected_constant, atol=1e-5), (
                f"Target velocity on unmodified frame {frame_idx} should be constant {constant_offset}. "
                f"Mean: {frame_velocity.mean()}, Expected: {constant_offset}"
            )

    @pytest.mark.L0
    def test_perfect_prediction_yields_zero_loss(
        self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape
    ):
        """
        Verify that when network predicts the exact target velocity (accounting for
        policy injection), the loss is zero.

        This is the correct way to test "constant velocity" with policy injection:
        first compute what the actual target is, then predict it exactly.
        """
        B, C, T, H, W = latent_shape

        x0 = torch.zeros(B, C, T, H, W, device=device)
        constant_offset = 2.5
        epsilon = torch.full((B, C, T, H, W), constant_offset, device=device)

        condition = create_condition(B, device)
        policy_inputs = create_policy_inputs(B, T, H, W, device)

        timesteps = torch.full((B, 1), 500.0, device=device)
        sigmas = torch.full((B, 1), 0.5, device=device)

        # First pass: get target velocity with dummy prediction
        model.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)
        output_first, _ = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0.clone(),
            condition=condition,
            epsilon_B_C_T_H_W=epsilon.clone(),
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        # Set network to predict the ACTUAL target velocity (including policy injection effects)
        actual_target = output_first["vt_target"]
        model.net.set_target_velocity(actual_target)

        # Second pass: should have zero loss
        _, loss_tensor = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0.clone(),
            condition=condition,
            epsilon_B_C_T_H_W=epsilon.clone(),
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        loss = loss_tensor.mean()
        assert loss.item() < 1e-8, f"Loss should be zero when prediction matches target. Got: {loss.item()}"

    @pytest.mark.L0
    def test_linear_trajectory_via_rectified_flow(
        self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape
    ):
        """
        Verify the ACTUAL RectifiedFlow.get_interpolation computes linear trajectories.
        x_t = epsilon * t + x0 * (1 - t)
        """
        B, C, T, H, W = latent_shape

        x0 = torch.zeros(B, C, T, H, W, device=device)  # clean
        epsilon = torch.ones(B, C, T, H, W, device=device)  # noise

        t1 = torch.full((B, 1), 0.3, device=device)
        t2 = torch.full((B, 1), 0.7, device=device)

        # Use ACTUAL RectifiedFlow.get_interpolation
        x_t1, v_t1 = model.rectified_flow.get_interpolation(epsilon, x0, t1)
        x_t2, v_t2 = model.rectified_flow.get_interpolation(epsilon, x0, t2)

        # Verify linearity: x_t2 - x_t1 = (t2 - t1) * (epsilon - x0)
        diff = x_t2 - x_t1
        expected_diff = (0.7 - 0.3) * (epsilon - x0)

        assert torch.allclose(diff, expected_diff, atol=1e-6), "Trajectory should be linear"

        # Verify velocity is constant and equals epsilon - x0
        assert torch.allclose(v_t1, v_t2, atol=1e-6), "Velocity should be constant"
        assert torch.allclose(v_t1, epsilon - x0, atol=1e-6), "Velocity should be epsilon - x0"


class TestPerfectPredictionCase:
    """
    Test perfect prediction: when network predicts target velocity exactly, loss is zero.

    This tests the ACTUAL compute_loss_rectified_flow method.
    """

    @pytest.mark.L0
    def test_zero_loss_with_perfect_prediction(
        self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape
    ):
        """Loss should be zero when network predicts target velocity exactly."""
        B, C, T, H, W = latent_shape

        x0 = torch.randn(B, C, T, H, W, device=device)
        epsilon = torch.randn(B, C, T, H, W, device=device)

        condition = create_condition(B, device)
        policy_inputs = create_policy_inputs(B, T, H, W, device)

        timesteps = torch.full((B, 1), 500.0, device=device)
        sigmas = torch.full((B, 1), 0.5, device=device)

        # First pass: get target velocity with zero prediction
        model.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)
        output_first, _ = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0.clone(),
            condition=condition,
            epsilon_B_C_T_H_W=epsilon.clone(),
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        # Set network to predict the actual target velocity
        target_velocity = output_first["vt_target"]
        model.net.set_target_velocity(target_velocity)

        # Second pass: should have zero loss
        output_batch, loss_tensor = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0.clone(),
            condition=condition,
            epsilon_B_C_T_H_W=epsilon.clone(),
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        loss = loss_tensor.mean()
        assert loss.item() < 1e-8, f"Loss should be zero for perfect prediction. Got: {loss.item()}"

    @pytest.mark.L0
    def test_nonzero_loss_with_wrong_prediction(
        self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape
    ):
        """Loss should be non-zero when network prediction differs from target."""
        B, C, T, H, W = latent_shape

        x0 = torch.randn(B, C, T, H, W, device=device)
        epsilon = torch.randn(B, C, T, H, W, device=device)

        condition = create_condition(B, device)
        policy_inputs = create_policy_inputs(B, T, H, W, device)

        timesteps = torch.full((B, 1), 500.0, device=device)
        sigmas = torch.full((B, 1), 0.5, device=device)

        # Set network to predict zeros (likely wrong)
        model.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)

        output_batch, loss_tensor = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0,
            condition=condition,
            epsilon_B_C_T_H_W=epsilon,
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        loss = loss_tensor.mean()
        assert loss.item() > 0.01, f"Loss should be non-zero for wrong prediction. Got: {loss.item()}"


class TestOutputStructure:
    """Test that compute_loss_rectified_flow returns correct output structure."""

    @pytest.mark.L0
    def test_output_contains_required_keys(self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape):
        """Verify output_batch contains all required keys."""
        B, C, T, H, W = latent_shape

        x0 = torch.randn(B, C, T, H, W, device=device)
        epsilon = torch.randn(B, C, T, H, W, device=device)
        condition = create_condition(B, device)
        policy_inputs = create_policy_inputs(B, T, H, W, device)

        timesteps = torch.full((B, 1), 500.0, device=device)
        sigmas = torch.full((B, 1), 0.5, device=device)

        model.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)

        output_batch, _ = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0,
            condition=condition,
            epsilon_B_C_T_H_W=epsilon,
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        required_keys = [
            "x0",
            "xt",
            "timesteps",
            "sigmas",
            "condition",
            "vt_pred",
            "vt_target",
            "velocity_mse_loss",
            "edm_loss",
            "velocity_loss_per_frame",
            "demo_sample_action_mse_loss",
            "demo_sample_action_l1_loss",
        ]

        for key in required_keys:
            assert key in output_batch, f"Missing required key: {key}"

    @pytest.mark.L0
    def test_output_shapes(self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape):
        """Verify output tensors have correct shapes."""
        B, C, T, H, W = latent_shape

        x0 = torch.randn(B, C, T, H, W, device=device)
        epsilon = torch.randn(B, C, T, H, W, device=device)
        condition = create_condition(B, device)
        policy_inputs = create_policy_inputs(B, T, H, W, device)

        timesteps = torch.full((B, 1), 500.0, device=device)
        sigmas = torch.full((B, 1), 0.5, device=device)

        model.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)

        output_batch, loss_tensor = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0,
            condition=condition,
            epsilon_B_C_T_H_W=epsilon,
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        assert output_batch["x0"].shape == (B, C, T, H, W)
        assert output_batch["xt"].shape == (B, C, T, H, W)
        assert output_batch["vt_pred"].shape == (B, C, T, H, W)
        assert output_batch["vt_target"].shape == (B, C, T, H, W)
        assert output_batch["velocity_loss_per_frame"].shape == (B, T)


class TestLossMasking:
    """Test loss masking functionality."""

    @pytest.mark.L1
    def test_action_loss_multiplier(self, device, latent_shape):
        """Test that action_loss_multiplier affects loss weighting."""
        B, C, T, H, W = latent_shape

        # Model with multiplier = 1
        model_base = create_model_with_mocked_deps(device)
        model_base.config.action_loss_multiplier = 1

        # Model with multiplier = 2
        model_mult = create_model_with_mocked_deps(device)
        model_mult.config.action_loss_multiplier = 2

        x0 = torch.randn(B, C, T, H, W, device=device)
        epsilon = torch.randn(B, C, T, H, W, device=device)
        condition = create_condition(B, device)
        policy_inputs = create_policy_inputs(B, T, H, W, device)

        timesteps = torch.full((B, 1), 500.0, device=device)
        sigmas = torch.full((B, 1), 0.5, device=device)

        # Both predict zeros
        model_base.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)
        model_mult.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)

        _, loss_base = model_base.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0.clone(),
            condition=condition,
            epsilon_B_C_T_H_W=epsilon.clone(),
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        _, loss_mult = model_mult.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0.clone(),
            condition=condition,
            epsilon_B_C_T_H_W=epsilon.clone(),
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        # With higher multiplier, loss should be >= base
        assert loss_mult.mean() >= loss_base.mean(), (
            f"Loss with multiplier should be >= base. Base: {loss_base.mean()}, Mult: {loss_mult.mean()}"
        )


class TestVelocityMathematicalProperties:
    """Test mathematical properties of the velocity field."""

    @pytest.mark.L0
    def test_velocity_independent_of_time(self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape):
        """
        Target velocity v = epsilon - x0 should be independent of time t
        (tested on unmodified frames to avoid policy injection effects).
        """
        B, C, T, H, W = latent_shape

        x0 = torch.randn(B, C, T, H, W, device=device)
        epsilon = torch.randn(B, C, T, H, W, device=device)
        condition = create_condition(B, device)
        policy_inputs = create_policy_inputs(B, T, H, W, device)

        model.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)

        # Unmodified frames (not affected by policy injection)
        unmodified_frames = [1, 3]

        velocities = []
        for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            timesteps = torch.full((B, 1), t_val * 1000, device=device)
            sigmas = torch.full((B, 1), t_val, device=device)

            output_batch, _ = model.compute_loss_rectified_flow(
                x0_B_C_T_H_W=x0.clone(),
                condition=condition,
                epsilon_B_C_T_H_W=epsilon.clone(),
                timesteps_B_T=timesteps,
                sigmas_B_T=sigmas,
                **policy_inputs,
            )
            # Only check unmodified frames
            vel_unmodified = output_batch["vt_target"][:, :, unmodified_frames, :, :]
            velocities.append(vel_unmodified)

        # All velocities on unmodified frames should be the same regardless of time
        for i, vel in enumerate(velocities[1:], 1):
            assert torch.allclose(vel, velocities[0], atol=1e-5), (
                f"Velocity on unmodified frames should be constant across time. "
                f"Mismatch at t={[0.1, 0.3, 0.5, 0.7, 0.9][i]}"
            )


# =============================================================================
# Latent Injection Tests
# =============================================================================


class TestActionChunkInjection:
    """
    Test the action chunk injection mechanism.

    The replace_latent_with_action_chunk function:
    1. Flattens action_chunk (B, chunk_size, action_dim) -> (B, chunk_size * action_dim)
    2. Repeats to fill the latent volume (C * H * W elements)
    3. Reshapes and places at x0[:, :, action_indices, :, :]
    """

    @pytest.mark.L0
    def test_action_injection_places_values_correctly(self, device, latent_shape):
        """Verify action chunk values are correctly placed in the latent."""
        B, C, T, H, W = latent_shape

        # Create x0 with known values (all zeros)
        x0 = torch.zeros(B, C, T, H, W, device=device)

        # Create action_chunk with known values (all ones)
        chunk_size = 16
        action_dim = 7
        action_chunk = torch.ones(B, chunk_size, action_dim, device=device)
        action_indices = torch.zeros(B, dtype=torch.long, device=device)  # Inject at frame 0

        # Inject action
        x0_modified = replace_latent_with_action_chunk(x0.clone(), action_chunk, action_indices)

        # Frame 0 should be modified (all 1s repeated to fill volume)
        action_frame = x0_modified[:, :, 0, :, :]
        assert action_frame.abs().sum() > 0, "Action frame should be modified"
        assert torch.allclose(action_frame, torch.ones_like(action_frame), atol=1e-5), (
            "Action frame should contain repeated 1s"
        )

        # Other frames should remain zeros
        for frame_idx in [1, 2, 3]:
            other_frame = x0_modified[:, :, frame_idx, :, :]
            assert torch.allclose(other_frame, torch.zeros_like(other_frame), atol=1e-5), (
                f"Frame {frame_idx} should remain unmodified (zeros)"
            )

    @pytest.mark.L0
    def test_action_injection_preserves_values_pattern(self, device, latent_shape):
        """Verify the action values pattern is preserved after injection."""
        B, C, T, H, W = latent_shape

        x0 = torch.zeros(B, C, T, H, W, device=device)

        # Create action_chunk with distinct values
        chunk_size = 8
        action_dim = 7
        # Each action element has a unique value: 0.1, 0.2, 0.3, ...
        action_chunk = torch.arange(chunk_size * action_dim, device=device, dtype=torch.float32)
        action_chunk = action_chunk.unsqueeze(0).expand(B, -1).reshape(B, chunk_size, action_dim) * 0.01

        action_indices = torch.zeros(B, dtype=torch.long, device=device)

        x0_modified = replace_latent_with_action_chunk(x0.clone(), action_chunk, action_indices)

        # Extract the action frame and flatten
        action_frame = x0_modified[:, :, 0, :, :]
        flat_action_frame = action_frame.reshape(B, -1)

        # The first (chunk_size * action_dim) elements should match the original action
        flat_action = action_chunk.reshape(B, -1)
        num_action_elements = flat_action.shape[1]

        assert torch.allclose(flat_action_frame[:, :num_action_elements], flat_action, atol=1e-5), (
            "First elements of action frame should match original action values"
        )

    @pytest.mark.L1
    def test_action_injection_with_different_indices(self, device, latent_shape):
        """Verify action injection works with different frame indices per batch."""
        B, C, T, H, W = latent_shape

        x0 = torch.zeros(B, C, T, H, W, device=device)

        chunk_size = 8
        action_dim = 7
        action_chunk = torch.ones(B, chunk_size, action_dim, device=device) * 2.0

        # Different indices for each batch item
        action_indices = torch.tensor([0, 1], dtype=torch.long, device=device)[:B]

        x0_modified = replace_latent_with_action_chunk(x0.clone(), action_chunk, action_indices)

        # Verify each batch item has injection at correct frame
        for b in range(B):
            idx = action_indices[b].item()
            injected_frame = x0_modified[b, :, idx, :, :]
            assert torch.allclose(injected_frame, torch.ones_like(injected_frame) * 2.0, atol=1e-5), (
                f"Batch {b}, frame {idx} should be injected with 2.0"
            )


class TestProprioInjection:
    """
    Test the proprioception injection mechanism.

    The replace_latent_with_proprio function:
    1. Takes proprio (B, proprio_dim)
    2. Repeats to fill the latent volume (C * H * W elements)
    3. Reshapes and places at x0[:, :, proprio_indices, :, :]
    """

    @pytest.mark.L0
    def test_proprio_injection_places_values_correctly(self, device, latent_shape):
        """Verify proprio values are correctly placed in the latent."""
        B, C, T, H, W = latent_shape

        x0 = torch.zeros(B, C, T, H, W, device=device)

        # Create proprio with known values
        proprio_dim = 7
        proprio = torch.ones(B, proprio_dim, device=device) * 3.0
        proprio_indices = torch.ones(B, dtype=torch.long, device=device)  # Inject at frame 1

        x0_modified = replace_latent_with_proprio(x0.clone(), proprio, proprio_indices)

        # Frame 1 should be modified
        proprio_frame = x0_modified[:, :, 1, :, :]
        assert proprio_frame.abs().sum() > 0, "Proprio frame should be modified"

        # Other frames should remain zeros
        for frame_idx in [0, 2, 3]:
            other_frame = x0_modified[:, :, frame_idx, :, :]
            assert torch.allclose(other_frame, torch.zeros_like(other_frame), atol=1e-5), (
                f"Frame {frame_idx} should remain unmodified"
            )

    @pytest.mark.L0
    def test_proprio_injection_preserves_values_pattern(self, device, latent_shape):
        """Verify the proprio values pattern is preserved after injection."""
        B, C, T, H, W = latent_shape

        x0 = torch.zeros(B, C, T, H, W, device=device)

        # Create proprio with distinct values per dimension
        proprio_dim = 9
        proprio = torch.arange(proprio_dim, device=device, dtype=torch.float32)
        proprio = proprio.unsqueeze(0).expand(B, -1) * 0.1

        proprio_indices = torch.full((B,), 2, dtype=torch.long, device=device)

        x0_modified = replace_latent_with_proprio(x0.clone(), proprio, proprio_indices)

        # Extract and flatten the proprio frame
        proprio_frame = x0_modified[:, :, 2, :, :]
        flat_proprio_frame = proprio_frame.reshape(B, -1)

        # First proprio_dim elements should match original proprio
        assert torch.allclose(flat_proprio_frame[:, :proprio_dim], proprio, atol=1e-5), (
            "First elements should match original proprio values"
        )


class TestValueInjection:
    """
    Test the value function return injection mechanism.

    Value injection:
    1. Takes value_function_return (B,) scalar per batch
    2. Broadcasts to fill entire latent volume (C, H, W) with the same scalar
    3. Places at x0[:, :, value_indices, :, :]
    """

    @pytest.mark.L0
    def test_value_injection_broadcasts_scalar(
        self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape
    ):
        """Verify value function return is broadcast correctly to fill latent volume."""
        B, C, T, H, W = latent_shape

        # Create x0 with zeros, epsilon with known values
        x0 = torch.zeros(B, C, T, H, W, device=device)
        epsilon = torch.zeros(B, C, T, H, W, device=device)

        condition = create_condition(B, device)

        # Create policy inputs with specific value_function_return
        value_return = 0.75
        policy_inputs = create_policy_inputs(B, T, H, W, device)
        policy_inputs["value_function_return"] = torch.full((B,), value_return, device=device)
        policy_inputs["value_indices"] = torch.full((B,), 2, dtype=torch.long, device=device)

        timesteps = torch.full((B, 1), 500.0, device=device)
        sigmas = torch.full((B, 1), 0.5, device=device)

        model.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)

        output_batch, _ = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0.clone(),
            condition=condition,
            epsilon_B_C_T_H_W=epsilon.clone(),
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        # The modified x0 is stored in output_batch["x0"]
        modified_x0 = output_batch["x0"]

        # Value frame (index 2) should be filled with value_return
        value_frame = modified_x0[:, :, 2, :, :]
        expected_value_frame = torch.full_like(value_frame, value_return)

        assert torch.allclose(value_frame, expected_value_frame, atol=1e-5), (
            f"Value frame should be filled with {value_return}. "
            f"Got mean: {value_frame.mean()}, expected: {value_return}"
        )

    @pytest.mark.L0
    def test_value_injection_per_batch_item(self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape):
        """Verify different value returns per batch item are handled correctly."""
        B, C, T, H, W = latent_shape

        x0 = torch.zeros(B, C, T, H, W, device=device)
        epsilon = torch.zeros(B, C, T, H, W, device=device)

        condition = create_condition(B, device)

        # Different value returns per batch item
        value_returns = torch.tensor([0.5, 0.9], device=device)[:B]

        policy_inputs = create_policy_inputs(B, T, H, W, device)
        policy_inputs["value_function_return"] = value_returns
        policy_inputs["value_indices"] = torch.full((B,), 2, dtype=torch.long, device=device)

        timesteps = torch.full((B, 1), 500.0, device=device)
        sigmas = torch.full((B, 1), 0.5, device=device)

        model.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)

        output_batch, _ = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0.clone(),
            condition=condition,
            epsilon_B_C_T_H_W=epsilon.clone(),
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        modified_x0 = output_batch["x0"]

        # Check each batch item has correct value
        for b in range(B):
            value_frame = modified_x0[b, :, 2, :, :]
            expected = value_returns[b].item()
            assert torch.allclose(value_frame, torch.full_like(value_frame, expected), atol=1e-5), (
                f"Batch {b} value frame should be {expected}, got mean: {value_frame.mean()}"
            )


class TestInjectionAffectsTargetVelocity:
    """
    Test that latent injection correctly affects the target velocity.

    Target velocity: v = epsilon - modified_x0
    So if we know what's injected into x0, we can predict the target velocity.
    """

    @pytest.mark.L0
    def test_action_injection_affects_velocity(
        self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape
    ):
        """
        Verify action injection affects target velocity at action frame.

        If epsilon[action_frame] = 0 and action_chunk fills with value A,
        then v[action_frame] = 0 - A = -A
        """
        B, C, T, H, W = latent_shape

        # Start with zeros for both x0 and epsilon
        x0 = torch.zeros(B, C, T, H, W, device=device)
        epsilon = torch.zeros(B, C, T, H, W, device=device)

        condition = create_condition(B, device)

        # Create action_chunk with known value
        action_value = 5.0
        policy_inputs = create_policy_inputs(B, T, H, W, device)
        policy_inputs["action_chunk"] = torch.full((B, 16, 7), action_value, device=device)
        policy_inputs["action_indices"] = torch.zeros(B, dtype=torch.long, device=device)  # Frame 0

        timesteps = torch.full((B, 1), 500.0, device=device)
        sigmas = torch.full((B, 1), 0.5, device=device)

        model.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)

        output_batch, _ = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0.clone(),
            condition=condition,
            epsilon_B_C_T_H_W=epsilon.clone(),
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        target_velocity = output_batch["vt_target"]

        # Action frame velocity should be -action_value (since epsilon=0, x0_modified=action_value)
        action_frame_velocity = target_velocity[:, :, 0, :, :]
        expected_velocity = -action_value

        assert torch.allclose(
            action_frame_velocity, torch.full_like(action_frame_velocity, expected_velocity), atol=1e-5
        ), f"Action frame velocity should be {expected_velocity}. Got mean: {action_frame_velocity.mean()}"

    @pytest.mark.L0
    def test_value_injection_affects_velocity(
        self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape
    ):
        """
        Verify value injection affects target velocity at value frame.

        If epsilon[value_frame] = 0 and value_function_return = V,
        then v[value_frame] = 0 - V = -V
        """
        B, C, T, H, W = latent_shape

        x0 = torch.zeros(B, C, T, H, W, device=device)
        epsilon = torch.zeros(B, C, T, H, W, device=device)

        condition = create_condition(B, device)

        value_return = 2.5
        policy_inputs = create_policy_inputs(B, T, H, W, device)
        policy_inputs["value_function_return"] = torch.full((B,), value_return, device=device)
        policy_inputs["value_indices"] = torch.full((B,), 2, dtype=torch.long, device=device)
        # Set action to known value to avoid interference
        policy_inputs["action_chunk"] = torch.zeros(B, 16, 7, device=device)

        timesteps = torch.full((B, 1), 500.0, device=device)
        sigmas = torch.full((B, 1), 0.5, device=device)

        model.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)

        output_batch, _ = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0.clone(),
            condition=condition,
            epsilon_B_C_T_H_W=epsilon.clone(),
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        target_velocity = output_batch["vt_target"]

        # Value frame velocity should be -value_return
        value_frame_velocity = target_velocity[:, :, 2, :, :]
        expected_velocity = -value_return

        assert torch.allclose(
            value_frame_velocity, torch.full_like(value_frame_velocity, expected_velocity), atol=1e-5
        ), f"Value frame velocity should be {expected_velocity}. Got mean: {value_frame_velocity.mean()}"

    @pytest.mark.L0
    def test_combined_injection_independence(
        self, model: CosmosPolicyDiffusionModelRectifiedFlow, device, latent_shape
    ):
        """
        Verify action and value injections are independent (affect different frames).
        """
        B, C, T, H, W = latent_shape

        x0 = torch.zeros(B, C, T, H, W, device=device)
        epsilon = torch.zeros(B, C, T, H, W, device=device)

        condition = create_condition(B, device)

        action_value = 3.0
        value_return = 7.0

        policy_inputs = create_policy_inputs(B, T, H, W, device)
        policy_inputs["action_chunk"] = torch.full((B, 16, 7), action_value, device=device)
        policy_inputs["action_indices"] = torch.zeros(B, dtype=torch.long, device=device)  # Frame 0
        policy_inputs["value_function_return"] = torch.full((B,), value_return, device=device)
        policy_inputs["value_indices"] = torch.full((B,), 2, dtype=torch.long, device=device)  # Frame 2

        timesteps = torch.full((B, 1), 500.0, device=device)
        sigmas = torch.full((B, 1), 0.5, device=device)

        model.net._target_velocity = torch.zeros(B, C, T, H, W, device=device)

        output_batch, _ = model.compute_loss_rectified_flow(
            x0_B_C_T_H_W=x0.clone(),
            condition=condition,
            epsilon_B_C_T_H_W=epsilon.clone(),
            timesteps_B_T=timesteps,
            sigmas_B_T=sigmas,
            **policy_inputs,
        )

        target_velocity = output_batch["vt_target"]

        # Action frame (0) should have velocity = -action_value
        action_velocity = target_velocity[:, :, 0, :, :].mean()
        assert abs(action_velocity.item() - (-action_value)) < 0.1, (
            f"Action frame velocity should be ~{-action_value}, got {action_velocity.item()}"
        )

        # Value frame (2) should have velocity = -value_return
        value_velocity = target_velocity[:, :, 2, :, :].mean()
        assert abs(value_velocity.item() - (-value_return)) < 0.1, (
            f"Value frame velocity should be ~{-value_return}, got {value_velocity.item()}"
        )

        # Unmodified frames (1, 3) should have zero velocity (epsilon=x0=0)
        for frame_idx in [1, 3]:
            frame_velocity = target_velocity[:, :, frame_idx, :, :]
            assert torch.allclose(frame_velocity, torch.zeros_like(frame_velocity), atol=1e-5), (
                f"Frame {frame_idx} should have zero velocity"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--L0"])
