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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from cosmos_predict2._src.imaginaire.utils.helper_test import RunIf
from cosmos_predict2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference

"""
Usage:
    # Run all tests including multi-GPU tests
    pytest -s cosmos_predict2/_src/transfer2/inference/inference_pipeline_test.py --all -k test_inference_pipeline

    # Run only L1 tests (basic unit tests)
    pytest -s cosmos_predict2/_src/transfer2/inference/inference_pipeline_test.py --L1 -k test_inference_pipeline

    # Run L2 tests (multi-GPU inference tests) - requires 4+ GPUs
    torchrun --nproc_per_node=4 -m pytest -s cosmos_predict2/_src/transfer2/inference/inference_pipeline_test.py --L2 -k test_inference_pipeline

    # Run L2 tests with 8 GPUs
    torchrun --nproc_per_node=8 -m pytest -s cosmos_predict2/_src/transfer2/inference/inference_pipeline_test.py --L2 -k test_inference_pipeline
"""


def create_random_video_tensor(shape: tuple, dtype: torch.dtype = torch.uint8):
    """Create a random video tensor with the specified shape and dtype."""
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=dtype)
    else:
        return torch.randn(shape, dtype=dtype)


def create_random_text_embeddings(shape: tuple, dtype: torch.dtype = torch.float32):
    """Create random text embeddings."""
    return torch.randn(shape, dtype=dtype)


def create_random_control_inputs(hint_keys=["edge"], batch_size=1, num_frames=93, height=128, width=128):
    """Create random control input tensors."""
    control_dict = {}
    for key in hint_keys:
        control_dict[f"control_input_{key}"] = torch.randn(batch_size, 3, num_frames, height, width)
    return control_dict


def get_test_video2world_config():
    """Create a minimal test config for Video2WorldModel that supports 93 frames."""
    from omegaconf import DictConfig

    from cosmos_predict2._src.imaginaire.utils.config_helper import override
    from cosmos_predict2._src.predict2.configs.common.defaults.ema import PowerEMAConfig
    from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import VideoPredictionConditioner
    from cosmos_predict2._src.predict2.configs.video2world.defaults.net import mini_net
    from cosmos_predict2._src.predict2.models.video2world_model import Video2WorldConfig

    # Create a custom tokenizer config that supports 93 frames per chunk
    tokenizer_config = DictConfig(
        {
            "_target_": "projects.cosmos.diffusion.v1.module.pretrained_vae.DummyJointImageVideoTokenizer",
            "name": "dummy_joint_image_video_test",
            "pixel_ch": 3,
            "latent_ch": 16,
            "pixel_chunk_duration": 93,
            "latent_chunk_duration": 24,
            "spatial_compression_factor": 8,
            "temporal_compression_factor": 4,
            "spatial_resolution": "720",
        }
    )

    config = Video2WorldConfig(
        tokenizer=tokenizer_config,
        conditioner=VideoPredictionConditioner,
        net=mini_net,
        ema=PowerEMAConfig,
        state_t=24,
    )
    return override(config)


@pytest.mark.L1
def test_inference_pipeline_get_num_chunks():
    """Test the _get_num_chunks method with various input sizes."""
    dtype = torch.uint8

    # Mock only the heavy model loading
    with patch("cosmos_predict2._src.transfer2.inference.inference_pipeline.load_model_from_checkpoint") as mock_load:
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_load.return_value = (mock_model, mock_config)

        pipeline = ControlVideo2WorldInference(
            registered_exp_name="test_experiment",
            checkpoint_paths="test_path",
            s3_credential_path="test_cred",
            num_video_frames_per_chunk=93,
            num_conditional_frames=1,
        )

    # Test case 1: Small video (< chunk size)
    input_frames = create_random_video_tensor((3, 50, 128, 128), dtype)
    num_total_frames, num_chunks, num_frames_per_chunk = pipeline._get_num_chunks(input_frames)
    assert num_total_frames == 50
    assert num_chunks == 1
    assert num_frames_per_chunk == 92  # 93 - 1

    # Test case 2: Exactly chunk size
    input_frames = create_random_video_tensor((3, 93, 128, 128), dtype)
    num_total_frames, num_chunks, num_frames_per_chunk = pipeline._get_num_chunks(input_frames)
    assert num_total_frames == 93
    assert num_chunks == 1
    assert num_frames_per_chunk == 92

    # Test case 3: Large video (> chunk size)
    input_frames = create_random_video_tensor((3, 200, 128, 128), dtype)
    num_total_frames, num_chunks, num_frames_per_chunk = pipeline._get_num_chunks(input_frames)
    assert num_total_frames == 200
    assert num_chunks == 3  # ceil((200-93)/92) + 1
    assert num_frames_per_chunk == 92

    print("_get_num_chunks test passed.")


@pytest.mark.L1
def test_inference_pipeline_pad_input_frames():
    """Test the _pad_input_frames method."""
    dtype = torch.uint8

    # Mock only the heavy model loading
    with patch("cosmos_predict2._src.transfer2.inference.inference_pipeline.load_model_from_checkpoint") as mock_load:
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_load.return_value = (mock_model, mock_config)

        pipeline = ControlVideo2WorldInference(
            registered_exp_name="test_experiment",
            checkpoint_paths="test_path",
            s3_credential_path="test_cred",
            num_video_frames_per_chunk=93,
        )

    # Test case 1: Video shorter than chunk size (needs padding)
    input_frames = create_random_video_tensor((3, 50, 128, 128), dtype)
    padded_frames = pipeline._pad_input_frames(input_frames, 50)
    assert padded_frames.shape[1] == 93  # Should be padded to chunk size
    # Check that padding uses the last frame
    assert torch.equal(padded_frames[:, 49, :, :], padded_frames[:, 50, :, :])

    # Test case 2: Video equal to chunk size (no padding needed)
    input_frames = create_random_video_tensor((3, 93, 128, 128), dtype)
    padded_frames = pipeline._pad_input_frames(input_frames, 93)
    assert padded_frames.shape[1] == 93  # Should remain same
    assert torch.equal(input_frames, padded_frames)

    # Test case 3: Video longer than chunk size (no padding needed)
    input_frames = create_random_video_tensor((3, 100, 128, 128), dtype)
    padded_frames = pipeline._pad_input_frames(input_frames, 100)
    assert padded_frames.shape[1] == 100  # Should remain same
    assert torch.equal(input_frames, padded_frames)

    print("_pad_input_frames test passed.")


@pytest.mark.L1
def test_inference_pipeline_multi_branch_generate_img2world():
    """Test the inference pipeline initialization with multi-branch checkpoints."""

    # Mock the model loading and multi-branch checkpoint loading
    with patch("cosmos_predict2._src.transfer2.inference.inference_pipeline.load_model_from_checkpoint") as mock_load:
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_load.return_value = (mock_model, mock_config)
        mock_model.load_multi_branch_checkpoints = MagicMock()

        # Test cases: (checkpoint_paths, skip_load_model, should_call_multi_branch)
        test_cases = [
            ("single_checkpoint", False, False),  # Single string
            (["single_checkpoint"], False, False),  # Single item list
            (["ckpt1", "ckpt2", "ckpt3"], False, True),  # Multiple checkpoints
            (["ckpt1", "ckpt2"], True, False),  # Multiple but skip_load_model=True
        ]

        for i, (checkpoint_paths, skip_load_model, should_call) in enumerate(test_cases):
            mock_model.reset_mock()

            # Create pipeline
            pipeline = ControlVideo2WorldInference(
                registered_exp_name="test_experiment",
                checkpoint_paths=checkpoint_paths,
                s3_credential_path="test_cred",
                skip_load_model=skip_load_model,
            )

            # Verify multi-branch loading behavior
            if should_call:
                mock_model.load_multi_branch_checkpoints.assert_called_once_with(checkpoint_paths=checkpoint_paths)
            else:
                mock_model.load_multi_branch_checkpoints.assert_not_called()

            # Verify pipeline attributes
            assert pipeline.model == mock_model
            assert pipeline.config == mock_config
            expected_first_path = checkpoint_paths if isinstance(checkpoint_paths, str) else checkpoint_paths[0]
            assert pipeline.checkpoint_path == expected_first_path

    print("Multi-branch generate_img2world test passed.")


@pytest.mark.L1
def test_inference_pipeline_get_data_batch_input():
    """Test the _get_data_batch_input method."""
    dtype = torch.uint8

    # Mock only the heavy model loading
    with patch("cosmos_predict2._src.transfer2.inference.inference_pipeline.load_model_from_checkpoint") as mock_load:
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_load.return_value = (mock_model, mock_config)

        pipeline = ControlVideo2WorldInference(
            registered_exp_name="test_experiment",
            checkpoint_paths="test_path",
            s3_credential_path="test_cred",
            num_video_frames_per_chunk=93,
        )

    # Create test input tensors
    video = create_random_video_tensor((3, 93, 128, 128), dtype)
    prev_output = create_random_video_tensor((3, 93, 128, 128), dtype).unsqueeze(0)  # Add batch dimension
    text_embeddings = create_random_text_embeddings((1, 77, 1024), torch.float32)
    fps = 24

    # Test without image context
    data_batch = pipeline._get_data_batch_input(video, prev_output, text_embeddings, fps)

    assert "video" in data_batch
    assert "t5_text_embeddings" in data_batch
    assert "input_video" in data_batch
    assert data_batch["video"].shape == prev_output.shape
    assert data_batch["t5_text_embeddings"].shape == text_embeddings.shape
    assert data_batch["control_weight"] == [1.0]

    # Test with image context
    image_context = create_random_video_tensor((3, 128, 128), dtype).unsqueeze(0)  # (B, C, H, W)
    data_batch_with_img = pipeline._get_data_batch_input(
        video, prev_output, text_embeddings, fps, image_context=image_context
    )

    assert "image_context" in data_batch_with_img
    assert data_batch_with_img["image_context"].shape == image_context.shape

    # Test with custom control weight
    data_batch_control = pipeline._get_data_batch_input(video, prev_output, text_embeddings, fps, control_weight="0.5")
    assert data_batch_control["control_weight"] == [0.5]

    print("_get_data_batch_input test passed.")


@pytest.mark.L1
def test_inference_pipeline_chunk_calculations():
    """Test various chunk calculation scenarios."""
    dtype = torch.uint8

    # Mock only the heavy model loading
    with patch("cosmos_predict2._src.transfer2.inference.inference_pipeline.load_model_from_checkpoint") as mock_load:
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_load.return_value = (mock_model, mock_config)

        # Test different chunk sizes
        test_cases = [
            (93, 1, 50, 1, 92),  # chunk_size=93, cond_frames=1, input_frames=50 -> 1 chunk
            (93, 1, 93, 1, 92),  # chunk_size=93, cond_frames=1, input_frames=93 -> 1 chunk
            (93, 1, 185, 2, 92),  # chunk_size=93, cond_frames=1, input_frames=185 -> 2 chunks
            (93, 1, 186, 3, 92),  # chunk_size=93, cond_frames=1, input_frames=186 -> 3 chunks (remainder case)
            (24, 1, 100, 5, 23),  # chunk_size=24, cond_frames=1, input_frames=100 -> 5 chunks
        ]

        for chunk_size, cond_frames, input_frames_count, expected_chunks, expected_frames_per_chunk in test_cases:
            pipeline = ControlVideo2WorldInference(
                registered_exp_name="test_experiment",
                checkpoint_paths="test_path",
                s3_credential_path="test_cred",
                num_video_frames_per_chunk=chunk_size,
                num_conditional_frames=cond_frames,
            )

            input_frames = create_random_video_tensor((3, input_frames_count, 64, 64), dtype)
            num_total_frames, num_chunks, num_frames_per_chunk = pipeline._get_num_chunks(input_frames)

            assert num_total_frames == input_frames_count
            assert num_chunks == expected_chunks, f"Expected {expected_chunks} chunks, got {num_chunks}"
            assert num_frames_per_chunk == expected_frames_per_chunk

    print("Chunk calculations test passed.")


@pytest.mark.L1
def test_inference_pipeline_tensor_shapes():
    """Test that tensor shapes are handled correctly."""
    dtype = torch.uint8

    with patch("cosmos_predict2._src.transfer2.inference.inference_pipeline.load_model_from_checkpoint") as mock_load:
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_load.return_value = (mock_model, mock_config)

        pipeline = ControlVideo2WorldInference(
            registered_exp_name="test_experiment",
            checkpoint_paths="test_path",
            s3_credential_path="test_cred",
        )

    # Test different tensor dimensions - using (C, T, H, W) format
    test_shapes = [
        (3, 10, 64, 64),
        (3, 93, 128, 128),
        (3, 200, 256, 256),
    ]

    for shape in test_shapes:
        input_frames = create_random_video_tensor(shape, dtype)

        # Test _get_num_chunks
        num_total_frames, num_chunks, num_frames_per_chunk = pipeline._get_num_chunks(input_frames)
        assert num_total_frames == shape[1]  # T dimension

        # Test _pad_input_frames
        padded = pipeline._pad_input_frames(input_frames, shape[1])
        assert padded.shape[0] == shape[0]  # C dimension unchanged
        assert padded.shape[2:] == shape[2:]  # spatial dims unchanged

        # Test data batch creation
        prev_output = create_random_video_tensor(shape, dtype).unsqueeze(0)  # Add batch dimension for data batch
        text_embeddings = create_random_text_embeddings((1, 512, 1024), torch.float32)

        data_batch = pipeline._get_data_batch_input(input_frames, prev_output, text_embeddings, 24)

        assert data_batch["video"].shape == prev_output.shape
        assert data_batch["input_video"].shape == input_frames.shape

    print("Tensor shapes test passed.")


@RunIf(min_gpus=4)
@pytest.mark.L2
def test_inference_pipeline_real_inference():
    """Test the actual inference pipeline with tensor inputs.

    Requires 4+ GPUs. Run with:
    torchrun --nproc_per_node=4 -m pytest -s cosmos_predict2/_src/transfer2/inference/inference_pipeline_test.py --L2 -k test_inference_pipeline_real_inference
    """
    dtype = torch.uint8

    # Mock only the model loading - everything else should be real
    with patch("cosmos_predict2._src.transfer2.inference.inference_pipeline.load_model_from_checkpoint") as mock_load:
        # Create a real model instance for testing with proper config
        from cosmos_predict2._src.predict2.models.video2world_model import Video2WorldModel

        # Create proper config that supports 93 frames
        config = get_test_video2world_config()

        # Create a real model
        model = Video2WorldModel(config)
        model.cuda()
        model.eval()
        model.on_train_start()  # Initialize model properly

        mock_load.return_value = (model, config)

        # Create pipeline (using 93 frames per chunk)
        pipeline = ControlVideo2WorldInference(
            registered_exp_name="test_experiment",
            checkpoint_paths="test_path",
            s3_credential_path="test_cred",
            num_video_frames_per_chunk=93,
            num_conditional_frames=1,
        )

        # Create test data matching real scenario (93 frames, 93 frames per chunk)
        input_video = create_random_video_tensor((3, 93, 128, 128), dtype)
        text_embeddings = create_random_text_embeddings((1, 77, 1024), torch.float32)

        # Test the core inference components directly
        # 1. Test _get_num_chunks
        num_total_frames, num_chunks, num_frames_per_chunk = pipeline._get_num_chunks(input_video)
        assert num_total_frames == 93
        assert num_chunks == 1  # 121 frames with 93 per chunk = 2 chunks
        assert num_frames_per_chunk == 92

        # 2. Test _pad_input_frames (no padding needed for this scenario)
        padded_video = pipeline._pad_input_frames(input_video, num_total_frames)
        assert torch.equal(input_video, padded_video)  # No padding needed

        # 3. Test _get_data_batch_input (test with first chunk)
        chunk_0_frames = input_video[:, 0:93]  # First chunk: 93 frames
        prev_output = create_random_video_tensor((3, 93, 128, 128), dtype).unsqueeze(0)  # Add batch dimension
        data_batch = pipeline._get_data_batch_input(
            chunk_0_frames,
            prev_output,
            text_embeddings,
            fps=24,
            control_weight="1.0",
        )

        # Verify data batch structure
        assert "video" in data_batch
        assert "t5_text_embeddings" in data_batch
        assert "input_video" in data_batch
        assert data_batch["control_weight"] == [1.0]

        # 4. Test control input augmentation
        from cosmos_predict2._src.transfer2.datasets.augmentors.control_input import get_augmentor_for_eval

        # Mock _maybe_torch_to_numpy to handle CUDA tensors properly
        def mock_maybe_torch_to_numpy(frames):
            """Mock to handle CUDA tensors properly for testing."""
            if hasattr(frames, "cpu"):
                return frames.cpu().numpy()
            elif hasattr(frames, "numpy"):
                return frames.numpy()
            else:
                return np.array(frames)

        # Add control inputs via augmentor
        with patch(
            "cosmos_predict2._src.transfer2.datasets.augmentors.control_input._maybe_torch_to_numpy",
            side_effect=mock_maybe_torch_to_numpy,
        ):
            augmented_batch = get_augmentor_for_eval(
                data_dict=data_batch,
                input_keys=["input_video"],
                output_keys=["edge"],
                preset_edge_threshold="medium",
            )

        # Verify control inputs were added
        assert "control_input_edge" in augmented_batch

        # 5. Test REAL model forward pass
        sample = model.generate_samples_from_batch(
            augmented_batch,
            n_sample=1,
            guidance=7,
            seed=42,
            is_negative_prompt=False,
        )
        assert sample is not None
        print(f"Generated sample shape: {sample.shape}")

        # 6. Test REAL model decoding
        decoded = model.decode(sample)
        assert decoded is not None
        print(f"Decoded video shape: {decoded.shape}")
        assert len(decoded.shape) == 5  # (B, C, T, H, W)
        assert decoded.shape[0] == 1  # Batch size
        assert decoded.shape[1] == 3  # Channels

        print("Real inference test passed.")


@RunIf(min_gpus=4)
@pytest.mark.L2
def test_inference_pipeline_multi_chunk_inference():
    """Test the inference pipeline with multiple chunks using tensor inputs.

    Requires 4+ GPUs. Run with:
    torchrun --nproc_per_node=4 -m pytest -s cosmos_predict2/_src/transfer2/inference/inference_pipeline_test.py --L2 -k test_inference_pipeline_multi_chunk_inference
    """
    dtype = torch.uint8

    # Mock only the model loading - use real model for forward pass
    with patch("cosmos_predict2._src.transfer2.inference.inference_pipeline.load_model_from_checkpoint") as mock_load:
        # Create a real model instance for testing with proper config
        from cosmos_predict2._src.predict2.models.video2world_model import Video2WorldModel

        # Create proper config that supports 93 frames
        config = get_test_video2world_config()

        # Create a real model
        model = Video2WorldModel(config)
        model.cuda()
        model.eval()
        model.on_train_start()  # Initialize model properly

        mock_load.return_value = (model, config)

        # Create pipeline (using 93 frames per chunk)
        pipeline = ControlVideo2WorldInference(
            registered_exp_name="test_experiment",
            checkpoint_paths="test_path",
            s3_credential_path="test_cred",
            num_video_frames_per_chunk=93,
            num_conditional_frames=1,
        )

        # Create longer video tensor requiring multiple chunks (121 frames, 93 frames per chunk)
        input_video = create_random_video_tensor((3, 121, 128, 128), dtype)  # 121 frames = 2 chunks
        text_embeddings = create_random_text_embeddings((1, 77, 1024), torch.float32)

        # Test multi-chunk calculations
        num_total_frames, num_chunks, num_frames_per_chunk = pipeline._get_num_chunks(input_video)
        assert num_total_frames == 121
        assert num_chunks == 2  # Should require 2 chunks
        assert num_frames_per_chunk == 92

        # Test that padding doesn't change the video for longer videos
        padded_video = pipeline._pad_input_frames(input_video, num_total_frames)
        assert torch.equal(input_video, padded_video)  # No padding needed for long videos

        # Test data batch creation for different chunk scenarios
        # Chunk 0: frames 0-93
        chunk_0_frames = input_video[:, 0:93]
        prev_output_0 = create_random_video_tensor((3, 93, 128, 128), dtype).unsqueeze(0)
        data_batch_0 = pipeline._get_data_batch_input(
            chunk_0_frames,
            prev_output_0,
            text_embeddings,
            fps=24,
            control_weight="1.0",
        )

        # Chunk 1: frames 92-121 (with overlap)
        chunk_1_frames = input_video[:, 92:121]  # 29 frames
        # Simulate the padding that happens in the actual pipeline
        if chunk_1_frames.shape[1] < 93:
            last_frame = chunk_1_frames[:, -1:, :, :]
            padding = last_frame.repeat(1, 93 - chunk_1_frames.shape[1], 1, 1)
            chunk_1_frames_padded = torch.cat([chunk_1_frames, padding], dim=1)
        else:
            chunk_1_frames_padded = chunk_1_frames

        prev_output_1 = create_random_video_tensor((3, 93, 128, 128), dtype).unsqueeze(0)  # Mock previous output
        data_batch_1 = pipeline._get_data_batch_input(
            chunk_1_frames_padded,
            prev_output_1,
            text_embeddings,
            fps=24,
            control_weight="1.0",
        )

        # Verify all data batches are valid
        for i, batch in enumerate([data_batch_0, data_batch_1]):
            assert "video" in batch, f"Chunk {i} missing video"
            assert "t5_text_embeddings" in batch, f"Chunk {i} missing text embeddings"
            assert "input_video" in batch, f"Chunk {i} missing input_video"
            assert batch["input_video"].shape[1] == 93, f"Chunk {i} input_video wrong size"

        # Mock _maybe_torch_to_numpy to handle CUDA tensors properly
        def mock_maybe_torch_to_numpy(frames):
            """Mock to handle CUDA tensors properly for testing."""
            if hasattr(frames, "cpu"):
                return frames.cpu().numpy()
            elif hasattr(frames, "numpy"):
                return frames.numpy()
            else:
                return np.array(frames)

        # Test REAL model calls for multiple chunks
        generated_chunks = []
        with patch(
            "cosmos_predict2._src.transfer2.datasets.augmentors.control_input._maybe_torch_to_numpy",
            side_effect=mock_maybe_torch_to_numpy,
        ):
            for i in range(2):
                # Use different data batch for each chunk
                current_batch = [data_batch_0, data_batch_1][i]

                sample = model.generate_samples_from_batch(
                    current_batch,
                    n_sample=1,
                    guidance=7,
                    seed=42 + i,
                    is_negative_prompt=False,
                )
                decoded = model.decode(sample)

                print(f"Chunk {i} - Sample shape: {sample.shape}, Decoded shape: {decoded.shape}")
                assert sample is not None
                assert decoded is not None
                assert len(decoded.shape) == 5  # (B, C, T, H, W)
                assert decoded.shape[0] == 1  # Batch size
                assert decoded.shape[1] == 3  # Channels

                generated_chunks.append(decoded)

        # Verify we generated 2 chunks
        assert len(generated_chunks) == 2
        print(f"Successfully generated {len(generated_chunks)} chunks with real model forward pass")

        print("Multi-chunk inference test passed.")


if __name__ == "__main__":
    test_inference_pipeline_get_num_chunks()
    test_inference_pipeline_pad_input_frames()
    test_inference_pipeline_multi_branch_generate_img2world()
    test_inference_pipeline_get_data_batch_input()
    test_inference_pipeline_chunk_calculations()
    test_inference_pipeline_tensor_shapes()
    print("All L1 tests passed!")
