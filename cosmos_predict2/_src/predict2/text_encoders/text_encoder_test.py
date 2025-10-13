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
Unit tests for the TextEncoder class.

Usage:
    pytest -s cosmos_predict2/_src/predict2/text_encoders/text_encoder_test.py --L0
    pytest -s cosmos_predict2/_src/predict2/text_encoders/text_encoder_test.py --L1
"""

import unittest

import pytest
import torch

from cosmos_predict2._src.common.types.embedding_concat_strategy import EmbeddingConcatStrategy
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder, TextEncoderConfig


class TestTextEncoder(unittest.TestCase):
    """Test the TextEncoder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TextEncoderConfig(
            compute_online=True,
            embedding_concat_strategy=str(EmbeddingConcatStrategy.MEAN_POOLING),
            n_layers_per_group=2,
        )

    @pytest.mark.L0
    def test_mean_normalize(self):
        """Test the mean_normalize static method."""
        # Create a test tensor
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Apply mean normalization
        normalized = TextEncoder.mean_normalize(tensor)

        # Check that the result has the same shape
        assert normalized.shape == tensor.shape

        # Check that each row has mean close to 0 and std close to 1
        for i in range(tensor.shape[0]):
            assert abs(normalized[i].mean().item()) < 1e-6
            assert abs(normalized[i].std().item() - 1.0) < 1e-6

    @pytest.mark.L1
    def test_compute_text_embeddings_online_full_concat(self):
        """Test text embedding computation with FULL_CONCAT strategy."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create encoder with FULL_CONCAT strategy
        config = TextEncoderConfig(
            compute_online=True,
            embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
            n_layers_per_group=2,
        )
        encoder = TextEncoder(config)

        # Test data
        data_batch = {"input_caption": ["A beautiful sunset", "A cat playing"]}

        # Compute embeddings
        embeddings = encoder.compute_text_embeddings_online(data_batch, "input_caption")

        log.info(f"Embeddings shape: {embeddings.shape}")

        # Verify the output is a tensor
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.dim() == 3  # [batch_size, seq_len, hidden_dim]
        assert embeddings.shape[0] == 2  # batch_size
        assert embeddings.shape[1] == 512  # sequence length
        assert embeddings.shape[2] == 3584 * 28  # hidden dimension (num_layers * hidden_dim)

    @pytest.mark.L0
    def test_config_defaults(self):
        """Test TextEncoderConfig default values."""
        config = TextEncoderConfig()

        assert config.compute_online is False
        assert config.embedding_concat_strategy == str(EmbeddingConcatStrategy.MEAN_POOLING)
        assert config.n_layers_per_group == 5
        assert "s3://bucket/cosmos_reasoning1" in config.ckpt_path
        assert config.model_config is not None


if __name__ == "__main__":
    # Set up test environment
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Run tests
    unittest.main(verbosity=2)
