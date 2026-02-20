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

import numpy as np
import torch

from cosmos_predict2._src.imaginaire.modules.edm_sde import EDMSDE


class HybridEDMSDE(EDMSDE):
    """
    Extended EDMSDE for Cosmos Policy that supports hybrid sigma distribution.

    This class adds a hybrid sampling strategy that combines:
    - 70% samples from the original log-normal distribution
    - 30% samples from a uniform distribution over higher sigma values

    This approach helps put more weight on higher sigma values during training.
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        hybrid_sigma_distribution: bool = False,
        uniform_lower: float = 1.0,  # Lower bound of uniform distribution from which to sample higher sigma values
        uniform_upper: float = 85.0,  # Upper bound of uniform distribution from which to sample higher sigma values
    ):
        super().__init__(p_mean=p_mean, p_std=p_std, sigma_max=sigma_max, sigma_min=sigma_min)
        self.hybrid_sigma_distribution = hybrid_sigma_distribution
        self.uniform_lower = uniform_lower
        self.uniform_upper = uniform_upper

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps (sigma values) for training.

        When hybrid_sigma_distribution is enabled, uses a mixture of:
        - Log-normal distribution (70% of samples)
        - Uniform distribution between uniform_lower and uniform_upper (30% of samples)

        Otherwise, uses the base class implementation (pure log-normal).

        Args:
            batch_size: Number of samples to generate

        Returns:
            Tensor of sigma values with shape (batch_size,)
        """
        # Hybrid distribution of sigma that puts more weight on higher values
        if self.hybrid_sigma_distribution:
            samples = torch.zeros(batch_size, device="cuda")

            # Generate uniform random values to decide which distribution to sample from
            # 70/30 split: Sample from original log-normal distribution (p=0.7) or uniform distribution (p=0.3)
            distribution_choice = torch.rand(batch_size, device="cuda") < 0.7

            # Count number of samples to draw from each distribution
            num_lognormal = distribution_choice.sum().item()
            num_uniform = batch_size - num_lognormal

            # Sample from original log-normal distribution (70% of samples)
            if num_lognormal > 0:
                cdf_vals = np.random.uniform(size=(num_lognormal))
                log_sigmas = torch.tensor([self.gaussian_dist.inv_cdf(cdf_val) for cdf_val in cdf_vals], device="cuda")
                samples[distribution_choice] = torch.exp(log_sigmas)

            # Sample from uniform distribution (30% of samples)
            if num_uniform > 0:
                # Uniform between uniform_lower and uniform_upper
                samples[~distribution_choice] = torch.empty(num_uniform, device="cuda").uniform_(
                    self.uniform_lower, self.uniform_upper
                )

            return samples

        # Fall back to base class implementation for standard log-normal sampling
        return super().sample_t(batch_size)
