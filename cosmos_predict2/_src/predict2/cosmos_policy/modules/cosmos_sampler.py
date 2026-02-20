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
Extended Sampler for Cosmos Policy with special handling for num_steps.

This sampler extends the base Sampler class to add:
- Adjusted num_steps logic when sample_clean is enabled
- Special case handling for num_steps==1
"""

from typing import Callable, List, Optional

import torch

from cosmos_predict2._src.imaginaire.functional.multi_step import is_multi_step_fn_supported
from cosmos_predict2._src.imaginaire.functional.runge_kutta import is_runge_kutta_fn_supported
from cosmos_predict2._src.imaginaire.modules.res_sampler import (
    Sampler,
    SamplerConfig,
    SolverConfig,
    SolverTimestampConfig,
    differential_equation_solver,
    get_rev_ts,
)


class CosmosPolicySampler(Sampler):
    """
    Extended Sampler for Cosmos Policy.

    Adds special handling for:
    - Adjusting num_steps when sample_clean is enabled (subtracts 1 for num_steps > 1)
    - Special case for num_steps==1 where we directly denoise without the solver loop
    """

    def __init__(self, cfg: Optional[SamplerConfig] = None):
        super().__init__(cfg)

    @torch.no_grad()
    def forward(
        self,
        x0_fn: Callable,
        x_sigma_max: torch.Tensor,
        num_steps: int = 35,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        solver_option: str = "2ab",
    ) -> torch.Tensor:
        in_dtype = x_sigma_max.dtype

        def float64_x0_fn(x_B_StateShape: torch.Tensor, t_B: torch.Tensor) -> torch.Tensor:
            return x0_fn(x_B_StateShape.to(in_dtype), t_B.to(in_dtype)).to(torch.float64)

        is_multistep = is_multi_step_fn_supported(solver_option)
        is_rk = is_runge_kutta_fn_supported(solver_option)
        assert is_multistep or is_rk, f"Only support multistep or Runge-Kutta method, got {solver_option}"

        solver_cfg = SolverConfig(
            s_churn=S_churn,
            s_t_max=S_max,
            s_t_min=S_min,
            s_noise=S_noise,
            is_multi=is_multistep,
            rk=solver_option,
            multistep=solver_option,
        )
        # NOTE (user): If the sampler adds an additional clean step, subtract 1 from num_steps to get correct total # steps
        # Only do this for num_steps > 1 (num_steps==1 is a special case)
        sample_clean = True
        if sample_clean and num_steps > 1:
            num_steps = num_steps - 1
        timestamps_cfg = SolverTimestampConfig(nfe=num_steps, t_min=sigma_min, t_max=sigma_max, order=rho)
        sampler_cfg = SamplerConfig(solver=solver_cfg, timestamps=timestamps_cfg, sample_clean=sample_clean)

        return self._forward_impl(float64_x0_fn, x_sigma_max, sampler_cfg, num_steps=num_steps).to(in_dtype)

    @torch.no_grad()
    def _forward_impl(
        self,
        denoiser_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noisy_input_B_StateShape: torch.Tensor,
        sampler_cfg: Optional[SamplerConfig] = None,
        callback_fns: Optional[List[Callable]] = None,
        num_steps: int = 35,
    ) -> torch.Tensor:
        """
        Internal implementation of the forward pass.

        Args:
            denoiser_fn: Function to denoise the input.
            noisy_input_B_StateShape: Input tensor with noise.
            sampler_cfg: Configuration for the sampler.
            callback_fns: List of callback functions to be called during sampling.
            num_steps: Number of denoising steps.

        Returns:
            torch.Tensor: Denoised output tensor.
        """
        sampler_cfg = self.cfg if sampler_cfg is None else sampler_cfg
        solver_order = 1 if sampler_cfg.solver.is_multi else int(sampler_cfg.solver.rk[0])
        num_timestamps = sampler_cfg.timestamps.nfe // solver_order

        sigmas_L = get_rev_ts(
            sampler_cfg.timestamps.t_min, sampler_cfg.timestamps.t_max, num_timestamps, sampler_cfg.timestamps.order
        ).to(noisy_input_B_StateShape.device)

        if num_steps > 1:
            # Normal sampling
            denoised_output = differential_equation_solver(
                denoiser_fn, sigmas_L, sampler_cfg.solver, callback_fns=callback_fns
            )(noisy_input_B_StateShape)

            if sampler_cfg.sample_clean:
                # Override denoised_output with fully denoised version
                ones = torch.ones(denoised_output.size(0), device=denoised_output.device, dtype=denoised_output.dtype)
                denoised_output = denoiser_fn(denoised_output, sigmas_L[-1] * ones)
        else:
            # Special case: num_steps==1
            denoised_output = noisy_input_B_StateShape
            ones = torch.ones(denoised_output.size(0), device=denoised_output.device, dtype=denoised_output.dtype)
            denoised_output = denoiser_fn(denoised_output, sigmas_L[0] * ones)

        return denoised_output
