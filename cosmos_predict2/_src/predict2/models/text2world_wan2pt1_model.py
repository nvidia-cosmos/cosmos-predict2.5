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

from __future__ import annotations

import collections
from contextlib import contextmanager
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import attrs
import numpy as np
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor
from torch.distributed._composable.fsdp import FSDPModule, fully_shard
from torch.distributed._tensor.api import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.utils.clip_grad import clip_grad_norm_

from cosmos_predict2._src.common.types.denoise_prediction import DenoisePrediction
from cosmos_predict2._src.common.utils.checkpointer import non_strict_load_model
from cosmos_predict2._src.common.utils.count_params import count_params
from cosmos_predict2._src.common.utils.fsdp_helper import hsdp_device_mesh
from cosmos_predict2._src.common.utils.optim_instantiate import get_base_scheduler
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import log, misc
from cosmos_predict2._src.imaginaire.utils.context_parallel import (
    broadcast,
    broadcast_split_tensor,
    cat_outputs_cp,
)
from cosmos_predict2._src.imaginaire.utils.ema import FastEmaModelUpdater
from cosmos_predict2._src.predict2.conditioner import DataType, Text2WorldCondition
from cosmos_predict2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_predict2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
from cosmos_predict2._src.predict2.models.text2world_model import EMAConfig
from cosmos_predict2._src.predict2.networks.model_weights_stats import WeightTrainingStat
from cosmos_predict2._src.predict2.schedulers.rectified_flow import RectifiedFlow
from cosmos_predict2._src.predict2.tokenizers.base_vae import BaseVAE
from cosmos_predict2._src.predict2.utils.dtensor_helper import (
    DTensorFastEmaModelUpdater,
    broadcast_dtensor_model_states,
)

IS_PREPROCESSED_KEY = "is_preprocessed"
NUM_EMBEDDING_PADDING_TOKENS = 512


@attrs.define(slots=False)
class Text2WorldModelWan2pt1Config:
    """
    Config for [DiffusionModel][projects.cosmos.diffusion.v2.models.t2v_model.DiffusionModel].
    """

    tokenizer: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None
    ema: EMAConfig = EMAConfig()

    fsdp_shard_size: int = 1
    precision: str = "bfloat16"
    input_data_key: str = "video"  # key to fetch input data from data_batch
    input_image_key: str = "images"  # key to fetch input image from data_batch
    input_caption_key: str = "ai_caption"  # Key used to fetch input captions
    use_torch_compile: bool = False

    state_ch: int = 16  # for latent model, ref to the latent channel number
    state_t: int = 8  # for latent model, ref to the latent number of frames
    resolution: str = "512"

    shift: int = 5
    use_dynamic_shift: bool = False
    train_time_weight: str = "uniform"


class WANDiffusionModel(ImaginaireModel):
    """
    Diffusion model.
    """

    def __init__(self, config: Text2WorldModelWan2pt1Config):
        super().__init__()

        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        self.flow_matching_kwargs = {"device": "cuda", "dtype": torch.float32}

        log.warning(f"WANDiffusionModel: precision {self.precision}")
        log.warning(f"Flow Matching: precision {self.flow_matching_kwargs['dtype']}")

        # 1. set data keys and data information
        # self.sigma_data = config.sigma_data
        self.setup_data_key()

        # 2. setup up diffusion processing and scaling~(pre-condition), sampler
        self.sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
        )

        # 3. tokenizer
        with misc.timer("WANDiffusionModel: set_up_tokenizer"):
            self.tokenizer: BaseVAE = lazy_instantiate(config.tokenizer)
            assert self.tokenizer.latent_ch == self.config.state_ch, (
                f"latent_ch {self.tokenizer.latent_ch} != state_shape {self.config.state_ch}"
            )

        # 5. create fsdp mesh if needed
        if config.fsdp_shard_size > 1:
            self.fsdp_device_mesh = hsdp_device_mesh(
                sharding_group_size=config.fsdp_shard_size,
            )
        else:
            self.fsdp_device_mesh = None

        # 6. diffusion neural networks part
        self.set_up_model()

        # 7. training states
        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

        # 8. rectified flow
        self.rectified_flow = RectifiedFlow(
            velocity_field=self.net,
            train_time_distribution="logitnormal",
            use_dynamic_shift=config.use_dynamic_shift,
            shift=config.shift,
            train_time_weight_method=config.train_time_weight,
            device=torch.device("cuda"),
            dtype=self.flow_matching_kwargs["dtype"],
        )

    def setup_data_key(self) -> None:
        self.input_data_key = self.config.input_data_key  # by default it is video key for Video diffusion model
        self.input_image_key = self.config.input_image_key

    def build_net(self):
        config = self.config
        init_device = "meta" if self.fsdp_device_mesh else "cpu"
        with misc.timer("Creating PyTorch model"):
            with torch.device(init_device):
                net = lazy_instantiate(config.net)

            self._param_count = count_params(net, verbose=False)

            if self.fsdp_device_mesh:
                net.fully_shard(mesh=self.fsdp_device_mesh)
                net = fully_shard(net, mesh=self.fsdp_device_mesh, reshard_after_forward=True)

            with misc.timer("meta to cuda and broadcast model states"):
                net.to_empty(device="cuda")
                # IMPORTANT: (qsh) model init should not depends on current tensor shape, or it can handle Dtensor shape.
                net.init_weights()

            if self.fsdp_device_mesh:
                broadcast_dtensor_model_states(net, self.fsdp_device_mesh)
                for name, param in net.named_parameters():
                    assert isinstance(param, DTensor), f"param should be DTensor, {name} got {type(param)}"
        return net

    @misc.timer("DiffusionModel: set_up_model")
    def set_up_model(self):
        config = self.config
        with misc.timer("Creating PyTorch model and ema if enabled"):
            self.conditioner = lazy_instantiate(config.conditioner)
            assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, (
                "conditioner should not have learnable parameters"
            )
            self.net = self.build_net()
            self._param_count = count_params(self.net, verbose=False)

            if config.ema.enabled:
                self.net_ema = self.build_net()
                self.net_ema.requires_grad_(False)

                if self.fsdp_device_mesh:
                    self.net_ema_worker = DTensorFastEmaModelUpdater()
                else:
                    self.net_ema_worker = FastEmaModelUpdater()

                s = config.ema.rate
                self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()

                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)
        torch.cuda.empty_cache()

    def apply_fsdp(self, dp_mesh: DeviceMesh) -> None:
        """Apply FSDP to the net and net_ema."""
        # Back-to-back fully_shard calls allow for wrapping submodules and the top-level module.
        self.net.fully_shard(mesh=dp_mesh)
        self.net = fully_shard(self.net, mesh=dp_mesh, reshard_after_forward=True)
        broadcast_dtensor_model_states(self.net, dp_mesh)
        if hasattr(self, "net_ema") and self.net_ema:
            self.net_ema.fully_shard(mesh=dp_mesh)
            self.net_ema = fully_shard(self.net_ema, mesh=dp_mesh, reshard_after_forward=True)
            broadcast_dtensor_model_states(self.net_ema, dp_mesh)
            self.net_ema_worker = DTensorFastEmaModelUpdater()
            # No need to copy weights to EMA when applying FSDP, it is already copied before applying FSDP.

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:
            config_model (ModelConfig): The config object for the model.

        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """
        optimizer = lazy_instantiate(optimizer_config, model=self.net)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        return optimizer, scheduler

    # ------------------------ training hooks ------------------------
    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """
        update the net_ema
        """
        del scheduler, optimizer

        if self.config.ema.enabled:
            # calculate beta for EMA update
            ema_beta = self.ema_beta(iteration)
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.ema.enabled:
            self.net_ema.to(dtype=torch.float32)
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()
        self.net = self.net.to(memory_format=memory_format, **self.tensor_kwargs)

        if hasattr(self.config, "use_torch_compile") and self.config.use_torch_compile:  # compatible with old config
            if torch.__version__ < "2.3":
                log.warning(
                    "torch.compile in Pytorch version older than 2.3 doesn't work well with activation checkpointing.\n"
                    "It's very likely there will be no significant speedup from torch.compile.\n"
                    "Please use at least 24.04 Pytorch container, or imaginaire4:v7 container."
                )
            # Increasing cache size. It's required because of the model size and dynamic input shapes resulting in
            # multiple different triton kernels. For 28 TransformerBlocks, the cache limit of 256 should be enough for
            # up to 9 different input shapes, as 28*9 < 256. If you have more Blocks or input shapes, and you observe
            # graph breaks at each Block (detectable with torch._dynamo.explain) or warnings about
            # exceeding cache limit, you may want to increase this size.
            # Starting with 24.05 Pytorch container, the default value is 256 anyway.
            # You can read more about it in the comments in Pytorch source code under path torch/_dynamo/cache_size.py.
            torch._dynamo.config.accumulated_cache_size_limit = 256
            # dynamic=False means that a separate kernel is created for each shape. It incurs higher compilation costs
            # at initial iterations, but can result in more specialized and efficient kernels.
            # dynamic=True currently throws errors in pytorch 2.3.
            self.net = torch.compile(self.net, dynamic=False, disable=not self.config.use_torch_compile)

    # ------------------------ training ------------------------

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the diffusion model.

        This method is responsible for executing one iteration of the model's training. It involves:
        1. Adding noise to the input data using the SDE process.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: The computed loss for the training step as a PyTorch Tensor.

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        self._update_train_stats(data_batch)
        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)

        # Sample pertubation noise levels and N(0, 1) noises
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.flow_matching_kwargs)
        batch_size = x0_B_C_T_H_W.size()[0]
        t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.flow_matching_kwargs)
        t_B = rearrange(t_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma

        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B = self.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
        )
        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, self.flow_matching_kwargs)
        sigmas = self.rectified_flow.get_sigmas(
            timesteps,
            self.flow_matching_kwargs,
        )
        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")
        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas)

        vt_pred_B_C_T_H_W = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            **condition.to_dict(),
        )

        time_weights_B = self.rectified_flow.train_time_weight(timesteps, self.flow_matching_kwargs)
        per_instance_loss = torch.mean(
            (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2, dim=list(range(1, vt_pred_B_C_T_H_W.dim()))
        )

        loss = torch.mean(time_weights_B * per_instance_loss)
        output_batch = {"edm_loss": loss}

        return output_batch, loss

    @staticmethod
    def get_context_parallel_group():
        if parallel_state.is_initialized():
            return parallel_state.get_context_parallel_group()
        return None

    def broadcast_split_for_model_parallelsim(self, x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T):
        """
        Broadcast and split the input data and condition for model parallelism.
        Currently, we only support context parallelism.
        """
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            x0_B_C_T_H_W = broadcast_split_tensor(x0_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            epsilon_B_C_T_H_W = broadcast_split_tensor(epsilon_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            if sigma_B_T is not None:
                assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
                if sigma_B_T.shape[-1] == 1:  # single sigma is shared across all frames
                    sigma_B_T = broadcast(sigma_B_T, cp_group)
                else:  # different sigma for each frame
                    sigma_B_T = broadcast_split_tensor(sigma_B_T, seq_dim=1, process_group=cp_group)
            if condition is not None:
                condition = condition.broadcast(cp_group)
            self.net.enable_context_parallel(cp_group)
        else:
            self.net.disable_context_parallel()

        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T

    def _update_train_stats(self, data_batch: dict[str, torch.Tensor]) -> None:
        is_image = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image else self.input_data_key
        if isinstance(self.net, WeightTrainingStat):
            if is_image:
                self.net.accum_image_sample_counter += data_batch[input_key].shape[0] * self.data_parallel_size
            else:
                self.net.accum_video_sample_counter += data_batch[input_key].shape[0] * self.data_parallel_size

    # ------------------------ Sampling ------------------------

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        _, x0, _ = self.get_data_and_condition(data_batch)  # we need always process the data batch first.
        is_image_batch = self.is_image_batch(data_batch)

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def x0_fn(noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            cond_v = self.denoise(noise_x, timestep, condition)
            uncond_v = self.denoise(noise_x, timestep, uncondition)
            noise_pred = uncond_v + guidance * (cond_v - uncond_v)
            return noise_pred

        return x0_fn

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            solver_option (str): differential equation solver option, default to "2ab"~(mulitstep solver)
        """

        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        noise = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape),
            torch.float32,
            self.tensor_kwargs["device"],
            seed,
        )

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.sample_scheduler.set_timesteps(num_steps, device=self.tensor_kwargs["device"], shift=shift)

        timesteps = self.sample_scheduler.timesteps

        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)
        latents = noise

        if self.net.is_context_parallel_enabled:
            latents = broadcast_split_tensor(tensor=latents, seq_dim=2, process_group=self.get_context_parallel_group())

        for _, t in enumerate(timesteps):
            latent_model_input = latents
            timestep = [t]

            timestep = torch.stack(timestep)

            noise_pred = x0_fn(latent_model_input, timestep.unsqueeze(0))
            temp_x0 = self.sample_scheduler.step(
                noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            latents = temp_x0.squeeze(0)

        if self.net.is_context_parallel_enabled:
            latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())

        return latents

    @torch.no_grad()
    def validation_step(
        self, data: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        pass

    @torch.no_grad()
    def forward(self, xt, t, condition: Text2WorldCondition):
        pass

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[Tensor, Tensor, Text2WorldCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # Latent state
        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_data_key]
        latent_state = self.encode(raw_state).contiguous().float()

        # Condition
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        return raw_state, latent_state, condition

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        input_key = self.input_data_key if input_key is None else input_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
                assert torch.all((data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)), (
                    f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
                )
            else:
                assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

    def _augment_image_dim_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        input_key = self.input_image_key if input_key is None else input_key
        if input_key in data_batch:
            # Check if the data has already been augmented and avoid re-augmenting
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert data_batch[input_key].shape[2] == 1, (
                    f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                )
                return
            else:
                data_batch[input_key] = rearrange(data_batch[input_key], "b c h w -> b c 1 h w").contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True

    # ------------------ Checkpointing ------------------

    def state_dict(self) -> Dict[str, Any]:
        net_state_dict = self.net.state_dict(prefix="net.")
        if self.config.ema.enabled:
            ema_state_dict = self.net_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)
        return net_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """
        Loads a state dictionary into the model and optionally its EMA counterpart.
        Different from torch strict=False mode, the method will not raise error for unmatched state shape while raise warning.

        Parameters:e
            state_dict (Mapping[str, Any]): A dictionary containing separate state dictionaries for the model and
                                            potentially for an EMA version of the model under the keys 'model' and 'ema', respectively.
            strict (bool, optional): If True, the method will enforce that the keys in the state dict match exactly
                                    those in the model and EMA model (if applicable). Defaults to True.
            assign (bool, optional): If True and in strict mode, will assign the state dictionary directly rather than
                                    matching keys one-by-one. This is typically used when loading parts of state dicts
                                    or using customized loading procedures. Defaults to False.
        """
        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v

        state_dict = _reg_state_dict

        if strict:
            reg_results: _IncompatibleKeys = self.net.load_state_dict(_reg_state_dict, strict=strict, assign=assign)

            if self.config.ema.enabled:
                ema_results: _IncompatibleKeys = self.net_ema.load_state_dict(
                    _ema_state_dict, strict=strict, assign=assign
                )

            return _IncompatibleKeys(
                missing_keys=reg_results.missing_keys + (ema_results.missing_keys if self.config.ema.enabled else []),
                unexpected_keys=reg_results.unexpected_keys
                + (ema_results.unexpected_keys if self.config.ema.enabled else []),
            )
        else:
            log.critical("load model in non-strict mode")
            log.critical(non_strict_load_model(self.net, _reg_state_dict), rank0_only=False)
            if self.config.ema.enabled:
                log.critical("load ema model in non-strict mode")
                log.critical(non_strict_load_model(self.net_ema, _ema_state_dict), rank0_only=False)

    # ------------------ public methods ------------------
    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: The calculated beta value.
        """
        iteration = iteration + self.config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.ema_exp_coefficient + 1)

    def model_param_stats(self) -> Dict[str, int]:
        return {"total_learnable_param_num": self._param_count}

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        """We hanlde two types of data_batch. One comes from a joint_dataloader where "dataset_name" can be used to differenciate image_batch and video_batch.
        Another comes from a dataloader which we by default assumes as video_data for video model training.
        """
        is_image = self.input_image_key in data_batch
        is_video = self.input_data_key in data_batch
        assert is_image != is_video, (
            "Only one of the input_image_key or input_data_key should be present in the data_batch."
        )
        return is_image

    def denoise(
        self, xt_B_C_T_H_W: torch.Tensor, timestep: torch.Tensor, condition: Text2WorldCondition
    ) -> DenoisePrediction:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            timestep (torch.Tensor): The timestep level.
            condition (Text2WorldCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """
        # forward pass through the network
        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=(xt_B_C_T_H_W).to(**self.tensor_kwargs),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=timestep,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        ).float()

        return net_output_B_C_T_H_W

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(state)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(latent)

    def get_video_height_width(self) -> Tuple[int, int]:
        return VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]

    def get_video_latent_height_width(self) -> Tuple[int, int]:
        height, width = VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]
        return height // self.tokenizer.spatial_compression_factor, width // self.tokenizer.spatial_compression_factor

    def get_num_video_latent_frames(self) -> int:
        return self.config.state_t

    @contextmanager
    def ema_scope(self, context=None, is_cpu=False):
        if self.config.ema.enabled:
            # https://github.com/pytorch/pytorch/issues/144289
            for module in self.net.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            self.net_ema_worker.cache(self.net.parameters(), is_cpu=is_cpu)
            self.net_ema_worker.copy_to(src_model=self.net_ema, tgt_model=self.net)
            if context is not None:
                log.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.config.ema.enabled:
                for module in self.net.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()
                self.net_ema_worker.restore(self.net.parameters())
                if context is not None:
                    log.info(f"{context}: Restored training weights")

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ):
        return clip_grad_norm_(
            self.net.parameters(),
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )
