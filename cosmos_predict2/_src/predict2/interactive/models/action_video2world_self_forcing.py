# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""
Action-conditioned Self-forcing DMD2 Distillation (RectifiedFlow) with KVCache rollout
"""

import uuid
from typing import Optional

import attrs
import numpy as np
import torch
from einops import rearrange
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor.api import DTensor

from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_predict2._src.imaginaire.modules.res_sampler import Sampler
from cosmos_predict2._src.imaginaire.utils import log, misc
from cosmos_predict2._src.imaginaire.utils.count_params import count_params
from cosmos_predict2._src.imaginaire.utils.ema import FastEmaModelUpdater
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.action.configs.action_conditioned.conditioner import ActionConditionedCondition
from cosmos_predict2._src.predict2.distill.models.video2world_model_distill_dmd2 import (
    Video2WorldModelDistillationConfigDMD2TrigFlow,
    Video2WorldModelDistillDMD2TrigFlow,
)
from cosmos_predict2._src.predict2.interactive.networks.utils import make_network_kv_cache
from cosmos_predict2._src.predict2.utils.dtensor_helper import broadcast_dtensor_model_states
from cosmos_predict2._src.predict2.utils.kv_cache import KVCacheConfig, VideoSeqPos


@attrs.define(slots=False)
class ActionVideo2WorldModelTrigflowSelfForcingDMD2Config(Video2WorldModelDistillationConfigDMD2TrigFlow):
    cache_frame_size: int = -1  # if -1, will use the same as the video frame size


class ActionVideo2WorldModelTrigflowSelfForcingDMD2(Video2WorldModelDistillDMD2TrigFlow):
    def __init__(self, config: ActionVideo2WorldModelTrigflowSelfForcingDMD2Config):
        super().__init__(config)
        # Latest decoded video for visualization callbacks
        self.latest_backward_simulation_video = None

    def is_image_batch(self, data_batch: dict) -> bool:
        """Always returns False (video batch) since we're processing video sequences."""
        return False

    def _update_train_stats(self, data_batch: dict[str, torch.Tensor]) -> None:
        self.net.accum_video_sample_counter += self.data_parallel_size

    # modified from distillation_base_mixin.p
    # to enable no_fsdp mode for causal student net
    def build_net(self, net_config_dict, no_fsdp=False):
        init_device = "meta"
        with misc.timer("Creating PyTorch model"):
            with torch.device(init_device):
                net = lazy_instantiate(net_config_dict)

            self._param_count = count_params(net, verbose=False)

            if self.fsdp_device_mesh and not no_fsdp:
                net.fully_shard(mesh=self.fsdp_device_mesh)
                net = fully_shard(net, mesh=self.fsdp_device_mesh, reshard_after_forward=True)

            with misc.timer("meta to cuda and broadcast model states"):
                net.to_empty(device="cuda")
                # IMPORTANT: model init should not depend on current tensor shape, or it can handle DTensor shape.
                net.init_weights()

            if self.fsdp_device_mesh and not no_fsdp:
                # recall model weight init; be careful for buffers!
                broadcast_dtensor_model_states(net, self.fsdp_device_mesh)
                for name, param in net.named_parameters():
                    assert isinstance(param, DTensor), f"param should be DTensor, {name} got {type(param)}"
        return net

    # modified from distillation_base_mixin.py
    # to enable no_fsdp mode for causal student net
    # and not load student weight (load from checkpointer load_path)
    @misc.timer("ActionVideo2WorldModelTrigflowSelfForcingDMD2: set_up_model")
    def set_up_model(self):
        config = self.config
        with misc.timer("Creating PyTorch model and ema if enabled"):
            # Initialize sampler after Module.__init__ to avoid early module assignment.
            if self.sampler is None:
                self.sampler = Sampler()
            self.conditioner = lazy_instantiate(config.conditioner)
            assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, (
                "conditioner should not have learnable parameters"
            )

            assert config.teacher_load_from.load_path, (
                "A pretrained teacher model checkpoint is required for distillation"
            )

            self.net_teacher = self.build_net(config.net_teacher)
            if self.config.init_student_with_teacher:
                log.info("==========Loading teacher checkpoint to TEACHER net (load teacher weight)==========")
                self.load_ckpt_to_net(self.net_teacher, config.teacher_load_from.load_path)

            self.net = self.build_net(config.net, no_fsdp=True)
            log.info("==========Loading student checkpoint to STUDENT net (no weight; no fsdp)==========")

            # fake score net for approximating score func of the student generator output
            if config.net_fake_score:
                # init fake score net with the teacher score func (teacher model)
                self.net_fake_score = self.build_net(config.net_fake_score)
                if self.config.init_student_with_teacher:
                    log.info("==========Loading teacher net weights to FAKE SCORE net (load teacher weight)==========")
                    to_load = {k: v for k, v in self.net_teacher.state_dict().items() if not k.endswith("_extra_state")}
                    res = self.net_fake_score.load_state_dict(to_load, strict=False)
                    missing = [k for k in res.missing_keys if not k.endswith("_extra_state")]
                    unexpected = [k for k in res.unexpected_keys if not k.endswith("_extra_state")]
                    if missing or unexpected:
                        log.warning(f"!!!!!!!!!!!!!!!!!Missing: {missing[:10]}, Unexpected: {unexpected}")
                    if not missing and not unexpected:
                        log.info("==========teacher -> fake score: All keys matched successfully.")
                assert self.loss_scale_sid > 0 or self.loss_scale_GAN_generator > 0
            else:
                self.net_fake_score = None

            # discriminator
            if config.net_discriminator_head:
                self.net_discriminator_head = self.build_net(config.net_discriminator_head)

                # assert self.loss_scale_GAN_generator > 0
                assert config.net_fake_score
                # assert self.net_discriminator_head.model_channels == self.net_fake_score.model_channels
                assert config.intermediate_feature_ids
                assert self.net_discriminator_head.num_branches == len(config.intermediate_feature_ids)
            else:
                self.net_discriminator_head = None

            # freeze models
            if self.net.use_crossattn_projection and self.disable_proj_grad:
                log.info("Freezing the CR1 embedding projection layer in student net..")
                self.net.crossattn_proj.requires_grad_(False)

            if self.net_fake_score and self.net_fake_score.use_crossattn_projection and self.disable_proj_grad:
                log.info("Freezing the CR1 embedding projection layer in fake score net..")
                self.net_fake_score.crossattn_proj.requires_grad_(False)

            log.info("Freezing teacher net..")
            self.net_teacher.requires_grad_(False)

            self._param_count = count_params(self.net, verbose=False)

            # create ema model
            if config.ema.enabled:
                self.net_ema = self.build_net(config.net, no_fsdp=True)
                self.net_ema.requires_grad_(False)

                self.net_ema_worker = FastEmaModelUpdater()

                s = config.ema.rate
                self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()

                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)
        torch.cuda.empty_cache()

    def denoise_edm_seq(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: ActionConditionedCondition,
        video_pos: VideoSeqPos,
        kv_cache_cfg: Optional[KVCacheConfig] = None,
    ) -> torch.Tensor:
        B, C, _, H, W = x_B_C_T_H_W.shape

        assert timesteps_B_T.ndim == 2, f"time shape {timesteps_B_T.shape} is not supported"
        time_B_1_T_1_1 = rearrange(timesteps_B_T, "b t -> b 1 t 1 1")

        # replace the noise level of the cond frames tox_B_C_T_H_W be the pre-defined conditional noise level (very low)
        # the scaling coefficients computed later will inherit the setting.
        condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(x_B_C_T_H_W)
        condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True).type_as(time_B_1_T_1_1)

        t_cond = torch.atan(torch.ones_like(time_B_1_T_1_1) * (self.config.sigma_conditional / self.sigma_data))
        time_B_1_T_1_1 = t_cond * condition_video_mask_B_1_T_1_1 + time_B_1_T_1_1 * (1 - condition_video_mask_B_1_T_1_1)

        # convert noise level time to EDM-formulation coefficients
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling_from_time(time_B_1_T_1_1)

        # EDM preconditioning
        net_state_in_B_C_T_H_W = x_B_C_T_H_W * c_in_B_1_T_1_1

        # forward pass through the network
        net_output_B_C_T_H_W = (
            self.net.forward_seq(
                x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(**self.tensor_kwargs),
                video_pos=video_pos,
                timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
                kv_cache_cfg=kv_cache_cfg,
                **condition.to_dict(),
            )
            .float()
            .to(dtype=x_B_C_T_H_W.dtype)
        )

        # EDM reconstruction of x0
        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * x_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W
        return x0_pred_B_C_T_H_W

    def backward_simulation(
        self,
        condition: ActionConditionedCondition,
        init_noise: torch.Tensor,
        n_steps: int,
        with_grad: bool = False,
        dump_iter: int | None = None,
    ) -> torch.Tensor:
        """Few-step causal AR student with KV cache sampling including actions.

        Delegates to generate_streaming_video with optional gradient on the last hop.
        """
        # Execute the unified path
        output_latents = self.generate_streaming_video(
            condition=condition,
            init_noise=init_noise,
            n_steps=n_steps,
            cache_frame_size=self.config.cache_frame_size,
            enable_grad_on_last_hop=with_grad,
            use_cuda_graphs=False,
        )

        if dump_iter is not None:
            # IMPORTANT: never keep grad graphs alive for debug visualization.
            # `output_latents` may carry a graph when `with_grad=True` (student phase).
            with torch.no_grad():
                video = self.decode(output_latents.detach())
            video = video.detach().cpu()
            uid = uuid.uuid4()
            save_img_or_video((1.0 + video[0]) / 2, f"out-{dump_iter:06d}-{uid}", fps=10)
            # Expose for interactive wandb callbacks
            self.latest_backward_simulation_video = video

        return output_latents

    @torch.no_grad()
    def generate_next_frame(
        self,
        condition,
        frame_noise: torch.Tensor,
        t_idx: int,
        start_idx: int,
        *,
        full_video_pos: VideoSeqPos,
        n_steps: int,
        enable_grad_on_last_hop: bool = False,
    ) -> torch.Tensor:
        assert n_steps >= 1
        t_steps = self.config.selected_sampling_time
        K = len(t_steps)
        assert 1 <= n_steps <= K
        B = frame_noise.shape[0]

        cur_video_pos = full_video_pos.frame(t_idx)
        kv_cache_cfg = KVCacheConfig(run_with_kv=True, store_kv=False, current_idx=t_idx)

        A = self.net._num_action_per_latent_frame

        frame_seq = frame_noise
        x0_pred_last = None
        for s_idx in range(n_steps):
            t_cur = t_steps[s_idx]
            t_tensor = torch.full((B, 1), float(t_cur), device=frame_noise.device, dtype=torch.bfloat16)

            is_final_hop = s_idx == n_steps - 1
            grad_ctx = torch.enable_grad if (enable_grad_on_last_hop and is_final_hop) else torch.no_grad
            with grad_ctx():
                condition_dict = condition.to_dict()
                zero_mask = condition.condition_video_input_mask_B_C_T_H_W[:, :, t_idx : t_idx + 1]  # should be zeros
                action_frame = condition.action[:, (t_idx - start_idx) * A : (t_idx - start_idx + 1) * A]
                condition_dict.update(
                    action=action_frame,
                    condition_video_input_mask_B_C_T_H_W=zero_mask,
                )
                condition_frame = ActionConditionedCondition(**condition_dict)
                x0_pred = self.denoise_edm_seq(
                    x_B_C_T_H_W=frame_seq,
                    timesteps_B_T=t_tensor,
                    condition=condition_frame,
                    video_pos=cur_video_pos,
                    kv_cache_cfg=kv_cache_cfg,
                )

            if s_idx == n_steps - 1:
                x0_pred_last = x0_pred
            else:
                t_next = t_steps[s_idx + 1]
                t_next_tensor = torch.tensor(t_next, device=frame_noise.device, dtype=torch.bfloat16)
                frame_seq = (
                    torch.cos(t_next_tensor) * x0_pred / self.sigma_data + torch.sin(t_next_tensor) * frame_noise
                )

        assert x0_pred_last is not None
        return x0_pred_last

    def generate_streaming_video(
        self,
        condition: ActionConditionedCondition,
        init_noise: torch.Tensor,
        n_steps: int,
        cache_frame_size: int = -1,
        enable_grad_on_last_hop: bool = False,
        use_cuda_graphs: bool = True,
    ) -> torch.Tensor:
        t_steps = self.config.selected_sampling_time
        K = len(t_steps)
        assert 1 <= n_steps <= K

        init_noise = init_noise.to(**self.tensor_kwargs)

        B, C, T, H, W = init_noise.shape
        start_idx = 1

        initial_latent = condition.gt_frames[:, :, :start_idx].clone()
        output_latents = torch.zeros([B, C, T, H, W], device=init_noise.device, dtype=init_noise.dtype)
        output_latents[:, :, :start_idx] = initial_latent

        token_h = H // self.net.patch_spatial
        token_w = W // self.net.patch_spatial

        max_cache_size = T if cache_frame_size == -1 else cache_frame_size
        make_network_kv_cache(self.net, max_cache_size=max_cache_size)

        # Pre-capture CUDA graphs for each frame in advance
        if self.net.use_cuda_graphs and use_cuda_graphs:
            self.net.precapture_cuda_graphs(
                batch_size=B,
                max_t=T,
                token_h=token_h,
                token_w=token_w,
                N_ctx=condition.crossattn_emb.shape[1],
                dtype=self.tensor_kwargs.get("dtype", init_noise.dtype),
                device=init_noise.device,
            )
        full_video_pos = VideoSeqPos(T=T, H=token_h, W=token_w)

        A = self.net._num_action_per_latent_frame

        if start_idx > 0:
            for f in range(start_idx):
                cur_video_pos = full_video_pos.frame(f)
                cur_frame = initial_latent[:, :, f : f + 1]
                kv_cache_cfg_prefill = KVCacheConfig(run_with_kv=True, store_kv=True, current_idx=f)

                condition_dict = condition.to_dict()
                zero_mask = condition.condition_video_input_mask_B_C_T_H_W[:, :, f : f + 1]  # should be zeros
                condition_dict.update(
                    action=None,
                    condition_video_input_mask_B_C_T_H_W=zero_mask,
                )
                condition_frame = ActionConditionedCondition(**condition_dict)
                _ = self.denoise_edm_seq(
                    x_B_C_T_H_W=cur_frame,
                    timesteps_B_T=torch.zeros(B, 1, device=init_noise.device, dtype=self.tensor_kwargs["dtype"]),
                    condition=condition_frame,
                    video_pos=cur_video_pos,
                    kv_cache_cfg=kv_cache_cfg_prefill,
                )

        for t_idx in range(start_idx, T):
            frame_noise = init_noise[:, :, t_idx : t_idx + 1]
            x0_pred_last = self.generate_next_frame(
                condition,
                frame_noise,
                t_idx,
                start_idx,
                full_video_pos=full_video_pos,
                n_steps=n_steps,
                enable_grad_on_last_hop=enable_grad_on_last_hop,
            )

            # Commit the newly generated frame
            output_latents[:, :, t_idx : t_idx + 1] = x0_pred_last

            # Prefill KV cache with the clean generated frame for future steps
            cur_video_pos = full_video_pos.frame(t_idx)
            kv_cache_cfg_prefill = KVCacheConfig(run_with_kv=True, store_kv=True, current_idx=t_idx)

            condition_dict = condition.to_dict()
            zero_mask = condition.condition_video_input_mask_B_C_T_H_W[:, :, t_idx : t_idx + 1]  # should be zeros
            action_frame = condition.action[:, (t_idx - start_idx) * A : (t_idx - start_idx + 1) * A]
            condition_dict.update(
                action=action_frame,
                condition_video_input_mask_B_C_T_H_W=zero_mask,
            )
            condition_frame = ActionConditionedCondition(**condition_dict)
            with torch.no_grad():
                _ = self.denoise_edm_seq(
                    x_B_C_T_H_W=x0_pred_last,
                    timesteps_B_T=torch.zeros(B, 1, device=init_noise.device, dtype=self.tensor_kwargs["dtype"]),
                    condition=condition_frame,
                    video_pos=cur_video_pos,
                    kv_cache_cfg=kv_cache_cfg_prefill,
                )

        return output_latents
