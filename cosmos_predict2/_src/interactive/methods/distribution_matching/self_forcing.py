# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""
Self-forcing DMD2 Distillation (RectifiedFlow) with KVCache rollout
"""

import collections
import uuid

import attrs
import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor.api import DTensor
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_predict2._src.imaginaire.modules.res_sampler import Sampler
from cosmos_predict2._src.imaginaire.utils import log, misc
from cosmos_predict2._src.imaginaire.utils.count_params import count_params
from cosmos_predict2._src.imaginaire.utils.ema import FastEmaModelUpdater
from cosmos_predict2._src.imaginaire.utils.high_sigma_strategy import HighSigmaStrategy as HighSigmaStrategy
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.interactive.configs.method_configs.config_dmd2 import DMD2Config
from cosmos_predict2._src.interactive.methods.distribution_matching.dmd2 import DMD2Model
from cosmos_predict2._src.interactive.utils.model_loader import get_storage_reader
from cosmos_predict2._src.predict2.action.configs.action_conditioned.conditioner import ActionConditionedCondition
from cosmos_predict2._src.predict2.modules.denoiser_scaling import (
    EDM_sCMWrapper,
    RectifiedFlow_sCMWrapper,
)
from cosmos_predict2._src.predict2.tokenizers.base_vae import BaseVAE
from cosmos_predict2._src.predict2.utils.dtensor_helper import broadcast_dtensor_model_states


@attrs.define(slots=False)
class SelfForcingModelConfig(DMD2Config):
    # Number of frames cached for rollout/simulation. (if -1, defaults to the full video frame size)
    cache_frame_size: int = -1

    init_student_with_teacher: bool = True
    disable_proj_grad: bool = True
    resize_online: bool = False
    vis_debug_every_n: int = 100


class SelfForcingModel(DMD2Model):
    # ------------------------ Initialization & configuration ------------------------
    def __init__(self, config: SelfForcingModelConfig):
        super().__init__(config)
        # Latest decoded video for visualization callbacks
        self.latest_backward_simulation_video = None
        self.neg_embed = None
        self.condition_postprocessor = None
        self.scaling_from_time = (
            EDM_sCMWrapper(config.sigma_data)
            if config.scaling == "edm"
            else RectifiedFlow_sCMWrapper(config.sigma_data)
        )

    def is_image_batch(self, data_batch: dict) -> bool:
        """Always returns False (video batch) since we're processing video sequences."""
        return False

    def _update_train_stats(self, data_batch: dict[str, torch.Tensor]) -> None:
        self.net.accum_video_sample_counter += self.data_parallel_size

    def _load_ckpt_to_net(
        self,
        net: torch.nn.Module,
        ckpt_path: str,
        prefix: str = "net_ema",
        credential_path: str | None = None,
    ) -> None:
        """Load a DCP checkpoint into a single network."""
        if (
            credential_path is None
            and hasattr(self.config, "teacher_load_from")
            and self.config.teacher_load_from is not None
        ):
            credential_path = self.config.teacher_load_from.credentials

        storage_reader = get_storage_reader(ckpt_path, credential_path)
        if ckpt_path.endswith(".dcp/model"):
            prefix = "net"
        _state_dict = get_model_state_dict(net)

        metadata = storage_reader.read_metadata()
        checkpoint_keys = metadata.state_dict_metadata.keys()

        model_keys = set(_state_dict.keys())

        # Add the prefix to the model keys for comparison
        prefixed_model_keys = {f"{prefix}.{k}" for k in model_keys}

        missing_keys = prefixed_model_keys - checkpoint_keys
        if missing_keys:
            log.warning(f"Missing keys in checkpoint: {missing_keys}")

        unexpected_keys = checkpoint_keys - prefixed_model_keys
        assert prefix in ["net", "net_ema"], "prefix must be either net or net_ema"
        # if load "net_ema." keys, those starting with "net." are fine to ignore in the checkpoint
        if prefix == "net_ema":
            unexpected_keys = [k for k in unexpected_keys if "net." not in k]
        else:
            unexpected_keys = [k for k in unexpected_keys if "net_ema." not in k]
        log.warning("Ignoring _extra_state keys..")
        unexpected_keys = [k for k in unexpected_keys if "_extra_state" not in k]
        if unexpected_keys:
            log.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        if not missing_keys and not unexpected_keys:
            log.info("All keys matched successfully.")

        _new_state_dict = collections.OrderedDict()
        for k in _state_dict.keys():
            _new_state_dict[f"{prefix}.{k}"] = _state_dict[k]
        dcp.load(_new_state_dict, storage_reader=storage_reader, planner=DefaultLoadPlanner(allow_partial_load=True))
        for k in _state_dict.keys():
            _state_dict[k] = _new_state_dict[f"{prefix}.{k}"]

        log.info(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=False)))
        del _state_dict, _new_state_dict

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

    # to enable no_fsdp mode for causal student net
    # and not load student weight (load from checkpointer load_path)
    @misc.timer("SelfForcingModel: build_model")
    def build_model(self):
        config = self.config
        with misc.timer("Creating PyTorch model and ema if enabled"):
            # Initialize sampler after Module.__init__ to avoid early module assignment.
            self.sampler = Sampler()
            self.conditioner = lazy_instantiate(config.conditioner)
            assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, (
                "conditioner should not have learnable parameters"
            )

            # Tokenizer
            self.tokenizer: BaseVAE = lazy_instantiate(self.config.tokenizer)  # type: ignore
            assert self.tokenizer.latent_ch == self.config.state_ch, (
                f"latent_ch {self.tokenizer.latent_ch} != state_shape {self.config.state_ch}"
            )

            assert config.teacher_load_from.load_path, (
                "A pretrained teacher model checkpoint is required for distillation"
            )

            self.net_teacher = self.build_net(config.net_teacher)
            if self.config.init_student_with_teacher:
                log.info("==========Loading teacher checkpoint to TEACHER net (load teacher weight)==========")
                self._load_ckpt_to_net(
                    self.net_teacher,
                    self.config.teacher_load_from.load_path,
                    credential_path=self.config.teacher_load_from.credentials,
                )

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
                assert self.config.loss_scale_sid > 0 or self.config.loss_scale_GAN_generator > 0
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
            if self.net.use_crossattn_projection and self.config.disable_proj_grad:
                log.info("Freezing the CR1 embedding projection layer in student net..")
                self.net.crossattn_proj.requires_grad_(False)

            if self.net_fake_score and self.net_fake_score.use_crossattn_projection and self.config.disable_proj_grad:
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

        self.denoiser_nets = {
            "teacher": self.net_teacher,
            "fake_score": self.net_fake_score,
            "student": self.net,
        }

        torch.cuda.empty_cache()

    def backward_simulation(
        self,
        condition: ActionConditionedCondition,
        init_noise: torch.Tensor,
        n_steps: int,
        with_grad: bool = False,
        dump_iter: int | None = None,
    ) -> torch.Tensor:
        """Few-step causal AR student with KV cache sampling.

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
