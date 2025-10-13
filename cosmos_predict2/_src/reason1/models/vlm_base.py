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

import os
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.reason1.configs.default.model_config import FSDP2ModelConfig
from cosmos_predict2._src.reason1.parallelisms.optimizer import build_lr_schedulers, build_optimizers
from cosmos_predict2._src.reason1.parallelisms.parallel_dims import ParallelDims
from cosmos_predict2._src.reason1.parallelisms.torchtitan_utilts import device_module, device_type
from cosmos_predict2._src.reason1.tokenizer.processor import Processor
from cosmos_predict2._src.reason1.utils import distributed
from cosmos_predict2._src.reason1.utils.parallelism import broadcast, broadcast_with_shape_check


def find_nonpersistent_zero_buffers(model: torch.nn.Module):
    """
    Loop over all submodules of a model and report buffers
    that are registered with persistent=False and are all zeros.

    Args:
        model (nn.Module): The PyTorch model (e.g. from transformers).

    Returns:
        List of dicts with module name, buffer name, and tensor info.
    """
    results = []
    current_device = torch.cuda.current_device()
    for module_name, module in model.named_modules():
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer is None:
                continue

            # check if it is marked as non-persistent
            if buffer_name in module._non_persistent_buffers_set:
                if hasattr(module, "rope_init_fn"):
                    buffer, module.attention_scaling = module.rope_init_fn(module.config, device=None)
                    module.register_buffer("inv_freq", buffer.to(current_device), persistent=False)
                log.info(f"[Check]buffer_name: [{buffer_name}] in module [{module_name}] is marked as non-persistent")
                # check if tensor is all zeros
                if torch.all(buffer == 0):
                    results.append(
                        {
                            "module": module_name,
                            "buffer": buffer_name,
                            "shape": tuple(buffer.shape),
                            "dtype": buffer.dtype,
                            "device": buffer.device,
                        }
                    )
    assert len(results) == 0, f"nonpersistent_zero_buffers should be empty, but got {results}"
    return results


class VLMBaseModel(torch.nn.Module):
    """
    A class for base VLM model, has the shared methods for all VLM models

    Methods:
        build_model: build the model, should be implemented by each VLM model
        maybe_freeze_pretrained_modules: freeze the pretrained modules
        init_optimizer_scheduler: initialize the optimizer and scheduler
        get_num_params: get the number of parameters in the model
        load_state_dict: load the state dict
        validation_step: validation step
        forward: forward pass, should be implemented by each VLM model
        training_step: training step
        init_weights: initialize the weights, should be implemented by each VLM model
    """

    def __init__(
        self,
        model_config: FSDP2ModelConfig,
        tokenizer: Processor,
    ) -> "AutoRegressiveModel":
        super().__init__()
        """
        Build a AutoRegressiveModel instance by initializing and loading a model checkpoint.

        Args:
            model_config (FSDP2ModelConfig): The model configuration for the AutoRegressiveModel instance.
            tokenizer (Tokenizer): The tokenizer for the AutoRegressiveModel instance.
            download_rank_sync (bool, optional): Whether to download the checkpoint in a rank-synchronized manner. Defaults to True.
        Returns:
            AutoRegressiveModel: An instance of the AutoRegressiveModel class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory.

        Note:
            This method sets the device to CUDA and loads the pre-trained model and tokenizer.
        """
        orig_precision = torch.get_default_dtype()
        precision = getattr(torch, model_config.precision)
        torch.set_default_dtype(precision)
        log.info(f"Setting torch default dtype from {orig_precision} to {precision}")
        self.tokenizer = tokenizer
        self.config = model_config
        self.precision = getattr(torch, model_config.precision)

        self.build_model(model_config)
        torch.set_default_dtype(orig_precision)  # Reset the default dtype to the original value
        log.info(f"Reset torch default dtype to {orig_precision}")

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        """The model preparation before the training is launched

        Args:
            memory_format (torch.memory_format): Memory format of the model.
        """
        pass

    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """Hook before zero_grad() is called.

        Args:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            iteration (int): Current iteration number.
        """
        pass

    def on_after_backward(self, iteration: int = 0) -> None:
        """Hook after loss.backward() is called.

        This method is called immediately after the backward pass, allowing for custom operations
        or modifications to be performed on the gradients before the optimizer step.

        Args:
            iteration (int): Current iteration number.
        """
        pass

    def maybe_freeze_pretrained_modules(self):
        if self.config.freeze_vision_encoder:
            log.info("Freezing vision_encoder")
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        if self.config.freeze_mm_projector:
            log.info("Freezing mm_projector")
            for param in self.mm_projector.parameters():
                param.requires_grad = False
        if self.config.freeze_llm:
            log.info("Freezing llm")
            for param in self.model.parameters():
                param.requires_grad = False
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Print the number in billions, or in the format of 1,000,000,000
        log.info(
            f"Total parameters: {total_params / 1e9:.2f}B, Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}"
        )

    def init_optimizer_scheduler(
        self, optimizer_config, scheduler_config
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:


        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """

        model_parts = []
        lr_multiplier = []
        if not self.config.freeze_vision_encoder and self.vision_encoder is not None:
            log.info(
                f"adding vision_encoder to optimizer, lr_multiplier: {self.config.optimizer.lr_multiplier_vision_encoder}"
            )
            model_parts.append(self.vision_encoder)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_vision_encoder)
        if not self.config.freeze_mm_projector and self.mm_projector is not None:
            log.info(
                f"adding mm_projector to optimizer, lr_multiplier: {self.config.optimizer.lr_multiplier_mm_projector}"
            )
            model_parts.append(self.mm_projector)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_mm_projector)
        if not self.config.freeze_llm:
            log.info(f"adding llm to optimizer, lr_multiplier: {self.config.optimizer.lr_multiplier_llm}")
            model_parts.append(self.model)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_llm)
        optimizers = build_optimizers(model_parts, self.config, lr_multiplier)
        lr_schedulers = build_lr_schedulers(optimizers, self.config)
        return optimizers, lr_schedulers

    def get_num_params(
        self,
    ) -> int:
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False):
        """
        Ignore the missing keys with substrings matching `substring_to_ignore` (e.g., "_extra_state" keys imposed by
        TransformerEngine for FP8).
        """
        actual_missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False, assign=assign)
        if strict:
            if len(actual_missing_keys) > 0 or len(unexpected_keys) > 0:
                raise ValueError(f"Missing keys: {actual_missing_keys}\n\nUnexpected keys: {unexpected_keys}")
        return _IncompatibleKeys(actual_missing_keys, unexpected_keys)

    @torch.no_grad()
    def validation_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Perform a validation step for the model, which is the same as the training step (but without backpropagation).
        """
        return self.training_step(data_batch, iteration)

    def init_weights(
        self,
        buffer_device: Optional[torch.device] = None,
    ):
        if self.config.model_type in ["qwen2_5"]:
            # For general HF model, we apply the nonpersistent zero buffers check and init weights
            find_nonpersistent_zero_buffers(self.model)
        else:
            self.model.init_weights(buffer_device)
        if self.vision_encoder is not None:
            if self.config.vision_encoder.startswith("siglip"):
                pass
            elif self.config.vision_encoder in [
                "internvit-300m-448px-v2.5",
                "internvit-6b-448px-v2.5",
            ]:
                self.vision_encoder.init_weights()
            else:
                self.vision_encoder.init_weights(buffer_device)
        if self.mm_projector is not None:
            self.mm_projector.init_weights()

    @property
    def cp_mesh(self):
        return None

    @property
    def tp_mesh(self):
        return None

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        output_batch = {}
        if iteration < 20:
            summary_str = f"data_batch: {data_batch.keys()}"
            for key in data_batch.keys():
                if isinstance(data_batch[key], torch.Tensor):
                    summary_str += f" | {key} shape: {data_batch[key].shape}, dtype: {data_batch[key].dtype}"
            for key in ["__url__", "__key__", "image_grid_thw", "video_grid_thw"]:
                if key in data_batch:
                    summary_str += f" | {key}: {data_batch[key]}"
            log.info(summary_str, rank0_only=False)

        # first, broadcast if needed
        if self.cp_mesh is not None:
            _broadcast_to_cp_or_tp_ranks(data_batch, self.cp_mesh)
        elif self.tp_mesh is not None:
            _broadcast_to_cp_or_tp_ranks(data_batch, self.tp_mesh)

        # continue training
        tokens = data_batch["tokens"]
        tokens = tokens.to(device="cuda")

        # Token Mask (Note: this is not attention mask)
        token_mask = data_batch.get("token_mask", None)
        apply_token_mask = token_mask is not None

        if token_mask is None:
            token_mask = torch.ones_like(tokens, dtype=torch.bool)
        token_mask = token_mask.to(device="cuda")

        if self.config.aux_loss_coeff > 0:
            logits, aux_loss_list = self(tokens, data_batch, return_aux_loss=True)
            if len(aux_loss_list) > 0:
                assert aux_loss_list[0] is not None
                aux_loss = sum(aux_loss_list)
                output_batch["aux_loss_sum"] = aux_loss
                for i, aux_loss in enumerate(aux_loss_list):
                    output_batch[f"aux_loss_{i}"] = aux_loss
            else:
                aux_loss = None
        else:
            logits = self(tokens, data_batch)
        # For auto-regressive models, the labels are the same as the
        # input tokens shifted by one position
        logits = logits[:, :-1]
        token_mask = token_mask[:, 1:]
        labels = tokens[:, 1:].clone()

        # The PyTorch default ignore_index for the cross-entropy loss is -100.
        ignore_index = -100
        if apply_token_mask:
            labels[~token_mask] = ignore_index
        num_assistant_tokens = token_mask.float().sum()
        current_num_assistant_tokens = token_mask.float().sum()
        batch_size_local = tokens.shape[0]
        batch_size_global = torch.tensor(tokens.shape[0], device=tokens.device)

        dist.all_reduce(num_assistant_tokens, op=dist.ReduceOp.SUM)  # Sum of all num tokens with loss
        dist.all_reduce(batch_size_global, op=dist.ReduceOp.SUM)  # Sum of num of sequences
        avg_num_assistant_tokens = num_assistant_tokens / batch_size_global
        if "padding_mask" in data_batch:
            padding_mask = data_batch["padding_mask"]
            num_real_tokens = (~padding_mask).float().sum()
            dist.all_reduce(num_real_tokens, op=dist.ReduceOp.SUM)  # Sum of all tokens excluding padding
            avg_num_real_tokens = num_real_tokens / batch_size_global
            max_num_real_tokens = (~padding_mask).float().sum(dim=-1).max()
            dist.all_reduce(max_num_real_tokens, op=dist.ReduceOp.MAX)
            min_num_real_tokens = (~padding_mask).float().sum(dim=-1).min()
            dist.all_reduce(min_num_real_tokens, op=dist.ReduceOp.MIN)
        else:
            # No padding mask means all tokens are real tokens
            num_real_tokens = torch.tensor(float(tokens.numel()), device=tokens.device)
            dist.all_reduce(num_real_tokens, op=dist.ReduceOp.SUM)  # Sum of all tokens (no padding)
            avg_num_real_tokens = num_real_tokens / batch_size_global
            max_num_real_tokens = torch.tensor(float(tokens.shape[1]), device=tokens.device)
            dist.all_reduce(max_num_real_tokens, op=dist.ReduceOp.MAX)
            min_num_real_tokens = torch.tensor(float(tokens.shape[1]), device=tokens.device)
            dist.all_reduce(min_num_real_tokens, op=dist.ReduceOp.MIN)

        output_batch.update(
            {
                "encode_tokens": tokens,
                "logits": logits.detach(),
                "labels": labels.detach(),
                "ignore_index": ignore_index,
                "avg_num_assistant_tokens": avg_num_assistant_tokens.detach().item(),
                "avg_num_real_tokens": avg_num_real_tokens.detach().item(),
                "max_num_real_tokens": max_num_real_tokens.detach().item(),
                "min_num_real_tokens": min_num_real_tokens.detach().item(),
                "current_num_assistant_tokens": token_mask.float().sum().detach().item(),
                "batch_size_local": batch_size_local,
            }
        )
        logits = logits.flatten(0, 1)
        labels = labels.flatten(0, 1)

        # Main cross entropy loss
        if self.config.loss_per_token:
            ce_loss = F.cross_entropy(
                input=logits,
                target=labels,
                ignore_index=ignore_index,  # ignore prompt (turn prompt tokens into pad_id here)
                reduction="sum",
            )

            ce_loss = ce_loss / (batch_size_local * avg_num_assistant_tokens).detach()
        else:
            ce_loss = F.cross_entropy(
                input=logits,
                target=labels,
                ignore_index=ignore_index,  # ignore prompt (turn prompt tokens into pad_id here)
            )

        # Z-loss
        if self.config.z_loss_coeff > 0:
            if isinstance(logits, DTensor):
                local_logits = logits.to_local()  # Convert to a local tensor
            else:
                local_logits = logits
            log_z_local = torch.logsumexp(local_logits, dim=-1)

            z_loss_local = self.config.z_loss_coeff * (log_z_local**2).mean()
            if isinstance(ce_loss, DTensor):
                z_loss_dtensor = DTensor.from_local(
                    z_loss_local,
                    device_mesh=ce_loss.device_mesh,  # use the same device mesh as ce_loss
                    placements=ce_loss.placements,  # use the same sharding/placement strategy
                )
            else:
                z_loss_dtensor = z_loss_local
            # Combined loss
            total_loss = ce_loss + z_loss_dtensor
        else:
            total_loss = ce_loss

        output_batch["ce_loss"] = ce_loss
        if self.config.aux_loss_coeff > 0 and aux_loss is not None:
            total_loss += aux_loss * self.config.aux_loss_coeff
        return output_batch, total_loss  # skip returning output logits

    # These methods should be implemented by each VLM model
    def build_model(self, model_config):
        raise NotImplementedError

    def forward(self, tokens, data_batch={}, start_pos: int = 0) -> torch.Tensor:
        """
        The forward pass of the model.
        Returns:
            logits (torch.Tensor): The logits of the model.
        """
        raise NotImplementedError


def _broadcast_to_cp_or_tp_ranks(data_batch: dict[str, torch.Tensor], cp_or_tp_mesh: DeviceMesh) -> bool:
    """Copies tensors in data_batch to the GPU and broadcasts across CP or TP ranks.

    The contents of data_batch are updated with the copied and broadcasted
    tensors. The inputs are replicated across CP ranks. The output logits
    and loss calculations are also replicated across CP ranks.

    Args:
        data_batch: Inputs (tokens, token_mask, images) needed for training.
        cp_or_tp_mesh: The DeviceMesh for context parallelism or tensor parallelism.
    """

    tokens = data_batch.get("tokens")
    data_batch["tokens"] = broadcast_with_shape_check(tokens, cp_or_tp_mesh)

    if "attention_mask" in data_batch:
        attention_mask = data_batch.get("attention_mask")
        data_batch["attention_mask"] = broadcast_with_shape_check(attention_mask, cp_or_tp_mesh)

    # Token Mask (Note: this is not attention mask)
    token_mask = data_batch.get("token_mask", None)
    if token_mask is None:
        token_mask = torch.ones_like(tokens, dtype=torch.bool)
    data_batch["token_mask"] = broadcast_with_shape_check(token_mask, cp_or_tp_mesh)

    if "padding_mask" in data_batch:
        padding_mask = data_batch["padding_mask"]
        data_batch["padding_mask"] = broadcast_with_shape_check(padding_mask, cp_or_tp_mesh)

    # Some rank may not have images, e.g. text data, remove images from all ranks in the group if first rank in the group doesn’t have it, otherwise, create it
    has_images = (
        torch.ones(1, dtype=torch.bool).to(device=tokens.device)
        if "images" in data_batch
        else torch.zeros(1, dtype=torch.bool).to(device=tokens.device)
    )
    has_images = broadcast_with_shape_check(has_images, cp_or_tp_mesh)
    if not has_images and "images" in data_batch:
        del data_batch["images"]
    elif has_images and "images" not in data_batch:
        data_batch["images"] = torch.zeros(1, 1, 3, 448, 448).to(
            device=tokens.device
        )  # randomly init a zero tensor, the shape will be aligned later in broadcast_with_shape_check

    images = data_batch.get("images", None)
    if images is not None:
        data_batch["images"] = broadcast_with_shape_check(images, cp_or_tp_mesh)

    image_grid_thw = data_batch.get("image_grid_thw", None)
    if image_grid_thw is not None:
        data_batch["image_grid_thw"] = broadcast(
            image_grid_thw, cp_or_tp_mesh
        )  # NOTE (maxzhaoshuol): no need to check shape

    videos = data_batch.get("videos", None)
    if videos is not None:
        data_batch["videos"] = broadcast_with_shape_check(videos, cp_or_tp_mesh)

    video_grid_thw = data_batch.get("video_grid_thw", None)
    if video_grid_thw is not None:
        data_batch["video_grid_thw"] = broadcast(
            video_grid_thw, cp_or_tp_mesh
        )  # NOTE (maxzhaoshuol): no need to check shape

    # broadcast the string to all ranks
    for key in ["__url__", "dialog_str", "__key__"]:
        if key not in data_batch:
            data_batch[key] = [f"placeholder_{key}"]
        data_batch[key] = broadcast_object(data_batch[key], cp_or_tp_mesh)
    if "dataset_name" not in data_batch:
        data_batch["dataset_name"] = "default"
    data_batch["dataset_name"] = broadcast_object(data_batch["dataset_name"], cp_or_tp_mesh)
    return


def broadcast_object(local_str: List[str], cp_or_tp_mesh: DeviceMesh):
    """
    Broadcast a string to all ranks.
    """
    group = cp_or_tp_mesh.get_group()
    gathered_list = [None for _ in range(dist.get_world_size(group=group))]
    dist.all_gather_object(gathered_list, local_str, group=group)
    output_str = gathered_list[0]
    return output_str


def init_mesh(model_config):
    world_size = distributed.get_world_size()
    parallel_dims = ParallelDims(
        dp_shard=model_config.training.data_parallel_shard_degree,
        dp_replicate=model_config.training.data_parallel_replicate_degree,
        cp=model_config.training.context_parallel_degree,
        tp=model_config.training.tensor_parallel_degree,
        pp=model_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not model_config.training.disable_loss_parallel,
    )
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    return world_mesh, parallel_dims
