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

from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.reason1.configs.default.model_config import FSDP2ModelConfig
from cosmos_predict2._src.reason1.models.vlm_base import VLMBaseModel, init_mesh
from cosmos_predict2._src.reason1.networks.qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLModel
from cosmos_predict2._src.reason1.networks.qwen2_5_vl import get_rope_index as get_rope_index_v2_5
from cosmos_predict2._src.reason1.networks.qwen2_vl import Qwen2VisionTransformerPretrainedModel, Qwen2VLModel
from cosmos_predict2._src.reason1.networks.qwen2_vl import get_rope_index as get_rope_index_v2
from cosmos_predict2._src.reason1.parallelisms.optimizer import build_lr_schedulers, build_optimizers
from cosmos_predict2._src.reason1.parallelisms.parallelize_qwen import parallelize_qwen
from cosmos_predict2._src.reason1.tokenizer.processor import Processor

try:
    from torch.distributed.tensor import Shard
except ImportError:
    print("torch.distributed.tensor is not available. DeepSeek model will not work.")

from transformers import AutoConfig, Qwen2Model


class QwenModel(VLMBaseModel):
    """
    A class to build and use a AutoRegressiveModel model for text generation.
    This class is mimicing Qwen2_5_VLForConditionalGenerationSimple

    Methods:
        generate: Generate text sequences based on provided prompts using the language generation model.
    """

    def __init__(
        self,
        model_config: FSDP2ModelConfig,
        tokenizer: Processor,
    ) -> "QwenModel":
        super().__init__(model_config, tokenizer)
        self.forward_time = []

    def build_model(self, model_config):
        if model_config.model_type == "qwen2_5_vl":
            self.visual = Qwen2_5_VisionTransformerPretrainedModel(model_config.vision_config)
            self.model = Qwen2_5_VLModel(model_config)
        elif model_config.model_type == "qwen2_vl":
            self.visual = Qwen2VisionTransformerPretrainedModel(model_config.vision_config)
            self.model = Qwen2VLModel(model_config)
        elif model_config.model_type == "qwen2_5":
            self.visual = None
            config = AutoConfig.from_pretrained(
                model_config.name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
            self.model = Qwen2Model(config)
            model_config.hidden_size = config.hidden_size
            model_config.vocab_size = config.vocab_size
            self.model.set_cp_mesh = lambda x: None
            self.model.cp_mesh = None
        else:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")
        self.vocab_size = model_config.vocab_size
        self.lm_head = nn.Linear(model_config.hidden_size, model_config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here]

        if torch.distributed.is_initialized():
            self.world_mesh, self.parallel_dims = init_mesh(model_config)
            parallelize_qwen(self, self.world_mesh, self.parallel_dims, model_config)
            self.model.set_cp_mesh(self.cp_mesh)

    @property
    def vision_encoder(self):
        # This is to be compatible with VLMBaseModel
        return self.visual

    @property
    def mm_projector(self):
        # This is to be compatible with VLMBaseModel
        if self.vision_encoder is not None:
            return self.visual.merger
        else:
            return None

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
        model_part_names = []
        lr_multiplier = []
        if not self.config.freeze_vision_encoder and self.vision_encoder is not None:
            log.info(
                f"adding vision_encoder to optimizer, lr_multiplier: {self.config.optimizer.lr_multiplier_vision_encoder}"
            )
            model_parts.append(self.visual.patch_embed)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_vision_encoder)
            model_part_names.append("visual.patch_embed")
            model_parts.append(self.visual.blocks)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_vision_encoder)
            model_part_names.append("visual.blocks")
        if not self.config.freeze_mm_projector and self.mm_projector is not None:
            log.info(
                f"adding mm_projector to optimizer, lr_multiplier: {self.config.optimizer.lr_multiplier_mm_projector}"
            )
            model_parts.append(self.visual.merger)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_mm_projector)
            model_part_names.append("visual.merger")
        if not self.config.freeze_llm:
            log.info(f"adding llm to optimizer, lr_multiplier: {self.config.optimizer.lr_multiplier_llm}")
            model_parts.append(self.model)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_llm)
            model_part_names.append("llm")

            model_parts.append(self.lm_head)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_llm)
            model_part_names.append("llm")
        optimizers = build_optimizers(model_parts, self.config, lr_multiplier, model_part_names)
        lr_schedulers = build_lr_schedulers(optimizers, self.config)
        return optimizers, lr_schedulers

    def maybe_freeze_pretrained_modules(self):
        if self.config.freeze_vision_encoder:
            log.info("Freezing vision_encoder")
            for param in self.visual.patch_embed.parameters():
                param.requires_grad = False
            for param in self.visual.blocks.parameters():
                param.requires_grad = False
        if self.config.freeze_mm_projector:
            log.info("Freezing mm_projector")
            for param in self.visual.merger.parameters():
                param.requires_grad = False
        if self.config.freeze_llm:
            log.info("Freezing llm")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Print the number in billions, or in the format of 1,000,000,000
        log.info(
            f"Total parameters: {total_params / 1e9:.2f}B, Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}"
        )

    @property
    def cp_mesh(self):
        if not torch.distributed.is_initialized():
            return None
        # when none of the parallelisms are enabled, the world_mesh.mesh_dim_names is None
        if self.world_mesh.mesh_dim_names is not None and "cp" in self.world_mesh.mesh_dim_names:
            return self.world_mesh["cp"]
        else:
            return None

    @property
    def tp_mesh(self):
        if not torch.distributed.is_initialized():
            return None
        # when none of the parallelisms are enabled, the world_mesh.mesh_dim_names is None
        if self.world_mesh.mesh_dim_names is not None and "tp" in self.world_mesh.mesh_dim_names:
            return self.world_mesh["tp"]
        else:
            return None

    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            # This is a trick to handle TP for LLM but no TP for vision encoder, we need to convert DTensor to regular tensor later
            is_inputs_embeds_dtensor = isinstance(inputs_embeds, DTensor)  # This is True for TP>1, False for TP=1
            if is_inputs_embeds_dtensor:
                target_device_mesh = inputs_embeds.device_mesh
                target_placements = inputs_embeds.placements
                inputs_embeds = inputs_embeds.full_tensor()

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if is_inputs_embeds_dtensor:
                inputs_embeds = (
                    DTensor.from_local(inputs_embeds, device_mesh=target_device_mesh)
                    .redistribute(placements=target_placements)
                    .to_local()
                )
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore.
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                if self.config.model_type == "qwen2_5_vl":
                    position_ids, rope_deltas = get_rope_index_v2_5(
                        self.config,
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        second_per_grid_ts,
                        attention_mask,
                    )
                elif self.config.model_type == "qwen2_vl":
                    position_ids, rope_deltas = get_rope_index_v2(
                        self.config,
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        attention_mask,
                    )
                elif self.config.model_type == "qwen2_5":
                    position_ids = None
                    rope_deltas = None
                else:
                    raise ValueError(f"Unsupported model type: {self.config.model_type}")
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(  # Qwen2_5_VLModel
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        if self.cp_mesh is not None:
            logits = DTensor.from_local(logits, device_mesh=self.cp_mesh, placements=[Shard(1)]).full_tensor()
        return logits

    def forward(self, tokens, data_batch={}, start_pos: int = 0) -> torch.Tensor:
        """
        The training step of the model, including the loss computation.
        """
        assert "pixel_values" not in data_batch, "pixel_values should not be in data_batch, use images instead"
        pixel_values = data_batch.get("images", None)
        image_grid_thw = data_batch.get("image_grid_thw", None)
        pixel_values_videos = data_batch.get("videos", None)
        video_grid_thw = data_batch.get("video_grid_thw", None)
        second_per_grid_ts = None
        if image_grid_thw is not None:
            assert len(image_grid_thw) == 1, "Only batch=1 is supported for now, due to `get_rope_index`"
            image_grid_thw = image_grid_thw[0]  # 1, N_img, 3 -> N_img, 3
            second_per_grid_ts = None
        if video_grid_thw is not None:
            assert len(video_grid_thw) == 1, "Only batch=1 is supported for now, due to `get_rope_index`"
            video_grid_thw = video_grid_thw[0]  # 1, N_video, 3 -> N_video, 3
            if "second_per_grid_ts" in data_batch:  # only 2.5VL has fps
                second_per_grid_ts = data_batch["second_per_grid_ts"][0]  # 1, N_video -> N_video
            else:
                second_per_grid_ts = None
        logits = self._forward(
            input_ids=tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )
        return logits

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if iteration < 20:
            if "raw_video" in data_batch:
                log.info(f"Raw video shape: {data_batch['raw_video'].shape}")
            if "videos" in data_batch:
                log.info(f"Processed video tokens shape: {data_batch['videos'].shape}")
                if "second_per_grid_ts" in data_batch:  # only 2.5VL has fps
                    log.info(f"second_per_grid_ts: {data_batch['second_per_grid_ts']}")
            if "images" in data_batch:
                log.info(f"images shape: {data_batch['images'].shape}")
        return super().training_step(data_batch, iteration)
