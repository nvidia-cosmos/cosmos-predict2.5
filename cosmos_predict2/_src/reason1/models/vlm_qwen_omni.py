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
This file is modified from cosmos_predict2/_src/reason1/models/language_model/vlm_qwen.py for omni1 model.
"""

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._tensor import DTensor
from transformers import AutoConfig, Qwen2Model

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.reason1.models.vlm_base import init_mesh
from cosmos_predict2._src.reason1.models.vlm_qwen import QwenModel
from cosmos_predict2._src.reason1.networks.qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLModel
from cosmos_predict2._src.reason1.networks.qwen2_5_vl import get_rope_index as get_rope_index_v2_5
from cosmos_predict2._src.reason1.networks.qwen2_vl import Qwen2VisionTransformerPretrainedModel, Qwen2VLModel
from cosmos_predict2._src.reason1.networks.qwen2_vl import get_rope_index as get_rope_index_v2
from cosmos_predict2._src.reason1.parallelisms.parallelize_qwen import parallelize_qwen


class QwenVLBaseModel(QwenModel):
    """
    This is a base class for QwenVL models.
    Here we override the forward method and the training_step method to
    obtain more intermediate results from the language model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        self.lm_head = torch.nn.Linear(model_config.hidden_size, model_config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here]

        if torch.distributed.is_initialized() and model_config.use_fsdp2:
            self.world_mesh, self.parallel_dims = init_mesh(model_config)
            parallelize_qwen(self, self.world_mesh, self.parallel_dims, model_config)
            self.model.set_cp_mesh(self.cp_mesh)

    """
    Copy from QwenModel.forward with MODIFICATIONS
    MODIFICATIONS: add "lm_outputs" to the batch output.
    """

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
                    self.rope_deltas = rope_deltas
                elif self.config.model_type == "qwen2_vl":
                    position_ids, rope_deltas = get_rope_index_v2(
                        self.config,
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        attention_mask,
                    )
                    self.rope_deltas = rope_deltas
                elif self.config.model_type == "qwen2_5":
                    position_ids = None
                    rope_deltas = None
                else:
                    raise ValueError(f"Unsupported model type: {self.config.model_type}")

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
        if hasattr(self, "cp_mesh") and self.cp_mesh is not None:
            logits = DTensor.from_local(logits, device_mesh=self.cp_mesh, placements=[Shard(1)]).full_tensor()
        return logits, outputs

    """
    Copy from QwenModel.forward with MODIFICATIONS
    MODIFICATIONS: adding the hidden states to the output.
    """

    def forward(self, tokens, data_batch={}, start_pos: int = 0) -> torch.Tensor:
        """
        The training step of the model, including the loss computation.
        """
        assert "pixel_values" not in data_batch, "pixel_values should not be in data_batch, use images instead"
        pixel_values = data_batch.get("images", None)
        image_grid_thw = data_batch.get("image_grid_thw", None)
        pixel_values_videos = data_batch.get("videos", None)
        video_grid_thw = data_batch.get("video_grid_thw", None)
        attention_mask = data_batch.get("padding_mask", None)

        attention_mask = data_batch.get("padding_mask", None)

        if image_grid_thw is not None:
            assert len(image_grid_thw) == 1, "Only batch=1 is supported for now, due to `get_rope_index`"
            image_grid_thw = image_grid_thw[0]  # 1, N_img, 3 -> N_img, 3
        if video_grid_thw is not None:
            assert len(video_grid_thw) == 1, "Only batch=1 is supported for now, due to `get_rope_index`"
            video_grid_thw = video_grid_thw[0]  # 1, N_video, 3 -> N_video, 3
        logits, outputs = self._forward(
            input_ids=tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        return logits, outputs

    """
    Copy from cosmos_predict2._src.reason1.models.vlm_base.VLMBaseModel.training_step with MODIFICATIONS
    MODIFICATIONS: adding the hidden states to the output_batch.
    """

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        output_batch = {}

        if iteration < 20:
            summary_str = f"data_batch: {data_batch.keys()}"
            for key in ["tokens", "images", "videos"]:
                if key in data_batch and isinstance(data_batch[key], torch.Tensor):
                    summary_str += f" | {key} shape: {data_batch[key].shape}"
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
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor):
                data_batch[k] = v.to(device="cuda")

        # Token Mask (Note: this is not attention mask)
        token_mask = data_batch.get("token_mask", None)
        apply_token_mask = token_mask is not None

        if token_mask is None:
            token_mask = torch.ones_like(tokens, dtype=torch.bool)
        token_mask = token_mask.to(device="cuda")
        logits, outputs = self(tokens, data_batch)

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
        else:
            avg_num_real_tokens = torch.tensor(0.0, device=tokens.device)
            max_num_real_tokens = torch.tensor(0.0, device=tokens.device)
            if iteration < 20:
                log.warning(
                    f"No padding mask found in data batch, set avg_num_real_tokens to 0, data_batch: {data_batch.keys()}"
                )

        output_batch.update(
            {
                "encode_tokens": tokens,
                "logits": logits.detach(),
                "labels": labels.detach(),
                "ignore_index": ignore_index,
                "avg_num_assistant_tokens": avg_num_assistant_tokens.detach().item(),
                "avg_num_real_tokens": avg_num_real_tokens.detach().item(),
                "max_num_real_tokens": max_num_real_tokens.detach().item(),
                "current_num_assistant_tokens": token_mask.float().sum().detach().item(),
                "batch_size_local": batch_size_local,
                "lm_last_hidden_state": outputs.last_hidden_state,  # bs, seq_len, hidden_size
                "lm_all_hidden_state": outputs.hidden_states,  # [bs, seq_len, hidden_size] * len(decoder_layers), including the input embedding before the first decoder layer
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

        return output_batch, total_loss
