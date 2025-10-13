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

import attrs
import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

from cosmos_predict2._src.common.types.embedding_concat_strategy import (
    EmbeddingConcatStrategy as EmbeddingConcatStrategy,
)
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.models.utils import load_state_dict, load_state_dict_from_folder
from cosmos_predict2._src.predict2.text_encoders.reason1 import QwenVLBaseModel
from cosmos_predict2._src.reason1.configs.default.model_config_qwen import QwenModelConfig, QwenVisionConfig
from cosmos_predict2._src.reason1.tokenizer.processor import build_tokenizer

NUM_EMBEDDING_PADDING_TOKENS = 512


@attrs.define(slots=False)
class TextEncoderConfig:
    """
    Config for the text encoder model
    """

    compute_online: bool = False
    embedding_concat_strategy: str = str(EmbeddingConcatStrategy.MEAN_POOLING)
    n_layers_per_group: int = 5
    ckpt_path: str = "s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/"
    s3_credential_path: str = "credentials/s3_checkpoint.secret"
    model_config: QwenVLBaseModel = L(QwenVLBaseModel)(
        model_config=L(QwenModelConfig)(
            tokenizer_type="Qwen/Qwen2.5-VL-7B-Instruct",
            name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
            hidden_size=3584,
            intermediate_size=18944,
            max_window_layers=28,
            num_attention_heads=28,
            num_hidden_layers=28,
            num_key_value_heads=4,
            tie_word_embeddings=False,
            vocab_size=152064,
            vision_config=L(QwenVisionConfig)(out_hidden_size=3584),
            output_hidden_states=True,
        ),
        tokenizer=L(build_tokenizer)(
            tokenizer_type="Qwen/Qwen2.5-VL-7B-Instruct",
        ),
    )


class TextEncoder:
    def __init__(self, config: TextEncoderConfig, device: str = "cuda"):
        self.config = config
        self.device = device

        log.info("Instantiating text encoder model...")
        with torch.device("meta"):
            self.model = lazy_instantiate(self.config.model_config)
        self.model.to_empty(device=self.device)
        with torch.no_grad():
            self.model.init_weights()
        from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path

        log.info(f"Loading checkpoint from {self.config.ckpt_path}.")
        ckpt_path = get_checkpoint_path(self.config.ckpt_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            is_fsdp = torch.distributed.get_world_size() > 1
        else:
            is_fsdp = False
        if os.path.isdir(ckpt_path):
            state_dict = load_state_dict_from_folder(ckpt_path)
        else:
            state_dict = load_state_dict(ckpt_path)
        # remove _extra_state
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith("._extra_state")}

        # Load Regular weights.
        if is_fsdp:
            set_model_state_dict(
                self.model,
                state_dict,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                    strict=False,
                ),
            )
        else:
            self.model.load_state_dict(state_dict, strict=False)

        del state_dict
        log.info(f"Finished loading checkpoint from {ckpt_path}.")
        self.model.eval()
        torch.cuda.empty_cache()
        log.info("Text encoder model instantiated")

    @staticmethod
    def mean_normalize(tensor: torch.Tensor) -> torch.Tensor:
        """
        Mean normalize a tensor by subtracting the mean and dividing by the standard deviation.

        Args:
        tensor (torch.tensor): The tensor to normalize

        Returns:
        torch.tensor: The normalized tensor
        """
        return (tensor - tensor.mean(dim=-1, keepdim=True)) / (tensor.std(dim=-1, keepdim=True) + 1e-8)

    def compute_text_embeddings_online(
        self, data_batch: dict[str, torch.Tensor], input_caption_key: str
    ) -> torch.Tensor:
        """
        Compute text embeddings for the given prompts.
        """
        assert self.model is not None, "Text encoder is not initialized"

        # Tokenize prompts
        input_ids_batch = []

        for sample_idx in range(len(data_batch[input_caption_key])):
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": data_batch[input_caption_key][sample_idx],
                        }
                    ],
                },
            ]
            tokenizer_output = self.model.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
            )
            input_ids = tokenizer_output["input_ids"]
            pad_id = self.model.tokenizer.pad_id

            # Do padding or truncation
            if NUM_EMBEDDING_PADDING_TOKENS > len(input_ids):
                # Do padding:
                pad_len = NUM_EMBEDDING_PADDING_TOKENS - len(input_ids)
                input_ids = input_ids.tolist() + [pad_id] * pad_len
            else:
                # Do truncation:
                input_ids = input_ids.tolist()[:NUM_EMBEDDING_PADDING_TOKENS]
            input_ids = torch.LongTensor(input_ids).to(device="cuda")
            input_ids_batch.append(input_ids)

        input_ids_batch = torch.stack(input_ids_batch, dim=0)

        # Compute text embeddings
        self.model = self.model.to(self.device)
        with torch.no_grad():
            _, outputs_batch = self.model(input_ids_batch, {})
        hidden_states = outputs_batch["hidden_states"]

        # # Skip the embeddings of the system prompt
        # hidden_states = hidden_states[:, num_system_prompt_tokens:]

        # Now compute the normalized embeddings
        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            normalized_state = self.mean_normalize(hidden_states[layer_idx])
            normalized_hidden_states.append(normalized_state)

        text_embeddings = None
        if self.config.embedding_concat_strategy == str(EmbeddingConcatStrategy.FULL_CONCAT):
            text_embeddings = torch.cat(normalized_hidden_states, dim=-1)
        elif self.config.embedding_concat_strategy == str(EmbeddingConcatStrategy.MEAN_POOLING):
            # Stack the normalized hidden states and calculate the mean
            text_embeddings = torch.stack(normalized_hidden_states)
            text_embeddings = text_embeddings.mean(dim=0)
        elif self.config.embedding_concat_strategy == str(EmbeddingConcatStrategy.POOL_EVERY_N_LAYERS_AND_CONCAT):
            # Split the l
            n_layers_per_group = self.config.n_layers_per_group
            text_embeddings = []
            for i in range(0, len(normalized_hidden_states), n_layers_per_group):
                group_embeddings = normalized_hidden_states[i : i + n_layers_per_group]
                group_embedding = torch.stack(group_embeddings)
                group_embedding = group_embedding.mean(dim=0)
                text_embeddings.append(group_embedding)
            text_embeddings = torch.cat(text_embeddings, dim=-1)
        else:
            raise ValueError(f"Invalid embedding_concat_strategy: {self.config.embedding_concat_strategy}")

        return text_embeddings


def get_reason1_embeddings(text: str):
    """
    Get reason1 embeddings for a given text.
    Output (1, seq len, d) embeddings
    """
    config = TextEncoderConfig(
        embedding_concat_strategy="full_concat",
    )
    text_encoder = TextEncoder(config)
    text_embeddings = text_encoder.compute_text_embeddings_online(
        {
            "text": [text],
        },
        "text",
    )
    return text_embeddings
