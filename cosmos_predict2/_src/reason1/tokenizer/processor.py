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

from typing import Optional

import numpy as np
import torch

try:
    from qwen_vl_utils import extract_vision_info, process_vision_info
except ImportError as e:
    print("qwen_vl_utils is not available. Reason1 model will not work.")

from transformers.models.auto.processing_auto import AutoProcessor

from cosmos_predict2._src.imaginaire.utils import log

_LOCK_TIMEOUT_SECONDS = 60


def build_tokenizer(
    tokenizer_type: str,
    cache_dir: Optional[str] = None,
):
    return Processor(tokenizer_type, cache_dir)


def flatten_content_list(messages):
    new_messages = []
    for message in messages:
        if "content" in message and isinstance(message["content"], list):
            text_list = [item["text"] for item in message["content"]]
            message["content"] = " ".join(text_list)
        new_messages.append(message)
    return new_messages


class Processor:
    # This is a wrapper around the AutoProcessor class to add some helper functions
    def __init__(self, name="Qwen/Qwen2.5-VL-3B-Instruct", cache_dir=None):
        self.name = name

        if name not in [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2-VL-2B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "Qwen/Qwen2.5-0.5B",
        ]:
            raise ValueError(f"Error loading processor {name}, please check if the tokenizer is available")
        if "VL" not in name:
            self.is_vision_tokenizer = False
        else:
            self.is_vision_tokenizer = True

        s3_uri = f"s3://bucket/cosmos_reasoning1/pretrained/Qwen_tokenizer/{name}/"
        from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path

        cache_dir = get_checkpoint_path(s3_uri)

        self.processor = AutoProcessor.from_pretrained(cache_dir)
        log.info("Successfully loaded processor from local cache")

        if hasattr(self.processor, "image_token"):
            self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        else:
            self.image_token_id = None
        if hasattr(self.processor, "video_token"):
            self.video_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.video_token)
        else:
            self.video_token_id = None

        if hasattr(self.processor, "tokenizer"):
            self.eos_id = self.processor.tokenizer.eos_token_id
            self.pad_id = self.processor.tokenizer.pad_token_id
        else:
            self.eos_id = self.processor.eos_token_id
            self.pad_id = self.processor.pad_token_id

    def apply_chat_template(
        self, messages, add_generation_prompt=False, return_tensors="pt", tokenize=True, add_vision_id=False
    ):
        assert tokenize, "tokenize must be True"
        if self.name.startswith("Qwen/Qwen2"):
            # Use manual workaround for add_vision_id bug
            if not self.is_vision_tokenizer:
                messages = flatten_content_list(messages)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                add_vision_id=add_vision_id,
            )
            image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)

            # add fps ourselves, as videos have been presampled according to the specified token length
            vision_infos = extract_vision_info(messages)
            fps_list = []
            for vision_info in vision_infos:
                if "video" in vision_info:
                    fps_list.append(vision_info["fps"])

            # process inputs
            if self.is_vision_tokenizer:
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=False,
                    return_tensors=return_tensors,
                    fps=fps_list,
                )
            else:
                inputs = self.processor(
                    text=[text],
                    padding=False,
                    return_tensors=return_tensors,
                )

            # save raw text
            inputs["text"] = text

            # Convert batch features into single features
            # By default, the processor returns a batch of features, but we use processor in dataloader, so we need to convert it to single features
            inputs["input_ids"] = inputs["input_ids"][0]  # N_dialogue, N_token -> N_token
            inputs["attention_mask"] = inputs["attention_mask"][0]  # N_dialogue, N_token -> N_token
            # inputs["image_grid_thw"]: N_img, 3
            # inputs["video_grid_thw"]: N_video, 3
        else:
            raise ValueError(f"apply_chat_template is not implemented for tokenizer_type {self.name}")

        return inputs

    def add_assistant_tokens_mask(self, tokens):
        """
        Add a mask to the assistant tokens.
        This is used to mask out tokens that are not generated by the assistant (e.g.,  system prompts, user prompts, chat templates), such that in the loss computation, only the tokens generated by the assistant are used.
        If there are multiple turns in the conversation, the mask will mask all the assistant tokens in each turn.

        Args:
            tokens (Union[List[int], torch.Tensor]): The tokens to add the mask to.
        Returns:
            Union[List[bool], torch.Tensor]: The mask. True for tokens generated by the assistant (i.e. should apply loss on), False for tokens not generated by the assistant.
        """
        if isinstance(tokens, torch.Tensor) and tokens.ndim == 2:
            mask = torch.stack([self.add_assistant_tokens_mask(tokens[i]) for i in range(tokens.shape[0])])
            assert mask.shape == tokens.shape
            return mask
        np_tokens = tokens.cpu().numpy() if isinstance(tokens, torch.Tensor) else np.array(tokens)
        assert np_tokens.ndim == 1

        if self.name.startswith("Qwen/Qwen2"):
            # Constants defining bos, eos and fixed offsets.
            BOS_TOKEN = "<|im_start|>"
            EOS_TOKEN = "<|im_end|>"
            ROLE = "assistant"
            # Offsets: skip the bos + "assistant\n" (always 3 tokens) and include the eos (+1) for supervision
            START_OFFSET = 3
            END_OFFSET = 1

            # Retrieve token IDs for the markers and the role.
            bos_token_id = self.processor.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
            eos_token_id = self.processor.tokenizer.convert_tokens_to_ids(EOS_TOKEN)
            role_id = self.processor.tokenizer.convert_tokens_to_ids(ROLE)

            # Locate all positions where the start and end markers appear.
            start_indices = np.where(np_tokens == bos_token_id)[0]
            end_indices = np.where(np_tokens == eos_token_id)[0]

            # Initialize the mask with False values.
            masks = np.zeros_like(np_tokens, dtype=bool)
            assert len(start_indices) == len(end_indices)
            # For each pair of bos/eos, check if the role is 'assistant'
            # and apply the mask accordingly.
            for start, end in zip(start_indices, end_indices):
                if np_tokens[start + 1] == role_id:
                    # Mask tokens from after the assistant header (start+3) to include the end marker (end+1)
                    masks[start + START_OFFSET : end + END_OFFSET] = True
        else:
            raise ValueError(f"add_assistant_tokens_mask is not implemented for tokenizer_type {self.name}")

        assert masks.shape == np_tokens.shape
        if isinstance(tokens, torch.Tensor):
            return torch.from_numpy(masks)
        else:
            return masks.tolist()

    def encode(self, *args, **kwargs):
        return self.processor.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.processor.decode(*args, **kwargs)


if __name__ == "__main__":
    """
    PYTHONPATH=. python3 cosmos_predict2/_src/reason1/tokenizer/processor.py
    """
    processor = Processor("Qwen/Qwen2.5-VL-3B-Instruct")
    print("done")
