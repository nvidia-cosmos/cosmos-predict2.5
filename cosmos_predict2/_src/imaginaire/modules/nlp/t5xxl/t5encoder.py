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

# pyrefly: ignore  # import-error
import gdown
import torch
from loguru import logger as logging
from transformers import T5Config, T5EncoderModel, T5Tokenizer, T5TokenizerFast
from transformers import logging as transformers_logging

# Suppresses a lot of unhelpful warnings from transformers.
transformers_logging.set_verbosity_error()


def download_file_from_google_drive(id, destination):
    r"""Download a file from google drive.

    Args:
        URL: GDrive file ID.
        destination: Path to save the file.

    Returns:

    """
    # download_file(f"https://docs.google.com/uc?export=download&id={URL}", destination)
    url = f"https://drive.google.com/uc?id={id}"
    logging.info(f"Download {url}")
    gdown.download(url, destination, quiet=False)


class T5Encoder(torch.nn.Module):
    def __init__(
        self, max_seq_len: int = 512, device: str = "cuda", return_offsets_mapping: bool = False, checkpoint: str = ""
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.return_offsets_mapping = return_offsets_mapping
        self.model_seq_len = 512
        # Initializing T5 model
        if return_offsets_mapping:
            # We need fast tokenizer to return offsets mapping.
            self.tokenizer = T5TokenizerFast.from_pretrained("t5-11b", model_max_length=self.model_seq_len)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained("t5-11b", model_max_length=self.model_seq_len)

        if checkpoint:
            logging.info(f"Checkpoint location: {checkpoint}")
            hard_coded_encoder_weight_location = checkpoint
            hard_coded_encoder_config_location = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "t5encoder.json"
            )
        else:
            # Here is a version that only loads the encoder.
            # https://drive.google.com/file/d/16Y5GoOIcZJBoSZklowVnYJxDKqOQEDUv
            hard_coded_encoder_weight_url = "16Y5GoOIcZJBoSZklowVnYJxDKqOQEDUv"
            hard_coded_encoder_weight_location = os.path.join(os.environ["TORCH_HOME"], "nlp/t5xxl/t5encoder.bin")
            hard_coded_encoder_config_location = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "t5encoder.json"
            )
            os.makedirs(os.path.dirname(hard_coded_encoder_weight_location), exist_ok=True)
            if not os.path.exists(hard_coded_encoder_weight_location):
                download_file_from_google_drive(hard_coded_encoder_weight_url, hard_coded_encoder_weight_location)
        self.model = T5EncoderModel.from_pretrained(
            hard_coded_encoder_weight_location,
            config=T5Config.from_json_file(hard_coded_encoder_config_location),
            low_cpu_mem_usage=True,
        )
        self.device = device
        self.to(device)
        # Below is the original version
        # self.model = T5EncoderModel.from_pretrained("t5-11b", low_cpu_mem_usage=True)

    def encode(self, text_batch):
        encoded = self.tokenizer.batch_encode_plus(
            text_batch,
            return_tensors="pt",
            padding="max_length",
            max_length=self.model_seq_len,
            truncation=True,
            return_offsets_mapping=self.return_offsets_mapping,
        )
        # We expect all the processing is done in GPU.
        input_ids = encoded.input_ids.to(self.device)
        attn_mask = encoded.attention_mask.to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attn_mask)
            encoded_text = output.last_hidden_state.detach()

        encoded_text = encoded_text[:, 0 : self.max_seq_len]
        attn_mask = attn_mask[:, 0 : self.max_seq_len]
        for bnum in range(encoded_text.shape[0]):
            nvalid_elem = attn_mask[bnum].sum().item()
            encoded_text[bnum][nvalid_elem:] = 0

        offsets_mapping = encoded["offset_mapping"] if self.return_offsets_mapping else None

        return encoded_text, attn_mask, offsets_mapping

    def _get_word_inds(self, text: str, word_ind: int, tokenizer: T5TokenizerFast):
        """
        Args:
            text (str): string in which we will search for the word
            word_ind (int): index of the word inside of this string, if split by whitespace
            tokenizer (T5TokenizerFast): The tokenizer to use on the prompts.

        Returns:
            positions (torch.Tensor): position(s) of the given word in the tokenized version of text.
        """
        words = text.split(" ")
        tokens = tokenizer(text, return_offsets_mapping=True)
        offset_mapping = tokens["offset_mapping"]

        word_to_token_map, current_word_tokens = [], []
        word_idx, char_start = 0, 0

        for i, (start, end) in enumerate(offset_mapping[:-1]):
            if start >= char_start and end <= char_start + len(words[word_idx]):
                current_word_tokens.append(i)
            if end >= char_start + len(words[word_idx]):
                word_to_token_map.append(current_word_tokens)
                current_word_tokens = []
                word_idx += 1
                char_start += len(words[word_idx - 1]) + 1  # Move to the next word

        positions = torch.tensor(word_to_token_map[word_ind])
        return positions

    def _get_replacement_mapper(self, x: str, y: str, tokenizer: T5TokenizerFast, max_len=512):
        """
        Args:
            x (str): Source prompt.
            y (str): Target prompt.
            tokenizer (T5TokenizerFast): The tokenizer to use on the prompts. Since we need offset maps,
                can only proceed if using T5TokenizerFast type.

        Returns:
            mapper (torch.Tensor): 2d tensor that aligns token indices from two input prompts
        """
        words_x = x.split(" ")
        words_y = y.split(" ")
        if len(words_x) != len(words_y):
            raise ValueError(
                f"attention replacement edit can only be applied on prompts with the same length"
                f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words."
            )
        inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
        inds_source = [
            self._get_word_inds(x, i, tokenizer) for i in inds_replace
        ]  # position(s) in a tokenized version of x
        inds_target = [self._get_word_inds(y, i, tokenizer) for i in inds_replace]
        mapper = torch.zeros((max_len, max_len))

        i = j = 0
        cur_inds = 0
        while i < max_len and j < max_len and inds_source:
            if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
                inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
                if len(inds_source_) == len(inds_target_):
                    mapper[inds_source_, inds_target_] = 1
                else:  # if a target token maps to multiple source tokens, divide its attention
                    ratio = 1 / len(inds_target_)
                    for i_t in inds_target_:
                        mapper[inds_source_, i_t] = ratio
                cur_inds += 1
                i += len(inds_source_)
                j += len(inds_target_)
            else:
                mapper[i, j] = 1
                i += 1
                j += 1

        return mapper.float().to(self.device)

    def get_replacement_mapper(self, prompts: list[str], max_len: int = 512):
        """
        Creates a mapping tensor that aligns token indices from two input prompts,
        indicating which tokens in prompt x should be replaced by tokens in prompt y.

        The resulting mapping tensor is used to transfer attention from the source prompt (x)
        to the target prompt (y) in a model that uses cross-attention.

        Eventually, this should be able to handle many target prompts. For now, we have limited to 1.

        If unexpected arguments, should return an empty map of 0s.

        Args:
            prompts (list[str]): Accepts the first as the source prompt and every subsequent as a target prompt.
            max_len (int): The max prompt length, in tokens, accepted; default dimension for mapper.

        Returns:
            mapper (torch.Tensor): 2d tensor that aligns token indices from two input prompts
        """
        if len(prompts) != 2 or not isinstance(self.tokenizer, T5TokenizerFast):
            return torch.zeros((max_len, max_len))
        src_seq, tgt_seq = prompts
        mapper = self._get_replacement_mapper(src_seq, tgt_seq, self.tokenizer, max_len)
        return mapper
