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

import logging
import os
import textwrap
import warnings
from typing import ClassVar

import qwen_vl_utils
import transformers
import vllm

from vqa.utils import is_verbose, log_verbose


class CosmosReasonModel:
    """
    Wrapper class for Cosmos-Reason1 model inference.

    This class encapsulates the model initialization and inference logic
    for the nvidia/Cosmos-Reason1-7B model.
    """

    # Hardcoded static parameters
    MODEL_NAME: ClassVar[str] = "nvidia/Cosmos-Reason1-7B"

    # Vision configuration parameters
    VISION_CONFIG: ClassVar[dict[str, int]] = {
        "fps": 4,  # Video frame rate
        "total_pixels": 6422528,  # Maximum number of pixels per video (8192 * 28**2)
    }

    # Sampling parameters for text generation
    SAMPLING_PARAMS: ClassVar[dict[str, int | float]] = {
        "max_tokens": 4096,  # Maximum number of tokens to generate
        "n": 1,  # Number of output sequences
        "temperature": 0.01,  # Controls randomness (lower = more deterministic)
        "seed": 42,  # Random seed for reproducibility
    }

    # System prompt
    SYSTEM_PROMPT: ClassVar[str] = (
        """Always respond in English.You are a helpful assistant that analyzes videos and answers questions about them. Provide clear, accurate, and detailed responses based on what you observe in the video."""
    )

    def __init__(self, model_name: str | None = None, revision: str | None = None):
        """
        Initialize the Cosmos-Reason1 model.

        Args:
            model_name: HuggingFace model name/path (default: nvidia/Cosmos-Reason1-7B)
            revision: Model revision (branch name, tag name, or commit id)
        """
        self.model_name = model_name or self.MODEL_NAME
        self.revision = revision

        # Configure logging based on VERBOSE environment variable
        is_verbose = os.environ.get("VERBOSE", "0") == "1"

        # Suppress warnings if not verbose
        if not is_verbose:
            warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
            warnings.filterwarnings("ignore", message=".*torchvision.*")
            # Suppress vLLM INFO logs
            logging.getLogger("vllm").setLevel(logging.WARNING)
            # Also suppress transformers logging
            logging.getLogger("transformers").setLevel(logging.WARNING)

        print(f"Loading model: {self.model_name}")
        if self.revision:
            print(f"  Revision: {self.revision}")

        # Initialize the model - don't pass any logging parameters to avoid breaking functionality
        self.llm = vllm.LLM(
            model=self.model_name,
            revision=self.revision,
            limit_mm_per_prompt={"image": 0, "video": 1},
            enforce_eager=True,
        )

        # Initialize the processor
        self.processor: transformers.Qwen2_5_VLProcessor = transformers.AutoProcessor.from_pretrained(
            self.model_name,
            use_fast=True,  # Suppress the fast processor warning
        )

        # Create sampling parameters
        self.sampling_params = vllm.SamplingParams(**self.SAMPLING_PARAMS)

        print("Model loaded successfully!")

    def create_conversation(
        self,
        user_prompt: str,
        video_path: str,
        system_prompt: str | None = None,
    ) -> list[dict]:
        """
        Create a conversation structure for the model.

        Args:
            user_prompt: The question to ask about the video
            video_path: Path to the video file
            system_prompt: System prompt (uses default if not provided)

        Returns:
            Conversation list of dictionaries
        """
        system_prompt = system_prompt or self.SYSTEM_PROMPT

        # Create user content with video and question
        user_content = [
            {"type": "video", "video": video_path, **self.VISION_CONFIG},
            {"type": "text", "text": user_prompt},
        ]

        # Build conversation
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        return conversation

    def infer(
        self,
        video_path: str,
        question: str,
        system_prompt: str | None = None,
        verbose: bool = False,
    ) -> str:
        """
        Run inference on a video with a question.

        Args:
            video_path: Path to the video file
            question: Question to ask about the video
            system_prompt: Optional custom system prompt
            verbose: Whether to print detailed output

        Returns:
            Model's response as a string
        """
        # Create conversation
        conversation = self.create_conversation(
            user_prompt=question,
            video_path=video_path,
            system_prompt=system_prompt,
        )

        log_verbose("\n" + "=" * 60)
        log_verbose("System Prompt:")
        log_verbose(textwrap.indent((system_prompt or self.SYSTEM_PROMPT).strip(), "  "))
        log_verbose("\nUser Question:")
        log_verbose(textwrap.indent(question.strip(), "  "))
        log_verbose("=" * 60 + "\n")

        # Apply chat template
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        # Process vision inputs
        image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
            conversation, return_video_kwargs=True
        )

        # Prepare multi-modal data
        mm_data = {}
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        # Prepare LLM inputs
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

        # Run inference
        # Suppress tqdm progress bars by redirecting stderr if not verbose
        # This only affects the output, not the execution
        if is_verbose():
            outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        else:
            import contextlib
            import io

            # Use StringIO to capture and discard stderr (progress bars)
            with contextlib.redirect_stderr(io.StringIO()):
                outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)

        # Extract response
        response = outputs[0].outputs[0].text

        return response
