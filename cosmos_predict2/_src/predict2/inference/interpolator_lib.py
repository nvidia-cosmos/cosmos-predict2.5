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

"""Library for video frame interpolation using diffusion models.

This module provides functionality to interpolate frames between two input frames,
effectively increasing the frame rate of videos using trained diffusion models.

Usage:
    interpolator = Interpolator(experiment_name, ckpt_path)
    output = interpolator(**input_args)
"""

import math
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from loguru import logger
from megatron.core import parallel_state

from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.imaginaire.utils.s3_utils import load_from_s3_with_cache
from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

_CONFIG_FILE = "cosmos_predict2/_src/predict2/configs/frame_interpolation/config.py"
_NEG_PROMPT_EMBEDDINGS_S3_PATH = "s3://bucket/projects/edify_video/v4/video_neg_prompt_embeddings_v0.pt"
_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)
_T5_MAX_LENGTH = 512
_T5_HIDDEN_DIM = 1024
_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def resize_video_spatially(video: torch.Tensor, resolution: list[int]) -> torch.Tensor:
    """Resize and center-crop video to target resolution while preserving aspect ratio.

    Args:
        video: Input video tensor of shape (B, C, T, H, W).
        resolution: Target resolution [H, W].

    Returns:
        Resized and cropped video tensor of shape (B, C, T, target_H, target_W).
    """
    b, _, _, h, w = video.shape
    # Reshape all frames into a batch of images for efficient processing
    image_batch = rearrange(video, "b c t h w -> (b t) c h w")

    target_h, target_w = resolution
    # Scale to ensure the smaller dimension matches target while preserving aspect ratio
    scaling_ratio = max((target_w / w), (target_h / h))
    resizing_shape = (int(math.ceil(scaling_ratio * h)), int(math.ceil(scaling_ratio * w)))

    # Apply resize and center crop operations
    image_resized = torchvision.transforms.functional.resize(image_batch, resizing_shape)
    image_resized = torchvision.transforms.functional.center_crop(image_resized, resolution)

    # Reshape back to video tensor format
    return rearrange(image_resized, "(b t) c h w -> b c t h w", b=b)


class Interpolator(nn.Module):
    """Video frame interpolation inference handler using diffusion models.

    Supports both single-GPU and multi-GPU inference with context parallelism.
    Loads trained diffusion models and generates interpolated frames between
    input frame pairs using optional text conditioning.
    """

    def __init__(self, experiment_name: str, ckpt_path: str, s3_credential_path: str, context_parallel_size: int = 1):
        """Initialize the interpolator inference handler.

        Args:
            experiment_name: Name of the experiment configuration.
            ckpt_path: Path to the model checkpoint (local or S3).
            s3_credential_path: Path to S3 credentials file.
            context_parallel_size: Number of GPUs for context parallelism.
        """
        super().__init__()
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.s3_credential_path = s3_credential_path
        self.context_parallel_size = context_parallel_size
        self.process_group = None

        # Initialize distributed processing for multi-GPU setups
        if self.context_parallel_size > 1:
            self._init_distributed()

        # Load diffusion model and configuration
        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file=_CONFIG_FILE,
            load_ema_to_reg=True,
            experiment_opts=[
                f"checkpoint.load_from_object_store.credentials={self.s3_credential_path}",
                f"checkpoint.save_to_object_store.credentials={self.s3_credential_path}",
                f"checkpoint.load_from_object_store.bucket={urlparse(self.ckpt_path).netloc or 'dummy_bucket'}",
                f"checkpoint.save_to_object_store.bucket={urlparse(self.ckpt_path).netloc or 'dummy_bucket'}",
                f"model.config.tokenizer.s3_credential_path={self.s3_credential_path}",
            ],
        )

        # Enable context parallelism for multi-GPU inference
        if self.context_parallel_size > 1:
            model.net.enable_context_parallel(self.process_group)

        self.model = model
        self.model_config = config.model.config
        self.precision = getattr(config.model, "precision", torch.bfloat16)
        self.neg_t5_embeddings = None

    def _init_distributed(self):
        """Initialize distributed processing for context parallelism."""
        # Setup distributed environment
        distributed.init()

        # Configure model parallel states for context parallelism
        parallel_state.initialize_model_parallel(
            context_parallel_size=self.context_parallel_size,
        )

        # Obtain process group for context parallel communication
        self.process_group = parallel_state.get_context_parallel_group()

        logger.info(f"Initialized context parallel with size {self.context_parallel_size}")
        logger.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        prompt: str | None,
        negative_prompt: str | None = None,
        use_neg_prompt: bool = True,
    ) -> dict:
        """Prepare input data batch for the diffusion model.

        Args:
            video: Input video tensor (B, C, T, H, W).
            prompt: Text prompt for conditioning (optional).
            negative_prompt: Custom negative prompt (optional).
            use_neg_prompt: Whether to include negative prompt embeddings.

        Returns:
            Dictionary containing the prepared data batch with proper device and dtype.
        """
        B, _, _, H, W = video.shape

        # Construct base data batch with required model inputs
        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            "fps": torch.randint(16, 32, (B,)).float(),  # Random FPS for model conditioning
            "padding_mask": torch.zeros(B, 1, H, W),  # No padding mask needed
            "num_conditional_frames": 1,  # Number of conditioning frames
        }

        # # Add positive prompt embeddings if provided
        # if prompt is not None:
        #     data_batch["t5_text_embeddings"] = get_text_embedding(prompt)

        # Compute text embeddings
        if self.model.text_encoder is not None:
            # Use empty string if prompt is None (for unconditional generation)
            effective_prompt = prompt if prompt is not None else ""
            data_batch["ai_caption"] = [effective_prompt]
            data_batch["t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [effective_prompt], "images": None},
                input_caption_key="ai_caption",
            )
            if use_neg_prompt:
                if negative_prompt is None:
                    negative_prompt = _DEFAULT_NEGATIVE_PROMPT
                data_batch["neg_t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                    data_batch={"ai_caption": [negative_prompt], "images": None},
                    input_caption_key="ai_caption",
                )
        else:
            # Use empty string if prompt is None (for unconditional generation)
            effective_prompt = prompt if prompt is not None else ""
            data_batch["t5_text_embeddings"] = get_text_embedding(effective_prompt)
            if use_neg_prompt:
                if negative_prompt is not None:
                    # Use custom negative prompt embeddings
                    logger.info(f"Using custom negative prompt: {negative_prompt}")
                    data_batch["neg_t5_text_embeddings"] = (
                        get_text_embedding(negative_prompt).cuda().to(dtype=self.precision)
                    )
                else:
                    # Load default negative embeddings from S3
                    if self.neg_t5_embeddings is None:
                        self._load_default_negative_embeddings()

                    # Create zero-padded tensor for negative embeddings
                    zeros_t5 = torch.zeros([1, _T5_MAX_LENGTH, _T5_HIDDEN_DIM], dtype=self.precision).cuda()
                    length = min(_T5_MAX_LENGTH, self.neg_t5_embeddings.shape[0])
                    zeros_t5[0, :length] = self.neg_t5_embeddings.to(dtype=self.precision).cuda()[:length]
                    data_batch["neg_t5_text_embeddings"] = zeros_t5

        # Move floating-point tensors to GPU with model precision
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                data_batch[k] = v.cuda().to(dtype=self.precision)

        # # Configure negative prompts for classifier-free guidance
        # if use_neg_prompt:
        #     if negative_prompt is not None:
        #         # Use custom negative prompt embeddings
        #         logger.info(f"Using custom negative prompt: {negative_prompt}")
        #         data_batch["neg_t5_text_embeddings"] = (
        #             get_text_embedding(negative_prompt).cuda().to(dtype=self.precision)
        #         )
        #     else:
        #         # Load default negative embeddings from S3
        #         if self.neg_t5_embeddings is None:
        #             self._load_default_negative_embeddings()

        #         # Create zero-padded tensor for negative embeddings
        #         zeros_t5 = torch.zeros([1, _T5_MAX_LENGTH, _T5_HIDDEN_DIM], dtype=self.precision).cuda()
        #         length = min(_T5_MAX_LENGTH, self.neg_t5_embeddings.shape[0])
        #         zeros_t5[0, :length] = self.neg_t5_embeddings.to(dtype=self.precision).cuda()[:length]
        #         data_batch["neg_t5_text_embeddings"] = zeros_t5

        #         # Use zero embeddings for positive prompt if none provided
        #         if prompt is None:
        #             data_batch["t5_text_embeddings"] = zeros_t5
        # else:
        #     zeros_t5 = torch.zeros([B, _T5_MAX_LENGTH, _T5_HIDDEN_DIM], dtype=self.precision).cuda()
        #     data_batch["neg_t5_text_embeddings"] = zeros_t5
        #     data_batch["t5_text_embeddings"] = zeros_t5

        return data_batch

    def synchronize(self):
        """Synchronize all processes in distributed mode."""
        if self.context_parallel_size > 1:
            import torch.distributed as dist

            dist.barrier()

    def cleanup(self):
        """Clean up distributed resources."""
        # Add synchronization before cleanup
        self.synchronize()

        if self.context_parallel_size > 1:
            import torch.distributed as dist
            from megatron.core import parallel_state

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()

    def _load_default_negative_embeddings(self):
        """Load default negative text embeddings from S3."""
        backend_args = {
            "backend": "s3",
            "path_mapping": None,
            "s3_credential_path": self.s3_credential_path,
        }
        self.neg_t5_embeddings = load_from_s3_with_cache(
            _NEG_PROMPT_EMBEDDINGS_S3_PATH,
            easy_io_kwargs={"map_location": torch.device(torch.cuda.current_device())},
            backend_args=backend_args,
        )

    def forward(
        self,
        prompt: str,
        input_video: torch.Tensor,
        guidance: int = -1,
        resolution: str = "1072,1920",
        seed: int = 1,
        negative_prompt: str = None,
    ) -> torch.Tensor:
        """Generate interpolated frames between input frame pair.

        Args:
            prompt: Text prompt for conditioning the interpolation.
            input_video: Input video batch of layout (B, C, T, H, W), range [-1..1],
                with first and last temporal frames being the conditioning frames,
                while the in-betweens to be interpolated are initialized as zeros.
            guidance: Classifier-free guidance scale.
            resolution: Target resolution as "H,W" string.
            seed: Random seed for reproducibility.
            negative_prompt: Custom negative prompt (optional).

        Returns:
            Generated video tensor (B, C, T, H, W) in range [-1, 1].
        """
        # Validate input tensor dimensions and temporal frames
        assert input_video.ndim == 5, "Input video must be a 5D tensor of layout (B, C, T, H, W)"
        assert input_video.shape[-3] == self.model_config.state_t, (
            "The number of temporal frames in the input video must match the state_t in the model config"
        )

        # Determine target resolution for processing
        if resolution is not None:
            video_resolution = tuple(int(x) for x in resolution.split(","))
        else:
            video_resolution = self.model.get_video_height_width()
        input_video = resize_video_spatially(input_video, video_resolution)

        # Determine if we should use CFG
        use_cfg = prompt is not None and guidance != -1

        # Convert from [-1,1] float range to [0,255] uint8 range expected by model
        input_video_uint8 = (_UINT8_MAX_F * (input_video + 1.0) / 2.0).to(dtype=torch.uint8)

        # Prepare model input data batch
        data_batch = self._get_data_batch_input(
            input_video_uint8,
            prompt,
            negative_prompt=negative_prompt,
            use_neg_prompt=use_cfg,  # Only use negative prompts when doing CFG
        )

        # Log current GPU memory usage
        # mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # logger.info(f"GPU memory usage after preparing data batch: {mem_bytes / (1024**3):.2f} GB")

        # Generate latent samples using diffusion model
        sample = self.model.generate_samples_from_batch(
            data_batch,
            n_sample=1,
            guidance=guidance,
            seed=seed,
            is_negative_prompt=use_cfg,  # Consistent with use_neg_prompt
        )

        # Decode latent samples back to video tensor
        return self.model.decode(sample)
