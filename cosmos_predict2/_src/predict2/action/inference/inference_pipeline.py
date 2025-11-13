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


import numpy as np
import torch
import torch.distributed as dist
import torchvision
from loguru import logger
from megatron.core import parallel_state

from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


class ActionVideo2WorldInference:
    """
    Handles the Video2World inference process, including model loading, data preparation,
    and video generation from an image/video and text prompt. Now supports context parallelism.
    """

    def __init__(self, experiment_name: str, ckpt_path: str, s3_credential_path: str, context_parallel_size: int = 1):
        """
        Initializes the Video2WorldInference class.

        Loads the diffusion model and its configuration based on the provided
        experiment name and checkpoint path. Sets up distributed processing if needed.

        Args:
            experiment_name (str): Name of the experiment configuration.
            ckpt_path (str): Path to the model checkpoint (local or S3).
            s3_credential_path (str): Path to S3 credentials file (if loading from S3).
            context_parallel_size (int): Number of GPUs for context parallelism.
        """
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.s3_credential_path = s3_credential_path
        self.context_parallel_size = context_parallel_size
        self.process_group = None

        # Initialize distributed processing if context parallel size > 1
        if self.context_parallel_size > 1:
            self._init_distributed()

        # Load the model and config
        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
            load_ema_to_reg=True,
        )

        # Enable context parallel on the model if using context parallelism
        if self.context_parallel_size > 1:
            model.net.enable_context_parallel(self.process_group)

        self.model = model
        self.config = config
        self.batch_size = 1
        self.neg_t5_embeddings = None

    def _init_distributed(self):
        """Initialize distributed processing for context parallelism."""

        # Initialize distributed environment
        distributed.init()

        # Initialize model parallel states
        parallel_state.initialize_model_parallel(
            context_parallel_size=self.context_parallel_size,
        )

        # Get the process group for context parallel
        self.process_group = parallel_state.get_context_parallel_group()

        logger.info(f"Initialized context parallel with size {self.context_parallel_size}")
        logger.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        prompt: str,
        num_conditional_frames: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        use_neg_prompt: bool = True,
    ):
        """
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str): The text prompt for conditioning.
            num_conditional_frames (int): Number of conditional frames to use.
            negative_prompt (str, optional): Custom negative prompt.
            use_neg_prompt (bool, optional): Whether to include negative prompt embeddings. Defaults to True.

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = video.shape

        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            "fps": torch.randint(16, 32, (self.batch_size,)).float(),  # Random FPS (might be used by model)
            "padding_mask": torch.zeros(self.batch_size, 1, H, W),  # Padding mask (assumed no padding here)
            "num_conditional_frames": num_conditional_frames,  # Specify number of conditional frames
        }

        if use_neg_prompt:
            assert negative_prompt is not None, "Negative prompt is required when use_neg_prompt is True"

        # Compute text embeddings
        if self.model.text_encoder is not None:
            data_batch["ai_caption"] = [prompt]
            data_batch["t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [prompt], "images": None},
                input_caption_key="ai_caption",
            )
            if use_neg_prompt:
                data_batch["neg_t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                    data_batch={"ai_caption": [negative_prompt], "images": None},
                    input_caption_key="ai_caption",
                )
        else:
            data_batch["t5_text_embeddings"] = get_text_embedding(prompt)
            if use_neg_prompt:
                data_batch["neg_t5_text_embeddings"] = get_text_embedding(negative_prompt)

        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

        return data_batch

    def step_inference_with_latents(
        self,
        img_array: np.ndarray,
        action: np.ndarray = None,
        guidance: int = 3,
        seed: int = 1,
        num_latent_conditional_frames: int = 1,
        query_steps: list[int] = None,
    ):
        """
        Runs a single inference step to generate the next video frame and the full video given an input image and action.
        """

        num_video_frames = action.shape[0] + 1

        img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0)
        vid_input = torch.cat([img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], dim=0)
        vid_input = (vid_input * 255.0).to(torch.uint8)  # Convert to uint8 range if needed (might depend on model)
        vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Add batch dim B=1 and permute

        # Prepare the data batch with text embeddings
        data_batch = self._get_data_batch_input(
            vid_input,
            prompt="",
            num_conditional_frames=num_latent_conditional_frames,
            negative_prompt="",
            use_neg_prompt=False,
        )

        data_batch["action"] = torch.from_numpy(action).cuda().to(dtype=torch.bfloat16)[None, ...]

        mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"GPU memory usage after getting data_batch: {mem_bytes / (1024**3):.2f} GB")

        # Generate latent samples using the diffusion model
        sample, latents_to_save = self.model.generate_samples_with_latents_from_batch(
            data_batch,
            n_sample=1,  # Generate one sample
            guidance=guidance,
            seed=seed,  # Fixed seed for reproducibility
            is_negative_prompt=True,  # Use classifier-free guidance
            query_steps=query_steps,
        )

        # Decode the latent sample into a video tensor
        video = self.model.decode(sample)

        video_normalized = (video - (-1)) / (1 - (-1))
        video_clamped = (torch.clamp(video_normalized[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        next_frame = torch.clamp(video_normalized[0, :, -1, :, :], 0, 1)
        next_frame = (next_frame * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return next_frame, video_clamped, latents_to_save

    def step_inference(
        self,
        img_array: np.ndarray,
        action: np.ndarray = None,
        guidance: int = 3,
        seed: int = 1,
        num_latent_conditional_frames: int = 1,
    ):
        """
        Runs a single inference step to generate the next video frame and the full video given an input image and action.

        Args:
            img_array (np.ndarray): Input image as a numpy array (H, W, C), typically the first frame.
            action (np.ndarray, optional): Action vector to condition the model. Should be shape (action_dim,) or (chunk_size, action_dim).
            guidance (int, optional): Guidance scale for classifier-free guidance. Default is 3.
            seed (int, optional): Random seed for reproducibility. Default is 1.
            num_latent_conditional_frames (int, optional): Number of conditional frames to use for the model. Default is 1.

        Returns:
            next_frame (np.ndarray): The next predicted frame as a numpy array (H, W, C), uint8.
            video_clamped (np.ndarray): The generated video as a numpy array (T, H, W, C), uint8.
        """
        num_video_frames = action.shape[0] + 1

        img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0)  # (1, H, W, C)
        vid_input = torch.cat([img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], dim=0)
        vid_input = (vid_input * 255.0).to(torch.uint8)  # Convert to uint8 range if needed (might depend on model)
        vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Add batch dim B=1 and permute

        # Prepare the data batch with text embeddings
        data_batch = self._get_data_batch_input(
            vid_input,
            prompt="",
            num_conditional_frames=num_latent_conditional_frames,
            negative_prompt="",
            use_neg_prompt=False,
        )

        data_batch["action"] = torch.from_numpy(action).cuda().to(dtype=torch.bfloat16)[None, ...]

        mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"GPU memory usage after getting data_batch: {mem_bytes / (1024**3):.2f} GB")

        # Generate latent samples using the diffusion model
        # Video should be of shape torch.Size([1, 3, 93, 192, 320]) # Note: Shape check comment
        sample = self.model.generate_samples_from_batch(
            data_batch,
            n_sample=1,  # Generate one sample
            guidance=guidance,
            seed=seed,  # Fixed seed for reproducibility
            is_negative_prompt=True,  # Use classifier-free guidance
        )

        # Decode the latent sample into a video tensor
        video = self.model.decode(sample)

        video_normalized = (video - (-1)) / (1 - (-1))
        video_clamped = (torch.clamp(video_normalized[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        next_frame = torch.clamp(video_normalized[0, :, -1, :, :], 0, 1)
        next_frame = (next_frame * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return next_frame, video_clamped

    def step_inference_multi_frame(
        self,
        video_array: np.ndarray,
        action: np.ndarray = None,
        guidance: int = 3,
        seed: int = 1,
        num_latent_conditional_frames: int = 2,
    ):
        """
        Runs a single inference step to generate the next video frame and the full video given an input image and action.

        Args:
            video_array (np.ndarray): Input video as a numpy array (T, H, W, C).
            action (np.ndarray, optional): Action vector to condition the model. Should be shape (action_dim,) or (chunk_size, action_dim).

            guidance (int, optional): Guidance scale for classifier-free guidance. Default is 3.
            seed (int, optional): Random seed for reproducibility. Default is 1.
            num_latent_conditional_frames (int, optional): Number of conditional frames to use for the model. Default is 1.

        Returns:
            next_frame (np.ndarray): The next predicted frame as a numpy array (H, W, C), uint8.
            video_clamped (np.ndarray): The generated video as a numpy array (T, H, W, C), uint8.
        """
        num_video_frames = action.shape[0] + 1 + (num_latent_conditional_frames - 1) * 4
        num_cond_image_frames = (num_latent_conditional_frames - 1) * 4 + 1

        assert num_cond_image_frames == video_array.shape[0], (
            "Number of conditional frames is not equal to the number of frames in the video"
        )
        assert action.shape[0] == num_video_frames - num_cond_image_frames, (
            "Number of action frames is not equal to the number of frames in the video"
        )

        video_tensor = torch.stack(
            [torchvision.transforms.functional.to_tensor(v) for v in video_array]
        )  # (T, C, H, W)
        vid_input = torch.cat(
            [
                video_tensor,
                torch.zeros_like(video_tensor[0][None, ...]).repeat(num_video_frames - num_cond_image_frames, 1, 1, 1),
            ],
            dim=0,
        )
        vid_input = (vid_input * 255.0).to(torch.uint8)  # Convert to uint8 range if needed (might depend on model)
        vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Add batch dim B=1 and permute

        # Prepare the data batch with text embeddings
        data_batch = self._get_data_batch_input(
            vid_input,
            prompt="",
            num_conditional_frames=num_latent_conditional_frames,
            negative_prompt="",
            use_neg_prompt=False,
        )

        zero_action = np.zeros(
            (4 * (num_latent_conditional_frames - 1), action.shape[1])
        )  # (4 * (num_latent_conditional_frames-1), action_dim)
        action_padded = np.concatenate([zero_action, action], axis=0)

        data_batch["action"] = torch.from_numpy(action_padded).cuda().to(dtype=torch.bfloat16)[None, ...]

        mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"GPU memory usage after getting data_batch: {mem_bytes / (1024**3):.2f} GB")

        # Generate latent samples using the diffusion model
        # Video should be of shape torch.Size([1, 3, 93, 192, 320]) # Note: Shape check comment
        sample = self.model.generate_samples_from_batch(
            data_batch,
            n_sample=1,  # Generate one sample
            guidance=guidance,
            seed=seed,  # Fixed seed for reproducibility
            is_negative_prompt=True,  # Use classifier-free guidance
        )

        # Decode the latent sample into a video tensor
        video = self.model.decode(sample)

        video_normalized = (video - (-1)) / (1 - (-1))
        video_clamped = (torch.clamp(video_normalized[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        next_frame = torch.clamp(video_normalized[0, :, -1, :, :], 0, 1)
        next_frame = (next_frame * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return next_frame, video_clamped

    def cleanup(self):
        """Clean up distributed resources."""
        if self.context_parallel_size > 1:
            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
