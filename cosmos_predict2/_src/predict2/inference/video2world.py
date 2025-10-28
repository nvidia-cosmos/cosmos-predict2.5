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
# Script for generating I2W videos in s3
PYTHONPATH=. python cosmos_predict2/_src/predict2/inference/video2world.py --experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4 --ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000 --save_root results/cli_debug_from_s3 --input_root /project/cosmos/ybalaji/data/internal_val_set_clean

# Script for text2world generation
export EXPERIMENT=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/inference/video2world.py \
--experiment=${EXPERIMENT} \
--ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/${EXPERIMENT}/checkpoints/iter_000025000 \
--save_root results/base_model/${EXPERIMENT}_025k_seed0_t2w \
--num_latent_conditional_frames=0 --seed=0 \
--input_root /project/cosmos/fangyinw/data/pbench/v0

# I2W with context parallel with 8 GPUs:
PYTHONPATH=. torchrun --nproc_per_node=8 cosmos_predict2/_src/predict2/inference/video2world.py --experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4 --ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000 --save_root results/cli_debug_from_s3 --input_root /project/cosmos/ybalaji/data/internal_val_set_clean --context_parallel_size 8

# V2W with context parallel with 8 GPUs:
PYTHONPATH=. torchrun --nproc_per_node=8 cosmos_predict2/_src/predict2/inference/video2world.py --experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4 --ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000 --save_root results/cli_debug_from_s3 --input_root pbench_upsampled_prompts --num_latent_conditional_frames=2 --context_parallel_size=8


Folder structure:
We assume the input root contains images and prompts in the following format:
input_root/
 ├── image_1.jpg
 ├── image_1.txt
 ├── image_2.jpg
 └── image_2.txt
 └── ...

or videos and prompts in the following format:
input_root/
 ├── video_1.mp4
 ├── video_1.txt
 ├── video_2.mp4
 └── video_2.txt
 └── ...
"""

import math
import os
from typing import TYPE_CHECKING

import torch
import torchvision
from megatron.core import parallel_state
from PIL import Image

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
_VIDEO_EXTENSIONS = [".mp4"]

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def resize_input(video: torch.Tensor, resolution: list[int]):
    r"""
    Resizes and crops the input video tensor while preserving aspect ratio.

    The video is first resized so that the smaller dimension matches the target resolution,
    preserving the aspect ratio. Then, it's center-cropped to the target resolution.

    Args:
        video (torch.Tensor): Input video tensor of shape (T, C, H, W).
        resolution (list[int]): Target resolution [H, W].

    Returns:
        torch.Tensor: Resized and cropped video tensor of shape (T, C, target_H, target_W).
    """

    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = resolution

    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    video_cropped = torchvision.transforms.functional.center_crop(video_resized, resolution)
    return video_cropped


def read_and_process_image(img_path: str, resolution: list[int], num_video_frames: int, resize: bool = True):
    """
    Reads an image, converts it to a video tensor, and processes it for model input.

    The image is loaded, converted to a tensor, and replicated to match the
    `num_video_frames`. It's then optionally resized and permuted to the
    standard video format (B, C, T, H, W).

    Args:
        img_path (str): Path to the input image file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        num_video_frames (int): The number of frames the output video tensor should have.
        resize (bool, optional): Whether to resize the image to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W).

    Raises:
        ValueError: If the image extension is not one of the supported types.
    """
    ext = os.path.splitext(img_path)[1]
    if ext not in _IMAGE_EXTENSIONS:
        raise ValueError(f"Invalid image extension: {ext}")

    # Read the image
    img = Image.open(img_path)

    # Convert to tensor
    img = torchvision.transforms.functional.to_tensor(img)
    # Create a video tensor by repeating the first frame
    vid_input = img.unsqueeze(0)  # Add temporal dimension T=1

    # Repeat the first frame to match the desired number of video frames
    # Note: The actual content for frames > 0 will be generated by the model.
    vid_input = torch.cat([vid_input, torch.zeros_like(vid_input).repeat(num_video_frames - 1, 1, 1, 1)], dim=0)
    vid_input = (vid_input * 255.0).to(torch.uint8)  # Convert to uint8 range if needed (might depend on model)
    if resize:
        # Resize and crop to the target resolution
        vid_input = resize_input(vid_input, resolution)

    # Convert to {B, C, T, H, W} format expected by the model
    vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Add batch dim B=1 and permute
    return vid_input


def read_and_process_video(
    video_path: str,
    resolution: list[int],
    num_video_frames: int,
    num_latent_conditional_frames: int = 2,
    resize: bool = True,
):
    """
    Reads a video, processes it for model input.

    The video is loaded using easy_io, and uses the last 4x(num_latent_conditional_frames - 1) + 1 from the video.
    If the video is shorter than num_video_frames, it pads with the last frame repeated.
    The first num_latent_conditional_frames are marked as conditioning frames.

    Args:
        video_path (str): Path to the input video file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        num_video_frames (int): Number of frames needed by the model (should equal model.tokenizer.get_pixel_num_frames(model.config.state_t)).
        num_latent_conditional_frames (int): Number of latent conditional frames from the input video (1 or 2).
        resize (bool, optional): Whether to resize the video to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W) where T equals num_video_frames.

    Raises:
        ValueError: If the video extension is not supported or other validation errors.

    Note:
        Uses the last 4x(num_latent_conditional_frames - 1) + 1 frames from the video. If video is shorter, pads with last frame repeated.
    """
    ext = os.path.splitext(video_path)[1]
    if ext.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Invalid video extension: {ext}")

    # Load video using easy_io
    try:
        video_frames, video_metadata = easy_io.load(video_path)  # Returns (T, H, W, C) numpy array
        log.info(f"Loaded video with shape {video_frames.shape}, metadata: {video_metadata}")
    except Exception as e:
        raise ValueError(f"Failed to load video {video_path}: {e}")

    # Convert numpy array to tensor and rearrange dimensions
    video_tensor = torch.from_numpy(video_frames).float() / 255.0  # Convert to [0, 1] range
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

    available_frames = video_tensor.shape[1]

    # Calculate how many frames to extract from input video
    frames_to_extract = 4 * (num_latent_conditional_frames - 1) + 1
    log.info(f"Will extract {frames_to_extract} frames from input video and pad to {num_video_frames}")

    # Validate num_latent_conditional_frames
    if num_latent_conditional_frames not in [1, 2]:
        raise ValueError(f"num_latent_conditional_frames must be 1 or 2, but got {num_latent_conditional_frames}")

    # Create output tensor with exact num_video_frames
    C, _, H, W = video_tensor.shape
    full_video = torch.zeros(C, num_video_frames, H, W)

    if available_frames < frames_to_extract:
        raise ValueError(
            f"Video has only {available_frames} frames but needs at least {frames_to_extract} frames for num_latent_conditional_frames={num_latent_conditional_frames}"
        )

    # Extract the last frames_to_extract from input video
    start_idx = available_frames - frames_to_extract
    extracted_frames = video_tensor[:, start_idx:, :, :]
    full_video[:, :frames_to_extract, :, :] = extracted_frames
    log.info(f"Extracted last {frames_to_extract} frames from video (frames {start_idx} to {available_frames - 1})")

    # Pad remaining frames with the last extracted frame
    if frames_to_extract < num_video_frames:
        last_frame = extracted_frames[:, -1:, :, :]  # (C, 1, H, W)
        padding_frames = num_video_frames - frames_to_extract
        last_frame_repeated = last_frame.repeat(1, padding_frames, 1, 1)  # (C, padding_frames, H, W)
        full_video[:, frames_to_extract:, :, :] = last_frame_repeated
        log.info(f"Padded {padding_frames} frames with last extracted frame")

    # Convert to the format expected by the rest of the pipeline
    full_video = full_video.permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)
    full_video = (full_video * 255.0).to(torch.uint8)  # Convert to uint8 range

    if resize:
        # Resize and crop to the target resolution
        full_video = resize_input(full_video, resolution)

    # Convert to {B, C, T, H, W} format expected by the model
    full_video = full_video.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Add batch dim B=1 and permute
    return full_video


class Video2WorldInference:
    """
    Handles the Video2World inference process, including model loading, data preparation,
    and video generation from an image/video and text prompt. Now supports context parallelism.
    """

    def __init__(
        self,
        experiment_name: str,
        ckpt_path: str,
        s3_credential_path: str,
        context_parallel_size: int = 1,
        config_file: str = "cosmos_predict2/_src/predict2/configs/video2world/config.py",
    ):
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
        experiment_opts = []
        if not INTERNAL:
            experiment_opts.append("~data_train")

        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file=config_file,
            load_ema_to_reg=True,
            experiment_opts=experiment_opts,
        )
        if TYPE_CHECKING:
            from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
                Video2WorldModelRectifiedFlow,
            )

            model: Video2WorldModelRectifiedFlow = model

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

        log.info(f"Initialized context parallel with size {self.context_parallel_size}")
        log.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        prompt: str,
        num_conditional_frames: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        use_neg_prompt: bool = True,
        camera: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
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
            camera: (torch.Tensor, optional) Target camera extrinsics and intrinsics for the K output videos, must be provided for camera conditioned model.
            action: (torch.Tensor, optional) Target robot action for the K output videos, must be provided for action conditioned model.

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = video.shape

        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            "camera": camera,
            "action": action.unsqueeze(0) if action is not None else None,
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

    def generate_vid2world(
        self,
        prompt: str,
        input_path: str | torch.Tensor | None,
        guidance: int = 7,
        num_video_frames: int = 77,
        num_latent_conditional_frames: int = 1,
        num_input_video: int = 1,
        num_output_video: int = 1,
        resolution: str = "192,320",
        seed: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        camera: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        num_steps: int = 35,
        offload_diffusion_model: bool = False,
        offload_text_encoder: bool = False,
        offload_tokenizer: bool = False,
    ):
        """
        Generates a video based on an input image or video and text prompt.

        Processes the input, prepares the data batch, runs the diffusion
        model sampling, and decodes the result into a video tensor.

        Args:
            prompt: The text prompt describing the desired video content/style.
            input_path: Path to the input image or video file or a torch.Tensor.
            guidance: Classifier-free guidance scale. Defaults to 7.
            num_video_frames: Number of video frames to generate. Defaults to 77.
            num_latent_conditional_frames : Number of latent conditional frames. Defaults to 1.
            resolution: Target video resolution in "H,W" format. Defaults to "192,320".
            seed: Random seed for reproducibility. Defaults to 1.
            negative_prompt: Custom negative prompt. Defaults to the predefined default negative prompt.
            camera: Target camera extrinsics and intrinsics for the K output videos. Must be provided if model is camera conditioned.
            action: Target robot action for the K output videos. Must be provided if model is action conditioned.
            num_steps: Number of generation steps. Defaults to 35.
            offload_diffusion_model: If True, offload diffusion model to CPU to save GPU memory. Defaults to False.
            offload_text_encoder: If True, offload text encoder to CPU to save GPU memory. Defaults to False.
            offload_tokenizer: If True, offload tokenizer to CPU to save GPU memory. Defaults to False.

        Returns:
            torch.Tensor: The generated video tensor (B, C, T, H, W) in the range [-1, 1].
        """
        assert camera is not None or action is not None or num_input_video == 1 and num_output_video == 1, (
            "expected num_output_video==1 and num_output_video==1 for no camera conditioning or action conditioning"
        )

        # Parse resolution string into tuple of integers
        if resolution == "none":
            h, w = self.model.get_video_height_width()
            video_resolution = (h, w)
        else:
            video_resolution = resolution.split(",")
            video_resolution = tuple([int(x) for x in video_resolution])
            assert len(video_resolution) == 2, "Resolution must be in 'H,W' format"

        # Get the correct number of frames needed by the model
        model_required_frames = self.model.tokenizer.get_pixel_num_frames(self.model.config.state_t)

        # Determine if input is image or video and process accordingly
        if input_path is None or num_latent_conditional_frames == 0:
            vid_input = torch.zeros(1, 3, model_required_frames, video_resolution[0], video_resolution[1]).to(
                torch.uint8
            )
        elif isinstance(input_path, str):
            ext = os.path.splitext(input_path)[1].lower()
            if ext in _IMAGE_EXTENSIONS:
                log.info(f"Processing image input: {input_path}")
                vid_input = read_and_process_image(
                    img_path=input_path,
                    resolution=video_resolution,
                    num_video_frames=model_required_frames,
                    resize=True,
                )
            elif ext in _VIDEO_EXTENSIONS:
                log.info(f"Processing video input: {input_path}")
                vid_input = read_and_process_video(
                    video_path=input_path,
                    resolution=video_resolution,
                    num_video_frames=model_required_frames,
                    num_latent_conditional_frames=num_latent_conditional_frames,
                    resize=True,
                )
            else:
                raise ValueError(
                    f"Unsupported file extension: {ext}. Supported extensions: {_IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS}"
                )
        elif isinstance(input_path, torch.Tensor):
            vid_input = input_path
        else:
            raise ValueError(f"Unsupported input_path type: {type(input_path)}")

        # Prepare the data batch with text embeddings
        # Note: TextEncoder.compute_text_embeddings_online() will automatically move its model to GPU
        data_batch = self._get_data_batch_input(
            video=vid_input,
            prompt=prompt,
            camera=camera,
            action=action,
            num_conditional_frames=num_latent_conditional_frames,
            negative_prompt=negative_prompt,
            use_neg_prompt=True,
        )

        mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        log.info(f"GPU memory usage after getting data_batch: {mem_bytes / (1024**3):.2f} GB")

        # Memory Optimization Step 1: Offload Text Encoder
        # Offload text encoder after computing embeddings to free memory
        if offload_text_encoder and self.model.text_encoder is not None:
            log.info("[Memory Optimization] Offloading text encoder to CPU")
            # TextEncoder is a wrapper class with self.model (the actual neural network)
            if hasattr(self.model.text_encoder, "model") and self.model.text_encoder.model is not None:
                self.model.text_encoder.model = self.model.text_encoder.model.to("cpu")
            torch.cuda.empty_cache()

        # Memory Optimization Step 2: Tokenizer Encoder
        # Load tokenizer encoder to GPU for encoding input video
        if offload_tokenizer:
            log.info("[Memory Optimization] Loading tokenizer encoder to GPU")
            if hasattr(self.model.tokenizer, "encoder") and self.model.tokenizer.encoder is not None:
                self.model.tokenizer.encoder = self.model.tokenizer.encoder.to("cuda")
            torch.cuda.empty_cache()

        # Memory Optimization Step 3: Diffusion Network
        # Load the main diffusion network to GPU for sampling
        if offload_diffusion_model:
            log.info("[Memory Optimization] Loading diffusion network to GPU")
            self.model.net = self.model.net.to("cuda")
            # Also load conditioner if it exists
            if hasattr(self.model, "conditioner") and self.model.conditioner is not None:
                self.model.conditioner = self.model.conditioner.to("cuda")
            torch.cuda.empty_cache()

        extra_kwargs = {}
        if camera is not None:
            extra_kwargs = {
                "num_input_video": num_input_video,
                "num_output_video": num_output_video,
            }

        # Generate latent samples using the diffusion model
        # Video should be of shape torch.Size([1, 3, 93, 192, 320]) # Note: Shape check comment
        log.info("[Memory Optimization] Starting latent sample generation")
        sample = self.model.generate_samples_from_batch(
            data_batch,
            n_sample=1,  # Generate one sample
            guidance=guidance,
            seed=seed,  # Fixed seed for reproducibility
            is_negative_prompt=True,  # Use classifier-free guidance
            num_steps=num_steps,
            **extra_kwargs,
        )

        # Memory Optimization Step 4: Offload Diffusion Network
        # Offload diffusion network after sampling to make room for decoder
        if offload_diffusion_model:
            log.info("[Memory Optimization] Offloading diffusion network to CPU")
            self.model.net = self.model.net.to("cpu")
            if hasattr(self.model, "conditioner") and self.model.conditioner is not None:
                self.model.conditioner = self.model.conditioner.to("cpu")
            # Also offload encoder since we only need decoder now
            if hasattr(self.model.tokenizer, "encoder") and self.model.tokenizer.encoder is not None:
                self.model.tokenizer.encoder = self.model.tokenizer.encoder.to("cpu")
            torch.cuda.empty_cache()

        # Memory Optimization Step 5: Load Decoder
        # Load tokenizer decoder to GPU for decoding latents
        if offload_tokenizer:
            log.info("[Memory Optimization] Loading tokenizer decoder to GPU")
            if hasattr(self.model.tokenizer, "decoder") and self.model.tokenizer.decoder is not None:
                self.model.tokenizer.decoder = self.model.tokenizer.decoder.to("cuda")
            torch.cuda.empty_cache()

        # Decode the latent samples
        if isinstance(sample, list):
            # Decode the latent sample into a video tensor
            video_list = []
            for sample_chunk in sample:
                video_chunk = self.model.decode(sample_chunk)
                video_list.append(video_chunk)
            video = torch.cat(video_list, dim=3)
        else:
            # Decode the latent sample into a video tensor
            video = self.model.decode(sample)

        # Memory Optimization Step 6: Final Cleanup
        # Offload decoder after decoding
        if offload_tokenizer:
            log.info("[Memory Optimization] Offloading tokenizer decoder to CPU")
            if hasattr(self.model.tokenizer, "decoder") and self.model.tokenizer.decoder is not None:
                self.model.tokenizer.decoder = self.model.tokenizer.decoder.to("cpu")
            torch.cuda.empty_cache()

        return video

    def cleanup(self):
        """Clean up distributed resources."""
        if self.context_parallel_size > 1:
            import torch.distributed as dist
            from megatron.core import parallel_state

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
