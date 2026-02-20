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

import contextlib
import os
import random
import time
from typing import Optional, Union

import torch

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import distributed, log, misc
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.interactive.utils.model_loader import (
    load_model_from_checkpoint as load_model_interactive,
)
from cosmos_predict2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_predict2._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY
from cosmos_predict2._src.predict2.utils.model_loader import (
    load_model_from_checkpoint as load_model_predict2,
)
from cosmos_predict2._src.transfer2.datasets.augmentors.control_input import get_augmentor_for_eval
from cosmos_predict2._src.transfer2.inference.utils import (
    get_t5_from_prompt,
    normalized_float_to_uint8,
    read_and_process_control_input,
    read_and_process_image_context,
    read_and_process_video,
    reshape_output_video_to_input_resolution,
    uint8_to_normalized_float,
)


def _maybe_get_timer(
    benchmark_timer: Optional[misc.TrainingTimer], func_name: str
) -> contextlib.nullcontext | misc.TrainingTimer:
    return benchmark_timer(func_name) if benchmark_timer is not None else contextlib.nullcontext()


class ControlVideo2WorldInference:
    """
    Handles the Control2Video inference process, including model loading, data preparation,
    and video transfer from an input video and text prompt.
    """

    def __init__(
        self,
        registered_exp_name: str,
        checkpoint_paths: Union[str, list[str]],
        s3_credential_path: str,
        exp_override_opts: Optional[list[str]] = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        cache_dir: Optional[str] = None,
        skip_load_model: bool = False,
        base_load_from: Optional[str] = None,
        use_cp_wan: bool = False,
        wan_cp_grid: tuple[int, int] = (-1, -1),
        benchmark_timer: Optional[misc.TrainingTimer] = None,
        cache_text_encoder: bool = True,
        use_cuda_graphs: bool = False,
        cfg_parallel: bool = False,
        hierarchical_cp: bool = False,
        config_file: str = "cosmos_predict2/_src/transfer2/configs/vid2vid_transfer/config.py",
    ):
        """
        Initializes the ControlVideo2WorldInference class.

        Loads the diffusion model and its configuration based on the provided
        experiment name and checkpoint path.

        Args:
            registered_exp_name (str): Name of the experiment configuration.
            checkpoint_paths (Union[str, list[str]]): Single checkpoint path or List of checkpoint paths for multi-branch models.
            s3_credential_path (str): Path to S3 credentials file for ckpt & negative embedding (if loading from S3).
            exp_override_opts (list[str]): List of experiment override options.
            process_group (torch.distributed.ProcessGroup): Process group for distributed training.
            cache_dir (str): Cache directory for storing pre-computed embeddings.
            skip_load_model (bool): Whether to skip loading model from checkpoint for multi-control models.
            use_cp_wan (bool, optional): Whether to use parallel tokenizer. Defaults to False.
            wan_cp_grid (tuple[int, int], optional): The grid for parallel tokenizer. Used only when use_cp_wan is True. Defaults to (1, cp_size).
            use_cuda_graphs (bool, optional): Whether to use CUDA Graphs for inference. Defaults to False.
            cfg_parallel (bool, optional): Whether to parallelize Classifier Free Guidance.
            hierarchical_cp (bool, optional): Whether to use hierarchical CP algorithm.
        """
        self.registered_exp_name = registered_exp_name
        self.checkpoint_path = checkpoint_paths if isinstance(checkpoint_paths, str) else checkpoint_paths[0]
        self.s3_credential_path = s3_credential_path
        self.cache_dir = cache_dir
        self.cache_text_encoder = cache_text_encoder
        if exp_override_opts is None:
            exp_override_opts = []
        # no need to load base model separately at inference
        exp_override_opts.append("++model.config.base_load_from=null")
        if use_cuda_graphs:
            exp_override_opts.append("model.config.net.use_cuda_graphs=True")
        # data_train is training-only; distilled interactive inference does not define it,
        # so avoid deleting it for interactive configs to prevent Hydra errors.
        if not INTERNAL and (not config_file or "interactive" not in config_file):
            exp_override_opts.append("~data_train")
        if hierarchical_cp:
            exp_override_opts.append("model.config.net.atten_backend='transformer_engine'")
        local_cache_dir = cache_dir if not (isinstance(checkpoint_paths, list) and len(checkpoint_paths) > 1) else None
        # Load the model and config. Each trained model's config is composed by
        # loading a pre-registered experiment config, and then (optionally) overriding with some command-line
        # arguments. That is done in experiment_list.py. Here we simply replicate that process.
        # Use interactive loader for distilled models (interactive config), predict2 loader otherwise
        if config_file and "interactive" in config_file:
            model, config = load_model_interactive(
                experiment_name=self.registered_exp_name,
                s3_checkpoint_dir=self.checkpoint_path,
                config_file=config_file,
                load_ema_to_reg=True,
                local_cache_dir=local_cache_dir,  # for multi-control models, need to load other branches before caching
                experiment_opts=exp_override_opts,
            )
        else:
            model, config = load_model_predict2(
                experiment_name=self.registered_exp_name,
                s3_checkpoint_dir=self.checkpoint_path,
                config_file=config_file,
                load_ema_to_reg=True,
                local_cache_dir=local_cache_dir,  # for multi-control models, need to load other branches before caching
                experiment_opts=exp_override_opts,
                cache_text_encoder=self.cache_text_encoder,
            )
        if (
            isinstance(checkpoint_paths, list) and len(checkpoint_paths) > 1 and not skip_load_model
        ):  # load other branches for multi-control models
            load_from_local = False
            if cache_dir is not None:
                # build a unique path for s3checkpoint dir
                local_s3_ckpt_fp = os.path.join(
                    cache_dir,
                    self.checkpoint_path.split("s3://")[1],
                    "torch_model",
                    f"_rank_{distributed.get_rank()}.pt",
                )
                if os.path.exists(local_s3_ckpt_fp):
                    load_from_local = True

            if load_from_local:
                log.info(f"Loading model cached locally from {local_s3_ckpt_fp}")
                model.load_state_dict(easy_io.load(local_s3_ckpt_fp))
            else:
                model.load_multi_branch_checkpoints(checkpoint_paths=checkpoint_paths)
                if cache_dir is not None:
                    log.info(f"Caching model state dict to {local_s3_ckpt_fp}")
                    easy_io.dump(model.state_dict(), local_s3_ckpt_fp)

        if base_load_from is not None:
            log.info(f"Loading base model from {base_load_from}")
            model.config.base_load_from = {
                "load_path": base_load_from,
                "credentials": s3_credential_path,
            }
            model.load_base_model(load_ema_to_reg=True)

        # Get text_encoder_class from model or config depending on model type
        if hasattr(model, "text_encoder_class"):
            self.text_encoder_class = model.text_encoder_class
        else:
            # For distilled models (DMD2Model), get from config
            self.text_encoder_class = config.model.config.text_encoder_class

        if process_group is not None:
            cp_comm_type = "a2a+p2p" if hierarchical_cp else "p2p"
            log.info(f"Enabling CP in base model with {cp_comm_type}\n")
            model.net.enable_context_parallel(process_group, cfg_parallel=cfg_parallel, cp_comm_type=cp_comm_type)

            cp_size = process_group.size()

            if use_cp_wan:
                wan_cp_grid = wan_cp_grid if wan_cp_grid != (-1, -1) else (1, cp_size)
                assert wan_cp_grid[0] * wan_cp_grid[1] == cp_size, (
                    "Parallel Tokenizer grid needs to multiply to CP size."
                )

                model.tokenizer.model.model = model.tokenizer.model.model.to("cuda")
                model.tokenizer.initialize_context_parallel(process_group, wan_cp_grid)

        self.model = model
        self.config = config
        self.batch_size = 1
        self.benchmark_timer = benchmark_timer

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        prev_output: torch.Tensor,
        text_embedding: torch.Tensor,
        fps: int,
        negative_prompt: str = None,
        control_weight: str = "1.0",
        image_context: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str): The text prompt for conditioning.

            image_context (torch.Tensor, optional): Image context tensor for conditioning. Can be (B, C, H, W).

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = prev_output.shape
        input_key = "video" if T > 1 else "images"

        data_batch = {
            "dataset_name": "video_data",
            input_key: prev_output.squeeze(2),
            "t5_text_embeddings": text_embedding,  # positive prompt embedding. Name has t5 but also supports Reason1.
            "fps": torch.randint(16, 32, (self.batch_size,)).cuda(),  # Random FPS (might be used by model)
            "padding_mask": torch.zeros(
                self.batch_size, 1, H, W, device="cuda"
            ),  # Padding mask (assumed no padding here)
            "num_conditional_frames": 1,  # Specify that the first frame is conditional
            "control_weight": [float(w) for w in control_weight.split(",")],
            "input_video": video,
        }

        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.to(dtype=torch.bfloat16, device="cuda", non_blocking=True)

        # Add image context
        if image_context is not None:
            data_batch["image_context"] = image_context.to(
                dtype=torch.bfloat16, device="cuda", non_blocking=True
            ).contiguous()

        # Handle negative prompts for classifier-free guidance
        if negative_prompt is not None:
            assert self.neg_t5_embeddings is not None, "Negative prompt embedding is not computed."
            data_batch["neg_t5_text_embeddings"] = self.neg_t5_embeddings

        return data_batch

    def _get_num_chunks(
        self, input_frames: torch.Tensor, num_video_frames_per_chunk: int, num_conditional_frames: int
    ) -> tuple[int, int, int]:
        """
        Get the number of chunks for chunk-wise long video generation.
        """
        # Frame number settting for chunk-wise long video generation
        num_total_frames = input_frames.shape[1]
        num_frames_per_chunk = num_video_frames_per_chunk - num_conditional_frames
        if num_video_frames_per_chunk == 1:
            num_chunks = 1
        else:
            num_generated_frames_vid2vid = num_total_frames - num_video_frames_per_chunk
            num_chunks = 1 + num_generated_frames_vid2vid // num_frames_per_chunk
            if num_generated_frames_vid2vid % num_frames_per_chunk != 0:
                num_chunks += 1

        return num_total_frames, num_chunks, num_frames_per_chunk

    def _pad_input_frames(
        self,
        input_frames: torch.Tensor,
        num_total_frames: int,
        num_video_frames_per_chunk: int,
        padding_mode: str = "reflect",
    ) -> torch.Tensor:
        """
        Pad input frames if total frames is less than chunk size
        """
        if num_total_frames < num_video_frames_per_chunk:
            # Check whether the input_frames is empty. If so, there is nothing to pad.
            if num_total_frames == 0:
                raise ValueError("No input frames; cannot pad. Verify that video frame counts match.")
            if padding_mode == "repeat":
                last_frame = input_frames[:, -1:, :, :]  # Get the last frame
                padding = last_frame.repeat(1, num_video_frames_per_chunk - num_total_frames, 1, 1)
                input_frames = torch.cat([input_frames, padding], dim=1)
            elif padding_mode == "reflect":
                while input_frames.shape[1] < num_video_frames_per_chunk:
                    padding = min(input_frames.shape[1] - 1, num_video_frames_per_chunk - input_frames.shape[1])
                    padding_frames = input_frames.flip(dims=[1])[:, :padding, :, :]
                    input_frames = torch.cat([input_frames, padding_frames], dim=1)
            else:
                raise ValueError(f"Invalid padding mode: {padding_mode}")
        return input_frames

    @torch.no_grad()
    def generate_img2world(
        self,
        prompt: str | torch.Tensor | list[str] | dict[str, str],
        video_path: str,
        guidance: int = 7,
        seed: int = 1,
        resolution: str = "720",
        num_conditional_frames: int = 1,
        num_video_frames_per_chunk: int = 93,
        num_steps: int = 35,
        control_weight: str = "1.0",
        sigma_max: float | None = None,
        hint_key: list[str] = ["edge"],
        preset_edge_threshold: str = "medium",
        preset_blur_strength: str = "medium",
        seg_control_prompt: str | None = None,
        input_control_video_paths: dict[str, str] | None = None,
        show_control_condition: bool = False,
        show_input: bool = False,
        image_context_path: Optional[str] = None,
        keep_input_resolution: bool = True,
        negative_prompt: str | None = None,
        max_frames: int | None = None,
        context_frame_idx: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], int, tuple[int, int]]:
        """
        Generates a video based on an input video and text prompt.
        Supports chunk-wise long video generation.

        Args:
            prompt (str): The text prompt describing the desired video content/style.
            video_path (str): Path to the input conditional video.
            guidance (int, optional): Classifier-free guidance scale. Defaults to 7.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            resolution (str, optional): Resolution of the video (720-default, 480, etc). Defaults to 720.
            image_context_path (str, optional): Path to image file to use as image context. If None, uses random frame from video. Will be ignored and use input video if context_frame_idx is provided.
            keep_input_resolution (bool, optional): Whether to keep the exact dimension of the. Defaults to True.
            negative_prompt (str, optional): Negative prompt for classifier-free guidance. Defaults to None.
            max_frames (int, optional): Maximum number of frames to read from the video. Defaults to None. 1 for image.
            context_frame_idx (int, optional): Frame index of the input video to use as image context. Defaults to None. In this case, can still use image_context_path to provide image context.
        Returns:
            torch.Tensor: The generated video tensor (B, C, T, H, W) in the range [-1, 1].
            dict[str, torch.Tensor]: Dictionary mapping hint key to the corresponding control input video tensor.
            int: Frames per second of the original input video.
            tuple[int, int]: Original height and width of the input video.

        Raises:
            ValueError: If the input video is empty or invalid.
        """
        # --------Input processing--------
        # Process input video and get meta info.
        log.info("Loading input video...")
        # aspect_ratio is width / height
        # input_frames is (C, T, H, W)
        input_frames, fps, aspect_ratio, original_hw = read_and_process_video(
            video_path, resolution=resolution, max_frames=max_frames
        )
        if input_frames.shape[1] == 0:
            raise ValueError("Input video is empty")

        # Get text context embeddings
        log.info("Computing prompt text embeddings...")
        with _maybe_get_timer(self.benchmark_timer, "get_text_embeddings"):
            if self.text_encoder_class == "T5":
                text_embeddings = get_t5_from_prompt(prompt, text_encoder_class="T5", cache_dir=self.cache_dir)
            else:
                text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                    {"ai_caption": [prompt], "images": None}, input_caption_key="ai_caption"
                )
            if negative_prompt:
                log.info("Computing negative prompt text embeddings...")
                if self.text_encoder_class == "T5":
                    neg_text_embeddings = get_t5_from_prompt(
                        negative_prompt, text_encoder_class="T5", cache_dir=self.cache_dir
                    )
                else:
                    neg_text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                        {"ai_caption": [negative_prompt], "images": None}, input_caption_key="ai_caption"
                    )
                self.neg_t5_embeddings = neg_text_embeddings

        # Process image context if provided; else will be None
        log.info("Processing image context if available...")
        with _maybe_get_timer(self.benchmark_timer, "preprocessing"):
            if context_frame_idx is not None:
                image_context_path = video_path
                log.info(f"Using context frame index: {context_frame_idx} from video path: {video_path}")
            image_context = read_and_process_image_context(
                image_context_path,
                resolution=(VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]),
                resize=True,
                context_frame_idx=context_frame_idx,
            )
            # Load control inputs from paths, or optionally compute on-the-fly, and add to data batch.
            log.info("Loading control inputs...")
            control_input_dict, mask_video_dict = read_and_process_control_input(
                video_path=video_path,
                input_control_paths=input_control_video_paths,
                hint_key=hint_key,
                resolution=resolution,
                seg_control_prompt=seg_control_prompt,
            )

            # -------- Stuff to handle chunk-wise long video generation --------
            num_total_frames, num_chunks, num_frames_per_chunk = self._get_num_chunks(
                input_frames, num_video_frames_per_chunk, num_conditional_frames
            )
            # Pad input frames if total frames is less than chunk size
            input_frames = self._pad_input_frames(input_frames, num_total_frames, num_video_frames_per_chunk)
            all_chunks, time_per_chunk = [], []
            # Initialize control_video_dict to accumulate control inputs across chunks
            control_video_dict = {}
            all_control_chunks = {key: [] for key in hint_key}
            # For first chunk, use zeros as input (after normalization it is 0)
            prev_output = torch.zeros_like(input_frames[:, :num_video_frames_per_chunk]).to(torch.uint8).cuda()[None]

        # --------Start of chunk-wise long video generation--------
        self.model.eval()
        for chunk_id in range(num_chunks):
            log.info(f"Generating chunk {chunk_id + 1}/{num_chunks}")
            with _maybe_get_timer(self.benchmark_timer, "generate_chunk"):
                start_time = time.perf_counter()

                # Calculate start frame for this chunk
                chunk_start_frame = chunk_id * num_frames_per_chunk
                chunk_end_frame = min(chunk_start_frame + num_video_frames_per_chunk, input_frames.shape[1])

                x_sigma_max = None
                if input_frames is not None:
                    cur_input_frames = input_frames[:, chunk_start_frame:chunk_end_frame]
                    cur_input_frames = self._pad_input_frames(
                        cur_input_frames, cur_input_frames.shape[1], num_video_frames_per_chunk
                    )
                    if sigma_max is not None:
                        x0 = uint8_to_normalized_float(cur_input_frames, dtype=torch.bfloat16)[None].cuda(
                            non_blocking=True
                        )
                        x0 = self.model.encode(x0).contiguous()
                        x_sigma_max = self.model.get_x_from_clean(x0, sigma_max, seed=(seed + chunk_id))

                if isinstance(text_embeddings, list):
                    text_emb_idx = min(chunk_id, len(text_embeddings) - 1)
                    text_embedding = text_embeddings[text_emb_idx]
                else:
                    text_embedding = text_embeddings

                # Prepare the data batch with current input. Note: this doesn't include control inputs yet.
                data_batch = self._get_data_batch_input(
                    cur_input_frames,
                    prev_output,
                    text_embedding,
                    fps,
                    negative_prompt=negative_prompt,
                    control_weight=control_weight,
                    image_context=image_context,
                )

                # Process control inputs as specified in the hint_key list.
                # If pre-computed control inputs are provided, load them into the data batch.
                for k, v in control_input_dict.items():
                    cur_control_input = v[:, chunk_start_frame:chunk_end_frame]
                    data_batch[k] = self._pad_input_frames(
                        cur_control_input, cur_control_input.shape[1], num_video_frames_per_chunk
                    )
                    if k == "control_input_inpaint_mask":
                        data_batch["control_input_inpaint"] = cur_input_frames
                # Otherwise, compute control inputs on-the-fly via the augmentor（applicable to edge and vis).
                data_batch = get_augmentor_for_eval(
                    data_dict=data_batch,
                    input_keys=["input_video"],
                    output_keys=hint_key,
                    preset_edge_threshold=preset_edge_threshold,
                    preset_blur_strength=preset_blur_strength,
                )

                if chunk_id == 0:
                    data_batch[NUM_CONDITIONAL_FRAMES_KEY] = 0
                else:
                    data_batch[NUM_CONDITIONAL_FRAMES_KEY] = (
                        1 + (num_conditional_frames - 1) // 4
                    )  # tokenizer temporal compression is 4x

                random.seed(seed)
                seed = random.randint(0, 1000000)
                log.info(f"Seed: {seed}")

                sample = self.model.generate_samples_from_batch(
                    data_batch,
                    n_sample=1,
                    guidance=guidance,
                    seed=seed,
                    is_negative_prompt=negative_prompt is not None,
                    x_sigma_max=x_sigma_max,
                    sigma_max=sigma_max,
                    num_steps=num_steps,
                )
                video = self.model.decode(sample).cpu()  # Shape: (1, C, T, H, W)

                # For visualization: concatenate condition and input videos with generated video
                video_cat = video
                conditions = []
                if show_input and input_frames is not None:
                    x0 = uint8_to_normalized_float(cur_input_frames, dtype=torch.bfloat16)[None]
                    x0 = x0.to(device=video_cat.device)
                    video_cat = torch.cat([x0, video_cat], dim=-1)

                # Accumulate control inputs for each chunk
                for key in hint_key:
                    control_input = data_batch["control_input_" + key]
                    if f"control_input_{key}_mask" in data_batch:
                        mask = data_batch[f"control_input_{key}_mask"].to(device=control_input.device)
                        control_input = (control_input + 1) / 2 * mask * 2 - 1

                    # Store control input for this chunk
                    if chunk_id == 0:
                        all_control_chunks[key].append(control_input)
                    else:
                        # For subsequent chunks, only append the non-overlapping frames
                        all_control_chunks[key].append(control_input[:, :, num_conditional_frames:, :, :])

                    if show_control_condition:
                        conditions += [control_input.to(device=video_cat.device)]

                if show_control_condition:
                    video_cat = torch.cat([*conditions, video_cat], dim=-1)

                if chunk_id == 0:
                    all_chunks.append(video_cat)
                else:
                    # For subsequent chunks, only append the non-overlapping frames
                    all_chunks.append(video_cat[:, :, num_conditional_frames:, :, :])

                # For next chunk, use last conditional_frames as input
                if chunk_id < num_chunks - 1:  # Don't need to prepare next input for last chunk
                    last_frames = video[
                        :, :, video.shape[2] - num_conditional_frames :, :, :
                    ]  # (1, C, num_conditional_frames, H, W)
                    # Convert to uint8 [0, 255]
                    last_frames_uint8 = normalized_float_to_uint8(last_frames)
                    # Create blank frames for the rest
                    blank_frames = torch.zeros(
                        (
                            1,
                            3,
                            num_video_frames_per_chunk - num_conditional_frames,
                            video.shape[-2],
                            video.shape[-1],
                        ),
                        dtype=torch.uint8,
                        device=video.device,
                    )
                    prev_output = torch.cat([last_frames_uint8, blank_frames], dim=2)
                end_time = time.perf_counter()
                time_per_chunk.append(end_time - start_time)

        with _maybe_get_timer(self.benchmark_timer, "postprocessing"):
            # Concatenate all chunks along time
            full_video = torch.cat(all_chunks, dim=2)  # (1, C, T, H, W)
            # Keep only the original number of frames
            full_video = full_video[:, :, :num_total_frames, :, :]

            full_video = full_video.cpu()
            # Concatenate all control chunks and trim to original frames
            for key in hint_key:
                if all_control_chunks[key]:
                    control_video_dict[key] = torch.cat(all_control_chunks[key], dim=2)  # (1, C, T, H, W)
                    # Keep only the original number of frames
                    control_video_dict[key] = control_video_dict[key][:, :, :num_total_frames, :, :]

            if keep_input_resolution:
                # reshape output video to match the input video resolution
                full_video = reshape_output_video_to_input_resolution(
                    full_video, hint_key, show_control_condition, show_input, original_hw
                )
                # Also resize control videos to match input resolution
                for key in hint_key:
                    if key in control_video_dict and control_video_dict[key] is not None:
                        control_video_dict[key] = reshape_output_video_to_input_resolution(
                            control_video_dict[key], [key], False, False, original_hw
                        )
        log.info(f"Average time per chunk: {sum(time_per_chunk) / len(time_per_chunk)}")
        return full_video, control_video_dict, mask_video_dict, fps, original_hw
