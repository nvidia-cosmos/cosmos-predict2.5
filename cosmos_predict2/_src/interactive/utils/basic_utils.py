# -----------------------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Code for building fast generative models for a variety of tasks
# from Fundamental Generative AI Research (GenAIR) team
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch

PRECISION_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


# def basic_clean(text):
#     """
#     Clean text by fixing encoding issues and unescaping HTML entities.
#     """
#     text = ftfy.fix_text(text)
#     text = html.unescape(html.unescape(text))
#     return text.strip()


# def whitespace_clean(text):
#     """
#     Clean text by replacing multiple spaces with a single space and removing leading/trailing whitespace.
#     """
#     text = re.sub(r"\s+", " ", text)
#     text = text.strip()
#     return text


# def prompt_clean(text):
#     """
#     Clean text by applying basic and whitespace cleaning.
#     """
#     text = whitespace_clean(basic_clean(text))
#     return text


# def ensure_trailing_slash(s):
#     return s if s.endswith("/") else s + "/"


# def get_batch_size_total(config: BaseConfig):
#     # accumulated batch size per GPU
#     batch_size = config.dataloader_train.batch_size * config.trainer.grad_accum_rounds
#     return batch_size * world_size()


# def to_str(obj: Any) -> str | Dict[Any, str]:
#     """Print the object in a readable format. Typically used for batches of data."""
#     if isinstance(obj, torch.Tensor):
#         return f"Tensor{list(obj.shape)}"
#     elif isinstance(obj, str):
#         dots = "..." if len(obj) > 10 else ""
#         return f"{dots}{obj[-10:]}"
#     elif isinstance(obj, Mapping):
#         return {k: to_str(v) for k, v in obj.items()}
#     elif isinstance(obj, Iterable):
#         return str([to_str(v) for v in obj])
#     return str(obj)


# def set_random_seed(
#     seed: int, iteration: int = 0, by_rank: bool = False, devices: List[torch.device | str | int] | None = None
# ) -> None:
#     """Set random seed for `random, numpy, Pytorch, cuda`.

#     Args:
#         seed (int): Random seed.
#         by_rank (bool): if set to true, each GPU will use a different random seed.
#         devices (List[torch.device] | None): devices to set the seed on. If None, will set the seed on all devices.
#     """
#     seed += iteration
#     if by_rank:
#         seed += get_rank()
#     seed %= 1 << 31
#     logger.info(f"Using random seed {seed}.")
#     random.seed(seed)
#     np.random.seed(seed)
#     if devices is None:
#         # sets seed on the current CPU & all GPUs
#         torch.manual_seed(seed)
#     else:
#         # set the seed on cpu
#         torch.default_generator.manual_seed(seed)
#         # set the seed on devices
#         for device in devices:
#             # get device index (as in torch.cuda.set_rng_state)
#             if isinstance(device, str):
#                 device = torch.device(device)
#             elif isinstance(device, int):
#                 device = torch.device("cuda", device)
#             idx = device.index
#             if idx is None:
#                 idx = torch.cuda.current_device()
#             torch.cuda.default_generators[idx].manual_seed(seed)


# @contextlib.contextmanager
# def set_tmp_random_seed(
#     seed, iteration: int = 0, by_rank: bool = False, devices: List[torch.device | str | int] | None = None
# ):
#     """A context manager to temporarily set the random seeds.

#     Args:
#         seed (int): Random seed.
#         iteration (int): Iteration number.
#         by_rank (bool): if set to true, each GPU will use a different random seed.
#         devices (List[torch.device] | None): devices to set the seed on. If None, will set the seed on all devices.
#     """
#     if seed is None:
#         yield
#         return

#     # Save the original random states
#     np_state = np.random.get_state()
#     py_state = random.getstate()

#     try:
#         # Fork torch state
#         with torch.random.fork_rng(devices=devices):
#             # Set the new seeds
#             set_random_seed(seed, iteration=iteration, by_rank=by_rank, devices=devices)
#             yield
#     finally:
#         # Restore the original random states
#         np.random.set_state(np_state)
#         random.setstate(py_state)


# def to(
#     data: Any,
#     device: str | torch.device | None = None,
#     dtype: torch.dtype | None = None,
# ) -> Any:
#     """Recursively cast data into the specified device, dtype, and/or memory_format.

#     The input data can be a tensor, a list of tensors, a dict of tensors.
#     See the documentation for torch.Tensor.to() for details.

#     Args:
#         data (Any): Input data.
#         device (str | torch.device): GPU device (default: None).
#         dtype (torch.dtype): data type (default: None).

#     Returns:
#         data (Any): Data cast to the specified device, dtype, and/or memory_format.
#     """
#     assert device is not None or dtype is not None, "at least one of device, dtype should be specified"
#     if isinstance(data, torch.Tensor):
#         is_cpu = (isinstance(device, str) and device == "cpu") or (
#             isinstance(device, torch.device) and device.type == "cpu"
#         )
#         if data.dtype == torch.int64:
#             # t variable is int64 for some networks (e.g. CogVideoX, Stable Diffusion)
#             dtype = torch.int64

#         data = data.to(
#             device=device,
#             dtype=dtype,
#             non_blocking=(not is_cpu),
#         )
#         return data
#     elif isinstance(data, (list, tuple)):
#         return type(data)(to(d, device, dtype) for d in data)
#     elif isinstance(data, dict):
#         return {k: to(v, device, dtype) for k, v in data.items()}
#     else:
#         return data


# def convert_cfg_to_dict(cfg) -> dict:
#     """Convert config to dictionary, handling both OmegaConf and attrs cases.

#     Args:
#         cfg: Either a DictConfig (from OmegaConf/Hydra) or Config (attrs class)

#     Returns:
#         Dictionary representation of the config
#     """
#     if isinstance(cfg, DictConfig):
#         # Production case: OmegaConf DictConfig
#         return OmegaConf.to_container(cfg, resolve=True)
#     else:
#         # Test case: attrs SampleTConfig class
#         return attrs.asdict(cfg)


# def detach(
#     data: Any,
# ) -> Any:
#     """Recursively detach data if it is a tensor.

#     Args:
#         data (Any): Input data.
#     Returns:
#         data (Any): Data detached from the computation graph.
#     """
#     if isinstance(data, torch.Tensor):
#         return data.detach()
#     elif isinstance(data, (list, tuple)):
#         return type(data)(detach(d) for d in data)
#     elif isinstance(data, dict):
#         return {k: detach(v) for k, v in data.items()}
#     else:
#         return data


# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ("yes", "true", "t", "1"):
#         return True
#     elif v.lower() in ("no", "false", "f", "0"):
#         return False
#     else:
#         raise ValueError("Boolean value expected.")


# def save_video(
#     video: torch.Tensor,
#     vae: WanVideoEncoder = None,
#     save_name: str = "sample0.mp4",
#     save_as_gif: bool = True,
#     fps: int = 16,
#     quality: int = 23,
#     debug_shapes: bool = False,
#     **kwargs,
# ):
#     """
#     Save video with basic quality control and silent encoding.

#     Args:
#         vae: Video encoder for decoding latents to frames
#         video: Video tensor to save [B, C, T, H, W]
#         save_name: Full path including filename
#         save_as_gif: Whether to save as GIF or MP4
#         fps: Frames per second for playback (not frame count)
#         quality: Video quality 0-51 (lower=better, default: 23)
#         debug_shapes: Print tensor shapes for debugging
#         **kwargs: Additional encoding parameters
#     """
#     if debug_shapes:
#         print(f"🔍 Video tensor input shape: {video.shape}")

#     frames = video
#     if vae is not None:
#         frames = vae.decode(frames)[0]
#         if debug_shapes:
#             print(f"🔍 After VAE decode shape: {frames.shape}")

#     frames = rearrange(frames, "C T H W -> T H W C")

#     if debug_shapes:
#         print(f"🔍 After rearrange shape: {frames.shape}")
#         print(f"🔍 Final frame count: {frames.shape[0]} frames")
#         print(f"🔍 Expected duration at {fps}fps: {frames.shape[0] / fps:.1f} seconds")

#     frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().to(dtype=torch.uint8)

#     # Ensure save directory exists
#     os.makedirs(os.path.dirname(save_name), exist_ok=True)

#     if save_as_gif:
#         # Save as GIF with proper extension handling
#         gif_filename = os.path.splitext(save_name)[0] + ".gif"
#         iio.imwrite(
#             gif_filename,
#             frames,
#             fps=fps,
#             loop=kwargs.get("loop", 0),
#             quantizer=kwargs.get("quantizer", "nq"),
#         )
#     else:
#         # Save as MP4 with silent encoding and quality control
#         output_params = [
#             "-loglevel",
#             "quiet",  # Silent encoding
#             "-hide_banner",  # No ffmpeg banner
#             "-nostats",  # No encoding stats
#             "-crf",
#             str(quality),  # Quality setting
#             "-preset",
#             kwargs.get("preset", "medium"),  # Encoding speed/quality balance
#         ]

#         iio.imwrite(
#             save_name,
#             frames,
#             fps=fps,
#             codec="libx264",  # Reliable, widely supported codec
#             output_params=output_params,
#         )


# def clear_gpu_memory():
#     """
#     Aggressively clear GPU memory and force garbage collection.

#     This function performs comprehensive memory cleanup including:
#     - PyTorch CUDA cache clearing
#     - GPU synchronization
#     - Python garbage collection
#     - Memory defragmentation
#     """
#     if torch.cuda.is_available():
#         # Clear PyTorch's CUDA cache
#         torch.cuda.empty_cache()

#         # Wait for all CUDA operations to complete
#         torch.cuda.synchronize()

#         # Reset peak memory statistics
#         torch.cuda.reset_peak_memory_stats()

#         # Force another cache clear after sync
#         torch.cuda.empty_cache()

#     # Force Python garbage collection multiple times
#     for _ in range(3):
#         gc.collect()

#     # Additional CUDA cleanup if available
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
