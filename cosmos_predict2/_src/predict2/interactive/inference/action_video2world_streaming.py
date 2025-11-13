# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""
Script for streaming action-conditioned video generation using KV cache in a naive way

Example:
python -m cosmos_predict2._src.predict2.interactive.inference.action_video2world_streaming \
    --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py \
    --experiment=interactive_self_forcing_action_trigflow \
    --ckpt_path s3://bucket/cosmos_predict2_action_conditioned/interactive/cosmos_predict2p5_2B_action_gr00t_gr1_warmup_LR5e-5_newdata/checkpoints/iter_000004000 \
    --input_json cosmos_predict2/_src/predict2/interactive/assets/example_action.json
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import mediapy
import numpy as np
import torch
import torchvision.transforms.functional as TF
from loguru import logger
from PIL import Image

try:
    from megatron.core import parallel_state
except Exception:  # pragma: no cover - allow running without megatron installed in lint envs

    class _DummyParallelState:
        def is_initialized(self):
            return False

        def get_context_parallel_group(self):
            return None

        def initialize_model_parallel(self, **kwargs):
            return None

        def destroy_model_parallel(self):
            return None

    parallel_state = _DummyParallelState()  # type: ignore

from cosmos_predict2._src.imaginaire.utils import distributed, misc
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.distill.utils.model_loader import load_model_from_checkpoint
from cosmos_predict2._src.predict2.interactive.datasets.utils import extract_cr1_embedding
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
)

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Action-conditioned streaming Video2World inference script")
    parser.add_argument(
        "--config", type=str, default="cosmos_predict2/_src/predict2/interactive/configs/config.py", help="Config file"
    )
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config name")
    parser.add_argument("--ckpt_path", type=str, default="", help="S3/local checkpoint path")
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument("--input_json", type=str, required=True, help="Path to JSON entries list")
    parser.add_argument("--resolution", type=str, default="none", help="Optional resolution H,W (e.g. 256,320)")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS for output")
    parser.add_argument("--num_steps", type=int, default=4, help="Student steps per frame")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--start_frame_idx", type=int, default=0, help="Start frame index for conditioning")
    parser.add_argument("--max_frames", type=int, default=13, help="Max frames for conditioning [default is 13]")
    parser.add_argument(
        "--cache_frame_size",
        type=int,
        default=-1,
        help="Cache frame size [-1 means using the same as the video frame size]",
    )
    parser.add_argument(
        "--cr1_embeddings_path",
        type=str,
        default="/project/cosmos/user/cr1_empty_string_text_embeddings.pt",
        help="Local path to CR1 empty-string text embeddings (.pt)",
    )
    parser.add_argument("--context_parallel_size", type=int, default=1, help="Context parallel size")
    return parser.parse_args()


class ActionStreamingInference:
    def __init__(
        self,
        config_path: str,
        experiment_name: str,
        ckpt_path: str,
        s3_credential_path: str,
        cr1_embeddings_path: str,
        context_parallel_size: int = 1,
    ) -> None:
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.s3_credential_path = s3_credential_path
        self.cr1_embeddings_path = cr1_embeddings_path
        self.context_parallel_size = context_parallel_size
        self.process_group = None

        if self.context_parallel_size > 1:
            self._init_distributed()

        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file=config_path,
            load_ema_to_reg=True,
            experiment_opts=["ckpt_type=dcp"],
            skip_teacher_init=True,
        )

        if self.context_parallel_size > 1:
            model.net.enable_context_parallel(self.process_group)  # type: ignore

        # assert isinstance(
        #     model, ActionVideo2WorldModelRFSelfForcingDMD2
        # ), "Loaded model is not ActionVideo2WorldModelRFSelfForcingDMD2; check experiment config."

        self.model = model
        self.config = config
        self.batch_size = 1

        # Load CR1 empty-string text embeddings once (CPU). Expected shapes: [B, T, D] or [T, D].
        extract_cr1_embedding(self.cr1_embeddings_path)
        _emb = torch.load(self.cr1_embeddings_path, map_location="cpu")
        if isinstance(_emb, (list, tuple)):
            _emb = _emb[0]
        if not torch.is_tensor(_emb):
            raise ValueError("Loaded CR1 embeddings are not a torch.Tensor")
        if _emb.dim() == 2:
            _emb = _emb.unsqueeze(0)  # [1, T, D]
        elif _emb.dim() != 3:
            raise ValueError(f"Unexpected CR1 embeddings dim: {_emb.dim()} (expected 2 or 3)")
        self.t5_text_embeddings_cpu = _emb  # cache on CPU; move per inference call
        logger.info(
            f"Loaded CR1 text embeddings: shape={tuple(self.t5_text_embeddings_cpu.shape)} from {self.cr1_embeddings_path}"
        )

    def _init_distributed(self) -> None:
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=self.context_parallel_size)
        self.process_group = parallel_state.get_context_parallel_group()
        logger.info(f"Initialized context parallel with size {self.context_parallel_size}")
        logger.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def _prepare_data_batch(
        self,
        video_b_c_t_h_w: torch.Tensor,
        actions_np: np.ndarray,
        fps: float,
        num_latent_conditional_frames: int = 1,
    ) -> Dict[str, Any]:
        _, _, _, H, W = video_b_c_t_h_w.shape
        data_batch: Dict[str, Any] = {
            "dataset_name": "video_data",
            "video": video_b_c_t_h_w,
            "action": torch.from_numpy(actions_np).float().unsqueeze(0),  # [1, T-1, A]
            "fps": torch.tensor([fps], dtype=torch.float32),
            "padding_mask": torch.zeros(1, 1, H, W, dtype=torch.float32),
            NUM_CONDITIONAL_FRAMES_KEY: num_latent_conditional_frames,
        }

        # Move tensors to GPU and convert to bfloat16 if floating point
        for k, v in list(data_batch.items()):
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)
            elif isinstance(v, torch.Tensor):
                data_batch[k] = v.cuda()
        return data_batch

    @torch.inference_mode()
    def generate_action_streaming(
        self,
        video_path: str,
        actions_np: np.ndarray,
        resolution_hw: Tuple[int, int] | None,
        num_steps: int,
        seed: int,
        start_frame_idx: int,
        max_frames: int,
        cache_frame_size: int,
    ) -> torch.Tensor:
        # Load input video and extract conditioning frames
        video_array = mediapy.read_video(video_path)  # [Tv, H, W, C]
        num_conditional_frames = 0

        # Determine target resolution
        if resolution_hw is None:
            Ht, Wt = int(video_array.shape[1]), int(video_array.shape[2])
        else:
            Ht, Wt = int(resolution_hw[0]), int(resolution_hw[1])

        # Convert the selected conditioning frame to uint8 CHW tensor; resize if requested
        cond_frames: List[torch.Tensor] = []
        frame = video_array[0]
        if (frame.shape[0], frame.shape[1]) != (Ht, Wt):
            frame = mediapy.resize_image(frame, (Ht, Wt))
        frame_uint8 = np.clip(np.round(frame), 0, 255).astype(np.uint8)
        frame_tensor = TF.to_tensor(Image.fromarray(frame_uint8)).unsqueeze(0)
        # to_tensor returns float [0,1]; convert to uint8 [0,255]
        frame_tensor = (frame_tensor * 255.0).to(torch.uint8)
        cond_frames.append(frame_tensor)

        final_video = [frame_tensor.float().permute(1, 0, 2, 3) / 128 - 1]

        ar_total = actions_np.shape[0] // 12
        for ar_idx in range(ar_total):
            first_stack = torch.cat(cond_frames, dim=0)  # [N,3,H,W] uint8
            zeros_tail = torch.zeros_like(first_stack[:1]).repeat(12, 1, 1, 1)
            vid_chw_t = torch.cat([first_stack, zeros_tail], dim=0)  # [T,3,H,W] uint8
            video_b_c_t_h_w = vid_chw_t.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [1,3,T,H,W]

            # Prepare data batch
            data_batch = self._prepare_data_batch(
                video_b_c_t_h_w=video_b_c_t_h_w,
                actions_np=actions_np[ar_idx * 12 : (ar_idx + 1) * 12],
                fps=4,
                num_latent_conditional_frames=num_conditional_frames,
            )

            # Normalize and augment dims to match training pipeline
            self.model._normalize_video_databatch_inplace(data_batch)
            self.model._augment_image_dim_inplace(data_batch)

            # Inject CR1 text embeddings and mask expected by the conditioner
            # Always make sure embeddings and mask are bf16
            t5 = self.t5_text_embeddings_cpu.to(device=self.model.tensor_kwargs["device"], dtype=torch.bfloat16)
            data_batch["t5_text_embeddings"] = t5
            data_batch["t5_text_mask"] = torch.ones(
                (t5.shape[0], t5.shape[1]), device=self.model.tensor_kwargs["device"], dtype=torch.bfloat16
            )

            # Final safety: ensure all floating inputs are bf16 before conditioning
            for k, v in list(data_batch.items()):
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    data_batch[k] = v.to(dtype=self.model.tensor_kwargs["dtype"])  # typically bf16

            # Build condition following the base model's logic
            is_image_batch = self.model.is_image_batch(data_batch)
            data_type = DataType.IMAGE if is_image_batch else DataType.VIDEO
            # Use model helper to construct condition on this batch
            _, x0, condition, _ = self.model.get_data_and_condition(data_batch, set_video_condition=False)

            # Ensure latent frames used for conditioning match model precision (bf16)
            x0 = x0.to(dtype=self.model.tensor_kwargs["dtype"])
            condition = condition.edit_data_type(data_type)
            condition = condition.set_video_condition(
                gt_frames=x0,
                random_min_num_conditional_frames=None,
                random_max_num_conditional_frames=None,
                num_conditional_frames=data_batch[NUM_CONDITIONAL_FRAMES_KEY],
            )
            _x0, condition, _uncond, _eps, _t = self.model.broadcast_split_for_model_parallelsim(
                x0, condition, None, None, None
            )

            # Init noise with correct latent shape
            _T, _H, _W = data_batch[self.model.input_data_key].shape[-3:]
            state_shape = (
                self.model.config.state_ch,
                int(self.model.tokenizer.get_latent_num_frames(_T)),
                _H // self.model.tokenizer.spatial_compression_factor,
                _W // self.model.tokenizer.spatial_compression_factor,
            )
            noise = misc.arch_invariant_rand(
                (1, *state_shape),
                torch.float32,
                self.model.tensor_kwargs["device"],
                seed + ar_idx,
            )

            # Clamp steps to the configured student schedule length
            if hasattr(self.model, "student_t_list"):
                K = len(self.model.student_t_list)
            elif hasattr(self.model, "config") and hasattr(self.model.config, "selected_sampling_time"):
                K = len(self.model.config.selected_sampling_time)
            else:
                raise AttributeError(
                    "Model does not define 'student_t_list' or 'config.selected_sampling_time' to determine steps"
                )
            n_steps = max(1, min(int(num_steps), K))

            # Run streaming generation in latent space and decode
            logger.info("generation start")
            latents = self.model.generate_streaming_video(
                condition, noise, n_steps=n_steps, cache_frame_size=cache_frame_size
            )  # type: ignore[arg-type]
            logger.info("generation end")

            logger.info("decoding start")
            video = self.model.decode(latents)
            video = video.clip(min=-1, max=1)
            print("video: ", video.shape)
            logger.info("decoding end")

            final_video.append(video[0, :, 1:].cpu())

            cond_frames = []
            next_input_frame = ((1.0 + video[0:1, :, -1]) / 2 * 255.0).to(torch.uint8)
            cond_frames.append(next_input_frame)

        return torch.cat(final_video, dim=1)

    def cleanup(self) -> None:
        if self.context_parallel_size > 1:
            import torch.distributed as dist

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def _process_entries(
    entries: List[Dict[str, Any]], args: argparse.Namespace, infer: ActionStreamingInference, rank0: bool
):
    if not isinstance(entries, list):
        raise ValueError("input_json must contain a list of entries")

    for entry in entries:
        input_video = entry.get("input_video")
        input_action_path = entry.get("input_action")
        output_video_path = entry.get("output_video")

        if input_video is None or input_action_path is None or output_video_path is None:
            logger.warning(
                "Entry missing one of required keys: 'input_video', 'input_action', 'output_video'; skipping"
            )
            continue

        # Ensure output directory exists
        out_dir = os.path.dirname(output_video_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Load actions from a .npy file and use directly
        actions = np.load(input_action_path)

        # Optional per-entry overrides
        if "resolution" in entry and isinstance(entry["resolution"], list) and len(entry["resolution"]) == 2:
            res_hw = (int(entry["resolution"][0]), int(entry["resolution"][1]))
        elif args.resolution != "none":
            try:
                h, w = map(int, args.resolution.split(","))
                res_hw = (h, w)
            except Exception:
                res_hw = None
        else:
            res_hw = None

        # Per-entry override for start_frame_idx if provided
        if "start_frame_idx" in entry:
            try:
                start_frame_idx = int(entry["start_frame_idx"])  # type: ignore[arg-type]
            except Exception:
                start_frame_idx = args.start_frame_idx
        else:
            start_frame_idx = args.start_frame_idx

        video = infer.generate_action_streaming(
            video_path=input_video,
            actions_np=actions,
            resolution_hw=res_hw,
            num_steps=args.num_steps,
            seed=args.seed,
            start_frame_idx=start_frame_idx,
            max_frames=args.max_frames,
            cache_frame_size=args.cache_frame_size,
        )

        if rank0:
            save_fp_wo_ext = output_video_path[:-4] if output_video_path.endswith(".mp4") else output_video_path
            save_img_or_video((1.0 + video) / 2, save_fp_wo_ext, fps=args.fps)
            logger.info(f"Saved video to {output_video_path}")


@torch.inference_mode()
def main() -> None:
    args = parse_arguments()

    infer = ActionStreamingInference(
        config_path=args.config,
        experiment_name=args.experiment,
        ckpt_path=args.ckpt_path,
        s3_credential_path=args.s3_cred,
        cr1_embeddings_path=args.cr1_embeddings_path,
        context_parallel_size=args.context_parallel_size,
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory usage after model load: {mem_bytes / (1024**3):.2f} GB")

    rank0 = True
    if args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    with open(args.input_json, "r") as f:
        entries = json.load(f)

    _process_entries(entries, args, infer, rank0)
    if args.context_parallel_size > 1:
        torch.distributed.barrier()

    infer.cleanup()


if __name__ == "__main__":
    main()
