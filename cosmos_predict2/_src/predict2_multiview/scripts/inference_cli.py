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
Inference script for constructing data_batch from videos and captions, then running predict2_multiview model.

Expected directory structure:
```
input_root/
├── videos/                                    # Video folder
│   ├── ftheta_camera_front_wide_120fov/      # or camera_front_wide_120fov/
│   │   ├── video_id_1.mp4
│   │   ├── video_id_2.mp4
│   │   └── ...
│   ├── ftheta_camera_cross_right_120fov/     # or camera_cross_right_120fov/
│   │   ├── video_id_1.mp4
│   │   ├── video_id_2.mp4
│   │   └── ...
│   ├── ftheta_camera_rear_right_70fov/       # or camera_rear_right_70fov/
│   ├── ftheta_camera_rear_tele_30fov/        # or camera_rear_tele_30fov/
│   ├── ftheta_camera_rear_left_70fov/        # or camera_rear_left_70fov/
│   ├── ftheta_camera_cross_left_120fov/      # or camera_cross_left_120fov/
│   └── ftheta_camera_front_tele_30fov/       # or camera_front_tele_30fov/
│
└── captions/                                  # Caption folder (optional, uses default prompt if not present)
    ├── ftheta_camera_front_wide_120fov/      # or camera_front_wide_120fov/
    │   ├── video_id_1.txt
    │   ├── video_id_2.txt
    │   └── ...
    ├── ftheta_camera_cross_right_120fov/     # or camera_cross_right_120fov/
    │   ├── video_id_1.txt
    │   ├── video_id_2.txt
    │   └── ...
    ├── ftheta_camera_rear_right_70fov/       # or camera_rear_right_70fov/
    ├── ftheta_camera_rear_tele_30fov/        # or camera_rear_tele_30fov/
    ├── ftheta_camera_rear_left_70fov/        # or camera_rear_left_70fov/
    ├── ftheta_camera_cross_left_120fov/      # or camera_cross_left_120fov/
    └── ftheta_camera_front_tele_30fov/       # or camera_front_tele_30fov/

Notes:
- The videos/ folder is required (unless num_conditional_frames=0, which uses dummy all-zero videos)
- The captions/ folder is optional; if not present, a preset default driving scene description is used
- Each camera's subfolder name supports two formats: "ftheta_{camera_name}" or "{camera_name}"
- video_id must be consistent across all camera folders
- All 7 camera views must have corresponding subfolders and files

Camera view to View Index mapping:
- camera_front_wide_120fov: 0
- camera_cross_right_120fov: 1
- camera_rear_right_70fov: 2
- camera_rear_tele_30fov: 3
- camera_rear_left_70fov: 4
- camera_cross_left_120fov: 5
- camera_front_tele_30fov: 6
```

Usage:
```bash
EXP=predict2p5_2b_mv_7train7_res720p_fps10_t24_frombase2p5avfinetune_alpamayo_only_allcaption_uniform_nofps
ckpt_path=s3://bucket/cosmos_predict2_multiview/cosmos2_mv2/predict2p5_2b_mv_7train7_res720p_fps10_t24_frombase2p5avfinetune_alpamayo_only_allcaption_uniform_nofps_resume1-0/checkpoints/iter_000024500/

PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_predict2._src.predict2_multiview.scripts.inference_cli \
    --experiment ${EXP} \
    --ckpt_path ${ckpt_path} \
    --context_parallel_size 8 \
    --input_root /project/cosmos/yiflu/project_official_i4/condition_assets/multiview-inference-assets-1203 \
    --num_conditional_frames 0 \
    --guidance 5 \
    --fps 10 \
    --save_root results/predict2_multiview_av_grid/ \
    --max_samples 5 --stack_mode grid
```

```bash
EXP=predict2p5_2b_mv_7train7_res480p_fps15_t24_alpamayo_only_allcaption_uniform_nofps
ckpt_path=s3://bucket/cosmos_predict2_multiview/cosmos2p5_mv/predict2p5_2b_mv_7train7_res480p_fps15_t24_alpamayo_only_allcaption_uniform_nofps-0/checkpoints/iter_000020000/

PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_predict2._src.predict2_multiview.scripts.inference_cli \
    --experiment ${EXP} \
    --ckpt_path ${ckpt_path} \
    --context_parallel_size 8 \
    --input_root /project/cosmos/yiflu/project_official_i4/condition_assets/multiview-inference-assets-1203 \
    --num_conditional_frames 1 \
    --guidance 5 \
    --fps 15 \
    --save_root results/predict2_multiview_av_480p_i2v_grid/ \
    --max_samples 5 --stack_mode grid \
    --target_height 480 --target_width 832 \
    model.config.net.init_cross_view_attn_weight_from=null
```
"""

import argparse
import os
from pathlib import Path

import torch as th
import torchvision

from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2_multiview.scripts.inference import Vid2VidInference

NUM_CONDITIONAL_FRAMES_KEY = "num_conditional_frames"

# Camera name to view index mapping
CAMERA_TO_VIEW_INDEX = {
    "camera_front_wide_120fov": 0,
    "camera_cross_right_120fov": 1,
    "camera_rear_right_70fov": 2,
    "camera_rear_tele_30fov": 3,
    "camera_rear_left_70fov": 4,
    "camera_cross_left_120fov": 5,
    "camera_front_tele_30fov": 6,
}

DEFAULT_CAMERA_ORDER = list(CAMERA_TO_VIEW_INDEX.keys())

# Camera-specific caption prefixes describing camera position and orientation
CAMERA_TO_CAPTION_PREFIX = {
    "camera_front_wide_120fov": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "camera_cross_right_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
    "camera_rear_right_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
    "camera_rear_tele_30fov": "The video is captured from a camera mounted on a car. The camera is facing backwards.",
    "camera_rear_left_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
    "camera_cross_left_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
    "camera_front_tele_30fov": "The video is captured from a telephoto camera mounted on a car. The camera is facing forward.",
}

DEFAULT_DRIVING_SCENE_PROMPT = """
A clear daytime driving scene on an open road. The weather is sunny with bright natural lighting and good visibility. 
The sky is partly cloudy with scattered white clouds. The road surface is dry and well-maintained. 
The overall atmosphere is calm and peaceful with moderate traffic conditions. The lighting creates clear 
shadows and provides excellent contrast for safe navigation."""


def load_video(video_path: str, target_frames: int = 93, target_size: tuple[int, int] = (720, 1280)) -> th.Tensor:
    """
    Load video and process it to target size and frame count.

    Args:
        video_path: Path to video file
        target_frames: Target number of frames
        target_size: Target resolution (H, W)

    Returns:
        Video tensor with shape (C, T, H, W), dtype uint8
    """
    try:
        # Load video using easy_io
        video_frames, video_metadata = easy_io.load(video_path)  # Returns (T, H, W, C) numpy array
    except Exception as e:
        raise ValueError(f"Failed to load video {video_path}: {e}")

    # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
    video_tensor = th.from_numpy(video_frames).float() / 255.0
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

    C, T, H, W = video_tensor.shape

    # Adjust frame count: if video is too long, take first target_frames; if too short, pad with last frame
    if T > target_frames:
        video_tensor = video_tensor[:, :target_frames, :, :]
    elif T < target_frames:
        # Pad with last frame
        last_frame = video_tensor[:, -1:, :, :]
        padding_frames = target_frames - T
        last_frame_repeated = last_frame.repeat(1, padding_frames, 1, 1)
        video_tensor = th.cat([video_tensor, last_frame_repeated], dim=1)

    # Convert to uint8: (C, T, H, W) -> (T, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    video_tensor = (video_tensor * 255.0).to(th.uint8)

    # Adjust resolution
    target_h, target_w = target_size
    if H != target_h or W != target_w:
        # Use resize and center crop
        video_tensor = resize_and_crop(video_tensor, target_size)

    # Convert back to (C, T, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)

    return video_tensor


def resize_and_crop(video: th.Tensor, target_size: tuple[int, int]) -> th.Tensor:
    """
    Resize video and center crop.

    Args:
        video: Input video with shape (T, C, H, W)
        target_size: Target resolution (H, W)

    Returns:
        Resized video with shape (T, C, target_H, target_W)
    """
    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = target_size

    # Calculate scaling ratio to match the smaller dimension to target
    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(scaling_ratio * orig_h), int(scaling_ratio * orig_w))

    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    video_cropped = torchvision.transforms.functional.center_crop(video_resized, target_size)

    return video_cropped


def load_multiview_videos(
    input_root: Path,
    video_id: str,
    camera_order: list[str],
    target_frames: int = 93,
    target_size: tuple[int, int] = (720, 1280),
) -> th.Tensor:
    """
    Load multi-view videos.

    Args:
        input_root: Input root directory
        video_id: Video ID (filename without extension)
        camera_order: List of camera names in order
        target_frames: Target number of frames per view
        target_size: Target resolution (H, W)

    Returns:
        Multi-view video tensor with shape (C, V*T, H, W)
    """
    videos_dir = input_root / "videos"
    video_tensors = []

    for camera in camera_order:
        if (videos_dir / f"ftheta_{camera}").exists():
            sub_dir = f"ftheta_{camera}"
        elif (videos_dir / camera).exists():
            sub_dir = camera
        else:
            raise FileNotFoundError(f"Folder not found: {videos_dir / f'ftheta_{camera}'} or {videos_dir / camera}")

        video_path = videos_dir / sub_dir / f"{video_id}.mp4"

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Load single view video: (C, T, H, W)
        video_tensor = load_video(str(video_path), target_frames, target_size)
        video_tensors.append(video_tensor)

    # Concatenate all views: (C, V*T, H, W)
    multiview_video = th.cat(video_tensors, dim=1)

    return multiview_video


def load_multiview_captions(
    input_root: Path, video_id: str, camera_order: list[str], add_camera_prefix: bool = True
) -> list[str]:
    """
    Load multi-view captions. Uses default prompt if captions directory does not exist.

    Args:
        input_root: Input root directory
        video_id: Video ID (filename without extension)
        camera_order: List of camera names in order
        add_camera_prefix: Whether to add camera-specific prefix to captions

    Returns:
        List of captions, one per view
    """
    captions_dir = input_root / "captions"

    # If captions directory does not exist, use default prompt
    if not captions_dir.exists():
        log.warning(
            f"Captions directory not found: {captions_dir}. Using default driving scene prompt for all cameras."
        )
        return [DEFAULT_DRIVING_SCENE_PROMPT] * len(camera_order)

    captions = []

    for camera in camera_order:
        if (captions_dir / f"ftheta_{camera}").exists():
            sub_dir = f"ftheta_{camera}"
        elif (captions_dir / camera).exists():
            sub_dir = camera
        else:
            raise FileNotFoundError(f"Folder not found: {captions_dir / f'ftheta_{camera}'} or {captions_dir / camera}")

        caption_filename = f"{sub_dir}/{video_id}.txt"
        caption_path = captions_dir / caption_filename

        if not caption_path.exists():
            raise FileNotFoundError(f"Caption file not found: {caption_path}")

        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        # Add camera-specific prefix if enabled
        if add_camera_prefix and camera in CAMERA_TO_CAPTION_PREFIX:
            caption = f"{CAMERA_TO_CAPTION_PREFIX[camera]} {caption}"

        captions.append(caption)

    return captions


def construct_data_batch(
    multiview_video: th.Tensor,
    captions: list[str],
    camera_order: list[str],
    num_conditional_frames: int = 0,
    fps: float = 15.0,
    target_frames_per_view: int = 93,
) -> dict:
    """
    Construct data_batch for model inference.

    Args:
        multiview_video: Multi-view video tensor with shape (C, V*T, H, W)
        captions: List of captions
        camera_order: List of camera names in order
        num_conditional_frames: Number of conditional frames
        fps: Frames per second
        target_frames_per_view: Number of frames per view

    Returns:
        data_batch dictionary
    """
    C, VT, H, W = multiview_video.shape
    n_views = len(camera_order)
    T = VT // n_views

    # Add batch dimension: (C, V*T, H, W) -> (1, C, V*T, H, W)
    multiview_video = multiview_video.unsqueeze(0)

    # Construct correct view_indices based on camera order
    # Each view's T frames all use that view's corresponding view index
    view_indices_list = []
    for camera in camera_order:
        view_idx = CAMERA_TO_VIEW_INDEX[camera]
        view_indices_list.extend([view_idx] * T)
    view_indices = th.tensor(view_indices_list, dtype=th.int64).unsqueeze(0)  # (1, V*T)

    # Construct view_indices_selection: view indices of cameras in camera_order
    view_indices_selection = th.tensor(
        [CAMERA_TO_VIEW_INDEX[camera] for camera in camera_order], dtype=th.int64
    ).unsqueeze(0)  # (1, n_views)

    # Find position of front_wide_120fov in camera_order as ref_cam_view_idx_sample_position
    ref_cam_position = (
        camera_order.index("camera_front_wide_120fov") if "camera_front_wide_120fov" in camera_order else 0
    )

    # Construct data_batch
    data_batch = {
        "video": multiview_video,
        "ai_caption": [captions],
        "view_indices": view_indices,  # (1, V*T), using correct view index
        "fps": th.tensor([fps], dtype=th.float64),
        "chunk_index": th.tensor([0], dtype=th.int64),
        "frame_indices": th.arange(target_frames_per_view).unsqueeze(0),  # (1, T)
        "num_video_frames_per_view": th.tensor([target_frames_per_view], dtype=th.int64),
        "view_indices_selection": view_indices_selection,  # (1, n_views), using correct view index
        "camera_keys_selection": [camera_order],
        "sample_n_views": th.tensor([n_views], dtype=th.int64),
        "padding_mask": th.zeros(1, 1, H, W, dtype=th.float32),
        "ref_cam_view_idx_sample_position": th.tensor([ref_cam_position], dtype=th.int64),
        "front_cam_view_idx_sample_position": [None],
        "original_hw": th.tensor([[[H, W]] * n_views], dtype=th.int64),  # (1, n_views, 2)
        NUM_CONDITIONAL_FRAMES_KEY: num_conditional_frames,
    }

    return data_batch


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict2 Multiview inference from videos and captions")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the checkpoint. If not provided, will use the one specify in the config",
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (number of GPUs to split context over). Set to 8 for 8 GPUs",
    )
    # Generation parameters
    parser.add_argument("--guidance", type=int, default=5, help="Guidance value")
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_conditional_frames", type=int, default=0, help="Number of conditional frames")
    parser.add_argument("--num_steps", type=int, default=35, help="Number of diffusion steps")
    # Input/output
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Input root directory containing videos/ and captions/ subdirectories",
    )
    parser.add_argument("--save_root", type=str, default="results/predict2_multiview_av/", help="Save root")
    parser.add_argument("--max_samples", type=int, default=5, help="Maximum number of samples to generate")
    parser.add_argument(
        "--stack_mode",
        type=str,
        default="time",
        choices=["height", "width", "time", "grid"],
        help="Video stacking mode for visualization. grid will create a 3x3 grid of views.",
    )
    # Video parameters
    parser.add_argument("--target_frames", type=int, default=93, help="Target number of frames per view")
    parser.add_argument("--target_height", type=int, default=720, help="Target video height")
    parser.add_argument("--target_width", type=int, default=1280, help="Target video width")
    # Caption parameters
    parser.add_argument(
        "--add_camera_prefix",
        action="store_true",
        default=True,
        help="Add camera-specific position/orientation prefix to captions",
    )
    parser.add_argument(
        "--no_camera_prefix",
        action="store_false",
        dest="add_camera_prefix",
        help="Do not add camera-specific prefix to captions",
    )
    # Experiment options
    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.
        For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser.parse_args()


def main():
    os.environ["NVTE_FUSED_ATTN"] = "0"
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    th.enable_grad(False)

    args = parse_arguments()

    # Initialize inference handler
    vid2vid_cli = Vid2VidInference(
        args.experiment,
        args.ckpt_path,
        args.s3_cred,
        context_parallel_size=args.context_parallel_size,
        experiment_opts=args.opts,
    )
    mem_bytes = th.cuda.memory_allocated(device=th.device("cuda" if th.cuda.is_available() else "cpu"))
    log.info(f"GPU memory usage after model dcp.load: {mem_bytes / (1024**3):.2f} GB")

    # Only process files on rank 0
    rank0 = True
    if args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    # Create output directory
    os.makedirs(args.save_root, exist_ok=True)

    input_root = Path(args.input_root)
    videos_dir = input_root / "videos"

    # Get all video IDs (from first camera directory)
    if (videos_dir / f"ftheta_{DEFAULT_CAMERA_ORDER[0]}").exists():
        first_camera_dir = videos_dir / f"ftheta_{DEFAULT_CAMERA_ORDER[0]}"
    else:
        first_camera_dir = videos_dir / DEFAULT_CAMERA_ORDER[0]

    video_files = sorted(first_camera_dir.glob("*.mp4"))
    video_ids = [f.stem for f in video_files[: args.max_samples]]

    log.info(f"Found {len(video_ids)} video IDs, processing {min(len(video_ids), args.max_samples)} samples")

    for i, video_id in enumerate(video_ids):
        if rank0:
            log.info(f"Processing sample {i + 1}/{len(video_ids)}: {video_id}")

        try:
            # Load multi-view captions
            captions = load_multiview_captions(
                input_root, video_id, DEFAULT_CAMERA_ORDER, add_camera_prefix=args.add_camera_prefix
            )

            # Decide whether to load real videos based on num_conditional_frames
            if args.num_conditional_frames == 0:
                # Pure text-to-video generation, use dummy all-zero video
                n_views = len(DEFAULT_CAMERA_ORDER)
                multiview_video = th.zeros(
                    3,  # C
                    n_views * args.target_frames,  # V*T
                    args.target_height,  # H
                    args.target_width,  # W
                    dtype=th.uint8,
                )
                if rank0:
                    log.info(f"Using dummy video (all zeros) for text-to-video generation: {multiview_video.shape}")
            else:
                # Need video conditioning, load real videos
                multiview_video = load_multiview_videos(
                    input_root,
                    video_id,
                    DEFAULT_CAMERA_ORDER,
                    target_frames=args.target_frames,
                    target_size=(args.target_height, args.target_width),
                )
                if rank0:
                    log.info(f"Loaded multiview video: {multiview_video.shape}")

            if rank0:
                log.info(f"Loaded {len(captions)} captions")
                log.info(f"First caption preview: {captions[0][:100]}...")

            # Construct data_batch
            data_batch = construct_data_batch(
                multiview_video,
                captions,
                DEFAULT_CAMERA_ORDER,
                num_conditional_frames=args.num_conditional_frames,
                fps=args.fps,
                target_frames_per_view=args.target_frames,
            )

            # Run inference, already arranged the video according to stack_mode
            video = vid2vid_cli.generate_from_batch(
                data_batch,
                guidance=args.guidance,
                seed=args.seed + i,
                num_steps=args.num_steps,
                stack_mode=args.stack_mode,
            )

            # Save results
            if rank0:
                save_name = f"inference_av_{video_id}"
                save_img_or_video(video[0], f"{args.save_root}/{save_name}", fps=args.fps)
                log.info(f"Saved video to {args.save_root}/{save_name}")

        except Exception as e:
            log.error(f"Error processing {video_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Synchronize all processes
    if args.context_parallel_size > 1:
        th.distributed.barrier()

    # Cleanup distributed resources
    vid2vid_cli.cleanup()


if __name__ == "__main__":
    main()
