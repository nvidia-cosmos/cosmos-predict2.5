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
Field-of-view–based memory retrieval and shared dataset utilities
"""

import io
import json
import random
import time
from http.client import IncompleteRead
from typing import List, Optional

import boto3
import botocore.exceptions as botocore_exceptions
import torch
from botocore.config import Config
from decord import VideoReader
from urllib3.exceptions import ProtocolError as URLLib3ProtocolError
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from urllib3.exceptions import SSLError as URLLib3SSLError

from cosmos_predict2._src.imaginaire.modules.camera import Camera
from cosmos_predict2._src.imaginaire.utils import log


def create_s3_client(credentials_path: str) -> boto3.client:
    """Create S3 client from credentials file with sensible retries and timeouts."""
    with open(credentials_path, "r") as f:
        credentials = json.load(f)

    client_config = Config(
        retries={"max_attempts": 10, "mode": "standard"},
        read_timeout=300,
        connect_timeout=30,
        max_pool_connections=64,
    )

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=credentials.get("aws_access_key_id"),
        aws_secret_access_key=credentials.get("aws_secret_access_key"),
        region_name=credentials.get("region_name", "us-east-1"),
        config=client_config,
    )
    return s3_client


def load_data_list(s3_client: boto3.client, bucket_name: str, data_list_key: str, source_path_for_log: str):
    """Load data list JSON from S3 and log summary."""
    response = s3_client.get_object(Bucket=bucket_name, Key=data_list_key)
    data_info = json.loads(response["Body"].read().decode("utf-8"))
    log.info(f"Loaded data list with {len(data_info['data_list'])} samples from {source_path_for_log}")
    return data_info


def get_full_s3_path(root_prefix: str, relative_path: str) -> str:
    """Convert relative path to full S3 key given a root prefix."""
    relative_path = relative_path.lstrip("/")
    return f"{root_prefix}/{relative_path}" if root_prefix else relative_path


def download_s3_object_to_bytes(s3_client: boto3.client, bucket: str, key: str) -> bytes:
    """Download S3 object to bytes with retry/backoff and robust error handling."""
    max_attempts = 7
    base_backoff = 0.5

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            buffer = io.BytesIO()
            # download_fileobj handles streaming internally and is resilient to some transient issues
            s3_client.download_fileobj(bucket, key, buffer)
            buffer.seek(0)
            data = buffer.read()
            return data
        except botocore_exceptions.ClientError as e:
            # Do not retry on permanent errors
            error_code = e.response.get("Error", {}).get("Code") if hasattr(e, "response") else None
            if error_code in {"NoSuchKey", "404"}:
                raise
            last_error = e
            log.info(
                f"[S3] ClientError while downloading s3://{bucket}/{key} (attempt {attempt + 1}/{max_attempts}): {e}",
                rank0_only=False,
            )
        except (
            botocore_exceptions.ResponseStreamingError,
            botocore_exceptions.EndpointConnectionError,
            botocore_exceptions.ConnectionClosedError,
            botocore_exceptions.ReadTimeoutError,
            URLLib3ProtocolError,
            URLLib3ReadTimeoutError,
            URLLib3SSLError,
            IncompleteRead,
            OSError,
            IOError,
        ) as e:
            last_error = e
            log.info(
                f"[S3] Transient error while downloading s3://{bucket}/{key} (attempt {attempt + 1}/{max_attempts}): {e}",
                rank0_only=False,
            )

        # Backoff and retry
        if attempt < max_attempts - 1:
            sleep_s = min(30.0, base_backoff * (2**attempt) + random.uniform(0, 1))
            log.info(f"[S3] Retrying in {sleep_s:.2f}s", rank0_only=False)
            time.sleep(sleep_s)
            continue

    assert last_error is not None
    raise last_error


def load_video_from_s3(
    s3_client: boto3.client,
    root_bucket: str,
    root_prefix: str,
    relative_video_path: str,
    video_size,
):
    """Load all video frames from S3 and return as tensor along with total frames and fps."""
    video_key = get_full_s3_path(root_prefix, relative_video_path)
    video_bytes = download_s3_object_to_bytes(s3_client, root_bucket, video_key)

    # Create BytesIO object for decord
    video_stream = io.BytesIO(video_bytes)

    # Load video using decord
    vr = VideoReader(video_stream)
    total_frames = len(vr)
    fps = vr.get_avg_fps()  # Get FPS from video

    # Load all frames
    frames = torch.from_numpy(vr[:].asnumpy()).permute(0, 3, 1, 2).float()  # (T,C,H,W)

    # Clean up video reader following video_parsing.py pattern
    vr.seek(0)  # set video reader point back to 0 to clean up cache
    del vr  # delete the reader to avoid memory leak

    # Resize frames to target size
    if frames.shape[-2:] != tuple(int(x) for x in video_size):
        frames = torch.nn.functional.interpolate(
            frames, size=tuple(int(x) for x in video_size), mode="bilinear", align_corners=False
        )

    # Convert to uint8 format (similar to dataset_video.py)
    frames = torch.clamp(frames, 0, 255).to(torch.uint8)

    # Clean up BytesIO object
    video_stream.close()

    return frames, total_frames, fps


def load_camera_from_s3(s3_client: boto3.client, root_bucket: str, root_prefix: str, relative_camera_path: str):
    """Load camera parameters from S3 text file."""
    camera_key = get_full_s3_path(root_prefix, relative_camera_path)
    camera_text = download_s3_object_to_bytes(s3_client, root_bucket, camera_key).decode("utf-8")

    lines = camera_text.strip().split("\n")

    # Parse camera parameters
    intrinsics_list = []
    extrinsics_list = []

    for line in lines:
        line = line.strip()

        tokens = line.split()
        assert len(tokens) == 11, f"Expected 11 camera parameters per line, got {len(tokens)}: {tokens}"
        values = list(map(float, tokens))

        intrinsics_list.append(values[:4])  # fx, fy, cx, cy
        extrinsics_list.append(values[4:])  # qx, qy, qz, qw, tx, ty, tz

    if not intrinsics_list:
        raise ValueError(f"No valid camera parameters found in {camera_key}")

    intrinsics = torch.tensor(intrinsics_list, dtype=torch.float32)  # (T, 4)
    extrinsics = torch.tensor(extrinsics_list, dtype=torch.float32)  # (T, 7)
    return {"intrinsics": intrinsics, "extrinsics": extrinsics}


def load_caption_from_s3(
    s3_client: boto3.client, root_bucket: str, root_prefix: str, relative_caption_path: str
) -> str:
    """Load caption from S3 text file."""
    if not relative_caption_path:
        return ""

    caption_key = get_full_s3_path(root_prefix, relative_caption_path)
    caption_text = download_s3_object_to_bytes(s3_client, root_bucket, caption_key).decode("utf-8")
    return caption_text.strip()


def get_camera_frustum_points(K, c2w, W, H, z_near=0.1, z_far=5.0):
    """
    Return 8 frustum corner points for a batch of cameras
    K: [B,4] (fx,fy,cx,cy)
    c2w: [B,4,4]
    """
    B = K.shape[0]
    device = K.device
    dtype = K.dtype

    # Pixel-space corners
    corners = torch.tensor([[0, 0], [W, 0], [0, H], [W, H]], dtype=torch.float32, device=device)  # [4,2]
    corners = corners.unsqueeze(0).expand(B, -1, -1)  # [B,4,2]

    # Promote to homogeneous pixel coords at z=1 and unproject using intrinsics
    K_mat = Camera.intrinsic_params_to_matrices(K)  # [B,3,3]
    corners_h = Camera.to_homogeneous(corners.to(dtype))  # [B,4,3] (x,y,1)
    rays_cam = Camera.image2camera(corners_h, K_mat)  # [B,4,3] points at z=1 in camera frame

    # Scale to near and far planes
    z_near_t = torch.as_tensor(z_near, dtype=rays_cam.dtype, device=rays_cam.device)
    z_far_t = torch.as_tensor(z_far, dtype=rays_cam.dtype, device=rays_cam.device)
    pts_near = rays_cam * z_near_t
    pts_far = rays_cam * z_far_t
    pts_all_cam = torch.cat([pts_near, pts_far], dim=1)  # [B,8,3]

    # Transform to world coordinates via Camera using world2cam
    w2c = Camera.invert_pose(c2w[..., :3, :4])  # [B,3,4]
    pts_world = Camera.camera2world(pts_all_cam, w2c)  # [B,8,3]
    return pts_world


def is_inside_frustum(X_world, K, c2w, W, H, z_near=0.1, z_far=5.0):
    """
    X_world: [B,N,3]
    K: [B,4], c2w: [B,4,4]
    Return mask: [B,N] bool
    """
    B, N, _ = X_world.shape

    # Convert to world2cam and project with intrinsics
    w2c = Camera.invert_pose(c2w[..., :3, :4])  # [B,3,4]
    K_mat = Camera.intrinsic_params_to_matrices(K)  # [B,3,3]

    X_cam = Camera.world2camera(X_world, w2c)  # [B,N,3]
    pix_h = Camera.camera2image(X_cam, K_mat)  # [B,N,3] = [fx x + cx z, fy y + cy z, z]

    z = X_cam[..., 2]
    z32 = z.to(torch.float32)
    eps = 1e-8 if z.dtype not in (torch.bfloat16, torch.float16) else 1e-2
    denom = z32.clamp_min(eps).unsqueeze(-1)
    uv32 = pix_h[..., :2].to(torch.float32) / denom
    u = uv32[..., 0].to(z.dtype)
    v = uv32[..., 1].to(z.dtype)

    z_near_t = torch.as_tensor(z_near, dtype=z.dtype, device=z.device)
    z_far_t = torch.as_tensor(z_far, dtype=z.dtype, device=z.device)
    inside = (z > z_near_t) & (z < z_far_t) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return inside


def fov_overlap_batch(K1, c2w1, K2, c2w2, W, H, n_samples=2000, z_near=0.01, z_far=1.0):
    """
    Monte Carlo FOV overlap between two batches of cameras
    K1,K2: [B,4]
    c2w1,c2w2: [B,4,4]
    Return: [B] overlap ratio in [0,1]
    """
    device = K1.device
    B = K1.shape[0]

    # get 8 corners of frustum1
    frustum1 = get_camera_frustum_points(K1, c2w1, W, H, z_near, z_far)  # [B,8,3]
    min_xyz, _ = frustum1.min(dim=1)  # [B,3]
    max_xyz, _ = frustum1.max(dim=1)  # [B,3]

    # uniform samples in bounding box for each batch
    # [B, n_samples, 3]
    rand = torch.rand(B, n_samples, 3, device=device)
    samples = rand * (max_xyz[:, None, :] - min_xyz[:, None, :]) + min_xyz[:, None, :]

    # check inside frustum1
    inside1 = is_inside_frustum(samples, K1, c2w1, W, H, z_near, z_far)  # [B,n_samples]

    # mask invalid ones
    valid_samples = samples.clone()
    valid_samples[~inside1] = float("nan")  # filter with NaN

    # check inside frustum2 (ignore NaN)
    inside2 = is_inside_frustum(valid_samples, K2, c2w2, W, H, z_near, z_far)  # [B,n_samples]

    # calculate overlap ratio
    num_valid = inside1.sum(dim=1).float()
    num_overlap = (inside1 & inside2).sum(dim=1).float()

    overlap_ratio = torch.where(num_valid > 0, num_overlap / num_valid, torch.zeros_like(num_valid))

    return overlap_ratio  # [B]


def select_closest_cameras(
    K_all, w2c_all, idx_selected, W, H, n_samples=2000, z_near=0.1, z_far=5.0, topk=20, device="cpu"
):
    """
    Select the cameras closest to the subsequence (based on FOV overlap in 3D)

    K_all: [T,4]  (fx,fy,cx,cy)
    w2c_all: [T,4,4]
    idx_selected: [S] selected camera sequence index
    W,H: image resolution
    Return: [topk] indices of closest cameras
    """

    w2c_all = Camera.get_relative_poses_wrt_frame0(w2c_all)
    c2w_all = Camera.invert_pose(w2c_all)

    T = K_all.shape[0]
    mask = torch.ones(T, dtype=torch.bool, device=device)
    mask[idx_selected] = False

    # subsequence cameras
    K_sub, c2w_sub = K_all[idx_selected], c2w_all[idx_selected]  # [S,4], [S,4,4]

    # remaining cameras
    K_rem, c2w_rem = K_all[mask], c2w_all[mask]
    idx_rem = torch.arange(T, device=device)[mask]  # [R]

    R = K_rem.shape[0]
    if R == 0:
        return idx_selected[0:1].repeat(topk)

    # pick evenly spaced S' cameras from subseq (length = topk)
    sub_idx = torch.linspace(0, K_sub.shape[0] - 1, steps=topk).long()  # [topk]
    K_sub_sel, c2w_sub_sel = K_sub[sub_idx], c2w_sub[sub_idx]  # [topk,4], [topk,4,4]

    # expand to [topk*R, ...]
    K1 = K_rem.unsqueeze(0).expand(topk, R, 4).reshape(-1, 4)
    c2w1 = c2w_rem.unsqueeze(0).expand(topk, R, 3, 4).reshape(-1, 3, 4)
    K2 = K_sub_sel.unsqueeze(1).expand(topk, R, 4).reshape(-1, 4)
    c2w2 = c2w_sub_sel.unsqueeze(1).expand(topk, R, 3, 4).reshape(-1, 3, 4)

    # compute FOV overlap in batch
    overlap = fov_overlap_batch(K1, c2w1, K2, c2w2, W, H, n_samples=n_samples, z_near=z_near, z_far=z_far)  # [topk*R]

    # reshape back
    overlap = overlap.view(topk, R)  # [topk, R]

    # Avoid corner cases
    overlap[:, -1] += 1e-4

    # take best match for each subseq camera
    scores, indices = torch.max(overlap, dim=1)  # [topk]
    selected_indices = idx_rem[indices]  # [topk]

    return selected_indices


def generate_random_up_down_sequence(
    length: int = 24,
    action_probability: float = 0.1,
    max_actions: int = 5,
    seed: Optional[int] = None,
    return_flags: bool = False,
) -> List[int]:
    """
    Generate a list of `length` integers starting at 0. By default the sequence
    increments toward `length - 1`. At each step, with probability
    `action_probability`, switch the current action to one of {-1 (backward), 0
    (stay), +1 (forward)} other than the current action. The action persists
    until switched, and the number of switches is capped by `max_actions`.
    Values are not clamped during generation; after generation,
    the entire sequence is offset by an integer so that all values lie within
    [0, length - 1]. If multiple offsets are valid, one is chosen at random.

    Args:
        length: Number of elements in the returned sequence (default: 24).
        action_probability: Chance in [0,1] to switch to a different action
            among {-1, 0, +1} at a step.
        max_actions: Maximum number of action switches allowed.
        seed: Optional randomness seed for reproducibility.

    Returns:
        If return_flags is False: List of integers of size `length` (values may
        underflow/overflow).
        If return_flags is True: Tuple (sequence, switch_flags, actions_used)
        where switch_flags is a boolean list of length `length` that marks the
        index before each action switch, and actions_used is the exact number of
        switches.
    """
    if length <= 0:
        return []

    rng = random.Random(seed)
    current_value = 0
    current_action = rng.choice([-1, 0, 1])  # random initial action
    sequence: List[int] = [current_value]
    switch_flags: List[bool] = [False] * length
    actions_used = 0  # count of action switches taken

    for _ in range(length - 1):
        # Optionally switch action among {-1, 0, +1}, excluding current one
        if actions_used < max_actions and rng.random() < action_probability:
            choices = (-1, 0, 1)
            alternatives = [a for a in choices if a != current_action]
            current_action = rng.choice(alternatives)
            actions_used += 1
            # Mark the element BEFORE the action change
            prev_index = len(sequence) - 1
            if 0 <= prev_index < length:
                switch_flags[prev_index] = True

        current_value = current_value + current_action
        sequence.append(current_value)

    # Offset the sequence so all values are in [0, length - 1]. Choose a random
    # valid offset if multiple exist.
    min_value = min(sequence)
    max_value = max(sequence)
    lower_offset = -min_value
    upper_offset = (length - 1) - max_value
    if lower_offset <= upper_offset:
        offset = rng.randrange(lower_offset, upper_offset + 1)
        sequence = [v + offset for v in sequence]

    if return_flags:
        return sequence, switch_flags, actions_used
    return sequence
