#!/usr/bin/env -S uv run --script
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

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "s5cmd",
#   "torch",
#   "tyro",
# ]
# [tool.uv.sources]
# torch = [{ index = "pytorch" }]
# [[tool.uv.index]]
# name = "pytorch"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///

"""Download distributed checkpoint from S3 and convert to pytorch checkpoint.

Usage:

```python ./scripts/convert_distcp_to_pt.py "/tmp/imaginaire4-output/cosmos_diffusion_v2/cosmos_nemo_posttraining/predict2_video2world_2b_cosmos_nemo_assets/checkpoints/iter_000000200/model" "checkpoints/nvidia/Cosmos-Predict2.5-2B/trained"
```
"""

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


@dataclass(frozen=True, kw_only=True)
class Args:
    input_dir: tyro.conf.Positional[str]
    """S3 URI of the checkpoint or path to the distcp directory."""
    output_dir: tyro.conf.Positional[Path]
    """Output directory to save the converted checkpoints."""

    ema: bool = True
    """Export EMA weights."""


def main():
    args = tyro.cli(Args, description=__doc__)

    pt_path = args.output_dir / "model.pt"
    pt_path.unlink(missing_ok=True)
    pt_ema_fp32_path = args.output_dir / "model_ema_fp32.pt"
    pt_ema_fp32_path.unlink(missing_ok=True)
    pt_ema_bf16_path = args.output_dir / "model_ema_bf16.pt"
    pt_ema_bf16_path.unlink(missing_ok=True)

    if args.input_dir.startswith("s3://"):
        input_s3 = args.input_dir.rstrip("/")
        input_s3 = input_s3.removesuffix("/model")
        distcp_dir = args.output_dir / "model"
        print(f"Downloading distcp to {distcp_dir}...")
        if distcp_dir.exists():
            cmd = ["s5cmd", "sync", "--exit-on-error"]
        else:
            cmd = ["s5cmd", "cp"]
        cmd.extend(
            [
                f"{input_s3}/model/*",
                f"{distcp_dir}",
            ]
        )
        try:
            print(shlex.join(cmd))
            # Capture output, since it is very verbose
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise e
    else:
        distcp_dir = Path(args.input_dir)

    # Convert distributed checkpoint to torch single checkpoint
    dcp_to_torch_save(distcp_dir, pt_path)
    print(f"Converted '{distcp_dir}' to '{pt_path}'")

    if not args.ema:
        return

    # Drop Reg keys and save EMA weights only in fp32 precision
    state_dict: dict[str, Any] = torch.load(pt_path, map_location="cpu", weights_only=False)
    state_dict_ema_fp32: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("net_ema."):
            key = key.replace("net_ema.", "net.")
            state_dict_ema_fp32[key] = value
    if not state_dict_ema_fp32:
        raise ValueError("Model doesn't contain EMA weights")
    torch.save(state_dict_ema_fp32, pt_ema_fp32_path)
    print(f"Saved EMA fp32 weights from '{pt_path}' to '{pt_ema_fp32_path}'")

    # Save EMA weights only in bf16 precision
    state_dict_ema_bf16: dict[str, Any] = {}
    for key, value in state_dict_ema_fp32.items():
        if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
            value = value.bfloat16()
        state_dict_ema_bf16[key] = value
    torch.save(state_dict_ema_bf16, pt_ema_bf16_path)
    print(f"fp32 -> bf16: '{pt_ema_fp32_path}' to '{pt_ema_bf16_path}'")


if __name__ == "__main__":
    main()
