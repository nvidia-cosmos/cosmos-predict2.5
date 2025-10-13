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

import importlib
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--training", action="store_true", help="Check training packages")
args = parser.parse_args()


def _test_flash_attn():
    import torch

    # Import after torch.
    from flash_attn import flash_attn_func

    device = "cuda"
    dtype = torch.float16

    batch_size = 2
    seqlen = 512
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)

    out = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=1.0 / (head_dim**0.5),
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
    )


def _flash_attn_is_ok():
    try:
        _test_flash_attn()
    except ImportError:
        return False
    except RuntimeError:
        return False
    return True


def check_packages(package_list, success_status=True):
    def print_success(package, version=None):
        if version:
            print(f"\033[92m[SUCCESS]\033[0m {package} found (v{version})")
        else:
            print(f"\033[92m[SUCCESS]\033[0m {package} found")

    def print_error(message):
        print(f"\033[91m[ERROR]\033[0m {message}")

    for package in package_list:
        if isinstance(package, tuple):
            found = False
            for alt_package in package:
                try:
                    module = importlib.import_module(alt_package)
                    version = getattr(module, "__version__", None)
                    print_success(alt_package, version)
                    found = True
                    break
                except ImportError:
                    continue
            if not found:
                print_error(f"None of the alternative packages found: \033[93m{', '.join(package)}\033[0m")
                success_status = False
        elif package == "apex":
            try:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", None)
                print_success(package, version)
                try:
                    from apex import multi_tensor_apply  # noqa: F401

                    print_success("apex.multi_tensor_apply")
                except ImportError:
                    print_error("apex.multi_tensor_apply not found")
                    success_status = False
            except ImportError:
                print_error("apex not found")
                success_status = False
        elif package == "transformer_engine":
            try:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", None)
                print_success(package, version)
                try:
                    import transformer_engine.pytorch  # noqa: F401

                    print_success("transformer_engine.pytorch")
                except ImportError:
                    print_error("transformer_engine.pytorch not found")
                    success_status = False
            except ImportError:
                print_error("transformer_engine not found")
                success_status = False
        else:
            try:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", None)
                print_success(package, version)
            except ImportError as e:
                print_error(f"Package not successfully imported: \033[93m{package}\033[0m")
                success_status = False

    if _flash_attn_is_ok():
        print(f"\033[92m[SUCCESS]\033[0m flash_attn_func succeeds")
    else:
        print(f"\033[91m[ERROR]\033[0m flash_attn_func fails")
        success_status = False

    return success_status


if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
    detected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"\033[91m[ERROR]\033[0m Python 3.10+ is required. You have: \033[93m{detected}\033[0m")
    sys.exit(1)

print("Attempting to import critical packages...")

packages = [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "transformer_engine",
    "megatron.core",
    ("flash_attn", "flash_attn_interface"),
    "natten",
]
packages_training = [
    "apex",
]

all_success = check_packages(packages)
if args.training:
    training_success = check_packages(packages_training)
    if not training_success:
        print("\033[93m[WARNING]\033[0m Training packages not found. Training features will be unavailable.")

if all_success:
    print("-----------------------------------------------------------")
    print("\033[92m[SUCCESS]\033[0m Cosmos-predict2 environment setup is successful!")
