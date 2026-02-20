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
Precomputes Reason1 text embeddings for LIBERO task descriptions and saves them to disk.

This is the Reason1 equivalent of save_libero_t5_text_embeddings.py, used for
rectified flow models (predict2.5) which require Reason1 embeddings instead of T5.

Usage:
    uv run -m cosmos_predict2._src.predict2.cosmos_policy.datasets.save_libero_reason1_text_embeddings \
        --data_dir /path/to/libero/data

    # With custom options:
    uv run -m cosmos_predict2._src.predict2.cosmos_policy.datasets.save_libero_reason1_text_embeddings \
        --data_dir /path/to/libero/data \
        --batch_size 4 \
        --output_filename reason1_embeddings.pkl

After running, update your training config to:
1. Point t5_text_embeddings_path to the generated reason1_embeddings.pkl
2. Set text_encoder_config=None in the model config to disable online computation
"""

import argparse

from cosmos_predict2._src.predict2.cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_predict2._src.predict2.cosmos_policy.datasets.reason1_embedding_utils import (
    generate_reason1_embeddings,
    save_reason1_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute Reason1 text embeddings for LIBERO task descriptions")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/lustre/fsw/portfolios/dir/users/user/data/libero_regen",
        help="Directory containing LIBERO dataset (HDF5 files)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save embeddings (defaults to data_dir if not specified)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for computing embeddings (reduce if OOM)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="reason1_embeddings.pkl",
        help="Output filename for embeddings pickle file",
    )
    parser.add_argument(
        "--embedding_strategy",
        type=str,
        default="full_concat",
        choices=["full_concat", "mean_pooling", "pool_every_n_layers_and_concat"],
        help="Strategy for combining hidden layer outputs (must match training config)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
        help="Path to Reason1 checkpoint",
    )
    parser.add_argument(
        "--s3_credential_path",
        type=str,
        default="credentials/s3_checkpoint.secret",
        help="Path to S3 credentials file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Use output_dir if specified, otherwise use data_dir
    output_dir = args.output_dir if args.output_dir else args.data_dir

    print("=" * 60)
    print("LIBERO Reason1 Embedding Precomputation")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output filename: {args.output_filename}")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding strategy: {args.embedding_strategy}")
    print("=" * 60)

    # Load dataset to get unique commands
    print("\nLoading LIBERO dataset...")
    dataset = LIBERODataset(data_dir=args.data_dir)

    # Convert set to sorted list for consistent ordering
    unique_commands = sorted(list(dataset.unique_commands))

    print(f"Found {len(unique_commands)} unique commands")
    print("\nSample commands:")
    for i, cmd in enumerate(unique_commands[:5]):
        print(f"  {i + 1}. {cmd}")
    if len(unique_commands) > 5:
        print(f"  ... and {len(unique_commands) - 5} more")

    # Generate embeddings
    print("\n" + "=" * 60)
    print("Computing Reason1 embeddings (this may take several minutes)...")
    print("=" * 60)

    reason1_embeddings = generate_reason1_embeddings(
        unique_commands,
        batch_size=args.batch_size,
        ckpt_path=args.ckpt_path,
        s3_credential_path=args.s3_credential_path,
        embedding_concat_strategy=args.embedding_strategy,
    )

    # Save embeddings
    print("\n" + "=" * 60)
    save_path = save_reason1_embeddings(
        reason1_embeddings,
        output_dir,
        args.output_filename,
    )

    # Print next steps
    print("\n" + "=" * 60)
    print("DONE! Next steps:")
    print("=" * 60)
    print(f"""
1. Update your dataset config to use the new embeddings:
   t5_text_embeddings_path="{save_path}"

2. Update your model config to disable online computation:
   model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
       config=dict(
           ...
           text_encoder_config=None,  # Disable online computation
       ),
   ),

3. This should give you ~2x faster training speed!
""")


if __name__ == "__main__":
    main()
