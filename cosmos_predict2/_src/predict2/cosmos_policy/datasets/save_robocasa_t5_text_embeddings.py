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
Precomputes T5 text embeddings for RoboCasa task descriptions and saves them to disk for faster training.

Usage:
    uv run -m cosmos_predict2._src.predict2.cosmos_policy.datasets.save_robocasa_t5_text_embeddings [--data_dir DATA_DIR] [--rollout_data_dir ROLLOUT_DATA_DIR]

Examples:
    uv run -m cosmos_predict2._src.predict2.cosmos_policy.datasets.save_robocasa_t5_text_embeddings --data_dir /lustre/fsw/portfolios/dir/users/user/data/robocasa/robocasa_regen_v2_1199succDemos/ --rollout_data_dir /lustre/fsw/portfolios/dir/users/user/data/robocasa/robocasa_regen_rollout_data_v2_1291episodes/
"""

import argparse

from cosmos_predict2._src.predict2.cosmos_policy.datasets.robocasa_dataset import RoboCasaDataset
from cosmos_predict2._src.predict2.cosmos_policy.datasets.t5_embedding_utils import (
    generate_t5_embeddings,
    save_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute T5 text embeddings for RoboCasa task descriptions")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/lustre/fsw/portfolios/dir/users/user/data/robocasa_regen",
        help="Directory containing RoboCasa dataset",
    )
    parser.add_argument(
        "--rollout_data_dir",
        type=str,
        default="/lustre/fsw/portfolios/dir/users/user/data/robocasa/robocasa_regen_rollout_data_v2_1291episodes/",
        help="Directory containing RoboCasa rollout data",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir

    print("Loading data...")
    dataset = RoboCasaDataset(
        data_dir=data_dir,
        rollout_data_dir=args.rollout_data_dir,
        lazy_load_demos=True,
        skip_computing_dataset_statistics=True,
    )

    t5_text_embeddings = generate_t5_embeddings(dataset.unique_commands)
    save_embeddings(t5_text_embeddings, data_dir, check_exists=True)


if __name__ == "__main__":
    main()
