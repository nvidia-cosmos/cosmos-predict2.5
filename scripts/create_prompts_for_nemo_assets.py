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

import argparse
import os
import sys

"""example command
python -m scripts.create_prompts_for_nemo_assets --dataset_path datasets/cosmos_nemo_assets
"""


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create text prompts for cosmos nemo assets")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/cosmos_nemo_assets",
        help="Root path to the dataset",
    )
    parser.add_argument("--prompt", type=str, default="A video of sks teal robot.", help="Text prompt for the dataset")
    parser.add_argument("--is_image", action="store_true", help="Set if the dataset is image-based")
    return parser.parse_args()


def main(args) -> None:
    images_dir = os.path.join(args.dataset_path, "images")
    videos_dir = os.path.join(args.dataset_path, "videos")
    if args.is_image and not os.path.exists(images_dir):
        sys.stderr.write(f"images dir: {images_dir} does not exist, please re-structure {args.dataset_path}\n")
        sys.exit(1)
    elif not args.is_image and not os.path.exists(videos_dir):
        sys.stderr.write(f"videos dir: {videos_dir} does not exist, please re-structure {args.dataset_path}\n")
        sys.exit(2)

    # Cosmos-NeMo-Assets come with videos only. A prompt is provided as an argument.
    metas_dir = os.path.join(args.dataset_path, "metas")
    os.makedirs(metas_dir, exist_ok=True)

    if args.is_image:
        metas_list = [
            os.path.join(metas_dir, filename.replace(".jpg", ".txt"))
            for filename in sorted(os.listdir(images_dir))
            if filename.endswith(".jpg")
        ]
    else:
        metas_list = [
            os.path.join(metas_dir, filename.replace(".mp4", ".txt"))
            for filename in sorted(os.listdir(videos_dir))
            if filename.endswith(".mp4")
        ]

    # Write txt files to match other dataset formats.
    print(f"Creating prompt files with text: {args.prompt}")
    created_count = 0
    for meta_filename in metas_list:
        if not os.path.exists(meta_filename):
            with open(meta_filename, "w") as fp:
                fp.write(args.prompt)
            created_count += 1

    print(f"Created {created_count} prompt files in {metas_dir}")
    print(f"Total prompt files: {len(metas_list)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
