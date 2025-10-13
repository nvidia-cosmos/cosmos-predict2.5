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

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Union

import click
import imageio as imageio_v1
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

CameraNames = ["front", "front_left", "front_right", "side_left", "side_right"]

SourceFps = 10  # waymo's recording fps

OUTPUT_DIR = Path("datasets/multiview/waymo")


def get_camera_name(name_int) -> str:
    return dataset_pb2.CameraName.Name.Name(name_int)


def load_captions():
    with open(os.path.join(OUTPUT_DIR, "waymo_caption.csv"), "r") as f:
        df = pd.read_csv(f).T
    df.index.name = "sample_id"
    df.reset_index(inplace=True)
    df.rename(inplace=True, columns={0: "caption"})
    df["view"] = f"pinhole_{CameraNames[0]}"
    df["tag"] = None
    return df


def convert_waymo_image(sample_id: str, dataset: tf.data.TFRecordDataset):
    """
    read all frames and convert the images to video format.
    """

    camera_name_to_image_sequence = {}
    for camera_name in CameraNames:
        camera_name_to_image_sequence[camera_name] = []

    for data in dataset:
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        for image_data in frame.images:
            camera_name = get_camera_name(image_data.name).lower()
            image_data_bytes = image_data.image
            image_data_tf_tensor = tf.image.decode_jpeg(image_data_bytes)
            image_data_numpy = image_data_tf_tensor.numpy()
            camera_name_to_image_sequence[camera_name].append(image_data_numpy)

    output_video_dir = OUTPUT_DIR / "input" / sample_id
    output_video_dir.mkdir(parents=True)

    for camera_name, image_sequence in tqdm(
        camera_name_to_image_sequence.items(), desc=f"Writing videos for {sample_id}"
    ):
        # waymo is recorded at 10 Hz
        output_video_path = output_video_dir / f"pinhole_{camera_name}.mp4"
        writer = imageio_v1.get_writer(
            output_video_path.as_posix(),
            fps=SourceFps,
            macro_block_size=None,  # This makes sure num_frames is correct (by default it is rounded to 16x).
        )
        for image in image_sequence:
            writer.append_data(image)
        writer.close()


def convert_waymo(
    captions: pd.DataFrame,
    waymo_tfrecord_filename: Union[str, Path],
):
    waymo_tfrecord_path = Path(waymo_tfrecord_filename)
    sample_id = waymo_tfrecord_path.stem.lstrip("segment-").rstrip("_with_camera_labels")

    if not waymo_tfrecord_path.exists():
        raise FileNotFoundError(f"Waymo tfrecord file not found: {waymo_tfrecord_path}")

    if (OUTPUT_DIR / "input" / sample_id).exists():
        print(f"Skipping {sample_id} because it already exists")
        os.remove(waymo_tfrecord_path)
        return

    dataset = tf.data.TFRecordDataset(waymo_tfrecord_path, compression_type="")

    convert_waymo_image(sample_id, dataset)

    captions[captions.sample_id == sample_id].drop(columns=["sample_id"]).to_json(
        OUTPUT_DIR / "input" / sample_id / "caption.jsonl", lines=True, orient="records"
    )

    os.remove(waymo_tfrecord_path)


@click.command()
@click.option("--num_workers", "-n", type=int, default=1, help="Number of workers")
def main(num_workers: int):
    all_filenames = list((OUTPUT_DIR / "downloads").glob("*.tfrecord"))
    print(f"Found {len(all_filenames)} tfrecords")

    captions = load_captions()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(convert_waymo, waymo_tfrecord_filename=filename, captions=captions)
            for filename in all_filenames
        ]

        for future in tqdm(as_completed(futures), total=len(all_filenames), desc="Converting tfrecords"):
            try:
                future.result()
            except Exception as e:
                print(f"Failed to convert due to error: {e}")


if __name__ == "__main__":
    main()
