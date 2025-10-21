# Auto Multiview Post-training with Waymo Dataset

This guide provides instructions on running post-training with the Cosmos-Predict2.5 2B Multiview model.

## Table of Contents

<!--TOC-->

- [Table of Contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [1. Preparing Data](#1-preparing-data)
  - [1.1 Downloading the Waymo dataset](#11-downloading-the-waymo-dataset)
- [2. Post-training](#2-post-training)
- [3. Inference with the Post-trained Checkpoint](#3-inference-with-the-post-trained-checkpoint)
  - [3.1 Converting DCP Checkpoint to Consolidated PyTorch Format](#31-converting-dcp-checkpoint-to-consolidated-pytorch-format)
  - [3.2 Running Inference](#32-running-inference)

<!--TOC-->

## Prerequisites

Before proceeding, please read the [Post-training Guide](./post-training.md) for detailed setup steps and important post-training instructions, including checkpointing and best practices. This will ensure you are fully prepared for post-training with Cosmos-Predict2.5.

## 1. Preparing Data

### 1.1 Downloading the Waymo dataset

To download the Waymo dataset, you'll need the Google Cloud CLI. If you have `sudo` access, you can install this via the following commands.
```
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-cli
```
Otherwise, you can follow the instructions here: https://cloud.google.com/sdk/docs/install#linux

Once installed, in `packages/cosmos-predict2`, run the following command to download the Waymo dataset:
```bash
gcloud init
bash scripts/download_waymo.sh datasets/multiview/waymo/ <number of desired files>
```

This will download the necessary files for the Waymo dataset as such:
```
datasets/multiview/waymo
├── downloads
│   └── segment-<sample_id>_with_camera_labels.tfrecord
└── waymo_all.json # json array of all sample ids (necessary for queries)
└── waymo_caption.csv # csv with a row for captions and columns for each sample id
```

The first time you run this, you will need to install the `waymo-open-dataset-tf-2-11-0` package, which you can do via the following command:
```bash
uv pip install waymo-open-dataset-tf-2-11-0==1.6.1
```

Installing this library may introduce tensorflow logs when running some of the scripts below. To avoid this, you can set the following environment variable:
```bash
export TF_CPP_MIN_LOG_LEVEL=2
```

You can then run the following command to convert the dataset into the appropriate format for the post-training pipeline.
```bash
python scripts/convert_waymo.py
```
This script converts the Waymo dataset into the following format:
```
datasets/multiview/waymo/input/<sample_id>
├── pinhole_front_left.mp4 # video from front left camera
└── pinhole_front_right.mp4 # video from front right camera
└── pinhole_front.mp4 # video from front camera
└── pinhole_side_left.mp4 # video from side left camera
└── pinhole_side_right.mp4 # video from side right camera
└── caption.jsonl # jsonl file with the columns: caption, view, tag
```

The caption.jsonl file contains all captions corresponding to the input. To assign separate captions for views, you can use the view column to specify which view the caption corresponds to. Currently, the dataloader is set to only use the front view caption, but this can be modified [here](../cosmos_predict2/_src/predict2_multiview/configs/vid2vid/defaults/local_dataloader.py) via `SINGLE_CAPTION_ONLY`. Additionally, you can specify a tag for each caption. When multiple captions with different tags are available for a view, one is chosen randomly based on the probabilities defined in [here](../cosmos_predict2/_src/predict2_multiview/datasets/local_dataset_train.py) via `caption_tag_ratios` in the dataloader configuration. This allows for stochastic caption selection during training.

## 2. Post-training

Finally, once you have the dataset downloaded and converted, you can run the following command to post-train with the Waymo dataset. We recommend you run post-training with 8 GPUs.
```bash
torchrun --nproc_per_node=8 -m scripts.train \
--config=cosmos_predict2/_src/predict2_multiview/configs/vid2vid/config.py \
-- \
experiment=predict2_multiview_post_train_waymo \
```

## 3. Inference with the Post-trained Checkpoint

### 3.1 Converting DCP Checkpoint to Consolidated PyTorch Format

Since the checkpoints are saved in DCP format during training, you need to convert them to consolidated PyTorch format (.pt) for inference. Use the `convert_distcp_to_pt.py` script:

```bash
# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py \
    ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict_v2p5/multiview/2b_waymo/checkpoints/iter_000002000/model/ \
    ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict_v2p5/multiview/2b_waymo/checkpoints/iter_000002000/
```

This conversion will create three files:

- `model.pt`: Full checkpoint containing both regular and EMA weights
- `model_ema_fp32.pt`: EMA weights only in float32 precision
- `model_ema_bf16.pt`: EMA weights only in bfloat16 precision (recommended for inference)

### 3.2 Running Inference

After converting the checkpoint, you can run inference with your post-trained model with the following command:

```bash
NUM_GPUS=8
torchrun --nproc_per_node=$NUM_GPUS examples/multiview.py \
--inference-type video2world \
--checkpoint-path ${IMAGINAIRE_OUTPUT_ROOT}cosmos_predict_v2p5/multiview/2b_waymo/checkpoints/iter_000000500/model_ema_bf16.pt \
--experiment predict2_multiview_post_train_waymo \
--use-config-dataloader
-o outputs/multiview
```

For more inference options and advanced usage, see [docs/inference_multiview.md](./inference_auto_multiview.md).
