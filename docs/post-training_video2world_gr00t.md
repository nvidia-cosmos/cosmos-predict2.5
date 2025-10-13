# Video2World Post-training for DreamGen Bench

This guide provides instructions on running post-training with the Cosmos-Predict2.5 Video2World 2B model.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Preparing Data](#1-preparing-data)
- [Post-training](#2-post-training)
- [Inference with the Post-trained checkpoint](#3-inference-with-the-post-trained-checkpoint)

## Prerequisites

Before proceeding, please read the [Post-training Guide](./post-training.md) for detailed setup steps and important post-training instructions, including checkpointing and best practices. This will ensure you are fully prepared for post-training with Cosmos-Predict2.5.

## 1. Preparing Data

### 1.1 Download DreamGen Bench Training Dataset
For training on the robotic training datasets from the DreamGen paper, please use the following command to download the GR1 training dataset from https://huggingface.co/datasets/nvidia/GR1-100.

```
# This command will download the videos for physical AI

hf download nvidia/GR1-100 --repo-type dataset --local-dir datasets/benchmark_train/hf_gr1/ && \
mkdir -p datasets/benchmark_train/gr1/videos && \
mv datasets/benchmark_train/hf_gr1/gr1/*mp4 datasets/benchmark_train/gr1/videos && \
mv datasets/benchmark_train/hf_gr1/metadata.csv datasets/benchmark_train/gr1/
```

### 1.2 Preprocess the data and verify the dataset folder format
Run the following command to create text prompt txt files for each video:
```
python -m scripts.create_prompts_for_gr1_dataset --dataset_path datasets/benchmark_train/gr1
```

Dataset folder format should be:
```
datasets/benchmark_train/gr1/
├── metas/
│   ├── *.txt
├── videos/
│   ├── *.mp4
├── metadata.csv
```

## 2. Post-training

Run the following command to execute an example post-training job with `GR1` data.
```bash
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiment=predict2_video2world_training_2b_groot_gr1_480
```

The model will be post-trained using the `GR1` dataset.
See the config `predict2_video2world_training_2b_groot_gr1_480` (../cosmos_predict2/experiments/base/groot.py) to understand how the dataloader is defined.

Checkpoints are saved to `${IMAGINAIRE_OUTPUT_ROOT}/PROJECT/GROUP/NAME/checkpoints`. By default, `IMAGINAIRE_OUTPUT_ROOT` is `/tmp/imaginaire4-output`. We strongly recommend setting `IMAGINAIRE_OUTPUT_ROOT` to a location with sufficient storage space for your checkpoints.

In the above example, `PROJECT` is `cosmos_predict_v2p5`, `GROUP` is `video2world`, `NAME` is `2b_groot_gr1_480`.

See the job config to understand how they are determined.
```python
predict2_video2world_training_2b_groot_gr1_480 = dict(
    dict(
        ...
        job=dict(
            project="cosmos_predict_v2p5",
            group="video2world",
            name="2b_groot_gr1_480",
        ),
        ...
    )
)
```

## 3. Inference with the Post-trained checkpoint

### 3.1 Converting DCP Checkpoint to Consolidated PyTorch Format

Since the checkpoints are saved in DCP format during training, you need to convert them to consolidated PyTorch format (.pt) for inference. Use the `convert_distcp_to_pt.py` script:

```bash
# Get path to the latest checkpoint
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_predict_v2p5/video2world/2b_groot_gr1_480/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

This conversion will create three files:

- `model.pt`: Full checkpoint containing both regular and EMA weights
- `model_ema_fp32.pt`: EMA weights only in float32 precision
- `model_ema_bf16.pt`: EMA weights only in bfloat16 precision (recommended for inference)

### 3.2 Running Inference

After converting the checkpoint, you can run inference with your post-trained model using a JSON configuration file that specifies the inference parameters (see `assets/sample_gr00t_dreams_gr1/gr00t_image2world.json` for an example). Note that we override the inference resolution in the JSON file to match the 480p training resolution.

```bash
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/sample_gr00t_dreams_gr1/gr00t_image2world.json \
  -o outputs/gr00t_gr1_sample \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_video2world_training_2b_groot_gr1_480
```

Generated videos will be saved to the output directory (e.g., `outputs/gr00t_gr1/`).

For more inference options and advanced usage, see [docs/inference.md](./inference.md).
