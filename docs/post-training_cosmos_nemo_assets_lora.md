# LoRA Post-training for Cosmos-NeMo-Assets

This guide provides instructions on running LoRA (Low-Rank Adaptation) post-training with the Cosmos-Predict2.5 models for both Video2World and Text2World generation tasks.

## Table of Contents

- [Prerequisites](#prerequisites)
- [What is LoRA?](#what-is-lora)
- [Preparing Data](#1-preparing-data)
- [LoRA Post-training](#2-lora-post-training)
  - [Video2World LoRA Training](#22-video2world-lora-training)
  - [Text2World LoRA Training](#23-text2world-lora-training)
- [Inference with LoRA Post-trained checkpoint](#3-inference-with-lora-post-trained-checkpoint)

## Prerequisites

Before proceeding, please read the [Post-training Guide](./post-training.md) for detailed setup steps and important post-training instructions, including checkpointing and best practices. This will ensure you are fully prepared for post-training with Cosmos-Predict2.5.

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that allows you to adapt large pre-trained models to specific domains or tasks by training only a small number of additional parameters.

### Key Benefits of LoRA Post-Training

- **Memory Efficiency**: Only trains ~1-2% of total model parameters
- **Faster Training**: Significantly reduced training time per iteration
- **Storage Efficiency**: LoRA checkpoints are much smaller than full model checkpoints
- **Flexibility**: Can maintain multiple LoRA adapters for different domains
- **Preserved Base Capabilities**: Retains the original model's capabilities while adding domain-specific improvements

### When to Use LoRA vs Full Fine-tuning

**Use LoRA when:**
- You have limited compute resources
- You want to create domain-specific adapters
- You need to preserve the base model's general capabilities
- You're working with smaller datasets

**Use full fine-tuning when:**
- You need maximum model adaptation
- You have sufficient compute and storage
- You're making fundamental changes to model behavior

## 1. Preparing Data

### 1.1 Understanding Training Data Requirements

The data preparation is identical for both Video2World and Text2World training:

- **Video2World**: Requires videos with text prompts. Uses conditional frames from the videos.
- **Text2World**: Also requires videos with text prompts. The videos serve as ground truth for computing the training loss, teaching the model to reconstruct videos given their text descriptions.

You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p. These videos should focus on the subject throughout the entire video so that each video chunk contains the subject.
You can use [nvidia/Cosmos-NeMo-Assets](https://huggingface.co/datasets/nvidia/Cosmos-NeMo-Assets) for post-training.

### 1.2 Downloading Cosmos-NeMo-Assets

To download the dataset, please follow the following instructions:
```bash
mkdir -p datasets/cosmos_nemo_assets/

# This command will download the videos for physical AI
hf download nvidia/Cosmos-NeMo-Assets \
  --repo-type dataset \
  --local-dir datasets/cosmos_nemo_assets/ \
  --include "*.mp4*"

mv datasets/cosmos_nemo_assets/nemo_diffusion_example_data datasets/cosmos_nemo_assets/videos
```

### 1.3 Preprocessing the Data

Cosmos-NeMo-Assets comes with a single caption for 4 long videos.

#### Creating Prompt Files

To generate text prompt files for each video in the dataset, use the provided preprocessing script:

```bash
# Create prompt files for all videos with a custom prompt
python -m scripts.create_prompts_for_nemo_assets \
    --dataset_path datasets/cosmos_nemo_assets \
    --prompt "A video of sks teal robot."
```

Dataset folder format:

```
datasets/cosmos_nemo_assets/
├── metas/
│   └── *.txt
└── videos/
    └── *.mp4
```

## 2. LoRA Post-training

### 2.1 Understanding the LoRA Configuration

The LoRA configurations for both Video2World and Text2World are defined in `cosmos_predict2/experiments/base/cosmos_nemo_assets_lora.py`. Both configurations share the same dataset and dataloader:

```python
# Shared dataset for both video2world and text2world LoRA training
example_dataset_cosmos_nemo_assets_lora = L(VideoDataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    num_frames=93,
    video_size=(704, 1280),
)

dataloader_train_cosmos_nemo_assets_lora = L(get_generic_dataloader)(
    dataset=example_dataset_cosmos_nemo_assets_lora,
    batch_size=1,
    num_workers=4,
    pin_memory=True,
)
```

#### Key Configuration Differences

**Video2World Configuration:**
```python
model=dict(
    config=dict(
        use_lora=True,
        lora_rank=32,              # Rank of LoRA adaptation matrices
        lora_alpha=32,             # LoRA scaling parameter
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        init_lora_weights=True,    # Properly initialize LoRA weights
    ),
),
```

**Text2World Configuration:**
```python
model=dict(
    config=dict(
        # Enable LoRA training
        use_lora=True,
        lora_rank=32,
        lora_alpha=32,
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        init_lora_weights=True,

        # Configure video2world model for text2world by setting conditional frames to 0
        min_num_conditional_frames=0,  # 0 frames for text2world mode
        max_num_conditional_frames=0,  # 0 frames for text2world mode
        conditional_frame_timestep=-1.0,  # Default -1 means not effective
        conditioning_strategy="frame_replace",
        denoise_replace_gt_frames=True,
    ),
),
```

**Important**: The video2world rectified flow model operates in different modes based on the number of conditional frames:
- **0 conditional frames**: Text2World generation
- **1 conditional frame**: Image2World generation
- **2+ conditional frames**: Video2World generation

### 2.2 Video2World LoRA Training

Run the following command to execute Video2World LoRA post-training:

```bash
torchrun --nproc_per_node=8 scripts/train.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=predict2_video2world_lora_training_2b_cosmos_nemo_assets
```

Checkpoints are saved to `${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict_v2p5/video2world_lora/2b_cosmos_nemo_assets_lora/checkpoints`.

### 2.3 Text2World LoRA Training

Run the following command to execute Text2World LoRA post-training:

```bash
torchrun --nproc_per_node=8 scripts/train.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=predict2_text2world_lora_training_2b_cosmos_nemo_assets
```

Checkpoints are saved to `${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict_v2p5/text2world_lora/2b_cosmos_nemo_assets_text2world_lora/checkpoints`.

**Note**: By default, `IMAGINAIRE_OUTPUT_ROOT` is `/tmp/imaginaire4-output`. We strongly recommend setting `IMAGINAIRE_OUTPUT_ROOT` to a location with sufficient storage space for your checkpoints.

## 3. Inference with LoRA Post-trained checkpoint

### 3.1 Converting DCP Checkpoint to Consolidated PyTorch Format

Since the checkpoints are saved in DCP format during training, you need to convert them to consolidated PyTorch format (.pt) for inference. Use the `convert_distcp_to_pt.py` script:

#### For Video2World:

```bash
# Get path to the latest checkpoint
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_predict_v2p5/video2world_lora/2b_cosmos_nemo_assets_lora/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

#### For Text2World:

```bash
# Get path to the latest checkpoint
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_predict_v2p5/text2world_lora/2b_cosmos_nemo_assets_text2world_lora/checkpoints
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

After converting the checkpoint, you can run inference with your post-trained model using a JSON configuration file that specifies the inference parameters.

#### Video2World Inference:

```bash
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/video2world_cosmos_nemo_assets/nemo_image2world.json \
  -o outputs/video2world_posttraining \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_video2world_lora_training_2b_cosmos_nemo_assets
```

#### Text2World Inference:

```bash
# For text2world generation using the video2world model
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/text2world_prompts.json \
  -o outputs/text2world_posttraining \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_text2world_lora_training_2b_cosmos_nemo_assets
```

Generated videos will be saved to the output directory (e.g., `outputs/video2world_posttraining/` or `outputs/text2world_posttraining/`).

For more inference options and advanced usage, see [docs/inference.md](./inference.md).
