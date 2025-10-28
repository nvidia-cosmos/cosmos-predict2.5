# LoRA Post-training for Cosmos-NeMo-Assets

This guide provides instructions on running LoRA (Low-Rank Adaptation) post-training with the Cosmos-Predict2.5 models for Text2World, Image2World, and Video2World generation tasks.

## Table of Contents

- [Prerequisites](#prerequisites)
- [What is LoRA?](#what-is-lora)
- [Preparing Data](#1-preparing-data)
- [LoRA Post-training](#2-lora-post-training)
  - [Configuration](#21-configuration)
  - [Training](#22-training)
- [Inference with LoRA Post-trained checkpoint](#3-inference-with-lora-post-trained-checkpoint)
  - [Converting DCP Checkpoint to Consolidated PyTorch Format](#31-converting-dcp-checkpoint-to-consolidated-pytorch-format)
  - [Running Inference](#32-running-inference)

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

The training approach uses the same video dataset to train all three generation modes:

- **Text2World (0 frames)**: Uses only text prompts, videos serve as ground truth for reconstruction
- **Image2World (1 frame)**: Uses first frame as condition, generates remaining frames
- **Video2World (2+ frames)**: Uses initial frames as condition, continues the video generation

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

### 2.1 Configuration

The LoRA configuration is defined in `cosmos_predict2/experiments/base/cosmos_nemo_assets_lora.py`:

```python
# Shared dataset for LoRA training
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

#### Key Configuration Parameters

```python
model=dict(
    config=dict(
        # Enable LoRA training
        use_lora=True,
        lora_rank=32,              # Rank of LoRA adaptation matrices
        lora_alpha=32,             # LoRA scaling parameter
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        init_lora_weights=True,    # Properly initialize LoRA weights

        # Training for all three modes
        min_num_conditional_frames=0,  # Allow text2world (0 frames)
        max_num_conditional_frames=2,  # Allow up to video2world (2 frames)

        # Probability distribution for each mode
        conditional_frames_probs={0: 0.333, 1: 0.333, 2: 0.334},
    ),
),
```

**Important**: The model operates in different modes based on the number of conditional frames:
- **0 conditional frames**: Text2World generation
- **1 conditional frame**: Image2World generation
- **2+ conditional frames**: Video2World generation

### 2.2 Training

Run the following command to execute  LoRA post-training:

```bash
torchrun --nproc_per_node=8 scripts/train.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=predict2_lora_training_2b_cosmos_nemo_assets
```

Checkpoints are saved to `${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict_v2p5/video2world_lora/2b_cosmos_nemo_assets_lora/checkpoints`.


**Note**: By default, `IMAGINAIRE_OUTPUT_ROOT` is `/tmp/imaginaire4-output`. We strongly recommend setting `IMAGINAIRE_OUTPUT_ROOT` to a location with sufficient storage space for your checkpoints.

## 3. Inference with LoRA Post-trained checkpoint

### 3.1 Converting DCP Checkpoint to Consolidated PyTorch Format

Since the checkpoints are saved in DCP format during training, you need to convert them to consolidated PyTorch format (.pt) for inference. Use the `convert_distcp_to_pt.py` script:

```bash
# Get path to the latest checkpoint
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_predict_v2p5/lora/2b_cosmos_nemo_assets_lora/checkpoints
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

The model can be used for any generation mode. Simply use the appropriate JSON configuration:

```bash
# Text2World generation (0 conditional frames)
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/text2world_prompts.json \
  -o outputs/text2world \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_lora_training_2b_cosmos_nemo_assets

# Image2World generation (1 conditional frame)
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/image2world_prompts.json \
  -o outputs/image2world \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_lora_training_2b_cosmos_nemo_assets
```

The model automatically detects the generation mode based on the input:
- Provide text only → Text2World generation
- Provide 1 image frame → Image2World generation
- Provide 2+ video frames → Video2World generation

Generated videos will be saved to the output directory.

For more inference options and advanced usage, see [docs/inference.md](./inference.md).
