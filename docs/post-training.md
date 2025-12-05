# Post-Training Guide

## Prerequisites

### 1. Environment Setup

Follow the [Setup guide](./setup.md) for general environment setup instructions, including installing dependencies.

### 2. Hugging Face Configuration

Model checkpoints are automatically downloaded during post-training if they are not present. Configure Hugging Face as follows:

```bash
# Login with your Hugging Face token (required for downloading models)
hf auth login

# Set custom cache directory for HF models
# Default: ~/.cache/huggingface
export HF_HOME=/path/to/your/hf/cache
```

> **💡 Tip**: Ensure you have sufficient disk space in `HF_HOME`.

### 3. Training Output Directory

Configure where training checkpoints and artifacts will be saved:

```bash
# Set output directory for training checkpoints and artifacts
# Default: /tmp/imaginaire4-output
export IMAGINAIRE_OUTPUT_ROOT=/path/to/your/output/directory
```

> **💡 Tip**: Ensure you have sufficient disk space in `IMAGINAIRE_OUTPUT_ROOT`.

## Weights & Biases (W&B) Logging

By default, training will attempt to log metrics to Weights & Biases. You have several options:

### Option 1: Enable W&B

To enable full experiment tracking with W&B:

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)
3. Set the environment variable:

    ```bash
    export WANDB_API_KEY=your_api_key_here
    ```

4. Launch training with the following command:

    ```bash
    EXP=your_experiment_name_here

    torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
      --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
      experiment=${EXP}
    ```

### Option 2: Disable W&B

Add `job.wandb_mode=disabled` to your training command to disable wandb logging:

```bash
EXP=your_experiment_name_here

torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=${EXP} \
  job.wandb_mode=disabled
```

## Checkpointing

Training uses two checkpoint formats, each optimized for different use cases:

### 1. Distributed Checkpoint (DCP) Format

**Primary format for training checkpoints.**

- **Structure**: Multi-file directory with sharded model weights
- **Used for**: Saving checkpoints during training, resuming training
- **Advantages**:
  - Efficient parallel I/O for multi-GPU training
  - Supports FSDP (Fully Sharded Data Parallel)
  - Optimized for distributed workloads

**Example directory structure:**

```
checkpoints/
├── iter_{NUMBER}/
│   ├── model/
│   │   ├── .metadata
│   │   └── __0_0.distcp
│   ├── optim/
│   ├── scheduler/
│   └── trainer/
└── latest_checkpoint.txt
```

### 2. Consolidated PyTorch (.pt) Format

**Single-file format for inference and distribution.**

- **Structure**: Single `.pt` file containing the complete model state
- **Used for**: Inference, model sharing, initial post-training
- **Advantages**:
  - Easy to distribute and version control
  - Standard PyTorch format
  - Simpler for single-GPU workflows

### Loading Checkpoints

The training system **supports loading from both formats**:

**Load DCP checkpoint (for resuming training):**

```python
load_path="checkpoints/nvidia/Cosmos-Predict2.5-2B/dcp"
```

**Load consolidated checkpoint (for starting post-training):**

```python
load_path="checkpoints/nvidia/Cosmos-Predict2.5-2B/consolidated/model.pt"
```

> **Note**: When you download pretrained models from Hugging Face, they are typically in consolidated `.pt` format. The training system will automatically load this format and begin training.

### Saving Checkpoints

**All checkpoints saved during training use DCP format**. This ensures:

- Consistent checkpoint structure across training runs
- Optimal performance for distributed training

## Post-training Examples

For detailed training examples and configuration options, see:

- [Video2World Post-Training with Cosmos Nemo Assets](./post-training_video2world_cosmos_nemo_assets.md)

For post-training example with 14B, see:

- [Video2World Post-training for DreamGen Bench](./post-training_video2world_gr00t.md)
