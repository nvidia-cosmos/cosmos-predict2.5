# Cosmos Predict2-Distill Codebase
Last update: Dec. 17, 2025

### Overview
This project folder contains core functionalities of distilling Cosmos-Predict/Transfer series of models using the [DMD2](https://arxiv.org/abs/2405.14867) algorithm.

The goal of distillation is distill a pre-trained video diffusion model (the teacher model) which typically requires tens of inference steps, into a few-step student model. The student and teacher nets typically have the same architecture. We also distill the classifier-free guidance (CFG) knowledge of the teacher into the student model, so that the student inference does not require CFG. This brings another 2x speedup.

During distillation training, an auxiliary critic network (often referred to as the "fake score net" in the literature and code) is trained alongside the student. The training process alternates between updating the student and critic networks.


> **Note:** This README covers code walk through and post-training distillation instructions (distilled post-training checkpoint). For detailed inference instructions using the distilled base model, please see `cosmos-predict2.5/docs/inference.md`.


### Key Differences from Standard Cosmos Training

- **Trainer and checkpointer** (see `checkpointer/` and `trainer/`): Since we train two networks (student and critic), we add support for saving and loading both.
- **Training step** (see `models/`): The training step alternates between updating the student and critic. The loss functions also differ from the standard diffusion/flow-matching loss.
- **Inference script** (see `inference/`): The student requires only 1–4 steps. We manually define the diffusion timestep (noise-level) schedule for student inference instead of relying on a third-party sampler.
- **Math formulation**: We use TrigFlow (introduced in the sCM paper: https://arxiv.org/abs/2410.11081) as a shared parameterization for both DMD2 and consistency distillation (coming soon). The parameterization is compatible with distilling a RectifiedFlow teacher model.


### What Remains Unchanged

The following aspects remain the same as standard Cosmos model training:
- **Data loading mechanism.**
- **Conditioning mechanism** via the `Conditioner` object: This primarily includes text embedding conditioning and first-frame conditioning in the Video2World case.
- **Network architectures**: The student and critic networks typically resemble and are initialized from the teacher network.

## Folder Structure

- **Networks** (`projects/cosmos/predict2_distill/networks/`): The discriminator networks defined here are optionally used in DMD2 distillation. Otherwise, the networks are the same as the base teacher model defined in `projects/cosmos/predict2`.

- **Models** (`projects/cosmos/predict2_distill/models/`): Contains the trainable Imaginaire models. These scripts define network initialization, loss functions, and training steps called by the trainer.
  - `distillation_mixin.py`: Common methods shared by both supported distillation methods.
  - Other files follow the naming convention `<teacher_model_type>_distill_<distillation_method>.py`.

- **Configs**: Defines Hydra-style configurations for all aspects of training.
  - `registry_predict2p5.py`: The entry-point config file that registers all per-component and per-experiment configs for distilling the Predict2.5 model. The trainer takes this file as input. Similarly, when distilling the Transfer2.5 model, use `registry_transfer2p5.py`.
  - `defaults/`: Default configurations for different training components: model architecture, data, checkpointer, callbacks, etc.
  - `experiment/`: Experiment-specific configurations. Override hyper-parameter values here as needed.


## How to Distill Your Own Model

To add distillation support for a custom Cosmos model, create two new scripts:

1. **Model class**: Add the distilled model class under `models/<teacher_model_type>_distill_<distillation_method>.py`.

2. **Experiment config**: Add the corresponding experiment config in `configs/experiment/experiments_<distillation_method>_<teacher_model_type>.py`. Refer to existing files as examples. Be sure to include a runnable, up-to-date single-node training command in your experiment config file.

### Training

Create your own dataset and change `data_train` to point to your dataset in cosmos_predict2/_src/predict2/distill/configs/experiment/experiments_dmd2_predict2p5.py. Then run the following command to launch an example distillation training

```bash
torchrun --nproc_per_node=4 --master_port=12340 -m scripts.train --config=cosmos_predict2/_src/predict2/distill/configs/registry_predict2p5.py -- experiment=dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V
```

This command:
- Registeres the trainer, optimizer, dataset/dataloader, and the predict2.5 distillation model to be trained as specified in the `registry_predict2p5.py`.
- Registers all experiments as defined in various scripts in `configs/experiment/` folder (those added to Hydra via calling `cs.store`).
- Starts training: the trainer iterates training steps by calling the 'training_step' function defined in the model.

```bash
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_predict2_distill/predict2_distill/dmd2_trigflow_distill_cosmos_predict2_2B_bidirectional_TnI2V/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python ./scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR

```


## Code Walkthrough

Below is a walkthrough using the Predict2.5 Video2World model as an example.

```python
class Video2WorldModelDistillDMD2TrigFlow(DistillationCoreMixin, TrigFlowMixin, Video2WorldModel):
    ...
```

The distillation model inherits from three classes:
- **`DistillationCoreMixin`**: Common distillation-related code.
- **`TrigFlowMixin`**: Training-time timestep sampling functions, since we use TrigFlow as a unified parameterization for both distillation methods.
- **`Video2WorldModel`**: The teacher model class (in this case, Predict2.5), allowing reuse of its tokenizer, data handling, conditioner, etc.

> **Note:** The order of inheritance matters.

### Training Step Implementation

The high-level `training_step` that alternates between student and critic phases is defined in `DistillationCoreMixin`. For DMD2, implement two methods in your model:

#### Student Phase (`training_step_generator`)

1. Freeze the critic (and discriminator if enabled); unfreeze the student.
2. Sample time and noise; generate few-step student samples from noise.
3. Re-noise the student-generated samples to the sampled time, then feed this re-noised state to the teacher twice (conditional/unconditional) to form the CFG target. Also feed the same re-noised state to the critic if enabled.
4. Compute DMD2 losses from teacher and critic predictions; backpropagate into the student only. Optionally include GAN terms if configured.

#### Critic Phase (`training_step_critic`)

1. Freeze the student; unfreeze the critic (and discriminator if enabled).
2. Generate student samples via a short backward simulation (a few reverse steps); re-noise to the sampled time.
3. Train the critic on these student samples to fit the denoising target. If a discriminator head is used, also run the real/noisy-real path and apply GAN loss.

### Key Functions

- **`backward_simulation`**: Performs a few reverse-time steps with the student to craft slightly denoised samples for critic training. This is not a full trajectory and is controlled by config.

- **`get_data_and_condition`**: For CFG distillation, the teacher needs both conditional and unconditional forwards, so this function returns `(raw_state, x0, condition, uncondition)`. For video, we use identical conditional-frame masks for both; the only difference is text conditioning.
