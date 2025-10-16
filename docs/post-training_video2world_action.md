# Video2World Post-training for Action-conditioned Video Prediction

We provide an example post-training instruction from a pre-trained video2world checkpoint.
Before proceeding, please read the [Post-training Guide](./post-training.md) for detailed setup steps and important post-training instructions, including checkpointing and best practices. This will ensure you are fully prepared for post-training with Cosmos-Predict2.5.

## 1. Preparing Data
### 1.1 Download Bridge training dataset
We use the train/validation splits of the Bridge dataset from IRASim for action-conditioned post-training.
To download and prepare the dataset, run the following commands under the `cosmos-predict2/` directory:
```
mkdir -p datasets && curl -L https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_train_data.tar.gz | tar -xvz -C datasets/
cd datasets
mv opensource_robotdata/bridge ./
```

Your dataset directory structure should look like this:
```
datasets/bridge/
├── annotations/
│   ├── *.json
├── videos/
    ├── *.mp4
```

Each JSON file in the `annotations/` folder contains the end-effector pose and gripper width of the robot arm for each frame in the corresponding video.
Specifically, each file includes:
- `state`: The end-effector pose of the robot arm at each timestep, represented as [x, y, z, roll, pitch, yaw].
    - (x, y, z) denotes the gripper's position in world coordinates.
    - (roll, pitch, yaw) describes its orientation in Euler angles.
- `continuous_gripper_state`: The width of the gripper at each timestep, indicating whether it is open or closed. A value of 0 means the gripper is open, and 1 means it is closed.
- `action`: The gripper's displacement at each timestep.
    - The first six dimensions represent displacement in (x, y, z, roll, pitch, yaw) within the gripper coordinate frame.
    - The last (seventh) dimension is a binary value indicating whether the gripper should open (1) or close (0).

We use this information as conditioning input for video generation.


## 2. Post-training

##### Cosmos-Predict2-2B-Video2World
Run the following command to launch an example post-training job using the Bridge dataset:
```bash
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320 ~dataloader_train.dataloaders

```
See `../cosmos_predict2/experiments/base/action.py` to understand how the dataloader is defined.
To add action as additional condition, we create new `conditioner` to support action in `cosmos_predict2/_src/predict2/configs/conditioner.py`.

##### Checkpoint Output Structure
Checkpoints are saved to ${IMAGINAIRE_OUTPUT_ROOT}/PROJECT/GROUP/NAME/checkpoints. By default, IMAGINAIRE_OUTPUT_ROOT is /tmp/imaginaire4-output. We strongly recommend setting IMAGINAIRE_OUTPUT_ROOT to a location with sufficient storage space for your checkpoints.

```
For the example command above:
- PROJECT: `cosmos_predict2_action_conditioned`
- GROUP: `cosmos_predict_v2p5`
- NAME: `2b_bridge_action_conditioned`

##### Configuration Snippet
Below is a configuration snippet defining the experiment setup:
```python
  AC_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B_256_320 = LazyDict(
    dict(
        defaults=[
            DEFAULT_CHECKPOINT.experiment,
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            {"override /data_train": "bridge_13frame_480_640_train"},
            {"override /data_val": "bridge_13frame_480_640_val"},
             "_self_",
        ],
        job=dict(
            group="cosmos_predict_v2p5",
            name="2b_bridge_action_conditioned",
            project="cosmos_predict2_action_conditioned",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=0.1,
        ),
    ),
)

```


## 3. Inference for Bridge

### 3.1 Converting DCP Checkpoint to Consolidated PyTorch Format

Since the checkpoints are saved in DCP format during training, you need to convert them to consolidated PyTorch format (.pt) for inference. Use the `convert_distcp_to_pt.py` script:

```bash

CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_bridge_action_conditioned/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python ./scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR

```

This conversion will create three files:

- `model.pt`: Full checkpoint containing both regular and EMA weights
- `model_ema_fp32.pt`: EMA weights only in float32 precision
- `model_ema_bf16.pt`: EMA weights only in bfloat16 precision (recommended for inference)


#### 3.2 Running inference
After converting the checkpoint, you can run inference with your post-trained checkpoint (e.g., at 1000 iterations) use the command below.
Specify the path to the checkpoint in the assets/action_conditioned/basic/inference_params.json file:
```
python examples/action_conditioned.py \
-i assets/action_conditioned/basic/inference_params.json -o outputs/action_conditioned/basic \
--checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
--experiment ac_reason_embeddings_rectified_flow_2b_256_320
```
