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

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]


# Cosmos-NeMo-Assets dataset and dataloader for LoRA training
example_dataset_cosmos_nemo_assets_lora = L(VideoDataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    num_frames=93,
    video_size=(704, 1280),
)

dataloader_train_cosmos_nemo_assets_lora = L(get_generic_dataloader)(
    dataset=example_dataset_cosmos_nemo_assets_lora,
    sampler=L(get_sampler)(dataset=example_dataset_cosmos_nemo_assets_lora),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# LoRA post-training configuration for all modes (text2world, image2world, video2world)
# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiment=predict2_lora_training_2b_cosmos_nemo_assets
predict2_lora_training_2b_cosmos_nemo_assets = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="lora",
        name="2b_cosmos_nemo_assets_lora",
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets_lora,
    checkpoint=dict(
        save_iter=200,
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_from_object_store=dict(
            enabled=False,
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[2_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=1000,
        callbacks=dict(
            heart_beat=dict(
                save_s3=False,
            ),
            iter_speed=dict(
                hit_thres=200,
                save_s3=False,
            ),
            device_monitor=dict(
                save_s3=False,
            ),
            every_n_sample_reg=dict(
                every_n=200,
                save_s3=False,
            ),
            every_n_sample_ema=dict(
                every_n=200,
                save_s3=False,
            ),
            wandb=dict(
                save_s3=False,
            ),
            wandb_10x=dict(
                save_s3=False,
            ),
            dataloader_speed=dict(
                save_s3=False,
            ),
        ),
    ),
    model=dict(
        config=dict(
            # Enable LoRA training
            use_lora=True,
            # LoRA configuration parameters
            lora_rank=32,
            lora_alpha=32,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
            # Training configuration for all three modes
            # The model will randomly sample between 0, 1, and 2 conditional frames during training
            min_num_conditional_frames=0,  # Allow text2world (0 frames)
            max_num_conditional_frames=2,  # Allow up to video2world (2 frames)
            # Probability distribution for sampling number of conditional frames
            # This controls how often each mode is trained:
            # - 0 frames: text2world (33.3%)
            # - 1 frame: image2world (33.3%)
            # - 2 frames: video2world (33.3%)
            conditional_frames_probs={0: 0.333, 1: 0.333, 2: 0.334},
            # Optional: set conditional_frame_timestep for better control
            conditional_frame_timestep=-1.0,  # Default -1 means not effective
            # Keep the default conditioning strategy
            conditioning_strategy="frame_replace",
            denoise_replace_gt_frames=True,
        ),
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
)

cs = ConfigStore.instance()

# Register the configurations with Hydra ConfigStore
for _item in [
    predict2_lora_training_2b_cosmos_nemo_assets,
]:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
