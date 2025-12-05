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
from cosmos_predict2._src.predict2.callbacks.validation_draw_sample import ValidationDrawSample
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]


# Cosmos-NeMo-Assets dataset and dataloader for LoRA training
# Text format configuration
example_dataset_cosmos_nemo_assets_lora_txt = L(VideoDataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    num_frames=93,
    video_size=(704, 1280),
)

dataloader_train_cosmos_nemo_assets_lora_txt = L(get_generic_dataloader)(
    dataset=example_dataset_cosmos_nemo_assets_lora_txt,
    sampler=L(get_sampler)(dataset=example_dataset_cosmos_nemo_assets_lora_txt),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# JSON format configuration with long prompts
# Training dataset
example_dataset_cosmos_nemo_assets_lora_json_train = L(VideoDataset)(
    dataset_dir="datasets/cosmos_nemo_assets_json/train",
    num_frames=93,
    video_size=(704, 1280),
    caption_format="json",
    prompt_type="long",
)

dataloader_train_cosmos_nemo_assets_lora_json = L(get_generic_dataloader)(
    dataset=example_dataset_cosmos_nemo_assets_lora_json_train,
    sampler=L(get_sampler)(dataset=example_dataset_cosmos_nemo_assets_lora_json_train),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# Validation dataset
example_dataset_cosmos_nemo_assets_lora_json_val = L(VideoDataset)(
    dataset_dir="datasets/cosmos_nemo_assets_json/validation",
    num_frames=93,
    video_size=(704, 1280),
    caption_format="json",
    prompt_type="long",
)

dataloader_val_cosmos_nemo_assets_lora_json = L(get_generic_dataloader)(
    dataset=example_dataset_cosmos_nemo_assets_lora_json_val,
    sampler=L(get_sampler)(dataset=example_dataset_cosmos_nemo_assets_lora_json_val),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# LoRA post-training configuration for all modes (text2world, image2world, video2world)
# Shared configuration parameters
_lora_defaults = [
    f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
    {"override /data_train": "mock"},
    {"override /data_val": "mock"},
    "_self_",
]

_lora_checkpoint_base = dict(
    # pyrefly: ignore  # missing-attribute
    load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
    load_from_object_store=dict(
        enabled=False,
    ),
    save_to_object_store=dict(
        enabled=False,
    ),
)

_lora_optimizer = dict(
    lr=2 ** (-14.5),
    weight_decay=0.001,
)

_lora_scheduler = dict(
    f_max=[0.5],
    f_min=[0.2],
    warm_up_steps=[2_000],
    cycle_lengths=[100000],
)

_lora_trainer = dict(
    run_validation=True,
    validation_iter=5,
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
        validation_draw_sample_reg=L(ValidationDrawSample)(
            n_samples=2,
            is_ema=False,
            save_s3=False,
            do_x0_prediction=False,
        ),
        validation_draw_sample_ema=L(ValidationDrawSample)(
            n_samples=2,
            is_ema=True,
            save_s3=False,
            do_x0_prediction=False,
        ),
    ),
)

_lora_model_config = dict(
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
)

_lora_model_parallel = dict(
    context_parallel_size=1,
)

# Text format experiment
# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiment=predict2_lora_training_2b_cosmos_nemo_assets_txt
predict2_lora_training_2b_cosmos_nemo_assets_txt = dict(
    defaults=_lora_defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="lora",
        name="2b_cosmos_nemo_assets_lora",
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets_lora_txt,
    checkpoint=dict(
        **_lora_checkpoint_base,
        save_iter=200,
    ),
    optimizer=_lora_optimizer,
    scheduler=_lora_scheduler,
    trainer=_lora_trainer,
    model=_lora_model_config,
    model_parallel=_lora_model_parallel,
)

# JSON format experiment with long prompts
# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiment=predict2_lora_training_2b_cosmos_nemo_assets_json
predict2_lora_training_2b_cosmos_nemo_assets_json = dict(
    defaults=_lora_defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="lora",
        name="2b_cosmos_nemo_assets_json_lora",
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets_lora_json,
    dataloader_val=dataloader_val_cosmos_nemo_assets_lora_json,
    checkpoint=dict(
        **_lora_checkpoint_base,
        save_iter=30,
    ),
    optimizer=_lora_optimizer,
    scheduler=_lora_scheduler,
    trainer=_lora_trainer,
    model=_lora_model_config,
    model_parallel=_lora_model_parallel,
)

cs = ConfigStore.instance()

# Register the configurations with Hydra ConfigStore
for _item in [
    predict2_lora_training_2b_cosmos_nemo_assets_txt,
    predict2_lora_training_2b_cosmos_nemo_assets_json,
]:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
