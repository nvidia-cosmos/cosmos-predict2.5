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

import torch
from einops import repeat

import cosmos_predict2._src.imaginaire.utils.distributed
from cosmos_predict2._src.imaginaire.utils import misc
from cosmos_predict2._src.imaginaire.utils.config_helper import override
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.configs.text2world.config import make_config
from cosmos_predict2._src.predict2.models.text2world_model import DiffusionModel

"""
torchrun --nproc_per_node=2 -m projects.cosmos.diffusion.v2.models.model_fsdp2_test
"""


def image_batch():
    batch_size = 1
    num_frame = 17
    image_batch_size = batch_size * num_frame // 2
    data_batch = {
        "dataset_name": "image_data",
        "images": torch.randn(batch_size * num_frame // 2, 3, 1024, 1024, dtype=torch.float32),
        "t5_text_embeddings": torch.randn(image_batch_size, 512, 1024, dtype=torch.float32),
        "t5_text_mask": torch.randint(0, 2, (image_batch_size, 512), dtype=torch.int64),
        "fps": torch.randint(16, 32, (image_batch_size,)).float(),
        "num_frames": torch.ones(image_batch_size) * 1.0,
        "image_size": repeat(
            torch.tensor([1024, 1024, 1024, 1024]),
            "... -> b ...",
            b=image_batch_size,
        ),
        "padding_mask": repeat(
            torch.zeros(size=(1, 1024, 1024)),
            "... -> b ...",
            b=image_batch_size,
        ),
    }
    return data_batch


def video_batch():
    batch_size = 1
    num_frame = 17
    # video batch
    data_batch = {
        "dataset_name": "video_data",
        "video": (torch.randn(batch_size, 3, num_frame, 1024, 1024) * 255).to(dtype=torch.uint8),
        "t5_text_embeddings": torch.randn(batch_size, 512, 1024, dtype=torch.float32),
        "t5_text_mask": torch.randint(0, 2, (batch_size, 512), dtype=torch.int64),
        "fps": torch.randint(16, 32, (batch_size,)).float(),
        "num_frames": torch.ones(batch_size) * num_frame,
        "image_size": repeat(
            torch.tensor([1024, 1024, 1024, 1024]),
            "... -> b ...",
            b=batch_size,
        ),
        "padding_mask": repeat(
            torch.zeros(size=(1, 1024, 1024)),
            "... -> b ...",
            b=batch_size,
        ),
    }
    return data_batch


def model_init_test():
    cosmos_predict2._src.imaginaire.utils.distributed.init()
    config = make_config()
    config = override(config, ["--", "experiment=error-free_fsdp_mock-data_base-cb"])
    easy_io.set_s3_backend(
        backend_args={
            "backend": "s3",
            "path_mapping": {
                "s3://rundir/": f"s3://{config.checkpoint.save_to_object_store.bucket}/{config.job.path}/",
            },
            "s3_credential_path": config.checkpoint.save_to_object_store.credentials,
        }
    )
    misc.set_random_seed(seed=config.trainer.seed, by_rank=True)
    model = DiffusionModel(config.model.config).cuda()
    model.on_train_start(torch.preserve_format)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-8)
    return model, optim


def model_forward_test():
    model, optim = model_init_test()
    model.on_train_start(torch.preserve_format)

    rank = torch.distributed.get_rank()

    image_batch_data = image_batch()
    video_batch_data = video_batch()

    for k, v in video_batch_data.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        video_batch_data[k] = _v

    output_batch, loss = model.training_step(video_batch_data, 1)
    loss.backward()
    optim.step()
    print(f"rank {rank} Video loss: {loss.item()}")
    model.on_before_zero_grad(None, None, iteration=1)

    for k, v in image_batch_data.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        image_batch_data[k] = _v
    output_batch, loss = model.training_step(image_batch_data, 2)
    loss.backward()
    model.clip_grad_norm_(1.0)
    print(f"rank {rank} Image loss: {loss.item()}")

    with model.ema_scope():
        print(f"rank {rank} ema works!")


if __name__ == "__main__":
    model_forward_test()
