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

"""
This file demonstrates how to use Context Parallelism (CP) in imaginaire4.

It has the following usages:
1. By reading the tests in order, you will learn how CP works and how to test its correctness.
2. Passing the tests ensure CP is working correctly. This is needed before merging MRs.

We annotate variables with the following shape suffixes:

B: batch size
S: sequence length
Q: query dimension
C: context dimension
H: number of heads
D: head dimension

Usage:
    torchrun --nproc_per_node=2 -m pytest -v --L1 projects/cosmos/diffusion/v2/context_parallel_test.py
"""

import pytest
import torch
import torch.distributed as dist
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed import get_process_group_ranks
from torch.nn.parallel import DistributedDataParallel as DDP

import cosmos_predict2._src.imaginaire.utils.distributed
from cosmos_predict2._src.common.utils.fsdp_helper import hsdp_device_mesh
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import distributed, misc
from cosmos_predict2._src.imaginaire.utils.config_helper import override
from cosmos_predict2._src.imaginaire.utils.context_parallel import split_inputs_cp
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.imaginaire.utils.helper_test import RunIf
from cosmos_predict2._src.imaginaire.utils.misc import set_random_seed
from cosmos_predict2._src.predict2.configs.common.defaults.ema import PowerEMAConfig
from cosmos_predict2._src.predict2.configs.common.defaults.tokenizer import DummyJointImageVideoConfig
from cosmos_predict2._src.predict2.configs.text2world.config import make_config
from cosmos_predict2._src.predict2.configs.text2world.defaults.conditioner import VideoConditionerFpsPaddingConfig
from cosmos_predict2._src.predict2.models.text2world_model import DiffusionModel as Text2WorldModel
from cosmos_predict2._src.predict2.models.text2world_model import Text2WorldModelConfig
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import Attention, MiniTrainDIT
from cosmos_predict2._src.predict2.utils.test_helper import compare_tensors


@pytest.mark.L1
@pytest.mark.parametrize("backend", ["torch", "transformer_engine"])
def test_self_attention(backend):
    batch_size = 2
    sequence_length = 16
    query_dim = 64
    n_heads = 4
    head_dim = 32

    attn = Attention(query_dim=query_dim, n_heads=n_heads, head_dim=head_dim, backend=backend).cuda()
    x_B_S_Q = torch.randn(batch_size, sequence_length, query_dim).cuda()
    y_B_S_Q = attn(x_B_S_Q)
    assert x_B_S_Q.shape == y_B_S_Q.shape

    q_B_S_H_D, k_B_S_H_D, v_B_S_H_D = attn.compute_qkv(x_B_S_Q)
    scale = head_dim**-0.5
    q_B_S_H_D = scale * q_B_S_H_D
    q_B_H_S_D = rearrange(q_B_S_H_D, "B S H D -> B H S D")
    k_B_H_S_D = rearrange(k_B_S_H_D, "B S H D -> B H S D")
    v_B_H_S_D = rearrange(v_B_S_H_D, "B S H D -> B H S D")
    attn_matrix = q_B_H_S_D @ k_B_H_S_D.transpose(-2, -1)
    attn_matrix = attn_matrix.softmax(dim=-1)
    z_B_H_S_D = attn_matrix @ v_B_H_S_D
    z_B_S_HD = rearrange(z_B_H_S_D, "B H S D -> B S (H D)")
    z_B_S_Q = attn.output_proj(z_B_S_HD)
    z_B_S_Q = attn.output_dropout(z_B_S_Q)
    tols = dict(atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(y_B_S_Q, z_B_S_Q, **tols)


@RunIf(min_gpus=2)
@pytest.mark.L1
def test_self_attention_with_cp():
    # Initialize parallel state
    cosmos_predict2._src.imaginaire.utils.distributed.init()
    parallel_state.initialize_model_parallel(context_parallel_size=2)
    process_group = parallel_state.get_context_parallel_group()

    # CP is only supported with bf16 or fp16.
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    # Set the parameters and create the attention module.
    batch_size = 2
    sequence_length = 64
    query_dim = 64
    n_heads = 4
    head_dim = 32
    attn = Attention(query_dim=query_dim, n_heads=n_heads, head_dim=head_dim).cuda()

    # DDP syncs the model parameters across GPUs.
    attn_ddp = DDP(attn, process_group=process_group)
    # Create inputs and broadcast them to all ranks.
    x_B_S_Q = torch.randn(batch_size, sequence_length, query_dim).cuda()
    dist.broadcast(x_B_S_Q, 0)
    # Run the attention module.
    y_without_cp_B_S_Q = attn_ddp(x_B_S_Q)
    # Gather outputs from all ranks and assert they are the same.
    gathered_y = [torch.randn_like(y_without_cp_B_S_Q) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_y, y_without_cp_B_S_Q)
    torch.testing.assert_close(gathered_y[0], gathered_y[1])
    # Compute the loss and backpropagate.
    loss_without_cp = y_without_cp_B_S_Q.mean()
    loss_without_cp.backward()
    # Store the gradients for later comparison.
    grads_without_cp = [p.grad.clone() for p in attn.parameters()]
    # Clear the gradients.
    attn_ddp.zero_grad()

    # Enable context parallelism.
    attn_ddp.module.set_context_parallel_group(
        process_group, get_process_group_ranks(process_group), torch.cuda.Stream()
    )
    # Split the inputs along the sequence dimension.
    x_B_S2_Q = split_inputs_cp(x=x_B_S_Q, seq_dim=1, cp_group=process_group)
    assert x_B_S2_Q.shape[1] == sequence_length // 2
    # Run the attention module with context parallelism.
    y_with_cp_B_S2_Q = attn_ddp(x_B_S2_Q)
    assert y_with_cp_B_S2_Q.shape[1] == sequence_length // 2
    # Compare the results with and without context parallelism.
    y_without_cp_B_S2_Q = split_inputs_cp(x=y_without_cp_B_S_Q, seq_dim=1, cp_group=process_group)
    tols = dict(atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(y_without_cp_B_S2_Q, y_with_cp_B_S2_Q, **tols)
    # Compute the loss and backpropagate.
    cp_size = len(get_process_group_ranks(process_group))
    loss = y_with_cp_B_S2_Q.mean() * cp_size
    loss.backward()
    # Compare the gradients with and without context parallelism.
    for p, grad_without_cp in zip(attn.parameters(), grads_without_cp):
        grad_with_cp = p.grad.clone()
        tols = dict(atol=8e-3, rtol=1.6e-2)
        torch.testing.assert_close(grad_with_cp, grad_without_cp, **tols)

    parallel_state.destroy_model_parallel()


"""
test:
torchrun --nproc_per_node=2 -m pytest -v --L1 projects/cosmos/diffusion/v2/context_parallel_test.py::test_dit_with_cp
"""

N_HEADS = 8
HEAD_DIM = 32
DIT_MODEL_CONFIG = L(MiniTrainDIT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=N_HEADS * HEAD_DIM,
    num_blocks=2,
    num_heads=N_HEADS,
    concat_padding_mask=False,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    extra_per_block_abs_pos_emb=True,
)


@RunIf(min_gpus=2)
@pytest.mark.L1
def test_dit_with_cp():
    # Initialize parallel state
    cosmos_predict2._src.imaginaire.utils.distributed.init()
    parallel_state.initialize_model_parallel(context_parallel_size=2)
    try:
        process_group = parallel_state.get_context_parallel_group()

        # CP is only supported with bf16 or fp16.
        # despite we are using bf16 in the real training, here we should float16 for high precision
        dtype = torch.float16
        torch.set_default_dtype(dtype)
        torch.manual_seed(0)

        # Set the parameters and create the attention module.
        batch_size = 1

        net = instantiate(DIT_MODEL_CONFIG).to(dtype).cuda()

        # Create inputs and broadcast them to all ranks.
        noise_labels = torch.randn(batch_size, 1).cuda().to(dtype=dtype)
        crossattn_emb = torch.randn(batch_size, 512, 1024).cuda().to(dtype=dtype)
        fps = torch.randint(16, 32, (batch_size,)).cuda().to(dtype=dtype)
        shape = [64, 512, 512]  # [T, H, W]
        x_B_C_T_H_W = torch.randn(batch_size, 16, shape[0] // 8, shape[1] // 16, shape[2] // 16).cuda().to(dtype=dtype)
        for input in [x_B_C_T_H_W, noise_labels, crossattn_emb, fps]:
            dist.broadcast(input, 0)

        # DDP syncs the model parameters across GPUs.
        net_ddp = DDP(net, process_group=process_group)
        # Run the attention module.
        y_without_cp_B_C_T_H_W = net(x_B_C_T_H_W, noise_labels, crossattn_emb, fps)

        # Gather outputs from all ranks and assert they are the same.
        gathered_y = [torch.randn_like(y_without_cp_B_C_T_H_W) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_y, y_without_cp_B_C_T_H_W)
        torch.testing.assert_close(gathered_y[0], gathered_y[1])
        # Compute the loss and backpropagate.
        loss_without_cp = y_without_cp_B_C_T_H_W.mean()
        loss_without_cp.backward()
        # Store the gradients for later comparison.
        grads_without_cp = []
        for p in net.parameters():
            if p.grad is not None:
                grads_without_cp.append(p.grad.clone())
        # Clear the gradients.
        net_ddp.zero_grad()

        # Enable context parallelism.
        # Feel free to take a look at how `enable_context_parallel` is implemented.
        # Right now we skip the context parallelism during cross attention.
        net_ddp.module.enable_context_parallel(process_group)

        # Run the model with context parallelism.
        x_B_C_T2_H_W = split_inputs_cp(x=x_B_C_T_H_W, seq_dim=2, cp_group=process_group)
        y_with_cp_B_C_T2_H_W = net_ddp(x_B_C_T2_H_W, noise_labels, crossattn_emb, fps)
        assert y_with_cp_B_C_T2_H_W.shape[2] == x_B_C_T_H_W.shape[2] // 2
        # Compare the results with and without context parallelism.
        y_without_cp_B_Q_T2_H_W = split_inputs_cp(x=y_without_cp_B_C_T_H_W, seq_dim=2, cp_group=process_group)
        tols = dict(atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(y_without_cp_B_Q_T2_H_W, y_with_cp_B_C_T2_H_W, **tols)

        # Compute the loss and backpropagate.
        loss = y_with_cp_B_C_T2_H_W.mean()
        loss.backward()
        # Collect the gradients.
        grads_with_cp = []
        names = []
        for name, p in net_ddp.module.named_parameters():
            if p.grad is not None:
                grads_with_cp.append(p.grad.clone())
                names.append(name)
        # Compare the gradients with and without context parallelism.
        compare_tensors(names, grads_with_cp, grads_without_cp, atol=0.01, rtol=0.05)
    finally:
        # Destroy parallel state
        parallel_state.destroy_model_parallel()


"""
test:
torchrun --nproc_per_node=2 -m pytest -v --L1 projects/cosmos/diffusion/v2/context_parallel_test.py::test_diffusion_model_with_cp2
torchrun --nproc_per_node=4 -m pytest -v --L1 projects/cosmos/diffusion/v2/context_parallel_test.py::test_diffusion_model_with_cp2
torchrun --nproc_per_node=8 -m pytest -v --L1 projects/cosmos/diffusion/v2/context_parallel_test.py::test_diffusion_model_with_cp2
"""


@RunIf(min_gpus=2)
@pytest.mark.L1
def test_diffusion_model_with_cp2():
    cosmos_predict2._src.imaginaire.utils.distributed.init()
    parallel_state.initialize_model_parallel(context_parallel_size=1)
    cp_size = 2
    cp_mesh = hsdp_device_mesh(sharding_group_size=cp_size)
    if False:
        text2world_model_config = Text2WorldModelConfig(
            tokenizer=DummyJointImageVideoConfig,
            conditioner=VideoConditionerFpsPaddingConfig,
            net=DIT_MODEL_CONFIG,
            ema=PowerEMAConfig,
            precision="float16",  # using float16 for high precision
            fsdp_shard_size=cp_size,
        )
    else:
        config = make_config()
        config = override(
            config,
            [
                "--",
                "experiment=text2world_overfit_ddp_video-only",
                "model.config.fsdp_shard_size=2",
                "model.config.net.num_blocks=2",
                "model.config.precision=float16",
                "tokenizer=dummy_tokenizer",
            ],
        )
        text2world_model_config = config.model.config
        easy_io.set_s3_backend(
            backend_args={
                "backend": "s3",
                "path_mapping": {
                    "s3://rundir/": f"s3://bucket/{config.job.path}/",
                },
                "s3_credential_path": config.checkpoint.save_to_object_store.credentials,
            }
        )

    dtype = torch.float16

    # text2world_model_config = override(text2world_model_config)
    # Imports config utilities
    misc.set_random_seed(seed=0, by_rank=True)

    # Create the config
    model = Text2WorldModel(text2world_model_config).cuda()
    model.on_train_start()

    # Define how to synthesize data
    def video_batch():
        batch_size = 2
        num_frames = 34

        # batch_size = 1
        # num_frames = 57

        # video batch
        data_batch = {
            "dataset_name": "video_data",
            "video": (torch.randn(batch_size, 3, num_frames, 256, 256) * 255).to(dtype=torch.uint8).cuda(),
            "t5_text_embeddings": torch.randn(batch_size, 512, 1024, dtype=dtype).cuda(),
            "fps": 24 * torch.ones(batch_size, dtype=dtype).cuda(),
            "image_size": 256 * torch.ones(batch_size, 4, dtype=dtype).cuda(),
            "num_frames": num_frames * torch.ones(batch_size, dtype=dtype).cuda(),
            "padding_mask": torch.zeros(batch_size, 1, 256, 256, dtype=dtype).cuda(),
        }
        return data_batch

    # First, let's examine the correctness of the model with only FSDP, without context parallelism.
    # To test the correctness of FSDP, we will sync the input data across all ranks.
    cp_process_group = cp_mesh.get_group(mesh_dim="shard")
    dp_process_group = cp_mesh.get_group(mesh_dim="replicate")
    min_cp_rank_in_current_group = min(get_process_group_ranks(cp_process_group))
    data_batch = video_batch()
    for value in data_batch.values():
        if isinstance(value, torch.Tensor):
            dist.broadcast(value, min_cp_rank_in_current_group, group=cp_process_group)
    set_random_seed(min_cp_rank_in_current_group, by_rank=False)
    output_batch_without_cp, loss_without_cp = model.training_step(data_batch, 0)
    # Gather x0_pred from all ranks and assert they are the same.
    x0_pred_without_cp_B_Q_T_H_W = output_batch_without_cp["model_pred"].x0
    # Gather losses from all ranks and assert they are the same.
    gathered_loss = [torch.randn_like(loss_without_cp) for _ in range(cp_size)]
    dist.all_gather(gathered_loss, loss_without_cp, group=cp_process_group)
    torch.testing.assert_close(gathered_loss[0], gathered_loss[1])
    gathered_x0_pred = [torch.randn_like(x0_pred_without_cp_B_Q_T_H_W) for _ in range(cp_size)]
    dist.all_gather(gathered_x0_pred, x0_pred_without_cp_B_Q_T_H_W, group=cp_process_group)
    torch.testing.assert_close(gathered_x0_pred[0], gathered_x0_pred[1])

    # make sure dp is working well, having different loss
    if dp_process_group.size() > 1:
        gathered_loss = [torch.randn_like(loss_without_cp) for _ in range(dp_process_group.size())]
        dist.all_gather(gathered_loss, loss_without_cp, group=dp_process_group)
        assert gathered_loss[0] != gathered_loss[1]

    # Backpropagate the loss.
    loss_without_cp.backward()
    # Store the gradients for later comparison.
    grads_without_cp = []
    for p in model.net.parameters():
        if p.grad is not None:
            grads_without_cp.append(p.grad.clone())
    # Clear the gradients.
    model.zero_grad()
    parallel_state.destroy_model_parallel()

    parallel_state.initialize_model_parallel(context_parallel_size=cp_size)

    # Therefore, in the following we explicitly make sure that the data is different across ranks.
    # Also, the random seed should be different across ranks.
    cp_group = parallel_state.get_context_parallel_group()
    # Set random seed to be different for the second GPU
    set_random_seed(0, by_rank=True)
    # Create different data for the second GPU
    if distributed.get_rank() != min_cp_rank_in_current_group:
        data_batch = video_batch()

    # Run the model with context parallelism.
    output_batch_with_cp, loss_with_cp = model.training_step(data_batch, 0)
    x0_pred_with_cp_B_Q_T2_H_W = output_batch_with_cp["model_pred"].x0
    assert x0_pred_with_cp_B_Q_T2_H_W.shape[2] == x0_pred_without_cp_B_Q_T_H_W.shape[2] // 2
    # Compare the results with and without context parallelism.
    x0_pred_without_cp_B_Q_T2_H_W = split_inputs_cp(x=x0_pred_without_cp_B_Q_T_H_W, seq_dim=2, cp_group=cp_group)
    tols = dict(atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(x0_pred_without_cp_B_Q_T2_H_W, x0_pred_with_cp_B_Q_T2_H_W, **tols)
    # Backpropagate the loss.
    loss_with_cp.backward()
    # Collect the gradients.
    grads_with_cp = []
    names = []
    for name, p in model.net.named_parameters():
        if p.grad is not None:
            grads_with_cp.append(p.grad.clone())
            names.append(name)
    # Compare the gradients with and without context parallelism.
    compare_tensors(names, grads_with_cp, grads_without_cp, atol=0.01, rtol=0.05)
