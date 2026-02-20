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
import torch.distributed as dist

from cosmos_predict2._src.imaginaire.utils.misc import get_local_tensor_if_DTensor


def calculate_adaptive_weight(scm_loss, sid_loss, last_layer=None):
    scm_grads = torch.autograd.grad(scm_loss.mean(), last_layer.full_tensor(), retain_graph=True)[0]
    sid_grads = torch.autograd.grad(sid_loss.mean(), last_layer.full_tensor(), retain_graph=True)[0]
    dist.all_reduce(scm_grads, op=dist.ReduceOp.AVG)
    dist.all_reduce(sid_grads, op=dist.ReduceOp.AVG)
    d_weight = torch.norm(scm_grads) / (torch.norm(sid_grads) + 1e-5)
    return d_weight.detach()


def update_master_weights(optimizer: torch.optim.Optimizer):
    if getattr(optimizer, "master_weights", False) and optimizer.param_groups_master is not None:
        params, master_params = [], []
        for group, group_master in zip(optimizer.param_groups, optimizer.param_groups_master):
            for p, p_master in zip(group["params"], group_master["params"]):
                params.append(get_local_tensor_if_DTensor(p.data))
                master_params.append(p_master.data)
        torch._foreach_copy_(params, master_params)


def concat_condition(cond1, cond2):
    kwargs1, kwargs2 = cond1.to_dict(skip_underscore=False), cond2.to_dict(skip_underscore=False)
    kwargs = {}
    for key, value in kwargs1.items():
        if value is not None and isinstance(value, torch.Tensor):
            kwargs[key] = torch.cat([value, kwargs2[key]], dim=0)
        else:
            kwargs[key] = value
    return type(cond1)(**kwargs)


def expand_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expands the input tensor `x` to have the same
    number of dimensions as the `target` tensor.

    Pads `x` with singleton dimensions on the end.

    # Example

    ```
    x = torch.ones(5)
    target = torch.ones(5, 10, 30, 1, 10)

    x = expand_like(x, target)
    print(x.shape) # <- [5, 1, 1, 1, 1]
    ```

    Args:
        x (torch.Tensor): The input tensor to expand.
        target (torch.Tensor): The target tensor whose shape length
            will be matched.

    Returns:
        torch.Tensor: The expanded tensor `x` with trailing singleton
            dimensions.
    """
    x = torch.atleast_1d(x)
    while len(x.shape) < len(target.shape):
        x = x[..., None]
    return x


# used for multiview model inference
def to_model_input(data_batch, model):
    """
    Similar to misc.to, but avoid converting uint8 "video" to float
    """
    for k, v in data_batch.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        data_batch[k] = _v
    return data_batch
