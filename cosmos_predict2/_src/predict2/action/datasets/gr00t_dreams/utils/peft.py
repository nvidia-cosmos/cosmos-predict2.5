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
from peft import LoraConfig, get_peft_model


def _wrap_forward(model):
    def _forward(inputs):
        backbone_inputs, action_inputs = model.prepare_input(inputs)
        backbone_outputs = model.backbone(backbone_inputs)
        action_head_outputs = model.action_head(backbone_outputs, action_inputs)
        model.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    model.forward = _forward
    return model


def get_lora_model(model, rank=32, lora_alpha=16, lora_dropout=0.1):
    target_modules = []

    # Inspect model structure to find the correct paths
    for name, module in model.named_modules():
        # Look for linear layers in attention mechanisms
        if isinstance(module, torch.nn.Linear):
            if any(x in name for x in ["q_proj", "v_proj", "to_q", "to_v", "k_proj", "to_k"]):
                target_modules.append(name)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model = _wrap_forward(model)

    return model
