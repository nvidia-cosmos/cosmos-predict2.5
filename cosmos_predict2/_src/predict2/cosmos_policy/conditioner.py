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
Extended Conditioner classes for Cosmos Policy.

This module provides mutable (frozen=False) versions of condition dataclasses
and a modified GeneralConditioner that skips uncondition when all dropout rates are 0.

Conditions need to be mutable for Cosmos Policy since it modifies parts of the condition
objects during training.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

import torch

from cosmos_predict2._src.predict2.conditioner import (
    DataType,
    broadcast_condition,
)
from cosmos_predict2._src.predict2.conditioner import (
    GeneralConditioner as _GeneralConditioner,
)

# Mutable versions of condition dataclasses for Cosmos Policy
# These are redefined with frozen=False to allow modification of attributes after creation


@dataclass(frozen=False)
class BaseCondition(ABC):
    """
    Mutable version of BaseCondition for Cosmos Policy.

    Attributes:
        _is_broadcasted: Flag indicating if parallel broadcast splitting
            has been performed. This is an internal implementation detail.
    """

    _is_broadcasted: bool = False

    def to_dict(self, skip_underscore: bool = True) -> Dict[str, Any]:
        """Converts the condition to a dictionary.

        Returns:
            Dictionary containing the condition's fields and values.
        """
        return {f.name: getattr(self, f.name) for f in fields(self) if not (f.name.startswith("_") and skip_underscore)}

    @property
    def is_broadcasted(self) -> bool:
        return self._is_broadcasted

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> BaseCondition:
        """Broadcasts and splits the condition across the checkpoint parallelism group.
        For most condition, such as Text2WorldCondition, we do not need split.

        Args:
            process_group: The process group for broadcast and split

        Returns:
            A new BaseCondition instance with the broadcasted and split condition.
        """
        if self.is_broadcasted:
            return self
        return broadcast_condition(self, process_group)


@dataclass(frozen=False)
class Text2WorldCondition(BaseCondition):
    """Mutable version of Text2WorldCondition for Cosmos Policy."""

    crossattn_emb: Optional[torch.Tensor] = None
    data_type: DataType = DataType.VIDEO
    padding_mask: Optional[torch.Tensor] = None
    fps: Optional[torch.Tensor] = None

    def edit_data_type(self, data_type: DataType) -> Text2WorldCondition:
        """Edit the data type of the condition.

        Args:
            data_type: The new data type.

        Returns:
            A new Text2WorldCondition instance with the new data type.
        """
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["data_type"] = data_type
        return type(self)(**kwargs)

    @property
    def is_video(self) -> bool:
        return self.data_type == DataType.VIDEO


class GeneralConditioner(_GeneralConditioner):
    """
    Extended GeneralConditioner for Cosmos Policy.

    Modifies get_condition_uncondition to skip uncondition generation when all dropout rates are 0,
    supporting always-conditional generation instead of CFG.
    """

    def get_condition_uncondition(
        self,
        data_batch: Dict,
    ) -> Tuple[Any, Any]:
        """
        NOTE (user): Modified to remove the "uncondition" since we are doing always-conditional generation instead of CFG.

        Processes the provided data batch to generate two sets of outputs: conditioned and unconditioned. This method
        manipulates the dropout rates of embedders to simulate two scenarios — one where all conditions are applied
        (conditioned), and one where they are removed or reduced to the minimum (unconditioned).

        This method first sets the dropout rates to zero for the conditioned scenario to fully apply the embedders' effects.
        For the unconditioned scenario, it sets the dropout rates to 1 (or to 0 if the initial unconditional dropout rate
        is insignificant) to minimize the embedders' influences, simulating an unconditioned generation.

        Parameters:
            data_batch (Dict): The input data batch that contains all necessary information for embedding processing. The
                            data is expected to match the required format and keys expected by the embedders.

        Returns:
            Tuple[Any, Any]: A tuple containing two condition:
                - The first one contains the outputs with all embedders fully applied (conditioned outputs).
                - The second one contains the outputs with embedders minimized or not applied (unconditioned outputs).
        """
        cond_dropout_rates, dropout_rates = {}, {}
        for emb_name, embedder in self.embedders.items():
            cond_dropout_rates[emb_name] = 0.0
            dropout_rates[emb_name] = 1.0  # Always drop embedders for uncondition (CFG)

        # Always generate both condition and uncondition for CFG
        condition: Any = self(data_batch, override_dropout_rate=cond_dropout_rates)
        un_condition: Any = self(data_batch, override_dropout_rate=dropout_rates)

        return condition, un_condition


class VideoConditioner(GeneralConditioner):
    """VideoConditioner using mutable Text2WorldCondition for Cosmos Policy."""

    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Text2WorldCondition:
        output = super()._forward(batch, override_dropout_rate)
        return Text2WorldCondition(**output)
