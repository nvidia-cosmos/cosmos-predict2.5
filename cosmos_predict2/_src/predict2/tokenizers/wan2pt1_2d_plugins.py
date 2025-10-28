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

# Original code author is Yan Wang, yanwa@nvidia.com
#
# The purpose of those plugins is to enable CP (at the moment splitting by video height) for Wan2.1 tokenizer inference,
# they work by wrapping original Conv/Attention modules' forward calls.
# In case of convolutional layers ranks receive/send required data from/to ranks that are handling adjacent parts of the video
# to properly compute convolution, pad their input with that data, run original module, and truncate output if needed.
# Attention is not parallelized, so all ranks gather (if needed) all parts of the video run attention and split result (if needed) back into chunks
#
# Only function `plugin_mount` should be used outside this file

from itertools import chain

import torch
import torch.distributed as distributed


def _create_adj_groups(
    grid_shape: tuple[int, int], cp_group: distributed.ProcessGroup
) -> tuple[list[distributed.ProcessGroup], list[distributed.ProcessGroup]]:
    grid_rows, grid_cols = grid_shape

    all_rank_groups = [None for _ in range(distributed.get_world_size())]
    group_ranks = distributed.get_process_group_ranks(cp_group) if cp_group is not None else []
    distributed.all_gather_object(all_rank_groups, group_ranks)

    all_groups = list(set(tuple(rank_group) for rank_group in all_rank_groups))
    global_rank = distributed.get_rank()
    in_row_groups = None
    in_col_groups = None

    for cp_group_ranks in all_groups:
        if len(cp_group_ranks) == 0:
            continue
        tmp_in_row_groups = []
        tmp_in_col_groups = []
        scaling_factor = min(cp_group_ranks)
        for row in range(grid_rows):
            in_row_adj_ranks_list = [
                (cp_group_ranks[row * grid_cols + col], cp_group_ranks[row * grid_cols + col + 1])
                for col in range(grid_cols - 1)
            ]

            adj_groups = [distributed.new_group(in_row_adj_ranks) for in_row_adj_ranks in in_row_adj_ranks_list]
            # print(f'{global_rank=}\t{adj_groups=}')
            # if all(adj_group == distributed.GroupMember.NON_GROUP_MEMBER for adj_group in adj_groups):
            #     continue
            tmp_in_row_groups.append(adj_groups)

        for col in range(grid_cols):
            in_col_adj_ranks_list = [
                (row * grid_cols + col + scaling_factor, (row + 1) * grid_cols + col + scaling_factor)
                for row in range(grid_rows - 1)
            ]

            adj_groups = [distributed.new_group(in_col_adj_ranks) for in_col_adj_ranks in in_col_adj_ranks_list]

            # if all(adj_group == distributed.GroupMember.NON_GROUP_MEMBER for adj_group in adj_groups):
            #     continue
            tmp_in_col_groups.append(adj_groups)

        if global_rank in cp_group_ranks:
            in_row_groups = tmp_in_row_groups
            in_col_groups = tmp_in_col_groups
    return in_row_groups, in_col_groups


class _ModulePlugin:
    def __init__(self, module, module_id, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups):
        self.module = module
        self.module_id = module_id
        self.enable = True
        self.implement_forward()
        self.plugin_config = plugin_config

        self.in_row_adj_groups = in_row_adj_groups
        self.in_col_adj_groups = in_col_adj_groups
        self.cp_group = cp_group
        self.group_rank = distributed.get_rank(group=cp_group)
        self.group_rank_to_global_rank = distributed.get_process_group_ranks(cp_group)
        self.cp_group_size = len(self.group_rank_to_global_rank)

        self.grid_shape = grid_shape
        self.grid_rows, self.grid_cols = grid_shape

        self.row_id = self.group_rank // self.grid_cols
        self.col_id = self.group_rank % self.grid_cols

        # Neighbor rank calculation
        self.left_neighbor = self.group_rank - 1 if self.col_id > 0 else None
        self.right_neighbor = self.group_rank + 1 if self.col_id < self.grid_cols - 1 else None
        self.top_neighbor = self.group_rank - self.grid_cols if self.row_id > 0 else None
        self.bottom_neighbor = self.group_rank + self.grid_cols if self.row_id < self.grid_rows - 1 else None

        self.my_row_groups = self.in_row_adj_groups[self.row_id]
        self.my_col_groups = self.in_col_adj_groups[self.col_id]

    def implement_forward(self):
        module = self.module
        if not hasattr(module, "old_forward"):
            module.old_forward = module.forward

        self.new_forward = self.get_new_forward()

        def forward(*args, **kwargs):
            self.update_config()
            return self.new_forward(*args, **kwargs) if self.enable else module.old_forward(*args, **kwargs)

        module.forward = forward

    def set_enable(self, enable=True):
        self.enable = enable

    def get_new_forward(self):
        raise NotImplementedError

    def update_config(self, config: dict | None = None):
        if config is None:
            config = self.plugin_config.get(self.module_id[0], {})

        for key, value in config.items():
            setattr(self, key, value)


class _Conv3DSafeNewPlugin(_ModulePlugin):
    def __init__(
        self,
        module,
        module_id,
        plugin_config=None,
        cp_group=None,
        grid_shape=None,
        in_row_adj_groups=None,
        in_col_adj_groups=None,
    ):
        super().__init__(module, module_id, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups)

        self.kernel_size = getattr(module, "kernel_size", (1, 1, 1))

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size, self.kernel_size)

        kernel_height = self.kernel_size[1]
        d_height = kernel_height - 1
        self.padding_left_height = d_height // 2
        self.padding_right_height = d_height - self.padding_left_height
        self.height_padding_flag = self.padding_left_height if d_height > 0 else 0

        kernel_width = self.kernel_size[2]
        d_width = kernel_width - 1
        self.padding_left_width = d_width // 2
        self.padding_right_width = d_width - self.padding_left_width
        self.width_padding_flag = self.padding_left_width if d_width > 0 else 0

    def pad_context_2d(self, h):
        """2D context padding: simultaneously perform padding on both height and width dimensions"""

        if self.width_padding_flag == 0 and self.height_padding_flag == 0:
            return h

        # First step: perform padding on width dimension (dim=4)
        if self.width_padding_flag > 0:
            h = self._pad_width_dimension(h)

        # Second step: perform padding on height dimension (dim=3)
        if self.height_padding_flag > 0:
            h = self._pad_height_dimension(h)

        return h

    def _pad_width_dimension(self, h):
        """Perform padding on width dimension (dim=4)"""
        # Only pad in necessary directions, no padding at boundaries
        contexts_to_concat = []

        # Left padding: only needed for non-leftmost columns
        if self.left_neighbor is not None:
            share_to_left = h[:, :, :, :, : self.padding_left_width].contiguous()
            if self.col_id % 2:
                # Odd column, handle left first
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_left, group=self.my_row_groups[self.col_id - 1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                # Even column, handle left later
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_left, group=self.my_row_groups[self.col_id - 1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            contexts_to_concat.append(left_context)

        # Add original data
        contexts_to_concat.append(h)

        # Right padding: only needed for non-rightmost columns
        if self.right_neighbor is not None:
            share_to_right = h[:, :, :, :, -self.padding_right_width :].contiguous()
            if self.col_id % 2:
                # Odd column, handle right later
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_right, group=self.my_row_groups[self.col_id])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                # Even column, handle right first
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_right, group=self.my_row_groups[self.col_id])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            contexts_to_concat.append(right_context)

        h_with_width_context = torch.cat(contexts_to_concat, dim=4)
        return h_with_width_context

    def _pad_height_dimension(self, h):
        """Perform padding on height dimension (dim=3)"""
        # Only pad in necessary directions, no padding at boundaries
        contexts_to_concat = []

        # Top padding: only needed for non-topmost rows
        if self.top_neighbor is not None:
            share_to_top = h[:, :, :, : self.padding_left_height].contiguous()
            if self.row_id % 2:
                # Odd row, handle top first
                padding_list = [torch.zeros_like(share_to_top) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_top, group=self.my_col_groups[self.row_id - 1])
                top_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                # Even row, handle top later
                padding_list = [torch.zeros_like(share_to_top) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_top, group=self.my_col_groups[self.row_id - 1])
                top_context = padding_list[0].to(h.device, non_blocking=True)
            contexts_to_concat.append(top_context)

        # Add original data
        contexts_to_concat.append(h)

        # Bottom padding: only needed for non-bottommost rows
        if self.bottom_neighbor is not None:
            share_to_bottom = h[:, :, :, -self.padding_right_height :].contiguous()
            if self.row_id % 2:
                # Odd row, handle bottom later
                padding_list = [torch.zeros_like(share_to_bottom) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_bottom, group=self.my_col_groups[self.row_id])
                bottom_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                # Even row, handle bottom first
                padding_list = [torch.zeros_like(share_to_bottom) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_bottom, group=self.my_col_groups[self.row_id])
                bottom_context = padding_list[1].to(h.device, non_blocking=True)
            contexts_to_concat.append(bottom_context)

        h_with_height_context = torch.cat(contexts_to_concat, dim=3)
        return h_with_height_context

    def get_new_forward(self):
        module = self.module

        def new_forward(hidden_states, cache_x=None):
            # If no padding is needed, directly use original forward
            if self.width_padding_flag == 0 and self.height_padding_flag == 0:
                return module.old_forward(hidden_states, cache_x)

            # Perform 2D context padding
            hidden_states = self.pad_context_2d(hidden_states)
            if cache_x is not None:
                cache_x = self.pad_context_2d(cache_x)

            # Execute convolution operation
            result = module.old_forward(hidden_states, cache_x)

            # Crop results, remove padding (only crop parts that actually had padding added)
            # First crop height dimension
            if self.height_padding_flag > 0:
                # Calculate actual cropping range
                start_h = self.padding_left_height if self.top_neighbor is not None else 0
                end_h = (
                    -self.padding_right_height
                    if (self.padding_right_height > 0 and self.bottom_neighbor is not None)
                    else None
                )
                if start_h > 0 or end_h is not None:
                    result = result[:, :, :, start_h:end_h]

            # Then crop width dimension
            if self.width_padding_flag > 0:
                # Calculate actual cropping range
                start_w = self.padding_left_width if self.left_neighbor is not None else 0
                end_w = (
                    -self.padding_right_width
                    if (self.padding_right_width > 0 and self.right_neighbor is not None)
                    else None
                )
                if start_w > 0 or end_w is not None:
                    result = result[:, :, :, :, start_w:end_w]
            return result

        return new_forward


class _Conv2DSafeNewPlugin(_ModulePlugin):
    def __init__(
        self,
        module,
        module_id,
        plugin_config=None,
        cp_group=None,
        grid_shape=None,
        in_row_adj_groups=None,
        in_col_adj_groups=None,
    ):
        super().__init__(module, module_id, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups)

        self.kernel_size = getattr(module, "kernel_size", (1, 1))
        self.stride = getattr(module, "stride", (1, 1))

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

        kernel_height = self.kernel_size[0]
        d_height = kernel_height - 1
        self.padding_left_height = d_height // 2
        self.padding_right_height = d_height - self.padding_left_height
        self.height_padding_flag = self.padding_left_height if d_height > 0 else 0

        kernel_width = self.kernel_size[1]
        d_width = kernel_width - 1
        self.padding_left_width = d_width // 2
        self.padding_right_width = d_width - self.padding_left_width
        self.width_padding_flag = self.padding_left_width if d_width > 0 else 0

    def pad_context_2d(self, h):
        if self.width_padding_flag == 0 and self.height_padding_flag == 0:
            return h

        if self.width_padding_flag > 0:
            h = self._pad_width_dimension(h)
        if self.height_padding_flag > 0:
            h = self._pad_height_dimension(h)

        return h

    def _pad_width_dimension(self, h):
        """Perform padding on width dimension (dim=3)"""
        # Only pad in necessary directions, no padding at boundaries
        contexts_to_concat = []

        # Left padding: only needed for non-leftmost columns
        if self.left_neighbor is not None:
            share_to_left = h[:, :, :, : self.padding_left_width].contiguous()
            if self.col_id % 2:
                # Odd column, handle left first
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_left, group=self.my_row_groups[self.col_id - 1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                # Even column, handle left later
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_left, group=self.my_row_groups[self.col_id - 1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            contexts_to_concat.append(left_context)

        # Add original data
        contexts_to_concat.append(h)

        # Right padding: only needed for non-rightmost columns
        if self.right_neighbor is not None:
            share_to_right = h[:, :, :, -self.padding_right_width :].contiguous()
            if self.col_id % 2:
                # Odd column, handle right later
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_right, group=self.my_row_groups[self.col_id])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                # Even column, handle right first
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_right, group=self.my_row_groups[self.col_id])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            contexts_to_concat.append(right_context)

        h_with_width_context = torch.cat(contexts_to_concat, dim=3)
        return h_with_width_context

    def _pad_height_dimension(self, h):
        """Perform padding on height dimension (dim=2)"""
        # Only pad in necessary directions, no padding at boundaries
        contexts_to_concat = []

        # Top padding: only needed for non-topmost rows
        if self.top_neighbor is not None:
            share_to_top = h[:, :, : self.padding_left_height].contiguous()
            if self.row_id % 2:
                # Odd row, handle top first
                padding_list = [torch.zeros_like(share_to_top) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_top, group=self.my_col_groups[self.row_id - 1])
                top_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                # Even row, handle top later
                padding_list = [torch.zeros_like(share_to_top) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_top, group=self.my_col_groups[self.row_id - 1])
                top_context = padding_list[0].to(h.device, non_blocking=True)
            contexts_to_concat.append(top_context)

        # Add original data
        contexts_to_concat.append(h)

        # Bottom padding: only needed for non-bottommost rows
        if self.bottom_neighbor is not None:
            share_to_bottom = h[:, :, -self.padding_right_height :].contiguous()
            if self.row_id % 2:
                padding_list = [torch.zeros_like(share_to_bottom) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_bottom, group=self.my_col_groups[self.row_id])
                bottom_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                padding_list = [torch.zeros_like(share_to_bottom) for _ in range(2)]
                distributed.all_gather(padding_list, share_to_bottom, group=self.my_col_groups[self.row_id])
                bottom_context = padding_list[1].to(h.device, non_blocking=True)
            contexts_to_concat.append(bottom_context)

        h_with_height_context = torch.cat(contexts_to_concat, dim=2)
        return h_with_height_context

    def get_new_forward(self):
        module = self.module

        def new_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = self.pad_context_2d(hidden_states)

            result = module.old_forward(hidden_states)

            if self.height_padding_flag:
                result = result[
                    :,
                    :,
                    (self.padding_left_height if self.top_neighbor is not None else None) : (
                        (-self.padding_right_height if self.padding_right_height > 0 else None)
                        if self.bottom_neighbor is not None
                        else None
                    ),
                ]

            if self.width_padding_flag:
                result = result[
                    :,
                    :,
                    :,
                    (self.padding_left_width if self.left_neighbor is not None else None) : (
                        (-self.padding_right_width if self.padding_right_width > 0 else None)
                        if self.right_neighbor is not None
                        else None
                    ),
                ]

            return result

        return new_forward


class _Conv2DSafeNewPluginStride2(_ModulePlugin):
    def __init__(
        self,
        module,
        module_id,
        plugin_config=None,
        cp_group=None,
        grid_shape=None,
        in_row_adj_groups=None,
        in_col_adj_groups=None,
    ):
        super().__init__(module, module_id, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups)

        self.kernel_size = getattr(module, "kernel_size", (1, 1))
        self.stride = getattr(module, "stride", (1, 1))

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

        kernel_height, kernel_width = self.kernel_size
        self.padding_height = (kernel_height - 1) // 2 if kernel_height > 1 else 0
        self.padding_width = (kernel_width - 1) // 2 if kernel_width > 1 else 0

        self.diagonal_sender = None
        self.diagonal_receiver = None

        self.right_sender = self.group_rank + 1 if self.col_id < self.grid_cols - 1 else None
        self.bottom_sender = self.group_rank + self.grid_cols if self.row_id < self.grid_rows - 1 else None
        self.left_receiver = self.group_rank - 1 if self.col_id > 0 else None
        self.top_receiver = self.group_rank - self.grid_cols if self.row_id > 0 else None

        if self.row_id < self.grid_rows - 1 and self.col_id < self.grid_cols - 1:
            self.diagonal_sender = self.group_rank + self.grid_cols + 1

        if self.row_id > 0 and self.col_id > 0:
            self.diagonal_receiver = self.group_rank - self.grid_cols - 1

    def pad_context_2d(self, h):
        if h is None or self.cp_group_size == 1:
            return h

        if self.padding_height == 0 and self.padding_width == 0:
            return h

        return self._pad_with_unidirectional_transfer(h)

    def _pad_with_unidirectional_transfer(self, h):
        if self.padding_width > 0:
            h = self._pad_width_with_concat(h)

        if self.padding_height > 0:
            h = self._pad_height_with_concat(h)

        if self.padding_width > 0 and self.padding_height > 0:
            h = self._handle_diagonal_transfer(h)

        return h

    def _pad_width_with_concat(self, h):
        contexts_to_concat = []
        contexts_to_concat.append(h)
        if self.right_sender is not None:
            tmp = torch.zeros(h.shape[0], h.shape[1], h.shape[2], self.padding_width, dtype=h.dtype, device=h.device)
            padding_list = [torch.zeros_like(tmp) for _ in range(2)]
            distributed.all_gather(padding_list, tmp, group=self.my_row_groups[self.col_id])
            contexts_to_concat.append(padding_list[1].to(h.device, non_blocking=True))

        if self.left_receiver is not None:
            left_boundary = h[:, :, :, : self.padding_width].contiguous()
            padding_list = [torch.zeros_like(left_boundary) for _ in range(2)]
            distributed.all_gather(padding_list, left_boundary, group=self.my_row_groups[self.col_id - 1])

        return torch.cat(contexts_to_concat, dim=3)

    def _pad_height_with_concat(self, h):
        contexts_to_concat = []
        contexts_to_concat.append(h)

        if self.bottom_sender is not None:
            tmp = torch.zeros(h.shape[0], h.shape[1], self.padding_height, h.shape[3], dtype=h.dtype, device=h.device)
            padding_list = [torch.zeros_like(tmp) for _ in range(2)]
            distributed.all_gather(padding_list, tmp, group=self.my_col_groups[self.row_id])
            contexts_to_concat.append(padding_list[1])

        if self.top_receiver is not None:
            top_boundary = h[:, :, : self.padding_height, :].contiguous()
            padding_list = [torch.zeros_like(top_boundary) for _ in range(2)]
            distributed.all_gather(padding_list, top_boundary, group=self.my_col_groups[self.row_id - 1])

        return torch.cat(contexts_to_concat, dim=2)

    def _handle_diagonal_transfer(self, h):
        if self.diagonal_sender is not None and self.right_sender is not None and self.bottom_sender is not None:
            # Receive data from bottom-right diagonal neighbor
            diagonal_data = torch.zeros(
                h.shape[0], h.shape[1], self.padding_height, self.padding_width, dtype=h.dtype, device=h.device
            )
            distributed.recv(diagonal_data, src=self.diagonal_sender)

            # Fill diagonal data to bottom-right corner
            # Since width and height padding have been performed earlier, h's size has increased
            # Need to place diagonal data at the bottom-right position
            original_h = h.shape[2] - self.padding_height
            original_w = h.shape[3] - self.padding_width
            h[:, :, original_h:, original_w:] = diagonal_data

        # Send data to top-left diagonal neighbor
        # Only devices that are not in the first row and not in the first column need to send diagonal data
        if self.diagonal_receiver is not None:
            # Send own top-left corner data to top-left diagonal neighbor
            # Take the top-left part of original data (data before padding)
            corner_data = h[:, :, : self.padding_height, : self.padding_width].contiguous()
            distributed.send(corner_data, dst=self.diagonal_receiver)

        return h

    def get_new_forward(self):
        module = self.module

        def new_forward(hidden_states):
            # Doing CP execution here causes torch.compile + CUDA graphs to deadlock, not sure why, so we just gather all data and, run convolution like we are not using CP, and return appropriate chunk
            # hidden_states = self.pad_context_2d(hidden_states)
            # return module.old_forward(hidden_states)

            is_last_row = self.row_id == self.grid_rows - 1
            is_last_col = self.col_id == self.grid_cols - 1

            # Remove ZeroPad from chunks
            if is_last_col:
                hidden_states = hidden_states[:, :, :, :-1]
            if is_last_row:
                hidden_states = hidden_states[:, :, :-1]

            gathered_tensors = [torch.zeros_like(hidden_states) for _ in range(self.cp_group_size)]
            distributed.all_gather(gathered_tensors, hidden_states.contiguous(), group=self.cp_group)

            combined_tensor = torch.cat(
                [torch.cat(gathered_tensors[c :: self.grid_cols], dim=2) for c in range(self.grid_cols)], dim=3
            )

            # Reapply ZeroPad to whole video
            combined_tensor = torch.nn.ZeroPad2d((0, 1, 0, 1)).eval()(combined_tensor)

            forward_output = module.old_forward(combined_tensor)

            chunk_h = forward_output.shape[2] // self.grid_rows
            chunk_w = forward_output.shape[3] // self.grid_cols

            local_output = forward_output[
                :,
                :,
                self.row_id * chunk_h : (self.row_id + 1) * chunk_h,
                self.col_id * chunk_w : (self.col_id + 1) * chunk_w,
            ].contiguous()

            return local_output

        return new_forward


class _WanAttentionPlugin(_ModulePlugin):
    def __init__(
        self,
        module,
        module_id,
        plugin_config=None,
        cp_group=None,
        grid_shape=None,
        in_row_adj_groups=None,
        in_col_adj_groups=None,
        all_gather_before_attention=False,
        cp_split_after_attention=True,
    ):
        self.all_gather_before_attention = all_gather_before_attention
        self.cp_split_after_attention = cp_split_after_attention

        super().__init__(module, module_id, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups)

    def get_new_forward(self):
        module = self.module

        def new_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            if self.all_gather_before_attention:
                gathered_tensors = [torch.zeros_like(hidden_states) for _ in range(self.cp_group_size)]
                distributed.all_gather(gathered_tensors, hidden_states, group=self.cp_group)

                combined_tensor = torch.cat(
                    [torch.cat(gathered_tensors[c :: self.grid_cols], dim=3) for c in range(self.grid_cols)], dim=4
                )
            else:
                combined_tensor = hidden_states

            forward_output = module.old_forward(combined_tensor)

            if self.cp_split_after_attention:
                chunk_h = forward_output.shape[3] // self.grid_rows
                chunk_w = forward_output.shape[4] // self.grid_cols

                local_output = forward_output[
                    :,
                    :,
                    :,
                    self.row_id * chunk_h : (self.row_id + 1) * chunk_h,
                    self.col_id * chunk_w : (self.col_id + 1) * chunk_w,
                ].contiguous()
            else:
                local_output = forward_output

            return local_output

        return new_forward


class _ResamplePlugin(_ModulePlugin):
    def __init__(
        self,
        module,
        module_id,
        plugin_config=None,
        cp_group=None,
        grid_shape=None,
        in_row_adj_groups=None,
        in_col_adj_groups=None,
    ):
        super().__init__(module, module_id, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups)

    def get_new_forward(self):
        module = self.module

        def new_forward(*args, **kwargs) -> torch.Tensor:
            return module.old_forward(*args, **kwargs)

        return new_forward

    def set_enable(self, enable=True):
        self.enable = enable

        if not (self.module.mode in ["downsample2d", "downsample3d"] and self.group_rank < self.cp_group_size - 1):
            return

        if self.enable is True:
            # Check if at boundaries
            is_last_row = self.row_id == self.grid_rows - 1
            is_last_col = self.col_id == self.grid_cols - 1

            # Determine padding based on position
            left_pad = 0
            right_pad = 1 if is_last_col else 0  # Last column needs right padding
            top_pad = 0
            bottom_pad = 1 if is_last_row else 0  # Last row needs bottom padding
            self.module.resample[0] = torch.nn.ZeroPad2d((left_pad, right_pad, top_pad, bottom_pad)).eval()
        else:
            self.module.resample[0] = torch.nn.ZeroPad2d((0, 1, 0, 1)).eval()


def plugin_mount(model, cp_group, grid_shape):
    """
    Register plugins and allow CP execution of Wan2.1 tokenizer

    Args:
        model (torch.nn.Module): instance of `projects.diffusion.v2.tokenizers.wan2pt1.WanVAE_`
        cp_group (distributed.ProcessGroup): CP group that will be used
        grid_shape (tuple[int, int]):

    Returns:
        plugins (dict[str, dict[str, _ModulePlugin]]): dict[layer_name, dict[plugin_id, _ModulePlugin]] dictionarly with plugins, allowing to turn them on/off
    """

    PLUGIN_CONFIG = {
        "attn": {
            "padding": 24,
            "top_k": 24,
            "top_k_chunk_size": 24,
            "attn_scale": 1.0,
            "token_num_scale": True,
            "dynamic_scale": True,
        },
        "conv_3d": {
            "padding": 1,
        },
        "conv_layer": {},
    }

    if cp_group is not None:
        group_rank_to_global_rank = distributed.get_process_group_ranks(cp_group)
        cp_group_size = len(group_rank_to_global_rank)
        assert cp_group_size == grid_shape[0] * grid_shape[1]

    in_row_adj_groups, in_col_adj_groups = _create_adj_groups(grid_shape, cp_group)
    if cp_group is None:
        return {}
    assert len(in_col_adj_groups) == grid_shape[1] and all(
        len(group) == grid_shape[0] - 1 for group in in_col_adj_groups
    )
    assert len(in_row_adj_groups) == grid_shape[0] and all(
        len(group) == grid_shape[1] - 1 for group in in_row_adj_groups
    )

    plugins = {}
    _conv_3d_plugin_mount(plugins, model, PLUGIN_CONFIG, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups)
    _conv_2d_plugin_stride2_mount(
        plugins, model, PLUGIN_CONFIG, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups
    )  # only for wan vae encoder
    _conv_2d_plugin_mount(plugins, model, PLUGIN_CONFIG, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups)
    _wanattention_plugin_mount(
        plugins, model, PLUGIN_CONFIG, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups
    )
    _resample_plugin_mount(plugins, model, PLUGIN_CONFIG, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups)

    return plugins


def _wanattention_plugin_mount(
    plugins: dict, model, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups
):
    plugins["wanattention"] = {}

    wanattention_gather_after = []
    wanattention_split_after = []
    for name, module in chain(model.named_modules()):
        if "middle" in name and module.__class__.__name__ == "AttentionBlock":
            (wanattention_gather_after if "encoder" in name else wanattention_split_after).append(module)

    for i, wanattention in enumerate(wanattention_gather_after):
        plugin_id = "wanattention", i
        plugins["wanattention"][plugin_id] = _WanAttentionPlugin(
            wanattention,
            plugin_id,
            plugin_config,
            cp_group,
            grid_shape,
            in_row_adj_groups,
            in_col_adj_groups,
            all_gather_before_attention=True,
            cp_split_after_attention=False,
        )

    for i, wanattention in enumerate(wanattention_split_after):
        plugin_id = "wanattention", i + len(wanattention_gather_after)
        plugins["wanattention"][plugin_id] = _WanAttentionPlugin(
            wanattention,
            plugin_id,
            plugin_config,
            cp_group,
            grid_shape,
            in_row_adj_groups,
            in_col_adj_groups,
            all_gather_before_attention=False,
            cp_split_after_attention=True,
        )


def _conv_3d_plugin_mount(
    plugins: dict, model, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups
):
    plugins["conv_3d"] = {}
    conv3d_s = []
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import CausalConv3d

    for name, module in chain(model.named_modules()):
        if any(
            p in name
            for p in [
                "encoder.middle.2",
                "encoder.head",
                "decoder.conv1",
                "decoder.middle.0",
            ]
        ):
            continue

        if (
            any(
                p in name
                for p in [
                    "conv1",
                    "conv2",
                ]
            )
            and "encoder" not in name
            and "decoder" not in name
        ):
            continue

        if isinstance(module, CausalConv3d) and module.kernel_size[1] > 1:
            conv3d_s.append(module)

    for i, conv in enumerate(conv3d_s):
        plugin_id = "conv_3d", i
        plugins["conv_3d"][plugin_id] = _Conv3DSafeNewPlugin(
            conv, plugin_id, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups
        )


def _conv_2d_plugin_stride2_mount(
    plugins: dict, model, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups
):
    plugins["conv_2d_stride2"] = {}
    conv2d_stride2_s = []

    for name, module in model.encoder.named_modules():
        if (
            any(
                p in name
                for p in [
                    "middle.2",
                    "head",
                ]
            )
            and ".resample" in name
            and module.__class__.__name__ == "Conv2d"
        ):
            continue
        if ".resample" in name and module.__class__.__name__ == "Conv2d":
            conv2d_stride2_s.append(module)

    for i, conv in enumerate(conv2d_stride2_s):
        plugin_id = "conv_2d_stride2", i
        plugins["conv_2d_stride2"][plugin_id] = _Conv2DSafeNewPluginStride2(
            conv, plugin_id, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups
        )


def _conv_2d_plugin_mount(
    plugins: dict, model, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups
):
    plugins["conv_2d"] = {}
    conv2d_s = []
    for name, module in model.decoder.named_modules():
        if any(p in name for p in ["conv1", "middle.0"]):
            continue
        if ".resample" in name and module.__class__.__name__ == "Conv2d":
            conv2d_s.append(module)

    for i, conv in enumerate(conv2d_s):
        plugin_id = "conv_2d", i
        plugins["conv_2d"][plugin_id] = _Conv2DSafeNewPlugin(
            conv, plugin_id, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups
        )


def _resample_plugin_mount(
    plugins: dict, model, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups
):
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Resample

    plugins["resample"] = {}
    resamples = []
    for name, module in model.named_modules():
        if isinstance(module, Resample) and module.mode in ["downsample2d", "downsample3d"]:
            resamples.append(module)

    for i, resample in enumerate(resamples):
        plugin_id = "resample", i
        plugins["resample"][plugin_id] = _ResamplePlugin(
            resample, plugin_id, plugin_config, cp_group, grid_shape, in_row_adj_groups, in_col_adj_groups
        )
