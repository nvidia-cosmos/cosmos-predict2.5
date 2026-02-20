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

import copy
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import omegaconf
import torch
from einops import rearrange
from hydra.core.config_store import ConfigStore
from torch.distributed import get_process_group_ranks

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast_split_tensor, find_split
from cosmos_predict2._src.predict2.conditioner import BooleanFlag, GeneralConditioner, ReMapkey, TextAttr
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import (
    Video2WorldCondition,
    Video2WorldConditionV2,
)
from cosmos_predict2._src.transfer2.networks.siglip2_image_context import SigLip2EmbImgContext


@dataclass(frozen=True)
class ControlVideo2WorldCondition(Video2WorldCondition):
    control_input_edge: Optional[torch.Tensor] = None
    control_input_vis: Optional[torch.Tensor] = None
    control_input_depth: Optional[torch.Tensor] = None
    control_input_seg: Optional[torch.Tensor] = None
    control_input_inpaint: Optional[torch.Tensor] = None
    control_input_edge_mask: Optional[torch.Tensor] = None
    control_input_vis_mask: Optional[torch.Tensor] = None
    control_input_depth_mask: Optional[torch.Tensor] = None
    control_input_seg_mask: Optional[torch.Tensor] = None
    control_input_inpaint_mask: Optional[torch.Tensor] = None
    control_input_hdmap_bbox: Optional[torch.Tensor] = None
    latent_control_input: Optional[torch.Tensor] = None
    control_context_scale: Optional[float] = 1.0

    def set_control_condition(
        self, latent_control_input: torch.Tensor, control_weight: float = 1.0
    ) -> "ControlVideo2WorldCondition":
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["latent_control_input"] = latent_control_input
        kwargs["control_context_scale"] = control_weight
        return type(self)(**kwargs)

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> "ControlVideo2WorldCondition":
        if self.is_broadcasted:
            return self
        # Handle parent class broadcasting
        parent_condition = super().broadcast(process_group)
        kwargs = parent_condition.to_dict(skip_underscore=False)

        # Handle control tensor broadcasting
        latent_control_input = self.latent_control_input
        if latent_control_input is not None and process_group is not None:
            cp_ranks = get_process_group_ranks(process_group)
            cp_size = len(cp_ranks)
            use_spatial_split = cp_size > latent_control_input.shape[2] or latent_control_input.shape[2] % cp_size != 0
            after_split_shape = find_split(latent_control_input.shape, cp_size) if use_spatial_split else None
            if latent_control_input.dim() == 5:  # B, C, T, H, W
                _, _, T, _, _ = latent_control_input.shape
                if T > 1 and process_group.size() > 1:
                    if use_spatial_split:
                        latent_control_input = rearrange(latent_control_input, "b c t h w -> b c (t h w)")
                    latent_control_input = broadcast_split_tensor(
                        latent_control_input, seq_dim=2, process_group=process_group
                    )
                    if use_spatial_split:
                        latent_control_input = rearrange(
                            latent_control_input,
                            "b c (t h w) -> b c t h w",
                            t=after_split_shape[0],
                            h=after_split_shape[1],
                        )
        control_context_scale = self.control_context_scale
        if isinstance(control_context_scale, torch.Tensor) and process_group is not None:
            if control_context_scale.dim() >= 5:  # B, T, H, W, D or N, B, T, H, W, D
                T = control_context_scale.shape[-4]
                if T > 1 and process_group.size() > 1:
                    seq_dim = control_context_scale.dim() - 4
                    control_context_scale = broadcast_split_tensor(
                        control_context_scale, seq_dim=seq_dim, process_group=process_group
                    )

        kwargs["latent_control_input"] = latent_control_input
        kwargs["control_context_scale"] = control_context_scale
        return type(self)(**kwargs)


@dataclass(frozen=True)
class ControlVideo2WorldConditionV2(Video2WorldConditionV2):
    control_input: Optional[torch.Tensor] = None
    latent_control_input: Optional[torch.Tensor] = None

    def set_control_condition(
        self, latent_control_input: torch.Tensor, control_weight: float = 1.0
    ) -> "ControlVideo2WorldConditionV2":
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["latent_control_input"] = latent_control_input
        kwargs["control_context_scale"] = control_weight
        return type(self)(**kwargs)

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> "ControlVideo2WorldConditionV2":
        if self.is_broadcasted:
            return self
        # Handle parent class broadcasting
        parent_condition = super().broadcast(process_group)
        kwargs = parent_condition.to_dict(skip_underscore=False)

        # Handle control tensor broadcasting
        latent_control_input = self.latent_control_input
        if latent_control_input is not None and process_group is not None:
            if latent_control_input.dim() == 5:  # B, C, T, H, W
                _, _, T, _, _ = latent_control_input.shape
                print("broadcasting latent_control_input", latent_control_input.shape)
                if T > 1 and process_group.size() > 1:
                    latent_control_input = broadcast_split_tensor(
                        latent_control_input, seq_dim=2, process_group=process_group
                    )

        kwargs["latent_control_input"] = latent_control_input
        return type(self)(**kwargs)


@dataclass(frozen=True)
class ControlVideo2WorldConditionImageContext(ControlVideo2WorldCondition):
    img_context_emb: Optional[torch.Tensor] = None


class ControlVideo2WorldConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> ControlVideo2WorldCondition:
        output = self._forward(batch, override_dropout_rate)
        return ControlVideo2WorldCondition(**output)

    def _forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Processes the input batch through all configured embedders, applying conditional dropout rates if specified.
        Output tensors for each key are concatenated along the dimensions specified in KEY2DIM.

        Parameters:
            batch (Dict): The input data batch to process.
            override_dropout_rate (Optional[Dict[str, float]]): Optional dictionary to override default dropout rates
                                                                per embedder key.

        Returns:
            Dict: A dictionary of output tensors concatenated by specified dimensions.

        Note:
            In case the network code is sensitive to the order of concatenation, you can either control the order via \
            config file or make sure the embedders return a unique key for each output.
        """
        output = defaultdict(list)
        if override_dropout_rate is None:
            override_dropout_rate = {}

        # make sure emb_name in override_dropout_rate is valid
        for emb_name in override_dropout_rate.keys():
            assert emb_name in self.embedders, f"invalid name found {emb_name}"

        for emb_name, embedder in self.embedders.items():
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                emb_out = {}
                if isinstance(embedder.input_key, str):
                    if embedder.input_key in batch:
                        emb_out = embedder(
                            embedder.random_dropout_input(
                                batch[embedder.input_key], override_dropout_rate.get(emb_name, None)
                            )
                        )
                elif isinstance(embedder.input_key, (list, omegaconf.listconfig.ListConfig)):
                    emb_out = embedder(
                        *[
                            embedder.random_dropout_input(batch.get(k), override_dropout_rate.get(emb_name, None), k)
                            for k in embedder.input_key
                            if k in batch
                        ]
                    )
                else:
                    raise KeyError(
                        f"Embedder '{embedder.__class__.__name__}' requires an 'input_key' attribute to be defined as either a string or list of strings"
                    )
            for k, v in emb_out.items():
                output[k].append(v)
        # Concatenate the outputs
        return {k: torch.cat(v, dim=self.KEY2DIM.get(k, -1)) for k, v in output.items()}


class ControlVideo2WorldConditionerV2(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> ControlVideo2WorldConditionV2:
        output = super()._forward(batch, override_dropout_rate)
        return ControlVideo2WorldConditionV2(**output)


class ControlVideo2WorldConditionerImageContext(ControlVideo2WorldConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> ControlVideo2WorldConditionImageContext:
        output = super()._forward(batch, override_dropout_rate)
        return ControlVideo2WorldConditionImageContext(**output)

    def get_condition_with_negative_prompt(
        self,
        data_batch: Dict,
    ) -> Tuple[Any, Any]:
        """
        Similar functionality as get_condition_uncondition
        But use negative prompts for unconditon, remove image context for classifier free guidance
        """
        cond_dropout_rates, uncond_dropout_rates = {}, {}
        for emb_name, embedder in self.embedders.items():
            cond_dropout_rates[emb_name] = 0.0
            if isinstance(embedder, TextAttr):
                uncond_dropout_rates[emb_name] = 0.0
            else:
                uncond_dropout_rates[emb_name] = 1.0 if embedder.dropout_rate > 1e-4 else 0.0

        data_batch_neg_prompt = copy.deepcopy(data_batch)
        if "neg_t5_text_embeddings" in data_batch_neg_prompt:
            if isinstance(data_batch_neg_prompt["neg_t5_text_embeddings"], torch.Tensor):
                data_batch_neg_prompt["t5_text_embeddings"] = data_batch_neg_prompt["neg_t5_text_embeddings"]
            data_batch_neg_prompt["image_context"] = None
        log.info("Getting condition with image context")
        condition: Any = self(data_batch, override_dropout_rate=cond_dropout_rates)
        log.info("Getting uncondition with negative prompt without image context")
        un_condition: Any = self(data_batch_neg_prompt, override_dropout_rate=uncond_dropout_rates)

        return condition, un_condition


_SHARED_CONFIG = dict(
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    text=L(TextAttr)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
    ),
    use_video_condition=L(BooleanFlag)(
        input_key="fps",
        output_key="use_video_condition",
        dropout_rate=0.2,
    ),
    control_input_edge=L(ReMapkey)(
        input_key="control_input_edge",
        output_key="control_input_edge",
        dropout_rate=0.0,
        dtype=None,
    ),
    control_input_vis=L(ReMapkey)(
        input_key="control_input_vis",
        output_key="control_input_vis",
        dropout_rate=0.0,
        dtype=None,
    ),
    control_input_depth=L(ReMapkey)(
        input_key="control_input_depth",
        output_key="control_input_depth",
        dropout_rate=0.0,
        dtype=None,
    ),
    control_input_seg=L(ReMapkey)(
        input_key="control_input_seg",
        output_key="control_input_seg",
        dropout_rate=0.0,
        dtype=None,
    ),
    control_input_inpaint=L(ReMapkey)(
        input_key="control_input_inpaint",
        output_key="control_input_inpaint",
        dropout_rate=0.0,
        dtype=None,
    ),
    control_input_edge_mask=L(ReMapkey)(
        input_key="control_input_edge_mask",
        output_key="control_input_edge_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    control_input_vis_mask=L(ReMapkey)(
        input_key="control_input_vis_mask",
        output_key="control_input_vis_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    control_input_depth_mask=L(ReMapkey)(
        input_key="control_input_depth_mask",
        output_key="control_input_depth_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    control_input_seg_mask=L(ReMapkey)(
        input_key="control_input_seg_mask",
        output_key="control_input_seg_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    control_input_inpaint_mask=L(ReMapkey)(
        input_key="control_input_inpaint_mask",
        output_key="control_input_inpaint_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
)

_SHARED_CONFIG_AV = copy.deepcopy(_SHARED_CONFIG)
_SHARED_CONFIG_AV.pop("control_input_edge")
_SHARED_CONFIG_AV.pop("control_input_vis")
_SHARED_CONFIG_AV.pop("control_input_depth")
_SHARED_CONFIG_AV.pop("control_input_seg")
_SHARED_CONFIG_AV.pop("control_input_edge_mask")
_SHARED_CONFIG_AV.pop("control_input_vis_mask")
_SHARED_CONFIG_AV.pop("control_input_depth_mask")
_SHARED_CONFIG_AV.pop("control_input_seg_mask")


_SHARED_CONFIG_AV["control_input_hdmap_bbox"] = L(ReMapkey)(
    input_key="control_input_hdmap_bbox",
    output_key="control_input_hdmap_bbox",
    dropout_rate=0.0,
    dtype=None,
)


VideoPredictionControlConditioner: LazyDict = L(ControlVideo2WorldConditioner)(
    **_SHARED_CONFIG,
)

VideoPredictionControlConditionerV2: LazyDict = L(ControlVideo2WorldConditionerV2)(
    **_SHARED_CONFIG,
)

VideoPredictionControlConditionerAV: LazyDict = L(ControlVideo2WorldConditioner)(
    **_SHARED_CONFIG_AV,
)


VideoPredictionControlConditionerImageContext: LazyDict = L(ControlVideo2WorldConditionerImageContext)(
    **_SHARED_CONFIG,
    reference_image_context=L(SigLip2EmbImgContext)(
        input_key=["images", "video", "image_context"],
        output_key=None,
        dropout_rate=0.0,
    ),
)


def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_control_conditioner",
        node=VideoPredictionControlConditioner,
    )

    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_control_conditioner_v2",
        node=VideoPredictionControlConditionerV2,
    )
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_control_conditioner_av",
        node=VideoPredictionControlConditionerAV,
    )
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_control_conditioner_image_context",
        node=VideoPredictionControlConditionerImageContext,
    )
