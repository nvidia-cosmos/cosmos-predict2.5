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

import attrs

IS_PREPROCESSED_KEY = "is_preprocessed"

import math

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.modules.edm_sde import EDMSDE
from cosmos_predict2._src.predict2.models.text2world_model import EMAConfig
from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoderConfig


@attrs.define(slots=False)
class Cosmos2InteractiveModelConfig:
    # ---------------- Basic infra setup ----------------
    fsdp_shard_size: int = 8
    grad_clip: bool = True
    loss_reduce: str = "mean"
    precision: str = "bfloat16"
    resolution: str = "720"
    state_ch: int = 16  # latent DiT input channel number
    state_t: int = 24  # latent DiT input temporal duration
    use_torch_compile: bool = False

    # ---------------- Model architecture / components ----------------
    conditioner: LazyDict = None  # type: ignore
    ema: EMAConfig = EMAConfig()
    net: LazyDict = None  # type: ignore
    text_encoder_class: str = "reason1p1_7B"
    text_encoder_config: TextEncoderConfig | None = None
    tokenizer: LazyDict = None  # type: ignore

    # ---------------- Diffusion / noise process ----------------
    rectified_flow_loss_weight_uniform: bool = True
    rectified_flow_t_scaling_factor: float = 1.0
    scaling: str = "rectified_flow"
    sde: LazyDict = L(EDMSDE)(
        p_mean=0.0,
        p_std=1.0,
        sigma_max=80,
        sigma_min=0.0002,
    )
    # Selected time below corresponds to a uniformly-spaced 4-step t in RF: [1.0, 0.75, 0.5, 0.25] with a shift of 5
    selected_sampling_time: list[float] = [math.pi / 2, math.atan(15), math.atan(5), math.atan(5 / 3)]
    sigma_data: float = 1.0
    # timestep shift used to sample the noise level to the teacher/critic models. Should match that of the teacher model config.
    timestep_shift: float = 5

    # ---------------- Video conditioning / noise adjustments ----------------
    conditional_frames_probs: dict[int, float] | None = None  # Probability distribution for conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    min_num_conditional_frames: int = 0  # Minimum number of latent conditional frames
    multiply_noise_by_video_len: bool = True  # whether or not adjust video noise according to the video length
    replace_cond_output_with_gt: bool = (
        True  # Replace conditioning frames in denoise output with GT to avoid loss on them
    )
    sigma_conditional: float = 0.0001  # Noise level used for conditional frames
    use_clean_cond_timesteps: bool = True  # Set conditional-frame timesteps / noise levels to be very low

    # ---------------- Data keys ----------------
    input_caption_key: str = "ai_caption"  # Key used to fetch input captions
    input_data_key: str = "video"  # key to fetch input data from data_batch
    input_image_key: str = "images"  # key to fetch input image from data_batch

    # ---------------- Optional extra condition postprocessing ----------------
    # For Predict2.5 model, this is None. If the teacher model requires customized get_data_and_condition logic,
    # e.g. Transfer2 control inputs, action/camera conditions, etc., we can move those condition processing steps
    # into this custom condition_postprocessor.
    condition_postprocessor: LazyDict | None = None

    # ---------------- misc ----------------
    use_neg_prompt_str: bool = False
    neg_embed_path: str = ""
    neg_prompt_str: str = (
        "The video captures a game playing, with bad crappy graphics and cartoonish frames. "
        "It represents a recording of old outdated games. The lighting looks very fake. "
        "The textures are very raw and basic. The geometries are very primitive. "
        "The images are very pixelated and of poor CG quality. There are many subtitles in the footage. "
        "Overall, the video is unrealistic at all."
    )
