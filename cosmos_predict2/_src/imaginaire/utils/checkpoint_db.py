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

"""Database of released checkpoints."""

import functools
import os
from functools import cached_property
from typing import Annotated

import pydantic
from huggingface_hub import hf_hub_download, snapshot_download
from typing_extensions import override

from cosmos_predict2._src.imaginaire.flags import EXPERIMENTAL_CHECKPOINTS, INTERNAL
from cosmos_predict2._src.imaginaire.utils import log


class _CheckpointUri(pydantic.BaseModel):
    """Config for checkpoint file/directory."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    metadata: dict = pydantic.Field(default_factory=dict)
    """File metadata.

    Only used for debugging.
    """

    def _download(self) -> str:
        raise NotImplementedError("Download method not implemented.")

    @cached_property
    def path(self) -> str:
        """Return S3 URI or local path."""
        return self._download()


def is_s3_uri(uri: str) -> str:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}. Must start with 's3://'")
    return uri.rstrip("/")


S3Uri = Annotated[str, pydantic.AfterValidator(is_s3_uri)]


class _CheckpointS3(_CheckpointUri):
    """Config for checkpoint on S3."""

    uri: S3Uri
    """S3 URI."""


class CheckpointFileS3(_CheckpointS3):
    """Config for checkpoint file on S3."""


class CheckpointDirS3(_CheckpointS3):
    """Config for checkpoint directory on S3."""


class _CheckpointHf(_CheckpointUri):
    """Config for checkpoint on Hugging Face."""

    repository: str
    """Repository id (organization/repository)."""
    revision: str
    """Git revision id which can be a branch name, a tag, or a commit hash."""


class CheckpointFileHf(_CheckpointHf):
    """Config for checkpoint file on Hugging Face."""

    filename: str
    """File name."""

    @override
    def _download(self) -> str:
        """Download checkpoint and return the local path."""
        download_kwargs = dict(
            repo_id=self.repository, repo_type="model", revision=self.revision, filename=self.filename
        )
        log.info(f"Downloading checkpoint file from Hugging Face with {download_kwargs}")
        path = hf_hub_download(**download_kwargs)
        assert os.path.exists(path), path
        return path


class CheckpointDirHf(_CheckpointHf):
    """Config for checkpoint directory on Hugging Face."""

    subdirectory: str = ""
    """Repository subdirectory."""
    include: tuple[str, ...] = ()
    """Include patterns.

    See https://huggingface.co/docs/huggingface_hub/en/guides/download#filter-files-to-download
    """
    exclude: tuple[str, ...] = ()
    """Exclude patterns.

    See https://huggingface.co/docs/huggingface_hub/en/guides/download#filter-files-to-download
    """

    @override
    def _download(self) -> str:
        """Download checkpoint and return the local path."""
        patterns: dict[str, list[str]] = {}
        if self.include:
            patterns["allow_patterns"] = list(self.include)
        else:
            patterns["allow_patterns"] = ["*"]
        if self.exclude:
            patterns["ignore_patterns"] = list(self.exclude)
        if self.subdirectory:
            patterns = {key: [os.path.join(self.subdirectory, x) for x in val] for key, val in patterns.items()}
        download_kwargs = dict(repo_id=self.repository, repo_type="model", revision=self.revision) | patterns
        log.info(f"Downloading checkpoint from Hugging Face with {download_kwargs}")
        path = snapshot_download(**download_kwargs)
        if self.subdirectory:
            path = os.path.join(path, self.subdirectory)
        assert os.path.exists(path), path
        return path


class CheckpointConfig(pydantic.BaseModel):
    """Config for checkpoint."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    uuid: str
    """Checkpoint UUID."""
    name: str
    """Checkpoint name.

    Only used for debugging.
    """
    metadata: dict = pydantic.Field(default_factory=dict)
    """Checkpoint metadata.

    Only used for debugging.
    """
    experiment: str | None = None
    """Experiment name."""

    s3: CheckpointFileS3 | CheckpointDirS3 | None = None
    """Config for checkpoint on S3."""
    hf: CheckpointFileHf | CheckpointDirHf | None = None
    """Config for checkpoint on Hugging Face."""

    @cached_property
    def path(self) -> str:
        """Return S3 URI or local path."""
        if INTERNAL and self.s3 is not None:
            return self.s3.uri
        if self.hf is None:
            raise ValueError(f"Checkpoint {self.name}({self.uuid}) is not available on Hugging Face.")
        log.info(f"Downloading checkpoint {self.name}({self.uuid})")
        return self.hf.path


_CHECKPOINTS_BY_UUID: dict[str, CheckpointConfig] = {}
_CHECKPOINTS_BY_S3: dict[str, CheckpointConfig] = {}


def _register_checkpoint(checkpoint_config: CheckpointConfig):
    if checkpoint_config.uuid in _CHECKPOINTS_BY_UUID:
        raise ValueError(f"Checkpoint UUID {checkpoint_config.uuid} already registered.")
    _CHECKPOINTS_BY_UUID[checkpoint_config.uuid] = checkpoint_config
    if checkpoint_config.s3 is not None:
        uri = checkpoint_config.s3.uri
        if uri in _CHECKPOINTS_BY_S3:
            raise ValueError(f"Checkpoint S3 {uri} already registered.")
        _CHECKPOINTS_BY_S3[uri] = checkpoint_config


_register_checkpoint(
    CheckpointConfig(
        uuid="4dbf13c6-1d30-4b02-99d6-75780dd8b744",
        name="google-t5/t5-11b",
        hf=CheckpointDirHf(
            repository="google-t5/t5-11b",
            revision="90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3",
            exclude=("tf_model.h5",),
        ),
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="a2944743-cf8d-427e-a6fc-b3c03d807064",
        name="meta-llama/Llama-Guard-3-8B",
        hf=CheckpointDirHf(
            repository="meta-llama/Llama-Guard-3-8B",
            revision="7327bd9f6efbbe6101dc6cc4736302b3cbb6e425",
            exclude=("original/*",),
        ),
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="9c7b7da4-2d95-45bb-9cb8-2eed954e9736",
        name="nvidia/Cosmos-Guardrail1",
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos-Guardrail1",
            revision="d6d4bfa899a71454a700907664f3e88f503950cf",
        ),
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="7219c6c7-f878-4137-bbdb-76842ea85e70",
        name="Qwen/Qwen2.5-VL-7B-Instruct",
        s3=CheckpointDirS3(
            uri="s3://bucket/cosmos_reasoning1/pretrained/Qwen_tokenizer/Qwen/Qwen2.5-VL-7B-Instruct",
        ),
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos-Experimental",
            revision="736a20b6cfbc38e42ba3f7e7d8efa1d886c20db1",
            subdirectory="7219c6c7-f878-4137-bbdb-76842ea85e70",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else CheckpointDirHf(
            repository="nvidia/Cosmos-Reason1-7B",
            revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
        ),
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="685afcaa-4de2-42fe-b7b9-69f7a2dee4d8",
        name="Wan2.1/vae",
        s3=CheckpointFileS3(
            uri="s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/Wan2.1_VAE.pth",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="736a20b6cfbc38e42ba3f7e7d8efa1d886c20db1",
            filename="685afcaa-4de2-42fe-b7b9-69f7a2dee4d8.pth",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else CheckpointFileHf(
            repository="nvidia/Cosmos-Predict2.5-2B",
            revision="6787e176dce74a101d922174a95dba29fa5f0c55",
            filename="tokenizer.pth",
        ),
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="cb3e3ffa-7b08-4c34-822d-61c7aa31a14f",
        name="nvidia/Cosmos-Reason1.1-7B",
        s3=CheckpointDirS3(
            uri="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="736a20b6cfbc38e42ba3f7e7d8efa1d886c20db1",
            filename="cb3e3ffa-7b08-4c34-822d-61c7aa31a14f/model.pt",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else CheckpointDirHf(
            repository="nvidia/Cosmos-Reason1-7B",
            revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
        ),
    ),
)

# -----------------------------------------------------------------------------
# Cosmos-Predict2.5-2B
# -----------------------------------------------------------------------------
_register_checkpoint(
    CheckpointConfig(
        uuid="d20b7120-df3e-4911-919d-db6e08bad31c",
        name="nvidia/Cosmos-Predict2.5-2B/base/pre-trained",
        experiment="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2",
        metadata={
            "resolution": "720p",
            "fps": 16,
        },
        s3=CheckpointDirS3(
            uri="s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2/checkpoints/iter_000023000/model",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
            filename="d20b7120-df3e-4911-919d-db6e08bad31c/model_ema_bf16.pt",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else CheckpointFileHf(
            repository="nvidia/Cosmos-Predict2.5-2B",
            revision="15a82a2ec231bc318692aa0456a36537c806e7d4",
            filename="base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",
        ),
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="81edfebe-bd6a-4039-8c1d-737df1a790bf",
        name="nvidia/Cosmos-Predict2.5-2B/base/post-trained",
        experiment="Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt",
        metadata={
            "resolution": "720p",
            "fps": 16,
        },
        s3=CheckpointFileS3(
            uri="s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_GRPO-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-posttrain_data-HQ_V7_RF_MERGE_LOCAL_ag_every2_guidance0_scorekeyoverall_reward_databeta0.01_mincon0/checkpoints/iter_000000288/model",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
            filename="81edfebe-bd6a-4039-8c1d-737df1a790bf/model_ema_bf16.pt",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else CheckpointFileHf(
            repository="nvidia/Cosmos-Predict2.5-2B",
            revision="15a82a2ec231bc318692aa0456a36537c806e7d4",
            filename="base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
        ),
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="6b9d7548-33bb-4517-b5e8-60caf47edba7",
        name="nvidia/Cosmos-Predict2.5-2B/auto/multiview",
        experiment="buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames_nofps",
        metadata={
            "resolution": "720p",
            "fps": 30,
            "views": 7,
            "frames": 29,
        },
        s3=CheckpointDirS3(
            uri="s3://bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames_nofps-0/checkpoints/iter_000005000/model",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
            filename="6b9d7548-33bb-4517-b5e8-60caf47edba7/model_ema_bf16.pt",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else CheckpointFileHf(
            repository="nvidia/Cosmos-Predict2.5-2B",
            revision="15a82a2ec231bc318692aa0456a36537c806e7d4",
            filename="auto/multiview/6b9d7548-33bb-4517-b5e8-60caf47edba7_ema_bf16.pt",
        ),
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="0e8177cc-0db5-4cfd-a8a4-b820c772f4fc",
        name="nvidia/Cosmos-Predict2.5-2B/robot/multiview",
        experiment="multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_multicam_syncam",
        metadata={
            "resolution": "720p",
            "fps": 16,
        },
        s3=CheckpointDirS3(
            uri="s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_multicam_syncam/checkpoints/iter_000002000/model",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
            filename="0e8177cc-0db5-4cfd-a8a4-b820c772f4fc/model_ema_bf16.pt",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else None,
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="7f6b99b7-7fac-4e74-8dbe-a394cb56ef99",
        name="nvidia/Cosmos-Predict2.5-2B/robot/multiview-agibot",
        experiment="multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_agibot",
        metadata={
            "resolution": "720p",
            "fps": 16,
        },
        s3=CheckpointDirS3(
            uri="s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_agibot/checkpoints/iter_000003000/model",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
            filename="7f6b99b7-7fac-4e74-8dbe-a394cb56ef99/model_ema_bf16.pt",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else None,
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="38c6c645-7d41-4560-8eeb-6f4ddc0e6574",
        name="nvidia/Cosmos-Predict2.5-2B/robot/action-cond",
        experiment="cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320",
        metadata={
            "resolution": "360p",
            "fps": 4,
        },
        s3=CheckpointDirS3(
            uri="s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320/checkpoints/iter_000016000/model",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="main",
            filename="38c6c645-7d41-4560-8eeb-6f4ddc0e6574/model_ema_bf16.pt",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else CheckpointFileHf(
            repository="nvidia/Cosmos-Predict2.5-2B",
            revision="main",
            filename="robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt",
        ),
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="24a3b7b8-6a3d-432d-b7d1-5d30b9229465",
        name="nvidia/Cosmos-Predict2.5-2B/transfer2.5",
        experiment="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only",
        metadata={
            "resolution": "720p",
            "fps": 16,
        },
        s3=CheckpointDirS3(
            uri="s3://bucket/cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only/checkpoints/iter_000037000/model",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
            filename="24a3b7b8-6a3d-432d-b7d1-5d30b9229465/model_ema_bf16.pt",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else None,
    ),
)

# -----------------------------------------------------------------------------
# Cosmos-Predict2.5-14B
# -----------------------------------------------------------------------------
_register_checkpoint(
    CheckpointConfig(
        uuid="54937b8c-29de-4f04-862c-e67b04ec41e8",
        name="nvidia/Cosmos-Predict2.5-14B/base/pre-trained",
        experiment="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma",
        metadata={
            "size": "14B",
            "resolution": "720p",
            "fps": 16,
        },
        s3=CheckpointFileS3(
            uri="s3://bucket/cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma/checkpoints/iter_000012500/model",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
            filename="54937b8c-29de-4f04-862c-e67b04ec41e8/model_ema_bf16.pt",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else CheckpointFileHf(
            repository="nvidia/Cosmos-Predict2.5-14B",
            revision="03eb354f35eae0d6e0c1be3c9f94d8551e125570",
            filename="base/pre-trained/54937b8c-29de-4f04-862c-e67b04ec41e8_ema_bf16.pt",
        ),
    ),
)

# -----------------------------------------------------------------------------
# Cosmos-Transfer2.5-2B
# -----------------------------------------------------------------------------
_register_checkpoint(
    CheckpointConfig(
        uuid="ecd0ba00-d598-4f94-aa09-e8627899c431",
        name="nvidia/Cosmos-Transfer2.5-2B/general/edge",
        experiment="edge_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow_mock_data",
        metadata={
            "resolution": "720p",
            "fps": 16,
        },
        s3=CheckpointDirS3(
            uri="s3://bucket/cosmos_transfer2/vid2vid_2B_control/edge_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow/checkpoints/iter_000029000/model",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
            filename="ecd0ba00-d598-4f94-aa09-e8627899c431/model_ema_bf16.pt",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else CheckpointFileHf(
            repository="nvidia/Cosmos-Transfer2.5-2B",
            revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
            filename="general/edge/ecd0ba00-d598-4f94-aa09-e8627899c431_ema_bf16.pt",
        ),
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab",
        name="nvidia/Cosmos-Transfer2.5-2B/general/seg",
        experiment="seg_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p2_20250823_64N_rectified_flow",
        metadata={
            "resolution": "720p",
            "fps": 16,
        },
        s3=CheckpointDirS3(
            uri="s3://bucket/cosmos_transfer2/vid2vid_2B_control/seg_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p2_20250823_64N_rectified_flow/checkpoints/iter_000031000/model",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
            filename="fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab/model_ema_bf16.pt",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else CheckpointFileHf(
            repository="nvidia/Cosmos-Transfer2.5-2B",
            revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
            filename="general/seg/fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab_ema_bf16.pt",
        ),
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1",
        name="nvidia/Cosmos-Transfer2.5-2B/general/blur",
        experiment="vis_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow",
        metadata={
            "resolution": "720p",
            "fps": 16,
        },
        s3=CheckpointDirS3(
            uri="s3://bucket/cosmos_transfer2/vid2vid_2B_control/vis_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N_rectified_flow/checkpoints/iter_000043000/model",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
            filename="20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1/model_ema_bf16.pt",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else CheckpointFileHf(
            repository="nvidia/Cosmos-Transfer2.5-2B",
            revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
            filename="general/blur/20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1_ema_bf16.pt",
        ),
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="0f214f66-ae98-43cf-ab25-d65d09a7e68f",
        name="nvidia/Cosmos-Transfer2.5-2B/general/depth",
        experiment="depth_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p1_20250823_64N_rectified_flow",
        metadata={
            "resolution": "720p",
            "fps": 16,
        },
        s3=CheckpointDirS3(
            uri="s3://bucket/cosmos_transfer2/vid2vid_2B_control/depth_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv4p1_20250823_64N_rectified_flow/checkpoints/iter_000028000/model",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
            filename="0f214f66-ae98-43cf-ab25-d65d09a7e68f/model_ema_bf16.pt",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else CheckpointFileHf(
            repository="nvidia/Cosmos-Transfer2.5-2B",
            revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
            filename="general/depth/0f214f66-ae98-43cf-ab25-d65d09a7e68f_ema_bf16.pt",
        ),
    ),
)

_register_checkpoint(
    CheckpointConfig(
        uuid="b5ab002d-a120-4fbf-a7f9-04af8615710b",
        name="nvidia/Cosmos-Transfer2.5-2B/auto/multiview",
        experiment="buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase5knofps_mads720pmulticaps29frames_world_scenario_resumefrom21k",
        metadata={
            "resolution": "720p",
            "fps": 16,
            "views": 7,
            "frames": 29,
        },
        s3=CheckpointDirS3(
            uri="s3://bucket/cosmos_transfer2_multiview/cosmos2_mv/buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase5knofps_mads720pmulticaps29frames_world_scenario_resumefrom21k-0/checkpoints/iter_000010000/model",
        ),
        hf=CheckpointFileHf(
            repository="nvidia/Cosmos-Experimental",
            revision="9a02ed8daa8c6c7718ac09da06488bfd1d363cb6",
            filename="b5ab002d-a120-4fbf-a7f9-04af8615710b/model_ema_bf16.pt",
        )
        if EXPERIMENTAL_CHECKPOINTS
        else CheckpointFileHf(
            repository="nvidia/Cosmos-Transfer2.5-2B",
            revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
            filename="auto/multiview/b5ab002d-a120-4fbf-a7f9-04af8615710b_ema_bf16.pt",
        ),
    ),
)


def get_checkpoint_by_uuid(checkpoint_uuid: str) -> CheckpointConfig:
    """Return checkpoint config for UUID."""
    if checkpoint_uuid not in _CHECKPOINTS_BY_UUID:
        raise ValueError(f"Checkpoint UUID {checkpoint_uuid} not found.")
    return _CHECKPOINTS_BY_UUID[checkpoint_uuid]


def get_checkpoint_by_s3(checkpoint_s3: str) -> CheckpointConfig:
    """Return checkpoint config for S3 URI."""
    checkpoint_s3 = checkpoint_s3.rstrip("/")
    if checkpoint_s3 not in _CHECKPOINTS_BY_S3:
        raise ValueError(f"Checkpoint S3 {checkpoint_s3} not found.")
    return _CHECKPOINTS_BY_S3[checkpoint_s3]


@functools.lru_cache
def get_checkpoint_path(checkpoint_uri: str) -> str:
    """Return checkpoint path for S3 URI or local path."""
    if INTERNAL:
        return checkpoint_uri
    checkpoint_uri = checkpoint_uri.rstrip("/")
    if checkpoint_uri.startswith("s3://"):
        return get_checkpoint_by_s3(checkpoint_uri).path
    if not os.path.exists(checkpoint_uri):
        raise ValueError(f"Checkpoint path {checkpoint_uri} does not exist.")
    return checkpoint_uri
