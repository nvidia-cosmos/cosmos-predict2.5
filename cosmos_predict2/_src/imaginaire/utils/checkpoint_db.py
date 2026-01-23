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
from typing import Annotated, TypeAlias

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


CheckpointS3: TypeAlias = CheckpointFileS3 | CheckpointDirS3


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


CheckpointHf: TypeAlias = CheckpointFileHf | CheckpointDirHf


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
    """Hydra experiment name."""
    config_file: str | None = None
    """Hydra config file."""

    s3: CheckpointS3 | None = None
    """Config for checkpoint on S3."""
    hf: CheckpointHf
    """Config for checkpoint on Hugging Face."""

    @cached_property
    def full_name(self) -> str:
        return f"{self.name}({self.uuid})"

    @cached_property
    def path(self) -> str:
        """Return S3 URI or local path."""
        if INTERNAL and self.s3 is not None:
            return self.s3.uri
        log.info(f"Downloading checkpoint {self.full_name}")
        return self.hf.path

    @classmethod
    def from_uuid(cls, uuid: str):
        return get_checkpoint_by_uuid(uuid)

    @classmethod
    def from_s3(cls, uri: str):
        return get_checkpoint_by_s3(uri)

    def register(self):
        register_checkpoint(self)


_CHECKPOINTS_BY_UUID: dict[str, CheckpointConfig] = {}
_CHECKPOINTS_BY_S3: dict[str, CheckpointConfig] = {}


def register_checkpoint(checkpoint_config: CheckpointConfig):
    if not EXPERIMENTAL_CHECKPOINTS:
        if checkpoint_config.hf.repository in ["nvidia/Cosmos-Experimental"]:
            # Don't register experimental checkpoints. An exception will be
            # raised in CI if the checkpoint is used without
            # EXPERIMENTAL_CHECKPOINTS.
            return
    if checkpoint_config.uuid in _CHECKPOINTS_BY_UUID:
        raise ValueError(f"Checkpoint UUID {checkpoint_config.uuid} already registered.")
    _CHECKPOINTS_BY_UUID[checkpoint_config.uuid] = checkpoint_config
    if checkpoint_config.s3 is not None:
        uri = checkpoint_config.s3.uri
        if uri in _CHECKPOINTS_BY_S3:
            raise ValueError(f"Checkpoint S3 {uri} already registered.")
        _CHECKPOINTS_BY_S3[uri] = checkpoint_config


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
def get_checkpoint_by_hf(checkpoint_hf: str) -> str:
    """Download checkpoint from HuggingFace and return local path."""
    # Parse hf://org/repo/path/to/file.pth
    assert checkpoint_hf.startswith("hf://"), f"Not a HuggingFace URI: {checkpoint_hf}"
    hf_path = checkpoint_hf[5:]  # Remove "hf://" prefix
    # Split into repo_id (org/repo) and filename (path/to/file.pth)
    parts = hf_path.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid HuggingFace URI format: {checkpoint_hf}. Expected format: hf://org/repo/path/to/file.pth"
        )
    repo_id = "/".join(parts[:2])  # org/repo
    filename = "/".join(parts[2:])  # path/to/file.pth
    log.info(f"Downloading checkpoint from HuggingFace: {repo_id}/{filename}")
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=filename,
    )
    assert os.path.exists(path), path
    return path


@functools.lru_cache
def get_checkpoint_path(checkpoint_uri: str) -> str:
    """Return checkpoint path for S3 URI, HuggingFace URI, or local path.

    Supports:
    - S3 URIs: s3://bucket/path/to/checkpoint
    - HuggingFace URIs: hf://org/repo/path/to/file.pth
    - Local paths: /path/to/checkpoint
    """
    if INTERNAL:
        return checkpoint_uri
    checkpoint_uri = checkpoint_uri.rstrip("/")
    if checkpoint_uri.startswith("s3://"):
        return get_checkpoint_by_s3(checkpoint_uri).path
    if checkpoint_uri.startswith("hf://"):
        return get_checkpoint_by_hf(checkpoint_uri)
    if not os.path.exists(checkpoint_uri):
        raise ValueError(f"Checkpoint path {checkpoint_uri} does not exist.")
    return checkpoint_uri
