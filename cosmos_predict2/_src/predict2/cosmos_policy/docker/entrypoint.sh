#!/usr/bin/env bash
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

set -euo pipefail

TARGET_USER="${HOST_USER_NAME:-cosmos}"
TARGET_UID="${HOST_USER_ID:-1000}"
TARGET_GID="${HOST_GROUP_ID:-1000}"
TARGET_HOME="/home/${TARGET_USER}"

ensure_group() {
    if getent group "${TARGET_GID}" >/dev/null; then
        existing_group="$(getent group "${TARGET_GID}" | cut -d: -f1)"
        if [[ "${existing_group}" != "${TARGET_USER}" ]]; then
            groupmod --new-name "${TARGET_USER}" "${existing_group}"
        fi
    elif getent group "${TARGET_USER}" >/dev/null; then
        groupmod --gid "${TARGET_GID}" "${TARGET_USER}"
    else
        groupadd --gid "${TARGET_GID}" "${TARGET_USER}"
    fi
}

ensure_user() {
    if id -u "${TARGET_USER}" >/dev/null 2>&1; then
        usermod --uid "${TARGET_UID}" --gid "${TARGET_GID}" --home "${TARGET_HOME}" --shell /bin/bash "${TARGET_USER}"
    elif getent passwd "${TARGET_UID}" >/dev/null; then
        existing_user="$(getent passwd "${TARGET_UID}" | cut -d: -f1)"
        usermod --login "${TARGET_USER}" --home "${TARGET_HOME}" --move-home --shell /bin/bash "${existing_user}"
        usermod --gid "${TARGET_GID}" "${TARGET_USER}"
    else
        useradd --uid "${TARGET_UID}" --gid "${TARGET_GID}" --home "${TARGET_HOME}" --create-home --shell /bin/bash "${TARGET_USER}"
    fi
}

prepare_home() {
    mkdir -p "${TARGET_HOME}"
    chown "${TARGET_UID}:${TARGET_GID}" "${TARGET_HOME}"
    export HOME="${TARGET_HOME}"
    export USER="${TARGET_USER}"
    export PATH="${TARGET_HOME}/.local/bin:${PATH}"
}

ensure_group
ensure_user
prepare_home

exec gosu "${TARGET_UID}:${TARGET_GID}" "$@"
