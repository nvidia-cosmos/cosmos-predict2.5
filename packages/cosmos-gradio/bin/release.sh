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

# Release a new version

if [ $# -lt 1 ]; then
    echo "Usage: $0 <pypi_token>"
    exit 1
fi
PYPI_TOKEN="$1"
shift

if [[ $(git status --porcelain) ]]; then
  echo "There are uncommitted changes. Please commit or stash them before proceeding."
  exit 1
fi

# Bump the version and tag the release
PACKAGE_VERSION=$(uv version --bump patch --short)
git add .
git commit -m "v$PACKAGE_VERSION"
git tag "v$PACKAGE_VERSION"

# Publish to PyPI
rm -rf dist
uv build
uv publish --token "$PYPI_TOKEN" "$@"
