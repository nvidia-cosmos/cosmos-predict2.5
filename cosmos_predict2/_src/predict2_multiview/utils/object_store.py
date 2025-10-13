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

from __future__ import annotations

import io
import json
import pickle
from typing import Any, Callable

import numpy as np
import torch
import yaml
from PIL import Image

from cosmos_predict2._src.imaginaire.utils import object_store


class ObjectStore(object_store.ObjectStore):
    def _load_object(
        self, key: str, type: str | None = None, load_func: Callable | None = None, encoding: str = "UTF-8"
    ) -> Any:
        """
        Overrides parent function to add support for loading jsonl objects.
        """
        assert type is not None or load_func is not None, "Either type or load_func should be specified."
        with io.BytesIO() as buffer:
            self.client.download_fileobj(Bucket=self.bucket, Key=key, Fileobj=buffer)
            buffer.seek(0)
            # Read from buffer for common data types.
            if type == "torch":
                object = torch.load(buffer, map_location=lambda storage, loc: storage)
            elif type == "torch.jit":
                object = torch.jit.load(buffer)
            elif type == "image":
                object = Image.open(buffer)
                object.load()
            elif type == "json":
                object = json.load(buffer)
            elif type == "jsonl":
                data = []
                for line in buffer:
                    data.append(json.loads(line))
                object = {"data": data}
            elif type == "pickle":
                object = pickle.load(buffer)
            elif type == "yaml":
                object = yaml.safe_load(buffer)
            elif type == "text":
                object = buffer.read().decode(encoding)
            elif type == "numpy":
                object = np.load(buffer, allow_pickle=True)
            # Read from buffer as raw bytes.
            elif type == "bytes":
                object = buffer.read()
            # Customized load_func should be provided.
            else:
                object = load_func(buffer)
        return object
