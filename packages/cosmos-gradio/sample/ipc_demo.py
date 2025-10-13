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

import json

from cosmos_gradio.model_ipc.model_server import ModelServer

"""
simple demo for command ipc:

To build a custom gradio app, the model IPC can be used to broadcast inference command to workers
and collect status from all workers

"""


def test_command_ipc():
    with ModelServer(num_gpus=2, factory_module="sample.sample_worker", factory_function="create_worker") as server:
        result = server.infer({"prompt": "a cat"})
        print(json.dumps(result, indent=4))

        result = server.infer({"prompt": "a dog"})
        print(json.dumps(result, indent=4))


if __name__ == "__main__":
    test_command_ipc()
