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

from cosmos_gradio.gradio_app.gradio_test_harness import TestHarness

from cosmos_predict2.gradio.sample_data import sample_request_image2world, sample_request_multiview

env_vars_video2world = {
    "MODEL_NAME": "video2world",
    "NUM_GPUS": "2",
}

env_vars_multiview = {
    "MODEL_NAME": "multiview",
    "NUM_GPUS": "8",
}

if __name__ == "__main__":
    TestHarness.test(
        server_module="cosmos_predict2.gradio.gradio_bootstrapper",
        env_vars=env_vars_video2world,
        sample_request=sample_request_image2world,
    )

    TestHarness.test(
        server_module="cosmos_predict2.gradio.gradio_bootstrapper",
        env_vars=env_vars_multiview,
        sample_request=sample_request_multiview,
    )
