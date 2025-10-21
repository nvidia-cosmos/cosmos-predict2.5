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

import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from cosmos_predict2.multiview import MultiviewInference
from cosmos_predict2.multiview_config import MultiviewInferenceArgumentsWithInputPaths, MultiviewSetupArguments


class Multiview_Worker:
    def __init__(
        self,
        num_gpus,  # todo not sure if anything else is supported
    ):
        assert num_gpus == 8, "Multiview currently requires 8 GPUs"
        setup_args = MultiviewSetupArguments(model="2B/auto/multiview", output_dir=Path("outputs"), keep_going=True)
        self.pipe = MultiviewInference(setup_args)

    def infer(self, args: dict):
        output_dir = args.pop("output_dir", "outputs")
        p = MultiviewInferenceArgumentsWithInputPaths(**args)
        output_videos = self.pipe.generate([p], Path(output_dir))
        return {
            "videos": output_videos,
        }
