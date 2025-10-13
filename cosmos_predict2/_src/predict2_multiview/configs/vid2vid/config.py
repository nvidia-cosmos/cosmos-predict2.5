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

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils.config_helper import import_all_modules_from_package
from cosmos_predict2._src.predict2.configs.video2world.config import make_config as vid2vid_make_config
from cosmos_predict2._src.predict2_multiview.configs.vid2vid.defaults.callbacks import (
    register_callbacks as register_callbacks_for_backward_compatibility,
)
from cosmos_predict2._src.predict2_multiview.configs.vid2vid.defaults.conditioner import register_conditioner
from cosmos_predict2._src.predict2_multiview.configs.vid2vid.defaults.model import register_model
from cosmos_predict2._src.predict2_multiview.configs.vid2vid.defaults.net import register_net


def make_config():
    c = vid2vid_make_config()
    c.job.project = "cosmos_predict2_multiview"
    register_conditioner()
    register_model()
    register_net()
    from cosmos_predict2._src.predict2_multiview.configs.vid2vid.defaults.local_dataloader import (
        register_waymo_dataloader,
    )

    register_waymo_dataloader()

    register_callbacks_for_backward_compatibility()

    import_all_modules_from_package("cosmos_predict2._src.predict2_multiview.configs.vid2vid.experiment", reload=True)
    if not INTERNAL:
        import_all_modules_from_package("cosmos_predict2.experiments.multiview", reload=True)
    return c
