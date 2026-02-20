# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""
Main configuration router for Cosmos Policy.

Currently provides access to the v2 configuration (for Cosmos-Predict2).
v1 config is deprecated. The version is determined by the COSMOS_VERSION environment variable.
If not set, we default to v2.
"""

import os

from cosmos_predict2._src.predict2.cosmos_policy.config.config_v2 import ConfigV2, make_config_v2

config_version = os.environ.get("COSMOS_VERSION", "v2").lower()

if config_version == "v2":
    Config = ConfigV2
    make_config = make_config_v2
else:
    raise ValueError(f"Invalid COSMOS_VERSION: {config_version}. Not implemented yet!")

# Export all the classes and functions
__all__ = [
    "Config",
    "ConfigV2",
    "make_config",
    "make_config_v2",
]
