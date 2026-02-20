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

"""
debug_deploy.py

Debug script to test the ALOHA deploy server by sending dummy observations.
Fully LLM-generated code.

Usage:
    uv run -m cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.aloha.debug_deploy \
        --policy_server_ip 127.0.0.1 \
        --num_queries 3

    # Test with different image sizes
    uv run -m cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.aloha.debug_deploy \
        --policy_server_ip 10.12.181.75 \
        --image_size 224 \
        --num_queries 1
"""

import json
import time
from dataclasses import dataclass

import draccus
import json_numpy
import numpy as np
import requests

json_numpy.patch()


@dataclass
class DebugConfig:
    # fmt: off
    policy_server_ip: str = "127.0.0.1"           # Policy server IP address
    policy_server_port: int = 8777                # Policy server port
    image_size: int = 480                         # Image size (height and width)
    num_queries: int = 1                          # Number of queries to send
    print_full_response: bool = False             # Whether to print full response (can be very long)
    # fmt: on


def create_dummy_observation(image_size: int):
    """Create a dummy observation with random images and proprio state."""
    # Create random images (RGB, uint8)
    primary_image = np.random.randint(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
    left_wrist_image = np.random.randint(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
    right_wrist_image = np.random.randint(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)

    # Create random proprio state (14-dim for ALOHA: 7 for left arm, 7 for right arm)
    proprio = np.random.randn(14).astype(np.float32)

    # Create observation dict
    observation = {
        "primary_image": primary_image.tolist(),
        "left_wrist_image": left_wrist_image.tolist(),
        "right_wrist_image": right_wrist_image.tolist(),
        "proprio": proprio.tolist(),
        "task_description": "fold shirt",
    }

    return observation


def send_observation_to_server(observation, server_endpoint: str):
    """Send observation to the policy server and get response."""
    try:
        start_time = time.time()
        response = requests.post(server_endpoint, json=observation, timeout=30)
        query_time = time.time() - start_time

        if response.status_code != 200:
            print(f"❌ Server returned status code: {response.status_code}")
            print(f"Response text: {response.text}")
            return None, query_time

        response_json = response.json()
        return response_json, query_time

    except requests.exceptions.Timeout:
        print("❌ Request timed out after 30 seconds")
        return None, None
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None, None


def print_response_metadata(response, query_time, query_idx):
    """Print metadata about the server response."""
    print(f"\n{'=' * 80}")
    print(f"QUERY {query_idx} - Response Metadata")
    print(f"{'=' * 80}")

    if response is None:
        print("❌ No response received")
        return

    print(f"✓ Query time: {query_time:.3f} seconds")
    print(f"✓ Response type: {type(response)}")

    if isinstance(response, dict):
        print(f"✓ Response keys: {list(response.keys())}")

        # Check for actions
        if "actions" in response:
            actions = response["actions"]
            if isinstance(actions, list):
                actions_array = np.array(actions)
                print("\n📊 Actions:")
                print("  - Type: list")
                print(f"  - Length: {len(actions)}")
                print(f"  - Shape (as array): {actions_array.shape}")
                print(f"  - Dtype (as array): {actions_array.dtype}")
                print(f"  - Min value: {actions_array.min():.4f}")
                print(f"  - Max value: {actions_array.max():.4f}")
                print(f"  - Mean value: {actions_array.mean():.4f}")
                print(f"  - First action: {actions[0]}")

        # Check for future image predictions
        if "future_image_predictions" in response:
            future_preds = response["future_image_predictions"]
            print("\n🖼️  Future Image Predictions:")
            if future_preds is None:
                print("  - None")
            elif isinstance(future_preds, dict):
                print("  - Type: dict")
                print(f"  - Keys: {list(future_preds.keys())}")
                for key, value in future_preds.items():
                    if value is not None:
                        if isinstance(value, (list, np.ndarray)):
                            value_array = np.array(value)
                            print(f"  - {key}:")
                            print(f"    - Shape: {value_array.shape}")
                            print(f"    - Dtype: {value_array.dtype}")
                            print(f"    - Min: {value_array.min()}, Max: {value_array.max()}")
                        else:
                            print(f"  - {key}: {type(value)}")
                    else:
                        print(f"  - {key}: None")

        # Check for value prediction
        if "value_prediction" in response:
            value_pred = response["value_prediction"]
            print("\n💎 Value Prediction:")
            print(f"  - Type: {type(value_pred)}")
            print(f"  - Value: {value_pred}")

        # Check for all_* fields (if return_all_query_results was True)
        all_keys = [k for k in response.keys() if k.startswith("all_")]
        if all_keys:
            print("\n📦 All Query Results (best-of-N):")
            for key in all_keys:
                value = response[key]
                if isinstance(value, list):
                    print(f"  - {key}: list with {len(value)} elements")
                else:
                    print(f"  - {key}: {type(value)}")

    elif isinstance(response, list):
        print("✓ Response is a list (actions only)")
        actions_array = np.array(response)
        print(f"  - Length: {len(response)}")
        print(f"  - Shape (as array): {actions_array.shape}")
        print(f"  - Dtype (as array): {actions_array.dtype}")
        print(f"  - Min value: {actions_array.min():.4f}")
        print(f"  - Max value: {actions_array.max():.4f}")
        print(f"  - Mean value: {actions_array.mean():.4f}")
        print(f"  - First action: {response[0]}")

    print(f"{'=' * 80}\n")


@draccus.wrap()
def debug_deploy(cfg: DebugConfig):
    """Main debug function."""
    print(f"\n{'=' * 80}")
    print("ALOHA Deploy Server Debug Script")
    print(f"{'=' * 80}")
    print(f"Server: http://{cfg.policy_server_ip}:{cfg.policy_server_port}/act")
    print(f"Image size: {cfg.image_size}x{cfg.image_size}")
    print(f"Number of queries: {cfg.num_queries}")
    print(f"{'=' * 80}\n")

    # Get server endpoint
    server_endpoint = f"http://{cfg.policy_server_ip}:{cfg.policy_server_port}/act"

    # Run queries
    total_query_time = 0.0
    successful_queries = 0

    for i in range(cfg.num_queries):
        print(f"\n📤 Sending query {i + 1}/{cfg.num_queries}...")

        # Create dummy observation
        observation = create_dummy_observation(cfg.image_size)
        print("  - Created dummy observation:")
        print(f"    - primary_image shape: {np.array(observation['primary_image']).shape}")
        print(f"    - left_wrist_image shape: {np.array(observation['left_wrist_image']).shape}")
        print(f"    - right_wrist_image shape: {np.array(observation['right_wrist_image']).shape}")
        print(f"    - proprio shape: {np.array(observation['proprio']).shape}")
        print(f"    - task_description: {observation['task_description']}")

        # Send to server
        response, query_time = send_observation_to_server(observation, server_endpoint)

        # Print metadata
        if response is not None:
            successful_queries += 1
            total_query_time += query_time
            print_response_metadata(response, query_time, i + 1)

            if cfg.print_full_response:
                print(f"\n📄 Full Response (Query {i + 1}):")
                print(json.dumps(response, indent=2, default=str))
                print(f"{'=' * 80}\n")
        else:
            print(f"❌ Query {i + 1} failed")

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total queries: {cfg.num_queries}")
    print(f"Successful queries: {successful_queries}")
    print(f"Failed queries: {cfg.num_queries - successful_queries}")
    if successful_queries > 0:
        print(f"Average query time: {total_query_time / successful_queries:.3f} seconds")
    print(f"{'=' * 80}\n")

    if successful_queries == cfg.num_queries:
        print("✅ All queries successful! Deploy server is working correctly.")
    elif successful_queries > 0:
        print("⚠️  Some queries failed. Check the error messages above.")
    else:
        print("❌ All queries failed. Deploy server may not be running or configured correctly.")


if __name__ == "__main__":
    debug_deploy()
