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
VQA Validation utilities.

This module provides validation utilities for VQA batch processing,
including import validation and directory/config resolution validation.
"""

import os
from pathlib import Path

from vqa.utils import log_info, log_verbose


class VQAValidator:
    """
    Validator for VQA batch processing.

    Provides methods to validate imports, directory resolution, and test config files
    before starting batch VQA evaluation.
    """

    @staticmethod
    def validate_imports() -> bool:
        """
        Validate that all VQA module imports work correctly.

        This function tests imports from evaluator.py and cosmos_reason_inference.py
        to catch any import errors early, even before processing videos.

        Returns:
            True if all imports succeed

        Raises:
            ImportError: If any required module cannot be imported
        """
        log_info("Validating VQA module imports...")

        try:
            # Test evaluator imports
            log_verbose("  ✓ Importing vqa.evaluator...")
            from vqa import evaluator  # noqa: F401

            # Test cosmos_reason_inference imports
            log_verbose("  ✓ Importing vqa.cosmos_reason_inference...")
            from vqa import cosmos_reason_inference  # noqa: F401

            # Test key dependencies
            log_verbose("  ✓ Importing yaml...")
            import yaml  # noqa: F401  # type: ignore[import-not-found]

            log_verbose("  ✓ Importing vllm...")
            import vllm  # noqa: F401  # type: ignore[import-not-found]

            log_verbose("  ✓ Importing qwen_vl_utils...")
            import qwen_vl_utils  # noqa: F401  # type: ignore[import-not-found]

            log_verbose("  ✓ Importing openai...")
            import openai  # noqa: F401  # type: ignore[import-not-found]

            log_info("✓ All VQA module imports validated successfully!\n")
            return True

        except ImportError as e:
            log_info(f"✗ Import validation failed: {e}")
            raise

    @staticmethod
    def validate_directory_resolution(
        directory: str | Path,
        package: str,
        video_test_config_map: dict[str, dict[str, str]],
        build_dir: str,
        base_dir: str | Path | None,
        skip_unmapped: bool = False,
    ) -> tuple[Path, Path]:
        """
        Validate and show directory resolution before processing.
        Also validates that test config YAML files exist.

        Args:
            directory: Video output directory path
            package: Package name
            video_test_config_map: Mapping of package -> video -> test config
            build_dir: Build directory path
            base_dir: Base directory for resolving relative paths
            skip_unmapped: Skip validation for videos without config mapping

        Returns:
            Tuple of (resolved_base_path, resolved_directory_path)

        Raises:
            FileNotFoundError: If directories or configs don't exist
            ValueError: If package not found in video_test_config_map
        """
        log_verbose("=" * 80)
        log_verbose("DIRECTORY & CONFIG RESOLUTION VALIDATION")
        log_verbose("=" * 80)

        # Validate package exists in VIDEO_TEST_CONFIG_MAP
        if package not in video_test_config_map:
            available_packages = ", ".join(video_test_config_map.keys())
            raise ValueError(
                f"Package '{package}' not found in VIDEO_TEST_CONFIG_MAP.\n  Available packages: {available_packages}"
            )

        package_config = video_test_config_map[package]
        log_verbose(f"✓ Package: {package} (has {len(package_config)} video config mappings)")

        # Resolve base directory
        if base_dir is not None:
            base_path = Path(base_dir).resolve()
            log_verbose(f"✓ Base directory (provided): {base_path}")
        else:
            base_path = Path.cwd()
            log_verbose(f"✓ Base directory (CWD): {base_path}")

        log_verbose(f"  Python CWD: {os.getcwd()}")
        log_verbose(f"  Base directory exists: {base_path.exists()}")

        # Resolve video directory
        dir_path = Path(directory)
        log_verbose(f"\n✓ Video directory (raw input): {dir_path}")
        log_verbose(f"  Is absolute: {dir_path.is_absolute()}")

        if dir_path.is_absolute():
            resolved_dir = dir_path
        else:
            resolved_dir = base_path / dir_path

        log_verbose(f"  Resolved to: {resolved_dir}")
        log_verbose(f"  Directory exists: {resolved_dir.exists()}")

        if not resolved_dir.exists():
            raise FileNotFoundError(
                f"Video directory not found: {resolved_dir}\n"
                f"  Raw input: {directory}\n"
                f"  Base path: {base_path}\n"
                f"  Please check that tests have generated videos in the expected location."
            )

        # Count and list MP4 files
        try:
            mp4_files = list(resolved_dir.rglob("*.mp4"))
            mp4_count = len(mp4_files)
            log_verbose(f"  MP4 files found: {mp4_count}")

            if mp4_count > 0:
                log_verbose(f"  Video files:")
                for mp4 in sorted(mp4_files)[:10]:  # Show first 10
                    log_verbose(f"    - {mp4.name}")
                if mp4_count > 10:
                    log_verbose(f"    ... and {mp4_count - 10} more")
        except Exception as e:
            log_info(f"  MP4 files found: (unable to count: {e})")
            mp4_files = []

        # Validate test config files
        log_info(f"\n✓ Test config validation:")

        missing_configs = []
        found_configs = []
        unmapped_videos = []

        for mp4 in mp4_files:
            video_filename = mp4.name

            if video_filename not in package_config:
                if not skip_unmapped:
                    unmapped_videos.append(video_filename)
                continue

            # Resolve test config path
            test_config_rel = package_config[video_filename]
            test_config_path = Path(build_dir) / package / test_config_rel

            if not test_config_path.is_absolute():
                test_config_resolved = base_path / test_config_path
            else:
                test_config_resolved = test_config_path

            if test_config_resolved.exists():
                found_configs.append((video_filename, test_config_resolved))
            else:
                missing_configs.append((video_filename, test_config_resolved))

        # Report results
        if found_configs:
            log_verbose(f"  ✓ Found {len(found_configs)} test config(s):")
            for video, config in found_configs[:5]:  # Show first 5
                log_verbose(f"    ✓ {video}")
                log_verbose(f"      → {config}")
            if len(found_configs) > 5:
                log_verbose(f"    ... and {len(found_configs) - 5} more")

        if unmapped_videos:
            log_verbose(f"  ⚠ {len(unmapped_videos)} unmapped video(s) (no config in VIDEO_TEST_CONFIG_MAP):")
            for video in unmapped_videos[:5]:
                log_verbose(f"    ⚠ {video}")
            if len(unmapped_videos) > 5:
                log_verbose(f"    ... and {len(unmapped_videos) - 5} more")
            if not skip_unmapped:
                raise ValueError(
                    f"Found {len(unmapped_videos)} unmapped videos without test configs.\n"
                    f"  Use --skip-unmapped flag to skip these videos, or add them to VIDEO_TEST_CONFIG_MAP."
                )

        if missing_configs and not skip_unmapped:
            log_info(f"  ✗ {len(missing_configs)} MISSING test config(s):")
            for video, config in missing_configs:
                log_info(f"    ✗ {video}")
                log_info(f"      → {config} (NOT FOUND)")
            raise FileNotFoundError(
                f"Missing {len(missing_configs)} test config file(s).\n"
                f"  Please check that test configs are in the correct location."
            )

        log_verbose("=" * 80)
        log_verbose("")

        return base_path, resolved_dir
