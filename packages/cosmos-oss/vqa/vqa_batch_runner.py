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
VQA Batch Runner - Orchestrates batch VQA evaluation across multiple videos.

This script finds all MP4 files in a directory and runs VQA evaluation on each
using the vqa_evaluator.py script. It uses a static mapping (VIDEO_TEST_CONFIG_MAP)
to associate video filenames with their corresponding test configuration files.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add the parent directory of 'vqa' to Python path to allow absolute imports
vqa_dir = Path(__file__).parent.resolve()  # .../packages/cosmos-oss/vqa
parent_dir = vqa_dir.parent.resolve()  # .../packages/cosmos-oss

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


def validate_vqa_imports() -> bool:
    """
    Validate that all VQA module imports work correctly.

    This function tests imports from vqa_evaluator.py and cosmos_reason_inference.py
    to catch any import errors early, even before processing videos.

    Returns:
        True if all imports succeed

    Raises:
        ImportError: If any required module cannot be imported
    """
    print("Validating VQA module imports...")

    try:
        # Test vqa_evaluator imports
        print("  ✓ Importing vqa.vqa_evaluator...")
        from vqa import vqa_evaluator  # noqa: F401

        # Test cosmos_reason_inference imports
        print("  ✓ Importing vqa.cosmos_reason_inference...")
        from vqa import cosmos_reason_inference  # noqa: F401

        # Test key dependencies
        print("  ✓ Importing yaml...")
        import yaml  # noqa: F401  # type: ignore[import-not-found]

        print("  ✓ Importing vllm...")
        import vllm  # noqa: F401  # type: ignore[import-not-found]

        print("  ✓ Importing qwen_vl_utils...")
        import qwen_vl_utils  # noqa: F401  # type: ignore[import-not-found]

        print("  ✓ Importing openai...")
        import openai  # noqa: F401  # type: ignore[import-not-found]

        print("✓ All VQA module imports validated successfully!\n")
        return True

    except ImportError as e:
        print(f"✗ Import validation failed: {e}")
        raise


def validate_directory_resolution(
    directory: str | Path,
    package: str,
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
        build_dir: Build directory path
        base_dir: Base directory for resolving relative paths
        skip_unmapped: Skip validation for videos without config mapping

    Returns:
        Tuple of (resolved_base_path, resolved_directory_path)

    Raises:
        FileNotFoundError: If directories or configs don't exist
        ValueError: If package not found in VIDEO_TEST_CONFIG_MAP
    """
    import os

    print("=" * 80)
    print("DIRECTORY & CONFIG RESOLUTION VALIDATION")
    print("=" * 80)

    # Validate package exists in VIDEO_TEST_CONFIG_MAP
    if package not in VIDEO_TEST_CONFIG_MAP:
        available_packages = ", ".join(VIDEO_TEST_CONFIG_MAP.keys())
        raise ValueError(
            f"Package '{package}' not found in VIDEO_TEST_CONFIG_MAP.\n  Available packages: {available_packages}"
        )

    package_config = VIDEO_TEST_CONFIG_MAP[package]
    print(f"✓ Package: {package} (has {len(package_config)} video config mappings)")

    # Resolve base directory
    if base_dir is not None:
        base_path = Path(base_dir).resolve()
        print(f"✓ Base directory (provided): {base_path}")
    else:
        base_path = Path.cwd()
        print(f"✓ Base directory (CWD): {base_path}")

    print(f"  Python CWD: {os.getcwd()}")
    print(f"  Base directory exists: {base_path.exists()}")

    # Resolve video directory
    dir_path = Path(directory)
    print(f"\n✓ Video directory (raw input): {dir_path}")
    print(f"  Is absolute: {dir_path.is_absolute()}")

    if dir_path.is_absolute():
        resolved_dir = dir_path
    else:
        resolved_dir = base_path / dir_path

    print(f"  Resolved to: {resolved_dir}")
    print(f"  Directory exists: {resolved_dir.exists()}")

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
        print(f"  MP4 files found: {mp4_count}")

        if mp4_count > 0:
            print(f"  Video files:")
            for mp4 in sorted(mp4_files)[:10]:  # Show first 10
                print(f"    - {mp4.name}")
            if mp4_count > 10:
                print(f"    ... and {mp4_count - 10} more")
    except Exception as e:
        print(f"  MP4 files found: (unable to count: {e})")
        mp4_files = []

    # Validate test config files
    print(f"\n✓ Test config validation:")
    print(f"  Build directory: {build_dir}")
    print(f"  Skip unmapped: {skip_unmapped}")

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
        test_config_path = Path(f"{build_dir}/{package}/{test_config_rel}")

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
        print(f"  ✓ Found {len(found_configs)} test config(s):")
        for video, config in found_configs[:5]:  # Show first 5
            print(f"    ✓ {video}")
            print(f"      → {config}")
        if len(found_configs) > 5:
            print(f"    ... and {len(found_configs) - 5} more")

    if unmapped_videos:
        print(f"  ⚠ {len(unmapped_videos)} unmapped video(s) (no config in VIDEO_TEST_CONFIG_MAP):")
        for video in unmapped_videos[:5]:
            print(f"    ⚠ {video}")
        if len(unmapped_videos) > 5:
            print(f"    ... and {len(unmapped_videos) - 5} more")
        if not skip_unmapped:
            raise ValueError(
                f"Found {len(unmapped_videos)} unmapped videos without test configs.\n"
                f"  Use --skip-unmapped flag to skip these videos, or add them to VIDEO_TEST_CONFIG_MAP."
            )

    if missing_configs:
        print(f"  ✗ {len(missing_configs)} MISSING test config(s):")
        for video, config in missing_configs:
            print(f"    ✗ {video}")
            print(f"      → {config} (NOT FOUND)")
        raise FileNotFoundError(
            f"Missing {len(missing_configs)} test config file(s).\n"
            f"  Please check that test configs are in the correct location."
        )

    print("=" * 80)
    print()

    return base_path, resolved_dir


# Static mapping of package names to their video test configurations
# First-level key: Package name (e.g., "cosmos-predict2", "cosmos-transfer2")
# Second-level key: MP4 filename (e.g., "video1.mp4")
# Value: Path to the test config YAML file (relative to package directory)
#        Final path will be: package_name/test_config_path (e.g., "cosmos-predict2/tests/vqa_questions/...")
VIDEO_TEST_CONFIG_MAP: dict[str, dict[str, str]] = {
    "cosmos-predict2": {
        "output_Digit_Lift_movie_image2world.mp4": "tests/vqa_questions/post_training/video2world_cosmos_nemo_assets.yaml",
        "rubiks_cube_on_shelf.mp4": "tests/vqa_questions/examples/video2world_cosmos_groot.yaml",
        "robot_pouring_image2world.mp4": "tests/vqa_questions/examples/robot_pouring.yaml",
        "robot_pouring_text2world.mp4": "tests/vqa_questions/examples/robot_pouring.yaml",
        "robot_pouring_video2world.mp4": "tests/vqa_questions/examples/robot_pouring.yaml",
        "urban_freeway_image2world.mp4": "tests/vqa_questions/examples/urban_freeway.yaml",
        "urban_freeway_video2world.mp4": "tests/vqa_questions/examples/urban_freeway.yaml",
        "urban_freeway_text2world.mp4": "tests/vqa_questions/examples/urban_freeway.yaml",
    },
    "cosmos-transfer2": {
        # Example entries (paths relative to cosmos-transfer2 package):
        # "transfer_example.mp4": "tests/vqa_questions/examples/transfer_example.yaml",
    },
}


def find_mp4_files(directory: str | Path) -> list[Path]:
    """
    Recursively find all MP4 files in a directory.

    Args:
        directory: Path to the directory to search

    Returns:
        List of Path objects for all found MP4 files (sorted)

    Raises:
        FileNotFoundError: If the directory doesn't exist
        ValueError: If the path is not a directory
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Find all .mp4 files recursively
    mp4_files = sorted(directory.rglob("*.mp4"))

    return mp4_files


def run_vqa_evaluator(
    video_path: Path,
    test_config_path: Path,
    model_name: str | None = None,
    revision: str | None = None,
    validate: bool = True,
    verbose: bool = False,
    output: str | None = None,
    threshold: float = 80.0,
) -> tuple[int, str, str]:
    """
    Run vqa_evaluator.py as a subprocess for a single video.

    Args:
        video_path: Path to the video file
        test_config_path: Path to the test config YAML file
        model_name: HuggingFace model identifier
        revision: Model revision
        validate: Whether to validate answers
        verbose: Verbose output
        output: Output JSON file path
        threshold: Success rate threshold

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    # Build command - use uv run to ensure dependencies are available
    cmd = [
        "uv",
        "run",
        "--project",
        str(vqa_dir),
        "python",
        str(vqa_dir / "vqa_evaluator.py"),
        "--video_path",
        str(video_path),
        "--test_config_path",
        str(test_config_path),
        "--threshold",
        str(threshold),
    ]

    if model_name:
        cmd.extend(["--model_name", model_name])

    if revision:
        cmd.extend(["--revision", revision])

    if not validate:
        cmd.append("--no-validate")

    if verbose:
        cmd.append("--verbose")

    if output:
        cmd.extend(["--output", output])

    # Run subprocess
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )

    return result.returncode, result.stdout, result.stderr


def run_batch_vqa(
    directory: str | Path,
    package: str,
    model_name: str | None = None,
    revision: str | None = None,
    validate: bool = True,
    verbose: bool = False,
    skip_unmapped: bool = False,
    output_dir: str | None = None,
    threshold: float = 80.0,
    build_dir: str = "build",
    base_dir: str | Path | None = None,
) -> dict[str, dict]:
    """
    Run VQA evaluation on all MP4 files in a directory.

    Args:
        directory: Path to the directory containing MP4 files
        package: Package name to look up in VIDEO_TEST_CONFIG_MAP
        model_name: HuggingFace model identifier
        revision: Model revision
        validate: Whether to validate answers
        verbose: Verbose output
        skip_unmapped: Skip videos without config in VIDEO_TEST_CONFIG_MAP
        output_dir: Directory to save individual video results
        threshold: Success rate threshold
        build_dir: Build directory path (default: "build")
        base_dir: Base directory for resolving relative paths (default: None = use CWD)

    Returns:
        Dictionary mapping video filenames to their results

    Raises:
        FileNotFoundError: If directory doesn't exist or no MP4 files found
        ValueError: If package not found in VIDEO_TEST_CONFIG_MAP
    """
    # Resolve base_dir (for resolving relative paths)
    if base_dir is not None:
        base_path = Path(base_dir).resolve()
    else:
        base_path = Path.cwd()

    directory = Path(directory)

    # Validate package exists in VIDEO_TEST_CONFIG_MAP
    if package not in VIDEO_TEST_CONFIG_MAP:
        available_packages = ", ".join(VIDEO_TEST_CONFIG_MAP.keys())
        raise ValueError(
            f"Package '{package}' not found in VIDEO_TEST_CONFIG_MAP. Available packages: {available_packages}"
        )

    package_config = VIDEO_TEST_CONFIG_MAP[package]
    print(f"Using package: {package}")
    print(f"Package has {len(package_config)} video config(s)")

    # Find all MP4 files
    print(f"\nSearching for MP4 files in: {directory}")
    mp4_files = find_mp4_files(directory)

    if not mp4_files:
        raise FileNotFoundError(f"No MP4 files found in directory: {directory}")

    print(f"Found {len(mp4_files)} MP4 files\n")

    # Create output directory if needed
    output_dir_path: Path | None = None
    if output_dir:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

    # Process each video file
    all_results: dict[str, dict] = {}

    for idx, video_path in enumerate(mp4_files, 1):
        video_filename = video_path.name
        print(f"{'=' * 80}")
        print(f"[{idx}/{len(mp4_files)}] Processing: {video_filename}")
        print(f"Full path: {video_path}")

        # Look up test config for this video in the package-specific config
        if video_filename not in package_config:
            if skip_unmapped:
                print(f"⚠ Skipping {video_filename}: No test config found for package '{package}'")
                all_results[video_filename] = {
                    "status": "skipped",
                    "reason": "No config mapping",
                }
                continue
            else:
                raise ValueError(
                    f"No test config found for video '{video_filename}' in package '{package}'. "
                    f"Add an entry to VIDEO_TEST_CONFIG_MAP['{package}'] or use --skip-unmapped flag."
                )

        test_config_path = package_config[video_filename]

        test_config_path_resolved = Path(f"{build_dir}/{package}/{test_config_path}")
        if not test_config_path_resolved.is_absolute():
            test_config_path_resolved = base_path / test_config_path_resolved

        print(f"Test config: {test_config_path_resolved}")

        # Prepare output file path for this video
        video_output: str | None = None
        if output_dir_path is not None:
            video_output = str(output_dir_path / f"{video_path.stem}_results.json")

        # Run VQA evaluator
        try:
            return_code, stdout, stderr = run_vqa_evaluator(
                video_path=video_path,
                test_config_path=test_config_path_resolved,
                model_name=model_name,
                revision=revision,
                validate=validate,
                verbose=verbose,
                output=video_output,
                threshold=threshold,
            )

            # Print output
            if stdout:
                print(stdout)
            if stderr:
                print(f"STDERR: {stderr}", file=sys.stderr)

            # Store results
            if video_output and Path(video_output).exists():
                with Path(video_output).open("r") as f:
                    video_results = json.load(f)
                all_results[video_filename] = {
                    "status": "success" if return_code == 0 else "failed",
                    "return_code": return_code,
                    "output_file": video_output,
                    "results": video_results,
                }
            else:
                all_results[video_filename] = {
                    "status": "success" if return_code == 0 else "failed",
                    "return_code": return_code,
                }

        except Exception as e:  # noqa: BLE001
            print(f"✗ Error processing {video_filename}: {e!s}")
            all_results[video_filename] = {
                "status": "error",
                "error": str(e),
            }

        print()

    return all_results


def print_summary(results: dict[str, dict], package: str, threshold: float = 80.0) -> None:
    """
    Print a summary of all VQA batch results.

    Args:
        results: Dictionary of results from run_batch_vqa
        package: Package name used
        threshold: Success rate threshold
    """
    print(f"\n{'=' * 80}")
    print("BATCH VQA SUMMARY")
    print(f"{'=' * 80}")
    print(f"Package: {package}")
    print(f"Total videos: {len(results)}")

    # Count statuses
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    failed_count = sum(1 for r in results.values() if r.get("status") == "failed")
    skipped_count = sum(1 for r in results.values() if r.get("status") == "skipped")
    error_count = sum(1 for r in results.values() if r.get("status") == "error")

    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")

    # Aggregate VQA checks statistics
    total_checks = 0
    total_passed = 0

    for _, result in results.items():
        if result.get("status") in ["success", "failed"] and "results" in result:
            video_results = result["results"]
            total_checks += video_results.get("total_checks", 0)
            total_passed += video_results.get("passed", 0)

    if total_checks > 0:
        success_rate = (total_passed / total_checks) * 100.0
        print("\nAggregate VQA Statistics:")
        print(f"Total checks: {total_checks}")
        print(f"Total passed: {total_passed}")
        print(f"Total failed: {total_checks - total_passed}")
        print(f"Overall success rate: {success_rate:.2f}%")
        print(f"Threshold: {threshold:.2f}%")
        print("-" * 80)
        if success_rate >= threshold:
            print(f"✓ SUCCESS: Success rate ({success_rate:.2f}%) meets threshold ({threshold:.2f}%)")
        else:
            print(f"✗ FAILURE: Success rate ({success_rate:.2f}%) below threshold ({threshold:.2f}%)")

    print(f"{'=' * 80}")


def main() -> None:
    """
    Main entry point for CLI usage.

    MANDATORY FIRST STEP: Validates all VQA imports before any processing.
    This ensures import errors are caught immediately in CI, even if:
    - No videos exist in the directory
    - Arguments are incorrect
    - Running with --validate-imports-only flag
    """
    # MANDATORY: Validate all imports first, before any argument parsing or processing
    print("=" * 80)
    print("MANDATORY IMPORT VALIDATION")
    print("=" * 80)
    try:
        validate_vqa_imports()
    except ImportError as e:
        print(f"\n✗ Import validation failed: {e!s}", file=sys.stderr)
        print("Please ensure all VQA dependencies are installed:", file=sys.stderr)
        print("  uv sync --project packages/cosmos-oss/vqa", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Run batch VQA inference on all MP4 videos in a directory. "
        "Uses VIDEO_TEST_CONFIG_MAP to map video filenames to their test configurations."
    )

    parser.add_argument(
        "--package",
        type=str,
        required=False,  # Not required when --validate-imports-only is used
        help=f"Package name to use for config lookup (available: {', '.join(VIDEO_TEST_CONFIG_MAP.keys())})",
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=False,  # Not required when --validate-imports-only is used
        help="Path to directory containing MP4 files (searched recursively)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision (branch name, tag name, or commit id)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate answers against expected keywords (default: True)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Disable answer validation",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output during inference",
    )
    parser.add_argument(
        "--skip-unmapped",
        action="store_true",
        help="Skip videos without config in VIDEO_TEST_CONFIG_MAP",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/vqa_batch",
        help="Directory to save individual video results (default: outputs/vqa_batch)",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=None,
        help="Path to save batch summary JSON file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="Success rate threshold percentage (default: 80.0)",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default="build",
        help="Build directory path (default: build)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for resolving relative paths (default: current working directory)",
    )
    parser.add_argument(
        "--validate-imports-only",
        action="store_true",
        help="Only validate imports and exit (useful for CI pre-checks)",
    )

    args = parser.parse_args()

    # If --validate-imports-only flag is set, exit after successful import validation
    if args.validate_imports_only:
        print("\n✓ Import validation completed successfully (imports-only mode)!")
        sys.exit(0)

    # Validate required arguments for normal operation
    if not args.package or not args.directory:
        parser.error("--package and --directory are required (unless using --validate-imports-only)")

    # Validate directory resolution and test configs before processing
    try:
        base_path, resolved_directory = validate_directory_resolution(
            directory=args.directory,
            package=args.package,
            build_dir=args.build_dir,
            base_dir=args.base_dir,
            skip_unmapped=args.skip_unmapped,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n✗ Validation failed: {e!s}", file=sys.stderr)
        sys.exit(1)

    # Run batch VQA
    try:
        results = run_batch_vqa(
            directory=args.directory,
            package=args.package,
            model_name=args.model_name,
            revision=args.revision,
            validate=args.validate,
            verbose=args.verbose,
            skip_unmapped=args.skip_unmapped,
            output_dir=args.output_dir,
            threshold=args.threshold,
            build_dir=args.build_dir,
            base_dir=args.base_dir,
        )

        # Print summary
        print_summary(results, args.package, args.threshold)

        # Save summary to file if requested
        if args.output_summary:
            summary_path = Path(args.output_summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            # Calculate aggregate statistics
            total_checks = 0
            total_passed = 0

            for result in results.values():
                if result.get("status") in ["success", "failed"] and "results" in result:
                    video_results = result["results"]
                    total_checks += video_results.get("total_checks", 0)
                    total_passed += video_results.get("passed", 0)

            success_rate = (total_passed / total_checks * 100.0) if total_checks > 0 else 0.0

            summary_data = {
                "package": args.package,
                "directory": str(args.directory),
                "total_videos": len(results),
                "total_checks": total_checks,
                "passed": total_passed,
                "failed": total_checks - total_passed,
                "success_rate": success_rate,
                "threshold": args.threshold,
                "meets_threshold": success_rate >= args.threshold,
                "videos": results,
            }

            with summary_path.open("w") as f:
                json.dump(summary_data, f, indent=2)

            print(f"\nBatch summary saved to: {args.output_summary}")

        # Exit with appropriate code
        success_count = sum(1 for r in results.values() if r.get("status") == "success")
        skipped_count = sum(1 for r in results.values() if r.get("status") == "skipped")
        total_count = len(results)

        # Pass if all tests are skipped
        if skipped_count == total_count and total_count > 0 and args.skip_unmapped:
            print("All tests are skipped. Exiting with success.")
            sys.exit(0)

        if args.validate and len(results) > 0:
            # Only count passed and failed tests (exclude skipped)
            total_passed = sum(
                result.get("results", {}).get("passed", 0) for result in results.values() if "results" in result
            )
            total_failed = sum(
                result.get("results", {}).get("failed", 0) for result in results.values() if "results" in result
            )
            total_actual_checks = total_passed + total_failed

            if total_actual_checks > 0:
                success_rate = (total_passed / total_actual_checks) * 100.0
                is_success = success_rate >= args.threshold
                sys.exit(0 if is_success else 1)
            else:
                # No checks to validate - pass by default
                print("No checks to validate. Exiting with success.")
                sys.exit(0)
        else:
            print("No checks to validate. Exiting with success.")
            sys.exit(0 if success_count > 0 else 1)

    except Exception as e:  # noqa: BLE001 - Catch all exceptions to report fatal errors and exit gracefully
        print(f"\n✗ Fatal error: {e!s}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
