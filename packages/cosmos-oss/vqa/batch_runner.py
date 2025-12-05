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
using the evaluator.py script. It uses a static mapping (VIDEO_TEST_CONFIG_MAP)
to associate video filenames with their corresponding test configuration files.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add the parent directory of 'vqa' to Python path to allow absolute imports
vqa_dir = Path(__file__).parent.resolve()  # .../packages/cosmos-oss/vqa
parent_dir = vqa_dir.parent.resolve()  # .../packages/cosmos-oss

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from vqa.check_prereqs import VQAValidator  # noqa: E402
from vqa.utils import find_mp4_files, log_info, log_verbose  # noqa: E402 # type: ignore[attr-defined]

# Static mapping of package names to their video test configurations
# First-level key: Package name (e.g., "cosmos-predict2", "cosmos-transfer2")
# Second-level key: MP4 filename (e.g., "video1.mp4")
# Value: Path to the test config YAML file (relative to package directory)
#        Final path will be: package_name/test_config_path (e.g., "cosmos-predict2/tests/vqa_questions/...")
VIDEO_TEST_CONFIG_MAP: dict[str, dict[str, str]] = {
    "cosmos-predict2": {
        "output_Digit_Lift_movie_image2world.mp4": "tests/vqa_questions/examples/video2world_cosmos_nemo_assets.yaml",
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


class VQABatchRunner:
    """
    Orchestrates VQA evaluation across multiple videos.

    This class manages the evaluation of multiple videos, reusing the same model instance
    for better performance. It handles video discovery, config mapping, and result aggregation.

    Attributes:
        package: Package name to look up in VIDEO_TEST_CONFIG_MAP
        model_name: HuggingFace model identifier
        revision: Model revision
        validate: Whether to validate answers
        verbose: Verbose output
        skip_unmapped: Skip videos without config mapping
        threshold: Success rate threshold
        build_dir: Build directory path
        base_dir: Base directory for resolving relative paths
    """

    def __init__(
        self,
        package: str,
        directory: str | Path,
        model_name: str | None = None,
        revision: str | None = None,
        validate: bool = True,
        verbose: bool = False,
        skip_unmapped: bool = False,
        threshold: float = 80.0,
        build_dir: str = "build",
        base_dir: str | Path | None = None,
        output_dir: str | None = None,
        output_summary: str | None = None,
    ) -> None:
        """
        Initialize the VQA batch runner.

        Args:
            package: Package name to look up in VIDEO_TEST_CONFIG_MAP
            directory: Path to directory containing MP4 files
            model_name: HuggingFace model identifier
            revision: Model revision
            validate: Whether to validate answers
            verbose: Verbose output
            skip_unmapped: Skip videos without config mapping
            threshold: Success rate threshold
            build_dir: Build directory path (default: "build")
            base_dir: Base directory for resolving relative paths (default: None = use CWD)
            output_dir: Directory to save individual video results
            output_summary: Path to save batch summary JSON file

        Raises:
            ValueError: If package not found in VIDEO_TEST_CONFIG_MAP
        """
        self.package = package
        self.directory = Path(directory)
        self.model_name = model_name
        self.revision = revision
        self.validate = validate
        self.verbose = verbose
        self.skip_unmapped = skip_unmapped
        self.threshold = threshold
        self.build_dir = build_dir
        self.output_dir = output_dir
        self.output_summary = output_summary

        # Resolve base_dir
        if base_dir is not None:
            self.base_path = Path(base_dir).resolve()
        else:
            self.base_path = Path.cwd()

        # Validate package exists
        if package not in VIDEO_TEST_CONFIG_MAP:
            available_packages = ", ".join(VIDEO_TEST_CONFIG_MAP.keys())
            raise ValueError(
                f"Package '{package}' not found in VIDEO_TEST_CONFIG_MAP. Available packages: {available_packages}"
            )

        self.package_config = VIDEO_TEST_CONFIG_MAP[package]
        self._evaluator = None

    @property
    def evaluator(self):
        """Get the evaluator instance, initializing if necessary."""
        if self._evaluator is None:
            from vqa.evaluator import VQAEvaluator

            self._evaluator = VQAEvaluator(
                model_name=self.model_name,
                revision=self.revision,
                verbose=self.verbose,
            )
        return self._evaluator

    @classmethod
    def from_args(cls, args) -> "VQABatchRunner":
        """
        Create a VQABatchRunner from parsed command-line arguments.

        Args:
            args: argparse.Namespace object with parsed arguments

        Returns:
            VQABatchRunner instance configured from arguments
        """
        return cls(
            package=args.package,
            directory=args.directory,
            model_name=args.model_name,
            revision=args.revision,
            validate=args.validate,
            verbose=args.verbose,
            skip_unmapped=args.skip_unmapped,
            threshold=args.threshold,
            build_dir=args.build_dir,
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            output_summary=args.output_summary,
        )

    def run_vqa_evaluator(
        self,
        video_path: Path,
        test_config_path: Path,
        validate: bool = True,
        output: str | None = None,
        threshold: float = 80.0,
    ) -> tuple[int, dict]:
        """
        Run VQA evaluation on a single video using direct function calls.

        Args:
            video_path: Path to the video file
            test_config_path: Path to the test config YAML file
            validate: Whether to validate answers
            output: Output JSON file path
            threshold: Success rate threshold

        Returns:
            Tuple of (exit_code, results_dict)
                exit_code: 0 if success rate >= threshold, 1 otherwise
                results_dict: Dictionary containing evaluation results
        """
        import json

        try:
            # Run VQA evaluation directly using the evaluator
            results = self.evaluator.run_batch(
                video_path=video_path,
                test_config_path=test_config_path,
                validate=validate,
            )

            # Calculate success metrics
            passed = sum(1 for r in results if r.validation_passed)
            total = len(results)
            success_rate = (passed / total * 100) if total > 0 else 0.0
            is_success = success_rate >= threshold

            # Build results dictionary
            results_dict = {
                "video_path": str(video_path),
                "test_config_path": str(test_config_path),
                "total_checks": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": success_rate,
                "results": [
                    {
                        "question": r.check.question,
                        "expected_answer": r.check.answer,
                        "actual_answer": r.actual_answer,
                        "expected_keywords": r.check.keywords,
                        "validation_passed": r.validation_passed,
                        "found_keywords": r.found_keywords,
                        "must_pass": r.check.must_pass,
                    }
                    for r in results
                ],
            }

            # Optionally save results to file
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w") as f:
                    json.dump(results_dict, f, indent=2)

            return (0 if is_success else 1, results_dict)

        except Exception as e:
            # Return error result
            return (1, {"error": str(e), "video_path": str(video_path)})

    def _resolve_test_config_path(self, video_filename: str) -> Path:
        """
        Resolve the test config path for a given video.

        Args:
            video_filename: Name of the video file

        Returns:
            Resolved Path to the test config file

        Raises:
            ValueError: If no config mapping found and skip_unmapped is False
        """
        if video_filename not in self.package_config:
            if self.skip_unmapped:
                raise ValueError(f"No config mapping for {video_filename} (skip_unmapped=True)")
            else:
                raise ValueError(
                    f"No test config found for video '{video_filename}' in package '{self.package}'. "
                    f"Add an entry to VIDEO_TEST_CONFIG_MAP['{self.package}'] or use --skip-unmapped flag."
                )

        test_config_path = self.package_config[video_filename]
        test_config_path_resolved = Path(self.build_dir) / self.package / test_config_path

        if not test_config_path_resolved.is_absolute():
            test_config_path_resolved = self.base_path / test_config_path_resolved

        return test_config_path_resolved

    def run_batch(self) -> dict[str, dict]:
        """
        Run VQA evaluation on all MP4 files in the configured directory.

        Returns:
            Dictionary mapping video filenames to their results

        Raises:
            FileNotFoundError: If directory doesn't exist or no MP4 files found
        """

        log_verbose(f"Using package: {self.package}")
        log_verbose(f"Package has {len(self.package_config)} video config(s)")

        # Find all MP4 files
        log_verbose(f"\nSearching for MP4 files in: {self.directory}")
        mp4_files = find_mp4_files(self.directory)

        if not mp4_files:
            raise FileNotFoundError(f"No MP4 files found in directory: {self.directory}")

        log_verbose(f"Found {len(mp4_files)} MP4 files\n")

        # Create output directory if needed
        output_dir_path: Path | None = None
        if self.output_dir:
            output_dir_path = Path(self.output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)

        # Process each video file
        all_results: dict[str, dict] = {}

        for idx, video_path in enumerate(mp4_files, 1):
            video_filename = video_path.name
            log_verbose(f"{'=' * 80}")
            log_verbose(f"[{idx}/{len(mp4_files)}] Processing: {video_filename}")
            log_verbose(f"Full path: {video_path}")

            try:
                test_config_path_resolved = self._resolve_test_config_path(video_filename)
                log_verbose(f"Test config: {test_config_path_resolved}")

                # Run VQA evaluation directly using VQAEvaluator
                # Suppress individual video summaries - we'll print batch summary at the end
                print_summary = True if os.getenv("VERBOSE", "0") == "1" else False
                results = self.evaluator.run_batch(
                    video_path=video_path,
                    test_config_path=test_config_path_resolved,
                    validate=self.validate,
                    print_summary=print_summary,
                )

                # Save individual video results if output_dir specified
                if output_dir_path is not None:
                    video_output = output_dir_path / f"{video_path.stem}_results.json"
                    self._save_results(video_output, video_path, test_config_path_resolved, results)

                # Determine success/failure
                passed = sum(1 for r in results if r.validation_passed)
                total = len(results)
                success_rate = (passed / total * 100) if total > 0 else 0.0
                is_success = success_rate >= self.threshold

                # Store results
                all_results[video_filename] = {
                    "status": "success" if is_success else "failed",
                    "total_checks": total,
                    "passed": passed,
                    "failed": total - passed,
                    "success_rate": success_rate,
                    "results": [
                        {
                            "question": r.check.question,
                            "expected_answer": r.check.answer,
                            "actual_answer": r.actual_answer,
                            "validation_passed": r.validation_passed,
                            "found_keywords": r.found_keywords,
                            "must_pass": r.check.must_pass,
                        }
                        for r in results
                    ],
                }

            except ValueError as e:
                if "No config mapping" in str(e) and self.skip_unmapped:
                    log_verbose(f"⚠ Skipping {video_filename}: {e}")
                    all_results[video_filename] = {
                        "status": "skipped",
                        "reason": "No config mapping",
                    }
                else:
                    raise

            except Exception as e:  # noqa: BLE001
                log_verbose(f"✗ Error processing {video_filename}: {e!s}")
                all_results[video_filename] = {
                    "status": "error",
                    "error": str(e),
                }

            log_verbose("")

        return all_results

    def _save_results(
        self,
        output_path: Path,
        video_path: Path,
        test_config_path: Path,
        results: list,
    ) -> None:
        """Save VQA results to a JSON file."""
        output_data = {
            "video_path": str(video_path),
            "test_config_path": str(test_config_path),
            "total_checks": len(results),
            "passed": sum(1 for r in results if r.validation_passed),
            "failed": sum(1 for r in results if not r.validation_passed),
            "results": [
                {
                    "question": r.check.question,
                    "expected_answer": r.check.answer,
                    "actual_answer": r.actual_answer,
                    "expected_keywords": r.check.keywords,
                    "validation_passed": r.validation_passed,
                    "found_keywords": r.found_keywords,
                    "must_pass": r.check.must_pass,
                }
                for r in results
            ],
        }

        with output_path.open("w") as f:
            json.dump(output_data, f, indent=2)


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

    This is a convenience function that wraps VQABatchRunner for backward compatibility.
    For better control and performance, use VQABatchRunner directly.

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
    # Use VQABatchRunner class for better performance (model reuse)
    runner = VQABatchRunner(
        package=package,
        directory=directory,
        model_name=model_name,
        revision=revision,
        validate=validate,
        verbose=verbose,
        skip_unmapped=skip_unmapped,
        threshold=threshold,
        build_dir=build_dir,
        base_dir=base_dir,
        output_dir=output_dir,
    )

    return runner.run_batch()


def print_video_results(results: dict[str, dict]) -> None:
    """
    Print individual video results in pytest-like format.

    Args:
        results: Dictionary of results from run_batch_vqa
    """
    log_info("\n")

    for video_name, result in results.items():
        # Get status
        status = result.get("status", "unknown").upper()

        # Build status line
        if status in ["SUCCESS", "FAILED"]:
            passed = result.get("passed", 0)
            total_checks = result.get("total_checks", 0)
            success_rate = result.get("success_rate", 0.0)
            status_line = f"[{success_rate:5.1f}%] {status:7s} {video_name} ({passed}/{total_checks} checks)"

            # For failed videos, print the failed questions
            if status == "FAILED" and "results" in result:
                failed_checks = [check for check in result["results"] if not check.get("validation_passed", False)]
                if failed_checks:
                    status_line += "\n  Failed questions:"
                    for check in failed_checks:
                        question = check.get("question", "Unknown question")
                        expected = check.get("expected_answer", "N/A")
                        actual = check.get("actual_answer", "N/A")
                        status_line += f"\n    - Q: {question}"
                        status_line += f"\n      Expected: {expected}"
                        status_line += f"\n      Got: {actual}"
        elif status == "SKIPPED":
            reason = result.get("reason", "unknown reason")
            status_line = f"[  N/A ] {status:7s} {video_name} ({reason})"
        elif status == "ERROR":
            error = result.get("error", "unknown error")
            # Truncate error message if too long
            error_msg = error if len(error) < 50 else f"{error[:47]}..."
            status_line = f"[  N/A ] {status:7s} {video_name} ({error_msg})"
        else:
            status_line = f"[  N/A ] {status:7s} {video_name}"

        log_info(status_line)


def print_summary(results: dict[str, dict], package: str, threshold: float = 80.0) -> None:
    """
    Print a summary of all VQA batch results.

    Args:
        results: Dictionary of results from run_batch_vqa
        package: Package name used
        threshold: Success rate threshold
    """
    log_info(f"\n{'=' * 80}")
    log_info("BATCH VQA SUMMARY")
    log_info(f"{'=' * 80}")
    log_info(f"Package: {package}")
    log_info(f"Total videos: {len(results)}")

    # Count statuses
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    failed_count = sum(1 for r in results.values() if r.get("status") == "failed")
    skipped_count = sum(1 for r in results.values() if r.get("status") == "skipped")
    error_count = sum(1 for r in results.values() if r.get("status") == "error")

    log_info(f"Successful: {success_count}")
    log_info(f"Failed: {failed_count}")
    log_info(f"Skipped: {skipped_count}")
    log_info(f"Errors: {error_count}")

    # Aggregate VQA checks statistics
    total_checks = 0
    total_passed = 0

    for _, result in results.items():
        if result.get("status") in ["success", "failed"] and "results" in result:
            total_checks += result.get("total_checks", 0)
            total_passed += result.get("passed", 0)

    if total_checks > 0:
        success_rate = (total_passed / total_checks) * 100.0
        log_info("\nAggregate VQA Statistics:")
        log_info(f"Total checks: {total_checks}")
        log_info(f"Total passed: {total_passed}")
        log_info(f"Total failed: {total_checks - total_passed}")
        log_info(f"Overall success rate: {success_rate:.2f}%")
        log_info(f"Threshold: {threshold:.2f}%")
        log_info("-" * 80)
        if success_rate >= threshold:
            log_info(f"✓ SUCCESS: Success rate ({success_rate:.2f}%) meets threshold ({threshold:.2f}%)")
        else:
            log_info(f"✗ FAILURE: Success rate ({success_rate:.2f}%) below threshold ({threshold:.2f}%)")

    log_info(f"{'=' * 80}")


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
        VQAValidator.validate_imports()
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
        default="vqa_output/summary.json",
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
        base_path, resolved_directory = VQAValidator.validate_directory_resolution(
            directory=args.directory,
            package=args.package,
            video_test_config_map=VIDEO_TEST_CONFIG_MAP,
            build_dir=args.build_dir,
            base_dir=args.base_dir,
            skip_unmapped=args.skip_unmapped,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n✗ Validation failed: {e!s}", file=sys.stderr)
        sys.exit(1)

    # Create VQA batch runner from parsed arguments and run
    try:
        runner = VQABatchRunner.from_args(args)
        results = runner.run_batch()

        # Print individual video results (pytest-like format)
        print_video_results(results)

        # Print summary
        print_summary(results, runner.package, runner.threshold)

        # Save summary to file if requested
        if runner.output_summary:
            summary_path = Path(runner.output_summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            # Calculate aggregate statistics
            total_checks = 0
            total_passed = 0

            for result in results.values():
                if result.get("status") in ["success", "failed"] and "results" in result:
                    total_checks += result.get("total_checks", 0)
                    total_passed += result.get("passed", 0)

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

            log_verbose(f"\nBatch summary saved to: {args.output_summary}")

        # Exit with appropriate code
        success_count = sum(1 for r in results.values() if r.get("status") == "success")
        skipped_count = sum(1 for r in results.values() if r.get("status") == "skipped")
        total_count = len(results)

        # Pass if all tests are skipped
        if skipped_count == total_count and total_count > 0 and args.skip_unmapped:
            log_verbose("All tests are skipped. Exiting with success.")
            sys.exit(0)

        if args.validate and len(results) > 0:
            # Only count passed and failed tests (exclude skipped)
            total_passed = sum(result.get("passed", 0) for result in results.values() if "results" in result)
            total_failed = sum(result.get("failed", 0) for result in results.values() if "results" in result)
            total_actual_checks = total_passed + total_failed

            if total_actual_checks > 0:
                success_rate = (total_passed / total_actual_checks) * 100.0
                is_success = success_rate >= args.threshold
                sys.exit(0 if is_success else 1)
            else:
                # No checks to validate - pass by default
                log_verbose("No checks to validate. Exiting with success.")
                sys.exit(0)
        else:
            log_verbose("No checks to validate. Exiting with success.")
            sys.exit(0 if success_count > 0 else 1)

    except Exception as e:  # noqa: BLE001 - Catch all exceptions to report fatal errors and exit gracefully
        print(f"\n✗ Fatal error: {e!s}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
