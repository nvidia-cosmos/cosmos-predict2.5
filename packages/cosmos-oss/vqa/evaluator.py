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

import argparse
import sys
from pathlib import Path

from vqa.check import VQACheck
from vqa.exceptions import MustPassCheckFailedError
from vqa.metrics import VQAMetrics
from vqa.result import VQAResult

# Add the parent directory of 'vqa' to Python path to allow absolute imports to work when running from any location
# This allows running the script as:
#   - python packages/cosmos-oss/vqa/evaluator.py (from repo root)
#   - python vqa/evaluator.py (from ci_cd directory)
#   - python evaluator.py (from vqa directory)
vqa_dir = Path(__file__).parent.resolve()  # .../packages/cosmos-oss/vqa
parent_dir = vqa_dir.parent.resolve()  # .../packages/cosmos-oss (or wherever vqa's parent is)

print(f"Adding {parent_dir} to Python path")
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from vqa.cosmos_reason_inference import CosmosReasonModel  # noqa: E402
from vqa.utils import log_info, log_verbose  # noqa: E402


class VQAEvaluator:
    """
    Evaluates VQA checks on a single video using Cosmos Reason model.

    This class manages the model lifecycle and provides methods to run VQA evaluation
    on a video. The model is lazy-loaded on first use and can be reused across multiple
    evaluations for better performance.

    Attributes:
        model_name: HuggingFace model identifier
        revision: Model revision (branch name, tag name, or commit id)
        verbose: Whether to print detailed output during inference
    """

    must_pass_check_mandatory: bool = False

    def __init__(
        self,
        model_name: str | None = None,
        revision: str | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the VQA batch evaluator.

        Args:
            model_name: HuggingFace model identifier (default: None, uses CosmosReasonModel.MODEL_NAME)
            revision: Model revision (branch name, tag name, or commit id)
            verbose: Whether to print detailed output during inference (default: False)
        """
        self.model_name = model_name
        self.revision = revision
        self.verbose = verbose
        self._model: CosmosReasonModel | None = None

    def _initialize_model(self) -> None:
        """Lazy-load the Cosmos Reason model."""
        if self._model is None:
            log_info("\nInitializing Cosmos Reason model...")
            self._model = CosmosReasonModel(model_name=self.model_name, revision=self.revision)
            log_info(f"Model loaded: {self._model.model_name}")

    @property
    def model(self) -> CosmosReasonModel:
        """Get the model instance, initializing if necessary."""
        self._initialize_model()
        assert self._model is not None
        return self._model

    def _load_checks(self, test_config_path: str | Path) -> list[VQACheck]:
        """
        Load VQA checks from a YAML configuration file.

        Args:
            test_config_path: Path to the YAML file containing VQA checks

        Returns:
            List of VQACheck objects
        """
        log_verbose(f"Loading VQA checks from: {test_config_path}")
        vqa_checks = VQACheck.load_from_yaml(test_config_path)
        log_verbose(f"Found {len(vqa_checks)} VQA checks")
        return vqa_checks

    def _run_single_check(
        self,
        vqa_check: VQACheck,
        video_path: str | Path,
        test_config_path: str | Path,
        validate: bool,
    ) -> VQAResult:
        """
        Run inference and validation for a single VQA check.

        Args:
            vqa_check: The VQA check to evaluate
            video_path: Path to the video file
            test_config_path: Path to the test config (for error reporting)
            validate: Whether to validate the answer

        Returns:
            VQAResult object with the evaluation results

        Raises:
            MustPassCheckFailedError: If a must-pass check fails validation
        """
        actual_answer = self.model.infer(
            video_path=str(video_path),
            question=vqa_check.question,
            verbose=self.verbose,
        )
        log_verbose(f"Actual: {actual_answer}")

        # Validate answer if requested
        if validate and (vqa_check.answer or vqa_check.keywords):
            validation_passed, found_keywords = vqa_check.validate(actual_answer)

            # Raise exception only for must-pass check failures
            if not validation_passed and vqa_check.must_pass:
                log_verbose(f"Validation: ✗ MUST-PASS CHECK FAILED")
                raise MustPassCheckFailedError(vqa_check, actual_answer, video_path, test_config_path)

            log_verbose(f"Validation: {'✓ PASSED' if validation_passed else '✗ FAILED'}")
            if found_keywords:
                log_verbose(f"Found keywords: {', '.join(found_keywords)}")
        else:
            validation_passed = True
            found_keywords = []

        return VQAResult(
            check=vqa_check,
            actual_answer=actual_answer,
            validation_passed=validation_passed,
            found_keywords=found_keywords,
        )

    def _handle_must_pass_failure(
        self,
        error: MustPassCheckFailedError,
        vqa_check: VQACheck,
    ) -> VQAResult:
        """
        Handle a must-pass check failure.

        Args:
            error: The MustPassCheckFailedError exception
            vqa_check: The VQA check that failed

        Returns:
            VQAResult object with failure information
        """
        return VQAResult(
            check=vqa_check,
            actual_answer=error.actual_answer,
            validation_passed=False,
            found_keywords=[],
        )

    def _handle_exception(
        self,
        error: Exception,
        vqa_check: VQACheck,
    ) -> VQAResult:
        """
        Handle a generic exception during check execution.

        Args:
            error: The exception that occurred
            vqa_check: The VQA check being evaluated

        Returns:
            VQAResult object with error information
        """
        error_msg = f"Error: {error!s}"
        log_verbose(f"Actual: {error_msg}")
        log_verbose("Validation: ✗ ERROR")

        return VQAResult(
            check=vqa_check,
            actual_answer=error_msg,
            validation_passed=False,
            found_keywords=[],
        )

    def run_batch(
        self,
        video_path: str | Path,
        test_config_path: str | Path,
        validate: bool = True,
        print_summary: bool = True,
    ) -> list[VQAResult]:
        """
        Run VQA inference on a video with multiple questions from a YAML file.

        Args:
            video_path: Path to the video file
            test_config_path: Path to the YAML file containing VQA checks
            validate: Whether to validate answers against expected keywords (default: True)
            print_summary: Whether to print summary at the end (default: True)

        Returns:
            List of VQAResult objects containing questions, answers, and validation results
        """
        # Load VQA checks
        vqa_checks = self._load_checks(test_config_path)

        # Initialize model (lazy loading)
        self._initialize_model()

        # Run inference for each VQA check
        results: list[VQAResult] = []
        log_verbose(f"\nProcessing video: {video_path}")
        log_verbose("=" * 80)

        for idx, vqa_check in enumerate(vqa_checks, 1):
            must_pass_indicator = " [MUST PASS]" if vqa_check.must_pass else ""
            log_verbose(f"\n[Question {idx}/{len(vqa_checks)}]{must_pass_indicator}")
            log_verbose(f"Q: {vqa_check.question}")
            log_verbose(f"Expected: {vqa_check.answer}")

            try:
                result = self._run_single_check(vqa_check, video_path, test_config_path, validate)
                results.append(result)

            except MustPassCheckFailedError as e:
                # Must-pass check failed - record the failure and continue to get full report
                result = self._handle_must_pass_failure(e, vqa_check)
                results.append(result)

                if self.must_pass_check_mandatory:
                    sys.exit(1)
                else:
                    log_verbose(f"Must-pass check failed: {vqa_check.question}")
                    log_verbose(f"Continuing to process remaining checks...")

            except Exception as e:  # noqa: BLE001 - Catch all errors to continue processing remaining checks
                result = self._handle_exception(e, vqa_check)
                results.append(result)

        # Print summary if requested
        if print_summary:
            self._print_summary(results, validate)

        return results

    def _print_summary(self, results: list[VQAResult], validate: bool) -> None:
        """
        Print a summary of the evaluation results.

        Args:
            results: List of VQAResult objects
            validate: Whether validation was performed
        """
        log_info("\n" + "=" * 80)
        log_info("SUMMARY")
        log_info("=" * 80)
        log_info(f"Total checks: {len(results)}")

        if validate:
            metrics = VQAMetrics.from_results(results)
            metrics.print_summary()


def run_vqa_batch(
    video_path: str | Path,
    test_config_path: str | Path,
    model_name: str | None = None,
    revision: str | None = None,
    validate: bool = True,
    verbose: bool = False,
    print_summary: bool = True,
) -> list[VQAResult]:
    """
    Run VQA inference on a video with multiple questions from a YAML file.

    This is a convenience function that wraps VQAEvaluator for backward compatibility.
    For better performance when processing multiple videos, use VQAEvaluator directly
    or use VQABatchRunner from batch_runner.py.

    Args:
        video_path: Path to the video file
        test_config_path: Path to the YAML file containing VQA checks
        model_name: HuggingFace model identifier (default: None, uses CosmosReasonModel.MODEL_NAME)
        revision: Model revision (branch name, tag name, or commit id)
        validate: Whether to validate answers against expected keywords (default: True)
        verbose: Whether to print detailed output during inference (default: False)
        print_summary: Whether to print summary at the end (default: True)

    Returns:
        List of VQAResult objects containing questions, answers, and validation results
    """
    evaluator = VQAEvaluator(model_name=model_name, revision=revision, verbose=verbose)
    return evaluator.run_batch(video_path, test_config_path, validate, print_summary)


def evaluate_success_rate(output_path: str | Path, threshold: float = 80.0) -> bool:
    """
    Evaluate the success rate from a VQA results JSON file.

    Args:
        output_path: Path to the JSON output file
        threshold: Success threshold percentage (default: 80.0)

    Returns:
        True if success rate >= threshold AND all must-pass checks passed, False otherwise

    Raises:
        FileNotFoundError: If the output file doesn't exist
        ValueError: If the JSON file is invalid or missing required fields
    """
    import json

    output_path = Path(output_path)

    if not output_path.exists():
        raise FileNotFoundError(f"Output file not found: {output_path}")

    with output_path.open("r") as f:
        data = json.load(f)

    # Validate required fields
    if "total_checks" not in data or "passed" not in data:
        raise ValueError("JSON file must contain 'total_checks' and 'passed' fields")

    total_checks = data["total_checks"]
    passed = data["passed"]

    if total_checks == 0:
        raise ValueError("Cannot evaluate success rate: total_checks is 0")

    # Calculate success rate
    success_rate = (passed / total_checks) * 100.0

    # Determine if it meets threshold
    is_success = success_rate >= threshold

    # Check must-pass checks
    if "results" in data:
        must_pass_checks = [r for r in data["results"] if r.get("must_pass", False)]
        if must_pass_checks:
            must_pass_failed = [r for r in must_pass_checks if not r.get("validation_passed", False)]
            if must_pass_failed:
                log_info(f"\n⚠ MUST-PASS checks failed: {len(must_pass_failed)}/{len(must_pass_checks)}")
                for r in must_pass_failed:
                    log_info(f"  ✗ {r.get('question', 'Unknown')}")
                is_success = False

    # Print results
    log_info("\n" + "=" * 80)
    log_info("VQA SUCCESS RATE EVALUATION")
    log_info("=" * 80)
    log_info(f"Total Checks: {total_checks}")
    log_info(f"Passed: {passed}")
    log_info(f"Failed: {data.get('failed', total_checks - passed)}")
    log_info(f"Success Rate: {success_rate:.2f}%")
    log_info(f"Threshold: {threshold:.2f}%")
    log_info("-" * 80)
    if is_success:
        log_info(f"✓ SUCCESS: Success rate ({success_rate:.2f}%) meets threshold ({threshold:.2f}%)")
    else:
        log_info(f"✗ FAILURE: Success rate ({success_rate:.2f}%) below threshold ({threshold:.2f}%)")
    log_info("=" * 80)

    return is_success


def main() -> None:
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(description="Run batch VQA inference on a video with questions from a YAML file")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--test_config_path",
        type=str,
        required=True,
        help="Path to the YAML test configuration file containing VQA checks",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=f"HuggingFace model identifier (default: {CosmosReasonModel.MODEL_NAME})",
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
        "--output",
        type=str,
        default="outputs/vqa_results.json",
        help="Optional path to save results as JSON file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="Success rate threshold percentage (default: 80.0)",
    )
    parser.add_argument(
        "--evaluate-only",
        type=str,
        default=None,
        metavar="JSON_FILE",
        help="Evaluate success rate from existing JSON file without running inference",
    )

    args = parser.parse_args()

    # Evaluate-only mode
    if args.evaluate_only:
        is_success = evaluate_success_rate(args.evaluate_only, threshold=args.threshold)
        exit(0 if is_success else 1)

    # Create VQA evaluator from parsed arguments
    evaluator = VQAEvaluator(
        model_name=args.model_name,
        revision=args.revision,
        verbose=args.verbose,
    )

    # Run batch VQA evaluation
    results = evaluator.run_batch(
        video_path=args.video_path,
        test_config_path=args.test_config_path,
        validate=args.validate,
    )

    # Optionally save results to file
    if args.output:
        import json

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to serializable format
        results_dict = []
        for result in results:
            results_dict.append(
                {
                    "question": result.check.question,
                    "expected_answer": result.check.answer,
                    "actual_answer": result.actual_answer,
                    "expected_keywords": result.check.keywords,
                    "validation_passed": result.validation_passed,
                    "found_keywords": result.found_keywords,
                    "must_pass": result.check.must_pass,
                }
            )

        output_data = {
            "video_path": str(args.video_path),
            "test_config_path": str(args.test_config_path),
            "total_checks": len(results),
            "passed": sum(1 for r in results if r.validation_passed),
            "failed": sum(1 for r in results if not r.validation_passed),
            "results": results_dict,
        }

        with output_path.open("w") as f:
            json.dump(output_data, f, indent=2)

        log_verbose(f"\nResults saved to: {args.output}")

        # Evaluate success rate
        is_success = evaluate_success_rate(args.output, threshold=args.threshold)
        exit(0 if is_success else 1)
    else:
        # Exit with appropriate code based on validation results
        if args.validate:
            passed = sum(1 for r in results if r.validation_passed)
            success_rate = (passed / len(results)) * 100.0
            is_success = success_rate >= args.threshold
            exit(0 if is_success else 1)


if __name__ == "__main__":
    main()
