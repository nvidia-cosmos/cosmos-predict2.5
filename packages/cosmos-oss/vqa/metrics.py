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

from dataclasses import dataclass

from vqa.result import VQAResult
from vqa.utils import log_info


@dataclass
class VQAMetrics:
    """
    Encapsulates metrics for a batch of VQA results.

    Attributes:
        total_checks: Total number of checks performed
        passed: Number of checks that passed validation
        failed: Number of checks that failed validation
        success_rate: Percentage of checks that passed (0-100)
        must_pass_total: Total number of must-pass checks
        must_pass_passed: Number of must-pass checks that passed
        must_pass_failed: Number of must-pass checks that failed
        failed_must_pass_questions: List of questions from failed must-pass checks
    """

    total_checks: int
    passed: int
    failed: int
    success_rate: float
    must_pass_total: int
    must_pass_passed: int
    must_pass_failed: int
    failed_must_pass_questions: list[str]

    @classmethod
    def from_results(cls, results: list[VQAResult]) -> "VQAMetrics":
        """
        Calculate metrics from a list of VQA results.

        Args:
            results: List of VQAResult objects

        Returns:
            VQAMetrics object with calculated metrics
        """
        total_checks = len(results)
        passed = sum(1 for r in results if r.validation_passed)
        failed = total_checks - passed
        success_rate = (passed / total_checks * 100) if total_checks > 0 else 0.0

        # Calculate must-pass metrics
        must_pass_checks = [r for r in results if r.check.must_pass]
        must_pass_total = len(must_pass_checks)
        must_pass_passed = sum(1 for r in must_pass_checks if r.validation_passed)
        must_pass_failed = must_pass_total - must_pass_passed

        # Collect failed must-pass questions
        failed_must_pass_questions = [r.check.question for r in must_pass_checks if not r.validation_passed]

        return cls(
            total_checks=total_checks,
            passed=passed,
            failed=failed,
            success_rate=success_rate,
            must_pass_total=must_pass_total,
            must_pass_passed=must_pass_passed,
            must_pass_failed=must_pass_failed,
            failed_must_pass_questions=failed_must_pass_questions,
        )

    def print_summary(self) -> None:
        """Print a formatted summary of the metrics."""
        log_info(f"Passed: {self.passed}")
        log_info(f"Failed: {self.failed}")
        log_info(f"Success rate: {self.success_rate:.1f}%")

        if self.must_pass_total > 0:
            log_info(f"\nMust-pass checks: {self.must_pass_total}")
            log_info(f"Must-pass passed: {self.must_pass_passed}")
            log_info(f"Must-pass failed: {self.must_pass_failed}")

            if self.must_pass_failed > 0:
                log_info("\n⚠ WARNING: Some MUST-PASS checks failed!")
                for question in self.failed_must_pass_questions:
                    log_info(f"  ✗ {question}")

    def has_failures(self) -> bool:
        """
        Check if there are any failures in the results.

        Returns:
            True if any must-pass checks failed, False otherwise
        """
        return self.must_pass_failed > 0

    def meets_threshold(self, threshold: float) -> bool:
        """
        Check if the success rate meets the given threshold.

        Args:
            threshold: Success rate threshold (0-100)

        Returns:
            True if success rate >= threshold AND no must-pass failures, False otherwise
        """
        return self.success_rate >= threshold and not self.has_failures()
