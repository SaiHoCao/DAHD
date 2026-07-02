"""Validation utilities for metrics data integrity.

Ensures that collected metrics conform to expected ranges and constraints
before storage or analysis.
"""

from __future__ import annotations

import logging
from typing import Optional

from .schema import PerTokenMetrics, PerSequenceMetrics, GlobalBenchmarkMetrics

logger = logging.getLogger(__name__)

VALID_DRAFT_MODES = {"parallel", "ar"}


def validate_per_token(metrics: PerTokenMetrics) -> list[str]:
    """Validate a single per-token metrics record.

    Args:
        metrics: The per-token metrics to validate.

    Returns:
        List of error messages. Empty list means valid.
    """
    errors: list[str] = []

    if not (0.0 <= metrics.probe_confidence <= 1.0):
        errors.append(
            f"probe_confidence={metrics.probe_confidence} out of range [0, 1]"
        )

    if metrics.draft_mode not in VALID_DRAFT_MODES:
        errors.append(
            f"draft_mode='{metrics.draft_mode}' not in {VALID_DRAFT_MODES}"
        )

    if metrics.draft_latency_ms < 0:
        errors.append(
            f"draft_latency_ms={metrics.draft_latency_ms} must be >= 0"
        )

    if metrics.token_pos < 0:
        errors.append(f"token_pos={metrics.token_pos} must be >= 0")

    if metrics.timestamp <= 0:
        errors.append(f"timestamp={metrics.timestamp} must be > 0")

    return errors


def validate_per_sequence(metrics: PerSequenceMetrics) -> list[str]:
    """Validate a single per-sequence metrics record.

    Args:
        metrics: The per-sequence metrics to validate.

    Returns:
        List of error messages. Empty list means valid.
    """
    errors: list[str] = []

    if not (0.0 <= metrics.acceptance_rate <= 1.0):
        errors.append(
            f"acceptance_rate={metrics.acceptance_rate} out of range [0, 1]"
        )

    if metrics.num_easy_tokens + metrics.num_hard_tokens > metrics.total_tokens_drafted:
        errors.append(
            f"num_easy({metrics.num_easy_tokens}) + num_hard({metrics.num_hard_tokens}) "
            f"> total_tokens_drafted({metrics.total_tokens_drafted})"
        )

    if metrics.total_tokens_accepted > metrics.total_tokens_drafted:
        errors.append(
            f"total_tokens_accepted({metrics.total_tokens_accepted}) "
            f"> total_tokens_drafted({metrics.total_tokens_drafted})"
        )

    if metrics.avg_draft_latency_ms < 0:
        errors.append(
            f"avg_draft_latency_ms={metrics.avg_draft_latency_ms} must be >= 0"
        )

    if metrics.avg_verify_latency_ms < 0:
        errors.append(
            f"avg_verify_latency_ms={metrics.avg_verify_latency_ms} must be >= 0"
        )

    if metrics.total_e2e_latency_ms < 0:
        errors.append(
            f"total_e2e_latency_ms={metrics.total_e2e_latency_ms} must be >= 0"
        )

    if not (0.0 <= metrics.avg_probe_confidence <= 1.0):
        errors.append(
            f"avg_probe_confidence={metrics.avg_probe_confidence} out of range [0, 1]"
        )

    return errors


def validate_global(metrics: GlobalBenchmarkMetrics) -> list[str]:
    """Validate global benchmark metrics.

    Args:
        metrics: The global metrics to validate.

    Returns:
        List of error messages. Empty list means valid.
    """
    errors: list[str] = []

    if metrics.avg_latency_p50_ms < 0:
        errors.append(f"avg_latency_p50_ms={metrics.avg_latency_p50_ms} must be >= 0")

    if metrics.avg_latency_p95_ms < 0:
        errors.append(f"avg_latency_p95_ms={metrics.avg_latency_p95_ms} must be >= 0")

    if metrics.throughput_tokens_per_sec < 0:
        errors.append(
            f"throughput_tokens_per_sec={metrics.throughput_tokens_per_sec} must be >= 0"
        )

    if not (0.0 <= metrics.acceptance_rate_global <= 1.0):
        errors.append(
            f"acceptance_rate_global={metrics.acceptance_rate_global} "
            f"out of range [0, 1]"
        )

    if metrics.speedup_vs_ar < 0:
        errors.append(f"speedup_vs_ar={metrics.speedup_vs_ar} must be >= 0")

    return errors


class MetricsValidator:
    """Batch validator that collects and reports validation errors.

    Processes batches of metrics and accumulates error reports for
    logging or raising exceptions.
    """

    def __init__(self, strict: bool = False):
        """Initialize the validator.

        Args:
            strict: If True, raise ValueError on first validation error.
                    If False, log warnings and continue.
        """
        self.strict = strict
        self._errors: list[dict] = []
        self._total_validated: int = 0
        self._total_invalid: int = 0

    def validate_token_batch(self, metrics_list: list[PerTokenMetrics]) -> int:
        """Validate a batch of per-token metrics.

        Args:
            metrics_list: List of per-token metrics to validate.

        Returns:
            Number of invalid records found.
        """
        invalid_count = 0
        for i, m in enumerate(metrics_list):
            errors = validate_per_token(m)
            self._total_validated += 1
            if errors:
                invalid_count += 1
                self._total_invalid += 1
                error_record = {
                    "type": "per_token",
                    "index": i,
                    "sequence_id": m.sequence_id,
                    "token_pos": m.token_pos,
                    "errors": errors,
                }
                self._errors.append(error_record)
                if self.strict:
                    raise ValueError(
                        f"Per-token validation failed: {errors}"
                    )
                else:
                    logger.warning(
                        f"Invalid per-token metric at index {i}: {errors}"
                    )
        return invalid_count

    def validate_sequence_batch(
        self, metrics_list: list[PerSequenceMetrics]
    ) -> int:
        """Validate a batch of per-sequence metrics.

        Args:
            metrics_list: List of per-sequence metrics to validate.

        Returns:
            Number of invalid records found.
        """
        invalid_count = 0
        for i, m in enumerate(metrics_list):
            errors = validate_per_sequence(m)
            self._total_validated += 1
            if errors:
                invalid_count += 1
                self._total_invalid += 1
                error_record = {
                    "type": "per_sequence",
                    "index": i,
                    "sequence_id": m.sequence_id,
                    "errors": errors,
                }
                self._errors.append(error_record)
                if self.strict:
                    raise ValueError(
                        f"Per-sequence validation failed: {errors}"
                    )
                else:
                    logger.warning(
                        f"Invalid per-sequence metric at index {i}: {errors}"
                    )
        return invalid_count

    @property
    def error_report(self) -> dict:
        """Generate a summary report of all validation errors.

        Returns:
            Dictionary with validation statistics and error details.
        """
        return {
            "total_validated": self._total_validated,
            "total_invalid": self._total_invalid,
            "invalid_rate": (
                self._total_invalid / self._total_validated
                if self._total_validated > 0
                else 0.0
            ),
            "errors": self._errors,
        }

    def reset(self) -> None:
        """Clear all accumulated errors and counters."""
        self._errors.clear()
        self._total_validated = 0
        self._total_invalid = 0
