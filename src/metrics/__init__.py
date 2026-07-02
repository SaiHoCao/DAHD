"""Metrics collection and analysis package for DAHD speculative decoding.

Provides tools for collecting per-token and per-sequence metrics,
persisting them to disk, and validating data integrity.
"""

from .schema import GlobalBenchmarkMetrics, PerSequenceMetrics, PerTokenMetrics
from .collector import PerTokenMetricsCollector, PerSequenceMetricsAggregator
from .storage import MetricsStore
from .validator import (
    MetricsValidator,
    validate_per_token,
    validate_per_sequence,
    validate_global,
)

__all__ = [
    # Schema
    "PerTokenMetrics",
    "PerSequenceMetrics",
    "GlobalBenchmarkMetrics",
    # Collection
    "PerTokenMetricsCollector",
    "PerSequenceMetricsAggregator",
    # Storage
    "MetricsStore",
    # Validation
    "MetricsValidator",
    "validate_per_token",
    "validate_per_sequence",
    "validate_global",
]
