"""Benchmark framework for DAHD speculative decoding.

Provides tools for fair latency measurement, task-specific evaluation,
statistical comparison, and benchmark orchestration.
"""

from .latency_measurement import (
    CUDATimer,
    adaptive_warmup,
    timed_run,
    remove_outliers,
)
from .harness import BenchmarkResult, FairBenchmarkHarness
from .task_runners import (
    TaskRunner,
    GSM8KRunner,
    MATHRunner,
    HumanEvalRunner,
    MTBenchRunner,
    CNNDailyMailRunner,
    get_task_runner,
    list_available_tasks,
)
from .statistical_tests import (
    ComparisonReport,
    test_bimodal_hypothesis,
    compare_two_methods,
    compute_confidence_interval,
    bootstrap_speedup,
)

__all__ = [
    # Latency measurement
    "CUDATimer",
    "adaptive_warmup",
    "timed_run",
    "remove_outliers",
    # Harness
    "BenchmarkResult",
    "FairBenchmarkHarness",
    # Task runners
    "TaskRunner",
    "GSM8KRunner",
    "MATHRunner",
    "HumanEvalRunner",
    "MTBenchRunner",
    "CNNDailyMailRunner",
    "get_task_runner",
    "list_available_tasks",
    # Statistical tests
    "ComparisonReport",
    "test_bimodal_hypothesis",
    "compare_two_methods",
    "compute_confidence_interval",
    "bootstrap_speedup",
]
