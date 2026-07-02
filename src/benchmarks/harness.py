"""Fair benchmark harness for comparing speculative decoding methods.

Provides a standardized framework for benchmarking different methods
under identical conditions with proper statistical analysis.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .latency_measurement import CUDATimer, adaptive_warmup, remove_outliers, timed_run
from .statistical_tests import compute_confidence_interval

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single method.

    Attributes:
        method: Name of the method benchmarked.
        latency_p50: Median latency in milliseconds.
        latency_p95: 95th percentile latency in milliseconds.
        latency_p99: 99th percentile latency in milliseconds.
        throughput_mean: Mean throughput in tokens/second.
        acceptance_rate_mean: Mean acceptance rate (for speculative methods).
        ci_95: 95% confidence interval for mean latency.
        raw_latencies: All measured latency values.
        num_runs: Number of measurement runs.
        warmup_iters: Number of warmup iterations performed.
    """

    method: str
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput_mean: float
    acceptance_rate_mean: float
    ci_95: tuple[float, float]
    raw_latencies: list[float] = field(default_factory=list)
    num_runs: int = 0
    warmup_iters: int = 0

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "method": self.method,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "throughput_mean": self.throughput_mean,
            "acceptance_rate_mean": self.acceptance_rate_mean,
            "ci_95": list(self.ci_95),
            "num_runs": self.num_runs,
            "warmup_iters": self.warmup_iters,
            "raw_latencies": self.raw_latencies,
        }


class FairBenchmarkHarness:
    """Standardized benchmark harness ensuring fair comparison.

    Ensures all methods are evaluated under identical conditions:
    - Same random seed for reproducibility
    - Same input data
    - Proper warmup before measurement
    - Statistical outlier removal
    - Confidence intervals for all metrics

    Args:
        model_name: Name/path of the model being benchmarked.
        task_name: Name of the evaluation task.
        seed: Random seed for reproducibility.
        device: CUDA device to use.
    """

    def __init__(
        self,
        model_name: str,
        task_name: str,
        seed: int = 42,
        device: str = "cuda:0",
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.seed = seed
        self.device = device
        self._results: dict[str, BenchmarkResult] = {}
        self._is_setup = False

    def setup(self) -> None:
        """Initialize the benchmark environment.

        Sets random seeds for reproducibility and prepares the device.
        """
        # Set seeds for reproducibility
        np.random.seed(self.seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        logger.info(
            f"Benchmark setup complete: model={self.model_name}, "
            f"task={self.task_name}, seed={self.seed}, device={self.device}"
        )
        self._is_setup = True

    def benchmark_method(
        self,
        method_name: str,
        method_impl: Callable[..., dict],
        num_runs: int = 30,
        outlier_percentile: float = 5.0,
    ) -> BenchmarkResult:
        """Benchmark a single method implementation.

        Performs adaptive warmup, collects measurements, removes outliers,
        and computes summary statistics with confidence intervals.

        Args:
            method_name: Identifier for this method.
            method_impl: Callable that returns a dict with at least:
                - "num_tokens": int (tokens generated)
                - "acceptance_rate": float (optional, for speculative methods)
            num_runs: Number of measurement iterations.
            outlier_percentile: Percentile for outlier removal.

        Returns:
            BenchmarkResult with all statistics.
        """
        if not self._is_setup:
            self.setup()

        logger.info(f"Benchmarking method: {method_name} ({num_runs} runs)")

        # Adaptive warmup
        warmup_count = adaptive_warmup(
            method_impl, args=(), target_cv=0.02, max_iters=100
        )

        # Measurement runs
        latencies: list[float] = []
        total_tokens: list[int] = []
        acceptance_rates: list[float] = []

        for run_idx in range(num_runs):
            timer = CUDATimer(device=self.device)
            timer.start()
            result = method_impl()
            elapsed = timer.stop()

            latencies.append(elapsed)
            total_tokens.append(result.get("num_tokens", 0))
            acceptance_rates.append(result.get("acceptance_rate", 0.0))

        # Remove outliers
        clean_latencies = remove_outliers(latencies, percentile=outlier_percentile)
        if not clean_latencies:
            clean_latencies = latencies  # Fallback if all removed

        # Compute statistics
        lat_array = np.array(clean_latencies)
        p50 = float(np.percentile(lat_array, 50))
        p95 = float(np.percentile(lat_array, 95))
        p99 = float(np.percentile(lat_array, 99))

        # Throughput: tokens / time
        mean_tokens = np.mean(total_tokens) if total_tokens else 0
        mean_latency_s = np.mean(clean_latencies) / 1000.0
        throughput = float(mean_tokens / mean_latency_s) if mean_latency_s > 0 else 0.0

        # Acceptance rate
        acceptance_mean = float(np.mean(acceptance_rates)) if acceptance_rates else 0.0

        # Confidence interval
        ci = compute_confidence_interval(clean_latencies, confidence=0.95)

        bench_result = BenchmarkResult(
            method=method_name,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            throughput_mean=throughput,
            acceptance_rate_mean=acceptance_mean,
            ci_95=ci,
            raw_latencies=latencies,
            num_runs=num_runs,
            warmup_iters=warmup_count,
        )

        self._results[method_name] = bench_result
        logger.info(
            f"  {method_name}: p50={p50:.2f}ms, p95={p95:.2f}ms, "
            f"throughput={throughput:.1f} tok/s, acceptance={acceptance_mean:.3f}"
        )
        return bench_result

    def compare_methods(self, results: Optional[dict[str, BenchmarkResult]] = None) -> str:
        """Generate a formatted comparison table of all benchmarked methods.

        Args:
            results: Dict of method_name -> BenchmarkResult. Uses internal
                     results if None.

        Returns:
            Formatted comparison table as a string.
        """
        if results is None:
            results = self._results

        if not results:
            return "No results to compare."

        # Header
        header = (
            f"{'Method':<20} {'P50(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10} "
            f"{'Tok/s':<10} {'Accept%':<10} {'CI95':<20}"
        )
        separator = "-" * len(header)
        lines = [separator, header, separator]

        for name, r in results.items():
            line = (
                f"{name:<20} {r.latency_p50:<10.2f} {r.latency_p95:<10.2f} "
                f"{r.latency_p99:<10.2f} {r.throughput_mean:<10.1f} "
                f"{r.acceptance_rate_mean * 100:<10.1f} "
                f"[{r.ci_95[0]:.2f}, {r.ci_95[1]:.2f}]"
            )
            lines.append(line)

        lines.append(separator)
        table = "\n".join(lines)
        print(table)
        return table

    def save_results(
        self,
        results: Optional[dict[str, BenchmarkResult]] = None,
        output_file: str | Path = "benchmark_results.json",
    ) -> None:
        """Save benchmark results to a JSON file.

        Args:
            results: Dict of method_name -> BenchmarkResult. Uses internal
                     results if None.
            output_file: Path to the output JSON file.
        """
        if results is None:
            results = self._results

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {
            "metadata": {
                "model": self.model_name,
                "task": self.task_name,
                "seed": self.seed,
                "device": self.device,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "results": {name: r.to_dict() for name, r in results.items()},
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        logger.info(f"Benchmark results saved to {output_path}")
