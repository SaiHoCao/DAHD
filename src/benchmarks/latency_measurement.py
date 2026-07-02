"""CUDA-aware latency measurement utilities.

Provides accurate GPU timing using CUDA events, adaptive warmup,
and outlier-robust measurement functions.
"""

from __future__ import annotations

import time
import logging
from typing import Callable, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.cuda

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


class CUDATimer:
    """High-precision GPU timer using CUDA events.

    Uses torch.cuda.Event for accurate measurement of GPU operations,
    avoiding CPU-GPU synchronization overhead in the measurement itself.

    Can be used as a context manager:
        with CUDATimer() as timer:
            # GPU operations
            ...
        elapsed = timer.elapsed_time_ms

    Or manually:
        timer = CUDATimer()
        timer.start()
        # GPU operations
        timer.stop()
        elapsed = timer.elapsed_time_ms
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize CUDA timer.

        Args:
            device: CUDA device string (e.g., "cuda:0"). Uses current device if None.
        """
        self._device = device
        self._start_event: Optional[Any] = None
        self._end_event: Optional[Any] = None
        self._elapsed_ms: float = 0.0
        self._running: bool = False

        if CUDA_AVAILABLE:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> "CUDATimer":
        """Record the start event on the current CUDA stream.

        Returns:
            Self for method chaining.
        """
        if CUDA_AVAILABLE:
            self._start_event.record()
        else:
            self._cpu_start = time.perf_counter()
        self._running = True
        return self

    def stop(self) -> float:
        """Record the end event and compute elapsed time.

        Returns:
            Elapsed time in milliseconds.
        """
        if not self._running:
            raise RuntimeError("Timer was not started. Call start() first.")

        if CUDA_AVAILABLE:
            self._end_event.record()
            torch.cuda.synchronize()
            self._elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            self._elapsed_ms = (time.perf_counter() - self._cpu_start) * 1000.0

        self._running = False
        return self._elapsed_ms

    @property
    def elapsed_time_ms(self) -> float:
        """Get the last measured elapsed time in milliseconds."""
        return self._elapsed_ms

    def __enter__(self) -> "CUDATimer":
        """Context manager entry: start the timer."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: stop the timer."""
        self.stop()


def adaptive_warmup(
    fn: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    target_cv: float = 0.02,
    max_iters: int = 100,
    window_size: int = 10,
) -> int:
    """Run warmup iterations until timing stabilizes.

    Executes the function repeatedly until the coefficient of variation (CV)
    of the most recent `window_size` measurements drops below `target_cv`.

    Args:
        fn: Function to warm up.
        args: Positional arguments for fn.
        kwargs: Keyword arguments for fn.
        target_cv: Target coefficient of variation (std/mean).
        max_iters: Maximum number of warmup iterations.
        window_size: Number of recent measurements to check CV over.

    Returns:
        Number of warmup iterations performed.
    """
    if kwargs is None:
        kwargs = {}

    timings: list[float] = []

    for i in range(max_iters):
        timer = CUDATimer()
        timer.start()
        fn(*args, **kwargs)
        elapsed = timer.stop()
        timings.append(elapsed)

        # Check stability after we have enough samples
        if len(timings) >= window_size:
            recent = timings[-window_size:]
            mean = np.mean(recent)
            if mean > 0:
                cv = np.std(recent) / mean
                if cv <= target_cv:
                    logger.debug(
                        f"Warmup stabilized after {i + 1} iterations "
                        f"(CV={cv:.4f} <= {target_cv})"
                    )
                    return i + 1

    logger.warning(
        f"Warmup did not stabilize within {max_iters} iterations "
        f"(final CV={np.std(timings[-window_size:]) / np.mean(timings[-window_size:]):.4f})"
    )
    return max_iters


def timed_run(
    fn: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    num_runs: int = 30,
    warmup_target_cv: float = 0.02,
    warmup_max_iters: int = 100,
) -> list[float]:
    """Perform a timed benchmark run with adaptive warmup.

    First performs warmup until timing stabilizes, then collects
    `num_runs` measurements.

    Args:
        fn: Function to benchmark.
        args: Positional arguments for fn.
        kwargs: Keyword arguments for fn.
        num_runs: Number of measurement runs after warmup.
        warmup_target_cv: CV target for warmup phase.
        warmup_max_iters: Maximum warmup iterations.

    Returns:
        List of elapsed times in milliseconds for each measurement run.
    """
    if kwargs is None:
        kwargs = {}

    # Warmup phase
    warmup_count = adaptive_warmup(
        fn, args, kwargs,
        target_cv=warmup_target_cv,
        max_iters=warmup_max_iters,
    )
    logger.info(f"Warmup completed in {warmup_count} iterations")

    # Measurement phase
    latencies: list[float] = []
    for _ in range(num_runs):
        timer = CUDATimer()
        timer.start()
        fn(*args, **kwargs)
        elapsed = timer.stop()
        latencies.append(elapsed)

    return latencies


def remove_outliers(data: list[float], percentile: float = 5.0) -> list[float]:
    """Remove outliers from timing data using percentile-based clipping.

    Removes data points below the `percentile`-th and above the
    `(100 - percentile)`-th percentiles.

    Args:
        data: List of measurement values.
        percentile: Percentage to clip from each tail (default 5%).

    Returns:
        Filtered list with outliers removed.
    """
    if not data:
        return []

    arr = np.array(data)
    lower = np.percentile(arr, percentile)
    upper = np.percentile(arr, 100.0 - percentile)
    filtered = arr[(arr >= lower) & (arr <= upper)]
    return filtered.tolist()
