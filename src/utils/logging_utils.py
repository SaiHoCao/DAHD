"""Logging utilities for DAHD experiments."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_logger(
    name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    format_string: str | None = None,
) -> logging.Logger:
    """Configure and return a Python logger.

    Creates a logger with both console and optional file handlers.
    The console handler uses a concise format, while the file handler
    includes timestamps and module information.

    Args:
        name: Logger name (typically __name__ of the calling module).
        log_file: Optional path to a log file. If provided, logs are also
            written to this file.
        level: Logging level (default: INFO).
        format_string: Optional custom format string. If None, uses defaults.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Console handler with concise format
    console_format = format_string or "%(levelname)s | %(name)s | %(message)s"
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console_handler)

    # File handler with detailed format
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_format = format_string or (
            "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
        )
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(file_format))
        logger.addHandler(file_handler)

    return logger


@dataclass
class ExperimentLogger:
    """Structured experiment logger for recording metadata and metrics.

    Records experiment lifecycle events including start time, configuration,
    GPU information, and key metrics. Provides methods for logging structured
    events that can later be analyzed.

    Attributes:
        experiment_name: Name of the experiment.
        log_dir: Directory for log files.
        start_time: Experiment start timestamp.
        gpu_info: GPU device information string.
        config_snapshot: Snapshot of the experiment configuration.
        events: List of logged events with timestamps.
    """

    experiment_name: str
    log_dir: str = "logs"
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    gpu_info: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set up the internal logger and log directory."""
        log_path = Path(self.log_dir) / f"{self.experiment_name}.log"
        self._logger = setup_logger(
            name=f"experiment.{self.experiment_name}",
            log_file=log_path,
        )
        self._start_perf = time.perf_counter()

    def log_config(self, config: Any) -> None:
        """Log the experiment configuration.

        Args:
            config: Configuration object (must support asdict or __dict__).
        """
        if hasattr(config, "__dataclass_fields__"):
            self.config_snapshot = asdict(config)
        elif hasattr(config, "__dict__"):
            self.config_snapshot = config.__dict__.copy()
        else:
            self.config_snapshot = {"raw": str(config)}

        self._logger.info(f"Configuration: {self.config_snapshot}")

    def log_gpu_info(self, gpu_info_str: str) -> None:
        """Log GPU device information.

        Args:
            gpu_info_str: String description of the GPU device.
        """
        self.gpu_info = gpu_info_str
        self._logger.info(f"GPU Info: {gpu_info_str}")

    def log_event(self, event_type: str, **kwargs: Any) -> None:
        """Log a structured event with a timestamp.

        Args:
            event_type: Type/category of the event (e.g., 'phase_start', 'metric').
            **kwargs: Additional key-value data to record with the event.
        """
        elapsed = time.perf_counter() - self._start_perf
        event = {
            "type": event_type,
            "elapsed_seconds": round(elapsed, 3),
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.events.append(event)
        self._logger.info(f"[{event_type}] {kwargs}")

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a numerical metric.

        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step/iteration number.
        """
        extra = {"metric_name": name, "value": value}
        if step is not None:
            extra["step"] = step
        self.log_event("metric", **extra)

    def log_phase_start(self, phase: int, description: str = "") -> None:
        """Log the start of an experiment phase.

        Args:
            phase: Phase number (1-5).
            description: Optional description of the phase.
        """
        self.log_event("phase_start", phase=phase, description=description)

    def log_phase_end(self, phase: int, summary: dict[str, Any] | None = None) -> None:
        """Log the end of an experiment phase.

        Args:
            phase: Phase number (1-5).
            summary: Optional summary metrics for the completed phase.
        """
        self.log_event("phase_end", phase=phase, summary=summary or {})

    def get_elapsed_time(self) -> float:
        """Get elapsed time since experiment start in seconds."""
        return time.perf_counter() - self._start_perf

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the experiment run.

        Returns:
            Dictionary containing experiment metadata and event counts.
        """
        return {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time,
            "elapsed_seconds": round(self.get_elapsed_time(), 3),
            "gpu_info": self.gpu_info,
            "total_events": len(self.events),
            "config": self.config_snapshot,
        }
