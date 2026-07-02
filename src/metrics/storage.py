"""Persistent storage for metrics using JSONL, Parquet, and CSV formats.

Provides a unified interface for reading and writing metrics at different
granularities (per-token, per-sequence, global).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import GlobalBenchmarkMetrics, PerSequenceMetrics, PerTokenMetrics

logger = logging.getLogger(__name__)


class MetricsStore:
    """Unified metrics storage backend.

    Manages file-based storage with appropriate formats for each metric type:
    - Per-token: JSONL (streaming append, high volume)
    - Per-sequence: Parquet (columnar, efficient for analysis)
    - Global: CSV (human-readable, append-friendly)

    Args:
        base_dir: Root directory for all metrics files.
    """

    # Subdirectory and file naming conventions
    PER_TOKEN_DIR = "per_token"
    PER_SEQUENCE_FILE = "per_sequence.parquet"
    GLOBAL_FILE = "global_metrics.csv"

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self._per_token_dir = self.base_dir / self.PER_TOKEN_DIR
        self._per_token_dir.mkdir(parents=True, exist_ok=True)

        self._per_sequence_path = self.base_dir / self.PER_SEQUENCE_FILE
        self._global_path = self.base_dir / self.GLOBAL_FILE

    def _get_per_token_path(self, task: str) -> Path:
        """Get the JSONL file path for a given task."""
        safe_task = task.replace("/", "_").replace("\\", "_")
        return self._per_token_dir / f"{safe_task}.jsonl"

    def append_per_token(self, metrics: PerTokenMetrics) -> None:
        """Append a single per-token metric to the appropriate JSONL file.

        Args:
            metrics: Per-token metrics to store.
        """
        path = self._get_per_token_path(metrics.task)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics.to_dict(), ensure_ascii=False) + "\n")
        except IOError as e:
            logger.error(f"Failed to append per-token metric: {e}")
            raise

    def append_per_token_batch(self, metrics_list: list[PerTokenMetrics]) -> None:
        """Append a batch of per-token metrics, grouped by task.

        Args:
            metrics_list: List of per-token metrics to store.
        """
        # Group by task for efficient file I/O
        by_task: dict[str, list[dict]] = {}
        for m in metrics_list:
            by_task.setdefault(m.task, []).append(m.to_dict())

        for task, records in by_task.items():
            path = self._get_per_token_path(task)
            try:
                with open(path, "a", encoding="utf-8") as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except IOError as e:
                logger.error(f"Failed to append per-token batch for task {task}: {e}")
                raise

    def save_per_sequence_batch(self, metrics_list: list[PerSequenceMetrics]) -> None:
        """Save per-sequence metrics to Parquet, appending to existing data.

        Args:
            metrics_list: List of per-sequence metrics to save.
        """
        if not metrics_list:
            return

        new_df = pd.DataFrame([m.to_dict() for m in metrics_list])

        # Convert mode_switch_points list to string for Parquet compatibility
        new_df["mode_switch_points"] = new_df["mode_switch_points"].apply(json.dumps)

        if self._per_sequence_path.exists():
            existing_df = pd.read_parquet(self._per_sequence_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        combined_df.to_parquet(self._per_sequence_path, index=False)
        logger.info(
            f"Saved {len(metrics_list)} per-sequence metrics "
            f"(total: {len(combined_df)} rows)"
        )

    def save_global_metrics(self, metrics: GlobalBenchmarkMetrics) -> None:
        """Append global benchmark metrics to the CSV file.

        Args:
            metrics: Global metrics for one benchmark run.
        """
        data = metrics.to_dict()
        # Flatten confidence_interval for CSV
        ci = data.pop("confidence_interval_95", [0.0, 0.0])
        data["ci_95_lower"] = ci[0] if isinstance(ci, (list, tuple)) else 0.0
        data["ci_95_upper"] = ci[1] if isinstance(ci, (list, tuple)) else 0.0

        file_exists = self._global_path.exists()

        try:
            with open(self._global_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(data.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)
        except IOError as e:
            logger.error(f"Failed to save global metrics: {e}")
            raise

        logger.info(f"Appended global metrics for {metrics.method}/{metrics.task}")

    def load_per_token_for_task(self, task: str) -> list[dict]:
        """Load all per-token metrics for a given task from JSONL.

        Args:
            task: The task name to load metrics for.

        Returns:
            List of metric dictionaries.
        """
        path = self._get_per_token_path(task)
        if not path.exists():
            logger.warning(f"No per-token data found for task: {task}")
            return []

        records: list[dict] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping malformed line {line_num} in {path}: {e}"
                        )
        except IOError as e:
            logger.error(f"Failed to read per-token data for task {task}: {e}")
            raise

        logger.info(f"Loaded {len(records)} per-token records for task: {task}")
        return records

    def load_per_sequence_as_dataframe(self) -> pd.DataFrame:
        """Load all per-sequence metrics as a pandas DataFrame.

        Returns:
            DataFrame with per-sequence metrics. Empty DataFrame if no data exists.
        """
        if not self._per_sequence_path.exists():
            logger.warning("No per-sequence data found.")
            return pd.DataFrame()

        df = pd.read_parquet(self._per_sequence_path)

        # Parse mode_switch_points back from JSON string
        if "mode_switch_points" in df.columns:
            df["mode_switch_points"] = df["mode_switch_points"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

        logger.info(f"Loaded {len(df)} per-sequence records")
        return df

    def load_global_metrics(self) -> pd.DataFrame:
        """Load all global benchmark metrics from CSV.

        Returns:
            DataFrame with global metrics. Empty DataFrame if no data exists.
        """
        if not self._global_path.exists():
            logger.warning("No global metrics data found.")
            return pd.DataFrame()

        df = pd.read_csv(self._global_path)
        logger.info(f"Loaded {len(df)} global metric records")
        return df

    def list_tasks(self) -> list[str]:
        """List all tasks that have per-token metrics recorded.

        Returns:
            List of task names.
        """
        tasks = []
        for path in self._per_token_dir.glob("*.jsonl"):
            tasks.append(path.stem)
        return sorted(tasks)
