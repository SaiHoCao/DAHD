"""Per-token and per-sequence metrics collection.

Provides buffered JSONL writing for per-token metrics and aggregation
logic for computing per-sequence summary statistics.
"""

from __future__ import annotations

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional

from .schema import PerTokenMetrics, PerSequenceMetrics

logger = logging.getLogger(__name__)


class PerTokenMetricsCollector:
    """Buffered collector that writes per-token metrics to JSONL files.

    Accumulates metrics in memory and flushes to disk when the buffer
    reaches capacity or when explicitly requested.

    Args:
        output_jsonl: Path to the output JSONL file.
        buffer_size: Number of records to buffer before auto-flushing.
    """

    def __init__(self, output_jsonl: str | Path, buffer_size: int = 1000):
        self.output_jsonl = Path(output_jsonl)
        self.buffer_size = buffer_size
        self._buffer: list[dict] = []
        self._total_written: int = 0

        # Ensure parent directory exists
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def record_token(
        self,
        token_id: int,
        token_pos: int,
        task: str,
        sequence_id: str,
        probe_confidence: float,
        input_hidden_entropy: float,
        is_accepted: bool,
        draft_mode: str,
        draft_latency_ms: float,
        device: str = "",
    ) -> None:
        """Record a single token metric entry.

        Args:
            token_id: Vocabulary ID of the token.
            token_pos: Position within the draft window.
            task: Benchmark task name.
            sequence_id: Parent sequence identifier.
            probe_confidence: Probe confidence score [0, 1].
            input_hidden_entropy: Hidden state entropy.
            is_accepted: Whether the token was accepted.
            draft_mode: "parallel" or "ar".
            draft_latency_ms: Draft latency in milliseconds.
            device: CUDA device string.
        """
        metric = PerTokenMetrics(
            token_id=token_id,
            token_pos=token_pos,
            task=task,
            sequence_id=sequence_id,
            probe_confidence=probe_confidence,
            input_hidden_entropy=input_hidden_entropy,
            is_accepted=is_accepted,
            draft_mode=draft_mode,
            draft_latency_ms=draft_latency_ms,
            timestamp=time.time(),
            device=device,
        )
        self._buffer.append(metric.to_dict())

        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def record_token_from_dataclass(self, metric: PerTokenMetrics) -> None:
        """Record a pre-constructed PerTokenMetrics object."""
        self._buffer.append(metric.to_dict())
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write all buffered metrics to the JSONL file and clear the buffer."""
        if not self._buffer:
            return

        try:
            with open(self.output_jsonl, "a", encoding="utf-8") as f:
                for record in self._buffer:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._total_written += len(self._buffer)
            logger.debug(
                f"Flushed {len(self._buffer)} records to {self.output_jsonl} "
                f"(total: {self._total_written})"
            )
        except IOError as e:
            logger.error(f"Failed to flush metrics to {self.output_jsonl}: {e}")
            raise
        finally:
            self._buffer.clear()

    @property
    def pending_count(self) -> int:
        """Number of records currently in the buffer."""
        return len(self._buffer)

    @property
    def total_written(self) -> int:
        """Total number of records written to disk."""
        return self._total_written

    def __del__(self) -> None:
        """Ensure remaining buffer is flushed on deletion."""
        try:
            self.flush()
        except Exception:
            pass


class PerSequenceMetricsAggregator:
    """Aggregates per-token data into per-sequence summary metrics.

    Tracks the state of a single generation sequence and computes
    summary statistics upon completion.
    """

    def __init__(self):
        self._sequence_id: Optional[str] = None
        self._task: Optional[str] = None
        self._prompt: Optional[str] = None
        self._tokens: list[dict] = []
        self._start_time: float = 0.0
        self._verify_latencies: list[float] = []

    def start_sequence(self, sequence_id: str, task: str, prompt: str = "") -> None:
        """Begin tracking a new sequence.

        Args:
            sequence_id: Unique identifier for this sequence.
            task: Benchmark task name.
            prompt: The input prompt (for reference).
        """
        self._sequence_id = sequence_id
        self._task = task
        self._prompt = prompt
        self._tokens = []
        self._start_time = time.time()
        self._verify_latencies = []

    def add_token(self, token_data: dict) -> None:
        """Add a token's metrics to the current sequence.

        Args:
            token_data: Dictionary with token-level metric fields.
                Expected keys: probe_confidence, is_accepted, draft_mode,
                draft_latency_ms, token_pos.
        """
        if self._sequence_id is None:
            raise RuntimeError("No active sequence. Call start_sequence() first.")
        self._tokens.append(token_data)

    def add_verify_latency(self, latency_ms: float) -> None:
        """Record a verification step latency.

        Args:
            latency_ms: Verification latency in milliseconds.
        """
        self._verify_latencies.append(latency_ms)

    def end_sequence(
        self, is_correct: bool = False, task_metric: float = 0.0
    ) -> PerSequenceMetrics:
        """Finalize the sequence and compute aggregated metrics.

        Args:
            is_correct: Whether the generation was correct.
            task_metric: Task-specific quality score.

        Returns:
            Aggregated PerSequenceMetrics for this sequence.
        """
        if self._sequence_id is None:
            raise RuntimeError("No active sequence. Call start_sequence() first.")

        end_time = time.time()
        total_e2e_latency_ms = (end_time - self._start_time) * 1000.0

        total_drafted = len(self._tokens)
        total_accepted = sum(1 for t in self._tokens if t.get("is_accepted", False))
        acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0

        # Probe confidence statistics
        confidences = [t.get("probe_confidence", 0.0) for t in self._tokens]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        num_easy = sum(1 for c in confidences if c > 0.7)
        num_hard = sum(1 for c in confidences if c < 0.5)

        # Latency statistics
        draft_latencies = [t.get("draft_latency_ms", 0.0) for t in self._tokens]
        avg_draft_latency = (
            sum(draft_latencies) / len(draft_latencies) if draft_latencies else 0.0
        )
        avg_verify_latency = (
            sum(self._verify_latencies) / len(self._verify_latencies)
            if self._verify_latencies
            else 0.0
        )

        # Mode counting and switch detection
        modes = [t.get("draft_mode", "ar") for t in self._tokens]
        num_ar = sum(1 for m in modes if m == "ar")
        num_parallel = sum(1 for m in modes if m == "parallel")

        # Detect mode switch points
        mode_switch_points: list[int] = []
        for i in range(1, len(modes)):
            if modes[i] != modes[i - 1]:
                mode_switch_points.append(i)

        # Total tokens generated = accepted tokens (in speculative decoding,
        # the accepted tokens form the final output)
        total_generated = total_accepted

        result = PerSequenceMetrics(
            sequence_id=self._sequence_id,
            task=self._task or "",
            total_tokens_generated=total_generated,
            total_tokens_drafted=total_drafted,
            total_tokens_accepted=total_accepted,
            acceptance_rate=acceptance_rate,
            avg_probe_confidence=avg_confidence,
            num_easy_tokens=num_easy,
            num_hard_tokens=num_hard,
            avg_draft_latency_ms=avg_draft_latency,
            avg_verify_latency_ms=avg_verify_latency,
            total_e2e_latency_ms=total_e2e_latency_ms,
            num_ar_rounds=num_ar,
            num_parallel_rounds=num_parallel,
            mode_switch_points=mode_switch_points,
            is_correct=is_correct,
            task_metric=task_metric,
        )

        # Reset state
        self._sequence_id = None
        self._task = None
        self._prompt = None
        self._tokens = []
        self._verify_latencies = []

        return result
