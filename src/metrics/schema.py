"""Schema definitions for DAHD speculative decoding metrics.

Defines dataclasses for per-token, per-sequence, and global benchmark metrics
used throughout the metrics collection and analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PerTokenMetrics:
    """Metrics collected for each individual drafted/verified token.

    Attributes:
        token_id: The vocabulary ID of the token.
        token_pos: Position within the draft window (0 ~ draft_length-1).
        task: The benchmark task name (e.g., "gsm8k", "humaneval").
        sequence_id: Unique identifier for the parent sequence.
        probe_confidence: Confidence score from the difficulty probe [0, 1].
        input_hidden_entropy: Entropy of the input hidden state.
        is_accepted: Whether this drafted token was accepted by the verifier.
        draft_mode: The drafting mode used ("parallel" or "ar").
        draft_latency_ms: Time taken to draft this token in milliseconds.
        timestamp: Unix timestamp when this metric was recorded.
        device: CUDA device identifier (e.g., "cuda:0").
    """

    token_id: int
    token_pos: int  # 0 ~ draft_length-1
    task: str
    sequence_id: str
    probe_confidence: float  # [0, 1]
    input_hidden_entropy: float
    is_accepted: bool
    draft_mode: str  # "parallel" or "ar"
    draft_latency_ms: float
    timestamp: float
    device: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "token_id": self.token_id,
            "token_pos": self.token_pos,
            "task": self.task,
            "sequence_id": self.sequence_id,
            "probe_confidence": self.probe_confidence,
            "input_hidden_entropy": self.input_hidden_entropy,
            "is_accepted": self.is_accepted,
            "draft_mode": self.draft_mode,
            "draft_latency_ms": self.draft_latency_ms,
            "timestamp": self.timestamp,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerTokenMetrics":
        """Construct from dictionary."""
        return cls(
            token_id=data["token_id"],
            token_pos=data["token_pos"],
            task=data["task"],
            sequence_id=data["sequence_id"],
            probe_confidence=data["probe_confidence"],
            input_hidden_entropy=data["input_hidden_entropy"],
            is_accepted=data["is_accepted"],
            draft_mode=data["draft_mode"],
            draft_latency_ms=data["draft_latency_ms"],
            timestamp=data["timestamp"],
            device=data.get("device", ""),
        )


@dataclass
class PerSequenceMetrics:
    """Aggregated metrics for a complete generation sequence.

    Attributes:
        sequence_id: Unique identifier for this sequence.
        task: The benchmark task name.
        total_tokens_generated: Total tokens in the final output.
        total_tokens_drafted: Total tokens proposed by the drafter.
        total_tokens_accepted: Total tokens accepted by the verifier.
        acceptance_rate: Ratio of accepted to drafted tokens.
        avg_probe_confidence: Mean probe confidence across all tokens.
        num_easy_tokens: Count of tokens with probe_conf > 0.7.
        num_hard_tokens: Count of tokens with probe_conf < 0.5.
        avg_draft_latency_ms: Mean draft latency per token.
        avg_verify_latency_ms: Mean verification latency per step.
        total_e2e_latency_ms: End-to-end latency for the full sequence.
        num_ar_rounds: Number of autoregressive drafting rounds.
        num_parallel_rounds: Number of parallel drafting rounds.
        mode_switch_points: Token positions where mode switched.
        is_correct: Whether the final answer is correct (task-specific).
        task_metric: Task-specific quality metric value.
    """

    sequence_id: str
    task: str
    total_tokens_generated: int
    total_tokens_drafted: int
    total_tokens_accepted: int
    acceptance_rate: float
    avg_probe_confidence: float
    num_easy_tokens: int  # probe_conf > 0.7
    num_hard_tokens: int  # probe_conf < 0.5
    avg_draft_latency_ms: float
    avg_verify_latency_ms: float
    total_e2e_latency_ms: float
    num_ar_rounds: int
    num_parallel_rounds: int
    mode_switch_points: list[int] = field(default_factory=list)
    is_correct: bool = False
    task_metric: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sequence_id": self.sequence_id,
            "task": self.task,
            "total_tokens_generated": self.total_tokens_generated,
            "total_tokens_drafted": self.total_tokens_drafted,
            "total_tokens_accepted": self.total_tokens_accepted,
            "acceptance_rate": self.acceptance_rate,
            "avg_probe_confidence": self.avg_probe_confidence,
            "num_easy_tokens": self.num_easy_tokens,
            "num_hard_tokens": self.num_hard_tokens,
            "avg_draft_latency_ms": self.avg_draft_latency_ms,
            "avg_verify_latency_ms": self.avg_verify_latency_ms,
            "total_e2e_latency_ms": self.total_e2e_latency_ms,
            "num_ar_rounds": self.num_ar_rounds,
            "num_parallel_rounds": self.num_parallel_rounds,
            "mode_switch_points": self.mode_switch_points,
            "is_correct": self.is_correct,
            "task_metric": self.task_metric,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerSequenceMetrics":
        """Construct from dictionary."""
        return cls(
            sequence_id=data["sequence_id"],
            task=data["task"],
            total_tokens_generated=data["total_tokens_generated"],
            total_tokens_drafted=data["total_tokens_drafted"],
            total_tokens_accepted=data["total_tokens_accepted"],
            acceptance_rate=data["acceptance_rate"],
            avg_probe_confidence=data["avg_probe_confidence"],
            num_easy_tokens=data["num_easy_tokens"],
            num_hard_tokens=data["num_hard_tokens"],
            avg_draft_latency_ms=data["avg_draft_latency_ms"],
            avg_verify_latency_ms=data["avg_verify_latency_ms"],
            total_e2e_latency_ms=data["total_e2e_latency_ms"],
            num_ar_rounds=data["num_ar_rounds"],
            num_parallel_rounds=data["num_parallel_rounds"],
            mode_switch_points=data.get("mode_switch_points", []),
            is_correct=data.get("is_correct", False),
            task_metric=data.get("task_metric", 0.0),
        )


@dataclass
class GlobalBenchmarkMetrics:
    """Global metrics aggregated across all sequences for a benchmark run.

    Attributes:
        benchmark_id: Unique identifier for this benchmark run.
        timestamp: ISO format timestamp of the run.
        model: Model name/path used.
        method: Speculative decoding method ("dahd", "parallel", "eagle", etc.).
        task: Benchmark task name.
        avg_latency_p50_ms: 50th percentile latency.
        avg_latency_p95_ms: 95th percentile latency.
        throughput_tokens_per_sec: Tokens generated per second.
        acceptance_rate_global: Global acceptance rate.
        acceptance_length_mean: Mean accepted draft length.
        speedup_vs_ar: Speedup compared to autoregressive baseline.
        pass_at_1: Pass@1 metric (for code generation tasks).
        exact_match: Exact match accuracy.
        bleu_score: BLEU score (for translation/summarization).
        num_samples: Number of samples evaluated.
        confidence_interval_95: 95% CI for the primary metric.
    """

    benchmark_id: str
    timestamp: str
    model: str
    method: str  # "dahd", "parallel", "eagle", etc.
    task: str
    avg_latency_p50_ms: float
    avg_latency_p95_ms: float
    throughput_tokens_per_sec: float
    acceptance_rate_global: float
    acceptance_length_mean: float
    speedup_vs_ar: float
    pass_at_1: float = 0.0
    exact_match: float = 0.0
    bleu_score: float = 0.0
    num_samples: int = 0
    confidence_interval_95: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_id": self.benchmark_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "method": self.method,
            "task": self.task,
            "avg_latency_p50_ms": self.avg_latency_p50_ms,
            "avg_latency_p95_ms": self.avg_latency_p95_ms,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "acceptance_rate_global": self.acceptance_rate_global,
            "acceptance_length_mean": self.acceptance_length_mean,
            "speedup_vs_ar": self.speedup_vs_ar,
            "pass_at_1": self.pass_at_1,
            "exact_match": self.exact_match,
            "bleu_score": self.bleu_score,
            "num_samples": self.num_samples,
            "confidence_interval_95": list(self.confidence_interval_95),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalBenchmarkMetrics":
        """Construct from dictionary."""
        ci = data.get("confidence_interval_95", (0.0, 0.0))
        if isinstance(ci, list):
            ci = tuple(ci)
        return cls(
            benchmark_id=data["benchmark_id"],
            timestamp=data["timestamp"],
            model=data["model"],
            method=data["method"],
            task=data["task"],
            avg_latency_p50_ms=data["avg_latency_p50_ms"],
            avg_latency_p95_ms=data["avg_latency_p95_ms"],
            throughput_tokens_per_sec=data["throughput_tokens_per_sec"],
            acceptance_rate_global=data["acceptance_rate_global"],
            acceptance_length_mean=data["acceptance_length_mean"],
            speedup_vs_ar=data["speedup_vs_ar"],
            pass_at_1=data.get("pass_at_1", 0.0),
            exact_match=data.get("exact_match", 0.0),
            bleu_score=data.get("bleu_score", 0.0),
            num_samples=data.get("num_samples", 0),
            confidence_interval_95=ci,
        )
