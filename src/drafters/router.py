"""Adaptive mode router for DAHD speculative decoding.

This module provides a standalone routing logic component that can be used
independently of the full DAHDDraftModule for experimentation and analysis.
It supports three routing strategies: probe_only, ema_only, and hybrid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RoutingDecision:
    """A single routing decision record.

    Attributes:
        mode: Selected drafting mode ('parallel' or 'ar').
        k: Number of draft tokens to generate.
        difficulty_score: Computed difficulty score.
        probe_confidence: Probe confidence used for this decision.
        ema_value: EMA acceptance rate used for this decision.
        strategy: Routing strategy that produced this decision.
    """

    mode: str
    k: int
    difficulty_score: float
    probe_confidence: float
    ema_value: float
    strategy: str


class AdaptiveModeRouter:
    """Adaptive routing between AR and Parallel drafting modes.

    Supports three routing strategies:
    - probe_only: Route based solely on the difficulty probe's confidence.
    - ema_only: Route based solely on the EMA of acceptance rates.
    - hybrid: Weighted combination of probe confidence and EMA (default).

    The router also maintains a history of routing decisions for analysis
    and visualization purposes.
    """

    def __init__(
        self,
        strategy: Literal["probe_only", "ema_only", "hybrid"] = "hybrid",
        probe_weight: float = 0.6,
        ema_alpha: float = 0.3,
        easy_threshold: float = 0.7,
        hard_threshold: float = 0.5,
        draft_length_easy: int = 6,
        draft_length_hard: int = 3,
        draft_length_medium: int = 4,
        history_maxlen: int = 1000,
    ):
        """Initialize AdaptiveModeRouter.

        Args:
            strategy: Routing strategy to use.
            probe_weight: Weight for probe confidence (only used in 'hybrid' mode).
            ema_alpha: Smoothing factor for EMA update.
            easy_threshold: Score above which sample is considered easy.
            hard_threshold: Score below which sample is considered hard.
            draft_length_easy: Draft length for easy (parallel) mode.
            draft_length_hard: Draft length for hard (AR) mode.
            draft_length_medium: Draft length for medium difficulty.
            history_maxlen: Maximum number of routing decisions to store.
        """
        self.strategy = strategy
        self.probe_weight = probe_weight
        self.ema_alpha = ema_alpha
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold
        self.draft_length_easy = draft_length_easy
        self.draft_length_hard = draft_length_hard
        self.draft_length_medium = draft_length_medium
        self.history_maxlen = history_maxlen

        # Internal state
        self._ema_acceptance_rate: float = 0.5
        self._history: list[RoutingDecision] = []
        self._total_decisions: int = 0

    @property
    def ema_acceptance_rate(self) -> float:
        """Current EMA of acceptance rate."""
        return self._ema_acceptance_rate

    @property
    def history(self) -> list[RoutingDecision]:
        """Routing decision history (up to history_maxlen entries)."""
        return self._history

    @property
    def total_decisions(self) -> int:
        """Total number of routing decisions made (including those evicted from history)."""
        return self._total_decisions

    def route(
        self,
        probe_confidence: float,
        ema_acceptance_rate: float | None = None,
    ) -> tuple[str, int]:
        """Make a routing decision based on the configured strategy.

        Args:
            probe_confidence: Confidence from the difficulty probe (0 to 1).
            ema_acceptance_rate: Optional explicit EMA acceptance rate.
                If None, uses the internal EMA state.

        Returns:
            Tuple of (mode, k) where mode is 'parallel' or 'ar'.
        """
        ema_value = ema_acceptance_rate if ema_acceptance_rate is not None else self._ema_acceptance_rate

        # Compute difficulty score based on strategy
        if self.strategy == "probe_only":
            difficulty_score = probe_confidence
        elif self.strategy == "ema_only":
            difficulty_score = ema_value
        else:  # hybrid
            difficulty_score = (
                self.probe_weight * probe_confidence
                + (1.0 - self.probe_weight) * ema_value
            )

        # Select mode and k based on difficulty
        if difficulty_score >= self.easy_threshold:
            mode = "parallel"
            k = self.draft_length_easy
        elif difficulty_score <= self.hard_threshold:
            mode = "ar"
            k = self.draft_length_hard
        else:
            mode = "ar"
            k = self.draft_length_medium

        # Record decision in history
        decision = RoutingDecision(
            mode=mode,
            k=k,
            difficulty_score=difficulty_score,
            probe_confidence=probe_confidence,
            ema_value=ema_value,
            strategy=self.strategy,
        )
        self._history.append(decision)
        self._total_decisions += 1

        # Evict old entries if history exceeds maxlen
        if len(self._history) > self.history_maxlen:
            self._history = self._history[-self.history_maxlen:]

        return (mode, k)

    def update_ema(self, acceptance_rate: float) -> None:
        """Update the EMA of acceptance rate with a new observation.

        Args:
            acceptance_rate: The acceptance rate from the most recent verification.
        """
        self._ema_acceptance_rate = (
            self.ema_alpha * acceptance_rate
            + (1.0 - self.ema_alpha) * self._ema_acceptance_rate
        )

    def reset(self) -> None:
        """Reset internal state (EMA and history)."""
        self._ema_acceptance_rate = 0.5
        self._history.clear()
        self._total_decisions = 0

    def get_stats(self) -> dict[str, float | int | str]:
        """Get routing statistics from the history.

        Returns:
            Dictionary with routing statistics including mode distribution,
            average difficulty scores, and configuration.
        """
        if not self._history:
            return {
                "strategy": self.strategy,
                "total_decisions": 0,
                "parallel_fraction": 0.0,
                "ar_fraction": 0.0,
                "avg_difficulty_score": 0.0,
                "avg_probe_confidence": 0.0,
                "avg_ema_value": 0.0,
                "avg_k": 0.0,
            }

        parallel_count = sum(1 for d in self._history if d.mode == "parallel")
        ar_count = len(self._history) - parallel_count

        avg_difficulty = sum(d.difficulty_score for d in self._history) / len(self._history)
        avg_probe = sum(d.probe_confidence for d in self._history) / len(self._history)
        avg_ema = sum(d.ema_value for d in self._history) / len(self._history)
        avg_k = sum(d.k for d in self._history) / len(self._history)

        return {
            "strategy": self.strategy,
            "total_decisions": self._total_decisions,
            "parallel_fraction": parallel_count / len(self._history),
            "ar_fraction": ar_count / len(self._history),
            "avg_difficulty_score": avg_difficulty,
            "avg_probe_confidence": avg_probe,
            "avg_ema_value": avg_ema,
            "avg_k": avg_k,
        }
