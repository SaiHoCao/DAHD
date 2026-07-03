"""DAHD (Difficulty-Adaptive Hybrid Drafting) draft module — 参考/探索性架构.

NOTE: 这是一个参考/探索性架构实现。论文中报告的 DAHD v2 端到端结果
使用的是 experiments/phase4_e2e/run_e2e_comparison.py 中的 SpecDecEngine。
本模块提供可复用的模块化组件，适用于未来的工程集成。

Components:
- SharedBottomLayer: Shared feature extraction using small Transformer layers
- ARBranch: Autoregressive drafting branch for hard samples
- ParallelBranch: Parallel drafting branch (Gumiho-style) for easy samples
- DifficultyProbe: Lightweight probe for estimating next-token difficulty
- DifficultyRouter: Routes between AR and Parallel branches based on difficulty
- DAHDDraftModule: The complete draft module combining all components
"""

from __future__ import annotations

import time
from typing import NamedTuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAHDDraftOutput(NamedTuple):
    """Output from the DAHD draft module.

    Attributes:
        draft_tokens: Predicted draft token IDs, shape (batch, k).
        difficulty_score: Computed difficulty score, shape (batch,).
        selected_mode: The drafting mode selected ('parallel' or 'ar').
        draft_k: Number of draft tokens generated.
        draft_latency_ms: Time taken for draft generation in milliseconds.
    """

    draft_tokens: torch.Tensor
    difficulty_score: torch.Tensor
    selected_mode: str
    draft_k: int
    draft_latency_ms: float


class SharedBottomLayer(nn.Module):
    """Shared bottom layer for feature extraction.

    Uses 1-2 lightweight Transformer encoder layers to process the hidden states
    from the target model into a shared representation used by both the AR
    and Parallel branches.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        """Initialize SharedBottomLayer.

        Args:
            hidden_dim: Input and output dimension of hidden states.
            num_layers: Number of Transformer encoder layers (1 or 2).
            nhead: Number of attention heads.
            dim_feedforward: Dimension of feedforward network.
            dropout: Dropout rate.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract shared representation from target model hidden states.

        Args:
            hidden_states: Hidden states from target model, shape (batch, seq_len, hidden_dim).

        Returns:
            Shared representation tensor, shape (batch, seq_len, hidden_dim).
        """
        out = self.encoder(hidden_states)
        return self.layer_norm(out)


class ARBranch(nn.Module):
    """Autoregressive drafting branch.

    Generates draft tokens sequentially, where each token prediction depends
    on the previous one. Suitable for hard samples where accuracy matters more.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, max_k: int = 8):
        """Initialize ARBranch.

        Args:
            hidden_dim: Dimension of the hidden representation.
            vocab_size: Size of the output vocabulary.
            max_k: Maximum number of tokens that can be generated.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_k = max_k

        # Prediction head: maps hidden states to vocabulary logits
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Embedding layer: maps predicted tokens back to hidden space for next step
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Recurrent update: combines current hidden state with new token embedding
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Layer norm for stability at each AR step
        self.step_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, shared_repr: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate k tokens autoregressively.

        Args:
            shared_repr: Shared representation from bottom layer,
                shape (batch, 1, hidden_dim). Uses the last token position.
            k: Number of tokens to generate.

        Returns:
            Tuple of:
                - draft_tokens: shape (batch, k)
                - draft_logits: shape (batch, k, vocab_size)
        """
        batch_size = shared_repr.size(0)
        device = shared_repr.device

        # Use the last position as the initial hidden state
        hidden = shared_repr[:, -1, :]  # (batch, hidden_dim)

        draft_tokens_list: list[torch.Tensor] = []
        draft_logits_list: list[torch.Tensor] = []

        for step in range(min(k, self.max_k)):
            # Predict next token
            logits = self.lm_head(hidden)  # (batch, vocab_size)
            token = logits.argmax(dim=-1)  # (batch,)

            draft_tokens_list.append(token)
            draft_logits_list.append(logits)

            # If not the last step, update hidden state for next prediction
            if step < k - 1:
                token_emb = self.token_embedding(token)  # (batch, hidden_dim)
                combined = torch.cat([hidden, token_emb], dim=-1)  # (batch, 2*hidden_dim)
                hidden = self.update_gate(combined)  # (batch, hidden_dim)
                hidden = self.step_norm(hidden)

        draft_tokens = torch.stack(draft_tokens_list, dim=1)  # (batch, k)
        draft_logits = torch.stack(draft_logits_list, dim=1)  # (batch, k, vocab_size)

        return draft_tokens, draft_logits


class ParallelBranch(nn.Module):
    """Parallel drafting branch (Gumiho-style).

    Generates all k draft tokens simultaneously using independent prediction heads.
    Suitable for easy samples where parallelism provides speed benefits.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, max_k: int = 8):
        """Initialize ParallelBranch.

        Args:
            hidden_dim: Dimension of the hidden representation.
            vocab_size: Size of the output vocabulary.
            max_k: Maximum number of tokens that can be generated in parallel.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_k = max_k

        # K independent prediction heads, one for each draft position
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, vocab_size),
            )
            for _ in range(max_k)
        ])

    def forward(self, shared_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate all draft tokens in parallel.

        Args:
            shared_repr: Shared representation from bottom layer,
                shape (batch, 1, hidden_dim) or (batch, seq_len, hidden_dim).
                Uses the last token position.

        Returns:
            Tuple of:
                - draft_tokens: shape (batch, max_k)
                - draft_logits: shape (batch, max_k, vocab_size)
        """
        # Use the last position's hidden state
        hidden = shared_repr[:, -1, :]  # (batch, hidden_dim)

        all_logits: list[torch.Tensor] = []
        for head in self.heads:
            logits = head(hidden)  # (batch, vocab_size)
            all_logits.append(logits)

        draft_logits = torch.stack(all_logits, dim=1)  # (batch, max_k, vocab_size)
        draft_tokens = draft_logits.argmax(dim=-1)  # (batch, max_k)

        return draft_tokens, draft_logits


class DifficultyProbe(nn.Module):
    """Lightweight difficulty probe.

    Predicts the next token and estimates difficulty based on the confidence
    (max probability) of the prediction. High confidence suggests an easy
    token, while low confidence suggests a hard one.
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        """Initialize DifficultyProbe.

        Args:
            hidden_dim: Dimension of the hidden representation.
            vocab_size: Size of the output vocabulary.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Single linear head for next-token prediction
        self.probe_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, vocab_size),
        )

    def forward(self, shared_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next token and estimate difficulty confidence.

        Args:
            shared_repr: Shared representation from bottom layer,
                shape (batch, seq_len, hidden_dim). Uses the last token position.

        Returns:
            Tuple of:
                - probe_token: Predicted next token ID, shape (batch,).
                - probe_confidence: Max probability (confidence), shape (batch,).
        """
        hidden = shared_repr[:, -1, :]  # (batch, hidden_dim)
        logits = self.probe_head(hidden)  # (batch, vocab_size)

        probs = F.softmax(logits, dim=-1)  # (batch, vocab_size)
        probe_confidence, probe_token = probs.max(dim=-1)  # both (batch,)

        return probe_token, probe_confidence


class DifficultyRouter:
    """Routes between AR and Parallel branches based on difficulty estimation.

    Combines probe confidence with an EMA of recent acceptance rates to
    compute a difficulty score, then selects the appropriate drafting mode
    and draft length.
    """

    def __init__(
        self,
        probe_weight: float = 0.6,
        ema_alpha: float = 0.3,
        easy_threshold: float = 0.7,
        hard_threshold: float = 0.5,
        draft_length_easy: int = 6,
        draft_length_hard: int = 3,
        draft_length_medium: int = 4,
    ):
        """Initialize DifficultyRouter.

        Args:
            probe_weight: Weight for probe confidence in difficulty computation.
                EMA weight is (1 - probe_weight).
            ema_alpha: Smoothing factor for EMA update (higher = more responsive).
            easy_threshold: Score above which the sample is considered easy.
            hard_threshold: Score below which the sample is considered hard.
            draft_length_easy: Draft length k for easy samples (parallel mode).
            draft_length_hard: Draft length k for hard samples (AR mode).
            draft_length_medium: Draft length k for medium samples (AR mode).
        """
        self.probe_weight = probe_weight
        self.ema_weight = 1.0 - probe_weight
        self.ema_alpha = ema_alpha
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold
        self.draft_length_easy = draft_length_easy
        self.draft_length_hard = draft_length_hard
        self.draft_length_medium = draft_length_medium

        # EMA state for acceptance rate tracking
        self._ema_acceptance_rate: float = 0.5  # Initialize at neutral

    @property
    def ema_acceptance_rate(self) -> float:
        """Current EMA of acceptance rate."""
        return self._ema_acceptance_rate

    def compute_difficulty(
        self, probe_confidence: float, recent_acceptance_rate: float | None = None
    ) -> float:
        """Compute difficulty score combining probe and EMA signals.

        The difficulty score is a weighted combination of the probe's confidence
        and the EMA of recent acceptance rates. Higher score = easier.

        Args:
            probe_confidence: Confidence from the difficulty probe (0 to 1).
            recent_acceptance_rate: Optional explicit acceptance rate to use
                instead of the internal EMA state.

        Returns:
            Difficulty score in [0, 1]. Higher means easier.
        """
        ema_value = recent_acceptance_rate if recent_acceptance_rate is not None else self._ema_acceptance_rate
        score = self.probe_weight * probe_confidence + self.ema_weight * ema_value
        return float(score)

    def select_mode(self, difficulty: float) -> tuple[str, int]:
        """Select drafting mode and draft length based on difficulty score.

        Args:
            difficulty: Difficulty score from compute_difficulty().

        Returns:
            Tuple of (mode, k) where mode is 'parallel' or 'ar',
            and k is the number of tokens to draft.
        """
        if difficulty >= self.easy_threshold:
            return ("parallel", self.draft_length_easy)
        elif difficulty <= self.hard_threshold:
            return ("ar", self.draft_length_hard)
        else:
            return ("ar", self.draft_length_medium)

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
        """Reset the EMA state to neutral."""
        self._ema_acceptance_rate = 0.5


class DAHDDraftModule(nn.Module):
    """Complete Difficulty-Adaptive Hybrid Drafting module.

    This module combines all DAHD components:
    1. SharedBottomLayer for shared feature extraction
    2. DifficultyProbe for estimating next-token difficulty
    3. DifficultyRouter for selecting the drafting mode
    4. ARBranch for hard samples (sequential, higher accuracy)
    5. ParallelBranch for easy samples (parallel, lower latency)

    The key insight is that easy tokens (high confidence) can be drafted in
    parallel (Gumiho-style) for speed, while hard tokens benefit from
    autoregressive generation (EAGLE-style) for accuracy.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        max_k: int = 8,
        num_shared_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        probe_weight: float = 0.6,
        ema_alpha: float = 0.3,
        easy_threshold: float = 0.7,
        hard_threshold: float = 0.5,
        draft_length_easy: int = 6,
        draft_length_hard: int = 3,
        draft_length_medium: int = 4,
    ):
        """Initialize DAHDDraftModule.

        Args:
            hidden_dim: Dimension of hidden states from the target model.
            vocab_size: Size of the token vocabulary.
            max_k: Maximum number of draft tokens.
            num_shared_layers: Number of Transformer layers in shared bottom.
            nhead: Number of attention heads in shared bottom.
            dim_feedforward: Feedforward dimension in shared bottom.
            dropout: Dropout rate for shared bottom layers.
            probe_weight: Weight for probe confidence in routing.
            ema_alpha: EMA smoothing factor for acceptance rate tracking.
            easy_threshold: Difficulty threshold for parallel mode.
            hard_threshold: Difficulty threshold for hard (short AR) mode.
            draft_length_easy: Draft length for easy samples.
            draft_length_hard: Draft length for hard samples.
            draft_length_medium: Draft length for medium samples.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_k = max_k

        # Shared feature extraction
        self.shared_bottom = SharedBottomLayer(
            hidden_dim=hidden_dim,
            num_layers=num_shared_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Difficulty estimation
        self.difficulty_probe = DifficultyProbe(hidden_dim, vocab_size)

        # Routing logic (non-parametric)
        self.router = DifficultyRouter(
            probe_weight=probe_weight,
            ema_alpha=ema_alpha,
            easy_threshold=easy_threshold,
            hard_threshold=hard_threshold,
            draft_length_easy=draft_length_easy,
            draft_length_hard=draft_length_hard,
            draft_length_medium=draft_length_medium,
        )

        # Drafting branches
        self.ar_branch = ARBranch(hidden_dim, vocab_size, max_k)
        self.parallel_branch = ParallelBranch(hidden_dim, vocab_size, max_k)

    def forward(
        self,
        hidden_states: torch.Tensor,
        recent_acceptance_rate: float | None = None,
    ) -> DAHDDraftOutput:
        """Generate draft tokens using difficulty-adaptive hybrid drafting.

        The forward pass:
        1. Extract shared representation via SharedBottomLayer
        2. Estimate difficulty via DifficultyProbe
        3. Route to AR or Parallel branch via DifficultyRouter
        4. Generate draft tokens using the selected branch

        Args:
            hidden_states: Hidden states from the target model's last layer,
                shape (batch, seq_len, hidden_dim).
            recent_acceptance_rate: Optional explicit acceptance rate for routing.
                If None, uses the router's internal EMA state.

        Returns:
            DAHDDraftOutput with draft tokens, difficulty info, and timing.
        """
        start_time = time.perf_counter()

        # Step 1: Shared feature extraction
        shared_repr = self.shared_bottom(hidden_states)

        # Step 2: Difficulty estimation
        _probe_token, probe_confidence = self.difficulty_probe(shared_repr)

        # Step 3: Route to appropriate branch
        # Use mean confidence across batch for routing decision
        mean_confidence = probe_confidence.mean().item()
        difficulty_score = self.router.compute_difficulty(
            mean_confidence, recent_acceptance_rate
        )
        mode, k = self.router.select_mode(difficulty_score)

        # Step 4: Generate draft tokens
        if mode == "parallel":
            draft_tokens, _draft_logits = self.parallel_branch(shared_repr)
            # Trim to requested k
            draft_tokens = draft_tokens[:, :k]
        else:
            draft_tokens, _draft_logits = self.ar_branch(shared_repr, k)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0

        # Return difficulty score as a tensor for consistency
        batch_size = hidden_states.size(0)
        difficulty_tensor = torch.full(
            (batch_size,), difficulty_score, device=hidden_states.device
        )

        return DAHDDraftOutput(
            draft_tokens=draft_tokens,
            difficulty_score=difficulty_tensor,
            selected_mode=mode,
            draft_k=k,
            draft_latency_ms=latency_ms,
        )

    def update_acceptance_rate(self, acceptance_rate: float) -> None:
        """Update the router's EMA with a new acceptance rate observation.

        Should be called after each verification step to keep the difficulty
        routing adaptive.

        Args:
            acceptance_rate: Acceptance rate from the most recent verification.
        """
        self.router.update_ema(acceptance_rate)

    def reset_router(self) -> None:
        """Reset the router's EMA state to neutral."""
        self.router.reset()

    def get_routing_stats(self) -> dict[str, float]:
        """Get current routing statistics.

        Returns:
            Dictionary with current EMA acceptance rate and thresholds.
        """
        return {
            "ema_acceptance_rate": self.router.ema_acceptance_rate,
            "easy_threshold": self.router.easy_threshold,
            "hard_threshold": self.router.hard_threshold,
            "probe_weight": self.router.probe_weight,
        }
