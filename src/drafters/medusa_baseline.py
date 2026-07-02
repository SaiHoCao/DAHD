"""Medusa baseline drafter implementation.

Medusa uses multiple independent MLP heads to predict future tokens in parallel.
Each head is responsible for predicting the token at a specific future position,
all operating on the same hidden state from the target model.

Reference: Cai et al., "Medusa: Simple LLM Inference Acceleration Framework
with Multiple Decoding Heads" (2024).
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.drafters.base import SpeculativeDrafter, DraftOutput, VerifyResult


class MedusaHead(nn.Module):
    """Single Medusa prediction head.

    A 2-layer MLP with residual connection for predicting the token
    at a specific future position.
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        """Initialize a Medusa head.

        Args:
            hidden_dim: Dimension of the input hidden state.
            vocab_size: Size of the output vocabulary.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict logits for this position.

        Args:
            hidden_states: Input hidden states, shape (batch, hidden_dim).

        Returns:
            Logits for the predicted token, shape (batch, vocab_size).
        """
        residual = hidden_states
        out = self.mlp(hidden_states)
        out = self.layer_norm(out + residual)
        logits = self.lm_head(out)
        return logits


class MedusaBaseline(SpeculativeDrafter):
    """Medusa-style parallel drafting baseline.

    All K draft tokens are predicted independently and in parallel,
    each by its own MLP head operating on the same target model hidden state.
    This provides low latency but may have lower acceptance rates for
    difficult tokens since there is no inter-token dependency modeling.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        max_draft_length: int = 8,
    ):
        """Initialize MedusaBaseline.

        Args:
            hidden_dim: Dimension of the target model's hidden states.
            vocab_size: Size of the token vocabulary.
            max_draft_length: Maximum number of parallel prediction heads (K).
        """
        super().__init__(hidden_dim, vocab_size, max_draft_length)

        # Create K independent prediction heads
        self.heads = nn.ModuleList([
            MedusaHead(hidden_dim, vocab_size)
            for _ in range(max_draft_length)
        ])

    def draft(self, hidden_states: torch.Tensor, k: int) -> DraftOutput:
        """Generate k draft tokens in parallel using independent heads.

        Args:
            hidden_states: Hidden states from the target model,
                shape (batch, seq_len, hidden_dim). Uses the last position.
            k: Number of draft tokens to generate (must be <= max_draft_length).

        Returns:
            DraftOutput with parallel-predicted draft tokens and logits.
        """
        start_time = time.perf_counter()

        # Use the last position's hidden state
        hidden = hidden_states[:, -1, :]  # (batch, hidden_dim)

        # Generate predictions from each head in parallel
        k_actual = min(k, self.max_draft_length)
        logits_list: list[torch.Tensor] = []

        for i in range(k_actual):
            logits = self.heads[i](hidden)  # (batch, vocab_size)
            logits_list.append(logits)

        draft_logits = torch.stack(logits_list, dim=1)  # (batch, k, vocab_size)
        draft_tokens = draft_logits.argmax(dim=-1)  # (batch, k)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0

        return DraftOutput(
            draft_tokens=draft_tokens,
            draft_logits=draft_logits,
            draft_latency_ms=latency_ms,
        )

    def verify(
        self,
        draft_tokens: torch.Tensor,
        target_model: nn.Module,
        input_ids: torch.Tensor,
    ) -> VerifyResult:
        """Verify draft tokens against the target model.

        Performs standard speculative decoding verification:
        1. Concatenate input_ids with draft_tokens
        2. Run target model forward pass on the full sequence
        3. Compare target logits with draft predictions
        4. Accept tokens greedily until first mismatch

        Args:
            draft_tokens: Draft token IDs, shape (batch, k).
            target_model: The target language model with a forward method.
            input_ids: Original input token IDs, shape (batch, seq_len).

        Returns:
            VerifyResult with acceptance information.
        """
        start_time = time.perf_counter()

        batch_size, k = draft_tokens.shape
        device = draft_tokens.device

        # Concatenate input with draft tokens for target model forward pass
        full_input = torch.cat([input_ids, draft_tokens], dim=1)

        # Run target model
        with torch.no_grad():
            target_outputs = target_model(full_input)
            # Get logits corresponding to draft positions
            # target_logits at position i predicts token at position i+1
            seq_len = input_ids.shape[1]
            target_logits = target_outputs.logits[:, seq_len - 1 : seq_len - 1 + k, :]

        # Greedy verification: accept while target agrees with draft
        target_tokens = target_logits.argmax(dim=-1)  # (batch, k)
        matches = (target_tokens == draft_tokens)  # (batch, k)

        # Find first mismatch position for each sample in batch
        accepted_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        for b in range(batch_size):
            for pos in range(k):
                if matches[b, pos]:
                    accepted_lengths[b] += 1
                else:
                    break

        # Gather accepted tokens
        max_accepted = accepted_lengths.max().item()
        accepted_tokens = draft_tokens[:, :max_accepted] if max_accepted > 0 else torch.zeros(
            batch_size, 0, dtype=torch.long, device=device
        )

        acceptance_rate = accepted_lengths.float() / k

        end_time = time.perf_counter()
        verify_latency_ms = (end_time - start_time) * 1000.0

        return VerifyResult(
            accepted_tokens=accepted_tokens,
            accepted_length=accepted_lengths,
            acceptance_rate=acceptance_rate,
            verify_latency_ms=verify_latency_ms,
        )
