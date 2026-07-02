"""Abstract base classes for speculative decoding drafters."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn


class DraftOutput(NamedTuple):
    """Output from a draft generation step.

    Attributes:
        draft_tokens: Predicted draft token IDs, shape (batch, k).
        draft_logits: Logits for each draft position, shape (batch, k, vocab_size).
        draft_latency_ms: Time taken for draft generation in milliseconds.
    """

    draft_tokens: torch.Tensor
    draft_logits: torch.Tensor
    draft_latency_ms: float


class VerifyResult(NamedTuple):
    """Result from verifying draft tokens against the target model.

    Attributes:
        accepted_tokens: The accepted token IDs, shape (batch, accepted_length).
        accepted_length: Number of accepted tokens per sample, shape (batch,).
        acceptance_rate: Fraction of draft tokens accepted, shape (batch,).
        verify_latency_ms: Time taken for verification in milliseconds.
    """

    accepted_tokens: torch.Tensor
    accepted_length: torch.Tensor
    acceptance_rate: torch.Tensor
    verify_latency_ms: float


class SpeculativeDrafter(nn.Module, ABC):
    """Abstract base class for speculative decoding drafters.

    All drafter implementations (Medusa, EAGLE, DAHD, etc.) should inherit
    from this class and implement the `draft` and `verify` methods.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, max_draft_length: int = 8):
        """Initialize the speculative drafter.

        Args:
            hidden_dim: Dimension of hidden states from the target model.
            vocab_size: Size of the token vocabulary.
            max_draft_length: Maximum number of tokens that can be drafted.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_draft_length = max_draft_length

    @abstractmethod
    def draft(self, hidden_states: torch.Tensor, k: int) -> DraftOutput:
        """Generate k draft tokens from the target model's hidden states.

        Args:
            hidden_states: Hidden states from the last layer of the target model,
                shape (batch, seq_len, hidden_dim). Typically the last token's
                hidden state is used: (batch, 1, hidden_dim).
            k: Number of draft tokens to generate.

        Returns:
            DraftOutput containing the generated draft tokens, logits, and latency.
        """
        ...

    @abstractmethod
    def verify(
        self,
        draft_tokens: torch.Tensor,
        target_model: nn.Module,
        input_ids: torch.Tensor,
    ) -> VerifyResult:
        """Verify draft tokens against the target model using speculative sampling.

        This performs the standard speculative decoding verification:
        1. Run target model on input_ids + draft_tokens in a single forward pass
        2. Compare target model logits with draft logits at each position
        3. Accept tokens that pass the acceptance criterion

        Args:
            draft_tokens: The draft token IDs to verify, shape (batch, k).
            target_model: The target language model.
            input_ids: Original input token IDs, shape (batch, seq_len).

        Returns:
            VerifyResult containing accepted tokens, lengths, rates, and latency.
        """
        ...

    def timed_draft(self, hidden_states: torch.Tensor, k: int) -> DraftOutput:
        """Draft with explicit CUDA timing using events.

        Args:
            hidden_states: Hidden states from the target model.
            k: Number of draft tokens to generate.

        Returns:
            DraftOutput with accurate GPU-timed latency measurement.
        """
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        output = self.draft(hidden_states, k)
        end_event.record()

        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)

        return DraftOutput(
            draft_tokens=output.draft_tokens,
            draft_logits=output.draft_logits,
            draft_latency_ms=latency_ms,
        )
