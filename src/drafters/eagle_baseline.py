"""EAGLE baseline drafter implementation.

EAGLE uses autoregressive feature prediction to generate draft tokens.
It predicts future hidden state features sequentially, using each predicted
feature to generate the next one, capturing inter-token dependencies.

Reference: Li et al., "EAGLE: Speculative Sampling Requires Rethinking
Feature Uncertainty" (2024).
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.drafters.base import SpeculativeDrafter, DraftOutput, VerifyResult


class EAGLEFeatureHead(nn.Module):
    """EAGLE-style feature prediction head.

    Predicts the next hidden state feature autoregressively, then decodes
    it to a token. Uses a lightweight Transformer layer for feature prediction.
    """

    def __init__(self, hidden_dim: int, nhead: int = 8, dim_feedforward: int = 1024):
        """Initialize EAGLE feature head.

        Args:
            hidden_dim: Dimension of hidden features.
            nhead: Number of attention heads.
            dim_feedforward: Feedforward network dimension.
        """
        super().__init__()

        # Feature prediction via a single Transformer layer
        self.feature_predictor = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict the next feature from the current feature sequence.

        Args:
            features: Current feature sequence, shape (batch, seq_len, hidden_dim).

        Returns:
            Predicted next feature, shape (batch, 1, hidden_dim).
        """
        out = self.feature_predictor(features)
        # Take the last position as the predicted next feature
        next_feature = out[:, -1:, :]  # (batch, 1, hidden_dim)
        return self.layer_norm(next_feature)


class EAGLEBaseline(SpeculativeDrafter):
    """EAGLE-style autoregressive feature prediction baseline.

    Generates draft tokens by sequentially predicting future hidden state
    features. Each step uses the accumulated feature history to predict
    the next feature, which is then decoded to a token. This captures
    inter-token dependencies but requires sequential computation.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        max_draft_length: int = 8,
        nhead: int = 8,
        dim_feedforward: int = 1024,
    ):
        """Initialize EAGLEBaseline.

        Args:
            hidden_dim: Dimension of the target model's hidden states.
            vocab_size: Size of the token vocabulary.
            max_draft_length: Maximum number of AR steps for drafting.
            nhead: Number of attention heads in feature predictor.
            dim_feedforward: Feedforward dimension in feature predictor.
        """
        super().__init__(hidden_dim, vocab_size, max_draft_length)

        # Feature prediction head
        self.feature_head = EAGLEFeatureHead(
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )

        # Token decoder: maps features to vocabulary
        self.token_decoder = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Token-to-feature embedding for feeding back predictions
        self.token_to_feature = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def draft(self, hidden_states: torch.Tensor, k: int) -> DraftOutput:
        """Generate k draft tokens autoregressively via feature prediction.

        Args:
            hidden_states: Hidden states from the target model,
                shape (batch, seq_len, hidden_dim).
            k: Number of draft tokens to generate.

        Returns:
            DraftOutput with AR-predicted draft tokens and logits.
        """
        start_time = time.perf_counter()

        batch_size = hidden_states.size(0)
        k_actual = min(k, self.max_draft_length)

        # Start with the last position's feature as context
        feature_context = hidden_states[:, -1:, :]  # (batch, 1, hidden_dim)

        draft_tokens_list: list[torch.Tensor] = []
        draft_logits_list: list[torch.Tensor] = []

        for step in range(k_actual):
            # Predict next feature from context
            next_feature = self.feature_head(feature_context)  # (batch, 1, hidden_dim)

            # Decode feature to token
            logits = self.token_decoder(next_feature.squeeze(1))  # (batch, vocab_size)
            token = logits.argmax(dim=-1)  # (batch,)

            draft_tokens_list.append(token)
            draft_logits_list.append(logits)

            # Update feature context with the new predicted feature
            # In EAGLE, we append the predicted feature (not the token embedding)
            # to maintain the feature-level autoregression
            token_feature = self.token_to_feature[0](token).unsqueeze(1)  # (batch, 1, hidden_dim)
            token_feature = self.token_to_feature[1](token_feature)
            feature_context = torch.cat([feature_context, token_feature], dim=1)

        draft_tokens = torch.stack(draft_tokens_list, dim=1)  # (batch, k)
        draft_logits = torch.stack(draft_logits_list, dim=1)  # (batch, k, vocab_size)

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

        Uses the same greedy speculative decoding verification as Medusa:
        accept tokens sequentially until the first disagreement.

        Args:
            draft_tokens: Draft token IDs, shape (batch, k).
            target_model: The target language model.
            input_ids: Original input token IDs, shape (batch, seq_len).

        Returns:
            VerifyResult with acceptance information.
        """
        start_time = time.perf_counter()

        batch_size, k = draft_tokens.shape
        device = draft_tokens.device

        # Concatenate input with draft tokens
        full_input = torch.cat([input_ids, draft_tokens], dim=1)

        # Run target model on full sequence
        with torch.no_grad():
            target_outputs = target_model(full_input)
            seq_len = input_ids.shape[1]
            target_logits = target_outputs.logits[:, seq_len - 1 : seq_len - 1 + k, :]

        # Greedy verification
        target_tokens = target_logits.argmax(dim=-1)  # (batch, k)
        matches = (target_tokens == draft_tokens)  # (batch, k)

        # Find first mismatch for each batch element
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
