#!/usr/bin/env python3
"""
EAGLE-3 Full Speculative Decoding Profiling Script.

Implements the complete EAGLE-3 draft head (including attention layer) and runs
a proper speculative decoding loop to measure acceptance rates.

Architecture (from SGLang reference):
- Input: concat of last 3 aux hidden states from target model → fc → hidden_size
- Decoder: 1 layer with GQA attention (32 Q heads, 8 KV heads, head_dim=128)
  - Input layer: RMSNorm(embeds) concat RMSNorm(hidden) → 8192 dim input to QKV
  - Post-attention: RMSNorm → SwiGLU MLP
- Output: Final RMSNorm → lm_head (32000 draft vocab) → d2t mapping to target vocab
"""

import json
import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ProfilingConfig:
    # Model paths
    target_model_path: str = "/mnt/nas1/hf/Qwen3-8B/"
    eagle3_weights_path: str = "/mnt/nas1/hf/qwen3_8b_eagle3/pytorch_model.bin"
    eagle3_config_path: str = "/mnt/nas1/hf/qwen3_8b_eagle3/config.json"
    eval_data_path: str = "/mnt/nas1/hf/qwen3_8b_eagle3/eagle_data.jsonl"

    # Model architecture
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    draft_vocab_size: int = 32000
    target_vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    num_target_layers: int = 36

    # Aux hidden state layers (before layer i in 0-indexed)
    # For Qwen3-8B: layers 2, 18, 33
    aux_layer_indices: List[int] = field(default_factory=lambda: [2, 18, 33])

    # Speculative decoding params
    num_draft_steps: int = 5  # K
    max_new_tokens: int = 64
    num_samples: int = 10  # Start with 10 for validation
    max_prompt_len: int = 512

    # Output
    output_dir: str = "/data1/caoshuaihu.csh/workspace/dahd_speculative_decoding/results/phase1_results_v2/"

    # Device
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


# ============================================================================
# RMSNorm Implementation
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).to(input_dtype)


# ============================================================================
# Rotary Position Embedding
# ============================================================================

def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 1000000.0,
                          device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Precompute the frequency tensor for RoPE (HuggingFace/Qwen3 style: half rotation)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [seq_len, head_dim//2]
    # HF style: duplicate freqs for full head_dim
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims (HuggingFace style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                     positions: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings (HuggingFace/Qwen3 style: half rotation).
    
    x: [batch, n_heads, seq_len, head_dim]
    positions: [seq_len] or [batch, seq_len]
    cos, sin: [max_seq_len, head_dim]
    """
    if positions.dim() == 1:
        pos_cos = cos[positions].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
        pos_sin = sin[positions].unsqueeze(0).unsqueeze(0)
    else:
        pos_cos = cos[positions].unsqueeze(1)  # [batch, 1, seq, head_dim]
        pos_sin = sin[positions].unsqueeze(1)

    return (x * pos_cos) + (rotate_half(x) * pos_sin)


# ============================================================================
# EAGLE-3 Draft Head Implementation
# ============================================================================

class Eagle3DraftHead(nn.Module):
    """
    Complete EAGLE-3 draft head with attention.

    Architecture:
    - fc: Linear(num_aux * hidden_size, hidden_size) - projects aux hidden states
    - Decoder layer 0 (input_layer):
      - hidden_norm: RMSNorm on projected hidden states
      - input_layernorm: RMSNorm on token embeddings
      - Concat [normed_embeds, normed_hidden] → 2*hidden_size
      - GQA Self-Attention (q_proj input: 2*hidden_size, kv_proj input: 2*hidden_size)
      - post_attention_layernorm + residual
      - SwiGLU MLP
    - Final norm + lm_head
    - d2t mapping
    """

    def __init__(self, config: ProfilingConfig):
        super().__init__()
        self.config = config
        hs = config.hidden_size
        n_heads = config.num_attention_heads
        n_kv = config.num_kv_heads
        hd = config.head_dim
        inter = config.intermediate_size

        # FC projection: 3*hidden → hidden
        self.fc = nn.Linear(hs * len(config.aux_layer_indices), hs, bias=False)

        # Decoder layer norms
        self.hidden_norm = RMSNorm(hs, config.rms_norm_eps)
        self.input_layernorm = RMSNorm(hs, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hs, config.rms_norm_eps)

        # Attention projections (input dim = 2*hidden_size for input layer)
        input_dim = 2 * hs
        self.q_proj = nn.Linear(input_dim, n_heads * hd, bias=False)
        self.k_proj = nn.Linear(input_dim, n_kv * hd, bias=False)
        self.v_proj = nn.Linear(input_dim, n_kv * hd, bias=False)
        self.o_proj = nn.Linear(n_heads * hd, hs, bias=False)

        # MLP (SwiGLU)
        self.gate_proj = nn.Linear(hs, inter, bias=False)
        self.up_proj = nn.Linear(hs, inter, bias=False)
        self.down_proj = nn.Linear(inter, hs, bias=False)

        # Final norm and LM head
        self.norm = RMSNorm(hs, config.rms_norm_eps)
        self.lm_head = nn.Linear(hs, config.draft_vocab_size, bias=False)

        # d2t mapping (draft token id → target token id)
        self.register_buffer('d2t', torch.zeros(config.draft_vocab_size, dtype=torch.long))

        # RoPE
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = hd
        self.n_rep = n_heads // n_kv  # GQA repeat factor

        # Precomputed RoPE will be set after moving to device
        self.rope_cos = None
        self.rope_sin = None

    def init_rope(self, max_seq_len: int = 256):
        """Initialize RoPE frequencies."""
        self.rope_cos, self.rope_sin = precompute_freqs_cis(
            self.head_dim, max_seq_len,
            theta=self.config.rope_theta,
            device=self.fc.weight.device,
            dtype=self.config.dtype
        )

    def load_weights(self, state_dict: dict):
        """Load weights from the EAGLE-3 checkpoint."""
        # d2t mapping
        d2t_diff = state_dict['d2t']
        self.d2t = d2t_diff + torch.arange(d2t_diff.shape[0])

        # FC
        self.fc.weight.data.copy_(state_dict['fc.weight'])

        # Attention
        self.q_proj.weight.data.copy_(state_dict['midlayer.self_attn.q_proj.weight'])
        self.k_proj.weight.data.copy_(state_dict['midlayer.self_attn.k_proj.weight'])
        self.v_proj.weight.data.copy_(state_dict['midlayer.self_attn.v_proj.weight'])
        self.o_proj.weight.data.copy_(state_dict['midlayer.self_attn.o_proj.weight'])

        # MLP
        self.gate_proj.weight.data.copy_(state_dict['midlayer.mlp.gate_proj.weight'])
        self.up_proj.weight.data.copy_(state_dict['midlayer.mlp.up_proj.weight'])
        self.down_proj.weight.data.copy_(state_dict['midlayer.mlp.down_proj.weight'])

        # Norms
        self.hidden_norm.weight.data.copy_(state_dict['midlayer.hidden_norm.weight'])
        self.input_layernorm.weight.data.copy_(state_dict['midlayer.input_layernorm.weight'])
        self.post_attention_layernorm.weight.data.copy_(state_dict['midlayer.post_attention_layernorm.weight'])
        self.norm.weight.data.copy_(state_dict['norm.weight'])

        # LM head
        self.lm_head.weight.data.copy_(state_dict['lm_head.weight'])

    def _attention(self, hidden_states: torch.Tensor, positions: torch.Tensor,
                   kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        GQA attention with RoPE and optional KV cache.

        Args:
            hidden_states: [batch, seq_len, 2*hidden_size] (concatenated input)
            positions: [seq_len] absolute positions
            kv_cache: tuple of (k_cache, v_cache) each [batch, n_kv, cache_len, head_dim]

        Returns:
            output: [batch, seq_len, hidden_size]
            new_kv_cache: tuple of (k_cache, v_cache)
        """
        bsz, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)  # [bsz, seq, n_heads * hd]
        k = self.k_proj(hidden_states)  # [bsz, seq, n_kv * hd]
        v = self.v_proj(hidden_states)  # [bsz, seq, n_kv * hd]

        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rotary_emb(q, self.rope_cos, self.rope_sin, positions)
        k = apply_rotary_emb(k, self.rope_cos, self.rope_sin, positions)

        # KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_kv_cache = (k, v)

        # GQA: repeat KV heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention with causal mask
        # Use causal when doing prefill (no cache, multi-token)
        # For decode (with cache, single new token), no causal needed
        use_causal = (kv_cache is None and seq_len > 1)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=use_causal
        )

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.o_proj(attn_out)

        return output, new_kv_cache

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU MLP."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)

    def forward(self, hidden_states: torch.Tensor, embeds: torch.Tensor,
                positions: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                is_first_step: bool = True):
        """
        Forward pass of the EAGLE-3 draft head.

        Args:
            hidden_states: [batch, seq_len, hidden_size] or [batch, seq_len, 3*hidden_size]
                If 3*hidden_size (first step from target), fc is applied.
                If hidden_size (subsequent steps from draft's own output), fc is skipped.
            embeds: [batch, seq_len, hidden_size] - token embeddings
            positions: [seq_len] - absolute positions for RoPE
            kv_cache: optional KV cache from previous draft steps
            is_first_step: whether this is the first draft step (uses target hidden)

        Returns:
            logits: [batch, seq_len, draft_vocab_size]
            aux_hidden: [batch, seq_len, hidden_size] - for next draft step
            new_kv_cache: updated KV cache
        """
        # Apply fc if input is from target (3*hidden_size)
        if hidden_states.shape[-1] != self.config.hidden_size:
            hidden_states = self.fc(hidden_states)

        # Decoder layer (input layer)
        residual = hidden_states
        hidden_normed = self.hidden_norm(hidden_states)
        embeds_normed = self.input_layernorm(embeds)

        # Concat [embeds, hidden] → 2*hidden_size
        concat_input = torch.cat([embeds_normed, hidden_normed], dim=-1)

        # Self attention
        attn_out, new_kv_cache = self._attention(concat_input, positions, kv_cache)

        # Post-attention: residual + norm
        hidden_states = attn_out + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        hidden_states = self._mlp(hidden_states)

        # Final residual
        hidden_states = hidden_states + residual

        # Save aux hidden (pre-norm) for next step
        aux_hidden = hidden_states.clone()

        # Final norm + lm_head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, aux_hidden, new_kv_cache

    def draft_token_to_target(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        """Convert draft token IDs to target vocab token IDs using d2t mapping."""
        return self.d2t[draft_token_ids]

    def target_token_to_draft(self, target_token_ids: torch.Tensor,
                               t2d_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert target token IDs to draft vocab space.
        For tokens not in draft vocab, return -1.
        """
        # t2d_mask is a boolean mask indicating which target tokens are in draft vocab
        # We need the actual mapping - but since d2t[i] = target_id for draft_id i,
        # we need the inverse. For simplicity, build a lookup.
        # This is precomputed in __init__ or load.
        pass


# ============================================================================
# Speculative Decoding Loop
# ============================================================================

class SpeculativeDecoder:
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype

        self.logger = logging.getLogger("SpecDecoder")
        self.logger.setLevel(logging.INFO)

        # Metrics storage
        self.per_token_metrics = []
        self.per_step_summaries = []

    def load_models(self):
        """Load target model and EAGLE-3 draft head."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.target_model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.info("Loading target model (Qwen3-8B)...")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.config.target_model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.target_model.eval()

        # Get the embedding layer for draft head
        self.embed_tokens = self.target_model.model.embed_tokens

        self.logger.info("Loading EAGLE-3 draft head weights...")
        eagle_state_dict = torch.load(
            self.config.eagle3_weights_path,
            map_location='cpu',
            weights_only=False
        )

        self.draft_head = Eagle3DraftHead(self.config)
        self.draft_head.load_weights(eagle_state_dict)
        self.draft_head = self.draft_head.to(device=self.device, dtype=self.dtype)
        self.draft_head.eval()
        self.draft_head.init_rope(max_seq_len=2048)

        # Move d2t to device
        self.draft_head.d2t = self.draft_head.d2t.to(self.device)

        # Build target-to-draft mapping for converting target tokens to draft space
        self._build_t2d_mapping(eagle_state_dict)

        self.logger.info("All models loaded successfully.")

    def _build_t2d_mapping(self, state_dict: dict):
        """Build reverse mapping from target vocab to draft vocab."""
        # d2t[draft_id] = target_id
        d2t = self.draft_head.d2t  # [32000] on device
        # Build reverse: t2d[target_id] = draft_id (or -1 if not mapped)
        self.t2d = torch.full((self.config.target_vocab_size,), -1,
                              dtype=torch.long, device=self.device)
        draft_ids = torch.arange(self.config.draft_vocab_size, device=self.device)
        target_ids = d2t.long()
        self.t2d[target_ids] = draft_ids
        self.logger.info(f"t2d mapping: {(self.t2d >= 0).sum().item()} target tokens mapped to draft vocab")

    def extract_aux_hidden_states(self, model_output) -> torch.Tensor:
        """
        Extract and concatenate auxiliary hidden states from target model output.

        For Qwen3-8B with layers_to_capture = [2, 18, 33]:
        - hidden_states[2] = output going INTO layer 2 = output of layer 1
        - hidden_states[18] = output going INTO layer 18 = output of layer 17
        - hidden_states[33] = output going INTO layer 33 = output of layer 32
        """
        all_hidden = model_output.hidden_states  # tuple of (n_layers+1) tensors
        # In HuggingFace: hidden_states[i] = output after layer i-1
        # "Before layer i" = hidden_states[i]
        aux_states = []
        for layer_idx in self.config.aux_layer_indices:
            aux_states.append(all_hidden[layer_idx])

        # Concatenate along hidden dim: [batch, seq, 3*hidden_size]
        return torch.cat(aux_states, dim=-1)

    @torch.no_grad()
    def run_target_forward(self, input_ids: torch.Tensor):
        """Run target model forward pass, returning logits and hidden states."""
        output = self.target_model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )
        return output

    @torch.no_grad()
    def eagle_prefill(self, target_output, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Run EAGLE model prefill on the full input sequence to build KV cache.
        
        This is critical: the eagle model's attention needs context from the full
        input sequence. Without prefill, subsequent decode steps have no history.
        
        Args:
            target_output: target model output with hidden_states
            input_ids: [1, N] input token ids
            
        Returns:
            aux_hidden: [1, 1, hidden_size] - aux hidden at the last position
            kv_cache: tuple of (k, v) from the full prefill
        """
        # Get aux hidden states at ALL positions
        aux_all = self.extract_aux_hidden_states(target_output)  # [1, N, 3*hidden_size]
        
        # Get embeddings for all tokens
        embeds_all = self.embed_tokens(input_ids)  # [1, N, hidden_size]
        
        # Positions for the full sequence
        N = input_ids.shape[1]
        positions = torch.arange(N, device=self.device)
        
        # Run eagle forward on the full sequence (prefill)
        logits, aux_hidden, kv_cache = self.draft_head(
            hidden_states=aux_all,
            embeds=embeds_all,
            positions=positions,
            kv_cache=None,
            is_first_step=True
        )
        
        # Return aux_hidden at the LAST position and the full KV cache
        return aux_hidden[:, -1:, :], kv_cache

    @torch.no_grad()
    def draft_phase(self, target_aux_hidden: torch.Tensor,
                    prev_token_id: int,
                    start_position: int,
                    eagle_kv_cache: Optional[Tuple] = None,
                    eagle_aux_hidden: Optional[torch.Tensor] = None) -> Tuple[List[int], List[float]]:
        """
        Auto-regressively generate K draft tokens.

        Args:
            target_aux_hidden: [1, 1, 3*hidden_size] - aux hidden from target at last position
            prev_token_id: the last accepted token (target vocab id)
            start_position: the absolute position for RoPE

        Returns:
            draft_tokens: list of K target-vocab token ids
            draft_confidences: list of K confidence scores (max prob)
        """
        K = self.config.num_draft_steps
        draft_tokens = []
        draft_confidences = []

        # Use prefilled KV cache and aux hidden if provided
        if eagle_aux_hidden is not None:
            hidden_states = eagle_aux_hidden  # [1, 1, hidden_size] from eagle prefill
        else:
            hidden_states = target_aux_hidden  # [1, 1, 3*hidden_size] fallback
        
        kv_cache = eagle_kv_cache  # May be None or prefilled

        for k in range(K):
            # Get embedding of previous token
            prev_token_tensor = torch.tensor([[prev_token_id]], device=self.device)
            # Convert target token to draft space for embedding lookup
            # Actually, we use the TARGET model's embeddings (shared)
            embeds = self.embed_tokens(prev_token_tensor)  # [1, 1, hidden_size]

            # Position for this draft step
            pos = torch.tensor([start_position + k], device=self.device)

            # Draft head forward
            logits, aux_hidden, kv_cache = self.draft_head(
                hidden_states=hidden_states,
                embeds=embeds,
                positions=pos,
                kv_cache=kv_cache,
                is_first_step=(k == 0)
            )

            # Get draft logits and pick token
            draft_logits = logits[0, -1, :]  # [draft_vocab_size]
            probs = F.softmax(draft_logits.float(), dim=-1)
            draft_token_id = probs.argmax().item()
            confidence = probs[draft_token_id].item()

            # Convert to target vocab
            target_token_id = self.draft_head.d2t[draft_token_id].item()

            draft_tokens.append(target_token_id)
            draft_confidences.append(confidence)

            # Update for next step
            prev_token_id = target_token_id
            hidden_states = aux_hidden  # [1, 1, hidden_size] - no fc needed

        return draft_tokens, draft_confidences

    @torch.no_grad()
    def verify_phase(self, input_ids: torch.Tensor, target_next: int,
                     draft_tokens: List[int],
                     prompt_len: int) -> Tuple[int, List[bool], List[int]]:
        """
        Verify draft tokens with the target model.

        Convention B: draft_tokens[i] predicts position N+1+i where N = len(input_ids).
        We verify by running target on [input_ids, target_next, draft_tokens] and checking
        if target_logits[N+i].argmax() == draft_tokens[i].

        Args:
            input_ids: [1, seq_len] current input sequence (length N)
            target_next: target model's greedy prediction at position N (always accepted)
            draft_tokens: list of K draft token ids (target vocab)
            prompt_len: length of the original prompt

        Returns:
            n_accepted: number of accepted draft tokens
            acceptance_mask: list of K booleans
            target_choices: list of K target model choices at each position
        """
        K = len(draft_tokens)

        # Build verification sequence: [input_ids, target_next, draft_tokens]
        extra_tokens = [target_next] + draft_tokens
        extra_tensor = torch.tensor([extra_tokens], device=self.device)
        verify_ids = torch.cat([input_ids, extra_tensor], dim=1)

        # Run target model on the full sequence
        output = self.target_model(
            input_ids=verify_ids,
            output_hidden_states=False,
            return_dict=True,
        )

        # Check each draft position
        # verify_ids layout: [input(N), target_next(1), draft_tokens(K)]
        # Logit at position N predicts token at N+1 → should match draft_tokens[0]
        # Logit at position N+1 predicts token at N+2 → should match draft_tokens[1]
        input_len = input_ids.shape[1]  # = N
        n_accepted = 0
        acceptance_mask = []
        target_choices = []

        for i in range(K):
            logit_pos = input_len + i  # N, N+1, ..., N+K-1
            target_logits = output.logits[0, logit_pos, :]
            target_choice = target_logits.argmax().item()
            target_choices.append(target_choice)

            if target_choice == draft_tokens[i]:
                n_accepted += 1
                acceptance_mask.append(True)
            else:
                acceptance_mask.append(False)
                break  # Stop at first rejection

        # Fill remaining positions
        while len(acceptance_mask) < K:
            acceptance_mask.append(False)
            logit_pos = input_len + len(target_choices)
            if logit_pos < output.logits.shape[1]:
                target_logits = output.logits[0, logit_pos, :]
                target_choices.append(target_logits.argmax().item())

        return n_accepted, acceptance_mask, target_choices

    @torch.no_grad()
    def run_single_prompt(self, prompt: str, prompt_idx: int) -> dict:
        """Run speculative decoding on a single prompt."""
        self.logger.info(f"Processing prompt {prompt_idx}: {prompt[:60]}...")

        # Format with chat template and tokenize
        formatted_prompt = self.format_prompt(prompt)
        inputs = self.tokenizer(
            formatted_prompt, return_tensors="pt", truncation=True,
            max_length=self.config.max_prompt_len
        ).to(self.device)
        input_ids = inputs['input_ids']  # [1, seq_len]
        original_prompt_len = input_ids.shape[1]

        # Metrics for this prompt
        step_metrics = []
        total_accepted = 0
        total_drafted = 0
        generated_tokens = 0

        step = 0
        while generated_tokens < self.config.max_new_tokens:
            current_len = input_ids.shape[1]

            # === Target model forward (get hidden states for draft) ===
            target_output = self.run_target_forward(input_ids)

            # Get the target model's next token prediction (greedy)
            target_next = target_output.logits[0, -1, :].argmax().item()

            # Check for EOS
            if target_next == self.tokenizer.eos_token_id:
                break

            # Extract aux hidden states at the last position
            aux_hidden = self.extract_aux_hidden_states(target_output)
            # Take only the last position: [1, 1, 3*hidden_size]
            aux_hidden_last = aux_hidden[:, -1:, :]

            # === Eagle Prefill: build KV cache from full context ===
            eagle_aux, eagle_kv = self.eagle_prefill(target_output, input_ids)

            # === Draft Phase (Convention B) ===
            # 1. Process target_next at position N (adds to eagle KV)
            # 2. Auto-regressively predict K tokens at positions N+1..N+K
            # Draft tokens predict positions N+1, N+2, ..., N+K
            draft_tokens, draft_confidences = self.draft_phase(
                target_aux_hidden=aux_hidden_last,
                prev_token_id=target_next,  # Feed target's prediction
                start_position=current_len,  # Position N (of target_next)
                eagle_kv_cache=eagle_kv,
                eagle_aux_hidden=eagle_aux,
            )

            # === Verify Phase ===
            # Draft tokens predict positions N+1..N+K
            # Verify by running target on [input_ids, target_next, draft_tokens]
            n_accepted, acceptance_mask, target_choices = self.verify_phase(
                input_ids, target_next, draft_tokens, original_prompt_len
            )

            # Record metrics
            for pos_k in range(len(acceptance_mask)):
                metric = {
                    'prompt_idx': prompt_idx,
                    'step': step,
                    'position': pos_k,
                    'draft_token': draft_tokens[pos_k],
                    'target_token': target_choices[pos_k] if pos_k < len(target_choices) else -1,
                    'accepted': acceptance_mask[pos_k],
                    'confidence': draft_confidences[pos_k],
                    'draft_token_text': self.tokenizer.decode([draft_tokens[pos_k]]),
                    'target_token_text': self.tokenizer.decode([target_choices[pos_k]]) if pos_k < len(target_choices) else '',
                }
                self.per_token_metrics.append(metric)

            step_metrics.append({
                'step': step,
                'n_accepted': n_accepted,
                'K': self.config.num_draft_steps,
                'acceptance_rate': n_accepted / self.config.num_draft_steps,
                'confidences': draft_confidences,
            })

            total_accepted += n_accepted
            total_drafted += self.config.num_draft_steps

            # Update input_ids (Convention B):
            # target_next is always accepted (came from target model)
            # Then accept draft tokens up to first rejection
            # Then add bonus token from target at rejection point
            new_tokens_list = [target_next]  # Always accepted

            if n_accepted > 0:
                new_tokens_list.extend(draft_tokens[:n_accepted])

            # Add bonus token (target's choice at the first rejected position)
            if n_accepted < self.config.num_draft_steps:
                bonus_token = target_choices[n_accepted] if n_accepted < len(target_choices) else target_next
                new_tokens_list.append(bonus_token)

            new_tokens_tensor = torch.tensor([new_tokens_list], device=self.device)
            input_ids = torch.cat([input_ids, new_tokens_tensor], dim=1)
            generated_tokens += len(new_tokens_list)

            step += 1

            # Safety: prevent infinite loop
            if step > 100:
                break

        # Summary for this prompt
        overall_acceptance = total_accepted / max(total_drafted, 1)
        summary = {
            'prompt_idx': prompt_idx,
            'prompt_len': original_prompt_len,
            'generated_tokens': generated_tokens,
            'total_steps': step,
            'total_accepted': total_accepted,
            'total_drafted': total_drafted,
            'overall_acceptance_rate': overall_acceptance,
            'step_details': step_metrics,
        }
        self.per_step_summaries.append(summary)

        self.logger.info(
            f"  Prompt {prompt_idx}: acceptance={overall_acceptance:.1%} "
            f"({total_accepted}/{total_drafted}), "
            f"generated={generated_tokens} tokens in {step} steps"
        )

        return summary

    def load_eval_data(self) -> List[str]:
        """Load evaluation prompts and format with chat template (thinking disabled)."""
        prompts = []
        with open(self.config.eval_data_path) as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data['query'])
        return prompts

    def format_prompt(self, query: str) -> str:
        """Format prompt using chat template with thinking disabled."""
        messages = [
            {"role": "user", "content": query}
        ]
        # Apply chat template with thinking disabled
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Fallback if enable_thinking not supported
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return text

    def run(self):
        """Main entry point."""
        # Setup output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Setup logging to file
        log_path = os.path.join(self.config.output_dir, 'run_log.txt')
        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.logger.info("=" * 60)
        self.logger.info("EAGLE-3 Full Speculative Decoding Profiling")
        self.logger.info("=" * 60)
        self.logger.info(f"Config: K={self.config.num_draft_steps}, "
                        f"max_new_tokens={self.config.max_new_tokens}, "
                        f"num_samples={self.config.num_samples}")

        # Load models
        self.load_models()

        # Load data
        prompts = self.load_eval_data()
        num_to_run = min(self.config.num_samples, len(prompts))
        self.logger.info(f"Running on {num_to_run} prompts (total available: {len(prompts)})")

        # Run speculative decoding
        start_time = time.time()
        for i in range(num_to_run):
            self.run_single_prompt(prompts[i], i)
            # Save intermediate results
            if (i + 1) % 5 == 0:
                self._save_results()

        elapsed = time.time() - start_time
        self.logger.info(f"\nTotal time: {elapsed:.1f}s for {num_to_run} prompts")

        # Final save
        self._save_results()

        # Run analysis
        self._run_analysis()

        self.logger.info("Done!")

    def _save_results(self):
        """Save collected metrics to files."""
        out_dir = self.config.output_dir

        # Per-token metrics
        with open(os.path.join(out_dir, 'per_token_metrics.jsonl'), 'w') as f:
            for m in self.per_token_metrics:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')

        # Per-step summary
        with open(os.path.join(out_dir, 'per_step_summary.json'), 'w') as f:
            json.dump(self.per_step_summaries, f, indent=2, ensure_ascii=False)

    def _run_analysis(self):
        """Run statistical analysis on collected data."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy import stats

        out_dir = self.config.output_dir
        K = self.config.num_draft_steps

        # === Per-position acceptance rate ===
        position_accepted = {k: [] for k in range(K)}
        for m in self.per_token_metrics:
            pos = m['position']
            if pos < K:
                position_accepted[pos].append(1 if m['accepted'] else 0)

        fig, ax = plt.subplots(figsize=(8, 5))
        positions = list(range(K))
        rates = [np.mean(position_accepted[k]) if position_accepted[k] else 0 for k in positions]
        counts = [len(position_accepted[k]) for k in positions]

        ax.bar(positions, rates, color='steelblue', alpha=0.8)
        ax.set_xlabel('Draft Position (k)')
        ax.set_ylabel('Acceptance Rate')
        ax.set_title('Per-Position Acceptance Rate')
        ax.set_xticks(positions)
        for i, (r, c) in enumerate(zip(rates, counts)):
            ax.text(i, r + 0.02, f'{r:.1%}\n(n={c})', ha='center', fontsize=9)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'per_position_acceptance.png'), dpi=150)
        plt.close()

        # === Acceptance distribution ===
        step_rates = [s['overall_acceptance_rate'] for s in self.per_step_summaries]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram of per-step acceptance rates
        all_step_accept = [s['acceptance_rate'] for summ in self.per_step_summaries
                          for s in summ['step_details']]
        axes[0].hist(all_step_accept, bins=K+1, range=(-0.5/K, 1+0.5/K),
                    color='coral', alpha=0.8, edgecolor='black')
        axes[0].set_xlabel('Step Acceptance Rate (n_accepted/K)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Distribution of Per-Step Acceptance Rates')
        axes[0].axvline(np.mean(all_step_accept), color='red', linestyle='--',
                       label=f'Mean={np.mean(all_step_accept):.2%}')
        axes[0].legend()

        # Confidence vs Acceptance scatter
        confs = [m['confidence'] for m in self.per_token_metrics]
        accs = [1 if m['accepted'] else 0 for m in self.per_token_metrics]
        axes[1].scatter(confs, accs, alpha=0.3, s=10, c='steelblue')
        axes[1].set_xlabel('Draft Confidence (max prob)')
        axes[1].set_ylabel('Accepted (1=yes, 0=no)')
        axes[1].set_title('Confidence vs Acceptance')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'acceptance_distribution.png'), dpi=150)
        plt.close()

        # === Bimodal Analysis ===
        bimodal_results = self._bimodal_analysis(confs)

        # === Summary statistics ===
        overall_rate = np.mean([m['accepted'] for m in self.per_token_metrics]) if self.per_token_metrics else 0
        bimodal_results['summary'] = {
            'overall_acceptance_rate': overall_rate,
            'per_position_rates': {str(k): rates[k] for k in positions},
            'mean_confidence': float(np.mean(confs)) if confs else 0,
            'num_prompts': len(self.per_step_summaries),
            'num_tokens_evaluated': len(self.per_token_metrics),
        }

        with open(os.path.join(out_dir, 'bimodal_analysis.json'), 'w') as f:
            json.dump(bimodal_results, f, indent=2)

        self.logger.info(f"\n{'='*40}")
        self.logger.info(f"RESULTS SUMMARY")
        self.logger.info(f"{'='*40}")
        self.logger.info(f"Overall acceptance rate: {overall_rate:.1%}")
        self.logger.info(f"Per-position rates: {[f'{r:.1%}' for r in rates]}")
        self.logger.info(f"Mean draft confidence: {np.mean(confs):.3f}" if confs else "No confidence data")

    def _bimodal_analysis(self, confidences: List[float]) -> dict:
        """Run bimodal statistical tests on confidence distribution."""
        from scipy import stats

        results = {}

        if len(confidences) < 10:
            results['note'] = 'Insufficient data for bimodal analysis'
            return results

        confs = np.array(confidences)

        # Hartigan's Dip Test (using a simple approximation)
        try:
            from scipy.stats import gaussian_kde
            # Simple approach: fit GMM with 1 and 2 components, compare BIC
            from sklearn.mixture import GaussianMixture

            confs_2d = confs.reshape(-1, 1)

            gmm1 = GaussianMixture(n_components=1, random_state=42).fit(confs_2d)
            gmm2 = GaussianMixture(n_components=2, random_state=42).fit(confs_2d)
            gmm3 = GaussianMixture(n_components=3, random_state=42).fit(confs_2d)

            bic1 = gmm1.bic(confs_2d)
            bic2 = gmm2.bic(confs_2d)
            bic3 = gmm3.bic(confs_2d)

            results['gmm'] = {
                'bic_1_component': float(bic1),
                'bic_2_components': float(bic2),
                'bic_3_components': float(bic3),
                'best_n_components': int(np.argmin([bic1, bic2, bic3]) + 1),
                'gmm2_means': gmm2.means_.flatten().tolist(),
                'gmm2_weights': gmm2.weights_.tolist(),
                'gmm2_variances': gmm2.covariances_.flatten().tolist(),
            }

            if bic2 < bic1:
                results['bimodal_evidence'] = 'GMM suggests bimodal (2-comp BIC < 1-comp BIC)'
            else:
                results['bimodal_evidence'] = 'No strong bimodal evidence from GMM'

        except ImportError as e:
            results['gmm_error'] = str(e)

        # Basic statistics
        results['descriptive'] = {
            'mean': float(np.mean(confs)),
            'std': float(np.std(confs)),
            'median': float(np.median(confs)),
            'skewness': float(stats.skew(confs)),
            'kurtosis': float(stats.kurtosis(confs)),
            'q25': float(np.percentile(confs, 25)),
            'q75': float(np.percentile(confs, 75)),
        }

        # Shapiro-Wilk test for normality
        if len(confs) <= 5000:
            stat, p_value = stats.shapiro(confs[:min(len(confs), 5000)])
            results['shapiro_wilk'] = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'is_normal': bool(p_value > 0.05)
            }

        return results


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="EAGLE-3 Full Speculative Decoding Profiling")
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of prompts to evaluate')
    parser.add_argument('--num-draft-steps', type=int, default=5,
                       help='Number of draft steps (K)')
    parser.add_argument('--max-new-tokens', type=int, default=64,
                       help='Max tokens to generate per prompt')
    args = parser.parse_args()

    config = ProfilingConfig(
        num_samples=args.num_samples,
        num_draft_steps=args.num_draft_steps,
        max_new_tokens=args.max_new_tokens,
    )

    decoder = SpeculativeDecoder(config)
    decoder.run()


if __name__ == "__main__":
    main()
