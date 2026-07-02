#!/usr/bin/env python3
"""DAHD vs EAGLE-3 End-to-End Speculative Decoding Comparison.

Implements a real speculative decoding loop and compares:
1. EAGLE-3 baseline: AR mode, K=5
2. Always Parallel: Hydra Parallel Branch, K=6
3. DAHD: Router dynamically selects AR or Parallel

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/phase4_e2e/run_dahd_e2e_eval.py [--num-prompts 5]
"""

import json
import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class E2EConfig:
    # Model paths
    target_model_path: str = "/mnt/nas1/hf/Qwen3-8B/"
    eagle3_weights_path: str = "/mnt/nas1/hf/qwen3_8b_eagle3/pytorch_model.bin"
    hydra_ckpt_path: str = "/data1/caoshuaihu.csh/workspace/dahd_speculative_decoding/checkpoints/hydra_parallel_branch_best.pt"
    router_ckpt_path: str = "/data1/caoshuaihu.csh/workspace/dahd_speculative_decoding/checkpoints/difficulty_router.pt"
    eval_data_path: str = "/mnt/nas1/hf/qwen3_8b_eagle3/eagle_data.jsonl"

    # Model architecture (Qwen3-8B)
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    draft_vocab_size: int = 32000
    target_vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    aux_layer_indices: List[int] = field(default_factory=lambda: [2, 18, 33])

    # Speculative decoding params
    eagle3_K: int = 5        # EAGLE-3 draft steps
    parallel_K: int = 6     # Parallel branch heads
    dahd_ar_K: int = 3      # DAHD AR mode steps (shorter)
    max_new_tokens: int = 128
    max_prompt_len: int = 512

    # Eval params
    num_prompts: int = 50    # Use last 50 from data
    warmup_prompts: int = 2  # Warmup prompts (not counted in metrics)

    # Output
    output_dir: str = "/data1/caoshuaihu.csh/workspace/dahd_speculative_decoding/results/phase4_results/"

    # Device
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16


# ============================================================================
# RMSNorm
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
# RoPE
# ============================================================================

def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 1000000.0,
                          device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                     positions: torch.Tensor) -> torch.Tensor:
    if positions.dim() == 1:
        pos_cos = cos[positions].unsqueeze(0).unsqueeze(0)
        pos_sin = sin[positions].unsqueeze(0).unsqueeze(0)
    else:
        pos_cos = cos[positions].unsqueeze(1)
        pos_sin = sin[positions].unsqueeze(1)
    return (x * pos_cos) + (rotate_half(x) * pos_sin)


# ============================================================================
# EAGLE-3 Draft Head (complete with GQA attention + RoPE + KV cache)
# ============================================================================

class Eagle3DraftHead(nn.Module):
    """Complete EAGLE-3 draft head with attention layer."""

    def __init__(self, config: E2EConfig):
        super().__init__()
        self.config = config
        hs = config.hidden_size
        n_heads = config.num_attention_heads
        n_kv = config.num_kv_heads
        hd = config.head_dim
        inter = config.intermediate_size

        self.fc = nn.Linear(hs * len(config.aux_layer_indices), hs, bias=False)
        self.hidden_norm = RMSNorm(hs, config.rms_norm_eps)
        self.input_layernorm = RMSNorm(hs, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hs, config.rms_norm_eps)

        input_dim = 2 * hs
        self.q_proj = nn.Linear(input_dim, n_heads * hd, bias=False)
        self.k_proj = nn.Linear(input_dim, n_kv * hd, bias=False)
        self.v_proj = nn.Linear(input_dim, n_kv * hd, bias=False)
        self.o_proj = nn.Linear(n_heads * hd, hs, bias=False)

        self.gate_proj = nn.Linear(hs, inter, bias=False)
        self.up_proj = nn.Linear(hs, inter, bias=False)
        self.down_proj = nn.Linear(inter, hs, bias=False)

        self.norm = RMSNorm(hs, config.rms_norm_eps)
        self.lm_head = nn.Linear(hs, config.draft_vocab_size, bias=False)
        self.register_buffer('d2t', torch.zeros(config.draft_vocab_size, dtype=torch.long))

        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = hd
        self.n_rep = n_heads // n_kv
        self.rope_cos = None
        self.rope_sin = None

    def init_rope(self, max_seq_len: int = 2048):
        self.rope_cos, self.rope_sin = precompute_freqs_cis(
            self.head_dim, max_seq_len,
            theta=self.config.rope_theta,
            device=self.fc.weight.device,
            dtype=self.config.dtype
        )

    def load_weights(self, state_dict: dict):
        d2t_diff = state_dict['d2t']
        self.d2t = d2t_diff + torch.arange(d2t_diff.shape[0])
        self.fc.weight.data.copy_(state_dict['fc.weight'])
        self.q_proj.weight.data.copy_(state_dict['midlayer.self_attn.q_proj.weight'])
        self.k_proj.weight.data.copy_(state_dict['midlayer.self_attn.k_proj.weight'])
        self.v_proj.weight.data.copy_(state_dict['midlayer.self_attn.v_proj.weight'])
        self.o_proj.weight.data.copy_(state_dict['midlayer.self_attn.o_proj.weight'])
        self.gate_proj.weight.data.copy_(state_dict['midlayer.mlp.gate_proj.weight'])
        self.up_proj.weight.data.copy_(state_dict['midlayer.mlp.up_proj.weight'])
        self.down_proj.weight.data.copy_(state_dict['midlayer.mlp.down_proj.weight'])
        self.hidden_norm.weight.data.copy_(state_dict['midlayer.hidden_norm.weight'])
        self.input_layernorm.weight.data.copy_(state_dict['midlayer.input_layernorm.weight'])
        self.post_attention_layernorm.weight.data.copy_(state_dict['midlayer.post_attention_layernorm.weight'])
        self.norm.weight.data.copy_(state_dict['norm.weight'])
        self.lm_head.weight.data.copy_(state_dict['lm_head.weight'])

    def _attention(self, hidden_states: torch.Tensor, positions: torch.Tensor,
                   kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.n_kv, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, self.rope_cos, self.rope_sin, positions)
        k = apply_rotary_emb(k, self.rope_cos, self.rope_sin, positions)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        use_causal = (kv_cache is None and seq_len > 1)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_out), new_kv_cache

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, hidden_states: torch.Tensor, embeds: torch.Tensor,
                positions: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                is_first_step: bool = True):
        if hidden_states.shape[-1] != self.config.hidden_size:
            hidden_states = self.fc(hidden_states)

        residual = hidden_states
        hidden_normed = self.hidden_norm(hidden_states)
        embeds_normed = self.input_layernorm(embeds)
        concat_input = torch.cat([embeds_normed, hidden_normed], dim=-1)

        attn_out, new_kv_cache = self._attention(concat_input, positions, kv_cache)
        hidden_states = attn_out + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._mlp(hidden_states)
        hidden_states = hidden_states + residual

        aux_hidden = hidden_states.clone()
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, aux_hidden, new_kv_cache


# ============================================================================
# Hydra Parallel Branch (same as training script)
# ============================================================================

class HydraParallelBranch(nn.Module):
    """Hydra-style sequential-dependent parallel heads."""

    def __init__(self, hidden_size: int = 4096, num_heads: int = 6,
                 draft_vocab_size: int = 32000):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.draft_vocab_size = draft_vocab_size

        self.fc = nn.Linear(hidden_size * 3, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, draft_vocab_size, bias=False)

        self.head_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_heads)
        ])
        self.head_linears = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_heads)
        ])
        self.head_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_heads - 1)
        ])

    def forward(self, hidden_concat: torch.Tensor) -> list:
        """
        Args:
            hidden_concat: [B, 12288] - concat of last 3 layer hidden states
        Returns:
            List of [B, draft_vocab_size] logits, one per head
        """
        shared = self.fc(hidden_concat)
        all_logits = []
        prev_hidden = shared

        for i in range(self.num_heads):
            if i > 0:
                projected = self.head_projections[i - 1](prev_hidden)
                curr_input = shared + projected
            else:
                curr_input = shared

            h = self.head_norms[i](curr_input)
            h = self.head_linears[i](h)
            prev_hidden = h
            h = F.silu(h)
            logit = self.lm_head(h)
            all_logits.append(logit)

        return all_logits

    def get_shared_repr(self, hidden_concat: torch.Tensor) -> torch.Tensor:
        """Get shared representation (fc output) for router input."""
        return self.fc(hidden_concat)


# ============================================================================
# Difficulty Router
# ============================================================================

class DifficultyRouter(nn.Module):
    """MLP router: predicts easy (parallel) vs hard (AR)."""

    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(shared_repr))


# ============================================================================
# End-to-End Evaluator
# ============================================================================

class E2EEvaluator:
    def __init__(self, config: E2EConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.logger = logging.getLogger("E2E")

    def load_all_models(self):
        """Load target model, EAGLE-3, Hydra Parallel Branch, and Router."""
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
        self.embed_tokens = self.target_model.model.embed_tokens

        # --- EAGLE-3 Draft Head ---
        self.logger.info("Loading EAGLE-3 draft head...")
        eagle_sd = torch.load(self.config.eagle3_weights_path, map_location='cpu', weights_only=False)
        self.eagle3 = Eagle3DraftHead(self.config)
        self.eagle3.load_weights(eagle_sd)
        self.eagle3 = self.eagle3.to(device=self.device, dtype=self.dtype)
        self.eagle3.eval()
        self.eagle3.init_rope(max_seq_len=2048)
        self.eagle3.d2t = self.eagle3.d2t.to(self.device)

        # Build t2d mapping (for EAGLE-3, uses decoded d2t)
        self.t2d = torch.full((self.config.target_vocab_size,), -1, dtype=torch.long, device=self.device)
        d2t = self.eagle3.d2t.long()
        draft_ids = torch.arange(self.config.draft_vocab_size, device=self.device)
        self.t2d[d2t] = draft_ids

        # Build RAW d2t for parallel branch (training used raw diff as direct mapping)
        # The EAGLE-3 checkpoint stores d2t in diff format: actual_target = raw + index
        # But the parallel branch training script used raw values directly as target_ids
        # So for parallel branch decoding, we must also use raw values.
        self.d2t_raw = torch.tensor(
            eagle_sd['d2t'].numpy(), dtype=torch.long, device=self.device
        )

        # --- Hydra Parallel Branch ---
        self.logger.info("Loading Hydra Parallel Branch...")
        self.parallel_branch = HydraParallelBranch(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.parallel_K,
            draft_vocab_size=self.config.draft_vocab_size,
        ).to(device=self.device, dtype=torch.float32)  # Keep float32 as trained
        hydra_ckpt = torch.load(self.config.hydra_ckpt_path, map_location='cpu', weights_only=False)
        self.parallel_branch.load_state_dict(hydra_ckpt["model_state_dict"])
        self.parallel_branch.eval()

        # --- Difficulty Router ---
        self.logger.info("Loading Difficulty Router...")
        self.router = DifficultyRouter(hidden_size=self.config.hidden_size).to(
            device=self.device, dtype=torch.float32  # Router trained in fp32
        )
        router_ckpt = torch.load(self.config.router_ckpt_path, map_location='cpu', weights_only=False)
        self.router.load_state_dict(router_ckpt["model_state_dict"])
        self.router.eval()

        self.logger.info("All models loaded successfully.")

    # ---------- Helper: extract aux hidden states ----------
    def extract_aux_hidden(self, model_output) -> torch.Tensor:
        """Extract and concat EAGLE-3 aux hidden states: [1, seq, 3*hidden_size].
        Uses layers [2, 18, 33] as required by EAGLE-3."""
        all_hidden = model_output.hidden_states
        aux_states = [all_hidden[idx] for idx in self.config.aux_layer_indices]
        return torch.cat(aux_states, dim=-1)

    def extract_last3_hidden(self, model_output) -> torch.Tensor:
        """Extract and concat LAST 3 layer hidden states: [1, seq, 3*hidden_size].
        The Hydra parallel branch was trained on last 3 layers (indices -3, -2, -1)."""
        all_hidden = model_output.hidden_states
        h_last3 = [all_hidden[-3], all_hidden[-2], all_hidden[-1]]
        return torch.cat(h_last3, dim=-1)

    # ---------- EAGLE-3 AR draft ----------
    @torch.no_grad()
    def eagle3_draft(self, target_output, input_ids: torch.Tensor,
                     prev_token_id: int, K: int) -> List[int]:
        """Generate K draft tokens using EAGLE-3 AR mode with prefill."""
        # Prefill: build eagle KV cache from full context
        aux_all = self.extract_aux_hidden(target_output)  # [1, N, 3*hs]
        embeds_all = self.embed_tokens(input_ids)  # [1, N, hs]
        N = input_ids.shape[1]
        positions = torch.arange(N, device=self.device)

        _, eagle_aux, eagle_kv = self.eagle3(
            hidden_states=aux_all,
            embeds=embeds_all,
            positions=positions,
            kv_cache=None,
            is_first_step=True,
        )
        # Take last position aux hidden
        hidden_states = eagle_aux[:, -1:, :]  # [1, 1, hs]

        # AR decode K tokens
        draft_tokens = []
        current_token = prev_token_id
        kv_cache = eagle_kv

        for k in range(K):
            token_tensor = torch.tensor([[current_token]], device=self.device)
            embeds = self.embed_tokens(token_tensor)  # [1, 1, hs]
            pos = torch.tensor([N + k], device=self.device)

            logits, aux_hidden, kv_cache = self.eagle3(
                hidden_states=hidden_states,
                embeds=embeds,
                positions=pos,
                kv_cache=kv_cache,
                is_first_step=(k == 0),
            )

            draft_logits = logits[0, -1, :]
            draft_token_id = draft_logits.argmax().item()
            target_token_id = self.eagle3.d2t[draft_token_id].item()
            draft_tokens.append(target_token_id)

            current_token = target_token_id
            hidden_states = aux_hidden

        return draft_tokens

    # ---------- Parallel Branch draft ----------
    @torch.no_grad()
    def parallel_draft(self, last3_hidden_last: torch.Tensor) -> List[int]:
        """Generate draft tokens using Hydra Parallel Branch (one forward pass).

        The parallel branch is trained on hidden states at position P and predicts
        tokens at positions P+1, P+2, ..., P+6. In the spec-decode loop:
        - hidden state is at position N-1 (last position of input_ids)
        - head_0 predicts position N = target_next (already guaranteed correct)
        - heads 1-5 predict positions N+1..N+5 = the actual draft tokens

        So we skip head_0 and use heads 1-5 as draft tokens (K=5).

        NOTE: The parallel branch was trained using the RAW d2t (diff format) as
        direct target_id mapping. So we must decode with d2t_raw (not decoded d2t).
        """
        # last3_hidden_last: [1, 1, 3*hs] → squeeze to [1, 3*hs]
        hidden_concat = last3_hidden_last.squeeze(1).float()  # float32 for parallel branch
        logits_list = self.parallel_branch(hidden_concat)  # list of 6 [1, draft_vocab]
        draft_tokens = []
        # Skip head_0 (predicts target_next position), use heads 1-5
        for logits in logits_list[1:]:
            draft_id = logits.argmax(dim=-1).item()
            # Use RAW d2t (not decoded) — matches training label construction
            target_id = self.d2t_raw[draft_id].item()
            draft_tokens.append(target_id)
        return draft_tokens

    @torch.no_grad()
    def parallel_draft_debug(self, last3_hidden_last: torch.Tensor, target_next: int) -> Tuple[List[int], dict]:
        """Debug version: check all heads including head_0 vs target_next."""
        hidden_concat = last3_hidden_last.squeeze(1).float()
        logits_list = self.parallel_branch(hidden_concat)

        all_predictions = []
        for logits in logits_list:
            draft_id = logits.argmax(dim=-1).item()
            # Use RAW d2t — matches training label construction
            target_id = self.d2t_raw[draft_id].item()
            all_predictions.append(target_id)

        # Check head_0 vs target_next
        head0_matches = (all_predictions[0] == target_next)
        debug_info = {
            "head0_pred": all_predictions[0],
            "target_next": target_next,
            "head0_matches_target_next": head0_matches,
            "all_heads_target_ids": all_predictions,
        }

        # Use heads 1-5 as draft
        draft_tokens = all_predictions[1:]
        return draft_tokens, debug_info

    # ---------- Verify draft tokens ----------
    @torch.no_grad()
    def verify_draft(self, input_ids: torch.Tensor, target_next: int,
                     draft_tokens: List[int]) -> Tuple[int, int]:
        """
        Verify draft tokens with target model.

        Layout: [input_ids(N), target_next(1), draft_tokens(K)]
        Logit at pos N predicts N+1 → should match draft_tokens[0]

        Returns:
            n_accepted: number of consecutively accepted draft tokens
            bonus_token: target model's token at first rejected position
        """
        K = len(draft_tokens)
        extra = [target_next] + draft_tokens
        extra_tensor = torch.tensor([extra], device=self.device)
        verify_ids = torch.cat([input_ids, extra_tensor], dim=1)

        output = self.target_model(
            input_ids=verify_ids,
            output_hidden_states=False,
            return_dict=True,
        )

        input_len = input_ids.shape[1]
        n_accepted = 0

        for i in range(K):
            logit_pos = input_len + i  # Position of target_next + accepted drafts
            target_choice = output.logits[0, logit_pos].argmax().item()
            if target_choice == draft_tokens[i]:
                n_accepted += 1
            else:
                break

        # Bonus token: target's choice at the rejection point
        bonus_pos = input_len + n_accepted
        bonus_token = output.logits[0, bonus_pos].argmax().item()

        return n_accepted, bonus_token

    # ---------- Single prompt evaluation ----------
    @torch.no_grad()
    def run_single_prompt(self, prompt: str, mode: str) -> Dict:
        """
        Run speculative decoding on a single prompt.

        Args:
            prompt: raw query text
            mode: "eagle3" | "parallel" | "dahd"

        Returns:
            Dict with metrics for this prompt.
        """
        # Tokenize
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(
            formatted, return_tensors="pt", truncation=True,
            max_length=self.config.max_prompt_len
        ).to(self.device)
        input_ids = inputs['input_ids']
        prompt_len = input_ids.shape[1]

        # Metrics
        total_accepted = 0
        total_drafted = 0
        total_steps = 0
        generated_tokens = 0
        router_easy_count = 0
        router_hard_count = 0
        easy_accepted_total = 0
        hard_accepted_total = 0

        # Timing
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        while generated_tokens < self.config.max_new_tokens:
            # Target model forward → hidden states
            need_hidden = (mode != "vanilla")
            target_output = self.target_model(
                input_ids=input_ids,
                output_hidden_states=need_hidden,
                return_dict=True,
            )

            # Target's greedy next token
            target_next = target_output.logits[0, -1].argmax().item()
            if target_next == self.tokenizer.eos_token_id:
                generated_tokens += 1
                break

            # Extract hidden states for different branches (skip for vanilla)
            if need_hidden:
                # EAGLE-3 uses aux layers [2, 18, 33]; Parallel branch uses last 3 layers
                last3_hidden = self.extract_last3_hidden(target_output)  # [1, seq, 3*hs]
                last3_hidden_last = last3_hidden[:, -1:, :]  # [1, 1, 3*hs]

            # --- Draft phase ---
            use_parallel = False
            if mode == "vanilla":
                # No drafting in vanilla mode — just accept target_next
                draft_tokens = []
            elif mode == "eagle3":
                draft_tokens = self.eagle3_draft(
                    target_output, input_ids, target_next,
                    K=self.config.eagle3_K
                )
            elif mode == "parallel":
                draft_tokens = self.parallel_draft(last3_hidden_last)
                use_parallel = True
            elif mode == "dahd":
                # Router decides (using last3 hidden as that's what it was trained on)
                shared_repr = self.parallel_branch.get_shared_repr(
                    last3_hidden_last.squeeze(1).to(self.parallel_branch.fc.weight.dtype)
                )  # [1, hs]
                difficulty = self.router(shared_repr.float()).item()
                if difficulty > 0.5:
                    # Easy → use parallel
                    draft_tokens = self.parallel_draft(last3_hidden_last)
                    use_parallel = True
                    router_easy_count += 1
                else:
                    # Hard → use EAGLE-3 AR with shorter K
                    draft_tokens = self.eagle3_draft(
                        target_output, input_ids, target_next,
                        K=self.config.dahd_ar_K
                    )
                    router_hard_count += 1

            K = len(draft_tokens)
            total_drafted += K

            if mode == "vanilla":
                # Vanilla: just accept target_next, no verify needed
                total_steps += 1
                new_tokens = [target_next]
                new_tensor = torch.tensor([new_tokens], device=self.device)
                input_ids = torch.cat([input_ids, new_tensor], dim=1)
                generated_tokens += 1
            else:
                # --- Verify phase ---
                n_accepted, bonus_token = self.verify_draft(input_ids, target_next, draft_tokens)
                total_accepted += n_accepted
                total_steps += 1

                if mode == "dahd":
                    if use_parallel:
                        easy_accepted_total += n_accepted
                    else:
                        hard_accepted_total += n_accepted

                # Advance: target_next + accepted drafts + bonus
                new_tokens = [target_next]
                if n_accepted > 0:
                    new_tokens.extend(draft_tokens[:n_accepted])
                new_tokens.append(bonus_token)

                new_tensor = torch.tensor([new_tokens], device=self.device)
                input_ids = torch.cat([input_ids, new_tensor], dim=1)
                generated_tokens += len(new_tokens)

            # Safety
            if total_steps > 200:
                break

        torch.cuda.synchronize()
        t_end = time.perf_counter()
        wall_time = t_end - t_start

        # Compute metrics
        avg_accepted = total_accepted / max(total_steps, 1)
        tokens_per_second = generated_tokens / max(wall_time, 1e-6)
        # For vanilla: 1 target forward per step
        # For spec-decode: 1 target forward (draft) + 1 target forward (verify) = 2 per step
        if mode == "vanilla":
            total_target_forwards = total_steps  # 1 forward per step
        else:
            total_target_forwards = total_steps * 2  # draft + verify per step
        tokens_per_forward = generated_tokens / max(total_target_forwards, 1)

        result = {
            "mode": mode,
            "prompt_len": prompt_len,
            "total_tokens_generated": generated_tokens,
            "total_steps": total_steps,
            "total_target_forwards": total_target_forwards,
            "total_drafted": total_drafted,
            "total_accepted": total_accepted,
            "avg_accepted_per_step": avg_accepted,
            "avg_tokens_per_target_forward": tokens_per_forward,
            "wall_clock_time_s": wall_time,
            "tokens_per_second": tokens_per_second,
        }

        if mode == "dahd":
            result["router_easy_count"] = router_easy_count
            result["router_hard_count"] = router_hard_count
            result["easy_avg_accepted"] = easy_accepted_total / max(router_easy_count, 1)
            result["hard_avg_accepted"] = hard_accepted_total / max(router_hard_count, 1)
            result["router_easy_ratio"] = router_easy_count / max(total_steps, 1)

        return result

    # ---------- Full evaluation ----------
    def run_evaluation(self):
        """Run full evaluation for all three modes."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Load prompts (last N from data)
        prompts = self._load_prompts()
        total_prompts = self.config.num_prompts
        self.logger.info(f"Loaded {len(prompts)} prompts, using last {total_prompts}")
        eval_prompts = prompts[-total_prompts:]

        modes = ["vanilla", "eagle3", "parallel", "dahd"]
        all_results = {m: [] for m in modes}

        # Warmup
        self.logger.info(f"Warming up with {self.config.warmup_prompts} prompts...")
        for i in range(min(self.config.warmup_prompts, len(eval_prompts))):
            self.run_single_prompt(eval_prompts[i], "eagle3")
        torch.cuda.empty_cache()

        # Run evaluation
        for mode in modes:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Running mode: {mode.upper()}")
            self.logger.info(f"{'='*60}")

            for i, prompt in enumerate(eval_prompts):
                result = self.run_single_prompt(prompt, mode)
                all_results[mode].append(result)

                if (i + 1) % 10 == 0 or i == 0:
                    if mode == "vanilla":
                        avg_tps = np.mean([r["tokens_per_second"] for r in all_results[mode]])
                        self.logger.info(f"  [{i+1}/{total_prompts}] avg_tps={avg_tps:.1f}")
                    else:
                        avg_acc = np.mean([r["avg_accepted_per_step"] for r in all_results[mode]])
                        avg_tps = np.mean([r["tokens_per_second"] for r in all_results[mode]])
                        self.logger.info(
                            f"  [{i+1}/{total_prompts}] "
                            f"avg_accepted={avg_acc:.3f}, avg_tps={avg_tps:.1f}"
                        )

            torch.cuda.empty_cache()

        # Compute aggregate metrics
        summary = self._compute_summary(all_results)

        # Save results
        self._save_results(all_results, summary)

        # Print comparison table
        self._print_comparison(summary)

        # Generate chart
        self._generate_chart(summary)

        return summary

    def _compute_summary(self, all_results: Dict) -> Dict:
        """Compute aggregate metrics for each mode."""
        summary = {}

        # First pass: compute raw metrics
        for mode, results in all_results.items():
            total_tokens = sum(r["total_tokens_generated"] for r in results)
            total_steps = sum(r["total_steps"] for r in results)
            total_forwards = sum(r["total_target_forwards"] for r in results)
            total_time = sum(r["wall_clock_time_s"] for r in results)
            total_accepted = sum(r["total_accepted"] for r in results)
            total_drafted = sum(r["total_drafted"] for r in results)

            avg_accepted_per_step = total_accepted / max(total_steps, 1)
            avg_tokens_per_forward = total_tokens / max(total_forwards, 1)
            overall_tps = total_tokens / max(total_time, 1e-6)
            avg_tokens_per_step = total_tokens / max(total_steps, 1)

            mode_summary = {
                "mode": mode,
                "total_tokens_generated": total_tokens,
                "total_steps": total_steps,
                "total_target_forwards": total_forwards,
                "avg_accepted_per_step": avg_accepted_per_step,
                "avg_tokens_per_step": avg_tokens_per_step,
                "avg_tokens_per_target_forward": avg_tokens_per_forward,
                "wall_clock_time_s": total_time,
                "tokens_per_second": overall_tps,
                "speedup_vs_vanilla": 1.0,  # Will be updated below
                "acceptance_rate": total_accepted / max(total_drafted, 1),
            }

            if mode == "dahd":
                easy_counts = [r.get("router_easy_count", 0) for r in results]
                hard_counts = [r.get("router_hard_count", 0) for r in results]
                mode_summary["router_easy_count"] = sum(easy_counts)
                mode_summary["router_hard_count"] = sum(hard_counts)
                mode_summary["router_easy_ratio"] = (
                    sum(easy_counts) / max(sum(easy_counts) + sum(hard_counts), 1)
                )
                easy_accepted = sum(r.get("router_easy_count", 0) * r.get("easy_avg_accepted", 0)
                                    for r in results)
                hard_accepted = sum(r.get("router_hard_count", 0) * r.get("hard_avg_accepted", 0)
                                    for r in results)
                mode_summary["easy_avg_accepted"] = easy_accepted / max(sum(easy_counts), 1)
                mode_summary["hard_avg_accepted"] = hard_accepted / max(sum(hard_counts), 1)

            summary[mode] = mode_summary

        # Second pass: compute speedup relative to vanilla's wall-clock TPS
        vanilla_tps = summary.get("vanilla", {}).get("tokens_per_second", 1.0)
        for mode in summary:
            if vanilla_tps > 0:
                summary[mode]["speedup_vs_vanilla"] = summary[mode]["tokens_per_second"] / vanilla_tps
            else:
                summary[mode]["speedup_vs_vanilla"] = summary[mode]["avg_tokens_per_step"]

        return summary

    def _save_results(self, all_results: Dict, summary: Dict):
        """Save results to files."""
        out_dir = self.config.output_dir

        # e2e_comparison.json
        with open(os.path.join(out_dir, "e2e_comparison.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # per_prompt_results.jsonl
        with open(os.path.join(out_dir, "per_prompt_results.jsonl"), "w") as f:
            for mode, results in all_results.items():
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        self.logger.info(f"Results saved to {out_dir}")

    def _print_comparison(self, summary: Dict):
        """Print formatted comparison table."""
        print("\n" + "=" * 80)
        print("END-TO-END COMPARISON RESULTS")
        print("=" * 80)
        header = (f"{'Mode':<12} {'AcceptRate':<12} {'Acc/Step':<10} {'Tok/Step':<10} "
                  f"{'Tok/s':<10} {'Speedup':<10} {'Time(s)':<10}")
        print(header)
        print("-" * 80)
        for mode in ["vanilla", "eagle3", "parallel", "dahd"]:
            if mode not in summary:
                continue
            s = summary[mode]
            print(f"{mode:<12} {s['acceptance_rate']:<12.4f} {s['avg_accepted_per_step']:<10.3f} "
                  f"{s['avg_tokens_per_step']:<10.3f} {s['tokens_per_second']:<10.1f} "
                  f"{s['speedup_vs_vanilla']:<10.3f} {s['wall_clock_time_s']:<10.1f}")
        print("=" * 80)

        # DAHD-specific
        if "dahd" in summary:
            s = summary["dahd"]
            print(f"\nDAHD Router Stats:")
            print(f"  Easy (Parallel) count: {s.get('router_easy_count', 0)}, "
                  f"ratio: {s.get('router_easy_ratio', 0):.3f}")
            print(f"  Hard (AR) count: {s.get('router_hard_count', 0)}")
            print(f"  Easy avg accepted: {s.get('easy_avg_accepted', 0):.3f}")
            print(f"  Hard avg accepted: {s.get('hard_avg_accepted', 0):.3f}")

    def _generate_chart(self, summary: Dict):
        """Generate comparison chart."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            modes = [m for m in ["vanilla", "eagle3", "parallel", "dahd"] if m in summary]
            labels_map = {
                "vanilla": "Vanilla\n(AR)",
                "eagle3": "EAGLE-3\n(AR, K=5)",
                "parallel": "Always Parallel\n(Hydra, K=5)",
                "dahd": "DAHD\n(Router)",
            }
            labels = [labels_map[m] for m in modes]

            fig, axes = plt.subplots(1, 3, figsize=(14, 5))

            # 1. Acceptance rate
            acc_rates = [summary[m]["acceptance_rate"] for m in modes]
            bars = axes[0].bar(labels, acc_rates, color=['#2196F3', '#4CAF50', '#FF9800'], alpha=0.8)
            axes[0].set_ylabel("Acceptance Rate")
            axes[0].set_title("Draft Token Acceptance Rate")
            axes[0].set_ylim(0, max(acc_rates) * 1.3)
            for bar, val in zip(bars, acc_rates):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', fontsize=10)

            # 2. Tokens per step (speedup)
            tok_per_step = [summary[m]["avg_tokens_per_step"] for m in modes]
            bars = axes[1].bar(labels, tok_per_step, color=['#2196F3', '#4CAF50', '#FF9800'], alpha=0.8)
            axes[1].set_ylabel("Tokens per Step")
            axes[1].set_title("Avg Tokens per Spec-Decode Step\n(= Speedup vs Vanilla)")
            axes[1].set_ylim(0, max(tok_per_step) * 1.3)
            axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Vanilla (1.0)')
            axes[1].legend()
            for bar, val in zip(bars, tok_per_step):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{val:.2f}', ha='center', fontsize=10)

            # 3. Tokens per second
            tps = [summary[m]["tokens_per_second"] for m in modes]
            bars = axes[2].bar(labels, tps, color=['#2196F3', '#4CAF50', '#FF9800'], alpha=0.8)
            axes[2].set_ylabel("Tokens/s")
            axes[2].set_title("Generation Throughput")
            axes[2].set_ylim(0, max(tps) * 1.3)
            for bar, val in zip(bars, tps):
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{val:.1f}', ha='center', fontsize=10)

            plt.tight_layout()
            chart_path = os.path.join(self.config.output_dir, "comparison_chart.png")
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Chart saved to {chart_path}")

        except Exception as e:
            self.logger.warning(f"Chart generation failed: {e}")

    def _format_prompt(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True,
            )
        return text

    def _load_prompts(self) -> List[str]:
        prompts = []
        with open(self.config.eval_data_path) as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data['query'])
        return prompts


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DAHD vs EAGLE-3 E2E Eval")
    parser.add_argument('--num-prompts', type=int, default=50)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--eagle3-k', type=int, default=5)
    parser.add_argument('--dahd-ar-k', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("E2E")

    config = E2EConfig(
        num_prompts=args.num_prompts,
        max_new_tokens=args.max_new_tokens,
        eagle3_K=args.eagle3_k,
        dahd_ar_K=args.dahd_ar_k,
        warmup_prompts=args.warmup,
    )

    logger.info("=" * 60)
    logger.info("DAHD vs EAGLE-3 End-to-End Speculative Decoding Eval")
    logger.info("=" * 60)
    logger.info(f"  Target model: {config.target_model_path}")
    logger.info(f"  EAGLE-3 K={config.eagle3_K}, Parallel K={config.parallel_K}")
    logger.info(f"  DAHD AR K={config.dahd_ar_K}")
    logger.info(f"  Max new tokens: {config.max_new_tokens}")
    logger.info(f"  Num prompts: {config.num_prompts}")

    evaluator = E2EEvaluator(config)
    evaluator.load_all_models()
    evaluator.run_evaluation()

    logger.info("Done!")


if __name__ == "__main__":
    main()
