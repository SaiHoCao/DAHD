#!/usr/bin/env python3
"""End-to-end comparison: Vanilla / EAGLE-3 / Gumiho-parallel / DAHD-3modal.

Key fixes vs v1:
  1. Target model uses KV cache throughout → verify cost O(K+1) not O(N²).
  2. CUDA synchronize around timed sections → accurate GPU wall-clock.
  3. Parallel branch uses Gumiho-style input: fc(cat(embed(target_next), hidden_t)).
  4. DAHD uses 3-modal routing: Easy / Medium / Hard with dynamic K.
  5. Metrics: report avg_tokens_per_step (= n_accepted + 2) alongside n_accepted.

Architecture compatibility:
  - If 'checkpoints/medusa/gumiho_best.pt' exists → use new GumihoBranch.
  - Falls back to old MedusaModel if only legacy checkpoint is found.

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/phase4_e2e/run_medusa_e2e_comparison.py
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments/phase1_profiling"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments/phase2_architecture"))

from run_eagle3_full_profiling import (
    Eagle3DraftHead, ProfilingConfig, RMSNorm,
    precompute_freqs_cis, apply_rotary_emb, rotate_half,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class E2EConfig:
    # Paths
    target_model_path:    str = "/mnt/nas1/hf/Qwen3-8B/"
    eagle3_weights_path:  str = "/mnt/nas1/hf/qwen3_8b_eagle3/pytorch_model.bin"
    eagle3_config_path:   str = "/mnt/nas1/hf/qwen3_8b_eagle3/config.json"
    eval_data_path:       str = "/mnt/nas1/hf/qwen3_8b_eagle3/eagle_data.jsonl"

    gumiho_ckpt: str = str(PROJECT_ROOT / "checkpoints/medusa/gumiho_best.pt")
    legacy_ckpt: str = str(PROJECT_ROOT / "checkpoints/medusa/medusa_best.pt")

    # Architecture
    hidden_size:          int         = 4096
    vocab_size:           int         = 151936
    num_attention_heads:  int         = 32
    num_kv_heads:         int         = 8
    head_dim:             int         = 128
    intermediate_size:    int         = 12288
    draft_vocab_size:     int         = 32000
    rms_norm_eps:         float       = 1e-6
    rope_theta:           float       = 1_000_000.0
    num_target_layers:    int         = 36
    aux_layer_indices:    List[int]   = field(default_factory=lambda: [2, 18, 33])

    # Gumiho parallel config
    gumiho_num_heads: int = 4    # heads predicting t+2..t+5
    gumiho_mlp_depth: int = 3

    # 3-modal DAHD thresholds and K
    easy_threshold:  float = 0.75   # top1_prob > threshold → Easy
    hard_threshold:  float = 0.50   # top1_prob < threshold → Hard
    k_easy:    int = 4   # Gumiho parallel, K-1=4 actual drafts past target_next
    k_medium:  int = 3   # EAGLE AR
    k_hard:    int = 2   # EAGLE AR (short to avoid tail rejection)
    eagle_k:   int = 5   # standalone EAGLE baseline

    # Evaluation
    max_new_tokens:    int = 128
    num_prompts:       int = 50
    max_prompt_len:    int = 512
    warmup_prompts:    int = 3
    per_prompt_timeout: int = 180

    # Output
    output_dir: str = str(PROJECT_ROOT / "results/phase4_medusa")

    device: str   = "cuda"
    dtype:  torch.dtype = torch.bfloat16


# ── KV-cache utilities ────────────────────────────────────────────────────────

def truncate_past_kv(past_kv, n_keep: int):
    """Keep only the first `n_keep` positions in a HF past_key_values object.

    Handles:
      - transformers 5.x DynamicCache  (has .crop() + .layers with 3-item iter)
      - transformers 4.x DynamicCache  (iterates as (k, v) per layer)
      - legacy tuple-of-tuples format
    """
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            import copy
            new_cache = copy.deepcopy(past_kv)
            new_cache.crop(n_keep)          # in-place on the deep copy
            return new_cache
    except (ImportError, AttributeError):
        pass

    # Legacy tuple-of-tuples: each element is (key, value)
    result = []
    for layer_data in past_kv:
        if isinstance(layer_data, (tuple, list)) and len(layer_data) >= 2:
            k, v = layer_data[0], layer_data[1]
            result.append((k[:, :, :n_keep, :], v[:, :, :n_keep, :]))
    return tuple(result)


# ── Gumiho / legacy parallel branch ──────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.linear = nn.Linear(size, size)
        nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class GumihoBranch(nn.Module):
    """Gumiho-style parallel heads: predict t+2..t+K+1 given (hidden_t, embed_{t+1})."""

    def __init__(self, hidden_size: int, vocab_size: int,
                 num_heads: int = 4, mlp_depth: int = 3):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        # Zero-init keeps early training stable (same as Gumiho's noResBlock)
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.fc.weight)
        self.heads = nn.ModuleList([
            nn.Sequential(*[ResBlock(hidden_size) for _ in range(mlp_depth)])
            for _ in range(num_heads)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._compiled = False

    def compile_for_inference(self):
        """torch.compile the MLP heads for faster parallel forward (Gumiho trick)."""
        if not self._compiled:
            for i, head in enumerate(self.heads):
                self.heads[i] = torch.compile(head, mode="reduce-overhead")
            self._compiled = True

    @torch.no_grad()
    def draft(self, hidden_t: torch.Tensor, target_next_id: int) -> List[int]:
        """
        Args:
            hidden_t:       [1, hidden_size]  – last-layer hidden at last accepted position
            target_next_id: int               – the target model's greedy next token (t+1)
        Returns:
            list of K predicted token IDs for positions t+2, t+3, ..., t+K+1
        """
        nxt = torch.tensor([[target_next_id]], device=hidden_t.device)
        embed = self.embed_tokens(nxt).squeeze(1)           # [1, hidden_size]
        x = self.fc(torch.cat([embed, hidden_t], dim=-1))   # [1, hidden_size]
        return [self.lm_head(h(x))[0].argmax().item() for h in self.heads]


class LegacyMedusaHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.linear  = nn.Linear(hidden_size, hidden_size)
        self.act     = nn.SiLU()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return self.lm_head(x + self.act(self.linear(x)))


class LegacyMedusaModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_heads=5):
        super().__init__()
        self.heads   = nn.ModuleList([LegacyMedusaHead(hidden_size, vocab_size) for _ in range(num_heads)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @torch.no_grad()
    def draft(self, hidden_t: torch.Tensor, skip_head0: bool = True) -> List[int]:
        all_logits = [h(hidden_t) for h in self.heads]
        start = 1 if skip_head0 else 0
        return [lg[0].argmax().item() for lg in all_logits[start:]]


# ── Engine ────────────────────────────────────────────────────────────────────

class SpecDecEngine:
    """Unified spec-dec engine with proper KV-cache management."""

    def __init__(self, config: E2EConfig):
        self.cfg    = config
        self.device = config.device
        self.dtype  = config.dtype
        self.log    = logging.getLogger("E2E")

    # ── Model loading ────────────────────────────────────────────────────────

    def load_models(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.log.info("Loading tokenizer …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.target_model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.log.info("Loading target model (Qwen3-8B) …")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.target_model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.target_model.eval()
        self.embed_tokens = self.target_model.model.embed_tokens

        self.log.info("Loading EAGLE-3 draft head …")
        eagle_cfg = ProfilingConfig(
            hidden_size=self.cfg.hidden_size,
            num_attention_heads=self.cfg.num_attention_heads,
            num_kv_heads=self.cfg.num_kv_heads,
            head_dim=self.cfg.head_dim,
            intermediate_size=self.cfg.intermediate_size,
            draft_vocab_size=self.cfg.draft_vocab_size,
            target_vocab_size=self.cfg.vocab_size,
            rms_norm_eps=self.cfg.rms_norm_eps,
            rope_theta=self.cfg.rope_theta,
            aux_layer_indices=self.cfg.aux_layer_indices,
        )
        self.eagle_head = Eagle3DraftHead(eagle_cfg)
        eagle_state = torch.load(
            self.cfg.eagle3_weights_path, map_location="cpu", weights_only=False
        )
        self.eagle_head.load_weights(eagle_state)
        self.eagle_head = self.eagle_head.to(device=self.device, dtype=self.dtype)
        self.eagle_head.eval()
        self.eagle_head.init_rope(max_seq_len=2048)
        self.eagle_head.d2t = self.eagle_head.d2t.to(self.device)

        # t2d mapping for EAGLE verification
        self.t2d = torch.full((self.cfg.vocab_size,), -1, dtype=torch.long, device=self.device)
        d2t = self.eagle_head.d2t.long()
        self.t2d[d2t] = torch.arange(self.cfg.draft_vocab_size, device=self.device)

        # ── Parallel branch ──────────────────────────────────────────────────
        self.log.info("Loading parallel branch …")
        if Path(self.cfg.gumiho_ckpt).exists():
            self.log.info(f"  → GumihoBranch from {self.cfg.gumiho_ckpt}")
            self.parallel_branch = GumihoBranch(
                hidden_size=self.cfg.hidden_size,
                vocab_size=self.cfg.vocab_size,
                num_heads=self.cfg.gumiho_num_heads,
                mlp_depth=self.cfg.gumiho_mlp_depth,
            )
            ckpt = torch.load(self.cfg.gumiho_ckpt, map_location="cpu", weights_only=False)
            self.parallel_branch.load_state_dict(ckpt["model_state_dict"])
            self.parallel_type = "gumiho"
        elif Path(self.cfg.legacy_ckpt).exists():
            self.log.warning(f"  → Legacy MedusaModel from {self.cfg.legacy_ckpt}")
            self.log.warning("    Retrain with train_medusa_heads.py to get GumihoBranch for better results.")
            self.parallel_branch = LegacyMedusaModel(
                hidden_size=self.cfg.hidden_size,
                vocab_size=self.cfg.vocab_size,
                num_heads=5,
            )
            ckpt = torch.load(self.cfg.legacy_ckpt, map_location="cpu", weights_only=False)
            # Legacy checkpoint has ResBlock + shared lm_head; adapt key names
            try:
                self.parallel_branch.load_state_dict(ckpt["model_state_dict"], strict=False)
            except Exception as e:
                self.log.warning(f"    Could not load legacy weights ({e}); parallel branch random-init.")
            self.parallel_type = "legacy"
        else:
            self.log.warning("No parallel branch checkpoint found — DAHD will only use EAGLE.")
            self.parallel_branch = None
            self.parallel_type   = "none"

        if self.parallel_branch is not None:
            self.parallel_branch = self.parallel_branch.to(device=self.device, dtype=self.dtype)
            self.parallel_branch.eval()
            # torch.compile for faster MLP forward (Gumiho reports ~15% gain)
            if self.parallel_type == "gumiho":
                try:
                    self.parallel_branch.compile_for_inference()
                    self.log.info("  GumihoBranch heads compiled with torch.compile.")
                except Exception as e:
                    self.log.warning(f"  torch.compile skipped: {e}")

        self.log.info(f"All models loaded. GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Prompt loading ───────────────────────────────────────────────────────

    def load_prompts(self) -> List[str]:
        prompts = []
        with open(self.cfg.eval_data_path) as f:
            for line in f:
                prompts.append(json.loads(line.strip())["query"])
        return prompts[-self.cfg.num_prompts:]

    def format_and_tokenize(self, prompts: List[str]) -> List[torch.Tensor]:
        all_ids = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            try:
                txt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception:
                txt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            ids = self.tokenizer(
                txt, return_tensors="pt", truncation=True,
                max_length=self.cfg.max_prompt_len,
            )["input_ids"].to(self.device)
            all_ids.append(ids)
        return all_ids

    # ── EAGLE helpers ────────────────────────────────────────────────────────

    def extract_aux_hidden(self, model_output) -> torch.Tensor:
        all_hs = model_output.hidden_states
        return torch.cat([all_hs[i] for i in self.cfg.aux_layer_indices], dim=-1)

    @torch.no_grad()
    def eagle_draft(
        self,
        eagle_aux:    torch.Tensor,
        eagle_kv:     Optional[tuple],
        prev_token:   int,
        start_pos:    int,
        K:            int,
    ) -> List[int]:
        hidden = eagle_aux
        kv     = eagle_kv
        tokens = []

        for k in range(K):
            emb  = self.embed_tokens(torch.tensor([[prev_token]], device=self.device))
            pos  = torch.tensor([start_pos + k], device=self.device)
            logits, aux_hidden, kv = self.eagle_head(
                hidden_states=hidden, embeds=emb,
                positions=pos, kv_cache=kv, is_first_step=(k == 0),
            )
            draft_logit    = logits[0, -1, :]
            draft_token_id = draft_logit.argmax().item()
            target_tok     = self.eagle_head.d2t[draft_token_id].item()
            tokens.append(target_tok)
            prev_token = target_tok
            hidden     = aux_hidden

        return tokens

    @torch.no_grad()
    def eagle_prefill(
        self, target_output, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, tuple]:
        aux_all   = self.extract_aux_hidden(target_output)
        embs_all  = self.embed_tokens(input_ids)
        N         = input_ids.shape[1]
        positions = torch.arange(N, device=self.device)
        _, aux_hidden, kv = self.eagle_head(
            hidden_states=aux_all, embeds=embs_all,
            positions=positions, kv_cache=None, is_first_step=True,
        )
        return aux_hidden[:, -1:, :], kv

    # ── Parallel branch draft ────────────────────────────────────────────────

    @torch.no_grad()
    def parallel_draft(self, hidden_t: torch.Tensor, target_next: int) -> List[int]:
        """Return draft tokens for positions t+2, t+3, ... (gumiho-style)."""
        if self.parallel_branch is None:
            return []
        if self.parallel_type == "gumiho":
            return self.parallel_branch.draft(hidden_t.unsqueeze(0), target_next)
        else:
            # Legacy: skip head_0 (it predicts t+1 which we already know)
            return self.parallel_branch.draft(
                hidden_t.unsqueeze(0).to(self.dtype), skip_head0=True
            )

    # ── Core verify step with KV cache ───────────────────────────────────────

    @torch.no_grad()
    def verify_with_kv(
        self,
        target_kv:    tuple,
        current_len:  int,
        target_next:  int,
        draft_tokens: List[int],
    ) -> Tuple[int, int, tuple, int, torch.Tensor]:
        """
        Verify draft tokens using the maintained target KV cache.

        Args:
            target_kv:    HF past_key_values (covers positions 0..current_len-1)
            current_len:  number of tokens already in the KV cache
            target_next:  the target model's greedy next token (position current_len)
            draft_tokens: K predicted tokens for positions current_len+1, ...

        Returns:
            n_accepted   – accepted draft count
            bonus_token  – target's next token after acceptance point
            new_kv       – updated KV cache (covers 0..current_len+n_accepted)
            new_len      – = current_len + n_accepted + 1
            last_hidden  – hidden state at the acceptance-point position
                           (use as hidden_t for next step's parallel branch)
            bonus_logit  – logits at acceptance point (for routing)
            verify_aux   – aux hidden states [1, K+1, 3*hs] from verify output
            verify_ids   – [1, K+1] token IDs fed during verify
        """
        K          = len(draft_tokens)
        verify_ids = torch.tensor(
            [[target_next] + draft_tokens], device=self.device
        )   # [1, K+1]

        verify_out = self.target_model(
            verify_ids,
            past_key_values=target_kv,
            use_cache=True,
            output_hidden_states=True,
        )

        # Greedy accept: logit[i] predicts what follows verify_ids[i]
        n_accepted = 0
        for i, dt in enumerate(draft_tokens):
            if verify_out.logits[0, i].argmax().item() == dt:
                n_accepted += 1
            else:
                break

        # logit at n_accepted position → bonus token AND next-step routing signal
        bonus_logit = verify_out.logits[0, n_accepted]   # [vocab_size]
        bonus       = bonus_logit.argmax().item()

        # Truncate KV to keep positions 0..current_len+n_accepted (inclusive)
        new_len = current_len + n_accepted + 1   # +1 for target_next
        new_kv  = truncate_past_kv(verify_out.past_key_values, new_len)

        # Hidden at the acceptance-point position (used as hidden_t next step)
        last_hidden = verify_out.hidden_states[-1][0, n_accepted]  # [hidden_size]

        # Extract aux hidden states from verify output for EAGLE KV update
        verify_aux = torch.cat(
            [verify_out.hidden_states[i] for i in self.cfg.aux_layer_indices],
            dim=-1,
        )  # [1, K+1, 3*hidden_size]

        return n_accepted, bonus, new_kv, new_len, last_hidden, bonus_logit, verify_aux, verify_ids

    # ── Generation loops ─────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_vanilla(self, input_ids: torch.Tensor) -> dict:
        """Vanilla AR with KV cache — proper O(N·d²) baseline."""
        eos = self.tokenizer.eos_token_id

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Prefill
        out      = self.target_model(input_ids, use_cache=True)
        past_kv  = out.past_key_values
        n_tokens = 0

        while n_tokens < self.cfg.max_new_tokens:
            next_tok = out.logits[0, -1].argmax().item()
            if next_tok == eos:
                break
            out     = self.target_model(
                torch.tensor([[next_tok]], device=self.device),
                past_key_values=past_kv, use_cache=True,
            )
            past_kv  = out.past_key_values
            n_tokens += 1

            if time.perf_counter() - t0 > self.cfg.per_prompt_timeout:
                break

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        return {
            "tokens_generated": n_tokens,
            "wall_time":        elapsed,
            "tokens_per_sec":   n_tokens / max(elapsed, 1e-6),
            "avg_accepted":     0.0,
            "avg_tokens_per_step": 1.0,
            "num_steps":        n_tokens,
        }

    @torch.no_grad()
    def update_eagle_kv(
        self,
        eagle_kv:     Optional[tuple],
        verify_aux:   torch.Tensor,
        verify_ids:   torch.Tensor,
        n_accepted:   int,
        old_len:      int,
    ) -> Tuple[torch.Tensor, tuple]:
        """
        Incrementally update EAGLE KV cache with accepted tokens from verify.

        Args:
            eagle_kv:   current EAGLE KV cache (covers 0..old_len-1)
            verify_aux: [1, K+1, 3*hs] aux hidden states from verify output
            verify_ids: [1, K+1] token IDs from verify
            n_accepted: number of accepted draft tokens
            old_len:    position offset (= current_len before verify)

        Returns:
            eagle_aux: [1, 1, hs] aux hidden for next draft
            eagle_kv:  updated EAGLE KV cache (covers 0..old_len+n_accepted)
        """
        n_update = n_accepted + 1  # target_next + n_accepted drafts
        eagle_aux_step = None

        for i in range(n_update):
            pos_i = old_len + i
            aux_i = verify_aux[:, i:i+1, :]   # [1, 1, 3*hs]
            emb_i = self.embed_tokens(verify_ids[:, i:i+1])  # [1, 1, hs]
            pos_tensor = torch.tensor([pos_i], device=self.device)
            _, eagle_aux_step, eagle_kv = self.eagle_head(
                hidden_states=aux_i, embeds=emb_i,
                positions=pos_tensor, kv_cache=eagle_kv,
                is_first_step=True,
            )

        return eagle_aux_step, eagle_kv

    @torch.no_grad()
    def generate_eagle3(self, input_ids: torch.Tensor) -> dict:
        """EAGLE-3 with target KV cache."""
        eos = self.tokenizer.eos_token_id
        K   = self.cfg.eagle_k

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Prefill
        prefill = self.target_model(
            input_ids, use_cache=True, output_hidden_states=True
        )
        target_kv   = prefill.past_key_values
        current_len = input_ids.shape[1]
        target_next = prefill.logits[0, -1].argmax().item()
        last_hidden = prefill.hidden_states[-1][0, -1]

        # Rebuild EAGLE KV from prefill
        eagle_aux, eagle_kv = self.eagle_prefill(prefill, input_ids)

        n_tokens = 0; n_acc_total = 0; n_steps = 0

        while n_tokens < self.cfg.max_new_tokens:
            if target_next == eos:
                break

            drafts = self.eagle_draft(
                eagle_aux, eagle_kv, target_next, current_len, K
            )

            old_len = current_len  # save before verify updates it
            n_acc, bonus, target_kv, current_len, last_hidden, _, verify_aux, verify_ids = \
                self.verify_with_kv(target_kv, old_len, target_next, drafts)

            n_tokens    += n_acc + 2    # target_next + accepted drafts + bonus
            n_acc_total += n_acc
            n_steps     += 1
            target_next  = bonus

            if bonus == eos:
                break
            if time.perf_counter() - t0 > self.cfg.per_prompt_timeout:
                break

            # Update EAGLE KV incrementally with accepted tokens' true hidden states
            eagle_aux, eagle_kv = self.update_eagle_kv(
                eagle_kv, verify_aux, verify_ids, n_acc, old_len
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        return {
            "tokens_generated":    n_tokens,
            "wall_time":           elapsed,
            "tokens_per_sec":      n_tokens / max(elapsed, 1e-6),
            "avg_accepted":        n_acc_total / max(n_steps, 1),
            "avg_tokens_per_step": n_tokens    / max(n_steps, 1),
            "num_steps":           n_steps,
        }

    @torch.no_grad()
    def generate_parallel(self, input_ids: torch.Tensor) -> dict:
        """Gumiho-style parallel branch with target KV cache."""
        eos = self.tokenizer.eos_token_id

        if self.parallel_branch is None:
            return {"tokens_generated": 0, "wall_time": 0, "tokens_per_sec": 0,
                    "avg_accepted": 0, "avg_tokens_per_step": 0, "num_steps": 0}

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        prefill = self.target_model(
            input_ids, use_cache=True, output_hidden_states=True
        )
        target_kv   = prefill.past_key_values
        current_len = input_ids.shape[1]
        target_next = prefill.logits[0, -1].argmax().item()
        last_hidden = prefill.hidden_states[-1][0, -1]

        n_tokens = 0; n_acc_total = 0; n_steps = 0

        while n_tokens < self.cfg.max_new_tokens:
            if target_next == eos:
                break

            drafts = self.parallel_draft(last_hidden, target_next)

            n_acc, bonus, target_kv, current_len, last_hidden, _, _, _ = \
                self.verify_with_kv(target_kv, current_len, target_next, drafts)

            n_tokens    += n_acc + 2
            n_acc_total += n_acc
            n_steps     += 1
            target_next  = bonus

            if bonus == eos:
                break
            if time.perf_counter() - t0 > self.cfg.per_prompt_timeout:
                break

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        return {
            "tokens_generated":    n_tokens,
            "wall_time":           elapsed,
            "tokens_per_sec":      n_tokens / max(elapsed, 1e-6),
            "avg_accepted":        n_acc_total / max(n_steps, 1),
            "avg_tokens_per_step": n_tokens    / max(n_steps, 1),
            "num_steps":           n_steps,
        }

    @torch.no_grad()
    def generate_dahd(self, input_ids: torch.Tensor) -> dict:
        """DAHD 3-modal: Easy=Gumiho-parallel, Medium=EAGLE(K=3), Hard=EAGLE(K=2)."""
        eos = self.tokenizer.eos_token_id
        T_easy = self.cfg.easy_threshold
        T_hard = self.cfg.hard_threshold

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        prefill = self.target_model(
            input_ids, use_cache=True, output_hidden_states=True
        )
        target_kv   = prefill.past_key_values
        current_len = input_ids.shape[1]

        last_logits = prefill.logits[0, -1]
        target_next = last_logits.argmax().item()
        last_hidden = prefill.hidden_states[-1][0, -1]
        eagle_aux, eagle_kv = self.eagle_prefill(prefill, input_ids)

        n_tokens = 0; n_acc_total = 0; n_steps = 0
        cnt_easy = 0; cnt_med = 0; cnt_hard = 0

        while n_tokens < self.cfg.max_new_tokens:
            if target_next == eos:
                break

            # ── Route ────────────────────────────────────────────────────────
            top1_prob = torch.softmax(last_logits.float(), dim=-1).max().item()

            if top1_prob > T_easy and self.parallel_branch is not None:
                # Easy: Gumiho parallel (O(1) draft cost)
                drafts = self.parallel_draft(last_hidden, target_next)
                cnt_easy += 1
            elif top1_prob > T_hard:
                # Medium: EAGLE AR, K=3
                drafts = self.eagle_draft(
                    eagle_aux, eagle_kv, target_next, current_len, self.cfg.k_medium
                )
                cnt_med += 1
            else:
                # Hard: EAGLE AR, K=2 (short chain to avoid tail rejection)
                drafts = self.eagle_draft(
                    eagle_aux, eagle_kv, target_next, current_len, self.cfg.k_hard
                )
                cnt_hard += 1

            old_len = current_len  # save before verify updates it
            n_acc, bonus, target_kv, current_len, last_hidden, bonus_logit, verify_aux, verify_ids = \
                self.verify_with_kv(target_kv, old_len, target_next, drafts)

            n_tokens    += n_acc + 2
            n_acc_total += n_acc
            n_steps     += 1
            target_next  = bonus

            if bonus == eos:
                break
            if time.perf_counter() - t0 > self.cfg.per_prompt_timeout:
                break

            # Update EAGLE KV incrementally with accepted tokens' true hidden states
            eagle_aux, eagle_kv = self.update_eagle_kv(
                eagle_kv, verify_aux, verify_ids, n_acc, old_len
            )
            # bonus_logit from verify already tells us next-step difficulty — free
            last_logits = bonus_logit

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_routed = cnt_easy + cnt_med + cnt_hard
        return {
            "tokens_generated":    n_tokens,
            "wall_time":           elapsed,
            "tokens_per_sec":      n_tokens / max(elapsed, 1e-6),
            "avg_accepted":        n_acc_total / max(n_steps, 1),
            "avg_tokens_per_step": n_tokens    / max(n_steps, 1),
            "num_steps":           n_steps,
            "easy_ratio":          cnt_easy / max(total_routed, 1),
            "medium_ratio":        cnt_med  / max(total_routed, 1),
            "hard_ratio":          cnt_hard / max(total_routed, 1),
        }


# ── Evaluation harness ────────────────────────────────────────────────────────

def run_method(engine, method_name: str, method_fn, all_input_ids: List[torch.Tensor],
               warmup: int, logger) -> List[dict]:
    results = []

    # Warmup
    for i in range(min(warmup, len(all_input_ids))):
        try:
            method_fn(all_input_ids[i])
        except Exception as e:
            logger.warning(f"  Warmup {i} failed: {e}")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Evaluate
    failed = 0
    for i in tqdm(range(len(all_input_ids)), desc=f"  {method_name}"):
        try:
            r = method_fn(all_input_ids[i])
            results.append(r)
        except Exception as e:
            failed += 1
            logger.warning(f"  Prompt {i} failed ({method_name}): {e}")
            results.append({
                "tokens_generated": 0, "wall_time": 0,
                "tokens_per_sec": 0, "avg_accepted": 0,
                "avg_tokens_per_step": 0, "num_steps": 0,
            })
            torch.cuda.empty_cache()

    valid = [r for r in results if r["tokens_generated"] > 0]
    if valid:
        avg_tps  = np.mean([r["tokens_per_sec"]      for r in valid])
        avg_acc  = np.mean([r["avg_accepted"]         for r in valid])
        avg_step = np.mean([r["avg_tokens_per_step"]  for r in valid])
        print(f"  {method_name:<12}: {avg_tps:6.1f} tok/s  "
              f"acc/step={avg_acc:.3f}  tok/step={avg_step:.2f}  failed={failed}")
    torch.cuda.empty_cache()
    return results


def summarise(results: List[dict], vanilla_tps: float) -> dict:
    valid = [r for r in results if r["tokens_generated"] > 0]
    if not valid:
        return {}
    tps_list  = [r["tokens_per_sec"]     for r in valid]
    acc_list  = [r["avg_accepted"]        for r in valid]
    step_list = [r["avg_tokens_per_step"] for r in valid]
    avg_tps   = float(np.mean(tps_list))
    s = {
        "avg_tokens_per_sec":      avg_tps,
        "std_tokens_per_sec":      float(np.std(tps_list)),
        "avg_accepted_per_step":   float(np.mean(acc_list)),
        "avg_tokens_per_step":     float(np.mean(step_list)),
        "speedup_vs_vanilla":      avg_tps / max(vanilla_tps, 1e-6),
        "num_prompts":             len(valid),
    }
    # Extra routing stats for DAHD
    if "easy_ratio" in valid[0]:
        s["easy_ratio"]   = float(np.mean([r.get("easy_ratio",   0) for r in valid]))
        s["medium_ratio"] = float(np.mean([r.get("medium_ratio", 0) for r in valid]))
        s["hard_ratio"]   = float(np.mean([r.get("hard_ratio",   0) for r in valid]))
    return s


# ── Main ─────────────────────────────────────────────────────────────────────

def run_evaluation():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    logger = logging.getLogger("E2E")

    cfg = E2EConfig()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("End-to-End Comparison (v2 — with KV cache + Gumiho branch + 3-modal DAHD)")
    print("=" * 70)
    print(f"  Target   : {cfg.target_model_path}")
    print(f"  EAGLE-3  : {cfg.eagle3_weights_path}")
    print(f"  Parallel : gumiho_best.pt (or legacy medusa_best.pt)")
    print(f"  Prompts  : {cfg.num_prompts}  |  MaxNewTokens: {cfg.max_new_tokens}")
    print(f"  DAHD thresholds: easy>{cfg.easy_threshold} (K={cfg.k_easy}), "
          f"medium ({cfg.hard_threshold}–{cfg.easy_threshold}) (K={cfg.k_medium}), "
          f"hard<{cfg.hard_threshold} (K={cfg.k_hard})")
    print()

    engine = SpecDecEngine(cfg)
    engine.load_models()

    prompts    = engine.load_prompts()
    all_ids    = engine.format_and_tokenize(prompts)
    logger.info(f"Loaded {len(all_ids)} prompts")

    methods = {
        "vanilla":  engine.generate_vanilla,
        "eagle3":   engine.generate_eagle3,
        "parallel": engine.generate_parallel,
        "dahd":     engine.generate_dahd,
    }

    all_results = {}
    for name, fn in methods.items():
        print(f"\n── {name.upper()} ─────────────────────────────────────────────")
        all_results[name] = run_method(
            engine, name, fn, all_ids,
            warmup=cfg.warmup_prompts, logger=logger,
        )
        # Save incremental
        with open(Path(cfg.output_dir) / f"incremental_{name}.json", "w") as f:
            json.dump(all_results[name], f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    vanilla_valid = [r for r in all_results["vanilla"] if r["tokens_generated"] > 0]
    vanilla_tps   = float(np.mean([r["tokens_per_sec"] for r in vanilla_valid])) \
                    if vanilla_valid else 1.0

    summary = {
        name: summarise(res, vanilla_tps)
        for name, res in all_results.items()
    }

    print("\n" + "=" * 70)
    print(f"{'Method':<12} {'tok/s':>8} {'speedup':>9} {'acc/step':>10} {'tok/step':>10}")
    print("─" * 55)
    for name, s in summary.items():
        if not s:
            continue
        routing = ""
        if "easy_ratio" in s:
            routing = (f"  [easy={s['easy_ratio']:.0%} "
                       f"med={s['medium_ratio']:.0%} "
                       f"hard={s['hard_ratio']:.0%}]")
        print(f"{name:<12} {s['avg_tokens_per_sec']:>8.1f} "
              f"{s['speedup_vs_vanilla']:>8.3f}x "
              f"{s['avg_accepted_per_step']:>10.3f} "
              f"{s['avg_tokens_per_step']:>10.2f}"
              f"{routing}")
    print("=" * 70)

    note = (
        "Note: avg_accepted_per_step counts only accepted DRAFT tokens "
        "(excludes target_next and bonus). avg_tokens_per_step = acc + 2."
    )
    print(f"\n{note}")

    save_data = {
        "config": {
            "target_model": cfg.target_model_path,
            "eagle_k": cfg.eagle_k,
            "k_easy": cfg.k_easy, "k_medium": cfg.k_medium, "k_hard": cfg.k_hard,
            "easy_threshold": cfg.easy_threshold,
            "hard_threshold": cfg.hard_threshold,
            "max_new_tokens": cfg.max_new_tokens,
            "num_prompts":    cfg.num_prompts,
            "parallel_type":  engine.parallel_type,
        },
        "summary":    summary,
        "per_prompt": {n: r for n, r in all_results.items()},
    }
    out_path = Path(cfg.output_dir) / "e2e_comparison_v2.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved: {out_path}")

    return summary


if __name__ == "__main__":
    run_evaluation()
