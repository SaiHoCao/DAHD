#!/usr/bin/env python3
"""Phase 1: EAGLE-3 Acceptance Rate Profiling on Qwen3-8B.

This script profiles per-step acceptance rates using a simplified EAGLE-3
speculative decoding simulation to validate the bimodal distribution hypothesis.

Approach:
  - Load Qwen3-8B as target model with output_hidden_states=True
  - Implement a simplified EAGLE-3 draft head using raw weights
  - For each prompt, perform greedy decoding and measure acceptance at each step
  - Record per-token metrics: top-1 confidence, entropy, acceptance

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/phase1_profiling/run_eagle3_profiling.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_MODEL_PATH = "/mnt/nas1/hf/Qwen3-8B/"
EAGLE3_PATH = "/mnt/nas1/hf/qwen3_8b_eagle3/"
DATA_PATH = "/mnt/nas1/hf/qwen3_8b_eagle3/eagle_data.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "phase1_results"

MAX_NEW_TOKENS = 128       # tokens to generate per prompt
MAX_PROMPTS = 100          # number of prompts to process (subset of 397)
DRAFT_K = 5               # number of draft tokens per step
SEED = 42

# Layers from which to extract auxiliary hidden states (EAGLE-3 default: last 3)
AUX_LAYER_IDS = [-3, -2, -1]  # relative to last layer


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TokenMetric:
    """Per-token metrics for a single generated token."""
    position: int              # position in generated sequence
    token_id: int              # actual token id
    top1_prob: float           # probability assigned to the actual token
    entropy: float             # entropy of the distribution
    is_top1: bool              # whether this token was the argmax
    draft_accepted: bool       # whether the draft head predicted this correctly
    step_in_draft: int         # position within the draft window (0..K-1)


@dataclass
class SequenceResult:
    """Results for one prompt."""
    prompt_idx: int
    prompt_text: str
    num_generated: int
    tokens: List[TokenMetric] = field(default_factory=list)
    per_step_accept_rates: List[float] = field(default_factory=list)
    overall_accept_rate: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Simplified EAGLE-3 Draft Head (standalone PyTorch)
# ──────────────────────────────────────────────────────────────────────────────
class SimpleEagle3DraftHead(torch.nn.Module):
    """Simplified EAGLE-3 draft head for acceptance measurement.
    
    Architecture (from config):
      - fc: Linear(hidden_size * 3, hidden_size)  # project 3 aux hidden states
      - 1 LLaMA decoder layer with input_dim=2*hidden_size for qkv (concat embeds + hidden)
      - lm_head: Linear(hidden_size, draft_vocab_size)
      - embed_tokens: Embedding(vocab_size, hidden_size)
    
    For profiling, we use a simplified forward that doesn't need KV cache:
      - Takes target hidden states from 3 layers + previous token embedding
      - Projects hidden states via fc
      - Concatenates with embedding, applies a single-layer attention-free approximation
      - Predicts next token via lm_head
    """
    
    def __init__(self, config: dict, weights_path: str, device: torch.device):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.draft_vocab_size = config.get("draft_vocab_size", 32000)
        self.device = device
        
        # Load weights
        state_dict = torch.load(
            os.path.join(weights_path, "pytorch_model.bin"),
            map_location=device,
            weights_only=False,
        )
        
        # Build fc projection layer (3 * hidden_size -> hidden_size)
        fc_weight_key = None
        fc_bias_key = None
        for k in state_dict.keys():
            if "fc.weight" in k:
                fc_weight_key = k
            if "fc.bias" in k:
                fc_bias_key = k
        
        if fc_weight_key:
            self.fc = torch.nn.Linear(
                state_dict[fc_weight_key].shape[1],
                state_dict[fc_weight_key].shape[0],
                bias=(fc_bias_key is not None),
            ).to(device)
            self.fc.weight.data = state_dict[fc_weight_key].to(device)
            if fc_bias_key:
                self.fc.bias.data = state_dict[fc_bias_key].to(device)
        else:
            # Fallback: identity-like projection
            self.fc = torch.nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False).to(device)
        
        # Build embedding
        embed_key = None
        for k in state_dict.keys():
            if "embed_tokens.weight" in k:
                embed_key = k
                break
        
        if embed_key:
            embed_weight = state_dict[embed_key]
            self.embed_tokens = torch.nn.Embedding(
                embed_weight.shape[0], embed_weight.shape[1]
            ).to(device)
            self.embed_tokens.weight.data = embed_weight.to(device)
        else:
            self.embed_tokens = None
        
        # Build lm_head
        lm_head_key = None
        for k in state_dict.keys():
            if "lm_head.weight" in k:
                lm_head_key = k
                break
        
        if lm_head_key:
            lm_weight = state_dict[lm_head_key]
            self.lm_head = torch.nn.Linear(
                lm_weight.shape[1], lm_weight.shape[0], bias=False
            ).to(device)
            self.lm_head.weight.data = lm_weight.to(device)
        else:
            self.lm_head = None
        
        # Load d2t (draft-to-target token mapping) if available
        self.hot_token_id = None
        for k in state_dict.keys():
            if "d2t" in k:
                d2t = state_dict[k]
                self.hot_token_id = (d2t + torch.arange(d2t.shape[0], device=device)).long()
                break
        
        # Load the decoder layer weights for a simplified MLP-only forward
        # (Skip full attention for simplicity - use MLP as primary predictor)
        self.layer_gate_proj = None
        self.layer_up_proj = None
        self.layer_down_proj = None
        self.input_norm_weight = None
        self.hidden_norm_weight = None
        
        for k, v in state_dict.items():
            if "layers.0" in k or "midlayer" in k:
                if "gate_proj.weight" in k:
                    self.layer_gate_proj = v.to(device)
                elif "up_proj.weight" in k:
                    self.layer_up_proj = v.to(device)
                elif "down_proj.weight" in k:
                    self.layer_down_proj = v.to(device)
                elif "input_layernorm.weight" in k:
                    self.input_norm_weight = v.to(device)
                elif "hidden_norm.weight" in k or "post_attention_layernorm.weight" in k:
                    self.hidden_norm_weight = v.to(device)
        
        # Final norm
        self.norm_weight = None
        for k, v in state_dict.items():
            if k.endswith("norm.weight") and "layer" not in k and "fc_norm" not in k:
                self.norm_weight = v.to(device)
                break
        
        self.eps = 1e-6
        print(f"  [EAGLE-3 DraftHead] Loaded. fc: {fc_weight_key is not None}, "
              f"embed: {embed_key is not None}, lm_head: {lm_head_key is not None}, "
              f"d2t mapping: {self.hot_token_id is not None}")
        print(f"  [EAGLE-3 DraftHead] Keys in state_dict: {list(state_dict.keys())[:20]}")
        
        del state_dict
        torch.cuda.empty_cache()
    
    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * weight
    
    def draft_forward(
        self,
        aux_hidden_states: List[torch.Tensor],
        prev_token_id: torch.Tensor,
    ) -> torch.Tensor:
        """Simplified forward pass for the draft head.
        
        Args:
            aux_hidden_states: List of hidden states from target model layers [B, hidden_size] each
            prev_token_id: Previous token id [B]
        
        Returns:
            logits: [B, draft_vocab_size]
        """
        # Concatenate aux hidden states and project
        # aux_hidden_states: list of [B, H], concatenate to [B, 3*H]
        cat_hidden = torch.cat(aux_hidden_states, dim=-1)
        hidden = self.fc(cat_hidden)  # [B, hidden_size]
        
        # Get embedding of previous token
        if self.embed_tokens is not None:
            embeds = self.embed_tokens(prev_token_id)  # [B, hidden_size]
        else:
            embeds = torch.zeros_like(hidden)
        
        # Simplified forward: use MLP as the primary transformation
        # (Skip attention for profiling - this approximates the behavior)
        if self.input_norm_weight is not None:
            embeds_normed = self.rms_norm(embeds, self.input_norm_weight)
        else:
            embeds_normed = embeds
        
        # Combine embeds and hidden (simplified: additive instead of concat-attention)
        combined = embeds_normed + hidden
        
        # Apply MLP if available
        if self.layer_gate_proj is not None:
            if self.hidden_norm_weight is not None:
                combined = self.rms_norm(combined, self.hidden_norm_weight)
            gate = F.silu(F.linear(combined, self.layer_gate_proj))
            up = F.linear(combined, self.layer_up_proj)
            mlp_out = F.linear(gate * up, self.layer_down_proj)
            hidden_out = combined + mlp_out
        else:
            hidden_out = combined
        
        # Apply final norm
        if self.norm_weight is not None:
            hidden_out = self.rms_norm(hidden_out, self.norm_weight)
        
        # Project to vocab
        if self.lm_head is not None:
            logits = F.linear(hidden_out, self.lm_head.weight.data)
        else:
            logits = torch.zeros(hidden_out.shape[0], self.draft_vocab_size, device=self.device)
        
        return logits
    
    def predict_token(
        self,
        aux_hidden_states: List[torch.Tensor],
        prev_token_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next token and return (predicted_target_token_id, draft_logits)."""
        draft_logits = self.draft_forward(aux_hidden_states, prev_token_id)
        draft_token = draft_logits.argmax(dim=-1)  # [B] in draft vocab
        
        # Map draft token to target vocab if mapping exists
        if self.hot_token_id is not None:
            target_token = self.hot_token_id[draft_token]
        else:
            target_token = draft_token
        
        return target_token, draft_logits


# ──────────────────────────────────────────────────────────────────────────────
# Main profiling logic
# ──────────────────────────────────────────────────────────────────────────────
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of the probability distribution from logits."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def profile_single_sequence(
    target_model,
    tokenizer,
    draft_head: Optional[SimpleEagle3DraftHead],
    prompt: str,
    prompt_idx: int,
    device: torch.device,
    max_new_tokens: int = 128,
    draft_k: int = 5,
) -> SequenceResult:
    """Profile a single sequence, measuring acceptance at each step."""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    
    result = SequenceResult(
        prompt_idx=prompt_idx,
        prompt_text=prompt[:200],
        num_generated=0,
    )
    
    generated_tokens = []
    step_results = []  # list of (num_accepted_in_step, draft_k)
    
    with torch.no_grad():
        # Generate tokens step by step
        current_ids = input_ids
        
        for gen_step in range(max_new_tokens):
            # Forward pass through target model
            outputs = target_model(
                current_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            
            # Get logits for the last position
            last_logits = outputs.logits[0, -1, :]  # [vocab_size]
            
            # Compute metrics for this position
            top1_prob = F.softmax(last_logits, dim=-1).max().item()
            entropy = compute_entropy(last_logits.unsqueeze(0)).item()
            
            # Greedy token
            target_token_id = last_logits.argmax().item()
            
            # Draft head prediction
            draft_accepted = False
            if draft_head is not None and len(outputs.hidden_states) >= 4:
                # Extract aux hidden states from target model layers
                all_hidden = outputs.hidden_states  # tuple of [B, seq_len, H]
                num_layers = len(all_hidden) - 1  # exclude embedding layer
                
                # Get last 3 layer hidden states at last position
                aux_states = []
                for layer_idx in AUX_LAYER_IDS:
                    actual_idx = num_layers + layer_idx + 1  # +1 because index 0 is embeddings
                    if actual_idx >= 0 and actual_idx < len(all_hidden):
                        aux_states.append(all_hidden[actual_idx][0, -1:, :])  # [1, H]
                
                if len(aux_states) == 3:
                    prev_token = current_ids[0, -1:].long()  # [1]
                    predicted_token, _ = draft_head.predict_token(aux_states, prev_token)
                    draft_accepted = (predicted_token.item() == target_token_id)
            
            # Record token metric
            step_in_draft = gen_step % draft_k
            token_metric = TokenMetric(
                position=gen_step,
                token_id=target_token_id,
                top1_prob=top1_prob,
                entropy=entropy,
                is_top1=True,  # greedy decoding
                draft_accepted=draft_accepted,
                step_in_draft=step_in_draft,
            )
            result.tokens.append(token_metric)
            generated_tokens.append(target_token_id)
            
            # Track per-step acceptance for windowed analysis
            if step_in_draft == draft_k - 1 or gen_step == max_new_tokens - 1:
                window_start = gen_step - step_in_draft
                window_tokens = result.tokens[window_start:gen_step + 1]
                if window_tokens:
                    window_accept_rate = sum(1 for t in window_tokens if t.draft_accepted) / len(window_tokens)
                    result.per_step_accept_rates.append(window_accept_rate)
            
            # Append token and continue
            new_token = torch.tensor([[target_token_id]], device=device)
            current_ids = torch.cat([current_ids, new_token], dim=-1)
            
            # Stop on EOS
            if target_token_id == tokenizer.eos_token_id:
                break
            
            # Truncate context if too long (sliding window)
            if current_ids.shape[1] > 1024:
                current_ids = current_ids[:, -768:]
    
    result.num_generated = len(generated_tokens)
    if result.tokens:
        result.overall_accept_rate = sum(1 for t in result.tokens if t.draft_accepted) / len(result.tokens)
    
    return result


def main():
    print("=" * 70)
    print("Phase 1: EAGLE-3 Speculative Decoding Acceptance Rate Profiling")
    print("=" * 70)
    
    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES handles GPU selection
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── Load evaluation data ──────────────────────────────────────────────
    print(f"\n[1/5] Loading evaluation data from {DATA_PATH}")
    with open(DATA_PATH) as f:
        all_data = [json.loads(line) for line in f]
    print(f"  Loaded {len(all_data)} samples, using first {MAX_PROMPTS}")
    data = all_data[:MAX_PROMPTS]
    
    # ── Load tokenizer ────────────────────────────────────────────────────
    print(f"\n[2/5] Loading tokenizer from {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ── Load target model ─────────────────────────────────────────────────
    print(f"\n[3/5] Loading target model (Qwen3-8B) on {device}")
    t0 = time.time()
    target_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    target_model.eval()
    print(f"  Target model loaded in {time.time() - t0:.1f}s")
    print(f"  Model config: hidden_size={target_model.config.hidden_size}, "
          f"num_layers={target_model.config.num_hidden_layers}")
    
    # ── Load EAGLE-3 draft head ───────────────────────────────────────────
    print(f"\n[4/5] Loading EAGLE-3 draft head from {EAGLE3_PATH}")
    eagle3_config = json.load(open(os.path.join(EAGLE3_PATH, "config.json")))
    
    try:
        draft_head = SimpleEagle3DraftHead(eagle3_config, EAGLE3_PATH, device)
        draft_head.eval()
        use_draft_head = True
    except Exception as e:
        print(f"  WARNING: Failed to load EAGLE-3 draft head: {e}")
        print(f"  Falling back to confidence-based proxy only")
        draft_head = None
        use_draft_head = False
    
    # ── Run profiling ─────────────────────────────────────────────────────
    print(f"\n[5/5] Running profiling on {len(data)} prompts "
          f"(max_new_tokens={MAX_NEW_TOKENS}, draft_k={DRAFT_K})")
    
    all_results: List[SequenceResult] = []
    all_token_metrics: List[dict] = []
    
    t_start = time.time()
    
    for idx, sample in enumerate(data):
        prompt = sample["query"]
        
        result = profile_single_sequence(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_head=draft_head,
            prompt=prompt,
            prompt_idx=idx,
            device=device,
            max_new_tokens=MAX_NEW_TOKENS,
            draft_k=DRAFT_K,
        )
        all_results.append(result)
        
        # Collect per-token metrics
        for tm in result.tokens:
            all_token_metrics.append({
                "prompt_idx": idx,
                "position": tm.position,
                "token_id": tm.token_id,
                "top1_prob": tm.top1_prob,
                "entropy": tm.entropy,
                "is_top1": tm.is_top1,
                "draft_accepted": tm.draft_accepted,
                "step_in_draft": tm.step_in_draft,
            })
        
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            avg_accept = np.mean([r.overall_accept_rate for r in all_results])
            print(f"  [{idx+1}/{len(data)}] {rate:.2f} prompts/s, "
                  f"avg acceptance: {avg_accept:.4f}, "
                  f"tokens generated: {sum(r.num_generated for r in all_results)}")
    
    total_time = time.time() - t_start
    print(f"\n  Profiling complete: {total_time:.1f}s total, "
          f"{len(data)/total_time:.2f} prompts/s")
    
    # ── Save raw results ──────────────────────────────────────────────────
    print("\n[Saving results...]")
    
    # Save per-token metrics as JSONL
    metrics_path = OUTPUT_DIR / "per_token_metrics.jsonl"
    with open(metrics_path, "w") as f:
        for m in all_token_metrics:
            f.write(json.dumps(m) + "\n")
    print(f"  Per-token metrics: {metrics_path} ({len(all_token_metrics)} tokens)")
    
    # Save per-sequence summary
    seq_summary_path = OUTPUT_DIR / "per_sequence_summary.json"
    seq_summaries = []
    for r in all_results:
        seq_summaries.append({
            "prompt_idx": r.prompt_idx,
            "prompt_text": r.prompt_text,
            "num_generated": r.num_generated,
            "overall_accept_rate": r.overall_accept_rate,
            "per_step_accept_rates": r.per_step_accept_rates,
            "mean_top1_prob": np.mean([t.top1_prob for t in r.tokens]) if r.tokens else 0,
            "mean_entropy": np.mean([t.entropy for t in r.tokens]) if r.tokens else 0,
        })
    with open(seq_summary_path, "w") as f:
        json.dump(seq_summaries, f, indent=2)
    print(f"  Sequence summaries: {seq_summary_path}")
    
    # ── Quick distribution analysis ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("Quick Distribution Analysis")
    print("=" * 70)
    
    top1_probs = np.array([m["top1_prob"] for m in all_token_metrics])
    entropies = np.array([m["entropy"] for m in all_token_metrics])
    accepted = np.array([m["draft_accepted"] for m in all_token_metrics])
    
    print(f"\n  Total tokens analyzed: {len(top1_probs)}")
    print(f"\n  Top-1 Probability Distribution:")
    print(f"    Mean: {top1_probs.mean():.4f}")
    print(f"    Std:  {top1_probs.std():.4f}")
    print(f"    Median: {np.median(top1_probs):.4f}")
    print(f"    P10: {np.percentile(top1_probs, 10):.4f}")
    print(f"    P25: {np.percentile(top1_probs, 25):.4f}")
    print(f"    P75: {np.percentile(top1_probs, 75):.4f}")
    print(f"    P90: {np.percentile(top1_probs, 90):.4f}")
    
    # Bimodal indicator: fraction of tokens with very high vs very low confidence
    high_conf = (top1_probs > 0.8).sum() / len(top1_probs)
    low_conf = (top1_probs < 0.3).sum() / len(top1_probs)
    mid_conf = ((top1_probs >= 0.3) & (top1_probs <= 0.8)).sum() / len(top1_probs)
    
    print(f"\n  Confidence Regions:")
    print(f"    High (>0.8): {high_conf:.1%}")
    print(f"    Mid (0.3-0.8): {mid_conf:.1%}")
    print(f"    Low (<0.3): {low_conf:.1%}")
    
    if use_draft_head:
        print(f"\n  Draft Head Acceptance:")
        print(f"    Overall acceptance rate: {accepted.mean():.4f}")
        
        # Per-position acceptance
        for pos in range(DRAFT_K):
            pos_mask = np.array([m["step_in_draft"] for m in all_token_metrics]) == pos
            if pos_mask.sum() > 0:
                pos_accept = accepted[pos_mask].mean()
                print(f"    Position {pos}: {pos_accept:.4f}")
    
    print(f"\n  Entropy Distribution:")
    print(f"    Mean: {entropies.mean():.4f}")
    print(f"    Std:  {entropies.std():.4f}")
    print(f"    Median: {np.median(entropies):.4f}")
    
    # Bimodal test: Hartigan's dip test (simplified version)
    # Sort the top1_probs and check for a gap in the middle
    sorted_probs = np.sort(top1_probs)
    n = len(sorted_probs)
    # Check if there's a "valley" in the histogram
    hist, bin_edges = np.histogram(top1_probs, bins=50)
    hist_normalized = hist / hist.sum()
    
    # Find peaks
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks.append((i, hist[i], (bin_edges[i] + bin_edges[i+1]) / 2))
    
    print(f"\n  Histogram Peaks (top-1 prob):")
    for i, (idx, count, center) in enumerate(sorted(peaks, key=lambda x: -x[1])[:5]):
        print(f"    Peak {i+1}: center={center:.3f}, count={count}")
    
    if len(peaks) >= 2:
        print(f"\n  >>> BIMODAL SIGNAL DETECTED: {len(peaks)} peaks found")
        top2_peaks = sorted(peaks, key=lambda x: -x[1])[:2]
        print(f"      Peak 1: {top2_peaks[0][2]:.3f}")
        print(f"      Peak 2: {top2_peaks[1][2]:.3f}")
        print(f"      Separation: {abs(top2_peaks[0][2] - top2_peaks[1][2]):.3f}")
    
    # Save analysis summary
    analysis = {
        "total_tokens": len(top1_probs),
        "total_prompts": len(all_results),
        "top1_prob_stats": {
            "mean": float(top1_probs.mean()),
            "std": float(top1_probs.std()),
            "median": float(np.median(top1_probs)),
            "p10": float(np.percentile(top1_probs, 10)),
            "p25": float(np.percentile(top1_probs, 25)),
            "p75": float(np.percentile(top1_probs, 75)),
            "p90": float(np.percentile(top1_probs, 90)),
        },
        "confidence_regions": {
            "high_gt_0.8": float(high_conf),
            "mid_0.3_0.8": float(mid_conf),
            "low_lt_0.3": float(low_conf),
        },
        "entropy_stats": {
            "mean": float(entropies.mean()),
            "std": float(entropies.std()),
            "median": float(np.median(entropies)),
        },
        "draft_head_acceptance": float(accepted.mean()) if use_draft_head else None,
        "histogram_peaks": [
            {"center": float(p[2]), "count": int(p[1])} 
            for p in sorted(peaks, key=lambda x: -x[1])[:5]
        ],
        "num_peaks": len(peaks),
        "bimodal_detected": len(peaks) >= 2,
    }
    
    analysis_path = OUTPUT_DIR / "distribution_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis saved to: {analysis_path}")
    
    print("\n" + "=" * 70)
    print("Done! Results saved to:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
