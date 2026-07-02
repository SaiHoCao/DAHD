#!/usr/bin/env python3
"""
Phase 1 Validation: Bimodal Distribution Analysis on Draft Acceptance Rate.

This script runs full EAGLE-3 speculative decoding (with GQA attention, RoPE, KV cache)
and collects BOTH:
  1. Target model top-1 confidence (target's softmax max prob at each position)
  2. Draft acceptance rate (whether EAGLE-3's draft token was accepted by target)

Then performs comprehensive bimodal statistical tests on both distributions.

Key distinction:
  - target_confidence: how confident the target model is about its prediction
  - draft_acceptance: whether the draft model's prediction matches target's choice
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

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ValidationConfig:
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
    aux_layer_indices: List[int] = field(default_factory=lambda: [2, 18, 33])

    # Speculative decoding params
    num_draft_steps: int = 5  # K
    max_new_tokens: int = 128  # More tokens for better statistics
    num_samples: int = 60  # 60 prompts for robust statistics
    max_prompt_len: int = 512

    # Output
    output_dir: str = "/data1/caoshuaihu.csh/workspace/dahd_speculative_decoding/results/phase1_validation/"

    # Device
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


# ============================================================================
# Import EAGLE-3 Draft Head from existing module
# ============================================================================

from run_eagle3_full_profiling import (
    Eagle3DraftHead, RMSNorm, precompute_freqs_cis,
    rotate_half, apply_rotary_emb
)


# ============================================================================
# Speculative Decoder with Target Confidence Collection
# ============================================================================

class ValidationDecoder:
    """Runs speculative decoding collecting both target confidence and acceptance."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype

        self.logger = logging.getLogger("Phase1Validation")
        self.logger.setLevel(logging.INFO)

        # Collected metrics
        self.per_token_data = []  # Every draft token with target conf + acceptance
        self.per_step_data = []   # Per speculative step summary
        self.per_prompt_data = [] # Per prompt summary

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
        self.draft_head.d2t = self.draft_head.d2t.to(self.device)

        # Build t2d mapping
        d2t = self.draft_head.d2t
        self.t2d = torch.full((self.config.target_vocab_size,), -1,
                              dtype=torch.long, device=self.device)
        draft_ids = torch.arange(self.config.draft_vocab_size, device=self.device)
        target_ids = d2t.long()
        self.t2d[target_ids] = draft_ids

        self.logger.info(f"Models loaded. t2d coverage: {(self.t2d >= 0).sum().item()}/{self.config.target_vocab_size}")

    def extract_aux_hidden_states(self, model_output) -> torch.Tensor:
        """Extract and concatenate aux hidden states from target model output."""
        all_hidden = model_output.hidden_states
        aux_states = [all_hidden[idx] for idx in self.config.aux_layer_indices]
        return torch.cat(aux_states, dim=-1)

    @torch.no_grad()
    def run_target_forward(self, input_ids: torch.Tensor):
        """Run target model forward pass with hidden states."""
        return self.target_model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )

    @torch.no_grad()
    def eagle_prefill(self, target_output, input_ids: torch.Tensor):
        """Run EAGLE prefill to build KV cache from full context."""
        aux_all = self.extract_aux_hidden_states(target_output)
        embeds_all = self.embed_tokens(input_ids)
        N = input_ids.shape[1]
        positions = torch.arange(N, device=self.device)

        logits, aux_hidden, kv_cache = self.draft_head(
            hidden_states=aux_all,
            embeds=embeds_all,
            positions=positions,
            kv_cache=None,
            is_first_step=True
        )
        return aux_hidden[:, -1:, :], kv_cache

    @torch.no_grad()
    def draft_phase(self, target_aux_hidden, prev_token_id, start_position,
                    eagle_kv_cache=None, eagle_aux_hidden=None):
        """Auto-regressively generate K draft tokens."""
        K = self.config.num_draft_steps
        draft_tokens = []
        draft_confidences = []

        if eagle_aux_hidden is not None:
            hidden_states = eagle_aux_hidden
        else:
            hidden_states = target_aux_hidden

        kv_cache = eagle_kv_cache

        for k in range(K):
            prev_token_tensor = torch.tensor([[prev_token_id]], device=self.device)
            embeds = self.embed_tokens(prev_token_tensor)
            pos = torch.tensor([start_position + k], device=self.device)

            logits, aux_hidden, kv_cache = self.draft_head(
                hidden_states=hidden_states,
                embeds=embeds,
                positions=pos,
                kv_cache=kv_cache,
                is_first_step=(k == 0)
            )

            draft_logits = logits[0, -1, :]
            probs = F.softmax(draft_logits.float(), dim=-1)
            draft_token_id = probs.argmax().item()
            confidence = probs[draft_token_id].item()

            target_token_id = self.draft_head.d2t[draft_token_id].item()

            draft_tokens.append(target_token_id)
            draft_confidences.append(confidence)

            prev_token_id = target_token_id
            hidden_states = aux_hidden

        return draft_tokens, draft_confidences

    @torch.no_grad()
    def verify_phase_with_confidence(self, input_ids, target_next, draft_tokens):
        """
        Verify draft tokens AND collect target model confidence.

        Returns:
            n_accepted: number of consecutively accepted draft tokens
            acceptance_mask: [K] booleans
            target_choices: [K] target model greedy choices
            target_confidences: [K] target model max softmax prob at each position
        """
        K = len(draft_tokens)

        # Build verification sequence: [input_ids, target_next, draft_tokens]
        extra_tokens = [target_next] + draft_tokens
        extra_tensor = torch.tensor([extra_tokens], device=self.device)
        verify_ids = torch.cat([input_ids, extra_tensor], dim=1)

        # Run target model
        output = self.target_model(
            input_ids=verify_ids,
            output_hidden_states=False,
            return_dict=True,
        )

        input_len = input_ids.shape[1]
        n_accepted = 0
        acceptance_mask = []
        target_choices = []
        target_confidences = []

        for i in range(K):
            logit_pos = input_len + i  # Position of target_next, then draft[0], ...
            target_logits = output.logits[0, logit_pos, :]

            # Target confidence: max softmax probability
            target_probs = F.softmax(target_logits.float(), dim=-1)
            target_choice = target_probs.argmax().item()
            target_conf = target_probs[target_choice].item()

            target_choices.append(target_choice)
            target_confidences.append(target_conf)

            if target_choice == draft_tokens[i]:
                n_accepted += 1
                acceptance_mask.append(True)
            else:
                acceptance_mask.append(False)
                break  # Stop at first rejection

        # Fill remaining positions (after first rejection)
        while len(acceptance_mask) < K:
            acceptance_mask.append(False)
            logit_pos = input_len + len(target_choices)
            if logit_pos < output.logits.shape[1]:
                target_logits = output.logits[0, logit_pos, :]
                target_probs = F.softmax(target_logits.float(), dim=-1)
                tc = target_probs.argmax().item()
                target_choices.append(tc)
                target_confidences.append(target_probs[tc].item())
            else:
                target_choices.append(-1)
                target_confidences.append(0.0)

        return n_accepted, acceptance_mask, target_choices, target_confidences

    def format_prompt(self, query: str) -> str:
        """Format prompt with chat template (thinking disabled)."""
        messages = [{"role": "user", "content": query}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return text

    @torch.no_grad()
    def run_single_prompt(self, prompt: str, prompt_idx: int) -> dict:
        """Run speculative decoding on one prompt, collecting full metrics."""
        self.logger.info(f"[{prompt_idx}] {prompt[:60]}...")

        formatted = self.format_prompt(prompt)
        inputs = self.tokenizer(
            formatted, return_tensors="pt", truncation=True,
            max_length=self.config.max_prompt_len
        ).to(self.device)
        input_ids = inputs['input_ids']
        prompt_len = input_ids.shape[1]

        step_metrics = []
        total_accepted = 0
        total_drafted = 0
        generated_tokens = 0
        step = 0

        while generated_tokens < self.config.max_new_tokens:
            current_len = input_ids.shape[1]

            # Target forward
            target_output = self.run_target_forward(input_ids)
            target_next = target_output.logits[0, -1, :].argmax().item()

            if target_next == self.tokenizer.eos_token_id:
                break

            # Also get target confidence at the "current" position (for the accepted token)
            target_last_probs = F.softmax(target_output.logits[0, -1, :].float(), dim=-1)
            target_last_conf = target_last_probs[target_next].item()

            # Eagle prefill
            aux_hidden = self.extract_aux_hidden_states(target_output)
            aux_hidden_last = aux_hidden[:, -1:, :]
            eagle_aux, eagle_kv = self.eagle_prefill(target_output, input_ids)

            # Draft phase
            draft_tokens, draft_confidences = self.draft_phase(
                target_aux_hidden=aux_hidden_last,
                prev_token_id=target_next,
                start_position=current_len,
                eagle_kv_cache=eagle_kv,
                eagle_aux_hidden=eagle_aux,
            )

            # Verify with confidence collection
            n_accepted, acceptance_mask, target_choices, target_confidences = \
                self.verify_phase_with_confidence(input_ids, target_next, draft_tokens)

            # Record per-token data
            for pos_k in range(self.config.num_draft_steps):
                self.per_token_data.append({
                    'prompt_idx': prompt_idx,
                    'step': step,
                    'draft_position': pos_k,
                    'abs_position': current_len + 1 + pos_k,
                    'draft_token': draft_tokens[pos_k],
                    'target_token': target_choices[pos_k],
                    'accepted': acceptance_mask[pos_k],
                    'draft_confidence': draft_confidences[pos_k],
                    'target_confidence': target_confidences[pos_k],
                })

            # Record per-step data
            acceptance_length = n_accepted  # How many consecutive accepted
            step_record = {
                'prompt_idx': prompt_idx,
                'step': step,
                'acceptance_length': acceptance_length,
                'acceptance_rate': n_accepted / self.config.num_draft_steps,
                'target_last_confidence': target_last_conf,
                'mean_target_confidence': float(np.mean(target_confidences)),
                'mean_draft_confidence': float(np.mean(draft_confidences)),
            }
            self.per_step_data.append(step_record)
            step_metrics.append(step_record)

            total_accepted += n_accepted
            total_drafted += self.config.num_draft_steps

            # Update input_ids
            new_tokens_list = [target_next]
            if n_accepted > 0:
                new_tokens_list.extend(draft_tokens[:n_accepted])
            if n_accepted < self.config.num_draft_steps:
                bonus = target_choices[n_accepted] if n_accepted < len(target_choices) else target_next
                new_tokens_list.append(bonus)

            new_tensor = torch.tensor([new_tokens_list], device=self.device)
            input_ids = torch.cat([input_ids, new_tensor], dim=1)
            generated_tokens += len(new_tokens_list)
            step += 1

            if step > 100:
                break

        overall_rate = total_accepted / max(total_drafted, 1)
        summary = {
            'prompt_idx': prompt_idx,
            'prompt_len': prompt_len,
            'generated_tokens': generated_tokens,
            'total_steps': step,
            'total_accepted': total_accepted,
            'total_drafted': total_drafted,
            'overall_acceptance_rate': overall_rate,
        }
        self.per_prompt_data.append(summary)

        self.logger.info(
            f"  -> acceptance={overall_rate:.1%} ({total_accepted}/{total_drafted}), "
            f"generated={generated_tokens} in {step} steps"
        )
        return summary

    def run(self):
        """Main entry point."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Logging setup
        log_path = os.path.join(self.config.output_dir, 'run_log.txt')
        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

        self.logger.info("=" * 60)
        self.logger.info("Phase 1 Validation: Bimodal Distribution on Draft Acceptance")
        self.logger.info("=" * 60)
        self.logger.info(f"K={self.config.num_draft_steps}, max_new_tokens={self.config.max_new_tokens}, "
                        f"num_samples={self.config.num_samples}")

        self.load_models()

        # Load data
        prompts = []
        with open(self.config.eval_data_path) as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data['query'])
        num_to_run = min(self.config.num_samples, len(prompts))
        self.logger.info(f"Running {num_to_run} prompts (available: {len(prompts)})")

        # Run
        start_time = time.time()
        for i in range(num_to_run):
            self.run_single_prompt(prompts[i], i)
            if (i + 1) % 10 == 0:
                self._save_raw_data()
                self.logger.info(f"  [Checkpoint] Saved after {i+1} prompts")

        elapsed = time.time() - start_time
        self.logger.info(f"\nTotal time: {elapsed:.1f}s for {num_to_run} prompts")

        self._save_raw_data()
        self.logger.info("Running bimodal analysis...")
        self._run_bimodal_analysis()
        self.logger.info("Done! Results saved to " + self.config.output_dir)

    def _save_raw_data(self):
        """Save raw collected data."""
        out = self.config.output_dir
        with open(os.path.join(out, 'per_token_data.jsonl'), 'w') as f:
            for d in self.per_token_data:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
        with open(os.path.join(out, 'per_step_data.json'), 'w') as f:
            json.dump(self.per_step_data, f, indent=2)
        with open(os.path.join(out, 'per_prompt_data.json'), 'w') as f:
            json.dump(self.per_prompt_data, f, indent=2)

    def _run_bimodal_analysis(self):
        """Comprehensive bimodal analysis on both target confidence and acceptance."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy import stats
        from sklearn.mixture import GaussianMixture
        import diptest

        out = self.config.output_dir
        K = self.config.num_draft_steps

        # =====================================================================
        # Extract data arrays
        # =====================================================================
        # 1. Target confidence at position 0 of each step (most informative)
        target_confs_pos0 = np.array([
            d['target_confidence'] for d in self.per_token_data if d['draft_position'] == 0
        ])
        # 2. All target confidences
        target_confs_all = np.array([d['target_confidence'] for d in self.per_token_data])
        # 3. Draft acceptance binary at position 0
        accept_pos0 = np.array([
            1.0 if d['accepted'] else 0.0 for d in self.per_token_data if d['draft_position'] == 0
        ])
        # 4. Acceptance length per step (continuous: 0,1,2,3,4,5)
        accept_lengths = np.array([d['acceptance_length'] for d in self.per_step_data])
        # 5. Acceptance rate per step (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        accept_rates = np.array([d['acceptance_rate'] for d in self.per_step_data])
        # 6. Per-prompt overall acceptance rate (continuous approximation)
        prompt_rates = np.array([d['overall_acceptance_rate'] for d in self.per_prompt_data])
        # 7. Windowed acceptance rate (window of 5 consecutive steps)
        window_size = 5
        windowed_rates = []
        for pidx in set(d['prompt_idx'] for d in self.per_step_data):
            steps_for_prompt = [d['acceptance_rate'] for d in self.per_step_data if d['prompt_idx'] == pidx]
            for i in range(0, len(steps_for_prompt) - window_size + 1, window_size):
                window = steps_for_prompt[i:i+window_size]
                windowed_rates.append(np.mean(window))
        windowed_rates = np.array(windowed_rates) if windowed_rates else np.array([])

        # =====================================================================
        # GMM Analysis Function
        # =====================================================================
        def gmm_analysis(data, name):
            """Fit GMM 1,2,3 components and return results."""
            if len(data) < 20:
                return {'error': f'Insufficient data for {name}: n={len(data)}'}
            data_2d = data.reshape(-1, 1)
            results = {}
            gmms = {}
            for n in [1, 2, 3]:
                gmm = GaussianMixture(n_components=n, random_state=42, n_init=5).fit(data_2d)
                gmms[n] = gmm
                results[f'bic_{n}comp'] = float(gmm.bic(data_2d))
                results[f'aic_{n}comp'] = float(gmm.aic(data_2d))
            best_n = int(np.argmin([results['bic_1comp'], results['bic_2comp'], results['bic_3comp']]) + 1)
            results['best_n_by_bic'] = best_n
            # 2-component details
            gmm2 = gmms[2]
            results['gmm2_means'] = sorted(gmm2.means_.flatten().tolist())
            results['gmm2_weights'] = gmm2.weights_.tolist()
            results['gmm2_stds'] = np.sqrt(gmm2.covariances_.flatten()).tolist()
            # Delta BIC
            results['delta_bic_2vs1'] = results['bic_2comp'] - results['bic_1comp']
            results['bimodal_by_bic'] = results['delta_bic_2vs1'] < -10  # Strong evidence
            return results, gmms[2]

        # =====================================================================
        # Dip Test Function
        # =====================================================================
        def dip_analysis(data, name):
            """Hartigan's dip test for unimodality."""
            if len(data) < 20:
                return {'error': f'Insufficient data: n={len(data)}'}
            dip_stat, p_value = diptest.diptest(data.astype(np.float64))
            return {
                'dip_statistic': float(dip_stat),
                'p_value': float(p_value),
                'reject_unimodal': bool(p_value < 0.05),
                'n_samples': len(data),
            }

        # =====================================================================
        # Run analyses
        # =====================================================================
        results = {}

        # --- Target Confidence Distribution ---
        self.logger.info("Analyzing target confidence distribution...")
        tc_gmm_res, tc_gmm2 = gmm_analysis(target_confs_pos0, "target_confidence_pos0")
        tc_dip_res = dip_analysis(target_confs_pos0, "target_confidence_pos0")
        results['target_confidence_gmm'] = tc_gmm_res
        results['target_confidence_dip'] = tc_dip_res

        # --- Draft Acceptance (multiple views) ---
        self.logger.info("Analyzing draft acceptance distributions...")

        # View 1: Acceptance length per step (0-5)
        al_gmm_res, al_gmm2 = gmm_analysis(accept_lengths.astype(float), "acceptance_length")
        al_dip_res = dip_analysis(accept_lengths.astype(float), "acceptance_length")
        results['acceptance_length_gmm'] = al_gmm_res
        results['acceptance_length_dip'] = al_dip_res

        # View 2: Per-step acceptance rate
        ar_gmm_res, ar_gmm2 = gmm_analysis(accept_rates, "acceptance_rate_per_step")
        ar_dip_res = dip_analysis(accept_rates, "acceptance_rate_per_step")
        results['acceptance_rate_per_step_gmm'] = ar_gmm_res
        results['acceptance_rate_per_step_dip'] = ar_dip_res

        # View 3: Windowed acceptance rate (smoother)
        if len(windowed_rates) >= 20:
            wr_gmm_res, wr_gmm2 = gmm_analysis(windowed_rates, "windowed_acceptance_rate")
            wr_dip_res = dip_analysis(windowed_rates, "windowed_acceptance_rate")
            results['windowed_acceptance_gmm'] = wr_gmm_res
            results['windowed_acceptance_dip'] = wr_dip_res

        # --- Correlation Analysis ---
        self.logger.info("Computing correlations...")
        # Per-token: target_confidence vs accepted (point-biserial)
        tc_all = np.array([d['target_confidence'] for d in self.per_token_data])
        acc_all = np.array([1.0 if d['accepted'] else 0.0 for d in self.per_token_data])
        pearson_r, pearson_p = stats.pearsonr(tc_all, acc_all)
        spearman_r, spearman_p = stats.spearmanr(tc_all, acc_all)
        # Point-biserial (same as Pearson for binary)
        pb_r, pb_p = stats.pointbiserialr(acc_all.astype(int), tc_all)

        results['correlation'] = {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'point_biserial_r': float(pb_r),
            'point_biserial_p': float(pb_p),
        }

        # Per-step: mean target confidence vs acceptance rate
        step_tc = np.array([d['mean_target_confidence'] for d in self.per_step_data])
        step_ar = np.array([d['acceptance_rate'] for d in self.per_step_data])
        if len(step_tc) > 5:
            sr, sp = stats.spearmanr(step_tc, step_ar)
            results['correlation']['step_level_spearman_r'] = float(sr)
            results['correlation']['step_level_spearman_p'] = float(sp)

        # --- Summary Statistics ---
        results['summary'] = {
            'num_prompts': len(self.per_prompt_data),
            'num_steps': len(self.per_step_data),
            'num_tokens_evaluated': len(self.per_token_data),
            'overall_acceptance_rate': float(np.mean(acc_all)),
            'mean_acceptance_length': float(np.mean(accept_lengths)),
            'mean_target_confidence': float(np.mean(tc_all)),
            'K': self.config.num_draft_steps,
        }

        # --- Conclusion ---
        tc_bimodal = tc_gmm_res.get('bimodal_by_bic', False)
        al_bimodal = al_gmm_res.get('bimodal_by_bic', False)
        ar_bimodal = ar_gmm_res.get('bimodal_by_bic', False)
        tc_dip_reject = tc_dip_res.get('reject_unimodal', False)
        al_dip_reject = al_dip_res.get('reject_unimodal', False)

        conclusion_lines = []
        conclusion_lines.append(f"Target Confidence bimodal: GMM={'YES' if tc_bimodal else 'NO'}, DipTest={'YES' if tc_dip_reject else 'NO'}")
        conclusion_lines.append(f"Acceptance Length bimodal: GMM={'YES' if al_bimodal else 'NO'}, DipTest={'YES' if al_dip_reject else 'NO'}")
        conclusion_lines.append(f"Acceptance Rate/step bimodal: GMM={'YES' if ar_bimodal else 'NO'}")
        conclusion_lines.append(f"Correlation (target_conf vs accepted): r={pearson_r:.3f}, p={pearson_p:.2e}")

        if tc_bimodal and (al_bimodal or ar_bimodal):
            conclusion_lines.append("CONCLUSION: Both distributions show bimodality. Target confidence is a valid proxy for draft acceptance difficulty.")
        elif tc_bimodal and not (al_bimodal or ar_bimodal):
            conclusion_lines.append("CONCLUSION: Only target confidence is bimodal; acceptance may not be. Need alternative difficulty metric.")
        elif not tc_bimodal and (al_bimodal or ar_bimodal):
            conclusion_lines.append("CONCLUSION: Acceptance is bimodal but confidence is not. Interesting divergence.")
        else:
            conclusion_lines.append("CONCLUSION: Neither distribution shows strong bimodality.")

        results['conclusion'] = conclusion_lines
        for line in conclusion_lines:
            self.logger.info(line)

        # Save results JSON
        with open(os.path.join(out, 'bimodal_validation_results.json'), 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # =====================================================================
        # Visualizations
        # =====================================================================
        self._plot_visualizations(
            target_confs_pos0, target_confs_all, accept_lengths, accept_rates,
            windowed_rates, tc_all, acc_all, tc_gmm2, results
        )

    def _plot_visualizations(self, target_confs_pos0, target_confs_all,
                             accept_lengths, accept_rates, windowed_rates,
                             tc_all, acc_all, tc_gmm2, results):
        """Generate all visualization plots."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.mixture import GaussianMixture
        from scipy.stats import gaussian_kde

        out = self.config.output_dir

        # === Plot 1: Target Confidence Distribution + GMM Fit ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: histogram with GMM overlay
        ax = axes[0]
        ax.hist(target_confs_pos0, bins=50, density=True, alpha=0.7, color='steelblue',
                edgecolor='white', label='Observed')
        # GMM fit overlay
        x_range = np.linspace(0, 1, 200).reshape(-1, 1)
        if tc_gmm2 is not None:
            from scipy.stats import norm
            for i in range(2):
                w = tc_gmm2.weights_[i]
                m = tc_gmm2.means_[i, 0]
                s = np.sqrt(tc_gmm2.covariances_[i, 0, 0])
                ax.plot(x_range, w * norm.pdf(x_range, m, s), '--', linewidth=2,
                       label=f'Comp {i+1}: μ={m:.2f}, σ={s:.2f}, w={w:.2f}')
            # Total GMM
            log_prob = tc_gmm2.score_samples(x_range)
            ax.plot(x_range, np.exp(log_prob), 'r-', linewidth=2, label='GMM-2 total')

        ax.set_xlabel('Target Model Top-1 Confidence')
        ax.set_ylabel('Density')
        ax.set_title('Target Confidence Distribution (Position 0)')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

        # Right: all positions
        ax = axes[1]
        ax.hist(target_confs_all, bins=50, density=True, alpha=0.7, color='coral', edgecolor='white')
        ax.set_xlabel('Target Model Top-1 Confidence')
        ax.set_ylabel('Density')
        ax.set_title(f'Target Confidence Distribution (All Positions, n={len(target_confs_all)})')
        ax.axvline(np.mean(target_confs_all), color='red', ls='--', label=f'Mean={np.mean(target_confs_all):.3f}')
        ax.legend()
        ax.set_xlim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(out, 'target_confidence_distribution.png'), dpi=150)
        plt.close()

        # === Plot 2: Draft Acceptance Distribution ===
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top-left: Acceptance length histogram
        ax = axes[0, 0]
        bins = np.arange(-0.5, self.config.num_draft_steps + 1.5, 1)
        counts, _, _ = ax.hist(accept_lengths, bins=bins, color='seagreen', alpha=0.8,
                               edgecolor='black', density=True)
        ax.set_xlabel('Acceptance Length (consecutive accepted tokens)')
        ax.set_ylabel('Density')
        ax.set_title(f'Acceptance Length Distribution (n={len(accept_lengths)})')
        ax.set_xticks(range(self.config.num_draft_steps + 1))
        for i, c in enumerate(counts):
            if c > 0:
                ax.text(i, c + 0.01, f'{c:.2f}', ha='center', fontsize=9)

        # Top-right: Per-step acceptance rate
        ax = axes[0, 1]
        ax.hist(accept_rates, bins=self.config.num_draft_steps + 1,
                range=(-0.1/self.config.num_draft_steps, 1 + 0.1/self.config.num_draft_steps),
                color='darkorange', alpha=0.8, edgecolor='black', density=True)
        ax.set_xlabel('Per-Step Acceptance Rate (n_accepted/K)')
        ax.set_ylabel('Density')
        ax.set_title('Per-Step Acceptance Rate Distribution')
        ax.axvline(np.mean(accept_rates), color='red', ls='--',
                  label=f'Mean={np.mean(accept_rates):.3f}')
        ax.legend()

        # Bottom-left: Windowed acceptance rate (if available)
        ax = axes[1, 0]
        if len(windowed_rates) >= 10:
            ax.hist(windowed_rates, bins=30, color='purple', alpha=0.7, edgecolor='white', density=True)
            ax.set_xlabel('Windowed Acceptance Rate (5-step window)')
            ax.set_ylabel('Density')
            ax.set_title(f'Windowed Acceptance Rate (n={len(windowed_rates)})')
            ax.axvline(np.mean(windowed_rates), color='red', ls='--',
                      label=f'Mean={np.mean(windowed_rates):.3f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Insufficient data for windowed analysis',
                   ha='center', va='center', transform=ax.transAxes)

        # Bottom-right: Per-prompt acceptance rates
        ax = axes[1, 1]
        prompt_rates = np.array([d['overall_acceptance_rate'] for d in self.per_prompt_data])
        ax.hist(prompt_rates, bins=20, color='teal', alpha=0.7, edgecolor='white', density=True)
        ax.set_xlabel('Per-Prompt Overall Acceptance Rate')
        ax.set_ylabel('Density')
        ax.set_title(f'Per-Prompt Acceptance Rate (n={len(prompt_rates)})')
        ax.axvline(np.mean(prompt_rates), color='red', ls='--',
                  label=f'Mean={np.mean(prompt_rates):.3f}')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(out, 'draft_acceptance_distribution.png'), dpi=150)
        plt.close()

        # === Plot 3: Confidence vs Acceptance ===
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Left: scatter (binary acceptance vs target confidence)
        ax = axes[0]
        # Jitter y for visibility
        jitter = np.random.normal(0, 0.02, size=len(acc_all))
        ax.scatter(tc_all, acc_all + jitter, alpha=0.15, s=8, c='steelblue')
        ax.set_xlabel('Target Model Confidence')
        ax.set_ylabel('Accepted (0/1, jittered)')
        ax.set_title('Target Confidence vs Draft Acceptance')
        ax.set_xlim(0, 1)
        # Add binned acceptance rate curve
        bins = np.linspace(0, 1, 21)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_rates = []
        for i in range(len(bins) - 1):
            mask = (tc_all >= bins[i]) & (tc_all < bins[i+1])
            if mask.sum() > 0:
                bin_rates.append(acc_all[mask].mean())
            else:
                bin_rates.append(np.nan)
        ax.plot(bin_centers, bin_rates, 'r-o', linewidth=2, markersize=5,
               label='Binned acceptance rate')
        ax.legend()

        # Middle: target confidence histogram split by accepted/rejected
        ax = axes[1]
        tc_accepted = tc_all[acc_all == 1]
        tc_rejected = tc_all[acc_all == 0]
        ax.hist(tc_accepted, bins=40, alpha=0.6, color='green', density=True,
               label=f'Accepted (n={len(tc_accepted)})')
        ax.hist(tc_rejected, bins=40, alpha=0.6, color='red', density=True,
               label=f'Rejected (n={len(tc_rejected)})')
        ax.set_xlabel('Target Model Confidence')
        ax.set_ylabel('Density')
        ax.set_title('Target Confidence: Accepted vs Rejected')
        ax.legend()

        # Right: acceptance rate by confidence bin
        ax = axes[2]
        valid_mask = ~np.isnan(bin_rates)
        ax.bar(bin_centers[valid_mask], np.array(bin_rates)[valid_mask],
              width=0.04, color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_xlabel('Target Confidence Bin')
        ax.set_ylabel('Acceptance Rate in Bin')
        ax.set_title('Acceptance Rate by Target Confidence')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', ls='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(out, 'confidence_vs_acceptance.png'), dpi=150)
        plt.close()

        self.logger.info("All plots saved.")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1 Validation: Bimodal on Draft Acceptance")
    parser.add_argument('--num-samples', type=int, default=60,
                       help='Number of prompts to evaluate')
    parser.add_argument('--num-draft-steps', type=int, default=5,
                       help='Number of draft steps (K)')
    parser.add_argument('--max-new-tokens', type=int, default=128,
                       help='Max tokens to generate per prompt')
    args = parser.parse_args()

    config = ValidationConfig(
        num_samples=args.num_samples,
        num_draft_steps=args.num_draft_steps,
        max_new_tokens=args.max_new_tokens,
    )

    decoder = ValidationDecoder(config)
    decoder.run()


if __name__ == "__main__":
    main()
