#!/usr/bin/env python3
"""Phase 1: Acceptance rate profiling for DAHD speculative decoding.

This script profiles per-token acceptance rates across different benchmark tasks
to validate the bimodal hypothesis — that acceptance rates exhibit distinct
"easy" and "hard" modes across different token positions and tasks.

Usage:
    python experiments/phase1_profiling/run_acceptance_profiling.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --tasks GSM8K,MATH,HumanEval \
        --num_samples 100 \
        --output_dir results/phase1_results

    # With a custom YAML config:
    python experiments/phase1_profiling/run_acceptance_profiling.py \
        --config configs/phase1.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import ExperimentConfig
from src.drafters.dahd_draft_module import DAHDDraftModule
from src.metrics.collector import PerTokenMetricsCollector
from src.utils.device_utils import setup_deterministic
from src.utils.logging_utils import ExperimentLogger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Profile per-token acceptance rates for bimodal hypothesis validation."
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Target model HuggingFace ID or local path (e.g., /path/to/Qwen2-1.5B-Instruct)."
    )
    parser.add_argument(
        "--tasks", type=str, default="GSM8K,MATH,HumanEval",
        help="Comma-separated list of benchmark tasks."
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="Number of samples per task."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/phase1_results",
        help="Directory to save profiling results."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional YAML config file path (overrides other args)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to run inference on."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Maximum number of new tokens to generate per sample."
    )
    parser.add_argument(
        "--draft_k", type=int, default=6,
        help="Default draft length (k) for profiling."
    )
    parser.add_argument(
        "--local_data_dir", type=str, default=None,
        help="Local directory containing dataset files. If provided, datasets will be loaded from here."
    )
    parser.add_argument(
        "--use_synthetic_data", action="store_true",
        help="Use synthetic placeholder data for offline testing (no network required)."
    )
    return parser.parse_args()


def main() -> None:
    """Main profiling pipeline."""
    args = parse_args()

    # Load config from YAML if provided, otherwise build from args
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig(
            phase=1,
            experiment_name="acceptance_profiling",
            seed=args.seed,
            device=args.device,
            target_model=args.model,
            tasks=[t.strip().lower().replace("-", "_") for t in args.tasks.split(",")],
            num_samples_per_task=args.num_samples,
            output_dir=args.output_dir,
        )

    # Setup deterministic execution and logging
    setup_deterministic(config.seed)
    exp_logger = ExperimentLogger(
        experiment_name="phase1_acceptance_profiling",
        log_dir=str(Path(config.output_dir) / "logs"),
    )
    exp_logger.log_config(config)
    exp_logger.log_phase_start(1, "Acceptance rate profiling")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load target model and tokenizer
    print(f"[Phase 1] Loading target model: {config.target_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.target_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(
        config.target_model,
        torch_dtype=torch.float16,
        device_map=config.device,
        trust_remote_code=True,
    )
    target_model.eval()

    # Initialize DAHD draft module
    hidden_dim = target_model.config.hidden_size
    vocab_size = target_model.config.vocab_size
    draft_module = DAHDDraftModule(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        max_k=args.draft_k,
        draft_length_easy=config.draft_length_easy,
        draft_length_hard=config.draft_length_hard,
        draft_length_medium=config.draft_length_medium,
        probe_weight=config.probe_weight,
        ema_alpha=config.ema_alpha,
        easy_threshold=config.easy_threshold,
        hard_threshold=config.hard_threshold,
    ).to(config.device).half()
    draft_module.eval()

    # Import task runners
    from src.benchmarks.task_runners import get_task_runner

    # Per-task acceptance rate statistics
    task_stats: dict[str, dict] = {}

    for task_name in config.tasks:
        print(f"\n[Phase 1] Profiling task: {task_name}")
        runner = get_task_runner(task_name)
        # Determine local path for this task if local_data_dir is specified
        local_path = None
        if args.local_data_dir:
            task_data_path = Path(args.local_data_dir) / task_name
            if task_data_path.exists():
                local_path = str(task_data_path)
        samples = runner.load_dataset(
            num_samples=config.num_samples_per_task,
            local_path=local_path,
            use_synthetic=args.use_synthetic_data,
        )

        if not samples:
            print(f"  WARNING: No samples loaded for {task_name}, skipping.")
            continue

        # Initialize per-token metrics collector for this task
        collector = PerTokenMetricsCollector(
            output_jsonl=output_dir / f"{task_name}_per_token.jsonl",
            buffer_size=500,
        )

        total_accepted = 0
        total_drafted = 0

        for sample_idx, sample in enumerate(samples):
            sequence_id = f"{task_name}_{sample_idx}_{uuid.uuid4().hex[:8]}"
            prompt = runner.format_prompt(sample)

            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = inputs["input_ids"].to(config.device)

            # Reset router EMA for each sequence
            draft_module.reset_router()

            # Speculative decoding loop
            with torch.no_grad():
                for step in range(args.max_new_tokens // args.draft_k):
                    # Get target model hidden states at the last position
                    outputs = target_model(input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]

                    # Draft tokens using DAHD module
                    draft_output = draft_module(hidden_states)
                    draft_tokens = draft_output.draft_tokens
                    k = draft_output.draft_k

                    # Verify: run target model on input + draft tokens
                    candidate_ids = torch.cat([input_ids, draft_tokens], dim=-1)
                    verify_outputs = target_model(candidate_ids)
                    verify_logits = verify_outputs.logits

                    # Check acceptance for each draft position
                    seq_len = input_ids.size(1)
                    num_accepted = 0
                    for pos in range(k):
                        target_token = verify_logits[0, seq_len + pos - 1].argmax().item()
                        draft_token = draft_tokens[0, pos].item()
                        is_accepted = (target_token == draft_token)

                        if is_accepted:
                            num_accepted += 1

                        # Record per-token metric
                        collector.record_token(
                            token_id=draft_token,
                            token_pos=pos,
                            task=task_name,
                            sequence_id=sequence_id,
                            probe_confidence=float(draft_output.difficulty_score[0].item()),
                            input_hidden_entropy=0.0,
                            is_accepted=is_accepted,
                            draft_mode=draft_output.selected_mode,
                            draft_latency_ms=draft_output.draft_latency_ms,
                            device=config.device,
                        )

                        total_drafted += 1
                        total_accepted += int(is_accepted)

                        if not is_accepted:
                            break

                    # Update EMA with acceptance rate for this step
                    step_acceptance = num_accepted / k if k > 0 else 0.0
                    draft_module.update_acceptance_rate(step_acceptance)

                    # Advance input_ids with accepted tokens + bonus token
                    accepted_tokens = draft_tokens[:, :num_accepted]
                    bonus_token = verify_logits[0, seq_len + num_accepted - 1].argmax()
                    bonus_token = bonus_token.unsqueeze(0).unsqueeze(0)
                    input_ids = torch.cat([input_ids, accepted_tokens, bonus_token], dim=-1)

                    # Check stopping conditions
                    if input_ids.size(1) >= 1024 + args.max_new_tokens:
                        break
                    if tokenizer.eos_token_id and input_ids[0, -1].item() == tokenizer.eos_token_id:
                        break

            if (sample_idx + 1) % 10 == 0:
                rate = total_accepted / total_drafted if total_drafted > 0 else 0.0
                print(f"  [{task_name}] {sample_idx + 1}/{len(samples)} samples, "
                      f"running acceptance rate: {rate:.4f}")

        # Flush remaining metrics
        collector.flush()

        # Compute task-level statistics
        task_acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0
        task_stats[task_name] = {
            "total_drafted": total_drafted,
            "total_accepted": total_accepted,
            "acceptance_rate": task_acceptance_rate,
        }
        exp_logger.log_metric(f"{task_name}_acceptance_rate", task_acceptance_rate)

    # Save summary statistics
    summary_path = output_dir / "profiling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(task_stats, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("Phase 1: Acceptance Rate Profiling — Summary")
    print("=" * 60)
    for task_name, stats in task_stats.items():
        print(f"  {task_name:15s}: acceptance_rate={stats['acceptance_rate']:.4f} "
              f"({stats['total_accepted']}/{stats['total_drafted']} tokens)")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")

    exp_logger.log_phase_end(1, summary=task_stats)


if __name__ == "__main__":
    main()
