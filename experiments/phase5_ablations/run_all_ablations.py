#!/usr/bin/env python3
"""Phase 5: Ablation experiments for DAHD speculative decoding.

Runs all predefined ablation configurations to quantify the contribution
of each DAHD component (AR branch, Parallel branch, dynamic routing,
EMA smoothing, shared backbone) to overall performance.

Usage:
    # Run all ablations
    python experiments/phase5_ablations/run_all_ablations.py \
        --checkpoint checkpoints/dahd_final.pt \
        --tasks GSM8K,MATH,HumanEval \
        --num_samples 100 \
        --output_dir results/phase5_results

    # Run specific ablations
    python experiments/phase5_ablations/run_all_ablations.py \
        --checkpoint checkpoints/dahd_final.pt \
        --ablations fixed_parallel,fixed_ar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import ExperimentConfig
from src.ablations.runner import AblationSuite
from src.ablations.ablation_config import get_all_ablation_configs, get_ablation_by_name
from src.analysis.pipeline import AnalysisPipeline
from src.benchmarks.task_runners import get_task_runner
from src.drafters.dahd_draft_module import DAHDDraftModule
from src.utils.device_utils import setup_deterministic
from src.utils.logging_utils import ExperimentLogger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 5: Run ablation experiments for DAHD component analysis."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained DAHD draft module checkpoint."
    )
    parser.add_argument(
        "--tasks", type=str, default="GSM8K,MATH,HumanEval",
        help="Comma-separated list of benchmark tasks."
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="Number of samples per task per ablation."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/phase5_results",
        help="Directory to save ablation results."
    )
    parser.add_argument(
        "--ablations", type=str, default="all",
        help="Comma-separated ablation names, or 'all' for all configs."
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Target model for hidden dim / vocab size inference."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device.")
    return parser.parse_args()


def main() -> None:
    """Main ablation experiment pipeline."""
    args = parse_args()
    setup_deterministic(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_logger = ExperimentLogger(
        experiment_name="phase5_ablations",
        log_dir=str(output_dir / "logs"),
    )
    exp_logger.log_phase_start(5, "Ablation experiments")

    tasks = [t.strip().lower().replace("-", "_") for t in args.tasks.split(",")]

    print("=" * 60)
    print("Phase 5: Ablation Experiments")
    print("=" * 60)
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Tasks:        {tasks}")
    print(f"  Num samples:  {args.num_samples}")
    print(f"  Ablations:    {args.ablations}")
    print()

    # Determine ablation configs
    if args.ablations.lower() == "all":
        ablation_configs = get_all_ablation_configs()
    else:
        ablation_names = [a.strip() for a in args.ablations.split(",")]
        ablation_configs = [get_ablation_by_name(name) for name in ablation_names]

    print(f"  Running {len(ablation_configs)} ablation(s):")
    for cfg in ablation_configs:
        print(f"    - {cfg.name}: {cfg.description[:60]}...")

    # Load DAHD model
    print("\n[1/4] Loading DAHD draft module...")
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    hidden_dim = model_config.hidden_size
    vocab_size = model_config.vocab_size

    draft_module = DAHDDraftModule(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        max_k=6,
    ).to(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"  ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    state_dict = torch.load(checkpoint_path, map_location=args.device)
    draft_module.load_state_dict(state_dict)
    draft_module.eval()
    print(f"  Loaded checkpoint: {checkpoint_path}")

    # Prepare task configurations
    print("\n[2/4] Loading task datasets...")
    task_configs = []
    for task_name in tasks:
        runner = get_task_runner(task_name)
        dataset = runner.load_dataset(num_samples=args.num_samples)
        task_configs.append({
            "name": task_name,
            "runner": runner,
            "dataset": dataset,
        })
        print(f"  {task_name}: {len(dataset)} samples loaded")

    # Run ablation suite
    print("\n[3/4] Running ablation suite...")
    suite = AblationSuite(
        model=draft_module,
        tasks=task_configs,
        output_dir=output_dir,
        num_samples=args.num_samples,
    )
    all_results = suite.run_all_ablations()

    # Generate report
    print("\n[4/4] Generating ablation reports...")
    report = suite.generate_report()

    # Run visualization analysis
    try:
        # Prepare ablation results for analysis pipeline
        ablation_metrics = {}
        for name, result in all_results.items():
            ablation_metrics[name] = result.get("aggregated", {})

        pipeline = AnalysisPipeline(data_dir=output_dir, output_dir=output_dir)
        pipeline.run_ablation_analysis(ablation_metrics)
        print(f"  Visualizations and LaTeX table saved to: {output_dir / 'tables'}")
    except Exception as e:
        print(f"  WARNING: Analysis pipeline failed: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    header = f"{'Configuration':<22} {'Speedup(x)':<12} {'Accept%':<10} {'Correct%':<10} {'Δ Speed':<10}"
    print(header)
    print("-" * len(header))
    for name, result in all_results.items():
        agg = result.get("aggregated", {})
        comp = result.get("comparison", {})
        speedup = agg.get("speedup", 0.0)
        acceptance = agg.get("acceptance_rate", 0.0) * 100
        correctness = agg.get("correctness", 0.0) * 100
        delta = comp.get("speedup_delta", 0.0)
        delta_str = f"{delta:+.3f}" if comp else "—"
        print(f"{name:<22} {speedup:<12.3f} {acceptance:<10.1f} {correctness:<10.1f} {delta_str:<10}")
    print("=" * 60)
    print(f"\nFull report: {output_dir / 'ablation_report.md'}")
    print(f"Results JSON: {output_dir / 'ablation_all_results.json'}")

    exp_logger.log_phase_end(5, summary={"num_ablations": len(all_results)})


if __name__ == "__main__":
    main()
