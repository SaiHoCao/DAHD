#!/usr/bin/env python3
"""Phase 1: Bimodal distribution hypothesis test for acceptance rates.

Loads per-token profiling data from Phase 1, runs statistical tests
(Hartigan's dip test, KL divergence, silhouette improvement) to validate
whether acceptance rates follow a bimodal distribution, and produces a
Go/No-Go decision for the DAHD architecture.

Usage:
    python experiments/phase1_profiling/run_bimodal_test.py \
        --data_dir results/phase1_results \
        --output_dir results/phase1_results \
        --significance_level 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.statistical_tests import test_bimodal_hypothesis
from src.analysis.pipeline import AnalysisPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Bimodal distribution hypothesis test on acceptance rate data."
    )
    parser.add_argument(
        "--data_dir", type=str, default="results/phase1_results",
        help="Directory containing phase1 per-token JSONL files."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/phase1_results",
        help="Directory to save test results and visualizations."
    )
    parser.add_argument(
        "--significance_level", type=float, default=0.05,
        help="Significance level for hypothesis tests."
    )
    parser.add_argument(
        "--kl_threshold", type=float, default=0.5,
        help="KL divergence threshold for Go decision."
    )
    return parser.parse_args()


def load_per_token_data(data_dir: Path) -> dict[str, list[dict]]:
    """Load all per-token JSONL files from the data directory.

    Args:
        data_dir: Directory containing *_per_token.jsonl files.

    Returns:
        Dictionary mapping task_name -> list of per-token metric dicts.
    """
    task_data: dict[str, list[dict]] = {}
    for jsonl_file in sorted(data_dir.glob("*_per_token.jsonl")):
        task_name = jsonl_file.stem.replace("_per_token", "")
        records = []
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            task_data[task_name] = records
            print(f"  Loaded {len(records)} records for task: {task_name}")
    return task_data


def main() -> None:
    """Run bimodal hypothesis test pipeline."""
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1: Bimodal Distribution Hypothesis Test")
    print("=" * 60)
    print(f"  Data directory:      {data_dir}")
    print(f"  Significance level:  {args.significance_level}")
    print(f"  KL threshold:        {args.kl_threshold}")
    print()

    # Load per-token data
    print("[Step 1] Loading per-token profiling data...")
    task_data = load_per_token_data(data_dir)

    if not task_data:
        print("ERROR: No per-token data found. Run run_acceptance_profiling.py first.")
        sys.exit(1)

    # Run bimodal hypothesis tests per task
    print("\n[Step 2] Running bimodal hypothesis tests...")
    all_results: dict[str, dict] = {}
    go_decisions: dict[str, bool] = {}

    for task_name, records in task_data.items():
        print(f"\n  --- Task: {task_name.upper()} ({len(records)} tokens) ---")

        # Extract probe confidence values for bimodality testing
        confidences = [r.get("probe_confidence", 0.0) for r in records]

        # Run the comprehensive bimodal test
        result = test_bimodal_hypothesis(confidences)
        all_results[task_name] = result

        # Go/No-Go decision
        dip_significant = result["dip_p_value"] < args.significance_level
        kl_sufficient = result["kl_divergence"] > args.kl_threshold
        is_go = dip_significant and kl_sufficient
        go_decisions[task_name] = is_go

        print(f"    Dip statistic:          {result['dip_statistic']:.4f}")
        print(f"    Dip p-value:            {result['dip_p_value']:.6f} "
              f"({'< ' + str(args.significance_level) if dip_significant else '>= ' + str(args.significance_level)})")
        print(f"    KL divergence:          {result['kl_divergence']:.4f} "
              f"({'> ' + str(args.kl_threshold) if kl_sufficient else '<= ' + str(args.kl_threshold)})")
        print(f"    Silhouette improvement: {result['silhouette_improvement']:.4f}")
        print(f"    Is bimodal (internal):  {result['is_bimodal']}")
        print(f"    GO decision:            {'GO ✓' if is_go else 'NO-GO ✗'}")

    # Run visualization via AnalysisPipeline
    print("\n[Step 3] Generating visualizations...")
    pipeline = AnalysisPipeline(data_dir=data_dir, output_dir=output_dir)
    try:
        pipeline.run_phase1_analysis()
        print("  Visualizations saved to:", output_dir / "figures")
    except Exception as e:
        print(f"  WARNING: Visualization generation failed: {e}")
        print("  Continuing without visualizations...")

    # Overall Go/No-Go decision
    overall_go = all(go_decisions.values()) if go_decisions else False

    # Generate report
    print("\n[Step 4] Generating bimodal test report...")
    report_lines = [
        "# Bimodal Hypothesis Test Report",
        "",
        "## Parameters",
        f"- Significance level (α): {args.significance_level}",
        f"- KL divergence threshold: {args.kl_threshold}",
        f"- Data directory: `{data_dir}`",
        "",
        "## Per-Task Results",
        "",
        "| Task | Dip Stat | Dip p-value | KL Div | Silhouette | Bimodal | Decision |",
        "|------|----------|-------------|--------|------------|---------|----------|",
    ]

    for task_name, result in all_results.items():
        decision = "GO" if go_decisions[task_name] else "NO-GO"
        report_lines.append(
            f"| {task_name.upper()} | {result['dip_statistic']:.4f} | "
            f"{result['dip_p_value']:.4f} | {result['kl_divergence']:.4f} | "
            f"{result['silhouette_improvement']:.4f} | "
            f"{'Yes' if result['is_bimodal'] else 'No'} | {decision} |"
        )

    report_lines.extend([
        "",
        "## Overall Decision",
        "",
        f"**{'GO — Proceed with DAHD dual-branch architecture' if overall_go else 'NO-GO (Plan B) — Use soft routing instead of hard mode switching'}**",
        "",
        "## Interpretation",
        "",
    ])

    if overall_go:
        report_lines.extend([
            "The acceptance rate distribution shows statistically significant bimodality",
            "across all tested tasks. This validates the core hypothesis that tokens can be",
            "meaningfully separated into 'easy' (high acceptance) and 'hard' (low acceptance)",
            "categories, supporting the DAHD dual-branch architecture design.",
        ])
    else:
        report_lines.extend([
            "The bimodality evidence is insufficient for one or more tasks.",
            "Recommendation: Fall back to Plan B with soft routing (continuous blending",
            "of AR and parallel branch outputs weighted by difficulty score) instead of",
            "hard mode switching.",
        ])

    report_path = output_dir / "bimodal_test_report.md"
    report_path.write_text("\n".join(report_lines))

    # Save raw results as JSON
    results_path = output_dir / "bimodal_test_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "per_task": all_results,
            "go_decisions": go_decisions,
            "overall_go": overall_go,
            "parameters": {
                "significance_level": args.significance_level,
                "kl_threshold": args.kl_threshold,
            },
        }, f, indent=2)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL DECISION:", "GO ✓" if overall_go else "NO-GO ✗ (Plan B)")
    print("=" * 60)
    print(f"  Report saved to: {report_path}")
    print(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
