#!/usr/bin/env python3
"""Analyze e2e_comparison_v2.json: paired bootstrap CI and win rates.

Produces:
  - Console report with DAHD vs EAGLE-3 statistical comparison
  - results/phase4_e2e/statistical_analysis_v2.json
"""

import json
import sys
import os
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.benchmarks.statistical_tests import paired_bootstrap_ci


def main():
    results_path = os.path.join(PROJECT_ROOT, "results", "phase4_e2e", "e2e_comparison_v2.json")
    with open(results_path) as f:
        data = json.load(f)

    per_prompt = data["per_prompt"]

    # Extract per-prompt tokens_per_sec
    dahd_tps = np.array([p["tokens_per_sec"] for p in per_prompt["dahd"]])
    eagle3_tps = np.array([p["tokens_per_sec"] for p in per_prompt["eagle3"]])
    parallel_tps = np.array([p["tokens_per_sec"] for p in per_prompt["parallel"]])
    vanilla_tps = np.array([p["tokens_per_sec"] for p in per_prompt["vanilla"]])

    print("=" * 60)
    print("DAHD v2 Statistical Analysis")
    print("=" * 60)
    print(f"Data source: {results_path}")
    print(f"Number of prompts: {len(dahd_tps)}")
    print()

    # DAHD vs EAGLE-3
    result_vs_eagle = paired_bootstrap_ci(dahd_tps, eagle3_tps)
    print("--- DAHD vs EAGLE-3 ---")
    print(f"  Mean diff (tok/s):  {result_vs_eagle['mean_diff']:+.2f}")
    print(f"  95% CI:             [{result_vs_eagle['ci_lower']:+.2f}, {result_vs_eagle['ci_upper']:+.2f}]")
    print(f"  p-value (one-sided):{result_vs_eagle['p_value']:.4f}")
    print(f"  Win rate:           {result_vs_eagle['win_rate']:.1%} "
          f"[{result_vs_eagle['win_rate_ci'][0]:.1%}, {result_vs_eagle['win_rate_ci'][1]:.1%}]")
    print()

    # DAHD vs Parallel
    result_vs_parallel = paired_bootstrap_ci(dahd_tps, parallel_tps)
    print("--- DAHD vs Parallel (Gumiho) ---")
    print(f"  Mean diff (tok/s):  {result_vs_parallel['mean_diff']:+.2f}")
    print(f"  95% CI:             [{result_vs_parallel['ci_lower']:+.2f}, {result_vs_parallel['ci_upper']:+.2f}]")
    print(f"  p-value (one-sided):{result_vs_parallel['p_value']:.4f}")
    print(f"  Win rate:           {result_vs_parallel['win_rate']:.1%} "
          f"[{result_vs_parallel['win_rate_ci'][0]:.1%}, {result_vs_parallel['win_rate_ci'][1]:.1%}]")
    print()

    # Summary statistics
    print("--- Summary (tok/s) ---")
    for name, arr in [("Vanilla", vanilla_tps), ("EAGLE-3", eagle3_tps),
                      ("Parallel", parallel_tps), ("DAHD", dahd_tps)]:
        print(f"  {name:12s}: {arr.mean():.2f} ± {arr.std():.2f} (median {np.median(arr):.2f})")
    print()

    # Save results
    output = {
        "dahd_vs_eagle3": result_vs_eagle,
        "dahd_vs_parallel": result_vs_parallel,
        "summary": {
            "vanilla": {"mean": float(vanilla_tps.mean()), "std": float(vanilla_tps.std())},
            "eagle3": {"mean": float(eagle3_tps.mean()), "std": float(eagle3_tps.std())},
            "parallel": {"mean": float(parallel_tps.mean()), "std": float(parallel_tps.std())},
            "dahd": {"mean": float(dahd_tps.mean()), "std": float(dahd_tps.std())},
        },
        "data_source": "results/phase4_e2e/e2e_comparison_v2.json",
    }

    out_path = os.path.join(PROJECT_ROOT, "results", "phase4_e2e", "statistical_analysis_v2.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Analyze e2e_comparison_v2.json: paired bootstrap CI and win rates.

Produces:
  - Console report with DAHD vs EAGLE-3 statistical comparison
  - results/phase4_e2e/statistical_analysis_v2.json
"""

import json
import sys
import os
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.benchmarks.statistical_tests import paired_bootstrap_ci


def main():
    results_path = os.path.join(PROJECT_ROOT, "results", "phase4_e2e", "e2e_comparison_v2.json")
    with open(results_path) as f:
        data = json.load(f)

    per_prompt = data["per_prompt"]

    # Extract per-prompt tokens_per_sec
    dahd_tps = np.array([p["tokens_per_sec"] for p in per_prompt["dahd"]])
    eagle3_tps = np.array([p["tokens_per_sec"] for p in per_prompt["eagle3"]])
    parallel_tps = np.array([p["tokens_per_sec"] for p in per_prompt["parallel"]])
    vanilla_tps = np.array([p["tokens_per_sec"] for p in per_prompt["vanilla"]])

    print("=" * 60)
    print("DAHD v2 Statistical Analysis")
    print("=" * 60)
    print(f"Data source: {results_path}")
    print(f"Number of prompts: {len(dahd_tps)}")
    print()

    # DAHD vs EAGLE-3
    result_vs_eagle = paired_bootstrap_ci(dahd_tps, eagle3_tps)
    print("--- DAHD vs EAGLE-3 ---")
    print(f"  Mean diff (tok/s):  {result_vs_eagle['mean_diff']:+.2f}")
    print(f"  95% CI:             [{result_vs_eagle['ci_lower']:+.2f}, {result_vs_eagle['ci_upper']:+.2f}]")
    print(f"  p-value (one-sided):{result_vs_eagle['p_value']:.4f}")
    print(f"  Win rate:           {result_vs_eagle['win_rate']:.1%} "
          f"[{result_vs_eagle['win_rate_ci'][0]:.1%}, {result_vs_eagle['win_rate_ci'][1]:.1%}]")
    print()

    # DAHD vs Parallel
    result_vs_parallel = paired_bootstrap_ci(dahd_tps, parallel_tps)
    print("--- DAHD vs Parallel (Gumiho) ---")
    print(f"  Mean diff (tok/s):  {result_vs_parallel['mean_diff']:+.2f}")
    print(f"  95% CI:             [{result_vs_parallel['ci_lower']:+.2f}, {result_vs_parallel['ci_upper']:+.2f}]")
    print(f"  p-value (one-sided):{result_vs_parallel['p_value']:.4f}")
    print(f"  Win rate:           {result_vs_parallel['win_rate']:.1%} "
          f"[{result_vs_parallel['win_rate_ci'][0]:.1%}, {result_vs_parallel['win_rate_ci'][1]:.1%}]")
    print()

    # Summary statistics
    print("--- Summary (tok/s) ---")
    for name, arr in [("Vanilla", vanilla_tps), ("EAGLE-3", eagle3_tps),
                      ("Parallel", parallel_tps), ("DAHD", dahd_tps)]:
        print(f"  {name:12s}: {arr.mean():.2f} ± {arr.std():.2f} (median {np.median(arr):.2f})")
    print()

    # Save results
    output = {
        "dahd_vs_eagle3": result_vs_eagle,
        "dahd_vs_parallel": result_vs_parallel,
        "summary": {
            "vanilla": {"mean": float(vanilla_tps.mean()), "std": float(vanilla_tps.std())},
            "eagle3": {"mean": float(eagle3_tps.mean()), "std": float(eagle3_tps.std())},
            "parallel": {"mean": float(parallel_tps.mean()), "std": float(parallel_tps.std())},
            "dahd": {"mean": float(dahd_tps.mean()), "std": float(dahd_tps.std())},
        },
        "data_source": "results/phase4_e2e/e2e_comparison_v2.json",
    }

    out_path = os.path.join(PROJECT_ROOT, "results", "phase4_e2e", "statistical_analysis_v2.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
