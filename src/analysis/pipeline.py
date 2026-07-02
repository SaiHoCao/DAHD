"""Analysis pipeline for DAHD speculative decoding experiments.

Orchestrates data loading, statistical computation, visualization generation,
and report output for all experimental phases.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.analysis.statistics import (
    compute_per_position_acceptance,
    compute_difficulty_distribution,
    compute_mode_switch_frequency,
    compute_correlation,
    compute_speedup_by_difficulty_bin,
)
from src.analysis.visualization import (
    plot_acceptance_histogram,
    plot_violin_acceptance_by_task,
    plot_scatter_confidence_vs_acceptance,
    plot_speedup_comparison_bar,
    plot_mode_switch_timeline,
    plot_speedup_vs_difficulty,
    set_paper_style,
)

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Orchestrates the full analysis pipeline for DAHD experiments.

    Supports three main analysis phases:
    1. Phase 1: Bimodal hypothesis validation (profiling data)
    2. End-to-end: Full method comparison analysis
    3. Ablation: Component importance analysis
    """

    def __init__(self, data_dir: str | Path, output_dir: str | Path) -> None:
        """Initialize the analysis pipeline.

        Args:
            data_dir: Directory containing raw experiment data (JSONL files).
            output_dir: Directory to write analysis outputs (figures, tables, reports).
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories for organized output
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.reports_dir = self.output_dir / "reports"
        for d in [self.figures_dir, self.tables_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_jsonl(self, filepath: Path) -> list[dict]:
        """Load a JSONL file into a list of dictionaries."""
        data = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _run_bimodal_test(self, confidences: np.ndarray) -> dict[str, Any]:
        """Run bimodality tests on confidence distribution.

        Tests include:
        - Hartigan's dip test (via approximation)
        - K-means with k=2 separation metric
        - KL divergence from uniform
        """
        from scipy.stats import entropy
        from scipy.cluster.vq import kmeans2

        results: dict[str, Any] = {}

        # Hartigan's dip test approximation
        # Using sorted data to approximate dip statistic
        sorted_data = np.sort(confidences)
        n = len(sorted_data)
        uniform_cdf = np.linspace(0, 1, n)
        empirical_cdf = np.arange(1, n + 1) / n
        dip_stat = float(np.max(np.abs(empirical_cdf - uniform_cdf)))
        results["dip_statistic"] = dip_stat
        results["dip_significant"] = dip_stat > 0.05  # Simplified threshold

        # K-means with k=2
        if len(confidences) > 10:
            centroids, labels = kmeans2(confidences.reshape(-1, 1), 2, minit="points")
            cluster_0 = confidences[labels == 0]
            cluster_1 = confidences[labels == 1]
            if len(cluster_0) > 0 and len(cluster_1) > 0:
                separation = abs(float(np.mean(cluster_0) - np.mean(cluster_1)))
                results["kmeans_separation"] = separation
                results["kmeans_cluster_sizes"] = [int(len(cluster_0)), int(len(cluster_1))]
            else:
                results["kmeans_separation"] = 0.0
                results["kmeans_cluster_sizes"] = [int(len(cluster_0)), int(len(cluster_1))]
        else:
            results["kmeans_separation"] = 0.0
            results["kmeans_cluster_sizes"] = [0, 0]

        # KL divergence from uniform
        hist_counts, _ = np.histogram(confidences, bins=50, range=(0, 1), density=True)
        hist_counts = hist_counts + 1e-10  # Avoid zeros
        uniform_dist = np.ones_like(hist_counts) / len(hist_counts)
        kl_div = float(entropy(hist_counts, uniform_dist))
        results["kl_divergence_from_uniform"] = kl_div

        # Go/No-Go decision
        results["is_bimodal"] = (
            results["dip_significant"]
            and results["kmeans_separation"] > 0.2
            and kl_div > 0.5
        )

        return results

    def run_phase1_analysis(self) -> dict[str, Any]:
        """Run Phase 1 analysis: bimodal hypothesis validation.

        Loads per-token profiling JSONL data, computes statistics,
        generates Figure 1 (acceptance distribution), runs bimodal tests,
        and outputs a profiling report.

        Returns:
            Dictionary containing all computed statistics and test results.
        """
        logger.info("Starting Phase 1 analysis: bimodal hypothesis validation")
        results: dict[str, Any] = {}

        # Load per-token data for each task
        tasks = ["gsm8k", "math", "humaneval"]
        all_token_data: dict[str, list[dict]] = {}

        for task in tasks:
            filepath = self.data_dir / f"{task}_per_token.jsonl"
            if filepath.exists():
                all_token_data[task] = self._load_jsonl(filepath)
                logger.info(f"Loaded {len(all_token_data[task])} records for {task}")
            else:
                logger.warning(f"Data file not found: {filepath}")

        if not all_token_data:
            logger.error("No task data found. Aborting Phase 1 analysis.")
            return {"error": "No data files found"}

        # Compute per-task statistics
        for task, token_data in all_token_data.items():
            results[task] = {}
            results[task]["per_position_acceptance"] = compute_per_position_acceptance(token_data)
            results[task]["difficulty_distribution"] = compute_difficulty_distribution(token_data)

            # Correlation between confidence and acceptance
            confidences = [d["probe_confidence"] for d in token_data]
            acceptances = [d["accepted"] for d in token_data]
            results[task]["correlation"] = compute_correlation(confidences, acceptances)

            # Generate per-task histogram
            plot_acceptance_histogram(
                token_data, task.upper(),
                self.figures_dir / f"acceptance_hist_{task}.png"
            )

            # Run bimodal test
            conf_arr = np.array(confidences)
            results[task]["bimodal_test"] = self._run_bimodal_test(conf_arr)

        # Generate cross-task violin plot
        acceptance_by_task = {}
        for task, token_data in all_token_data.items():
            acceptance_by_task[task.upper()] = [d["accepted"] for d in token_data]
        plot_violin_acceptance_by_task(
            acceptance_by_task, self.figures_dir / "violin_acceptance_by_task.png"
        )

        # Write profiling report
        self._write_profiling_report(results)

        logger.info("Phase 1 analysis complete")
        return results

    def _write_profiling_report(self, results: dict[str, Any]) -> None:
        """Write Phase 1 profiling report as markdown."""
        report_path = self.reports_dir / "profiling_report.md"
        lines = [
            "# Phase 1: Acceptance Rate Profiling Report",
            "",
            "## Summary",
            "",
        ]

        for task, task_results in results.items():
            if task == "error":
                continue
            lines.append(f"### {task.upper()}")
            if "difficulty_distribution" in task_results:
                dist = task_results["difficulty_distribution"]
                lines.append(f"- Mean confidence: {dist['mean']:.4f}")
                lines.append(f"- Std: {dist['std']:.4f}")
                lines.append(f"- Median: {dist['median']:.4f}")
            if "bimodal_test" in task_results:
                bt = task_results["bimodal_test"]
                lines.append(f"- Bimodal: {'YES' if bt['is_bimodal'] else 'NO'}")
                lines.append(f"- Dip statistic: {bt['dip_statistic']:.4f}")
                lines.append(f"- K-means separation: {bt['kmeans_separation']:.4f}")
                lines.append(f"- KL divergence: {bt['kl_divergence_from_uniform']:.4f}")
            if "correlation" in task_results:
                corr = task_results["correlation"]
                lines.append(f"- Pearson r (conf vs acc): {corr['pearson_r']:.4f} (p={corr['pearson_p']:.2e})")
            lines.append("")

        report_path.write_text("\n".join(lines))
        logger.info(f"Profiling report written to {report_path}")

    def run_e2e_analysis(self) -> dict[str, Any]:
        """Run end-to-end experiment results analysis.

        Loads per-sequence data for all methods, computes speedup and
        acceptance rates, generates comparison figures, and outputs results CSV.

        Returns:
            Dictionary containing aggregated results across all methods and tasks.
        """
        logger.info("Starting end-to-end analysis")
        results: dict[str, Any] = {}

        # Load per-sequence results for each method
        methods_dir = self.data_dir / "methods"
        if not methods_dir.exists():
            methods_dir = self.data_dir

        method_results: dict[str, pd.DataFrame] = {}
        method_files = list(self.data_dir.glob("*_per_sequence.csv"))

        for fpath in method_files:
            method_name = fpath.stem.replace("_per_sequence", "")
            method_results[method_name] = pd.read_csv(fpath)
            logger.info(f"Loaded {len(method_results[method_name])} sequences for {method_name}")

        if not method_results:
            logger.warning("No per-sequence CSV files found")
            return {"error": "No per-sequence data found"}

        # Compute speedup comparison
        speedup_dict: dict[str, dict[str, float]] = {}
        for method, df in method_results.items():
            if "task" in df.columns and "speedup" in df.columns:
                speedup_dict[method] = df.groupby("task")["speedup"].mean().to_dict()

        if speedup_dict:
            plot_speedup_comparison_bar(
                speedup_dict, self.figures_dir / "speedup_comparison.png"
            )

        # Compute speedup vs difficulty for DAHD
        if "dahd" in method_results:
            dahd_df = method_results["dahd"]
            if "avg_probe_confidence" in dahd_df.columns and "speedup" in dahd_df.columns:
                bin_data = compute_speedup_by_difficulty_bin(dahd_df)
                plot_speedup_vs_difficulty(
                    bin_data, self.figures_dir / "speedup_vs_difficulty.png"
                )
                results["speedup_by_difficulty"] = bin_data.to_dict("records")

        # Output main results CSV
        all_rows = []
        for method, df in method_results.items():
            for task in df["task"].unique() if "task" in df.columns else ["all"]:
                task_df = df[df["task"] == task] if "task" in df.columns else df
                row = {
                    "method": method,
                    "task": task,
                    "mean_speedup": float(task_df["speedup"].mean()) if "speedup" in task_df else 0,
                    "mean_acceptance": float(task_df["acceptance_rate"].mean()) if "acceptance_rate" in task_df else 0,
                    "mean_correctness": float(task_df["correct"].mean()) if "correct" in task_df else 0,
                    "num_samples": len(task_df),
                }
                all_rows.append(row)

        results_df = pd.DataFrame(all_rows)
        results_df.to_csv(self.tables_dir / "main_results.csv", index=False)
        results["main_results"] = all_rows

        logger.info("End-to-end analysis complete")
        return results

    def run_ablation_analysis(self, ablation_results: dict[str, dict]) -> dict[str, Any]:
        """Run ablation study results analysis.

        Args:
            ablation_results: Dict mapping ablation_name -> {metric: value}.

        Returns:
            Dictionary containing processed ablation results.
        """
        logger.info("Starting ablation analysis")

        # Convert to DataFrame
        rows = []
        for ablation_name, metrics in ablation_results.items():
            row = {"ablation": ablation_name, **metrics}
            rows.append(row)

        ablation_df = pd.DataFrame(rows)
        ablation_df.to_csv(self.tables_dir / "ablation_results.csv", index=False)

        # Generate LaTeX table
        latex_lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Ablation Study Results}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Configuration & Speedup ($\times$) & Acceptance (\%) & Correctness (\%) \\",
            r"\midrule",
        ]

        for _, row in ablation_df.iterrows():
            speedup = row.get("speedup", 0)
            acceptance = row.get("acceptance_rate", 0) * 100
            correctness = row.get("correctness", 0) * 100
            latex_lines.append(
                f"  {row['ablation']} & {speedup:.2f} & {acceptance:.1f} & {correctness:.1f} \\\\"
            )

        latex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        latex_path = self.tables_dir / "ablation_table.tex"
        latex_path.write_text("\n".join(latex_lines))
        logger.info(f"LaTeX table written to {latex_path}")

        return {"ablation_df": ablation_df.to_dict("records"), "latex_path": str(latex_path)}
