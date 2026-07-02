"""Paper figure generation for DAHD speculative decoding.

Generates all publication-quality figures and tables for the paper,
including acceptance distributions, speedup curves, method comparisons,
and mode switch visualizations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.statistics import (
    compute_difficulty_distribution,
    compute_speedup_by_difficulty_bin,
)
from src.analysis.visualization import (
    set_paper_style,
    COLORS,
    TASK_COLORS,
)

logger = logging.getLogger(__name__)


class PaperFigureGenerator:
    """Generates all figures and tables for the DAHD paper.

    Produces publication-ready outputs including:
    - Figure 1: Acceptance rate distributions (bimodal evidence)
    - Figure 2: Speedup vs difficulty curves
    - Figure 3: Method comparison bar charts
    - Figure 4: Mode switch timeline visualization
    - Table 1: Main results (speedup + acceptance + correctness)
    - Table 2: Ablation study results
    """

    def __init__(self, data_dir: str | Path, output_dir: str | Path) -> None:
        """Initialize the paper figure generator.

        Args:
            data_dir: Directory containing processed experiment data.
            output_dir: Directory to write generated figures and tables.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_paper_style()

    def _load_jsonl(self, filepath: Path) -> list[dict]:
        """Load a JSONL file into a list of dictionaries."""
        data = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def generate_figure_1_acceptance_distribution(self) -> Path:
        """Generate Figure 1: Acceptance rate distributions for 3 tasks.

        Creates a 1×3 subplot figure showing easy/hard colored histograms
        for GSM8K, MATH, and HumanEval respectively.

        Returns:
            Path to the saved figure.
        """
        set_paper_style()
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

        tasks = [("gsm8k", "GSM8K"), ("math", "MATH"), ("humaneval", "HumanEval")]

        for ax, (task_key, task_label) in zip(axes, tasks):
            filepath = self.data_dir / f"{task_key}_per_token.jsonl"
            if not filepath.exists():
                ax.set_title(f"{task_label} (no data)")
                continue

            token_data = self._load_jsonl(filepath)

            easy_rates = [d["acceptance_rate"] for d in token_data
                         if d.get("difficulty_label") == "easy"]
            hard_rates = [d["acceptance_rate"] for d in token_data
                         if d.get("difficulty_label") == "hard"]

            bins = np.linspace(0, 1, 30)
            ax.hist(easy_rates, bins=bins, alpha=0.7, color=COLORS["easy"],
                    label=f"Easy (n={len(easy_rates)})", edgecolor="white", linewidth=0.3)
            ax.hist(hard_rates, bins=bins, alpha=0.7, color=COLORS["hard"],
                    label=f"Hard (n={len(hard_rates)})", edgecolor="white", linewidth=0.3)

            ax.set_xlabel("Acceptance Rate")
            ax.set_title(task_label)
            ax.legend(frameon=True, fontsize=8)
            ax.set_xlim(0, 1)

        axes[0].set_ylabel("Count")

        fig.suptitle("Figure 1: Bimodal Acceptance Rate Distribution", fontsize=13, y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / "figure_1_acceptance_distribution.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Figure 1 saved to {output_path}")
        return output_path

    def generate_figure_2_speedup_curve(self) -> Path:
        """Generate Figure 2: Speedup vs difficulty curves per task.

        Creates a line plot showing how speedup varies with task difficulty
        (probe confidence), with separate colored lines for each task.

        Returns:
            Path to the saved figure.
        """
        set_paper_style()
        fig, ax = plt.subplots(figsize=(7, 5))

        tasks = [("gsm8k", "GSM8K"), ("math", "MATH"), ("humaneval", "HumanEval"), ("mbpp", "MBPP")]

        for task_key, task_label in tasks:
            filepath = self.data_dir / f"{task_key}_per_sequence.csv"
            if not filepath.exists():
                continue

            df = pd.read_csv(filepath)
            if "avg_probe_confidence" not in df.columns or "speedup" not in df.columns:
                continue

            bin_data = compute_speedup_by_difficulty_bin(df, num_bins=8)
            if len(bin_data) == 0:
                continue

            color = TASK_COLORS.get(task_label, "#333333")
            ax.plot(bin_data["bin_center"], bin_data["mean_speedup"],
                    "-o", color=color, label=task_label, markersize=5, linewidth=1.8)
            ax.fill_between(
                bin_data["bin_center"],
                bin_data["mean_speedup"] - bin_data["std_speedup"],
                bin_data["mean_speedup"] + bin_data["std_speedup"],
                alpha=0.15, color=color,
            )

        ax.set_xlabel("Average Probe Confidence (→ Higher = Easier)")
        ax.set_ylabel("Speedup (×)")
        ax.set_title("Figure 2: Speedup vs. Task Difficulty")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(frameon=True)

        plt.tight_layout()
        output_path = self.output_dir / "figure_2_speedup_curve.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Figure 2 saved to {output_path}")
        return output_path

    def generate_figure_3_method_comparison(self) -> Path:
        """Generate Figure 3: Grouped bar chart comparing methods across tasks.

        Creates a grouped bar chart with 4 tasks × 6 methods showing
        speedup values for each combination.

        Returns:
            Path to the saved figure.
        """
        set_paper_style()
        fig, ax = plt.subplots(figsize=(12, 5))

        # Load results for all methods
        methods = ["dahd", "vanilla_sd", "eagle", "medusa", "gumiho", "baseline_ar"]
        method_labels = ["DAHD (Ours)", "Vanilla SD", "EAGLE", "Medusa", "Gumiho", "AR Baseline"]
        tasks = ["GSM8K", "MATH", "HumanEval", "MBPP"]

        results_path = self.data_dir / "main_results.csv"
        if results_path.exists():
            results_df = pd.read_csv(results_path)
        else:
            # Create placeholder data structure
            logger.warning("main_results.csv not found, using placeholder")
            results_df = pd.DataFrame(columns=["method", "task", "mean_speedup"])

        x = np.arange(len(tasks))
        n_methods = len(methods)
        width = 0.12

        method_colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

        for i, (method, label) in enumerate(zip(methods, method_labels)):
            method_data = results_df[results_df["method"] == method] if len(results_df) > 0 else pd.DataFrame()
            values = []
            for task in tasks:
                task_row = method_data[method_data["task"] == task] if len(method_data) > 0 else pd.DataFrame()
                if len(task_row) > 0:
                    values.append(float(task_row["mean_speedup"].iloc[0]))
                else:
                    values.append(0.0)

            offset = (i - n_methods / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=label,
                   color=method_colors[i], edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Task")
        ax.set_ylabel("Speedup (×)")
        ax.set_title("Figure 3: Method Comparison Across Tasks")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks)
        ax.legend(frameon=True, loc="upper left", ncol=2, fontsize=8)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        plt.tight_layout()
        output_path = self.output_dir / "figure_3_method_comparison.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Figure 3 saved to {output_path}")
        return output_path

    def generate_figure_4_mode_switch_vis(self) -> Path:
        """Generate Figure 4: Mode switch timeline visualization.

        Creates a visualization showing how the DAHD model switches between
        parallel and AR modes during inference on example sequences.

        Returns:
            Path to the saved figure.
        """
        set_paper_style()
        fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=False)

        # Load example sequences
        examples_path = self.data_dir / "mode_switch_examples.jsonl"
        if examples_path.exists():
            examples = self._load_jsonl(examples_path)[:3]
        else:
            # Generate synthetic example for illustration
            logger.warning("mode_switch_examples.jsonl not found, generating synthetic data")
            examples = []
            for i, pattern in enumerate(["mostly_parallel", "mixed", "mostly_ar"]):
                n_tokens = 80 + i * 20
                positions = list(range(n_tokens))
                if pattern == "mostly_parallel":
                    modes = ["parallel"] * n_tokens
                    for j in range(20, 35):
                        modes[j] = "ar"
                elif pattern == "mixed":
                    modes = []
                    for j in range(n_tokens):
                        modes.append("parallel" if (j // 10) % 2 == 0 else "ar")
                else:
                    modes = ["ar"] * n_tokens
                    for j in range(0, 15):
                        modes[j] = "parallel"
                examples.append({"token_pos": positions, "mode": modes})

        labels = ["Easy Example (GSM8K)", "Medium Example (MATH)", "Hard Example (HumanEval)"]

        for ax, example, label in zip(axes, examples, labels):
            positions = np.array(example["token_pos"])
            modes = example["mode"]
            mode_y = np.array([1 if m == "parallel" else 0 for m in modes])

            for i in range(len(positions) - 1):
                color = COLORS["parallel"] if modes[i] == "parallel" else COLORS["ar"]
                ax.fill_between(
                    [positions[i], positions[i + 1]],
                    [mode_y[i], mode_y[i]],
                    alpha=0.7, color=color, linewidth=0
                )

            n_switches = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i - 1])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["AR", "Parallel"])
            ax.set_title(f"{label} ({n_switches} switches)", fontsize=10)
            ax.set_xlim(positions[0], positions[-1])

        axes[-1].set_xlabel("Token Position")
        fig.suptitle("Figure 4: Mode Switch Visualization", fontsize=12, y=1.01)

        plt.tight_layout()
        output_path = self.output_dir / "figure_4_mode_switch.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Figure 4 saved to {output_path}")
        return output_path

    def generate_table_1_main_results(self) -> Path:
        """Generate Table 1: Main results in LaTeX format.

        Produces a LaTeX table showing speedup, acceptance rate, and
        correctness for all methods across all tasks.

        Returns:
            Path to the saved LaTeX file.
        """
        results_path = self.data_dir / "main_results.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)
        else:
            logger.warning("main_results.csv not found, generating empty table")
            df = pd.DataFrame(columns=["method", "task", "mean_speedup", "mean_acceptance", "mean_correctness"])

        methods = ["dahd", "vanilla_sd", "eagle", "medusa", "gumiho", "baseline_ar"]
        method_labels = {
            "dahd": r"\textbf{DAHD (Ours)}",
            "vanilla_sd": "Vanilla SD",
            "eagle": "EAGLE",
            "medusa": "Medusa",
            "gumiho": "Gumiho",
            "baseline_ar": "AR Baseline",
        }
        tasks = ["GSM8K", "MATH", "HumanEval", "MBPP"]

        latex_lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Main Results: Comparison of speculative decoding methods across reasoning and coding tasks.}",
            r"\label{tab:main_results}",
            r"\resizebox{\textwidth}{!}{",
            r"\begin{tabular}{l" + "ccc" * len(tasks) + "}",
            r"\toprule",
        ]

        # Header row
        header = r"Method"
        for task in tasks:
            header += rf" & \multicolumn{{3}}{{c}}{{{task}}}"
        header += r" \\"
        latex_lines.append(header)

        subheader = ""
        for _ in tasks:
            subheader += r" & Speedup & Accept. & Corr."
        subheader += r" \\"
        latex_lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}")
        latex_lines.append(subheader)
        latex_lines.append(r"\midrule")

        # Data rows
        for method in methods:
            label = method_labels.get(method, method)
            row = label
            method_df = df[df["method"] == method] if len(df) > 0 else pd.DataFrame()

            for task in tasks:
                task_df = method_df[method_df["task"] == task] if len(method_df) > 0 else pd.DataFrame()
                if len(task_df) > 0:
                    speedup = task_df["mean_speedup"].iloc[0]
                    accept = task_df["mean_acceptance"].iloc[0] * 100
                    correct = task_df["mean_correctness"].iloc[0] * 100
                    row += f" & {speedup:.2f}$\\times$ & {accept:.1f}\\% & {correct:.1f}\\%"
                else:
                    row += r" & -- & -- & --"

            row += r" \\"
            latex_lines.append(row)

        latex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
        ])

        output_path = self.output_dir / "table_1_main_results.tex"
        output_path.write_text("\n".join(latex_lines))
        logger.info(f"Table 1 saved to {output_path}")
        return output_path

    def generate_table_2_ablation(self) -> Path:
        """Generate Table 2: Ablation study results in LaTeX format.

        Produces a LaTeX table showing the effect of removing each
        component on speedup, acceptance, and correctness.

        Returns:
            Path to the saved LaTeX file.
        """
        ablation_path = self.data_dir / "ablation_results.csv"
        if ablation_path.exists():
            df = pd.read_csv(ablation_path)
        else:
            logger.warning("ablation_results.csv not found, generating placeholder table")
            df = pd.DataFrame({
                "ablation": ["Full DAHD", "Fixed Parallel", "Fixed AR",
                            "Fixed Split (Gumiho)", "Probe Only (no EMA)", "No Sharing"],
                "speedup": [0.0] * 6,
                "acceptance_rate": [0.0] * 6,
                "correctness": [0.0] * 6,
            })

        latex_lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Ablation Study: Impact of each DAHD component.}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Configuration & Speedup ($\times$) & Accept. (\%) & Corr. (\%) \\",
            r"\midrule",
        ]

        for _, row in df.iterrows():
            name = row["ablation"]
            speedup = row.get("speedup", 0)
            acceptance = row.get("acceptance_rate", 0) * 100
            correctness = row.get("correctness", 0) * 100

            if "Full" in str(name) or "DAHD" in str(name):
                line = rf"\textbf{{{name}}} & \textbf{{{speedup:.2f}}} & \textbf{{{acceptance:.1f}}} & \textbf{{{correctness:.1f}}} \\"
            else:
                line = f"{name} & {speedup:.2f} & {acceptance:.1f} & {correctness:.1f} \\\\"

            latex_lines.append(line)

        latex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        output_path = self.output_dir / "table_2_ablation.tex"
        output_path.write_text("\n".join(latex_lines))
        logger.info(f"Table 2 saved to {output_path}")
        return output_path

    def generate_all(self) -> dict[str, Path]:
        """Generate all paper figures and tables.

        Returns:
            Dictionary mapping figure/table names to their output paths.
        """
        logger.info("Generating all paper figures and tables...")
        outputs: dict[str, Path] = {}

        outputs["figure_1"] = self.generate_figure_1_acceptance_distribution()
        outputs["figure_2"] = self.generate_figure_2_speedup_curve()
        outputs["figure_3"] = self.generate_figure_3_method_comparison()
        outputs["figure_4"] = self.generate_figure_4_mode_switch_vis()
        outputs["table_1"] = self.generate_table_1_main_results()
        outputs["table_2"] = self.generate_table_2_ablation()

        logger.info(f"All outputs generated: {list(outputs.keys())}")
        return outputs
