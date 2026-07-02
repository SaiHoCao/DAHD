"""Visualization utilities for DAHD speculative decoding experiments.

Provides publication-quality plotting functions using matplotlib and seaborn.
"""

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


# Color palette for paper figures
COLORS = {
    "easy": "#4CAF50",
    "hard": "#F44336",
    "parallel": "#2196F3",
    "ar": "#FF9800",
    "dahd": "#9C27B0",
    "baseline": "#607D8B",
}

TASK_COLORS = {
    "GSM8K": "#1f77b4",
    "MATH": "#ff7f0e",
    "HumanEval": "#2ca02c",
    "MBPP": "#d62728",
}


def set_paper_style() -> None:
    """Configure matplotlib for publication-quality figures.

    Sets font sizes, line widths, and color schemes suitable for
    academic papers (typically 2-column format).
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    sns.set_palette("Set2")


def plot_acceptance_histogram(
    data: list[dict],
    task_name: str,
    output_path: str | Path,
    figsize: tuple[float, float] = (6, 4),
) -> None:
    """Plot acceptance rate histogram colored by easy/hard classification.

    Args:
        data: List of dicts with 'acceptance_rate' and 'difficulty_label' keys.
        task_name: Name of the task for the title.
        output_path: Path to save the figure.
        figsize: Figure size in inches.
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    easy_rates = [d["acceptance_rate"] for d in data if d["difficulty_label"] == "easy"]
    hard_rates = [d["acceptance_rate"] for d in data if d["difficulty_label"] == "hard"]

    bins = np.linspace(0, 1, 30)
    ax.hist(easy_rates, bins=bins, alpha=0.7, color=COLORS["easy"],
            label=f"Easy (n={len(easy_rates)})", edgecolor="white", linewidth=0.5)
    ax.hist(hard_rates, bins=bins, alpha=0.7, color=COLORS["hard"],
            label=f"Hard (n={len(hard_rates)})", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Acceptance Rate")
    ax.set_ylabel("Count")
    ax.set_title(f"Acceptance Rate Distribution — {task_name}")
    ax.legend(frameon=True, framealpha=0.9)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_violin_acceptance_by_task(
    data_dict: dict[str, list[float]],
    output_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> None:
    """Plot violin plots comparing acceptance rates across tasks.

    Args:
        data_dict: Dict mapping task_name -> list of acceptance rates.
        output_path: Path to save the figure.
        figsize: Figure size in inches.
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for seaborn
    plot_data = []
    for task, rates in data_dict.items():
        for r in rates:
            plot_data.append({"Task": task, "Acceptance Rate": r})

    df = pd.DataFrame(plot_data)
    sns.violinplot(data=df, x="Task", y="Acceptance Rate", ax=ax,
                   inner="box", palette="Set2", linewidth=1.0)

    ax.set_ylabel("Acceptance Rate")
    ax.set_xlabel("")
    ax.set_title("Acceptance Rate Distribution by Task")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_scatter_confidence_vs_acceptance(
    data: list[dict],
    output_path: str | Path,
    figsize: tuple[float, float] = (6, 5),
) -> None:
    """Plot scatter of probe confidence vs acceptance rate.

    Args:
        data: List of dicts with 'probe_confidence' and 'acceptance_rate'.
        output_path: Path to save the figure.
        figsize: Figure size in inches.
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    x = [d["probe_confidence"] for d in data]
    y = [d["acceptance_rate"] for d in data]

    ax.scatter(x, y, alpha=0.3, s=10, color=COLORS["dahd"], edgecolors="none")

    # Add trend line
    if len(x) > 10:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, p(x_line), "--", color="black", linewidth=1.5,
                label=f"Linear fit (slope={z[0]:.3f})")
        ax.legend(frameon=True)

    ax.set_xlabel("Probe Confidence")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Probe Confidence vs. Acceptance Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_speedup_comparison_bar(
    results_dict: dict[str, dict[str, float]],
    output_path: str | Path,
    figsize: tuple[float, float] = (10, 5),
) -> None:
    """Plot grouped bar chart comparing speedup across methods and tasks.

    Args:
        results_dict: Nested dict {method_name: {task_name: speedup}}.
        output_path: Path to save the figure.
        figsize: Figure size in inches.
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    methods = list(results_dict.keys())
    tasks = list(next(iter(results_dict.values())).keys())
    n_methods = len(methods)
    n_tasks = len(tasks)

    x = np.arange(n_tasks)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        values = [results_dict[method].get(task, 0) for task in tasks]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Task")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Speedup Comparison: DAHD vs. Baselines")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend(frameon=True, loc="upper left", ncol=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_mode_switch_timeline(
    sequence_data: pd.DataFrame,
    output_path: str | Path,
    figsize: tuple[float, float] = (12, 3),
) -> None:
    """Plot mode switch timeline for a single inference sequence.

    Args:
        sequence_data: DataFrame with columns 'token_pos' and 'mode' ('parallel' or 'ar').
        output_path: Path to save the figure.
        figsize: Figure size in inches.
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    positions = sequence_data["token_pos"].values
    modes = sequence_data["mode"].values

    # Map modes to y-values
    mode_y = np.array([1 if m == "parallel" else 0 for m in modes])

    # Color segments
    for i in range(len(positions) - 1):
        color = COLORS["parallel"] if modes[i] == "parallel" else COLORS["ar"]
        ax.fill_between(
            [positions[i], positions[i + 1]],
            [mode_y[i], mode_y[i]],
            alpha=0.6, color=color, linewidth=0
        )

    # Mark switch points
    switch_points = []
    for i in range(1, len(modes)):
        if modes[i] != modes[i - 1]:
            switch_points.append(positions[i])
            ax.axvline(x=positions[i], color="black", linestyle=":", linewidth=0.5, alpha=0.5)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["AR", "Parallel"])
    ax.set_xlabel("Token Position")
    ax.set_title(f"Mode Switch Timeline ({len(switch_points)} switches)")
    ax.set_xlim(positions[0], positions[-1])

    # Add legend
    patches = [
        mpatches.Patch(color=COLORS["parallel"], alpha=0.6, label="Parallel"),
        mpatches.Patch(color=COLORS["ar"], alpha=0.6, label="AR"),
    ]
    ax.legend(handles=patches, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_speedup_vs_difficulty(
    bin_data: pd.DataFrame,
    output_path: str | Path,
    figsize: tuple[float, float] = (7, 5),
) -> None:
    """Plot speedup as a function of difficulty (probe confidence).

    Args:
        bin_data: DataFrame with columns 'bin_center', 'mean_speedup', 'std_speedup'.
        output_path: Path to save the figure.
        figsize: Figure size in inches.
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    x = bin_data["bin_center"].values
    y = bin_data["mean_speedup"].values
    yerr = bin_data["std_speedup"].values

    ax.plot(x, y, "-o", color=COLORS["dahd"], markersize=6, linewidth=2)
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=COLORS["dahd"])

    ax.set_xlabel("Average Probe Confidence (Difficulty)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Speedup vs. Task Difficulty")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Add annotation for key insight
    max_idx = np.argmax(y)
    ax.annotate(
        f"Peak: {y[max_idx]:.2f}×",
        xy=(x[max_idx], y[max_idx]),
        xytext=(x[max_idx] + 0.05, y[max_idx] + 0.1),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
