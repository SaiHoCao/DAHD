"""Analysis module for DAHD speculative decoding experiments.

Provides statistical analysis, visualization, and pipeline orchestration
for evaluating difficulty-aware hybrid draft models.
"""

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
from src.analysis.pipeline import AnalysisPipeline
from src.analysis.paper_figures import PaperFigureGenerator

__all__ = [
    "compute_per_position_acceptance",
    "compute_difficulty_distribution",
    "compute_mode_switch_frequency",
    "compute_correlation",
    "compute_speedup_by_difficulty_bin",
    "plot_acceptance_histogram",
    "plot_violin_acceptance_by_task",
    "plot_scatter_confidence_vs_acceptance",
    "plot_speedup_comparison_bar",
    "plot_mode_switch_timeline",
    "plot_speedup_vs_difficulty",
    "set_paper_style",
    "AnalysisPipeline",
    "PaperFigureGenerator",
]
