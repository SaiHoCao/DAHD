"""Statistical analysis utilities for DAHD speculative decoding.

Provides functions for computing acceptance rates, difficulty distributions,
mode switch frequencies, correlations, and speedup analyses.
"""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def compute_per_position_acceptance(token_data: list[dict]) -> dict[int, float]:
    """Compute average acceptance rate at each token position.

    Args:
        token_data: List of dicts, each with keys 'token_pos' and 'accepted' (bool/int).

    Returns:
        Dictionary mapping token position -> mean acceptance rate.
    """
    position_groups: dict[int, list[float]] = defaultdict(list)
    for entry in token_data:
        pos = int(entry["token_pos"])
        accepted = float(entry["accepted"])
        position_groups[pos].append(accepted)

    result: dict[int, float] = {}
    for pos in sorted(position_groups.keys()):
        values = position_groups[pos]
        result[pos] = float(np.mean(values))

    return result


def compute_difficulty_distribution(token_data: list[dict]) -> dict[str, Any]:
    """Compute statistical summary of token difficulty distribution.

    Args:
        token_data: List of dicts, each with key 'probe_confidence' (float in [0, 1]).

    Returns:
        Dictionary with keys: mean, std, median, q25, q75, histogram_bins, histogram_counts.
    """
    confidences = np.array([entry["probe_confidence"] for entry in token_data])

    if len(confidences) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "histogram_bins": [],
            "histogram_counts": [],
        }

    counts, bin_edges = np.histogram(confidences, bins=50, range=(0.0, 1.0))

    return {
        "mean": float(np.mean(confidences)),
        "std": float(np.std(confidences)),
        "median": float(np.median(confidences)),
        "q25": float(np.percentile(confidences, 25)),
        "q75": float(np.percentile(confidences, 75)),
        "histogram_bins": bin_edges.tolist(),
        "histogram_counts": counts.tolist(),
    }


def compute_mode_switch_frequency(sequence_data: pd.DataFrame) -> dict[str, float]:
    """Compute mode switch frequency statistics across sequences.

    Args:
        sequence_data: DataFrame with columns 'sequence_id' and 'mode_switches'
                       (list or count of switches per sequence), and 'seq_length'.

    Returns:
        Dictionary with keys:
            - avg_switches_per_seq: Mean number of switches per sequence
            - switch_density: Mean switches per token (switches / seq_length)
            - std_switches: Standard deviation of switch counts
            - max_switches: Maximum switches in any sequence
    """
    if "num_switches" in sequence_data.columns:
        switch_counts = sequence_data["num_switches"].values
    elif "mode_switches" in sequence_data.columns:
        switch_counts = sequence_data["mode_switches"].apply(
            lambda x: len(x) if isinstance(x, list) else int(x)
        ).values
    else:
        raise ValueError("DataFrame must have 'num_switches' or 'mode_switches' column")

    seq_lengths = sequence_data["seq_length"].values if "seq_length" in sequence_data.columns else None

    avg_switches = float(np.mean(switch_counts))
    std_switches = float(np.std(switch_counts))
    max_switches = int(np.max(switch_counts))

    if seq_lengths is not None:
        densities = switch_counts / np.maximum(seq_lengths, 1)
        switch_density = float(np.mean(densities))
    else:
        switch_density = 0.0

    return {
        "avg_switches_per_seq": avg_switches,
        "switch_density": switch_density,
        "std_switches": std_switches,
        "max_switches": max_switches,
    }


def compute_correlation(x: list[float], y: list[float]) -> dict[str, float]:
    """Compute Pearson and Spearman correlations between two variables.

    Args:
        x: First variable values.
        y: Second variable values.

    Returns:
        Dictionary with keys: pearson_r, pearson_p, spearman_rho, spearman_p.
    """
    x_arr = np.array(x, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)

    if len(x_arr) < 3 or len(y_arr) < 3:
        return {
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
        }

    # Remove NaN pairs
    valid_mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_clean = x_arr[valid_mask]
    y_clean = y_arr[valid_mask]

    if len(x_clean) < 3:
        return {
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
        }

    pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
    spearman_rho, spearman_p = stats.spearmanr(x_clean, y_clean)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def compute_speedup_by_difficulty_bin(
    sequence_data: pd.DataFrame, num_bins: int = 10
) -> pd.DataFrame:
    """Compute speedup statistics grouped by difficulty bins.

    Args:
        sequence_data: DataFrame with columns 'avg_probe_confidence' and 'speedup'.
        num_bins: Number of bins to divide the confidence range into.

    Returns:
        DataFrame with columns: bin_center, bin_low, bin_high,
        mean_speedup, std_speedup, count, mean_confidence.
    """
    df = sequence_data.copy()

    # Create difficulty bins based on avg_probe_confidence
    df["difficulty_bin"] = pd.cut(
        df["avg_probe_confidence"], bins=num_bins, labels=False
    )

    bin_edges = np.linspace(
        df["avg_probe_confidence"].min(),
        df["avg_probe_confidence"].max(),
        num_bins + 1,
    )

    results = []
    for i in range(num_bins):
        bin_mask = df["difficulty_bin"] == i
        bin_data = df[bin_mask]

        if len(bin_data) == 0:
            continue

        results.append({
            "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
            "bin_low": float(bin_edges[i]),
            "bin_high": float(bin_edges[i + 1]),
            "mean_speedup": float(bin_data["speedup"].mean()),
            "std_speedup": float(bin_data["speedup"].std()),
            "count": int(len(bin_data)),
            "mean_confidence": float(bin_data["avg_probe_confidence"].mean()),
        })

    return pd.DataFrame(results)
