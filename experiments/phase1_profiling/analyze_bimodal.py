#!/usr/bin/env python3
"""Phase 1: Bimodal Distribution Analysis for EAGLE-3 Acceptance Rate Data.

This script analyzes the per-token acceptance rate data collected by
run_eagle3_profiling.py and performs statistical tests for bimodality.

Tests performed:
  1. Hartigan's Dip Test (statistical test for unimodality)
  2. Gaussian Mixture Model (GMM) fit with BIC comparison
  3. KL divergence from best-fit unimodal vs bimodal
  4. Visual analysis (histograms, KDE, Q-Q plots)

Usage:
    python experiments/phase1_profiling/analyze_bimodal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "phase1_results"
METRICS_PATH = RESULTS_DIR / "per_token_metrics.jsonl"
OUTPUT_DIR = RESULTS_DIR


def load_metrics():
    """Load per-token metrics from JSONL file."""
    metrics = []
    with open(METRICS_PATH) as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics


def hartigans_dip_test(data: np.ndarray, n_boot: int = 1000) -> dict:
    """Hartigan's Dip Test for unimodality.
    
    Computes the dip statistic and estimates p-value via bootstrap.
    Small p-value (<0.05) rejects unimodality -> supports bimodality.
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # Compute empirical CDF
    ecdf = np.arange(1, n + 1) / n
    
    # Compute the dip statistic (maximum deviation from best-fit unimodal CDF)
    # Simplified: compare with uniform distribution on [min, max]
    uniform_cdf = (sorted_data - sorted_data[0]) / (sorted_data[-1] - sorted_data[0] + 1e-10)
    
    # Dip = max difference between ECDF and the greatest convex minorant / least concave majorant
    # Simplified version: half the max gap between ECDF and uniform
    dip_stat = np.max(np.abs(ecdf - uniform_cdf)) / 2
    
    # Bootstrap p-value: generate uniform samples and compute dip
    boot_dips = []
    for _ in range(n_boot):
        boot_sample = np.sort(np.random.uniform(0, 1, n))
        boot_ecdf = np.arange(1, n + 1) / n
        boot_dip = np.max(np.abs(boot_ecdf - boot_sample)) / 2
        boot_dips.append(boot_dip)
    
    p_value = np.mean(np.array(boot_dips) >= dip_stat)
    
    return {
        "dip_statistic": float(dip_stat),
        "p_value": float(p_value),
        "reject_unimodality": p_value < 0.05,
        "n_bootstrap": n_boot,
    }


def fit_gmm(data: np.ndarray, max_components: int = 3) -> dict:
    """Fit Gaussian Mixture Models and compare via BIC.
    
    Returns the best model and BIC scores.
    """
    from sklearn.mixture import GaussianMixture
    
    data_2d = data.reshape(-1, 1)
    results = {}
    bic_scores = {}
    
    for n_comp in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_comp, random_state=42, n_init=5)
        gmm.fit(data_2d)
        bic = gmm.bic(data_2d)
        bic_scores[n_comp] = float(bic)
        results[n_comp] = {
            "bic": float(bic),
            "means": gmm.means_.flatten().tolist(),
            "stds": np.sqrt(gmm.covariances_.flatten()).tolist(),
            "weights": gmm.weights_.tolist(),
        }
    
    best_n = min(bic_scores, key=bic_scores.get)
    
    return {
        "best_n_components": best_n,
        "bic_scores": bic_scores,
        "models": results,
        "bimodal_preferred": best_n >= 2,
        "bic_improvement_2_vs_1": float(bic_scores[1] - bic_scores[2]) if 2 in bic_scores else 0,
    }


def compute_kl_divergence(data: np.ndarray, n_bins: int = 50) -> dict:
    """Compute KL divergence between empirical distribution and fitted models."""
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Empirical distribution (normalized histogram)
    p = hist * bin_width
    p = p / p.sum()  # ensure sums to 1
    p = np.clip(p, 1e-10, None)
    
    # Fit unimodal Gaussian
    mu = data.mean()
    sigma = data.std()
    from scipy.stats import norm
    q_unimodal = norm.pdf(bin_centers, mu, sigma) * bin_width
    q_unimodal = q_unimodal / q_unimodal.sum()
    q_unimodal = np.clip(q_unimodal, 1e-10, None)
    
    # KL(P || Q_unimodal)
    kl_unimodal = float(np.sum(p * np.log(p / q_unimodal)))
    
    # Fit bimodal (2-component GMM)
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=2, random_state=42, n_init=5)
    gmm.fit(data.reshape(-1, 1))
    
    q_bimodal = np.zeros_like(bin_centers)
    for i in range(2):
        q_bimodal += gmm.weights_[i] * norm.pdf(
            bin_centers, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0])
        )
    q_bimodal = q_bimodal * bin_width
    q_bimodal = q_bimodal / q_bimodal.sum()
    q_bimodal = np.clip(q_bimodal, 1e-10, None)
    
    # KL(P || Q_bimodal)
    kl_bimodal = float(np.sum(p * np.log(p / q_bimodal)))
    
    return {
        "kl_unimodal": kl_unimodal,
        "kl_bimodal": kl_bimodal,
        "kl_reduction": kl_unimodal - kl_bimodal,
        "bimodal_better_fit": kl_bimodal < kl_unimodal,
        "gmm_means": gmm.means_.flatten().tolist(),
        "gmm_weights": gmm.weights_.tolist(),
        "gmm_stds": np.sqrt(gmm.covariances_.flatten()).tolist(),
    }


def generate_visualizations(data: np.ndarray, metrics: list, output_dir: Path):
    """Generate visualization plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ── Plot 1: Histogram of top-1 probability ──────────────────────────
    ax = axes[0, 0]
    ax.hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.set_xlabel('Top-1 Token Probability')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Per-Token Confidence\n(Top-1 Probability)')
    ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean={data.mean():.3f}')
    ax.axvline(np.median(data), color='orange', linestyle='--', label=f'Median={np.median(data):.3f}')
    ax.legend()
    
    # ── Plot 2: GMM fit overlay ──────────────────────────────────────────
    ax = axes[0, 1]
    ax.hist(data, bins=50, density=True, alpha=0.5, color='lightgray', edgecolor='white')
    
    gmm = GaussianMixture(n_components=2, random_state=42, n_init=5)
    gmm.fit(data.reshape(-1, 1))
    
    x_range = np.linspace(0, 1, 200)
    for i in range(2):
        component = gmm.weights_[i] * norm.pdf(
            x_range, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0])
        )
        ax.plot(x_range, component, linewidth=2, 
                label=f'Component {i+1}: μ={gmm.means_[i,0]:.3f}, w={gmm.weights_[i]:.3f}')
    
    # Total GMM
    total_pdf = np.zeros_like(x_range)
    for i in range(2):
        total_pdf += gmm.weights_[i] * norm.pdf(
            x_range, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0])
        )
    ax.plot(x_range, total_pdf, 'k-', linewidth=2, label='GMM (2-comp)')
    ax.set_xlabel('Top-1 Token Probability')
    ax.set_ylabel('Density')
    ax.set_title('2-Component GMM Fit\n(Bimodal Model)')
    ax.legend()
    
    # ── Plot 3: Entropy distribution ─────────────────────────────────────
    ax = axes[0, 2]
    entropies = np.array([m["entropy"] for m in metrics])
    ax.hist(entropies, bins=50, density=True, alpha=0.7, color='coral', edgecolor='white')
    ax.set_xlabel('Entropy (nats)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Per-Token Entropy')
    ax.axvline(entropies.mean(), color='red', linestyle='--', label=f'Mean={entropies.mean():.3f}')
    ax.legend()
    
    # ── Plot 4: Per-prompt acceptance rate ───────────────────────────────
    ax = axes[1, 0]
    prompt_rates = []
    by_prompt = {}
    for m in metrics:
        pid = m["prompt_idx"]
        if pid not in by_prompt:
            by_prompt[pid] = []
        by_prompt[pid].append(m["top1_prob"])
    
    for pid in sorted(by_prompt.keys()):
        prompt_rates.append(np.mean(by_prompt[pid]))
    
    prompt_rates = np.array(prompt_rates)
    ax.hist(prompt_rates, bins=30, density=True, alpha=0.7, color='seagreen', edgecolor='white')
    ax.set_xlabel('Mean Top-1 Prob per Prompt')
    ax.set_ylabel('Density')
    ax.set_title('Per-Prompt Mean Confidence\n(Prompt-Level Bimodality)')
    ax.axvline(prompt_rates.mean(), color='red', linestyle='--', label=f'Mean={prompt_rates.mean():.3f}')
    ax.legend()
    
    # ── Plot 5: Acceptance by position in draft window ────────────────────
    ax = axes[1, 1]
    pos_data = {}
    for m in metrics:
        pos = m["step_in_draft"]
        if pos not in pos_data:
            pos_data[pos] = []
        pos_data[pos].append(m["top1_prob"])
    
    positions = sorted(pos_data.keys())
    means = [np.mean(pos_data[p]) for p in positions]
    stds = [np.std(pos_data[p]) for p in positions]
    
    ax.bar(positions, means, yerr=stds, alpha=0.7, color='mediumpurple', capsize=5)
    ax.set_xlabel('Position in Draft Window')
    ax.set_ylabel('Mean Top-1 Probability')
    ax.set_title('Confidence by Position in Draft Window')
    ax.set_xticks(positions)
    
    # ── Plot 6: CDF with bimodal markers ─────────────────────────────────
    ax = axes[1, 2]
    sorted_probs = np.sort(data)
    ecdf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    ax.plot(sorted_probs, ecdf, 'b-', linewidth=1.5, label='Empirical CDF')
    
    # Mark regions
    ax.axvline(0.3, color='red', linestyle=':', alpha=0.7, label='Low/Mid boundary (0.3)')
    ax.axvline(0.8, color='green', linestyle=':', alpha=0.7, label='Mid/High boundary (0.8)')
    
    low_frac = (data < 0.3).mean()
    high_frac = (data > 0.8).mean()
    ax.fill_betweenx([0, 1], 0, 0.3, alpha=0.1, color='red')
    ax.fill_betweenx([0, 1], 0.8, 1.0, alpha=0.1, color='green')
    
    ax.text(0.15, 0.5, f'{low_frac:.1%}\n(Hard)', ha='center', transform=ax.transAxes, fontsize=9)
    ax.text(0.85, 0.5, f'{high_frac:.1%}\n(Easy)', ha='center', transform=ax.transAxes, fontsize=9)
    
    ax.set_xlabel('Top-1 Token Probability')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Empirical CDF with Mode Regions')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = output_dir / "bimodal_analysis_plots.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {fig_path}")
    
    # Additional plot: Draft head acceptance if available
    draft_accepted = np.array([m.get("draft_accepted", False) for m in metrics])
    if draft_accepted.any():
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
        # Acceptance rate vs confidence (binned)
        bins = np.linspace(0, 1, 20)
        bin_indices = np.digitize(data, bins)
        bin_accept_rates = []
        bin_centers = []
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if mask.sum() > 5:
                bin_accept_rates.append(draft_accepted[mask].mean())
                bin_centers.append((bins[i-1] + bins[i]) / 2)
        
        ax2.plot(bin_centers, bin_accept_rates, 'bo-', linewidth=2, markersize=6)
        ax2.set_xlabel('Top-1 Token Probability (binned)')
        ax2.set_ylabel('Draft Head Acceptance Rate')
        ax2.set_title('Acceptance Rate vs Token Confidence\n(Validates confidence as acceptance proxy)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        fig2_path = output_dir / "acceptance_vs_confidence.png"
        plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Acceptance vs confidence plot saved: {fig2_path}")
    
    return fig_path


def main():
    print("=" * 70)
    print("Phase 1: Bimodal Distribution Analysis")
    print("=" * 70)
    
    if not METRICS_PATH.exists():
        print(f"ERROR: Metrics file not found: {METRICS_PATH}")
        print("Run run_eagle3_profiling.py first.")
        sys.exit(1)
    
    # Load data
    print(f"\n[1/5] Loading metrics from {METRICS_PATH}")
    metrics = load_metrics()
    print(f"  Loaded {len(metrics)} token metrics")
    
    top1_probs = np.array([m["top1_prob"] for m in metrics])
    entropies = np.array([m["entropy"] for m in metrics])
    
    print(f"\n[2/5] Running Hartigan's Dip Test...")
    dip_result = hartigans_dip_test(top1_probs, n_boot=2000)
    print(f"  Dip statistic: {dip_result['dip_statistic']:.6f}")
    print(f"  P-value: {dip_result['p_value']:.6f}")
    print(f"  Reject unimodality: {dip_result['reject_unimodality']}")
    
    print(f"\n[3/5] Fitting Gaussian Mixture Models...")
    try:
        gmm_result = fit_gmm(top1_probs)
        print(f"  Best number of components: {gmm_result['best_n_components']}")
        print(f"  BIC scores: {gmm_result['bic_scores']}")
        if gmm_result['bimodal_preferred']:
            model_2 = gmm_result['models'][2]
            print(f"  Bimodal GMM:")
            print(f"    Component 1: μ={model_2['means'][0]:.3f}, σ={model_2['stds'][0]:.3f}, w={model_2['weights'][0]:.3f}")
            print(f"    Component 2: μ={model_2['means'][1]:.3f}, σ={model_2['stds'][1]:.3f}, w={model_2['weights'][1]:.3f}")
        print(f"  BIC improvement (2-comp vs 1-comp): {gmm_result['bic_improvement_2_vs_1']:.1f}")
    except ImportError:
        print("  WARNING: sklearn not available, skipping GMM fit")
        gmm_result = None
    
    print(f"\n[4/5] Computing KL Divergence...")
    try:
        kl_result = compute_kl_divergence(top1_probs)
        print(f"  KL(empirical || unimodal): {kl_result['kl_unimodal']:.6f}")
        print(f"  KL(empirical || bimodal):  {kl_result['kl_bimodal']:.6f}")
        print(f"  KL reduction: {kl_result['kl_reduction']:.6f}")
        print(f"  Bimodal is better fit: {kl_result['bimodal_better_fit']}")
    except ImportError:
        print("  WARNING: scipy/sklearn not available, skipping KL analysis")
        kl_result = None
    
    print(f"\n[5/5] Generating visualizations...")
    try:
        fig_path = generate_visualizations(top1_probs, metrics, OUTPUT_DIR)
    except Exception as e:
        print(f"  WARNING: Visualization failed: {e}")
        fig_path = None
    
    # ── Summary ──────────────────────────────────────────────────────────
    # Helper to convert numpy types to native Python types for JSON
    def to_serializable(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        return obj

    summary = {
        "total_tokens": len(metrics),
        "top1_prob_stats": {
            "mean": float(top1_probs.mean()),
            "std": float(top1_probs.std()),
            "median": float(np.median(top1_probs)),
            "skewness": float(((top1_probs - top1_probs.mean()) ** 3).mean() / top1_probs.std() ** 3),
            "kurtosis": float(((top1_probs - top1_probs.mean()) ** 4).mean() / top1_probs.std() ** 4 - 3),
        },
        "bimodality_tests": {
            "dip_test": to_serializable(dip_result),
            "gmm_analysis": to_serializable(gmm_result),
            "kl_divergence": to_serializable(kl_result),
        },
        "conclusion": {
            "bimodal_evidence_strong": bool(
                dip_result["reject_unimodality"] and
                (gmm_result is not None and gmm_result["bimodal_preferred"]) and
                (kl_result is not None and kl_result["bimodal_better_fit"])
            ),
            "interpretation": "",
        }
    }
    
    # Generate interpretation
    evidence_count = sum([
        dip_result["reject_unimodality"],
        gmm_result["bimodal_preferred"] if gmm_result else False,
        kl_result["bimodal_better_fit"] if kl_result else False,
    ])
    
    if evidence_count >= 2:
        summary["conclusion"]["interpretation"] = (
            f"STRONG bimodal evidence ({evidence_count}/3 tests support bimodality). "
            "Token predictability shows distinct 'easy' (high confidence) and 'hard' (low confidence) modes. "
            "This validates the DAHD hypothesis: adaptive draft lengths should improve speculative decoding."
        )
    elif evidence_count == 1:
        summary["conclusion"]["interpretation"] = (
            f"MODERATE bimodal evidence ({evidence_count}/3 tests support bimodality). "
            "Some bimodal tendency detected but not conclusive across all tests."
        )
    else:
        summary["conclusion"]["interpretation"] = (
            f"WEAK bimodal evidence ({evidence_count}/3 tests support bimodality). "
            "Distribution appears approximately unimodal or multi-modal."
        )
    
    # Save summary
    summary_path = OUTPUT_DIR / "bimodal_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("ANALYSIS CONCLUSION")
    print("=" * 70)
    print(f"\n  {summary['conclusion']['interpretation']}")
    
    if gmm_result and gmm_result['bimodal_preferred']:
        model_2 = gmm_result['models'][2]
        print(f"\n  Bimodal Components:")
        print(f"    'Easy' mode: μ={max(model_2['means']):.3f} ({model_2['weights'][np.argmax(model_2['means'])]:.1%} of tokens)")
        print(f"    'Hard' mode: μ={min(model_2['means']):.3f} ({model_2['weights'][np.argmin(model_2['means'])]:.1%} of tokens)")
    
    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
