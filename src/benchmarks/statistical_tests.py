"""Statistical testing utilities for benchmark comparison.

Provides hypothesis tests for bimodal distribution detection,
method comparison with proper statistical rigor, and bootstrap
confidence intervals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


@dataclass
class ComparisonReport:
    """Report from comparing two benchmark methods statistically.

    Attributes:
        method_a: Name of first method.
        method_b: Name of second method.
        mean_a: Mean of method A measurements.
        mean_b: Mean of method B measurements.
        std_a: Standard deviation of method A.
        std_b: Standard deviation of method B.
        p_value: P-value from the chosen hypothesis test.
        significant: Whether the difference is statistically significant (p < 0.05).
        effect_size: Cohen's d effect size.
        test_used: Name of the statistical test used.
        ci_diff_95: 95% CI for the difference in means.
        normality_a: P-value from Shapiro-Wilk test on A.
        normality_b: P-value from Shapiro-Wilk test on B.
    """

    method_a: str
    method_b: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    p_value: float
    significant: bool
    effect_size: float
    test_used: str
    ci_diff_95: tuple[float, float] = (0.0, 0.0)
    normality_a: float = 0.0
    normality_b: float = 0.0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        sig_str = "YES" if self.significant else "NO"
        return (
            f"Comparison: {self.method_a} vs {self.method_b}\n"
            f"  Means: {self.mean_a:.3f} vs {self.mean_b:.3f}\n"
            f"  Test: {self.test_used}, p={self.p_value:.6f}\n"
            f"  Significant: {sig_str}, Effect size (Cohen's d): {self.effect_size:.3f}\n"
            f"  95% CI of difference: [{self.ci_diff_95[0]:.3f}, {self.ci_diff_95[1]:.3f}]"
        )


def test_bimodal_hypothesis(data: list[float]) -> dict:
    """Test whether data follows a bimodal distribution.

    Uses multiple approaches:
    1. Hartigan's dip test for unimodality
    2. KL divergence vs. best-fit unimodal Gaussian
    3. K-means (k=2) silhouette score improvement

    Args:
        data: List of scalar measurements to test.

    Returns:
        Dictionary with:
            - dip_statistic: Hartigan's dip test statistic
            - dip_p_value: P-value for the dip test
            - kl_divergence: KL divergence from unimodal Gaussian
            - silhouette_improvement: Improvement in silhouette from k=2 vs k=1
            - is_bimodal: Boolean verdict
    """
    arr = np.array(data, dtype=np.float64)
    n = len(arr)

    if n < 10:
        logger.warning("Too few samples for bimodality test")
        return {
            "dip_statistic": 0.0,
            "dip_p_value": 1.0,
            "kl_divergence": 0.0,
            "silhouette_improvement": 0.0,
            "is_bimodal": False,
        }

    # 1. Hartigan's Dip Test (manual implementation)
    dip_stat, dip_p = _hartigan_dip_test(arr)

    # 2. KL Divergence vs unimodal Gaussian
    kl_div = _kl_divergence_vs_gaussian(arr)

    # 3. Silhouette score improvement with k=2
    sil_improvement = _silhouette_improvement(arr)

    # Decision: bimodal if at least 2 of 3 criteria met
    criteria_met = 0
    if dip_p < 0.05:
        criteria_met += 1
    if kl_div > 0.1:
        criteria_met += 1
    if sil_improvement > 0.2:
        criteria_met += 1

    is_bimodal = criteria_met >= 2

    return {
        "dip_statistic": float(dip_stat),
        "dip_p_value": float(dip_p),
        "kl_divergence": float(kl_div),
        "silhouette_improvement": float(sil_improvement),
        "is_bimodal": is_bimodal,
    }


def _hartigan_dip_test(data: np.ndarray) -> tuple[float, float]:
    """Compute Hartigan's dip test statistic and p-value.

    Simplified implementation using the empirical CDF approach.

    Args:
        data: Sorted array of values.

    Returns:
        Tuple of (dip_statistic, p_value).
    """
    try:
        import diptest
        stat, p_value = diptest.diptest(data)
        return stat, p_value
    except ImportError:
        pass

    # Manual implementation: compute dip statistic
    sorted_data = np.sort(data)
    n = len(sorted_data)
    ecdf = np.arange(1, n + 1) / n

    # Compute the greatest convex minorant (GCM) and least concave majorant (LCM)
    # Simplified: use the maximum deviation from uniform CDF on [min, max]
    uniform_cdf = (sorted_data - sorted_data[0]) / (sorted_data[-1] - sorted_data[0] + 1e-10)

    # Dip = max deviation between ECDF and best-fit uniform
    dip_stat = np.max(np.abs(ecdf - uniform_cdf)) / 2.0

    # Approximate p-value using the asymptotic distribution
    # Under H0 (unimodal), sqrt(n) * dip ~ distribution
    # Use conservative approximation
    test_stat = np.sqrt(n) * dip_stat
    # Approximate p-value (conservative)
    p_value = np.exp(-2.0 * test_stat ** 2) if test_stat > 0 else 1.0
    p_value = min(p_value, 1.0)

    return float(dip_stat), float(p_value)


def _kl_divergence_vs_gaussian(data: np.ndarray) -> float:
    """Compute KL divergence between data histogram and best-fit Gaussian.

    Args:
        data: Array of values.

    Returns:
        KL divergence value.
    """
    # Fit Gaussian
    mu, sigma = np.mean(data), np.std(data)
    if sigma < 1e-10:
        return 0.0

    # Create histogram
    n_bins = max(10, int(np.sqrt(len(data))))
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1] - bin_edges[0]

    # Gaussian PDF at bin centers
    gaussian_pdf = stats.norm.pdf(bin_centers, mu, sigma)

    # Avoid zeros
    eps = 1e-10
    hist_norm = hist + eps
    gaussian_norm = gaussian_pdf + eps

    # Normalize to proper probability distributions
    hist_norm = hist_norm / (hist_norm.sum() * bin_width)
    gaussian_norm = gaussian_norm / (gaussian_norm.sum() * bin_width)

    # KL divergence: sum p(x) * log(p(x) / q(x))
    kl = np.sum(hist_norm * np.log(hist_norm / gaussian_norm)) * bin_width
    return max(0.0, float(kl))


def _silhouette_improvement(data: np.ndarray) -> float:
    """Compute silhouette score improvement from k=1 to k=2 clustering.

    Args:
        data: Array of values.

    Returns:
        Silhouette score for k=2 (higher = more bimodal).
    """
    X = data.reshape(-1, 1)

    if len(X) < 4:
        return 0.0

    # k=2 clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Check if both clusters have points
    if len(set(labels)) < 2:
        return 0.0

    sil_score = silhouette_score(X, labels)
    return float(sil_score)


def compare_two_methods(
    a: list[float],
    b: list[float],
    method_a_name: str = "A",
    method_b_name: str = "B",
    alpha: float = 0.05,
) -> ComparisonReport:
    """Compare two methods using appropriate statistical tests.

    Automatically selects between parametric (t-test) and non-parametric
    (Mann-Whitney U) tests based on normality of the data.

    Args:
        a: Measurements for method A.
        b: Measurements for method B.
        method_a_name: Label for method A.
        method_b_name: Label for method B.
        alpha: Significance level.

    Returns:
        ComparisonReport with full statistical analysis.
    """
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)

    mean_a, mean_b = float(np.mean(arr_a)), float(np.mean(arr_b))
    std_a, std_b = float(np.std(arr_a, ddof=1)), float(np.std(arr_b, ddof=1))

    # Normality tests (Shapiro-Wilk)
    norm_a_p = 1.0
    norm_b_p = 1.0
    if len(arr_a) >= 3:
        _, norm_a_p = stats.shapiro(arr_a)
    if len(arr_b) >= 3:
        _, norm_b_p = stats.shapiro(arr_b)

    # Choose test based on normality
    both_normal = norm_a_p > alpha and norm_b_p > alpha

    if both_normal:
        # Welch's t-test (doesn't assume equal variance)
        t_stat, p_value = stats.ttest_ind(arr_a, arr_b, equal_var=False)
        test_used = "Welch's t-test"
    else:
        # Mann-Whitney U test (non-parametric)
        u_stat, p_value = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
        test_used = "Mann-Whitney U"

    significant = p_value < alpha

    # Cohen's d effect size
    pooled_std = np.sqrt((std_a ** 2 + std_b ** 2) / 2.0)
    effect_size = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

    # 95% CI for the difference in means (using bootstrap)
    ci_diff = _bootstrap_ci_diff(arr_a, arr_b, confidence=0.95)

    return ComparisonReport(
        method_a=method_a_name,
        method_b=method_b_name,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        p_value=float(p_value),
        significant=significant,
        effect_size=float(effect_size),
        test_used=test_used,
        ci_diff_95=ci_diff,
        normality_a=float(norm_a_p),
        normality_b=float(norm_b_p),
    )


def _bootstrap_ci_diff(
    a: np.ndarray, b: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 1000
) -> tuple[float, float]:
    """Bootstrap confidence interval for the difference in means."""
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_bootstrap):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(sample_a) - np.mean(sample_b))

    diffs = np.array(diffs)
    alpha = 1 - confidence
    lower = float(np.percentile(diffs, 100 * alpha / 2))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return (lower, upper)


def compute_confidence_interval(
    data: list[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Compute confidence interval for the mean using t-distribution.

    Args:
        data: List of measurements.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    arr = np.array(data, dtype=np.float64)
    n = len(arr)

    if n < 2:
        mean = float(np.mean(arr)) if n > 0 else 0.0
        return (mean, mean)

    mean = float(np.mean(arr))
    se = float(stats.sem(arr))
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    margin = t_crit * se
    return (mean - margin, mean + margin)


def bootstrap_speedup(
    method_latencies: list[float],
    baseline_latencies: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for speedup ratio.

    Speedup is defined as baseline_mean / method_mean.

    Args:
        method_latencies: Latency measurements for the proposed method.
        baseline_latencies: Latency measurements for the baseline (AR).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.

    Returns:
        Tuple of (mean_speedup, ci_lower, ci_upper).
    """
    method_arr = np.array(method_latencies, dtype=np.float64)
    baseline_arr = np.array(baseline_latencies, dtype=np.float64)

    rng = np.random.default_rng(42)
    speedups = []

    for _ in range(n_bootstrap):
        method_sample = rng.choice(method_arr, size=len(method_arr), replace=True)
        baseline_sample = rng.choice(baseline_arr, size=len(baseline_arr), replace=True)

        method_mean = np.mean(method_sample)
        baseline_mean = np.mean(baseline_sample)

        if method_mean > 0:
            speedups.append(baseline_mean / method_mean)

    if not speedups:
        return (0.0, 0.0, 0.0)

    speedups = np.array(speedups)
    mean_speedup = float(np.mean(speedups))

    alpha = 1 - confidence
    ci_lower = float(np.percentile(speedups, 100 * alpha / 2))
    ci_upper = float(np.percentile(speedups, 100 * (1 - alpha / 2)))

    return (mean_speedup, ci_lower, ci_upper)
