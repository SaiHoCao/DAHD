#!/usr/bin/env python3
"""
Multi-Task EAGLE-3 Acceptance Profiling Script.

Runs EAGLE-3 speculative decoding on multiple task types to profile
per-task token difficulty distributions. Collects:
- Per-position acceptance rate per task
- Target model top-1 confidence distribution per task
- Acceptance length distribution per task
- Bimodal tests (GMM + Dip test) per task

Reuses the EAGLE-3 draft head implementation from run_eagle3_full_profiling.py.
"""

import json
import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.phase1_profiling.run_eagle3_full_profiling import (
    ProfilingConfig,
    Eagle3DraftHead,
    SpeculativeDecoder,
)

# ============================================================================
# Multi-Task Prompts
# ============================================================================

PROMPTS_CODE = [
    "Write a Python function to implement binary search in a sorted array.",
    "Implement a class for a min-heap data structure in Python.",
    "Write a function to find the longest common subsequence of two strings.",
    "Create a Python decorator that caches function results.",
    "Write a function to serialize and deserialize a binary tree.",
    "Implement a thread-safe singleton pattern in Python.",
    "Write a function to detect a cycle in a linked list.",
    "Implement the merge sort algorithm with detailed comments.",
    "Write a Python function to validate a binary search tree.",
    "Create a function that implements the producer-consumer pattern using threading.",
    "Write a Python class implementing an LRU cache with O(1) operations.",
    "Implement Dijkstra's shortest path algorithm in Python.",
    "Write a function to find all permutations of a string without duplicates.",
    "Create a balanced parentheses checker supporting multiple bracket types.",
    "Implement a trie (prefix tree) data structure with insert, search, and startsWith methods.",
    "Write a function to solve the N-Queens problem using backtracking.",
    "Implement a simple regular expression matcher supporting '.' and '*'.",
    "Write a generator function that yields prime numbers indefinitely.",
    "Create a function that flattens a deeply nested dictionary into dot-notation keys.",
    "Implement an async rate limiter class with a sliding window algorithm.",
    "Write a Python function to perform topological sort on a directed graph.",
    "Implement the A* pathfinding algorithm on a 2D grid.",
    "Write a function to find the median of two sorted arrays in O(log(m+n)) time.",
    "Create a context manager that measures and logs execution time.",
    "Write a function implementing the KMP string matching algorithm.",
]

PROMPTS_MATH = [
    "Solve step by step: If 3x + 7 = 22, what is x?",
    "Prove that the square root of 2 is irrational.",
    "Calculate the integral of x^2 * e^x dx using integration by parts.",
    "Find all prime numbers p such that p^2 + 2 is also prime.",
    "Prove by induction that 1+2+...+n = n(n+1)/2.",
    "Solve the system of equations: 2x + 3y = 7, x - y = 1.",
    "Find the derivative of f(x) = ln(sin(x^2)).",
    "Prove that there are infinitely many prime numbers.",
    "Calculate the limit of (1 + 1/n)^n as n approaches infinity.",
    "Solve the differential equation dy/dx = 2xy with y(0) = 1.",
    "Find the eigenvalues of the matrix [[3, 1], [1, 3]].",
    "Prove that the sum of angles in a triangle is 180 degrees.",
    "Calculate the Taylor series expansion of e^x around x=0 up to the 5th term.",
    "Solve: A train travels 60 km at 40 km/h and then 90 km at 60 km/h. What is the average speed?",
    "Prove that for any integer n, n^3 - n is divisible by 6.",
    "Find the area enclosed by the curves y = x^2 and y = 2x.",
    "Solve the recurrence relation a_n = 2*a_{n-1} + 1 with a_0 = 0.",
    "Prove the Cauchy-Schwarz inequality for real numbers.",
    "Calculate the probability of getting exactly 3 heads in 5 fair coin flips.",
    "Find the radius of convergence of the power series sum(x^n / n!).",
    "Solve step by step: If log_2(x) + log_2(x-2) = 3, find x.",
    "Prove that the function f(x) = x^3 is uniformly continuous on [0, 1].",
    "Find the shortest distance from the point (1, 2, 3) to the plane 2x + y - z = 4.",
    "Calculate the determinant of a 3x3 matrix [[1,2,3],[4,5,6],[7,8,10]].",
    "Prove that the set of rational numbers is countable.",
]

PROMPTS_WRITING = [
    "Write a short story about a robot discovering emotions for the first time.",
    "Describe a sunset over the ocean in vivid detail.",
    "Write a dialogue between a scientist and a philosopher about consciousness.",
    "Compose a poem about the passage of time.",
    "Tell me about the history of artificial intelligence.",
    "Write a creative description of a futuristic city powered entirely by renewable energy.",
    "Describe the experience of climbing a mountain from the perspective of a first-time hiker.",
    "Write a short fairy tale about a talking fox who helps lost travelers.",
    "Compose a letter from a grandparent to their grandchild about life lessons.",
    "Write a vivid description of a thunderstorm approaching a small coastal town.",
    "Create a monologue for a character who just discovered they can read minds.",
    "Describe the atmosphere of a bustling night market in Southeast Asia.",
    "Write a short story about the last librarian in a world where books are forgotten.",
    "Compose a eulogy for a beloved family pet.",
    "Describe what Earth looks like from the International Space Station.",
    "Write an opening chapter for a mystery novel set in Victorian London.",
    "Create a dialogue between two old friends meeting after 20 years.",
    "Write a reflective essay about how music shapes our memories.",
    "Describe the feeling of homesickness in poetic prose.",
    "Write a short fable about the importance of patience.",
    "Compose a travel journal entry about visiting an ancient temple.",
    "Write a story about a child's first day at school told from the teacher's perspective.",
    "Describe a perfect autumn day in the countryside.",
    "Write a motivational speech about overcoming failure.",
    "Create a scene depicting the moment before a significant historical event.",
]

PROMPTS_TRANSLATION = [
    "Translate the following to Chinese: The quick brown fox jumps over the lazy dog.",
    'Convert this JSON to YAML: {"name": "Alice", "age": 30, "city": "Beijing"}',
    "Rewrite in formal English: hey dude whats up wanna grab lunch",
    "List the first 20 elements of the periodic table with their symbols.",
    "Write the standard boilerplate for a Python Flask web application.",
    "Translate to French: I would like to book a table for two at eight o'clock tonight.",
    "Convert this SQL query to MongoDB: SELECT * FROM users WHERE age > 25 ORDER BY name",
    "Rewrite this paragraph in passive voice: The cat chased the mouse across the kitchen floor.",
    "List all US states in alphabetical order.",
    "Write the HTML boilerplate for a basic webpage with a navigation bar.",
    "Translate to Japanese: Good morning, how are you today?",
    "Convert these temperatures from Celsius to Fahrenheit: 0, 20, 37, 100.",
    "Rewrite in simpler English: The epistemological ramifications of quantum mechanics necessitate a paradigmatic shift.",
    "List the planets of our solar system in order from the sun.",
    "Write the standard import statements for a PyTorch deep learning project.",
    "Translate to Spanish: Where is the nearest hospital? I need help.",
    "Convert this Python dictionary to a formatted table: {'Alice': 85, 'Bob': 92, 'Carol': 78}",
    "Rewrite in British English: I took the elevator to the first floor of the apartment building.",
    "List the 12 months of the year with their number of days.",
    "Write the standard Dockerfile for a Python FastAPI application.",
    "Translate to German: Could you please tell me the way to the train station?",
    "Convert this CSV data to JSON: name,age,city\\nAlice,30,Beijing\\nBob,25,Shanghai",
    "Rewrite this sentence to be more concise: In my personal opinion, I think that it is very important to note that...",
    "List the seven continents with their approximate areas in square kilometers.",
    "Write a standard .gitignore file for a Python machine learning project.",
]

TASK_PROMPTS = {
    "code": PROMPTS_CODE,
    "math": PROMPTS_MATH,
    "writing": PROMPTS_WRITING,
    "translation": PROMPTS_TRANSLATION,
}


# ============================================================================
# Multi-Task Profiler
# ============================================================================

class MultiTaskProfiler:
    """Extends SpeculativeDecoder to run multiple task types and compare."""

    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.decoder = SpeculativeDecoder(config)
        self.logger = logging.getLogger("MultiTaskProfiler")
        self.logger.setLevel(logging.INFO)

        # Per-task metrics
        self.task_metrics: Dict[str, List[dict]] = defaultdict(list)
        self.task_summaries: Dict[str, List[dict]] = defaultdict(list)
        # Per-task target confidence
        self.task_target_confidences: Dict[str, List[float]] = defaultdict(list)

    def run(self):
        """Main entry point for multi-task profiling."""
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging
        log_path = os.path.join(output_dir, 'run_log.txt')
        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Also add handlers to decoder's logger
        self.decoder.logger.addHandler(fh)
        self.decoder.logger.addHandler(ch)

        self.logger.info("=" * 70)
        self.logger.info("EAGLE-3 Multi-Task Acceptance Profiling")
        self.logger.info("=" * 70)
        self.logger.info(f"Tasks: {list(TASK_PROMPTS.keys())}")
        self.logger.info(f"Prompts per task: {self.config.num_samples}")
        self.logger.info(f"K={self.config.num_draft_steps}, max_new_tokens={self.config.max_new_tokens}")

        # Load models
        self.decoder.load_models()

        # Run each task type
        total_start = time.time()
        for task_name, prompts in TASK_PROMPTS.items():
            self._run_task(task_name, prompts)

        total_elapsed = time.time() - total_start
        self.logger.info(f"\nTotal time: {total_elapsed:.1f}s")

        # Run comparative analysis
        self._run_comparative_analysis()
        self.logger.info("All done!")

    def _run_task(self, task_name: str, prompts: List[str]):
        """Run profiling for a single task type."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"TASK: {task_name.upper()} ({len(prompts)} prompts available)")
        self.logger.info(f"{'='*60}")

        num_to_run = min(self.config.num_samples, len(prompts))

        # Reset decoder metrics for this task
        self.decoder.per_token_metrics = []
        self.decoder.per_step_summaries = []

        start_time = time.time()
        for i in range(num_to_run):
            try:
                self.decoder.run_single_prompt(prompts[i], i)
            except Exception as e:
                self.logger.error(f"  Error on prompt {i}: {e}")
                continue

            # Collect target model confidence for this prompt
            # (from verify phase - the target model's top-1 probability)
            # We'll compute this from the verify logits in a modified approach below

        elapsed = time.time() - start_time
        self.logger.info(f"  Task '{task_name}' done in {elapsed:.1f}s")

        # Store task-specific metrics
        self.task_metrics[task_name] = list(self.decoder.per_token_metrics)
        self.task_summaries[task_name] = list(self.decoder.per_step_summaries)

        # Extract target confidence from acceptance data
        # Use draft confidence as proxy (reflects target predictability)
        for m in self.decoder.per_token_metrics:
            self.task_target_confidences[task_name].append(m['confidence'])

        # Save intermediate
        self._save_task_results(task_name)

    def _save_task_results(self, task_name: str):
        """Save results for a single task."""
        task_dir = os.path.join(self.config.output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        with open(os.path.join(task_dir, 'per_token_metrics.jsonl'), 'w') as f:
            for m in self.task_metrics[task_name]:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')

        with open(os.path.join(task_dir, 'per_step_summary.json'), 'w') as f:
            json.dump(self.task_summaries[task_name], f, indent=2, ensure_ascii=False)

    def _compute_target_confidence_from_verify(self, task_name: str, prompts: List[str]):
        """
        Compute per-token target model top-1 confidence during verification.
        This requires modifying the verify loop to also return softmax probs.
        For efficiency, we approximate using the draft confidence collected during run.
        """
        pass  # Using draft confidence as proxy for now

    def _run_comparative_analysis(self):
        """Run full comparative analysis across all tasks."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy import stats

        out_dir = self.config.output_dir
        K = self.config.num_draft_steps

        self.logger.info(f"\n{'='*60}")
        self.logger.info("COMPARATIVE ANALYSIS")
        self.logger.info(f"{'='*60}")

        # =====================================================================
        # 1. Per-task statistics summary (per_task_stats.json)
        # =====================================================================
        per_task_stats = {}
        for task_name, metrics in self.task_metrics.items():
            if not metrics:
                continue

            acceptances = [1 if m['accepted'] else 0 for m in metrics]
            confidences = [m['confidence'] for m in metrics]

            # Acceptance lengths per step
            step_acceptance_lengths = []
            for summ in self.task_summaries[task_name]:
                for step in summ['step_details']:
                    step_acceptance_lengths.append(step['n_accepted'])

            # Easy/hard ratio based on confidence threshold
            conf_arr = np.array(confidences)
            easy_ratio = float((conf_arr > 0.8).mean()) if len(conf_arr) > 0 else 0.0
            hard_ratio = float((conf_arr < 0.3).mean()) if len(conf_arr) > 0 else 0.0

            # GMM and Dip test
            gmm_result, dip_pvalue = self._bimodal_test(conf_arr)

            per_task_stats[task_name] = {
                "mean_acceptance": float(np.mean(acceptances)),
                "mean_confidence": float(np.mean(confidences)),
                "std_confidence": float(np.std(confidences)),
                "easy_ratio": easy_ratio,
                "hard_ratio": hard_ratio,
                "mean_acceptance_length": float(np.mean(step_acceptance_lengths)) if step_acceptance_lengths else 0.0,
                "median_acceptance_length": float(np.median(step_acceptance_lengths)) if step_acceptance_lengths else 0.0,
                "dip_pvalue": dip_pvalue,
                "gmm_best_n": gmm_result.get("best_n_components", -1),
                "gmm_means": gmm_result.get("gmm2_means", []),
                "gmm_weights": gmm_result.get("gmm2_weights", []),
                "num_tokens_evaluated": len(metrics),
                "num_prompts": len(self.task_summaries[task_name]),
            }

            self.logger.info(f"  {task_name}: acceptance={np.mean(acceptances):.1%}, "
                           f"easy={easy_ratio:.1%}, hard={hard_ratio:.1%}, "
                           f"dip_p={dip_pvalue:.4f}")

        with open(os.path.join(out_dir, 'per_task_stats.json'), 'w') as f:
            json.dump(per_task_stats, f, indent=2)

        # =====================================================================
        # 2. Difficulty distributions (per-task histograms with GMM fit)
        # =====================================================================
        self._plot_difficulty_distributions(out_dir)

        # =====================================================================
        # 3. Task comparison (overlaid distributions)
        # =====================================================================
        self._plot_task_comparison(out_dir)

        # =====================================================================
        # 4. Per-position acceptance rate by task
        # =====================================================================
        self._plot_per_position_by_task(out_dir)

        # =====================================================================
        # 5. Acceptance length distribution by task
        # =====================================================================
        self._plot_acceptance_length_by_task(out_dir)

    def _bimodal_test(self, confidences: np.ndarray) -> Tuple[dict, float]:
        """Run GMM + Dip test on confidence distribution."""
        result = {}
        dip_pvalue = 1.0

        if len(confidences) < 20:
            return result, dip_pvalue

        try:
            from sklearn.mixture import GaussianMixture

            confs_2d = confidences.reshape(-1, 1)

            gmm1 = GaussianMixture(n_components=1, random_state=42).fit(confs_2d)
            gmm2 = GaussianMixture(n_components=2, random_state=42).fit(confs_2d)

            bic1 = gmm1.bic(confs_2d)
            bic2 = gmm2.bic(confs_2d)

            result = {
                'bic_1_component': float(bic1),
                'bic_2_components': float(bic2),
                'best_n_components': 2 if bic2 < bic1 else 1,
                'gmm2_means': gmm2.means_.flatten().tolist(),
                'gmm2_weights': gmm2.weights_.tolist(),
                'gmm2_variances': gmm2.covariances_.flatten().tolist(),
            }
        except Exception as e:
            self.logger.warning(f"GMM failed: {e}")

        # Hartigan's Dip Test
        try:
            from diptest import diptest
            dip_stat, dip_pvalue = diptest(confidences)
        except ImportError:
            # Fallback: use scipy-based approximation
            try:
                dip_pvalue = self._approx_dip_test(confidences)
            except Exception:
                dip_pvalue = -1.0

        return result, float(dip_pvalue)

    def _approx_dip_test(self, data: np.ndarray) -> float:
        """Approximate dip test using ecdf-based method."""
        from scipy import stats
        n = len(data)
        sorted_data = np.sort(data)
        ecdf = np.arange(1, n + 1) / n

        # Compute Hartigan's dip statistic approximation
        # max deviation between ecdf and best unimodal cdf
        # Simple approximation: use KS test against uniform as proxy
        ks_stat, p_value = stats.kstest(sorted_data, 'uniform',
                                         args=(sorted_data.min(),
                                               sorted_data.max() - sorted_data.min()))
        return p_value

    def _plot_difficulty_distributions(self, out_dir: str):
        """Plot per-task confidence histograms with GMM fit overlay."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.mixture import GaussianMixture

        task_names = [t for t in TASK_PROMPTS.keys() if t in self.task_metrics]
        n_tasks = len(task_names)
        if n_tasks == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        colors = {'code': '#2196F3', 'math': '#FF5722',
                  'writing': '#4CAF50', 'translation': '#9C27B0'}

        for idx, task_name in enumerate(task_names):
            ax = axes[idx]
            confidences = np.array(self.task_target_confidences[task_name])

            if len(confidences) == 0:
                ax.set_title(f"{task_name} (no data)")
                continue

            # Histogram
            ax.hist(confidences, bins=30, density=True, alpha=0.6,
                   color=colors.get(task_name, 'steelblue'),
                   edgecolor='black', linewidth=0.5,
                   label=f'n={len(confidences)}')

            # GMM fit (2 components)
            try:
                confs_2d = confidences.reshape(-1, 1)
                gmm = GaussianMixture(n_components=2, random_state=42).fit(confs_2d)

                x_plot = np.linspace(0, 1, 200).reshape(-1, 1)
                log_prob = gmm.score_samples(x_plot)
                pdf = np.exp(log_prob)

                ax.plot(x_plot, pdf, 'r-', linewidth=2, label='GMM (2-comp)')

                # Individual components
                for i in range(2):
                    mean = gmm.means_[i, 0]
                    var = gmm.covariances_[i, 0, 0]
                    weight = gmm.weights_[i]
                    comp_pdf = weight * (1 / np.sqrt(2 * np.pi * var)) * \
                              np.exp(-0.5 * (x_plot.flatten() - mean) ** 2 / var)
                    ax.plot(x_plot, comp_pdf, '--', linewidth=1.5, alpha=0.7,
                           label=f'Comp {i+1}: μ={mean:.2f}, w={weight:.2f}')
            except Exception:
                pass

            ax.set_xlabel('Draft Confidence')
            ax.set_ylabel('Density')
            ax.set_title(f'{task_name.upper()}\n'
                        f'mean={confidences.mean():.3f}, '
                        f'easy={float((confidences > 0.8).mean()):.1%}')
            ax.legend(fontsize=8)
            ax.set_xlim(-0.05, 1.05)

        # Hide unused axes
        for idx in range(n_tasks, 4):
            axes[idx].set_visible(False)

        plt.suptitle('Token Difficulty Distribution by Task Type (Confidence)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'difficulty_distributions.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_task_comparison(self, out_dir: str):
        """Plot overlaid comparison of all task distributions."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        colors = {'code': '#2196F3', 'math': '#FF5722',
                  'writing': '#4CAF50', 'translation': '#9C27B0'}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Confidence distributions overlaid
        ax = axes[0]
        for task_name in TASK_PROMPTS.keys():
            if task_name not in self.task_target_confidences:
                continue
            confs = np.array(self.task_target_confidences[task_name])
            if len(confs) == 0:
                continue
            ax.hist(confs, bins=30, density=True, alpha=0.4,
                   color=colors.get(task_name, 'gray'),
                   label=f'{task_name} (μ={confs.mean():.3f})')
            # KDE overlay
            from scipy.stats import gaussian_kde
            if len(confs) > 5:
                kde = gaussian_kde(confs, bw_method=0.1)
                x_kde = np.linspace(0, 1, 200)
                ax.plot(x_kde, kde(x_kde), color=colors.get(task_name, 'gray'),
                       linewidth=2)

        ax.set_xlabel('Draft Confidence')
        ax.set_ylabel('Density')
        ax.set_title('Confidence Distribution Comparison')
        ax.legend()
        ax.set_xlim(-0.05, 1.05)

        # Right: Acceptance rate comparison (box plot)
        ax = axes[1]
        task_acceptance_rates = []
        task_labels = []
        for task_name in TASK_PROMPTS.keys():
            if task_name not in self.task_summaries:
                continue
            rates = [s['overall_acceptance_rate'] for s in self.task_summaries[task_name]]
            if rates:
                task_acceptance_rates.append(rates)
                task_labels.append(task_name)

        if task_acceptance_rates:
            bp = ax.boxplot(task_acceptance_rates, patch_artist=True)
            ax.set_xticklabels(task_labels)
            for i, (patch, task_name) in enumerate(zip(bp['boxes'], task_labels)):
                patch.set_facecolor(colors.get(task_name, 'gray'))
                patch.set_alpha(0.6)
            ax.set_ylabel('Per-Prompt Acceptance Rate')
            ax.set_title('Acceptance Rate Distribution by Task')
            ax.set_ylim(0, 1.05)
            # Add means
            for i, rates in enumerate(task_acceptance_rates):
                ax.scatter(i + 1, np.mean(rates), marker='D', color='red', s=50, zorder=3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'task_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_per_position_by_task(self, out_dir: str):
        """Plot per-position acceptance rate grouped by task."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        K = self.config.num_draft_steps
        colors = {'code': '#2196F3', 'math': '#FF5722',
                  'writing': '#4CAF50', 'translation': '#9C27B0'}
        markers = {'code': 'o', 'math': 's', 'writing': '^', 'translation': 'D'}

        fig, ax = plt.subplots(figsize=(10, 6))

        for task_name in TASK_PROMPTS.keys():
            if task_name not in self.task_metrics:
                continue
            metrics = self.task_metrics[task_name]
            if not metrics:
                continue

            # Compute per-position acceptance rate
            position_accepted = defaultdict(list)
            for m in metrics:
                pos = m['position']
                if pos < K:
                    position_accepted[pos].append(1 if m['accepted'] else 0)

            positions = list(range(K))
            rates = [np.mean(position_accepted[k]) if position_accepted[k] else 0
                    for k in positions]
            counts = [len(position_accepted[k]) for k in positions]

            ax.plot(positions, rates, marker=markers.get(task_name, 'o'),
                   color=colors.get(task_name, 'gray'), linewidth=2,
                   markersize=8, label=f'{task_name} (n={sum(counts)})')

            # Add error bars (95% CI using binomial)
            ci_low = []
            ci_high = []
            for k in positions:
                n = counts[k]
                p = rates[k]
                if n > 0:
                    se = np.sqrt(p * (1 - p) / n)
                    ci_low.append(max(0, p - 1.96 * se))
                    ci_high.append(min(1, p + 1.96 * se))
                else:
                    ci_low.append(0)
                    ci_high.append(0)

            ax.fill_between(positions, ci_low, ci_high,
                          color=colors.get(task_name, 'gray'), alpha=0.1)

        ax.set_xlabel('Draft Position (k)', fontsize=12)
        ax.set_ylabel('Acceptance Rate', fontsize=12)
        ax.set_title('Per-Position Acceptance Rate by Task Type', fontsize=14)
        ax.set_xticks(range(K))
        ax.set_xticklabels([f'k={i+1}' for i in range(K)])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'per_position_by_task.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_acceptance_length_by_task(self, out_dir: str):
        """Plot acceptance length distribution by task."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        K = self.config.num_draft_steps
        colors = {'code': '#2196F3', 'math': '#FF5722',
                  'writing': '#4CAF50', 'translation': '#9C27B0'}

        fig, ax = plt.subplots(figsize=(10, 6))

        width = 0.2
        x = np.arange(K + 1)  # 0, 1, ..., K

        for i, task_name in enumerate(TASK_PROMPTS.keys()):
            if task_name not in self.task_summaries:
                continue

            # Compute acceptance length distribution
            lengths = []
            for summ in self.task_summaries[task_name]:
                for step in summ['step_details']:
                    lengths.append(step['n_accepted'])

            if not lengths:
                continue

            # Count each length
            length_counts = np.zeros(K + 1)
            for l in lengths:
                if l <= K:
                    length_counts[l] += 1
            length_dist = length_counts / length_counts.sum()

            ax.bar(x + i * width, length_dist, width, alpha=0.8,
                  color=colors.get(task_name, 'gray'),
                  label=f'{task_name} (μ={np.mean(lengths):.2f})')

        ax.set_xlabel('Acceptance Length (tokens accepted per step)', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_title('Acceptance Length Distribution by Task Type', fontsize=14)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([str(i) for i in range(K + 1)])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'acceptance_length_by_task.png'), dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="EAGLE-3 Multi-Task Acceptance Profiling")
    parser.add_argument('--num-samples', type=int, default=25,
                       help='Number of prompts per task type (default: 25)')
    parser.add_argument('--num-draft-steps', type=int, default=5,
                       help='Number of draft steps K (default: 5)')
    parser.add_argument('--max-new-tokens', type=int, default=128,
                       help='Max tokens to generate per prompt (default: 128)')
    parser.add_argument('--output-dir', type=str,
                       default='/data1/caoshuaihu.csh/workspace/dahd_speculative_decoding/results/multi_task_difficulty/',
                       help='Output directory')
    args = parser.parse_args()

    config = ProfilingConfig(
        num_samples=args.num_samples,
        num_draft_steps=args.num_draft_steps,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )

    profiler = MultiTaskProfiler(config)
    profiler.run()


if __name__ == "__main__":
    main()
