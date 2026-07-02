"""Ablation experiment runner for DAHD speculative decoding.

Provides execution and comparison infrastructure for running ablation
experiments and aggregating results.
"""

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from src.ablations.ablation_config import AblationConfig, get_all_ablation_configs

logger = logging.getLogger(__name__)


class AblationExperiment:
    """Runs a single ablation experiment and compares with baseline.

    Applies configuration modifications to a model, executes inference
    on specified tasks, and computes performance deltas against baseline.
    """

    def __init__(self, config: AblationConfig, baseline_config: dict[str, Any]) -> None:
        """Initialize the ablation experiment.

        Args:
            config: The ablation configuration specifying what to modify.
            baseline_config: The full baseline model/scheduler configuration dict.
        """
        self.config = config
        self.baseline_config = baseline_config
        self.ablation_config = self._build_ablation_config()
        self.results: dict[str, Any] = {}

    def _build_ablation_config(self) -> dict[str, Any]:
        """Build the ablated configuration by applying modifications."""
        ablated = copy.deepcopy(self.baseline_config)
        for key_path, value in self.config.modifications.items():
            parts = key_path.split(".")
            target = ablated
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        return ablated

    def apply_ablation(self, model: Any) -> Any:
        """Apply ablation modifications to the model.

        Args:
            model: The DAHD draft model instance to modify.

        Returns:
            The modified model with ablation applied.
        """
        model_copy = copy.deepcopy(model)

        mods = self.config.modifications

        # Apply scheduler modifications
        if hasattr(model_copy, "scheduler"):
            scheduler = model_copy.scheduler
            if "scheduler.mode" in mods:
                scheduler.mode = mods["scheduler.mode"]
            if "scheduler.fixed_mode" in mods:
                scheduler.fixed_mode = mods["scheduler.fixed_mode"]
            if "scheduler.enable_ar_branch" in mods:
                scheduler.enable_ar_branch = mods["scheduler.enable_ar_branch"]
            if "scheduler.enable_parallel_branch" in mods:
                scheduler.enable_parallel_branch = mods["scheduler.enable_parallel_branch"]
            if "scheduler.enable_mode_switch" in mods:
                scheduler.enable_mode_switch = mods["scheduler.enable_mode_switch"]
            if "scheduler.use_ema" in mods:
                scheduler.use_ema = mods["scheduler.use_ema"]
            if "scheduler.ema_alpha" in mods:
                scheduler.ema_alpha = mods["scheduler.ema_alpha"]
            if "scheduler.enable_probe" in mods:
                scheduler.enable_probe = mods["scheduler.enable_probe"]
            if "scheduler.ar_prefix_length" in mods:
                scheduler.ar_prefix_length = mods["scheduler.ar_prefix_length"]
            if "scheduler.parallel_suffix_length" in mods:
                scheduler.parallel_suffix_length = mods["scheduler.parallel_suffix_length"]

        # Apply model architecture modifications
        if "model.share_backbone" in mods and hasattr(model_copy, "share_backbone"):
            model_copy.share_backbone = mods["model.share_backbone"]
        if "model.independent_branches" in mods and hasattr(model_copy, "independent_branches"):
            model_copy.independent_branches = mods["model.independent_branches"]

        logger.info(f"Applied ablation: {self.config.name}")
        return model_copy

    def run(
        self,
        tasks: list[dict[str, Any]],
        model: Any,
        num_samples: int = 100,
    ) -> dict[str, Any]:
        """Execute the ablation experiment.

        Args:
            tasks: List of task configurations (each with 'name', 'dataset', etc.).
            model: The model to apply ablation to and evaluate.
            num_samples: Number of samples per task to evaluate.

        Returns:
            Dictionary of results with keys per task and aggregated metrics.
        """
        logger.info(f"Running ablation '{self.config.name}' on {len(tasks)} tasks, "
                    f"{num_samples} samples each")

        ablated_model = self.apply_ablation(model)
        task_results: dict[str, dict[str, float]] = {}

        for task_config in tasks:
            task_name = task_config["name"]
            dataset = task_config.get("dataset", [])

            # Take subset of samples
            samples = dataset[:num_samples] if len(dataset) > num_samples else dataset

            # Collect metrics
            speedups = []
            acceptance_rates = []
            correct_count = 0
            total_time = 0.0

            for sample in samples:
                start_time = time.time()

                # Run inference with ablated model
                if hasattr(ablated_model, "generate"):
                    output = ablated_model.generate(
                        sample.get("input_ids"),
                        max_new_tokens=sample.get("max_new_tokens", 256),
                    )
                    elapsed = time.time() - start_time
                    total_time += elapsed

                    # Extract metrics from output
                    if hasattr(output, "metrics"):
                        speedups.append(output.metrics.get("speedup", 1.0))
                        acceptance_rates.append(output.metrics.get("acceptance_rate", 0.0))
                    if hasattr(output, "correct"):
                        correct_count += int(output.correct)

            # Aggregate per-task
            task_results[task_name] = {
                "speedup": float(np.mean(speedups)) if speedups else 0.0,
                "acceptance_rate": float(np.mean(acceptance_rates)) if acceptance_rates else 0.0,
                "correctness": correct_count / max(len(samples), 1),
                "total_time": total_time,
                "num_samples": len(samples),
            }

        self.results = {
            "ablation_name": self.config.name,
            "task_results": task_results,
            "aggregated": self.aggregate_results(task_results),
        }
        return self.results

    def aggregate_results(self, task_results: Optional[dict] = None) -> dict[str, float]:
        """Aggregate results across all tasks into summary metrics.

        Args:
            task_results: Per-task results dict. If None, uses self.results.

        Returns:
            Dictionary with mean speedup, acceptance, and correctness.
        """
        if task_results is None:
            task_results = self.results.get("task_results", {})

        if not task_results:
            return {"speedup": 0.0, "acceptance_rate": 0.0, "correctness": 0.0}

        speedups = [r["speedup"] for r in task_results.values()]
        acceptances = [r["acceptance_rate"] for r in task_results.values()]
        correctnesses = [r["correctness"] for r in task_results.values()]

        return {
            "speedup": float(np.mean(speedups)),
            "acceptance_rate": float(np.mean(acceptances)),
            "correctness": float(np.mean(correctnesses)),
        }

    def compare_with_baseline(
        self,
        ablation_results: dict[str, float],
        baseline_results: dict[str, float],
    ) -> dict[str, float]:
        """Compare ablation results against baseline.

        Args:
            ablation_results: Aggregated ablation metrics.
            baseline_results: Aggregated baseline metrics.

        Returns:
            Dictionary with delta values for each metric.
        """
        return {
            "speedup_delta": ablation_results["speedup"] - baseline_results["speedup"],
            "acceptance_delta": ablation_results["acceptance_rate"] - baseline_results["acceptance_rate"],
            "correctness_delta": ablation_results["correctness"] - baseline_results["correctness"],
            "speedup_ratio": (
                ablation_results["speedup"] / baseline_results["speedup"]
                if baseline_results["speedup"] > 0 else 0.0
            ),
        }


class AblationSuite:
    """Manages and executes a full suite of ablation experiments.

    Runs all predefined ablation configurations, collects results,
    and generates a comprehensive report.
    """

    def __init__(
        self,
        model: Any,
        tasks: list[dict[str, Any]],
        output_dir: str | Path,
        num_samples: int = 100,
    ) -> None:
        """Initialize the ablation suite.

        Args:
            model: The full DAHD model to ablate.
            tasks: List of task configurations for evaluation.
            output_dir: Directory to save results and reports.
            num_samples: Number of samples per task per ablation.
        """
        self.model = model
        self.tasks = tasks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.all_results: dict[str, dict] = {}
        self.baseline_results: Optional[dict[str, float]] = None

    def run_all_ablations(self) -> dict[str, dict]:
        """Run all 5 ablation experiments plus baseline.

        Returns:
            Dictionary mapping ablation name -> full results dict.
        """
        logger.info("Starting ablation suite with all configurations")

        configs = get_all_ablation_configs()
        baseline_config = self._extract_model_config()

        # Run baseline first
        logger.info("Running baseline (full DAHD)...")
        baseline_exp = AblationExperiment(
            AblationConfig(name="full_dahd", description="Full DAHD (baseline)", modifications={}),
            baseline_config,
        )
        baseline_result = baseline_exp.run(self.tasks, self.model, self.num_samples)
        self.baseline_results = baseline_result["aggregated"]
        self.all_results["full_dahd"] = baseline_result

        # Run each ablation
        for config in configs:
            logger.info(f"Running ablation: {config.name}")
            experiment = AblationExperiment(config, baseline_config)
            result = experiment.run(self.tasks, self.model, self.num_samples)

            # Compute delta against baseline
            result["comparison"] = experiment.compare_with_baseline(
                result["aggregated"], self.baseline_results
            )
            self.all_results[config.name] = result

        # Save results
        results_path = self.output_dir / "ablation_all_results.json"
        serializable = {k: self._make_serializable(v) for k, v in self.all_results.items()}
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"All ablation results saved to {results_path}")

        return self.all_results

    def _extract_model_config(self) -> dict[str, Any]:
        """Extract configuration from the current model state."""
        config: dict[str, Any] = {"scheduler": {}, "model": {}}
        if hasattr(self.model, "config"):
            config = copy.deepcopy(vars(self.model.config)) if hasattr(self.model.config, "__dict__") else {}
        return config

    def _make_serializable(self, obj: Any) -> Any:
        """Convert results to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(x) for x in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def generate_report(self) -> str:
        """Generate a markdown report summarizing ablation results.

        Returns:
            Markdown-formatted report string.
        """
        lines = [
            "# Ablation Study Report",
            "",
            "## Summary",
            "",
            "| Configuration | Speedup (×) | Acceptance (%) | Correctness (%) | Δ Speedup |",
            "|---|---|---|---|---|",
        ]

        for name, result in self.all_results.items():
            agg = result.get("aggregated", {})
            comp = result.get("comparison", {})
            speedup = agg.get("speedup", 0)
            acceptance = agg.get("acceptance_rate", 0) * 100
            correctness = agg.get("correctness", 0) * 100
            delta = comp.get("speedup_delta", 0)
            delta_str = f"{delta:+.3f}" if comp else "—"
            lines.append(f"| {name} | {speedup:.3f} | {acceptance:.1f} | {correctness:.1f} | {delta_str} |")

        lines.extend([
            "",
            "## Interpretation",
            "",
            "- **fixed_parallel**: Tests necessity of AR branch for hard tokens",
            "- **fixed_ar**: Tests value of parallel branch for easy tokens",
            "- **fixed_split_gumiho**: Tests dynamic vs static mode allocation",
            "- **probe_only**: Tests contribution of EMA smoothing",
            "- **no_sharing**: Tests benefit of parameter sharing between branches",
            "",
        ])

        report = "\n".join(lines)
        report_path = self.output_dir / "ablation_report.md"
        report_path.write_text(report)
        logger.info(f"Ablation report written to {report_path}")

        return report
