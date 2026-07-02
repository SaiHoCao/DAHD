"""Experiment configuration management for DAHD Speculative Decoding."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ExperimentConfig:
    """Configuration for DAHD speculative decoding experiments.

    Attributes:
        phase: Experiment phase (1-5).
        experiment_name: Human-readable name for the experiment run.
        seed: Random seed for reproducibility.
        device: Target device (e.g., 'cuda:0').
        target_model: HuggingFace model ID or path for the target LLM.
        draft_model: Optional HuggingFace model ID or path for the draft model.
        tasks: List of evaluation tasks.
        num_samples_per_task: Number of samples per evaluation task.
        batch_size: Inference batch size.
        warmup_runs: Number of warmup runs before timing.
        timed_runs: Number of timed runs for latency measurement.
        warmup_stability_cv: Coefficient of variation threshold for warmup stability.
        probe_weight: Weight for probe confidence in difficulty scoring.
        ema_weight: Weight for EMA acceptance rate in difficulty scoring.
        ema_alpha: Exponential moving average smoothing factor.
        easy_threshold: Difficulty score above which a sample is considered easy.
        hard_threshold: Difficulty score below which a sample is considered hard.
        draft_length_easy: Draft length (k) for easy samples.
        draft_length_hard: Draft length (k) for hard samples.
        draft_length_medium: Draft length (k) for medium-difficulty samples.
        output_dir: Directory for experiment outputs.
        checkpoint_dir: Directory for model checkpoints.
    """

    # Experiment identification
    phase: int = 1
    experiment_name: str = "dahd_experiment"
    seed: int = 42
    device: str = "cuda:0"

    # Model configuration
    target_model: str = "meta-llama/Llama-2-7b-hf"
    draft_model: Optional[str] = None

    # Task configuration
    tasks: list[str] = field(default_factory=lambda: ["gsm8k", "humaneval", "mt_bench"])
    num_samples_per_task: int = 100
    batch_size: int = 1

    # Timing configuration
    warmup_runs: int = 5
    timed_runs: int = 30
    warmup_stability_cv: float = 0.02

    # Difficulty routing parameters
    probe_weight: float = 0.6
    ema_weight: float = 0.4
    ema_alpha: float = 0.3

    # Threshold configuration
    easy_threshold: float = 0.7
    hard_threshold: float = 0.5

    # Draft length configuration
    draft_length_easy: int = 6
    draft_length_hard: int = 3
    draft_length_medium: int = 4

    # Output paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 1 <= self.phase <= 5:
            raise ValueError(f"phase must be between 1 and 5, got {self.phase}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if not 0.0 <= self.probe_weight <= 1.0:
            raise ValueError(f"probe_weight must be in [0, 1], got {self.probe_weight}")
        if not 0.0 <= self.ema_weight <= 1.0:
            raise ValueError(f"ema_weight must be in [0, 1], got {self.ema_weight}")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")
        if self.hard_threshold >= self.easy_threshold:
            raise ValueError(
                f"hard_threshold ({self.hard_threshold}) must be less than "
                f"easy_threshold ({self.easy_threshold})"
            )
        if self.draft_length_hard > self.draft_length_easy:
            raise ValueError(
                f"draft_length_hard ({self.draft_length_hard}) should not exceed "
                f"draft_length_easy ({self.draft_length_easy})"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            An ExperimentConfig instance populated from the YAML file.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            yaml_path: Path where the YAML configuration will be written.
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_output_path(self) -> Path:
        """Get the full output directory path for this experiment."""
        path = Path(self.output_dir) / f"phase{self.phase}" / self.experiment_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_checkpoint_path(self) -> Path:
        """Get the full checkpoint directory path for this experiment."""
        path = Path(self.checkpoint_dir) / f"phase{self.phase}" / self.experiment_name
        path.mkdir(parents=True, exist_ok=True)
        return path
