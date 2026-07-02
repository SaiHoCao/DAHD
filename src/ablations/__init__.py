"""Ablation experiment framework for DAHD speculative decoding.

Provides configuration, execution, and fairness verification for
systematic ablation studies of DAHD components.
"""

from src.ablations.ablation_config import (
    AblationConfig,
    FIXED_PARALLEL,
    FIXED_AR,
    FIXED_SPLIT_GUMIHO,
    PROBE_ONLY,
    NO_SHARING,
    get_all_ablation_configs,
    get_ablation_by_name,
)
from src.ablations.runner import AblationExperiment, AblationSuite
from src.ablations.fairness import AblationFairnessChecker

__all__ = [
    "AblationConfig",
    "FIXED_PARALLEL",
    "FIXED_AR",
    "FIXED_SPLIT_GUMIHO",
    "PROBE_ONLY",
    "NO_SHARING",
    "get_all_ablation_configs",
    "get_ablation_by_name",
    "AblationExperiment",
    "AblationSuite",
    "AblationFairnessChecker",
]
