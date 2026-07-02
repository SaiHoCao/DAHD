"""Ablation experiment configurations for DAHD speculative decoding.

Defines the 5 core ablation experiments that isolate each component's
contribution to the overall DAHD system performance.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment.

    Attributes:
        name: Short identifier for this ablation (e.g., 'fixed_parallel').
        description: Human-readable description of what is ablated.
        modifications: Dictionary of model/config parameters to override.
            Keys are dot-separated paths (e.g., 'scheduler.mode'), values
            are the override values.
    """
    name: str
    description: str
    modifications: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"AblationConfig(name='{self.name}', mods={list(self.modifications.keys())})"


# ============================================================================
# Predefined Ablation Configurations
# ============================================================================

FIXED_PARALLEL = AblationConfig(
    name="fixed_parallel",
    description=(
        "Fixed Parallel mode: Always use parallel (blockwise) decoding, "
        "disabling the AR branch entirely. Tests whether adaptive switching "
        "provides benefit over always-parallel."
    ),
    modifications={
        "scheduler.mode": "fixed",
        "scheduler.fixed_mode": "parallel",
        "scheduler.enable_ar_branch": False,
        "scheduler.enable_mode_switch": False,
    },
)

FIXED_AR = AblationConfig(
    name="fixed_ar",
    description=(
        "Fixed AR mode: Always use autoregressive decoding from the draft "
        "model, disabling the parallel branch. Tests whether the parallel "
        "branch contributes to overall speedup."
    ),
    modifications={
        "scheduler.mode": "fixed",
        "scheduler.fixed_mode": "ar",
        "scheduler.enable_parallel_branch": False,
        "scheduler.enable_mode_switch": False,
    },
)

FIXED_SPLIT_GUMIHO = AblationConfig(
    name="fixed_split_gumiho",
    description=(
        "Gumiho-style fixed split: Use AR for the first 2 draft tokens, "
        "then switch to parallel for the remaining 5 tokens. Tests whether "
        "dynamic switching outperforms a simple fixed split strategy."
    ),
    modifications={
        "scheduler.mode": "fixed_split",
        "scheduler.ar_prefix_length": 2,
        "scheduler.parallel_suffix_length": 5,
        "scheduler.enable_mode_switch": False,
        "scheduler.enable_probe": False,
    },
)

PROBE_ONLY = AblationConfig(
    name="probe_only",
    description=(
        "Probe-only switching: Use only the lightweight probe's instantaneous "
        "confidence score for mode switching, without the EMA smoothing. "
        "Tests whether EMA stabilization improves switching decisions."
    ),
    modifications={
        "scheduler.mode": "adaptive",
        "scheduler.use_ema": False,
        "scheduler.confidence_source": "probe_only",
        "scheduler.ema_alpha": 0.0,
    },
)

NO_SHARING = AblationConfig(
    name="no_sharing",
    description=(
        "No parameter sharing: AR and parallel branches use completely "
        "independent parameters (no shared backbone layers). Tests whether "
        "weight sharing between branches is beneficial."
    ),
    modifications={
        "model.share_backbone": False,
        "model.shared_layers": 0,
        "model.independent_branches": True,
    },
)

# Registry of all ablation configs
_ALL_ABLATIONS: dict[str, AblationConfig] = {
    "fixed_parallel": FIXED_PARALLEL,
    "fixed_ar": FIXED_AR,
    "fixed_split_gumiho": FIXED_SPLIT_GUMIHO,
    "probe_only": PROBE_ONLY,
    "no_sharing": NO_SHARING,
}


def get_all_ablation_configs() -> list[AblationConfig]:
    """Return all predefined ablation configurations.

    Returns:
        List of all 5 AblationConfig instances.
    """
    return list(_ALL_ABLATIONS.values())


def get_ablation_by_name(name: str) -> AblationConfig:
    """Retrieve an ablation configuration by its name.

    Args:
        name: The ablation name (e.g., 'fixed_parallel', 'probe_only').

    Returns:
        The corresponding AblationConfig.

    Raises:
        KeyError: If the ablation name is not found.
    """
    if name not in _ALL_ABLATIONS:
        available = ", ".join(_ALL_ABLATIONS.keys())
        raise KeyError(f"Unknown ablation '{name}'. Available: {available}")
    return _ALL_ABLATIONS[name]
