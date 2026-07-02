"""Speculative decoding drafter modules.

This package provides:
- SpeculativeDrafter: Abstract base class for all drafters
- DAHDDraftModule: The core Difficulty-Adaptive Hybrid Drafting module
- MedusaBaseline: Medusa-style parallel draft heads baseline
- EAGLEBaseline: EAGLE-style autoregressive feature prediction baseline
- AdaptiveModeRouter: Standalone adaptive routing logic
"""

from src.drafters.base import SpeculativeDrafter, DraftOutput, VerifyResult
from src.drafters.dahd_draft_module import (
    SharedBottomLayer,
    ARBranch,
    ParallelBranch,
    DifficultyProbe,
    DifficultyRouter,
    DAHDDraftModule,
    DAHDDraftOutput,
)
from src.drafters.medusa_baseline import MedusaBaseline
from src.drafters.eagle_baseline import EAGLEBaseline
from src.drafters.router import AdaptiveModeRouter

__all__ = [
    "SpeculativeDrafter",
    "DraftOutput",
    "VerifyResult",
    "SharedBottomLayer",
    "ARBranch",
    "ParallelBranch",
    "DifficultyProbe",
    "DifficultyRouter",
    "DAHDDraftModule",
    "DAHDDraftOutput",
    "MedusaBaseline",
    "EAGLEBaseline",
    "AdaptiveModeRouter",
]
