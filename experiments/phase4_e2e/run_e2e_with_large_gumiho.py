#!/usr/bin/env python3
"""Run end-to-end evaluation with the large-trained Gumiho checkpoint.

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/phase4_e2e/run_e2e_with_large_gumiho.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments/phase4_e2e"))

# Monkey-patch the config to use the new checkpoint
import run_e2e_comparison as e2e_module

# Override the default checkpoint path
original_config_init = e2e_module.E2EConfig.__init__

def patched_init(self):
    # Call dataclass auto-generated init (since it's a dataclass, just set attributes)
    pass

# Simply override at module level before evaluation runs
e2e_module.E2EConfig.gumiho_checkpoint_path = str(PROJECT_ROOT / "checkpoints/gumiho_large/gumiho_best.pt")
e2e_module.E2EConfig.output_dir = str(PROJECT_ROOT / "results/phase4_e2e_large")

if __name__ == "__main__":
    Path(e2e_module.E2EConfig.output_dir).mkdir(parents=True, exist_ok=True)
    e2e_module.run_evaluation()
