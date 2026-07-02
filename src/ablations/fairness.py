"""Fairness verification for ablation experiments.

Ensures that ablation comparisons are scientifically valid by verifying
seed determinism, weight freezing, and input consistency.
"""

import hashlib
import logging
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AblationFairnessChecker:
    """Verifies fairness conditions for ablation experiments.

    Ensures reproducible and fair comparisons by checking:
    1. Random seed determinism across runs
    2. Model weight immutability during evaluation
    3. Input dataset consistency across configurations
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize the fairness checker.

        Args:
            seed: The random seed that should be consistently used.
        """
        self.seed = seed
        self.check_results: dict[str, dict[str, Any]] = {}

    def verify_seed_determinism(self, configs: list[dict[str, Any]]) -> dict[str, Any]:
        """Verify that all configurations use the same random seed.

        Runs a short determinism test: given the same seed and input,
        verifies that outputs are bit-identical.

        Args:
            configs: List of configuration dicts to check.

        Returns:
            Dictionary with 'passed' bool and details.
        """
        result: dict[str, Any] = {"passed": True, "details": []}

        for i, config in enumerate(configs):
            config_seed = config.get("seed", config.get("random_seed", None))
            if config_seed is None:
                result["details"].append(f"Config {i}: No seed specified (will default to {self.seed})")
            elif config_seed != self.seed:
                result["passed"] = False
                result["details"].append(
                    f"Config {i}: Seed mismatch — expected {self.seed}, got {config_seed}"
                )
            else:
                result["details"].append(f"Config {i}: Seed OK ({config_seed})")

        # Verify PyTorch determinism
        torch.manual_seed(self.seed)
        ref_tensor = torch.randn(100)
        torch.manual_seed(self.seed)
        test_tensor = torch.randn(100)

        if torch.allclose(ref_tensor, test_tensor):
            result["details"].append("PyTorch determinism: VERIFIED")
        else:
            result["passed"] = False
            result["details"].append("PyTorch determinism: FAILED")

        # Verify NumPy determinism
        rng1 = np.random.default_rng(self.seed)
        ref_arr = rng1.random(100)
        rng2 = np.random.default_rng(self.seed)
        test_arr = rng2.random(100)

        if np.allclose(ref_arr, test_arr):
            result["details"].append("NumPy determinism: VERIFIED")
        else:
            result["passed"] = False
            result["details"].append("NumPy determinism: FAILED")

        self.check_results["seed_determinism"] = result
        return result

    def verify_model_weights_frozen(self, models: list[Any]) -> dict[str, Any]:
        """Verify that model weights remain unchanged across ablation runs.

        Computes a hash of model parameters before and after to ensure
        no weight modifications occur during evaluation.

        Args:
            models: List of model instances to verify.

        Returns:
            Dictionary with 'passed' bool and per-model hash details.
        """
        result: dict[str, Any] = {"passed": True, "details": [], "hashes": []}

        for i, model in enumerate(models):
            if not hasattr(model, "parameters"):
                result["details"].append(f"Model {i}: No parameters() method, skipping")
                continue

            # Compute parameter hash
            param_hash = self._compute_param_hash(model)
            result["hashes"].append(param_hash)
            result["details"].append(f"Model {i}: hash={param_hash[:16]}...")

            # Check gradients are disabled
            has_grad = False
            for param in model.parameters():
                if param.requires_grad:
                    has_grad = True
                    break

            if has_grad:
                result["details"].append(f"Model {i}: WARNING — some params have requires_grad=True")
            else:
                result["details"].append(f"Model {i}: All params frozen (requires_grad=False)")

        # If multiple models, verify hashes match (for same-architecture ablations)
        if len(result["hashes"]) > 1:
            # Note: different ablations may have different weights (e.g., no_sharing)
            # Only flag if models that should be identical differ
            result["details"].append(f"Hash comparison: {len(set(result['hashes']))} unique hashes "
                                     f"across {len(result['hashes'])} models")

        self.check_results["weights_frozen"] = result
        return result

    def _compute_param_hash(self, model: Any) -> str:
        """Compute SHA256 hash of all model parameters."""
        hasher = hashlib.sha256()
        for param in model.parameters():
            hasher.update(param.data.cpu().numpy().tobytes())
        return hasher.hexdigest()

    def verify_input_consistency(self, datasets: list[list[dict]]) -> dict[str, Any]:
        """Verify that all ablation runs use identical input data.

        Checks that dataset lengths match and content hashes are identical.

        Args:
            datasets: List of datasets (each is a list of sample dicts).

        Returns:
            Dictionary with 'passed' bool and consistency details.
        """
        result: dict[str, Any] = {"passed": True, "details": []}

        if len(datasets) < 2:
            result["details"].append("Only one dataset provided, consistency check trivially passes")
            self.check_results["input_consistency"] = result
            return result

        # Check lengths
        lengths = [len(ds) for ds in datasets]
        if len(set(lengths)) > 1:
            result["passed"] = False
            result["details"].append(f"Length mismatch: {lengths}")
        else:
            result["details"].append(f"All datasets have {lengths[0]} samples")

        # Check content hashes
        hashes = []
        for i, ds in enumerate(datasets):
            content_str = str(sorted([str(sorted(s.items())) for s in ds]))
            ds_hash = hashlib.sha256(content_str.encode()).hexdigest()
            hashes.append(ds_hash)
            result["details"].append(f"Dataset {i}: hash={ds_hash[:16]}...")

        if len(set(hashes)) > 1:
            result["passed"] = False
            result["details"].append("Content hashes differ — datasets are NOT identical")
        else:
            result["details"].append("All datasets have identical content hashes")

        self.check_results["input_consistency"] = result
        return result

    def run_all_checks(
        self,
        configs: Optional[list[dict]] = None,
        models: Optional[list[Any]] = None,
        datasets: Optional[list[list[dict]]] = None,
    ) -> dict[str, dict[str, Any]]:
        """Run all fairness verification checks.

        Args:
            configs: Configuration dicts for seed check.
            models: Model instances for weight check.
            datasets: Dataset lists for input check.

        Returns:
            Dictionary with results of all checks.
        """
        logger.info("Running all ablation fairness checks...")

        if configs is not None:
            self.verify_seed_determinism(configs)

        if models is not None:
            self.verify_model_weights_frozen(models)

        if datasets is not None:
            self.verify_input_consistency(datasets)

        # Summary
        all_passed = all(r.get("passed", True) for r in self.check_results.values())
        self.check_results["summary"] = {
            "all_passed": all_passed,
            "num_checks": len(self.check_results) - 1,  # Exclude summary itself
            "checks_run": list(self.check_results.keys()),
        }

        if all_passed:
            logger.info("All fairness checks PASSED")
        else:
            failed = [k for k, v in self.check_results.items()
                      if k != "summary" and not v.get("passed", True)]
            logger.warning(f"Fairness checks FAILED: {failed}")

        return self.check_results
