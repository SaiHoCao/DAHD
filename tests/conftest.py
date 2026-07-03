"""Pytest configuration and shared fixtures for DAHD correctness tests."""

import json
import os
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def project_root():
    """Return the project root path."""
    return PROJECT_ROOT


@pytest.fixture
def e2e_v2_results():
    """Load the canonical e2e_comparison_v2.json results."""
    path = os.path.join(PROJECT_ROOT, "results", "phase4_e2e", "e2e_comparison_v2.json")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def e2e_config(e2e_v2_results):
    """Extract the E2E config from results."""
    return e2e_v2_results["config"]


@pytest.fixture
def per_prompt_results(e2e_v2_results):
    """Extract per-prompt results, transposed to per-prompt-index dicts.

    The raw JSON stores per_prompt as {method: [result_per_prompt, ...]}.
    This fixture transposes it to [{method: result, ...}, ...] for
    easier iteration in tests.
    """
    raw = e2e_v2_results["per_prompt"]
    num_prompts = len(next(iter(raw.values())))
    result = []
    for i in range(num_prompts):
        prompt_dict = {}
        for method, results_list in raw.items():
            if i < len(results_list):
                prompt_dict[method] = results_list[i]
        result.append(prompt_dict)
    return result
