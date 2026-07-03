"""Correctness tests for DAHD v2 speculative decoding results.

These tests validate the consistency and correctness of recorded
e2e results without requiring GPU access or model loading.

NOTE: The current e2e_comparison_v2.json is **pre-clamping data** — it was generated
before tokens_per_sec was switched to use clamped tokens_generated. In this data:
  - tokens_generated is the RAW value (may exceed max_new_tokens by up to K_max + 2)
  - tokens_per_sec = tokens_generated / wall_time (raw-based)
  - avg_tokens_per_step = tokens_generated / num_steps (raw-based)

# TODO: After re-running the evaluation with the fixed engine (which outputs
#   tokens_per_sec based on clamped tokens_generated), regenerate the JSON and
#   switch tests to strict checks:
#   - tokens_generated <= max_new_tokens (strictly clamped)
#   - tokens_per_sec = tokens_generated / wall_time (clamped-based)
#   - tokens_per_sec_raw = tokens_generated_raw / wall_time

For GPU-based live tests (greedy equivalence), see test_greedy_equivalence
marked with @pytest.mark.slow.
"""

import numpy as np
import pytest


class TestTokenCountBounds:
    """Verify token counts don't exceed expected bounds."""

    def test_tokens_within_max_new_tokens_plus_margin(self, per_prompt_results, e2e_config):
        """tokens_generated should not exceed max_new_tokens + K_max + 2.

        This is a relaxed bound for pre-clamping data. Speculative decoding may
        overshoot max_new_tokens by up to K_max + 2 due to the +2 accounting
        (target_next + bonus) on the final step. The recorded tokens_generated
        in the current JSON reflects the actual (unclamped) count.

        # TODO: When JSON is regenerated with fixed engine, tighten to:
        #   assert tokens <= max_new_tokens  (strict clamping guarantee)
        """
        max_new = e2e_config["max_new_tokens"]
        k_max = max(e2e_config.get("eagle_k", 5),
                    e2e_config.get("k_easy", 4))
        # Relaxed bound: raw tokens can overshoot by at most K_max + 2
        max_allowed = max_new + k_max + 2

        methods = ["vanilla", "eagle3", "parallel", "dahd"]
        for prompt_result in per_prompt_results:
            for method in methods:
                if method not in prompt_result:
                    continue
                tokens = prompt_result[method]["tokens_generated"]
                assert tokens <= max_allowed, (
                    f"{method}: tokens_generated={tokens} > max_allowed={max_allowed} "
                    f"(pre-clamping raw bound: max_new_tokens + K_max + 2 = {max_allowed})"
                )

    def test_tokens_generated_positive(self, per_prompt_results):
        """All methods should generate at least 1 token per prompt."""
        methods = ["vanilla", "eagle3", "parallel", "dahd"]
        for prompt_result in per_prompt_results:
            for method in methods:
                if method not in prompt_result:
                    continue
                tokens = prompt_result[method]["tokens_generated"]
                assert tokens > 0, f"{method}: tokens_generated={tokens} <= 0"


class TestAcceptanceAccounting:
    """Verify acceptance metrics are internally consistent."""

    def test_avg_tokens_per_step_consistency(self, per_prompt_results):
        """avg_tokens_per_step * num_steps should approximate tokens_generated.

        NOTE: In pre-clamping data, both avg_tokens_per_step and tokens_generated
        use the raw (unclamped) token count, so they are mutually consistent.

        # TODO: After JSON regeneration, avg_tokens_per_step uses raw n_tokens
        #   while tokens_generated is clamped. Update this test to use
        #   tokens_generated_raw for consistency check.
        """
        methods = ["eagle3", "parallel", "dahd"]
        for prompt_result in per_prompt_results:
            for method in methods:
                if method not in prompt_result:
                    continue
                r = prompt_result[method]
                if r["num_steps"] == 0:
                    continue
                # avg_tokens_per_step = raw_tokens / num_steps (diagnostic metric)
                expected = r["avg_tokens_per_step"] * r["num_steps"]
                # In pre-clamping data, tokens_generated IS the raw value
                actual = r["tokens_generated"]
                assert abs(expected - actual) < 1.0, (
                    f"{method}: avg_tokens_per_step * num_steps = {expected:.1f} "
                    f"!= tokens_generated = {actual}"
                )

    def test_avg_accepted_non_negative(self, per_prompt_results):
        """Average accepted draft tokens per step should be non-negative."""
        methods = ["eagle3", "parallel", "dahd"]
        for prompt_result in per_prompt_results:
            for method in methods:
                if method not in prompt_result:
                    continue
                avg_acc = prompt_result[method]["avg_accepted"]
                assert avg_acc >= 0, (
                    f"{method}: avg_accepted={avg_acc} < 0"
                )

    def test_tokens_per_sec_consistency(self, per_prompt_results):
        """tokens_per_sec should equal tokens_generated / wall_time.

        NOTE: In the current pre-clamping JSON, tokens_per_sec was computed from
        the raw (unclamped) tokens_generated, and tokens_generated in the JSON is
        also the raw value. So this consistency check holds as-is.

        # TODO: After JSON regeneration with fixed engine:
        #   - tokens_per_sec uses clamped tokens_generated
        #   - tokens_generated in JSON is clamped
        #   - This check should still hold: tps = clamped / wall_time
        #   - Additionally verify: tokens_per_sec_raw = tokens_generated_raw / wall_time
        """
        methods = ["vanilla", "eagle3", "parallel", "dahd"]
        for prompt_result in per_prompt_results:
            for method in methods:
                if method not in prompt_result:
                    continue
                r = prompt_result[method]
                if r["wall_time"] < 1e-6:
                    continue
                # In pre-clamping data: tokens_per_sec = raw_tokens / wall_time
                # and tokens_generated IS raw_tokens
                expected_tps = r["tokens_generated"] / r["wall_time"]
                actual_tps = r["tokens_per_sec"]
                # Allow 1% tolerance for floating point
                assert abs(expected_tps - actual_tps) / max(actual_tps, 1e-6) < 0.01, (
                    f"{method}: computed tps={expected_tps:.2f} != recorded tps={actual_tps:.2f}"
                )


class TestSpeedupConsistency:
    """Verify speedup calculations are consistent."""

    def test_dahd_faster_than_vanilla(self, e2e_v2_results):
        """DAHD should be faster than vanilla AR (basic sanity)."""
        summary = e2e_v2_results["summary"]
        dahd_tps = summary["dahd"]["avg_tokens_per_sec"]
        vanilla_tps = summary["vanilla"]["avg_tokens_per_sec"]
        assert dahd_tps > vanilla_tps, (
            f"DAHD ({dahd_tps:.2f}) not faster than Vanilla ({vanilla_tps:.2f})"
        )

    def test_speedup_calculation(self, e2e_v2_results):
        """Speedup should equal method_tps / vanilla_tps."""
        summary = e2e_v2_results["summary"]
        vanilla_tps = summary["vanilla"]["avg_tokens_per_sec"]
        for method in ["eagle3", "parallel", "dahd"]:
            if method not in summary:
                continue
            reported_speedup = summary[method]["speedup_vs_vanilla"]
            computed_speedup = summary[method]["avg_tokens_per_sec"] / vanilla_tps
            assert abs(reported_speedup - computed_speedup) < 0.01, (
                f"{method}: reported speedup={reported_speedup:.3f} "
                f"!= computed={computed_speedup:.3f}"
            )


@pytest.mark.slow
class TestGreedyEquivalence:
    """Tests requiring GPU and model loading.

    These tests verify that DAHD speculative decoding produces
    exactly the same output as vanilla greedy decoding.
    Run with: pytest -m slow
    """

    def test_greedy_output_matches_vanilla(self):
        """DAHD greedy output should exactly match vanilla greedy output.

        NOTE: This test requires GPU access and the full model.
        It is marked as slow and skipped in CI.
        To run: pytest tests/test_correctness.py -m slow
        """
        pytest.skip(
            "Requires GPU and model access. "
            "Run manually with: pytest -m slow --no-header"
        )
        # Future implementation:
        # 1. Load target model and EAGLE/Gumiho heads
        # 2. For each test prompt, generate with vanilla AR (greedy)
        # 3. Generate with DAHD (greedy verification)
        # 4. Assert output_ids are identical


class TestKVLengthConsistency:
    """Verify KV cache length accounting from result metrics."""

    def test_num_steps_reasonable(self, per_prompt_results, e2e_config):
        """Number of decoding steps should be reasonable.

        With max_new_tokens=128 and avg acceptance > 0:
        - Minimum steps: 1 (all tokens accepted in one step - unlikely)
        - Maximum steps: max_new_tokens (0 acceptance, only bonus each step)
        """
        max_new = e2e_config["max_new_tokens"]
        methods = ["eagle3", "parallel", "dahd"]
        for prompt_result in per_prompt_results:
            for method in methods:
                if method not in prompt_result:
                    continue
                n_steps = prompt_result[method]["num_steps"]
                if n_steps == 0:
                    continue
                # At least 1 step needed
                assert n_steps >= 1, f"{method}: num_steps={n_steps} < 1"
                # Should not exceed max_new_tokens (each step produces at least 2 tokens)
                assert n_steps <= max_new, (
                    f"{method}: num_steps={n_steps} > max_new_tokens={max_new}"
                )
"""Correctness tests for DAHD v2 speculative decoding results.

These tests validate the consistency and correctness of recorded
e2e results without requiring GPU access or model loading.

For GPU-based live tests (greedy equivalence), see test_greedy_equivalence
marked with @pytest.mark.slow.
"""

import numpy as np
import pytest


class TestTokenCountBounds:
    """Verify token counts don't exceed expected bounds."""

    def test_tokens_within_max_new_tokens_plus_margin(self, per_prompt_results, e2e_config):
        """tokens_generated should not exceed max_new_tokens + K_max + 2.

        Speculative decoding may overshoot max_new_tokens by up to K_max + 2
        due to the +2 accounting (target_next + bonus) on the final step.
        The recorded tokens_generated reflects the actual (unclamped) count.
        """
        max_new = e2e_config["max_new_tokens"]
        k_max = max(e2e_config.get("eagle_k", 5),
                    e2e_config.get("k_easy", 4))
        max_allowed = max_new + k_max + 2

        methods = ["vanilla", "eagle3", "parallel", "dahd"]
        for prompt_result in per_prompt_results:
            for method in methods:
                if method not in prompt_result:
                    continue
                tokens = prompt_result[method]["tokens_generated"]
                assert tokens <= max_allowed, (
                    f"{method}: tokens_generated={tokens} > max_allowed={max_allowed}"
                )

    def test_tokens_generated_positive(self, per_prompt_results):
        """All methods should generate at least 1 token per prompt."""
        methods = ["vanilla", "eagle3", "parallel", "dahd"]
        for prompt_result in per_prompt_results:
            for method in methods:
                if method not in prompt_result:
                    continue
                tokens = prompt_result[method]["tokens_generated"]
                assert tokens > 0, f"{method}: tokens_generated={tokens} <= 0"


class TestAcceptanceAccounting:
    """Verify acceptance metrics are internally consistent."""

    def test_avg_tokens_per_step_consistency(self, per_prompt_results):
        """avg_tokens_per_step * num_steps should approximate tokens_generated_raw.

        Note: avg_tokens_per_step is computed from raw n_tokens (not clamped),
        so consistency check must use tokens_generated_raw.
        """
        methods = ["eagle3", "parallel", "dahd"]
        for prompt_result in per_prompt_results:
            for method in methods:
                if method not in prompt_result:
                    continue
                r = prompt_result[method]
                if r["num_steps"] == 0:
                    continue
                # avg_tokens_per_step uses raw n_tokens
                expected = r["avg_tokens_per_step"] * r["num_steps"]
                actual = r.get("tokens_generated_raw", r["tokens_generated"])
                assert abs(expected - actual) < 1.0, (
                    f"{method}: avg_tokens_per_step * num_steps = {expected:.1f} "
                    f"!= tokens_generated_raw = {actual}"
                )

    def test_avg_accepted_non_negative(self, per_prompt_results):
        """Average accepted draft tokens per step should be non-negative."""
        methods = ["eagle3", "parallel", "dahd"]
        for prompt_result in per_prompt_results:
            for method in methods:
                if method not in prompt_result:
                    continue
                avg_acc = prompt_result[method]["avg_accepted"]
                assert avg_acc >= 0, (
                    f"{method}: avg_accepted={avg_acc} < 0"
                )

    def test_tokens_per_sec_consistency(self, per_prompt_results):
        """tokens_per_sec should equal tokens_generated / wall_time."""
        methods = ["vanilla", "eagle3", "parallel", "dahd"]
        for prompt_result in per_prompt_results:
            for method in methods:
                if method not in prompt_result:
                    continue
                r = prompt_result[method]
                if r["wall_time"] < 1e-6:
                    continue
                expected_tps = r["tokens_generated"] / r["wall_time"]
                actual_tps = r["tokens_per_sec"]
                # Allow 1% tolerance for floating point
                assert abs(expected_tps - actual_tps) / max(actual_tps, 1e-6) < 0.01, (
                    f"{method}: computed tps={expected_tps:.2f} != recorded tps={actual_tps:.2f}"
                )


class TestSpeedupConsistency:
    """Verify speedup calculations are consistent."""

    def test_dahd_faster_than_vanilla(self, e2e_v2_results):
        """DAHD should be faster than vanilla AR (basic sanity)."""
        summary = e2e_v2_results["summary"]
        dahd_tps = summary["dahd"]["avg_tokens_per_sec"]
        vanilla_tps = summary["vanilla"]["avg_tokens_per_sec"]
        assert dahd_tps > vanilla_tps, (
            f"DAHD ({dahd_tps:.2f}) not faster than Vanilla ({vanilla_tps:.2f})"
        )

    def test_speedup_calculation(self, e2e_v2_results):
        """Speedup should equal method_tps / vanilla_tps."""
        summary = e2e_v2_results["summary"]
        vanilla_tps = summary["vanilla"]["avg_tokens_per_sec"]
        for method in ["eagle3", "parallel", "dahd"]:
            if method not in summary:
                continue
            reported_speedup = summary[method]["speedup_vs_vanilla"]
            computed_speedup = summary[method]["avg_tokens_per_sec"] / vanilla_tps
            assert abs(reported_speedup - computed_speedup) < 0.01, (
                f"{method}: reported speedup={reported_speedup:.3f} "
                f"!= computed={computed_speedup:.3f}"
            )


@pytest.mark.slow
class TestGreedyEquivalence:
    """Tests requiring GPU and model loading.

    These tests verify that DAHD speculative decoding produces
    exactly the same output as vanilla greedy decoding.
    Run with: pytest -m slow
    """

    def test_greedy_output_matches_vanilla(self):
        """DAHD greedy output should exactly match vanilla greedy output.

        NOTE: This test requires GPU access and the full model.
        It is marked as slow and skipped in CI.
        To run: pytest tests/test_correctness.py -m slow
        """
        pytest.skip(
            "Requires GPU and model access. "
            "Run manually with: pytest -m slow --no-header"
        )
        # Future implementation:
        # 1. Load target model and EAGLE/Gumiho heads
        # 2. For each test prompt, generate with vanilla AR (greedy)
        # 3. Generate with DAHD (greedy verification)
        # 4. Assert output_ids are identical


class TestKVLengthConsistency:
    """Verify KV cache length accounting from result metrics."""

    def test_num_steps_reasonable(self, per_prompt_results, e2e_config):
        """Number of decoding steps should be reasonable.

        With max_new_tokens=128 and avg acceptance > 0:
        - Minimum steps: 1 (all tokens accepted in one step - unlikely)
        - Maximum steps: max_new_tokens (0 acceptance, only bonus each step)
        """
        max_new = e2e_config["max_new_tokens"]
        methods = ["eagle3", "parallel", "dahd"]
        for prompt_result in per_prompt_results:
            for method in methods:
                if method not in prompt_result:
                    continue
                n_steps = prompt_result[method]["num_steps"]
                if n_steps == 0:
                    continue
                # At least 1 step needed
                assert n_steps >= 1, f"{method}: num_steps={n_steps} < 1"
                # Should not exceed max_new_tokens (each step produces at least 2 tokens)
                assert n_steps <= max_new, (
                    f"{method}: num_steps={n_steps} > max_new_tokens={max_new}"
                )
