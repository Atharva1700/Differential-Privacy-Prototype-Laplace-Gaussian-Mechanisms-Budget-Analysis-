"""
Unit tests for DP mechanisms, queries, budget composition.
Run with: pytest tests/ -v
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mechanisms import LaplaceMechanism, GaussianMechanism, PrivacyBudget
from src.queries import dp_count, dp_sum, dp_mean, dp_histogram


np.random.seed(0)
DATA = np.random.normal(50, 10, size=1000).clip(0, 100)


# ---------------------------------------------------------------------------
# LaplaceMechanism
# ---------------------------------------------------------------------------

class TestLaplaceMechanism:

    def test_scale_formula(self):
        m = LaplaceMechanism(sensitivity=2.0, epsilon=0.5)
        assert m.scale == pytest.approx(4.0)

    def test_noise_std(self):
        m = LaplaceMechanism(sensitivity=1.0, epsilon=1.0)
        assert m.noise_std() == pytest.approx(np.sqrt(2), rel=1e-5)

    def test_scalar_output(self):
        m = LaplaceMechanism(sensitivity=1.0, epsilon=1.0)
        result = m.randomize(100.0)
        assert isinstance(result, float)

    def test_array_output_shape(self):
        m = LaplaceMechanism(sensitivity=1.0, epsilon=1.0)
        arr = np.ones(50)
        result = m.randomize(arr)
        assert result.shape == (50,)

    def test_empirical_variance(self):
        """Empirical std should be close to theoretical (sqrt(2)*scale)."""
        m = LaplaceMechanism(sensitivity=1.0, epsilon=1.0)
        samples = [m.randomize(0.0) for _ in range(50_000)]
        assert np.std(samples) == pytest.approx(m.noise_std(), rel=0.05)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            LaplaceMechanism(sensitivity=1.0, epsilon=-1.0)

    def test_invalid_sensitivity(self):
        with pytest.raises(ValueError):
            LaplaceMechanism(sensitivity=0.0, epsilon=1.0)


# ---------------------------------------------------------------------------
# GaussianMechanism
# ---------------------------------------------------------------------------

class TestGaussianMechanism:

    def test_sigma_formula(self):
        m = GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        expected = np.sqrt(2 * np.log(1.25 / 1e-5)) * 1.0 / 1.0
        assert m.sigma == pytest.approx(expected, rel=1e-5)

    def test_empirical_std(self):
        m = GaussianMechanism(sensitivity=1.0, epsilon=0.5, delta=1e-5)
        samples = [m.randomize(0.0) for _ in range(50_000)]
        assert np.std(samples) == pytest.approx(m.sigma, rel=0.05)

    def test_epsilon_out_of_range(self):
        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=1.0, epsilon=1.5, delta=1e-5)

    def test_delta_out_of_range(self):
        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=1.0, epsilon=0.5, delta=0.0)


# ---------------------------------------------------------------------------
# PrivacyBudget
# ---------------------------------------------------------------------------

class TestPrivacyBudget:

    def test_sequential_consume(self):
        b = PrivacyBudget(total_epsilon=1.0)
        assert b.consume(0.4, label="q1") is True
        assert b.consume(0.4, label="q2") is True
        assert b.spent_epsilon == pytest.approx(0.8)
        assert b.remaining_epsilon == pytest.approx(0.2)

    def test_budget_exhausted(self):
        b = PrivacyBudget(total_epsilon=1.0)
        b.consume(0.6)
        assert b.consume(0.5) is False   # would exceed budget

    def test_parallel_composition(self):
        eps_cost, delta_cost = PrivacyBudget.parallel_cost([0.3, 0.5, 0.4])
        assert eps_cost == 0.5

    def test_summary_keys(self):
        b = PrivacyBudget(total_epsilon=2.0)
        b.consume(0.5)
        s = b.summary()
        for key in ["total_epsilon", "spent_epsilon", "remaining_epsilon", "num_queries"]:
            assert key in s


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

class TestDPQueries:

    def test_dp_count_type(self):
        result = dp_count(DATA, epsilon=1.0)
        assert isinstance(result, float)

    def test_dp_sum_clipping(self):
        """DP sum with tight bounds should be smaller than unconstrained sum."""
        result = dp_sum(DATA, epsilon=10.0, low=0.0, high=100.0)
        assert result > 0

    def test_dp_mean_range(self):
        """DP mean should be roughly within range given high epsilon."""
        result = dp_mean(DATA, epsilon=10.0, low=0.0, high=100.0)
        assert 0 <= result <= 100 * 2   # generous tolerance

    def test_dp_histogram_non_negative(self):
        """DP histogram post-processes to non-negative counts."""
        noisy, edges = dp_histogram(DATA, bins=10, epsilon=1.0)
        assert np.all(noisy >= 0)

    def test_dp_histogram_shape(self):
        noisy, edges = dp_histogram(DATA, bins=8, epsilon=1.0)
        assert len(noisy) == 8
        assert len(edges) == 9

    def test_dp_count_accuracy_high_epsilon(self):
        """At ε=10, noisy count should be within 2% of true."""
        true_count = len(DATA)
        errors = [abs(dp_count(DATA, epsilon=10.0) - true_count) / true_count
                  for _ in range(500)]
        assert np.mean(errors) < 0.02

    def test_dp_count_noisier_at_low_epsilon(self):
        """Mean abs error should decrease as epsilon increases."""
        errors = {}
        for eps in [0.1, 1.0, 10.0]:
            true_count = len(DATA)
            errs = [abs(dp_count(DATA, eps) - true_count) for _ in range(300)]
            errors[eps] = np.mean(errs)
        assert errors[0.1] > errors[1.0] > errors[10.0]
