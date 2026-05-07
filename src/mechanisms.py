"""
Differential Privacy Mechanisms — Laplace & Gaussian
"""

import numpy as np
from typing import Union


class LaplaceMechanism:
    """
    Laplace Mechanism for (epsilon)-differential privacy.
    Noise scale b = delta_f / epsilon
    """

    def __init__(self, sensitivity: float, epsilon: float):
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.scale = sensitivity / epsilon

    def randomize(self, true_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Add Laplace noise to a scalar or array query result."""
        shape = np.shape(true_value) if np.shape(true_value) else None
        noise = np.random.laplace(loc=0.0, scale=self.scale, size=shape)
        return true_value + noise

    def noise_std(self) -> float:
        """Standard deviation of the Laplace distribution = sqrt(2) * scale."""
        return np.sqrt(2) * self.scale

    def __repr__(self):
        return f"LaplaceMechanism(sensitivity={self.sensitivity}, epsilon={self.epsilon}, scale={self.scale:.4f})"


class GaussianMechanism:
    """
    Gaussian Mechanism for (epsilon, delta)-differential privacy.
    sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
    Valid for epsilon in (0, 1].
    """

    def __init__(self, sensitivity: float, epsilon: float, delta: float):
        if not (0 < epsilon <= 1):
            raise ValueError("epsilon must be in (0, 1] for Gaussian mechanism")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        if sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon

    def randomize(self, true_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Add Gaussian noise to a scalar or array query result."""
        shape = np.shape(true_value) if np.shape(true_value) else None
        noise = np.random.normal(loc=0.0, scale=self.sigma, size=shape)
        return true_value + noise

    def noise_std(self) -> float:
        return self.sigma

    def __repr__(self):
        return (f"GaussianMechanism(sensitivity={self.sensitivity}, "
                f"epsilon={self.epsilon}, delta={self.delta}, sigma={self.sigma:.4f})")


class PrivacyBudget:
    """
    Tracks sequential and parallel privacy budget composition.

    Sequential composition: total_epsilon = sum(epsilon_i), total_delta = sum(delta_i)
    Parallel composition:   total_epsilon = max(epsilon_i), total_delta = max(delta_i)
    """

    def __init__(self, total_epsilon: float, total_delta: float = 0.0):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.queries: list = []

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.total_epsilon - self.spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        return max(0.0, self.total_delta - self.spent_delta)

    def consume(self, epsilon: float, delta: float = 0.0, label: str = "") -> bool:
        """Sequential composition: consume epsilon/delta for one query."""
        if self.spent_epsilon + epsilon > self.total_epsilon + 1e-9:
            return False
        if delta > 0 and self.spent_delta + delta > self.total_delta + 1e-9:
            return False
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self.queries.append({"label": label, "epsilon": epsilon, "delta": delta})
        return True

    @staticmethod
    def parallel_cost(epsilons: list, deltas: list = None):
        """Parallel composition cost: max over disjoint partitions."""
        if deltas is None:
            deltas = [0.0] * len(epsilons)
        return max(epsilons), max(deltas)

    def summary(self) -> dict:
        return {
            "total_epsilon": self.total_epsilon,
            "spent_epsilon": round(self.spent_epsilon, 6),
            "remaining_epsilon": round(self.remaining_epsilon, 6),
            "total_delta": self.total_delta,
            "spent_delta": round(self.spent_delta, 6),
            "num_queries": len(self.queries),
        }

    def __repr__(self):
        return (f"PrivacyBudget(total={self.total_epsilon}, spent={self.spent_epsilon:.4f}, "
                f"remaining={self.remaining_epsilon:.4f})")
