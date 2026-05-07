"""
DP-enabled query functions: count, sum, mean, histogram.
Each function applies the appropriate mechanism calibrated to query sensitivity.
"""

import numpy as np
import pandas as pd
from src.mechanisms import LaplaceMechanism, GaussianMechanism


# ---------------------------------------------------------------------------
# Count query  (L1 sensitivity = 1)
# ---------------------------------------------------------------------------

def dp_count(data: np.ndarray, epsilon: float) -> float:
    """Differentially private count query using Laplace mechanism."""
    true_count = float(len(data))
    mech = LaplaceMechanism(sensitivity=1.0, epsilon=epsilon)
    return mech.randomize(true_count)


# ---------------------------------------------------------------------------
# Sum query  (L1 sensitivity = data_range = high - low)
# ---------------------------------------------------------------------------

def dp_sum(data: np.ndarray, epsilon: float, low: float, high: float) -> float:
    """
    Differentially private sum query.
    Data is clipped to [low, high] before summing.
    L1 sensitivity = high - low.
    """
    clipped = np.clip(data, low, high)
    true_sum = float(np.sum(clipped))
    sensitivity = high - low
    mech = LaplaceMechanism(sensitivity=sensitivity, epsilon=epsilon)
    return mech.randomize(true_sum)


# ---------------------------------------------------------------------------
# Mean query  (sensitivity = (high - low) / n  for known n)
# ---------------------------------------------------------------------------

def dp_mean(data: np.ndarray, epsilon: float, low: float, high: float) -> float:
    """
    Differentially private mean using Laplace mechanism.
    Sensitivity of mean = (high - low) / n.
    """
    n = len(data)
    clipped = np.clip(data, low, high)
    true_mean = float(np.mean(clipped))
    sensitivity = (high - low) / n
    mech = LaplaceMechanism(sensitivity=sensitivity, epsilon=epsilon)
    return mech.randomize(true_mean)


# ---------------------------------------------------------------------------
# Histogram query  (L1 sensitivity = 2, but parallel composition -> 1)
# ---------------------------------------------------------------------------

def dp_histogram(data: np.ndarray, bins: int, epsilon: float,
                 low: float = None, high: float = None) -> tuple:
    """
    Differentially private histogram using Laplace mechanism.
    Under parallel composition each bin sees sensitivity=1.
    Returns (noisy_counts, bin_edges).
    """
    lo = float(np.min(data)) if low is None else low
    hi = float(np.max(data)) if high is None else high
    true_counts, bin_edges = np.histogram(data, bins=bins, range=(lo, hi))
    # parallel composition: sensitivity=1 per bin
    mech = LaplaceMechanism(sensitivity=1.0, epsilon=epsilon)
    noisy_counts = mech.randomize(true_counts.astype(float))
    noisy_counts = np.maximum(noisy_counts, 0)   # post-process: non-negative
    return noisy_counts, bin_edges


# ---------------------------------------------------------------------------
# Gaussian mechanism equivalents
# ---------------------------------------------------------------------------

def dp_count_gaussian(data: np.ndarray, epsilon: float, delta: float) -> float:
    """(epsilon, delta)-DP count using Gaussian mechanism."""
    true_count = float(len(data))
    mech = GaussianMechanism(sensitivity=1.0, epsilon=epsilon, delta=delta)
    return mech.randomize(true_count)


def dp_sum_gaussian(data: np.ndarray, epsilon: float, delta: float,
                    low: float, high: float) -> float:
    """(epsilon, delta)-DP sum using Gaussian mechanism."""
    clipped = np.clip(data, low, high)
    true_sum = float(np.sum(clipped))
    sensitivity = high - low
    mech = GaussianMechanism(sensitivity=sensitivity, epsilon=epsilon, delta=delta)
    return mech.randomize(true_sum)
