"""
Empirical privacy-utility tradeoff analysis across epsilon values.
Runs Monte Carlo simulations to measure query accuracy vs. noise.
"""

import numpy as np
import pandas as pd
from src.mechanisms import LaplaceMechanism, GaussianMechanism
from src.queries import dp_count, dp_sum, dp_mean, dp_histogram


EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
N_TRIALS = 1000
DELTA = 1e-5


def relative_error(true_val: float, noisy_val: float) -> float:
    if true_val == 0:
        return abs(noisy_val)
    return abs(noisy_val - true_val) / abs(true_val)


def accuracy_within_threshold(errors: np.ndarray, threshold: float = 0.05) -> float:
    """Fraction of trials with relative error <= threshold."""
    return float(np.mean(errors <= threshold))


def analyze_count_query(data: np.ndarray, epsilons=None) -> pd.DataFrame:
    """Analyze DP count query across epsilon values."""
    if epsilons is None:
        epsilons = EPSILON_VALUES
    true_val = float(len(data))
    rows = []
    for eps in epsilons:
        errors = []
        for _ in range(N_TRIALS):
            noisy = dp_count(data, eps)
            errors.append(relative_error(true_val, noisy))
        errors = np.array(errors)
        mech = LaplaceMechanism(sensitivity=1.0, epsilon=eps)
        rows.append({
            "query": "count",
            "epsilon": eps,
            "true_value": true_val,
            "mean_abs_error": float(np.mean(np.abs(errors) * true_val)),
            "mean_relative_error": float(np.mean(errors)),
            "std_relative_error": float(np.std(errors)),
            "accuracy_5pct": accuracy_within_threshold(errors, 0.05),
            "accuracy_1pct": accuracy_within_threshold(errors, 0.01),
            "noise_scale": mech.scale,
            "noise_std": mech.noise_std(),
        })
    return pd.DataFrame(rows)


def analyze_sum_query(data: np.ndarray, low: float, high: float,
                      epsilons=None) -> pd.DataFrame:
    """Analyze DP sum query across epsilon values."""
    if epsilons is None:
        epsilons = EPSILON_VALUES
    true_val = float(np.sum(np.clip(data, low, high)))
    rows = []
    for eps in epsilons:
        errors = []
        for _ in range(N_TRIALS):
            noisy = dp_sum(data, eps, low, high)
            errors.append(relative_error(true_val, noisy))
        errors = np.array(errors)
        sensitivity = high - low
        mech = LaplaceMechanism(sensitivity=sensitivity, epsilon=eps)
        rows.append({
            "query": "sum",
            "epsilon": eps,
            "true_value": true_val,
            "mean_abs_error": float(np.mean(np.abs(errors) * abs(true_val))),
            "mean_relative_error": float(np.mean(errors)),
            "std_relative_error": float(np.std(errors)),
            "accuracy_5pct": accuracy_within_threshold(errors, 0.05),
            "accuracy_1pct": accuracy_within_threshold(errors, 0.01),
            "noise_scale": mech.scale,
            "noise_std": mech.noise_std(),
        })
    return pd.DataFrame(rows)


def analyze_mean_query(data: np.ndarray, low: float, high: float,
                       epsilons=None) -> pd.DataFrame:
    """Analyze DP mean query across epsilon values."""
    if epsilons is None:
        epsilons = EPSILON_VALUES
    n = len(data)
    true_val = float(np.mean(np.clip(data, low, high)))
    rows = []
    for eps in epsilons:
        errors = []
        for _ in range(N_TRIALS):
            clipped = np.clip(data, low, high)
            sensitivity = (high - low) / n
            mech = LaplaceMechanism(sensitivity=sensitivity, epsilon=eps)
            noisy = mech.randomize(true_val)
            errors.append(relative_error(true_val, noisy))
        errors = np.array(errors)
        sensitivity = (high - low) / n
        mech = LaplaceMechanism(sensitivity=sensitivity, epsilon=eps)
        rows.append({
            "query": "mean",
            "epsilon": eps,
            "true_value": true_val,
            "mean_abs_error": float(np.mean(np.abs(errors) * abs(true_val))),
            "mean_relative_error": float(np.mean(errors)),
            "std_relative_error": float(np.std(errors)),
            "accuracy_5pct": accuracy_within_threshold(errors, 0.05),
            "accuracy_1pct": accuracy_within_threshold(errors, 0.01),
            "noise_scale": mech.scale,
            "noise_std": mech.noise_std(),
        })
    return pd.DataFrame(rows)


def analyze_histogram_query(data: np.ndarray, bins: int = 10,
                             epsilons=None) -> pd.DataFrame:
    """Analyze DP histogram (avg per-bin accuracy) across epsilon values."""
    if epsilons is None:
        epsilons = EPSILON_VALUES
    true_counts, bin_edges = np.histogram(data, bins=bins)
    rows = []
    for eps in epsilons:
        bin_errors = []
        for _ in range(N_TRIALS):
            noisy_counts, _ = dp_histogram(data, bins=bins, epsilon=eps)
            errs = [relative_error(t, n) for t, n in zip(true_counts.astype(float), noisy_counts)]
            bin_errors.append(np.mean(errs))
        bin_errors = np.array(bin_errors)
        mech = LaplaceMechanism(sensitivity=1.0, epsilon=eps)
        rows.append({
            "query": "histogram",
            "epsilon": eps,
            "mean_relative_error": float(np.mean(bin_errors)),
            "std_relative_error": float(np.std(bin_errors)),
            "accuracy_5pct": accuracy_within_threshold(bin_errors, 0.05),
            "accuracy_10pct": accuracy_within_threshold(bin_errors, 0.10),
            "noise_scale": mech.scale,
            "noise_std": mech.noise_std(),
        })
    return pd.DataFrame(rows)


def analyze_gaussian_vs_laplace(data: np.ndarray, epsilons=None,
                                 delta: float = DELTA) -> pd.DataFrame:
    """Compare Laplace vs Gaussian mechanism noise std and accuracy for count."""
    if epsilons is None:
        epsilons = [e for e in EPSILON_VALUES if e <= 1.0]
    true_val = float(len(data))
    rows = []
    for eps in epsilons:
        lap = LaplaceMechanism(sensitivity=1.0, epsilon=eps)
        gauss = GaussianMechanism(sensitivity=1.0, epsilon=eps, delta=delta)

        lap_errors, gauss_errors = [], []
        for _ in range(N_TRIALS):
            lap_errors.append(relative_error(true_val, lap.randomize(true_val)))
            gauss_errors.append(relative_error(true_val, gauss.randomize(true_val)))

        rows.append({
            "epsilon": eps,
            "delta": delta,
            "laplace_noise_std": lap.noise_std(),
            "gaussian_noise_std": gauss.noise_std(),
            "laplace_accuracy_5pct": accuracy_within_threshold(np.array(lap_errors), 0.05),
            "gaussian_accuracy_5pct": accuracy_within_threshold(np.array(gauss_errors), 0.05),
        })
    return pd.DataFrame(rows)


def run_full_analysis(data: np.ndarray, low: float, high: float,
                      bins: int = 10) -> dict:
    """Run all analyses and return a dict of DataFrames."""
    print("[*] Analyzing count query ...")
    df_count = analyze_count_query(data)

    print("[*] Analyzing sum query ...")
    df_sum = analyze_sum_query(data, low, high)

    print("[*] Analyzing mean query ...")
    df_mean = analyze_mean_query(data, low, high)

    print("[*] Analyzing histogram query ...")
    df_hist = analyze_histogram_query(data, bins=bins)

    print("[*] Comparing Laplace vs Gaussian ...")
    df_comparison = analyze_gaussian_vs_laplace(data)

    return {
        "count": df_count,
        "sum": df_sum,
        "mean": df_mean,
        "histogram": df_hist,
        "comparison": df_comparison,
    }
