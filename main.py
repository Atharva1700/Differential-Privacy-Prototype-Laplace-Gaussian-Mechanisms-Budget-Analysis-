"""
Differential Privacy Prototype — Main Runner
============================================
Runs Laplace/Gaussian mechanisms, epsilon-budget analysis,
privacy-utility tradeoff experiments, and generates all plots.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure src is importable from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.mechanisms import LaplaceMechanism, GaussianMechanism, PrivacyBudget
from src.queries import dp_count, dp_sum, dp_mean, dp_histogram
from src.analysis import run_full_analysis
from src.visualization import generate_all_plots

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Synthetic Dataset
# ---------------------------------------------------------------------------

def make_dataset(n: int = 5000):
    """Generate synthetic employee dataset."""
    ages       = np.random.normal(loc=38, scale=10, size=n).clip(18, 75)
    salaries   = np.random.lognormal(mean=10.8, sigma=0.5, size=n).clip(30_000, 500_000)
    tenure     = np.random.exponential(scale=5, size=n).clip(0, 40)
    return {"ages": ages, "salaries": salaries, "tenure": tenure}


# ---------------------------------------------------------------------------
# 2. Mechanism demo
# ---------------------------------------------------------------------------

def demo_mechanisms(data: dict):
    print("\n" + "=" * 60)
    print("  MECHANISM DEMO")
    print("=" * 60)

    ages = data["ages"]
    true_count = len(ages)
    true_mean  = float(np.mean(ages))
    true_sum   = float(np.sum(np.clip(ages, 18, 75)))

    print(f"\nDataset: n={true_count}, mean_age={true_mean:.2f}, sum_age={true_sum:.0f}")

    print("\n--- Laplace Mechanism (count, ε=1.0) ---")
    lap = LaplaceMechanism(sensitivity=1.0, epsilon=1.0)
    print(lap)
    for _ in range(5):
        print(f"  noisy count = {lap.randomize(float(true_count)):.1f}  (true={true_count})")

    print("\n--- Gaussian Mechanism (count, ε=0.5, δ=1e-5) ---")
    gauss = GaussianMechanism(sensitivity=1.0, epsilon=0.5, delta=1e-5)
    print(gauss)
    for _ in range(5):
        print(f"  noisy count = {gauss.randomize(float(true_count)):.1f}  (true={true_count})")

    print("\n--- DP Histogram (ages, bins=8, ε=1.0) ---")
    noisy_hist, edges = dp_histogram(ages, bins=8, epsilon=1.0, low=18, high=75)
    true_hist, _      = np.histogram(ages, bins=8, range=(18, 75))
    for i, (t, n) in enumerate(zip(true_hist, noisy_hist)):
        bar = "[" + "#" * int(t / true_hist.max() * 30) + "]"
        print(f"  [{edges[i]:.0f}-{edges[i+1]:.0f}]  true={t:4d}  noisy={n:6.1f}  {bar}")


# ---------------------------------------------------------------------------
# 3. Budget demo
# ---------------------------------------------------------------------------

def demo_budget():
    print("\n" + "=" * 60)
    print("  PRIVACY BUDGET COMPOSITION DEMO")
    print("=" * 60)

    budget = PrivacyBudget(total_epsilon=2.0, total_delta=1e-4)
    print(f"\nInitial budget: {budget}")

    queries = [
        ("count_active_users",   0.5,  0.0),
        ("sum_page_views",       0.5,  0.0),
        ("mean_session_duration",0.4,  1e-5),
        ("histogram_regions",    0.5,  0.0),
        ("count_new_signups",    0.5,  0.0),   # this one may fail
    ]

    for label, eps, delta in queries:
        ok = budget.consume(eps, delta=delta, label=label)
        status = "OK" if ok else "REJECTED (budget exhausted)"
        print(f"  {label:<30} ε={eps}  δ={delta}  -> {status}")

    print(f"\nFinal: {budget.summary()}")

    print("\n--- Parallel Composition Example ---")
    epsilons = [0.3, 0.5, 0.4, 0.6]   # queries on disjoint partitions
    par_eps, par_delta = PrivacyBudget.parallel_cost(epsilons)
    print(f"  Per-partition epsilons: {epsilons}")
    print(f"  Parallel composition cost: ε={par_eps}  (vs sequential: ε={sum(epsilons)})")


# ---------------------------------------------------------------------------
# 4. Full analysis + plots
# ---------------------------------------------------------------------------

def run_analysis(data: dict, output_dir: str = "outputs"):
    print("\n" + "=" * 60)
    print("  PRIVACY-UTILITY TRADEOFF ANALYSIS")
    print("=" * 60)

    ages     = data["ages"]
    salaries = data["salaries"]

    print("\n[*] Running Monte Carlo experiments (1000 trials per ε) ...")
    results = run_full_analysis(ages, low=18.0, high=75.0, bins=10)

    # Print summary table
    print("\n  COUNT QUERY RESULTS:")
    df_count = results["count"]
    print(df_count[["epsilon", "mean_relative_error", "accuracy_5pct",
                    "noise_scale", "noise_std"]].to_string(index=False, float_format="{:.4f}".format))

    print("\n  LAPLACE vs GAUSSIAN COMPARISON:")
    df_cmp = results["comparison"]
    print(df_cmp.to_string(index=False, float_format="{:.4f}".format))

    # Export CSVs
    Path(output_dir).mkdir(exist_ok=True)
    for key, df in results.items():
        csv_path = f"{output_dir}/analysis_{key}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path}")

    print("\n[*] Generating visualizations ...")
    generate_all_plots(results, ages, output_dir=output_dir)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  DIFFERENTIAL PRIVACY PROTOTYPE")
    print("  Laplace/Gaussian Mechanisms + ε-Budget Analysis")
    print("=" * 60)

    print("\n[*] Generating synthetic dataset (n=5000) ...")
    data = make_dataset(n=5000)
    print(f"    ages:     mean={data['ages'].mean():.1f}, std={data['ages'].std():.1f}")
    print(f"    salaries: mean=${data['salaries'].mean():,.0f}")
    print(f"    tenure:   mean={data['tenure'].mean():.1f} yrs")

    demo_mechanisms(data)
    demo_budget()
    results = run_analysis(data, output_dir="outputs")

    print("\n" + "=" * 60)
    print("  DONE — outputs/ contains all plots and CSVs")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
