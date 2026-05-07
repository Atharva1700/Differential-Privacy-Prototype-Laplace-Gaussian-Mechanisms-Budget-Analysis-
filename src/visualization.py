"""
Visualization: noise magnitude vs query accuracy curves.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path

PALETTE = {
    "laplace":  "#2563EB",
    "gaussian": "#DC2626",
    "count":    "#7C3AED",
    "sum":      "#059669",
    "mean":     "#D97706",
    "histogram":"#DB2777",
    "grid":     "#E5E7EB",
    "bg":       "#F9FAFB",
}


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, linestyle="--", alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def plot_noise_vs_epsilon(output_dir: str = "outputs"):
    """Plot Laplace/Gaussian noise std vs epsilon."""
    Path(output_dir).mkdir(exist_ok=True)
    epsilons = np.linspace(0.05, 5.0, 200)
    sensitivity = 1.0
    delta = 1e-5

    lap_std = np.sqrt(2) * sensitivity / epsilons
    gauss_std = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / np.minimum(epsilons, 1.0)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.plot(epsilons, lap_std, color=PALETTE["laplace"], lw=2.5, label="Laplace (pure ε-DP)")
    eps_gauss = epsilons[epsilons <= 1.0]
    gauss_std_plot = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / eps_gauss
    ax.plot(eps_gauss, gauss_std_plot, color=PALETTE["gaussian"], lw=2.5,
            linestyle="--", label=f"Gaussian ((ε,δ)-DP, δ={delta})")

    for eps_mark in [0.1, 0.5, 1.0, 2.0, 5.0]:
        lap_val = np.sqrt(2) * sensitivity / eps_mark
        ax.axvline(eps_mark, color="#9CA3AF", lw=0.8, linestyle=":")
        ax.scatter([eps_mark], [lap_val], color=PALETTE["laplace"], zorder=5, s=50)

    _style_ax(ax, title="Noise Magnitude vs. Privacy Budget (ε)",
              xlabel="Privacy Budget ε", ylabel="Noise Std Dev (σ)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 5.0)
    ax.set_ylim(0, min(lap_std.max(), 30))
    fig.tight_layout()
    path = f"{output_dir}/noise_vs_epsilon.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_accuracy_curves(results: dict, output_dir: str = "outputs"):
    """Plot accuracy (% trials within 5% error) vs epsilon for all query types."""
    Path(output_dir).mkdir(exist_ok=True)

    query_keys = ["count", "sum", "mean", "histogram"]
    colors = [PALETTE["count"], PALETTE["sum"], PALETTE["mean"], PALETTE["histogram"]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    axes = axes.flatten()

    for ax, key, color in zip(axes, query_keys, colors):
        df = results[key]
        acc_col = "accuracy_5pct" if "accuracy_5pct" in df.columns else "accuracy_10pct"
        epsilons = df["epsilon"].values
        accuracy = df[acc_col].values * 100

        ax.fill_between(epsilons, accuracy, alpha=0.15, color=color)
        ax.plot(epsilons, accuracy, "o-", color=color, lw=2.5, ms=8,
                label=f"{key} query")
        ax.axhline(94, color="#374151", lw=1.2, linestyle="--", alpha=0.7,
                   label="94% accuracy")
        ax.axhline(80, color="#9CA3AF", lw=0.8, linestyle=":", alpha=0.7)
        ax.set_ylim(0, 105)
        ax.set_xticks(epsilons)

        # annotate each point
        for eps, acc in zip(epsilons, accuracy):
            ax.annotate(f"{acc:.0f}%", (eps, acc), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color=color)

        threshold_label = "5% error threshold"
        _style_ax(ax, title=f"DP {key.capitalize()} Query — Accuracy vs ε",
                  xlabel="Privacy Budget ε", ylabel=f"% Trials within {threshold_label}")
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Privacy-Utility Tradeoff: Query Accuracy across ε=[0.1, 0.5, 1.0, 2.0, 5.0]",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = f"{output_dir}/accuracy_vs_epsilon.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_laplace_vs_gaussian(df_comparison: pd.DataFrame, output_dir: str = "outputs"):
    """Bar chart comparing Laplace vs Gaussian accuracy."""
    Path(output_dir).mkdir(exist_ok=True)
    epsilons = df_comparison["epsilon"].values
    x = np.arange(len(epsilons))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")

    # Noise std comparison
    ax1.bar(x - width/2, df_comparison["laplace_noise_std"], width,
            label="Laplace", color=PALETTE["laplace"], alpha=0.85)
    ax1.bar(x + width/2, df_comparison["gaussian_noise_std"], width,
            label="Gaussian", color=PALETTE["gaussian"], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"ε={e}" for e in epsilons])
    _style_ax(ax1, title="Noise Std Dev: Laplace vs Gaussian",
              xlabel="Privacy Budget ε", ylabel="Noise Std Dev")
    ax1.legend(fontsize=9)

    # Accuracy comparison
    ax2.bar(x - width/2, df_comparison["laplace_accuracy_5pct"] * 100, width,
            label="Laplace", color=PALETTE["laplace"], alpha=0.85)
    ax2.bar(x + width/2, df_comparison["gaussian_accuracy_5pct"] * 100, width,
            label="Gaussian", color=PALETTE["gaussian"], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"ε={e}" for e in epsilons])
    ax2.set_ylim(0, 110)
    _style_ax(ax2, title="Accuracy (within 5% error): Laplace vs Gaussian",
              xlabel="Privacy Budget ε", ylabel="% Accurate Trials")
    ax2.legend(fontsize=9)

    fig.suptitle("Laplace vs Gaussian Mechanism Comparison (Count Query)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = f"{output_dir}/laplace_vs_gaussian.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_dp_histogram_example(data: np.ndarray, epsilon: float = 1.0,
                               bins: int = 10, output_dir: str = "outputs"):
    """Visual comparison of true vs DP histogram."""
    from src.queries import dp_histogram
    Path(output_dir).mkdir(exist_ok=True)

    true_counts, bin_edges = np.histogram(data, bins=bins)
    noisy_counts, _ = dp_histogram(data, bins=bins, epsilon=epsilon)

    x = np.arange(bins)
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
    ax.bar(x - width/2, true_counts, width, label="True Histogram", color="#1E40AF", alpha=0.85)
    ax.bar(x + width/2, noisy_counts, width, label=f"DP Histogram (ε={epsilon})",
           color="#EF4444", alpha=0.85)

    bin_labels = [f"{bin_edges[i]:.1f}" for i in range(bins)]
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=8)
    _style_ax(ax, title=f"True vs Differentially Private Histogram (ε={epsilon})",
              xlabel="Bin Lower Bound", ylabel="Count")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = f"{output_dir}/dp_histogram_example.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_budget_composition(output_dir: str = "outputs"):
    """Visualize sequential budget consumption over queries."""
    from src.mechanisms import PrivacyBudget
    Path(output_dir).mkdir(exist_ok=True)

    budget = PrivacyBudget(total_epsilon=3.0)
    queries = [
        ("Count age>30", 0.5), ("Sum salaries", 0.8),
        ("Mean tenure", 0.4), ("Histogram dept", 1.0),
        ("Count remote", 0.3),
    ]
    spent = [0.0]
    labels = ["Start"]
    for label, eps in queries:
        ok = budget.consume(eps, label=label)
        spent.append(budget.spent_epsilon)
        labels.append(f"{label}\n(+ε={eps})" + ("" if ok else " ✗"))

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
    ax.step(range(len(spent)), spent, where="post", color=PALETTE["laplace"], lw=2.5)
    ax.fill_between(range(len(spent)), spent, step="post", alpha=0.15, color=PALETTE["laplace"])
    ax.axhline(budget.total_epsilon, color=PALETTE["gaussian"], lw=2, linestyle="--",
               label=f"Total budget ε={budget.total_epsilon}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylim(0, budget.total_epsilon * 1.15)
    _style_ax(ax, title="Sequential Privacy Budget Composition",
              xlabel="Query", ylabel="Cumulative ε Spent")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = f"{output_dir}/budget_composition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def generate_all_plots(results: dict, data: np.ndarray, output_dir: str = "outputs"):
    paths = []
    print("[*] Plotting noise vs epsilon ...")
    paths.append(plot_noise_vs_epsilon(output_dir))
    print("[*] Plotting accuracy curves ...")
    paths.append(plot_accuracy_curves(results, output_dir))
    print("[*] Plotting Laplace vs Gaussian ...")
    paths.append(plot_laplace_vs_gaussian(results["comparison"], output_dir))
    print("[*] Plotting DP histogram example ...")
    paths.append(plot_dp_histogram_example(data, epsilon=1.0, output_dir=output_dir))
    print("[*] Plotting budget composition ...")
    paths.append(plot_budget_composition(output_dir))
    return paths
