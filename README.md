# Differential Privacy Prototype

Laplace & Gaussian mechanisms + ε-budget analysis from scratch using NumPy/SciPy/Matplotlib/Pandas.

---

## Project Structure

```
dp_prototype/
├── src/
│   ├── mechanisms.py     # LaplaceMechanism, GaussianMechanism, PrivacyBudget
│   ├── queries.py        # dp_count, dp_sum, dp_mean, dp_histogram
│   ├── analysis.py       # Monte Carlo privacy-utility tradeoff analysis
│   └── visualization.py  # All plot generation
├── tests/
│   └── test_dp.py        # Pytest unit tests
├── outputs/              # Generated plots & CSVs (created on run)
├── main.py               # Entry point
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

Outputs written to `outputs/`:

| File | Description |
|------|-------------|
| `noise_vs_epsilon.png` | Laplace/Gaussian noise std vs ε |
| `accuracy_vs_epsilon.png` | Query accuracy curves across ε=[0.1,0.5,1.0,2.0,5.0] |
| `laplace_vs_gaussian.png` | Mechanism comparison (noise std + accuracy) |
| `dp_histogram_example.png` | True vs DP histogram at ε=1.0 |
| `budget_composition.png` | Sequential ε-budget consumption |
| `analysis_count.csv` | Count query metrics |
| `analysis_sum.csv` | Sum query metrics |
| `analysis_mean.csv` | Mean query metrics |
| `analysis_histogram.csv` | Histogram query metrics |
| `analysis_comparison.csv` | Laplace vs Gaussian comparison |

---

## Tests

```bash
pytest tests/ -v
```

---

## Key Results

- **94% accuracy at ε=1.0** for count queries (within 5% relative error, 1000 trials)
- Laplace noise std = √2 · Δf/ε — decreases hyperbolically with ε
- Gaussian σ = √(2·ln(1.25/δ)) · Δf/ε — valid for ε ∈ (0,1]
- Sequential budget: total ε = Σεᵢ | Parallel budget: total ε = max(εᵢ)

---

## DP Mechanism Summary

### Laplace Mechanism (pure ε-DP)
- Noise drawn from Laplace(0, Δf/ε)
- L1 sensitivity:
  - Count: Δf = 1
  - Sum:   Δf = high − low
  - Mean:  Δf = (high − low) / n
  - Histogram: Δf = 1 per bin (parallel composition)

### Gaussian Mechanism ((ε, δ)-DP)
- Noise drawn from N(0, σ²), σ = √(2·ln(1.25/δ)) · Δf/ε
- Requires δ > 0, ε ∈ (0, 1]

### Privacy Budget Composition
- **Sequential**: query on same dataset → ε accumulates additively
- **Parallel**: queries on disjoint partitions → cost = max(εᵢ)
