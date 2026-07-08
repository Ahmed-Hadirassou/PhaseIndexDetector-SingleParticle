# PhaseIndex — a physics-inspired market crisis detector

A market-regime detector that treats each trading day as a particle moving
in a learned, input-conditioned Ginzburg-Landau bistable potential, evolved
by a velocity-Verlet integrator under Langevin thermostatting with a memory
(generalized Langevin / Mori-Zwanzig) term. The model outputs a continuous
phase index **Psi** in `[0, 1]`: how far the market has moved from a
"stable" basin toward a "crisis" basin. Per-day friction (`gamma`),
temperature (`T`), and dissipation (`alpha`) are predicted by a small
encoder network rather than fixed constants.

This version (**V30.4**) adds Gaussian-smoothed soft training labels and
Stochastic Weight Averaging on top of the V30.3 architecture. See the
module docstring in `main_v304_soft_labels.py` for the full changelog and
methodology notes.

## Repository layout

| File | Role |
|---|---|
| `main_v304_soft_labels.py` | Model, training loop, data pipeline, entry point (`python main_v304_soft_labels.py`) |
| `features.py` | 200 features across 5 classes: statistical, volatility, momentum, market, macro |
| `velocity_features.py` | 4 features: first/second differences of realized volatility |
| `spectral_features.py` | 40 wavelet-decomposition features (**not currently wired into the pipeline** — see note below) |
| `config.py` | Ticker universe and shared constants read by `features.py` |
| `evaluation_metrics.py` | Statistical indicators beyond the base report: MCC, balanced accuracy, Brier Skill Score, calibration slope/intercept, block-bootstrap confidence intervals, run-history comparison |
| `visualization.py` | Report figures: ROC/PR curves, reliability diagram, confusion matrices, Psi time series, threshold sweep, physics correlations, bootstrap distributions, training curves |

## Running it

```bash
pip install -r requirements.txt
python main_v304_soft_labels.py
```

Downloads the ticker universe via `yfinance`, builds the feature set,
trains a 3-seed ensemble, runs calibration, and writes:

- A console report (detection metrics, calibration, invariants, named
  crises in the test window, threshold sweep, physics correlations,
  bootstrap confidence intervals, version-over-version comparison)
- `results/figures/*.png` — the figures listed above
- `results/run_history.json` — every run's headline metrics, keyed by
  version, so the next version compares against this one automatically

## Before publishing numbers from this repo

Two things worth resolving first, both documented in code where they occur:

1. **Calibration/train overlap.** The default split fits temperature
   scaling on a calibration window that is a *subset* of the training
   window (see `split_data()`). This does not affect the headline
   detection metrics (recall, precision, F1, ROC-AUC, PR-AUC — all
   computed on a genuinely held-out test set), but it can optimistically
   bias the calibration numbers (ECE, Brier, log-loss). Set
   `CFG.STRICT_CALIBRATION_HOLDOUT = True` for a fully disjoint 3-way
   split before quoting those three numbers specifically.
2. **`spectral_features.py` is not imported by the pipeline.** The
   feature-partition indices (`SLOW_IDX` / `FAST_IDX`) are sized for the
   204-column set without it. Confirm this exclusion is intentional.


