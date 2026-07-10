# PhaseIndex — a physics-inspired market crisis detector

[![tests](https://github.com/Ahmed-Hadirassou/PhaseIndexDetector-SingleParticle/actions/workflows/tests.yml/badge.svg)](https://github.com/Ahmed-Hadirassou/PhaseIndexDetector-SingleParticle/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/github/license/Ahmed-Hadirassou/PhaseIndexDetector-SingleParticle)](LICENSE)

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

*Why "SingleParticle"?* This detector represents the whole market as one
particle in one potential well. A multi-particle formulation (one particle
per sector, coupled) is a separate, heavier architecture — naming this
repo explicitly keeps the two from being conflated.

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
| `tests/` | pytest suite — labels, loss, one physics step, report arithmetic, statistics, every plot function |

## Running it

```bash
pip install -r requirements.txt
python main_v304_soft_labels.py
```

Downloads the ticker universe via `yfinance`, builds the feature set,
trains a 3-seed ensemble, calibrates it two ways (see *Calibration*
below), and writes:

- A console report (detection metrics, both calibrations, invariants,
  named crises in the test window, threshold sweep, physics correlations,
  bootstrap confidence intervals, version-over-version comparison)
- `results/figures/*.png` — the figures listed above
- `results/run_history.json` — every run's headline metrics, keyed by
  version, so the next version compares against this one automatically

Total runtime is roughly 6-7 minutes on CPU (two 3-seed ensembles: the
final model plus the calibration proxy — see below).

## Results

From a real run (3-seed ensemble, test window 2020-01-01 through the end
of the sample). Re-running will not reproduce these to the last decimal —
training is stochastic even with fixed seeds — but should land within
noise of them.

**Detection (test set, never seen in training)**

| Metric | Base (Psi ≥ 0.65) | Calibrated (tau* = 0.25) |
|---|---|---|
| Accuracy | 91.45% | 90.97% |
| Recall | 55.62% | 73.75% |
| Precision | 83.18% | 69.41% |
| F1 | 66.67% | 71.52% |
| F2 | — | 72.84% |
| FPR | 2.04% | 5.90% |

| Invariant | Value |
|---|---|
| ROC-AUC | 0.9100 |
| PR-AUC | 0.7535 |
| Cohen's d | 1.97 |
| MCC | 0.662 |
| Balanced accuracy | 83.92% |

Block-bootstrap 95% CI (day-level resampling within the test window, not
a claim about generalization to a future crisis — see *Calibration and
statistical rigor* below): ROC-AUC [0.819, 0.981], PR-AUC [0.453, 0.936].

**By named crisis inside the test window**

| Crisis | Recall (base) | Recall (calibrated) |
|---|---|---|
| COVID-19 crash (Feb-Apr 2020) | 86.3% | 90.2% |
| 2022 rate-hike drawdown | 41.3% | 66.1% |

The gap between these two is real and worth stating up front rather than
averaging away: this detector is markedly better at fast, liquidity-panic
crises (COVID-like) than at slow, rate-driven grinds (2022-like). See the
module docstring's account of the `STRICT_CALIBRATION_HOLDOUT` experiment
for a concrete illustration of why — removing a single training-set
example of the "slow grind" pattern was enough to roughly halve recall on
that crisis type specifically, which says the model leans on very few
exemplars for it.

**Calibration**

Two temperatures are fit and reported side by side; see *Calibration and
statistical rigor* for what each means and why both are shown.

| | ECE | Brier | Log-loss | Calib. slope (1.0 = ideal) |
|---|---|---|---|---|
| In-sample T_calib | 2.66% | 0.0674 | 0.2575 | 0.689 |
| Proxy (out-of-sample) T_calib | **2.04%** | **0.0669** | **0.2502** | **0.784** |

The proxy calibration is better on every one of these, not just cleaner
in principle — use it as the headline calibration number.

**Adding figures to this section**

```bash
python main_v304_soft_labels.py   # writes results/figures/*.png
git add results/
git commit -m "Add example run output"
git push
```

then embed the most informative ones here, for example:

```markdown
![Phase index over the test window](results/figures/04_psi_timeseries.png)
![ROC and precision-recall curves](results/figures/01_roc_pr.png)
![Reliability diagram](results/figures/02_calibration.png)
```

## Calibration and statistical rigor

Three things worth understanding before citing numbers from this repo,
all documented in more detail in code where they occur:

1. **Why two calibrations are reported.** The calibration window
   (2017-2019) is a subset of the training window (`CFG.
   STRICT_CALIBRATION_HOLDOUT = False` by default), so a temperature
   fit directly on it (`T_calib`, "in-sample" above) is fit on model
   outputs the model has already trained on. Excluding that window from
   training instead — `STRICT_CALIBRATION_HOLDOUT = True` — was tried
   and rejected as the default: it removes the `2018-10-01..12-31` crisis
   from training entirely (the model's only training-set example of a
   slow, rate-driven drawdown), which measurably hurt 2022-crisis recall
   and every other detection metric, without even improving ECE. Full
   story, including the exact before/after numbers, in the module
   docstring.

   The fix that worked: `CFG.PROXY_CALIBRATION = True` (default) trains a
   second, disposable ensemble on data strictly before 2017, evaluates
   *that* ensemble on the calibration window (genuinely out-of-sample for
   it), and fits a temperature from those outputs instead. That
   temperature is then applied to the fully-trained final model's test
   predictions — so it costs an extra ~2 minutes of training and touches
   nothing about what the final model learns from. Table above shows the
   result: better ECE, Brier, log-loss, and calibration slope than the
   in-sample version, simultaneously. The one assumption this rests on —
   that the proxy ensemble's degree of overconfidence transfers
   reasonably to the final ensemble's — is stated explicitly in
   `train_proxy_calibration_ensemble()`; it is a materially better
   assumption than the in-sample alternative, not a proof.

2. **`spectral_features.py` is not imported by the pipeline.** The
   feature-partition indices (`SLOW_IDX` / `FAST_IDX`) are sized for the
   204-column set without it. Still open — confirm this exclusion is
   intentional before publishing, or wire it in.

3. **Only two crisis episodes fall inside the test window** (COVID 2020,
   the 2022 drawdown). The block-bootstrap confidence intervals reported
   above account for day-to-day autocorrelation within the test window,
   but no resampling scheme manufactures more independent crisis episodes
   than the historical record contains. Treat the reported intervals as
   "uncertainty from resampling this window," not "uncertainty over a
   future, unseen crisis" — and say so if quoting them.

If any number from this repo appears in a public place (a repo
description, a social card, a paper abstract), keep it in sync with
whatever a current run of `main_v304_soft_labels.py` actually produces —
a caveat that only lives in this file is easy to miss if the number was
seen somewhere else first.

## Tests

```bash
pytest
```

Runs on every push via GitHub Actions (badge above). Every test operates
on synthetic or hand-picked inputs — the suite never calls
`yfinance.download()` or trains a real model, so it stays fast, free, and
independent of Yahoo Finance's rate limits. A full download-and-train run
is a separate, manual step (`python main_v304_soft_labels.py`), not
something CI should do on every push.

## License

MIT (see `LICENSE`). If GitHub's license badge above isn't rendering,
check that the file contains an unmodified standard MIT template — GitHub
detects license type by matching file content, and a hand-edited file can
fail that match even though it's still legally a valid license.
