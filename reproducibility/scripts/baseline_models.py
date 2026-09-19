"""
baseline_models.py

Finance baselines for comparing against PhaseIndex, evaluated on the exact
same walk-forward folds as walk_forward.py. Every baseline is fit ONLY on
data up to train_end and then evaluated causally (no lookahead) over
[test_start, test_end] -- same protocol PhaseIndex itself follows, so
ROC-AUC / PR-AUC are directly comparable fold by fold.

Four baselines. Chosen because practitioners actually run these, not
because they are textbook curiosities -- and each has a real citation:

  1. VIX level               Whaley, R. E. (2000). "The Investor Fear
                              Gauge." Journal of Portfolio Management,
                              26(3), 12-17. Whaley identifies ~30 as an
                              informal alert threshold separating high-
                              from low-volatility regimes.

  2. GARCH(1,1) volatility   Bollerslev, T. (1986). "Generalized
                              Autoregressive Conditional Heteroskedas-
                              ticity." Journal of Econometrics, 31(3),
                              307-327. The standard practitioner
                              volatility model -- almost every real risk
                              desk runs some GARCH variant.

  3. Markov-switching        Hamilton, J. D. (1989). "A New Approach to
                              the Economic Analysis of Nonstationary Time
                              Series and the Business Cycle." Econometrica,
                              57(2), 357-384. Already cited as related work
                              in the PhaseIndex paper -- this makes it an
                              actual comparison, not just a citation.

  4. Trend / moving-average  Faber, M. T. (2007). "A Quantitative Approach
                              to Tactical Asset Allocation." Journal of
                              Wealth Management, 9(4), 69-79. Faber's own
                              rule is a 10-MONTH SMA on monthly data; the
                              ~200-trading-day window below is the standard
                              daily-frequency analogue used throughout the
                              practitioner/CTA literature, not Faber's
                              literal parameter -- said plainly here so the
                              paper doesn't misattribute the exact number.

Install once (Colab or local):
    pip install arch statsmodels --break-system-packages

Every function shares the signature
    fn(all_prices, train_end, test_start, test_end) -> pd.Series
so they can be looped over identically in run_baseline_comparison.py.
`all_prices` is exactly what main_v304_soft_labels.load_price_data(CFG)
returns as its second element -- no separate data download needed, VIX
(CFG.TICKERS_EXTRA) is already in there under the column name "VIX".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def vix_signal(all_prices: pd.DataFrame, train_end: str, test_start: str, test_end: str,
               **_ignored) -> pd.Series:
    """Raw VIX level over the test window. No fitting step -- VIX is
    itself already a forward-looking (implied) volatility index, so
    train_end plays no role here; kept as a parameter only so every
    baseline in this module shares one call signature."""
    vix = all_prices["VIX"]
    return vix.loc[test_start:test_end].rename("vix_level")


def garch_signal(all_prices: pd.DataFrame, train_end: str, test_start: str, test_end: str,
                  target: str = "SPY") -> pd.Series:
    """GARCH(1,1) conditional volatility. Parameters (omega, alpha, beta)
    are estimated ONLY on returns up to train_end via .fit(); the model
    is then re-instantiated on the full train+test return series and
    FIXED at those parameters via .fix() (not re-fit) to filter forward
    through the test window. This is a causal recursive filter -- test
    returns never inform the parameter estimates, only the filter's own
    running state, the same causality PhaseIndex's own stateful
    inference relies on."""
    px = all_prices[target]
    ret = 100 * px.pct_change().dropna()  # percent returns: standard for numerical stability in arch

    train = ret.loc[:train_end]
    full = ret.loc[:test_end]

    am_train = arch_model(train, vol="GARCH", p=1, q=1, dist="normal", mean="constant")
    res_train = am_train.fit(disp="off")

    am_full = arch_model(full, vol="GARCH", p=1, q=1, dist="normal", mean="constant")
    filtered = am_full.fix(res_train.params)
    cond_vol = filtered.conditional_volatility

    return cond_vol.loc[test_start:test_end].rename("garch_vol")


def markov_switching_signal(all_prices: pd.DataFrame, train_end: str, test_start: str, test_end: str,
                             target: str = "SPY") -> pd.Series:
    """Hamilton (1989) 2-regime Markov-switching model on returns, same
    fit-on-train / filter-on-full-with-fixed-params logic as garch_signal.
    Uses FILTERED (not smoothed) probabilities: smoothed probabilities
    use the whole sample in both directions, so a day early in the test
    window would be informed by days later in the test window -- exactly
    the kind of leakage this whole exercise is trying to avoid. Filtered
    probabilities are strictly causal, one direction only.

    Score = filtered probability of being in the higher-variance regime
    (the regime index with the larger fitted sigma2 -- regime labels 0/1
    are otherwise arbitrary and can flip depending on the data)."""
    px = all_prices[target]
    ret = px.pct_change().dropna()

    train = ret.loc[:train_end]
    full = ret.loc[:test_end]

    mod_train = MarkovRegression(train, k_regimes=2, trend="c", switching_variance=True)
    res_train = mod_train.fit()

    crisis_regime = int(res_train.params.filter(like="sigma2").values.argmax())

    mod_full = MarkovRegression(full, k_regimes=2, trend="c", switching_variance=True)
    filt = mod_full.filter(res_train.params)
    prob_crisis = filt.filtered_marginal_probabilities[crisis_regime]

    return prob_crisis.loc[test_start:test_end].rename("markov_prob_crisis")


def trend_signal(all_prices: pd.DataFrame, train_end: str, test_start: str, test_end: str,
                  target: str = "SPY", window: int = 200) -> pd.Series:
    """Faber-style trend rule (see module docstring re: 200d vs Faber's
    literal 10-month/monthly parameter). Score = -(price - SMA) / SMA,
    so higher score = further BELOW the moving average = more
    crisis-like, same orientation convention as the other three signals
    (higher score -> more likely crisis) so ROC-AUC needs no sign flip
    across baselines. A rolling window ending at t -> no fitting, no
    leakage by construction; train_end is unused, kept for signature
    consistency with the other three functions."""
    px = all_prices[target]
    sma = px.rolling(window).mean()
    score = -(px - sma) / sma
    return score.loc[test_start:test_end].rename(f"trend_{window}d")


BASELINES = {
    "vix": vix_signal,
    "garch_1_1": garch_signal,
    "markov_switching": markov_switching_signal,
    "trend_ma200": trend_signal,
}
