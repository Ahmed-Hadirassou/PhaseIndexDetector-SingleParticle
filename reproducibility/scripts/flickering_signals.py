"""
flickering_signals.py

A third leading-indicator family, complementary to Delta_FDT and Kramers
(physics_signals.py), grounded in a different branch of the same
critical-transitions literature: flickering (Wang, Dearing, Langdon,
Zhang, Yang, Dakos and Scheffer, Nature 492, 419, 2012; Dakos, Van Nes
and Scheffer, Theoretical Ecology 6, 309, 2013).

WHY THIS IS THE RIGHT NEXT SIGNAL, NOT JUST ANOTHER ONE

Guttal et al. (2016) found no critical slowing down before US market
crashes -- no rising autocorrelation -- and read this as evidence
against the fold-flattening picture. The mechanism ablation just run on
this project measured the same thing from inside a trained model: the
fixed-potential branch (FL) matched the per-day potential (LL) on 5 of
6 crises. The well does not need to move. Both findings point the same
way, independently.

The flickering literature is the mechanistic account of what a bistable
system does INSTEAD, when the well itself is not changing shape: under
strong enough noise, the state occasionally crosses toward the other
basin and back before finally committing, well before the transition is
final. This is not a new hypothesis bolted onto the model -- it is the
literature's own answer to the question the FL result already raised.
Dakos et al. (2013) give the simplest operational form directly:
flickering is often visible as rising variance in the state variable
over a rolling window, without needing autocorrelation to rise at all.

WHAT IS COMPUTED, AND FROM WHAT

Psi(t) already IS the model's own order parameter -- a learned
compression of the latent state into [0, 1] where the two wells sit at
the extremes. Rather than reconstruct the two wells in raw latent space
(a genuine ambiguity: the potential's basins are a sphere in the
4-dimensional z, ||z||^2 = b/2a, oriented by the tilt term c . mean(z),
not a simple +/- line), flickering is measured directly on Psi, which
is exactly the scalar order parameter the flickering literature
operates on in every cited application. This also means NOTHING new
needs to be extracted from the trained model: Psi(t) is the same array
already produced by every walk-forward run in this project.

Two operationalizations, both causal (rolling window ending at t, nothing
from t+1 onward):

    flicker_var(t)     rolling variance of Psi over the trailing window
    flicker_cross(t)   rolling rate of crossings of the trailing median
                        (excursions across the window's own center,
                        Wang et al.'s literal "switching back and forth")

Window default: 20 trading days (about one month), the standard
short-window choice in the empirical early-warning-signal literature;
distinct from and much shorter than Chiarella's 150-200 day trend
horizon used elsewhere in this project, because flickering is a
microstructure phenomenon, not a trend.
"""
from __future__ import annotations

import numpy as np


def flicker_var(psi: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling variance of Psi, causal (uses psi[t-window+1 : t+1]).
    First `window - 1` entries are NaN: not enough history yet."""
    n = len(psi)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        out[t] = np.var(psi[t - window + 1: t + 1])
    return out


def flicker_cross_rate(psi: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling rate of crossings of the window's own trailing median:
    the fraction of consecutive-day pairs in the window that land on
    opposite sides of that window's median. This is Wang et al.'s
    "switching back and forth" made concrete and causal."""
    n = len(psi)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        seg = psi[t - window + 1: t + 1]
        med = np.median(seg)
        side = seg > med
        out[t] = np.mean(side[1:] != side[:-1])
    return out


def fill_lead_in(x: np.ndarray) -> np.ndarray:
    """Replace the leading NaNs (not enough history for the window yet)
    with the first valid value, so the array can be scored over the
    same test window as every other signal without shrinking it."""
    out = x.copy()
    valid = np.where(~np.isnan(out))[0]
    if len(valid) and valid[0] > 0:
        out[: valid[0]] = out[valid[0]]
    return out
