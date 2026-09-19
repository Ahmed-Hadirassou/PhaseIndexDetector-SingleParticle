"""
mechanism_ablation.py

Two by two ablation of WHAT MOVES in the physics, motivated by a
specific empirical claim in the critical-transitions literature.

Guttal, Raghavendra, Goel and Hoarau (PLOS ONE 11, e0144198, 2016)
tested the Scheffer early-warning signature on the Dow Jones before
1929, 1987, 2000 and 2008 and found NO critical slowing down: lag-1
autocorrelation did not rise before any crash. Variance did. Their
reading: financial crashes are not fold bifurcations (a well flattening
until the state falls out) but stochastically driven transitions (the
noise intensity rising while the well stays put). Diks, Hommes and Wang
(2015) reached similar scepticism from a different angle.

PhaseIndex learns both mechanisms separately, every day: the well shape
through (a, b) and the noise through the thermostat (gamma, T). So the
question "which route does a learned physical model attribute crises
to" can be asked directly, by freezing one and leaving the other free:

    LL  potential per day,  thermostat per day     (reference)
    FL  potential FIXED,    thermostat per day     ("temperature route")
    LF  potential per day,  thermostat FIXED       ("barrier route")
    FF  both fixed                                  (null: fixed physics)

If FL matches LL, the well never needed to move: crises are the noise
rising, in agreement with Guttal et al., and the entire slow encoder
(80 inputs, one hidden layer) can be deleted. If LF matches LL, the
opposite. If FF matches LL, the learned physics is decorative and the
model is really "fast encoder -> initial condition -> fixed dynamics ->
readout", which would be a serious finding against the paper's framing.
If neither fixed branch matches, both mechanisms are load-bearing.

"Fixed" means a learned scalar, one per quantity, bounded exactly as
the per-day version is bounded (a, b > 0.1 via softplus; gamma in (0,1)
and T in [TEMP_MIN, TEMP_MAX] via sigmoid), so a fixed branch can only
lose by losing the ability to vary in time, never by losing range.

Every branch uses the scalar tilt c = 2 tanh(g z(M_t)) that the earlier
four-branch ablation showed to be indistinguishable from the network
version, so that tilt is held constant here and the ONLY thing changing
between branches is what moves.

Interface note: the fixed potential deletes param_net rather than
ignoring it, so the removed parameters leave the optimizer too. The
fixed thermostat keeps the fast encoder's other heads (z, p, f_ext, phi,
alpha) untouched; only gam_enc and tmp_enc are bypassed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import main_v304_soft_labels as mvsl
from main_v304_soft_labels import CFG, ParametricBistablePotential, FastEncoder
import tilt_ablation as tilt

BRANCHES = ("LL", "FL", "LF", "FF")
_ORIG_POTENTIAL = ParametricBistablePotential
_ORIG_FAST = FastEncoder
MODE = "LL"


class FixedPotential(ParametricBistablePotential):
    """a, b as two learned scalars; c = 2 tanh(g z(M_t)). No encoder.
    Inherits energy/gradient/forward, so the Verlet contract holds."""

    def __init__(self, slow_dim, latent_dim, hidden=32, macro_skip_dim=0):
        super().__init__(slow_dim - 1, latent_dim, hidden, macro_skip_dim)
        del self.param_net                                   # remove the ML, not just bypass it
        self._n_base = slow_dim - 1
        self.raw_a = nn.Parameter(torch.tensor(0.5))
        self.raw_b = nn.Parameter(torch.tensor(0.5))
        self.g = nn.Parameter(torch.tensor(1.0))

    def get_params(self, x_slow, x_macro_skip=None, vol_signal=None):
        n = x_slow.shape[0]
        a = (F.softplus(self.raw_a) + 0.1).expand(n, 1)
        b = (F.softplus(self.raw_b) + 0.1).expand(n, 1)
        z_m = x_slow[:, self._n_base: self._n_base + 1]
        c = torch.tanh(self.g * z_m) * 2.0
        return a, b, c


class FixedThermostatEncoder(FastEncoder):
    """gamma and T as two learned scalars, same bounds as the per-day
    heads. Everything else the fast encoder produces stays per day."""

    def __init__(self, fast_dim, latent_dim, hidden=32):
        super().__init__(fast_dim, latent_dim, hidden)
        del self.gam_enc
        del self.tmp_enc
        self.raw_gamma = nn.Parameter(torch.tensor(0.0))
        self.raw_temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        n = x.shape[0]
        gamma = torch.sigmoid(self.raw_gamma).expand(n, 1)
        temp = (CFG.TEMP_MIN + (CFG.TEMP_MAX - CFG.TEMP_MIN) * torch.sigmoid(self.raw_temp)).expand(n, 1)
        return (self.z_enc(x), self.p_enc(x) * 0.1, self.f_enc(x) * 0.1,
                gamma, temp, self.mem_enc(x) * 0.5,
                CFG.ALPHA_MIN + (CFG.ALPHA_MAX - CFG.ALPHA_MIN) * self.alp_enc(x))


def install(mode: str) -> None:
    global MODE
    if mode not in BRANCHES:
        raise ValueError(f"mode must be one of {BRANCHES}, got {mode!r}")
    MODE = mode
    if mode[0] == "L":
        tilt.install("physics")                              # per-day a,b + scalar tilt (verified earlier)
    else:
        mvsl.ParametricBistablePotential = FixedPotential
    mvsl.FastEncoder = FixedThermostatEncoder if mode[1] == "F" else _ORIG_FAST


def restore() -> None:
    mvsl.ParametricBistablePotential = _ORIG_POTENTIAL
    mvsl.FastEncoder = _ORIG_FAST
    tilt.restore()


def learned_scalars(model) -> dict:
    """The fixed values a branch settled on, for interpretation."""
    out = {}
    pot = model.potential
    if isinstance(pot, FixedPotential):
        out["a"] = float(F.softplus(pot.raw_a) + 0.1)
        out["b"] = float(F.softplus(pot.raw_b) + 0.1)
        out["barrier_dV"] = out["b"] ** 2 / (4 * out["a"])
    enc = model.fast_enc
    if isinstance(enc, FixedThermostatEncoder):
        out["gamma"] = float(torch.sigmoid(enc.raw_gamma))
        out["temp"] = float(CFG.TEMP_MIN + (CFG.TEMP_MAX - CFG.TEMP_MIN) * torch.sigmoid(enc.raw_temp))
    g = getattr(pot, "g", None)
    if g is not None:
        out["g"] = float(g)
    return out
