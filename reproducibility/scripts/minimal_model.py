"""
minimal_model.py

The subtraction continues from FF. The 2x2 mechanism ablation showed
that freezing every physical parameter (potential AND thermostat) to
six learned scalars costs nothing on ROC-AUC (FF 0.8822 vs LL 0.8849)
and cuts seed variance by 6.8x. What FF still learns per day is the
fast encoder: five small MLPs (124 -> 32 -> out) producing the initial
condition (z, p), the external force f_ext and the memory terms
(phi, alpha), plus the readout head. That is the last piece of ML in
the model. These two branches ask whether it needs to be ML at all.

    FF_lin   same as FF, but every encoder MLP is replaced by a single
             linear map. If this matches FF, the encoder's nonlinearity
             is decorative: the market state enters the physics through
             a linear projection of the features.

    FF_vol   FF_lin restricted to the 40 volatility features only. The
             single-fold group ablation found volatility the only group
             whose removal costs more than seed noise; this is the
             six-fold, five-seed test of that finding taken to its
             conclusion. If this matches FF, the entire model is:
             40 volatility features -> linear map -> fixed double well ->
             Verlet -> small readout. Roughly 700 parameters against
             31,381, with the physics fixed at six constants.

Neither branch touches the potential, the thermostat, the integrator
or the readout head, so a loss can only come from the encoder.

Parameter counts (with LATENT_DIM = 4):
    LL       31,381
    FF       20,628
    FF_lin    ~1,900   (5 linear maps 124 -> 14 outputs, + scalars + head)
    FF_vol      ~720   (5 linear maps  40 -> 14 outputs, + scalars + head)

The reason this is the right next experiment rather than any addition:
every subtraction so far has been free. Nine additions in this project
have all failed. The evidence says the model is over-specified, and the
way to find out by how much is to keep removing until something breaks.
Whatever survives is the model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import main_v304_soft_labels as mvsl
from main_v304_soft_labels import CFG
import mechanism_ablation as mech
from mechanism_ablation import FixedThermostatEncoder, FixedPotential

BRANCHES = ("FF_lin", "FF_vol")
VOL_IDX = list(range(40, 80))                 # the volatility block of the feature matrix
_ORIG_FAST_IDX, _ORIG_FAST_DIM = list(CFG.FAST_IDX), CFG.FAST_DIM


class LinearFixedEncoder(FixedThermostatEncoder):
    """FixedThermostatEncoder with every MLP collapsed to one Linear.
    Output activations and scalings are kept identical to FastEncoder
    so the downstream physics sees the same ranges."""

    def __init__(self, fast_dim, latent_dim, hidden=32):
        super().__init__(fast_dim, latent_dim, hidden)
        self.z_enc = nn.Sequential(nn.Linear(fast_dim, latent_dim), nn.Tanh())
        self.p_enc = nn.Linear(fast_dim, latent_dim)
        self.f_enc = nn.Linear(fast_dim, latent_dim)
        self.mem_enc = nn.Sequential(nn.Linear(fast_dim, 1), nn.Sigmoid())
        self.alp_enc = nn.Sequential(nn.Linear(fast_dim, 1), nn.Sigmoid())


def install(mode: str) -> None:
    if mode not in BRANCHES:
        raise ValueError(f"mode must be one of {BRANCHES}, got {mode!r}")
    mech.install("FF")                                  # fixed potential + fixed thermostat
    mvsl.FastEncoder = LinearFixedEncoder
    if mode == "FF_vol":
        CFG.FAST_IDX, CFG.FAST_DIM = list(VOL_IDX), len(VOL_IDX)
    else:
        CFG.FAST_IDX, CFG.FAST_DIM = list(_ORIG_FAST_IDX), _ORIG_FAST_DIM


def restore() -> None:
    mech.restore()
    CFG.FAST_IDX, CFG.FAST_DIM = list(_ORIG_FAST_IDX), _ORIG_FAST_DIM
