"""
physics_signals.py

Two physics-first signals, both computed AFTER training from quantities
the trained model already produces. Neither requires touching the
architecture, the loss, or the training loop, which is the point: they
test whether the model's physical parameters carry more information
than the single readout Psi extracts from them, without adding a
single degree of freedom that could destabilize what is already learned
(the failure mode of all four rejected extensions in the paper).

1. FLUCTUATION-DISSIPATION VIOLATION, Delta_FDT(t)

   The Langevin thermostat injects noise sigma = sqrt(2 gamma T dt),
   which is the equilibrium fluctuation-dissipation relation. For a
   thermalized system, equipartition then requires <|p|^2>/d = T. The
   paper's own equipartition diagnostic found this violated, tried to
   restore it with colored noise, and found no gain. This module takes
   the opposite view: in non-equilibrium statistical mechanics the
   DEGREE of FDT violation is itself a physical observable, a measure of
   how far a driven system sits from equilibrium (the effective-
   temperature literature: Cugliandolo, Kurchan and Peliti, Phys. Rev. E
   55, 3898, 1997). A market approaching a crisis is a driven system
   leaving equilibrium. So the violation is used here as a candidate
   early-warning signal in its own right, not as a defect to repair.

       Delta_FDT(t) = | |p_N(t)|^2 / d  -  T(t) |

   with p_N the momentum after the final Verlet step and T the learned
   temperature, both read directly from the trained model.

2. KRAMERS RATE WITH ITS PREFACTORS

   The paper's Kramers diagnostic uses R ~ gamma^-1 exp(-DV/T), which
   drops every prefactor and assumes the high-friction regime throughout,
   while acknowledging that the learned gamma crosses the Kramers
   turnover. This module computes the rate properly for the quartic
   potential V(r) = a r^4 - b r^2:

       well minimum at r^2 = b/2a,  V''(min) = 4b   ->  omega_0 = 2 sqrt(b)
       barrier at r = 0,           |V''(0)| = 2b   ->  omega_b = sqrt(2b)
       barrier height              DV = b^2 / 4a

   and reports the spatial-diffusion rate (moderate to high friction,
   Kramers 1940), the energy-diffusion rate (low friction), a simple
   harmonic-mean bridge across the turnover, and the dimensionless
   friction Gamma/omega_b that says which regime each day is in. The
   friction Gamma is the CONTINUOUS-time coefficient: the discrete update
   p <- (1 - gamma) p corresponds to dp/dt = -(gamma/dt) p, so
   Gamma = gamma / DT with DT = 0.1.

   None of these is a rate in calendar units; all are rank-preserving
   proxies for the same reason the paper gives. But they are DIFFERENT
   rank proxies from the crude one, because the prefactor depends on
   (a, b, gamma) in a way that does not reduce to a monotone transform
   of gamma^-1 exp(-DV/T). Whether they rank crisis days better is
   therefore a real empirical question.

Reference for the rate formulas: Hanggi, Talkner and Borkovec, Reaction-
rate theory: fifty years after Kramers, Rev. Mod. Phys. 62, 251 (1990).
"""
from __future__ import annotations

import numpy as np
import torch

import main_v304_soft_labels as mvsl
from main_v304_soft_labels import CFG


def collect_physics_trajectory(models, X, device, latent_dim):
    """Same sequential, stateful pass as run_stateful_inference with
    collect_physics=True, plus the one thing that function does not
    collect: the final momentum p_N, needed for Delta_FDT. Returns
    ensemble-averaged arrays over days. TCN branch omitted on purpose
    (that flag is disabled in every reported result)."""
    x_t = torch.tensor(X, dtype=torch.float32).to(device)
    keys = ("psi", "gamma", "temp", "a", "b", "p2")
    acc = {k: [] for k in keys}
    for model in models:
        model.eval()
        with torch.no_grad():
            state = torch.zeros((1, latent_dim), device=device)
            run = {k: [] for k in keys}
            for t in range(x_t.shape[0]):
                psi_t, _, state = model(x_t[t:t + 1], s_initial=state)
                run["psi"].append(psi_t.cpu().item())
                run["gamma"].append(model._last_gamma.cpu().item())
                run["temp"].append(model._last_temp.cpu().item())
                run["a"].append(model._last_a.cpu().item())
                run["b"].append(model._last_b.cpu().item())
                run["p2"].append(float((model._last_p ** 2).sum().cpu().item()))
            for k in keys:
                acc[k].append(np.array(run[k]))
    return {k: np.mean(acc[k], axis=0) for k in keys}


def fdt_violation(p2: np.ndarray, temp: np.ndarray, d: int) -> np.ndarray:
    """|<p^2>/d - T|: zero for a thermalized system, positive either way
    the system departs from equipartition. Absolute value on purpose:
    a market can be hotter or colder than its own thermostat, and both
    are departures from equilibrium."""
    return np.abs(p2 / d - temp)


def kramers_rates(a: np.ndarray, b: np.ndarray, gamma: np.ndarray, temp: np.ndarray,
                  dt: float = CFG.DT) -> dict:
    """All four rate proxies plus the regime indicator, elementwise."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    gamma = np.asarray(gamma, float); temp = np.asarray(temp, float)
    eps = 1e-9
    dV = b ** 2 / (4.0 * a + eps)
    omega0 = 2.0 * np.sqrt(np.clip(b, eps, None))
    omegab = np.sqrt(2.0 * np.clip(b, eps, None))
    Gamma = gamma / dt                                # continuous friction
    boltz = np.exp(-dV / np.clip(temp, eps, None))

    crude = boltz / np.clip(gamma, eps, None)                         # what the paper uses
    spatial = (omega0 / (2 * np.pi)) * (np.sqrt(Gamma ** 2 / 4 + omegab ** 2) - Gamma / 2) / omegab * boltz
    energy = Gamma * (dV / np.clip(temp, eps, None)) * (omega0 / (2 * np.pi)) * boltz
    turnover = 1.0 / (1.0 / np.clip(spatial, eps, None) + 1.0 / np.clip(energy, eps, None))
    return {
        "crude": crude, "spatial": spatial, "energy": energy, "turnover": turnover,
        "dimless_friction": Gamma / omegab, "barrier": dV,
        "escape_prob_20d": 1.0 - np.exp(-turnover * 20.0),
    }
