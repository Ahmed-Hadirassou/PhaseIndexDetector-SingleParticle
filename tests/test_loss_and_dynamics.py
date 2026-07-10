"""Tests for hooke_loss, verlet_step, and psi_to_regime.

hooke_loss and verlet_step need real torch tensors (they call
torch.where / .pow() / torch.randn_like internally), so these tests
instantiate the real torch backend declared in requirements.txt -- no
stubbing. They do not train or instantiate PhaseIndexDetector, so they
stay fast.
"""
import numpy as np
import pytest
import torch

import main_v304_soft_labels as mvsl


# ---------------------------------------------------------------------------
# hooke_loss
# ---------------------------------------------------------------------------

def test_hooke_loss_is_never_negative():
    torch.manual_seed(0)
    psi = torch.rand(64)
    labels = torch.rand(64)
    loss = mvsl.hooke_loss(psi, labels)
    assert loss.item() >= 0.0


def test_hooke_loss_is_zero_when_psi_sits_exactly_on_target():
    labels = torch.tensor([0.0, 0.5, 1.0])
    target = mvsl.CFG.ANCHOR_STABLE + (mvsl.CFG.ANCHOR_CRISIS - mvsl.CFG.ANCHOR_STABLE) * labels
    loss = mvsl.hooke_loss(target, labels)
    assert loss.item() == 0.0


def test_hooke_loss_penalizes_undershoot_harder_on_crisis_days():
    # Same magnitude of undershoot below the target, once on a pure crisis
    # day (label=1, stiffness should reach ASYM_CRISIS_UNDER_K=3x) and once
    # on a pure stable day (label=0, stiffness=1x). The crisis-day loss must
    # be strictly larger for an identical error magnitude.
    undershoot = 0.05
    target_crisis = mvsl.CFG.ANCHOR_STABLE + (mvsl.CFG.ANCHOR_CRISIS - mvsl.CFG.ANCHOR_STABLE) * 1.0
    target_stable = mvsl.CFG.ANCHOR_STABLE + (mvsl.CFG.ANCHOR_CRISIS - mvsl.CFG.ANCHOR_STABLE) * 0.0

    psi_crisis = torch.tensor([target_crisis - undershoot])
    psi_stable = torch.tensor([target_stable - undershoot])

    loss_crisis = mvsl.hooke_loss(psi_crisis, torch.tensor([1.0])).item()
    loss_stable = mvsl.hooke_loss(psi_stable, torch.tensor([0.0])).item()

    assert loss_crisis > loss_stable
    # The ratio should approach ASYM_CRISIS_UNDER_K since both start from the
    # same squared error and only the stiffness multiplier differs.
    assert loss_crisis / loss_stable == pytest.approx(mvsl.CFG.ASYM_CRISIS_UNDER_K, rel=1e-6)


def test_hooke_loss_does_not_extra_penalize_false_alarms_beyond_1x():
    # Overshooting the target on a stable day (false alarm) keeps stiffness
    # at ASYM_STABLE_OVER_K = 1.0, i.e. the same as a plain squared error.
    overshoot = 0.05
    target_stable = mvsl.CFG.ANCHOR_STABLE
    psi = torch.tensor([target_stable + overshoot])
    labels = torch.tensor([0.0])
    loss = mvsl.hooke_loss(psi, labels).item()
    assert loss == pytest.approx(overshoot ** 2, rel=1e-6)


# ---------------------------------------------------------------------------
# verlet_step
# ---------------------------------------------------------------------------

class _ConstantGradientPotential:
    """Deterministic stand-in for ParametricBistablePotential: returns a
    fixed gradient regardless of z, so the step's arithmetic can be checked
    by hand."""

    def __init__(self, grad_value):
        self.grad_value = grad_value

    def __call__(self, x_slow, z, vol_signal=None):
        return None, torch.full_like(z, self.grad_value)


def test_verlet_step_matches_hand_computed_update_with_zero_friction():
    # With gamma=0 and training=False, there is no thermal noise term, so
    # the update is exactly: p_next = p - dt*grad_V + dt*f_ext - phi*s
    #                          z_next = z + dt*p_next
    #                          s_next = alpha*s + dt*p_next
    z = torch.zeros(1, 2)
    p = torch.zeros(1, 2)
    s = torch.zeros(1, 2)
    x_slow = torch.zeros(1, 4)
    f_ext = torch.full((1, 2), 0.5)
    dt = 0.1
    grad_value = 0.2
    phi = 0.1
    alpha = 0.9

    z_next, p_next, s_next = mvsl.verlet_step(
        z, p, s, x_slow, _ConstantGradientPotential(grad_value), f_ext, dt,
        gamma=0.0, temp=0.0, phi=phi, alpha=alpha, training=False,
    )

    expected_p = (1.0 - 0.0) * p - dt * grad_value + dt * f_ext - phi * s
    expected_z = z + dt * expected_p
    expected_s = alpha * s + dt * expected_p

    assert torch.allclose(p_next, expected_p, atol=1e-6)
    assert torch.allclose(z_next, expected_z, atol=1e-6)
    assert torch.allclose(s_next, expected_s, atol=1e-6)


def test_verlet_step_preserves_shape():
    batch, latent = 5, 4
    z = torch.randn(batch, latent)
    p = torch.randn(batch, latent)
    s = torch.randn(batch, latent)
    x_slow = torch.randn(batch, 8)
    f_ext = torch.randn(batch, latent)

    z_next, p_next, s_next = mvsl.verlet_step(
        z, p, s, x_slow, _ConstantGradientPotential(0.1), f_ext, 0.1,
        gamma=0.1, temp=0.05, phi=0.1, alpha=0.8, training=True,
    )
    assert z_next.shape == (batch, latent)
    assert p_next.shape == (batch, latent)
    assert s_next.shape == (batch, latent)


# ---------------------------------------------------------------------------
# psi_to_regime
# ---------------------------------------------------------------------------

def test_psi_to_regime_buckets_match_cfg_bands():
    # CFG.REGIME_BANDS = [0.35, 0.50, 0.65] -> 4 buckets: Stable / Pre-alert
    # / Transition / Crisis.
    psi = np.array([0.10, 0.40, 0.55, 0.90])
    regimes = mvsl.psi_to_regime(psi)
    assert list(regimes) == [0, 1, 2, 3]


def test_psi_to_regime_boundary_is_inclusive():
    band = mvsl.CFG.REGIME_BANDS[0]
    psi = np.array([band - 1e-9, band, band + 1e-9])
    regimes = mvsl.psi_to_regime(psi)
    assert list(regimes) == [0, 1, 1]

