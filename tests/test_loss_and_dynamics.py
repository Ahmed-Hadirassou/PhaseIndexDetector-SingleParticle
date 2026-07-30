"""Tests for hooke_loss, verlet_step, and psi_to_regime.

hooke_loss and verlet_step need real torch tensors (they call
torch.where / .pow() / torch.randn_like internally), so these tests
instantiate the real torch backend declared in requirements.txt -- no
stubbing. They do not train or instantiate PhaseIndexDetector, so they
stay fast.
"""
import numpy as np
import pandas as pd
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

    def __call__(self, x_slow, z, x_macro_skip=None, vol_signal=None):
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



def test_verlet_step_colored_noise_injection_point():
    # The FDT-2 colored noise (CFG.FDT2_COLORED_NOISE) must enter the
    # momentum BEFORE the position/state updates, exactly like the white
    # noise -- so z_next and s_next both reflect the perturbed momentum.
    torch.manual_seed(0)
    z, p, s = torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4)
    f_ext = torch.zeros(2, 4)
    cn = torch.randn(2, 4) * 0.01
    dt = 0.1
    pot = _ConstantGradientPotential(0.5)
    z1, p1, s1 = mvsl.verlet_step(z, p, s, None, pot, f_ext, dt,
                                   gamma=0.0, phi=0.2, alpha=0.8, training=False)
    z2, p2, s2 = mvsl.verlet_step(z, p, s, None, pot, f_ext, dt,
                                   gamma=0.0, phi=0.2, alpha=0.8, training=False,
                                   colored_noise=cn)
    assert torch.allclose(p2, p1 + cn)
    assert torch.allclose(z2, z1 + dt * cn)
    assert torch.allclose(s2, s1 + dt * cn)


def test_verlet_step_colored_noise_none_is_identical():
    # colored_noise=None (the default) must leave the step bit-identical
    # to the pre-FDT2 behavior.
    torch.manual_seed(1)
    z, p, s = torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4)
    f_ext = torch.zeros(2, 4)
    pot = _ConstantGradientPotential(-0.3)
    out_default = mvsl.verlet_step(z, p, s, None, pot, f_ext, 0.1,
                                    gamma=0.1, phi=0.3, alpha=0.7, training=False)
    out_none = mvsl.verlet_step(z, p, s, None, pot, f_ext, 0.1,
                                 gamma=0.1, phi=0.3, alpha=0.7, training=False,
                                 colored_noise=None)
    for a, b in zip(out_default, out_none):
        assert torch.equal(a, b)


def test_tcn_causality_no_future_leakage():
    # The current-day (last position) output must not be reachable from
    # a change confined to the current day when inspecting an EARLIER
    # position's intermediate representation -- i.e. earlier positions
    # never see later days. Checked at the first conv layer directly.
    torch.manual_seed(0)
    tcn = mvsl.CausalTCN(fast_dim=6, channels=8, out_dim=4)
    tcn.eval()
    window = 5
    x_a = torch.randn(1, window, 6)
    x_b = x_a.clone()
    x_b[0, -1, :] = torch.randn(6) * 100  # wildly different LAST day only
    with torch.no_grad():
        h_a = tcn._causal_conv(tcn.conv1, tcn.project(x_a).transpose(1, 2))
        h_b = tcn._causal_conv(tcn.conv1, tcn.project(x_b).transpose(1, 2))
    assert torch.allclose(h_a[:, :, -2], h_b[:, :, -2], atol=1e-6), (
        "causality violated: an earlier position's representation depends "
        "on a later (more recent) day's value"
    )


def test_tcn_receptive_field_covers_full_window_no_gaps():
    # Every position in a TCN_WINDOW-length window must influence the
    # final output -- a naive dilation choice (dilation=3 for the second
    # of two kernel=2 layers) was tried and found to leave position 2 of
    # a 5-day window completely unreachable despite positions 0,1,3,4 all
    # reaching it. The (1,2,4) doubling schedule fixes this.
    torch.manual_seed(1)
    window = mvsl.CFG.TCN_WINDOW
    tcn = mvsl.CausalTCN(fast_dim=6, channels=8, out_dim=4)
    tcn.eval()
    x = torch.zeros(1, window, 6)
    with torch.no_grad():
        baseline = tcn(x)
    for pos in range(window):
        x_p = x.clone()
        x_p[0, pos, :] = 5.0
        with torch.no_grad():
            out_p = tcn(x_p)
        assert not torch.allclose(baseline, out_p), (
            f"window position {pos} never reaches the output -- a gap in "
            f"the receptive field"
        )


def test_tcn_gradients_flow_to_every_parameter():
    torch.manual_seed(2)
    tcn = mvsl.CausalTCN(fast_dim=6, channels=8, out_dim=4)
    x = torch.randn(3, 5, 6, requires_grad=False)
    out = tcn(x)
    out.sum().backward()
    for name, p in tcn.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert p.grad.norm().item() > 0, f"{name} received a zero gradient"


def test_dynamic_k_fn_backward_compatible():
    # persistence=None, or DYNAMIC_K_FN=False even with persistence
    # supplied, must reproduce hooke_loss's original behavior exactly.
    torch.manual_seed(0)
    psi = torch.rand(10)
    labels = torch.rand(10)
    original = mvsl.hooke_loss(psi, labels)

    mvsl.CFG.DYNAMIC_K_FN = False
    persistence = torch.tensor([0., 5., 10., 0., 3., 8., 0., 0., 1., 20.])
    with_persistence_flag_off = mvsl.hooke_loss(psi, labels, persistence=persistence)
    assert with_persistence_flag_off.item() == original.item()

    none_persistence = mvsl.hooke_loss(psi, labels, persistence=None)
    assert none_persistence.item() == original.item()


def test_dynamic_k_fn_ceiling_at_zero_persistence_is_unchanged():
    # At persistence=0, the effective k_FN ceiling must equal
    # ASYM_CRISIS_UNDER_K exactly, regardless of LAMBDA/TAU -- tanh(0)=0.
    torch.manual_seed(0)
    mvsl.CFG.DYNAMIC_K_FN = True
    psi = torch.tensor([0.9])   # far below target for label=1 -> triggers k_fn branch
    labels = torch.tensor([1.0])
    persistence_zero = torch.tensor([0.0])
    loss_dynamic = mvsl.hooke_loss(psi, labels, persistence=persistence_zero)

    mvsl.CFG.DYNAMIC_K_FN = False
    loss_static = mvsl.hooke_loss(psi, labels)
    assert abs(loss_dynamic.item() - loss_static.item()) < 1e-6


def test_credit_persistence_no_lookahead():
    rng = np.random.default_rng(3)
    n = 200
    dates = pd.bdate_range("2019-01-01", periods=n)
    hyg = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    ief = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    all_prices = pd.DataFrame({"HYG": hyg, "IEF": ief}, index=dates)

    full = mvsl.compute_credit_persistence(all_prices)
    for cutoff in [40, 90, 150, n - 1]:
        truncated = mvsl.compute_credit_persistence(all_prices.iloc[:cutoff + 1])
        assert np.allclose(full.iloc[:cutoff + 1].values, truncated.values), (
            f"look-ahead leakage detected: truncating at day {cutoff} changed a "
            f"value that should only depend on days <= {cutoff}"
        )


def test_split_data_persistence_alignment():
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.bdate_range("2015-01-01", periods=n)
    X = rng.normal(size=(n, 204)).astype(np.float32)
    y_hard = (rng.uniform(0, 1, n) < 0.15).astype(int)
    y_soft = y_hard.astype(np.float32)
    persistence = np.arange(n, dtype=np.float32)

    split = mvsl.split_data(dates, X, y_hard, y_soft, mvsl.CFG, persistence=persistence)
    mask_train = np.array(dates < pd.Timestamp("2020-01-01"))
    assert np.array_equal(split["persistence_train"], persistence[mask_train])
    assert split["persistence_train"].shape[0] == split["X_train"].shape[0]
