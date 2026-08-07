"""Smoke tests that run without the RadioMapSeer dataset.

These check the parts a reviewer is most likely to scrutinise: the link budget,
the outage labelling, the geometry of the physics features, and the routing
arithmetic. Run with:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from vaeevt import (
    Config,
    SNRScaler,
    compute_geometric_features,
    compute_permap_outage_mask,
    convert_gain_to_snr,
    fit_gpd_anchors,
    sharpen,
    stack_input_tensor,
)
from vaeevt.features import N_CHANNELS


@pytest.fixture
def cfg():
    return Config(map_size=64, los_ray_steps=100, shadow_steps=30)


@pytest.fixture
def scene(cfg):
    """A 64x64 scene with one rectangular building offset from the transmitter."""
    building = np.zeros((cfg.map_size, cfg.map_size), dtype=np.float32)
    building[28:36, 20:44] = 1.0
    return building, (10, 32)  # (tx_y, tx_x)


def test_noise_floor_matches_paper():
    cfg = Config()
    assert cfg.noise_floor_dbm == pytest.approx(-104.0, abs=1e-6)


def test_gain_to_snr_endpoints():
    cfg = Config()
    snr = convert_gain_to_snr(
        np.array([0.0, 255.0]), cfg.noise_floor_dbm, cfg.gain_min_dbm, cfg.gain_span_db
    )
    # -150 dBm and -30 dBm against a -104 dBm noise floor
    assert snr[0] == pytest.approx(-46.0)
    assert snr[1] == pytest.approx(74.0)


def test_scaler_roundtrip():
    scaler = SNRScaler(-46.0, 39.2)
    values = np.array([-46.0, 0.0, 39.2])
    assert np.allclose(scaler.inverse_transform(scaler.transform(values)), values,
                       atol=1e-3)
    # Values outside the percentile bounds saturate.
    assert scaler.transform(np.array([-100.0]))[0] == 0.0


def test_outage_mask_hits_target_fraction():
    rng = np.random.default_rng(0)
    snr = rng.normal(0.0, 10.0, size=(256, 256)).astype(np.float32)
    valid = np.ones_like(snr, dtype=bool)

    mask, threshold = compute_permap_outage_mask(snr, valid, 0.001,
                                                 rng=np.random.default_rng(1))
    target = int(np.ceil(valid.sum() * 0.001))
    assert mask.sum() == pytest.approx(target, rel=0.05)
    assert snr[mask].max() <= threshold + 1e-5


def test_outage_mask_strict_mode_matches_definition():
    """In strict mode the mask is exactly {gamma <= gamma_th}, per Eq. (2)."""
    snr = np.tile(np.arange(-50, 50, dtype=np.float32), (100, 1))
    valid = np.ones_like(snr, dtype=bool)
    mask, threshold = compute_permap_outage_mask(snr, valid, 0.10, strict=True)
    assert np.array_equal(mask, snr <= threshold)


def test_outage_mask_excludes_building_pixels():
    snr = np.full((64, 64), 10.0, dtype=np.float32)
    snr[:10, :10] = -60.0  # deep fade, but declared as building
    valid = np.ones_like(snr, dtype=bool)
    valid[:10, :10] = False
    mask, _ = compute_permap_outage_mask(snr, valid, 0.01)
    assert not mask[:10, :10].any()


def test_gpd_fit_recovers_known_parameters():
    from scipy.stats import genpareto

    xi_true, beta_true = 0.2, 5.0
    draws = genpareto.rvs(xi_true, loc=0, scale=beta_true, size=20000,
                          random_state=0)
    xi_hat, beta_hat, _p99 = fit_gpd_anchors(draws)
    assert xi_hat == pytest.approx(xi_true, abs=0.05)
    assert beta_hat == pytest.approx(beta_true, rel=0.1)


def test_feature_channels_shape_and_range(cfg, scene):
    building, (tx_y, tx_x) = scene
    feats = compute_geometric_features(tx_y, tx_x, building, cfg)
    X = stack_input_tensor(feats, threshold_norm=0.15)

    assert X.shape == (cfg.map_size, cfg.map_size, N_CHANNELS)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_all_channels_zero_inside_buildings(cfg, scene):
    building, (tx_y, tx_x) = scene
    feats = compute_geometric_features(tx_y, tx_x, building, cfg)
    inside = building > 0.5
    for name in ("los_mask", "los_dist", "nlos_shadow", "nlos_depth",
                 "shadow_edge", "dist_all"):
        assert np.allclose(feats[name][inside], 0.0), f"{name} leaks into buildings"


def test_outage_prior_floor_inside_buildings(cfg, scene):
    """The outage prior is zeroed on buildings and then clipped to
    [prior_clip_lo, prior_clip_hi], so building pixels carry the floor value
    rather than exactly zero.

    This is harmless -- every loss and metric masks building pixels via
    m(p) = 1 - B(p) -- but it is a real property of the published code and is
    asserted here so that it cannot change silently. See KNOWN_DEVIATIONS.md.
    """
    building, (tx_y, tx_x) = scene
    feats = compute_geometric_features(tx_y, tx_x, building, cfg)
    inside = building > 0.5
    assert np.allclose(feats["outage_prior"][inside], cfg.prior_clip_lo)


def test_los_mask_shadows_behind_building(cfg, scene):
    """Pixels directly behind the building must be NLoS; pixels beside it LoS."""
    building, (tx_y, tx_x) = scene
    feats = compute_geometric_features(tx_y, tx_x, building, cfg)
    los = feats["los_mask"]

    assert los[50, 32] < 0.5, "pixel in the geometric shadow should be NLoS"
    assert los[50, 5] > 0.5, "pixel with a clear path should be LoS"
    assert los[tx_y + 2, tx_x] > 0.5, "pixel next to the transmitter should be LoS"


def test_nlos_depth_increases_with_obstruction(cfg):
    """Two walls should register a greater penetration depth than one."""
    M = cfg.map_size
    one_wall = np.zeros((M, M), dtype=np.float32)
    one_wall[30:33, :] = 1.0
    two_walls = one_wall.copy()
    two_walls[40:43, :] = 1.0

    d1 = compute_geometric_features(5, M // 2, one_wall, cfg)["nlos_depth"]
    d2 = compute_geometric_features(5, M // 2, two_walls, cfg)["nlos_depth"]
    assert d2[55, M // 2] > d1[55, M // 2]


def test_routing_blend_is_a_convex_combination():
    mu = np.array([10.0, 10.0, 10.0])
    tail = np.array([-40.0, -40.0, -40.0])
    pi = np.array([0.0, 0.5, 1.0])
    pred = (1 - pi) * mu + pi * tail
    assert pred[0] == pytest.approx(10.0)
    assert pred[2] == pytest.approx(-40.0)
    assert min(mu[1], tail[1]) <= pred[1] <= max(mu[1], tail[1])


def test_sharpening_is_monotone_and_centred_at_tau():
    pi = np.linspace(0.0, 1.0, 101)
    out = sharpen(pi, tau=0.4, alpha=20.0)
    assert np.all(np.diff(out) > 0)
    assert sharpen(np.array([0.4]), 0.4, 20.0)[0] == pytest.approx(0.5)


@pytest.mark.slow
def test_model_forward_pass_shapes():
    """Requires TensorFlow. Marked slow: run with `pytest -m slow`."""
    from vaeevt import build_model

    cfg = Config(latent_dim=4)
    evt = {
        "empirical_xi": -0.35,
        "empirical_beta": 0.16,
        "empirical_outage_frac": 0.001,
        "avg_threshold_norm": 0.05,
    }
    model = build_model(evt, cfg)
    x = np.zeros((2, 256, 256, 10), dtype=np.float32)
    mu, log_sigma, y_tail, exc, pi = model.generate_maps(x)
    for tensor in (mu, log_sigma, y_tail, exc, pi):
        assert tuple(tensor.shape) == (2, 256, 256, 1)
    assert float(pi.numpy().max()) <= 1.0
    assert float(y_tail.numpy().min()) >= 0.0


# ----------------------------------------------------------------------
# Routing threshold t*
# ----------------------------------------------------------------------
def test_pinned_tau_is_used_verbatim():
    """A pinned tau must be passed through, not re-optimised."""
    from vaeevt.metrics import evaluate_predictions

    rng = np.random.default_rng(0)
    n = 20000
    is_out = (rng.random(n) < 0.01).astype(int)
    pi = np.clip(rng.beta(2, 50, n) + 0.6 * is_out, 0, 1)
    flat = {
        "y_db": rng.normal(0, 10, n),
        "mu_db": rng.normal(0, 10, n),
        "tail_db": rng.normal(-40, 2, n),
        "pi": pi,
        "is_outage": is_out,
        "is_los": rng.random(n) < 0.25,
    }
    out = evaluate_predictions(flat, alpha_sharp=20.0, tau=0.4192)
    assert out["tau"] == pytest.approx(0.4192)


def test_tau_choice_changes_outage_rmse_materially():
    """Regression guard for the Table I / Fig. 2 discrepancy.

    A low tau saturates the sharpened mask, collapsing the prediction onto the
    tail head. Because the tail head is accurate inside the outage region, the
    outage RMSE drops for reasons unrelated to model quality. Reporting two
    figures computed at different tau is therefore not comparable.
    """
    from vaeevt.metrics import blend, sharpen

    n = 5000
    mu = np.full(n, 0.0)      # bulk head: wrong inside outage
    tail = np.full(n, -40.0)  # tail head: correct inside outage
    truth = np.full(n, -40.0)
    pi = np.full(n, 0.3)      # ambiguous routing

    rmse_low = np.sqrt(np.mean((blend(mu, tail, sharpen(pi, 0.02, 20.0)) - truth) ** 2))
    rmse_high = np.sqrt(np.mean((blend(mu, tail, sharpen(pi, 0.99, 20.0)) - truth) ** 2))

    assert rmse_low < 1.0, "low tau should route to the tail head"
    assert rmse_high > 30.0, "high tau should route to the bulk head"
