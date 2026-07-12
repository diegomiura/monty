"""Dixon-Coles correction: tau math, mass conservation, rho estimation,
and the validation gate's drop behavior."""
import numpy as np
import pandas as pd
import pytest

from src.models.poisson import (
    DC_RHO_GRID,
    PoissonGoalModel,
    estimate_dc_rho,
    scoreline_log_likelihood,
)
from src.prediction.score_matrix import apply_dixon_coles, build_matrix, outcome_probs


def _sample_dc_scores(rng, n: int, la: float, lb: float, rho: float):
    """Sample exact scores from a DC-corrected grid (ground truth known)."""
    m, _ = build_matrix(la, lb, 10)
    m = apply_dixon_coles(m, la, lb, rho)
    flat = (m / m.sum()).ravel()
    idx = rng.choice(len(flat), size=n, p=flat)
    g = m.shape[0]
    return (idx // g).astype(float), (idx % g).astype(float)


def test_rho_zero_is_identity():
    m, _ = build_matrix(1.4, 1.1)
    assert np.array_equal(apply_dixon_coles(m, 1.4, 1.1, 0.0), m)


def test_mass_preserved_and_only_low_cells_change():
    m, _ = build_matrix(1.4, 1.1)
    dc = apply_dixon_coles(m, 1.4, 1.1, -0.08)
    assert dc.sum() == pytest.approx(m.sum(), abs=1e-12)
    changed = np.argwhere(~np.isclose(dc, m, rtol=1e-12))
    assert set(map(tuple, changed)) <= {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_negative_rho_direction():
    """rho < 0 must raise P(0-0) and P(1-1), lower P(1-0) and P(0-1)."""
    m, _ = build_matrix(1.4, 1.1)
    dc = apply_dixon_coles(m, 1.4, 1.1, -0.08)
    assert dc[0, 0] > m[0, 0] and dc[1, 1] > m[1, 1]
    assert dc[1, 0] < m[1, 0] and dc[0, 1] < m[0, 1]


def test_outcome_probs_still_sum_to_one():
    m, _ = build_matrix(1.7, 0.9)
    p = outcome_probs(apply_dixon_coles(m, 1.7, 0.9, -0.1))
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert (p >= 0).all()


def test_estimator_recovers_negative_rho():
    rng = np.random.default_rng(7)
    n = 20000
    la, lb = np.full(n, 1.35), np.full(n, 1.10)
    ga, gb = _sample_dc_scores(rng, n, 1.35, 1.10, -0.10)
    rho_hat = estimate_dc_rho(ga, gb, la, lb)
    assert -0.14 < rho_hat < -0.06


def test_estimator_near_zero_for_independent_data():
    rng = np.random.default_rng(8)
    n = 20000
    ga = rng.poisson(1.35, n).astype(float)
    gb = rng.poisson(1.10, n).astype(float)
    la, lb = np.full(n, 1.35), np.full(n, 1.10)
    rho_hat = estimate_dc_rho(ga, gb, la, lb)
    assert abs(rho_hat) < 0.02


def test_grid_range():
    assert DC_RHO_GRID.min() >= -0.15 and DC_RHO_GRID.max() <= 0.0501


def test_scoreline_ll_prefers_true_model():
    rng = np.random.default_rng(9)
    n = 20000
    la, lb = np.full(n, 1.35), np.full(n, 1.10)
    ga, gb = _sample_dc_scores(rng, n, 1.35, 1.10, -0.10)
    assert scoreline_log_likelihood(ga, gb, la, lb, -0.10) > scoreline_log_likelihood(
        ga, gb, la, lb, 0.0
    )


def _synthetic_poisson_frame(rng, n: int, rho: float) -> pd.DataFrame:
    """Feature frame the Poisson model can consume, with scores drawn from a
    DC grid with known rho (features are noise, so lambdas ~ the mean)."""
    cols = {}
    for side in ("a", "b"):
        for f in ("elo", "ew_gf", "ew_ga", "ppm10", "gd10", "comp_ppm10", "cs10"):
            cols[f"{f}_{side}"] = rng.normal(size=n)
    cols["elo_a"] = rng.normal(1500, 100, n)
    cols["elo_b"] = rng.normal(1500, 100, n)
    cols["home_edge"] = np.zeros(n)
    for c in ("neutral", "knockout", "comp_friendly", "comp_qualifier",
              "comp_nations_league", "comp_continental_final", "comp_world_cup"):
        cols[c] = np.zeros(n)
    X = pd.DataFrame(cols)
    ga, gb = _sample_dc_scores(rng, n, 1.3, 1.3, rho)
    X["goals_a_90"], X["goals_b_90"] = ga, gb
    X["goals_90_confirmed"] = True
    X["outcome"] = np.select([ga > gb, ga < gb], [0, 2], default=1)
    return X


def test_gate_drops_wrong_rho():
    """Data generated with rho=+0.05; a model forced to rho=-0.10 must be
    rejected by the validation gate and reset to 0."""
    from src.models.train_pipeline import dixon_coles_gate

    rng = np.random.default_rng(10)
    X = _synthetic_poisson_frame(rng, 8000, rho=+0.05)
    model = PoissonGoalModel().fit(
        X, X["goals_a_90"].to_numpy(), X["goals_b_90"].to_numpy(), estimate_rho=False
    )
    model.rho_ = -0.10
    report = dixon_coles_gate(model, X)
    assert report["kept"] is False
    assert model.rho_ == 0.0


def test_gate_keeps_correct_rho():
    """Data generated with rho=-0.10 and a model that estimated close to it
    must pass the gate."""
    from src.models.train_pipeline import dixon_coles_gate

    rng = np.random.default_rng(11)
    fit = _synthetic_poisson_frame(rng, 20000, rho=-0.10)
    val = _synthetic_poisson_frame(rng, 20000, rho=-0.10)
    model = PoissonGoalModel().fit(
        fit, fit["goals_a_90"].to_numpy(), fit["goals_b_90"].to_numpy(), estimate_rho=True
    )
    assert -0.14 < model.rho_ < -0.06
    report = dixon_coles_gate(model, val)
    assert report["kept"] is True
    assert model.rho_ != 0.0
