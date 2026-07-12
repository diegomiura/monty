"""Poisson expected-goals model.

Every match becomes two team-rows (attacking perspective): the target is
the attacking team's 90-minute goals; features describe the attacker, the
defender and match context. A single sklearn PoissonRegressor is fit on the
stacked rows, then lambda_a / lambda_b are predicted per match.

Matches whose 90-minute score is unconfirmed (non-World-Cup shootout
matches where the recorded draw score may include extra time) are excluded
from training to avoid target contamination.

Dixon-Coles correction: `estimate_dc_rho` profile-maximizes the exact-score
log-likelihood over the DC dependence parameter given predicted lambdas
(only the four low-score cells contribute, and the DC pmf is exactly
normalized, so the profile reduces to the sum of log-tau terms). The model
stores `rho_`; train_pipeline keeps it only when it improves unseen
chronological validation, otherwise forces it back to 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ATT_FEATS = ["elo", "ew_gf", "ppm10", "gd10", "comp_ppm10"]
DEF_FEATS = ["elo", "ew_ga", "cs10"]
CTX_FEATS = [
    "neutral", "knockout", "comp_friendly", "comp_qualifier",
    "comp_nations_league", "comp_continental_final", "comp_world_cup",
]
LAMBDA_CLIP = (0.05, 6.0)
DC_RHO_GRID = np.arange(-0.15, 0.0501, 0.0025)


def _dc_log_tau(goals_a: np.ndarray, goals_b: np.ndarray,
                la: np.ndarray, lb: np.ndarray, rho: float) -> np.ndarray:
    """Per-match log tau term of the Dixon-Coles pmf (0 outside the four
    low-score cells). Returns -inf rows if rho is outside the valid range."""
    tau = np.ones(len(goals_a))
    m00 = (goals_a == 0) & (goals_b == 0)
    m01 = (goals_a == 0) & (goals_b == 1)
    m10 = (goals_a == 1) & (goals_b == 0)
    m11 = (goals_a == 1) & (goals_b == 1)
    tau[m00] = 1.0 - la[m00] * lb[m00] * rho
    tau[m01] = 1.0 + la[m01] * rho
    tau[m10] = 1.0 + lb[m10] * rho
    tau[m11] = 1.0 - rho
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tau > 0, np.log(np.maximum(tau, 1e-300)), -np.inf)


def estimate_dc_rho(goals_a: np.ndarray, goals_b: np.ndarray,
                    la: np.ndarray, lb: np.ndarray,
                    grid: np.ndarray = DC_RHO_GRID) -> float:
    """Profile-MLE of the DC rho given lambdas; grid values that make any
    observed low-score tau non-positive are invalid and skipped."""
    best_rho, best_ll = 0.0, -np.inf
    for rho in grid:
        ll = _dc_log_tau(goals_a, goals_b, la, lb, float(rho)).sum()
        if np.isfinite(ll) and ll > best_ll:
            best_rho, best_ll = float(rho), ll
    return best_rho


def scoreline_log_likelihood(goals_a: np.ndarray, goals_b: np.ndarray,
                             la: np.ndarray, lb: np.ndarray, rho: float = 0.0) -> float:
    """Mean log-probability of the observed exact 90' score under the
    (optionally DC-corrected) Poisson model. Higher is better."""
    from scipy import stats

    ll = stats.poisson.logpmf(goals_a, la) + stats.poisson.logpmf(goals_b, lb)
    if rho != 0.0:
        ll = ll + _dc_log_tau(goals_a, goals_b, la, lb, rho)
    return float(np.mean(ll))


def _team_rows(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Stack (team_a attacking, team_b attacking) rows; returns (rows_a, rows_b)."""

    def side(att: str, dfn: str, home_sign: float) -> np.ndarray:
        cols = [X[f"{f}_{att}"].to_numpy() for f in ATT_FEATS]
        cols += [X[f"{f}_{dfn}"].to_numpy() for f in DEF_FEATS]
        cols += [(X[f"{f}_{att}"] - X[f"{f}_{dfn}"]).to_numpy() for f in ("elo",)]
        cols += [home_sign * X["home_edge"].to_numpy()]
        cols += [X[c].to_numpy() for c in CTX_FEATS]
        return np.column_stack(cols)

    return side("a", "b", +1.0), side("b", "a", -1.0)


FEATURE_NAMES = (
    [f"att_{f}" for f in ATT_FEATS]
    + [f"def_{f}" for f in DEF_FEATS]
    + ["elo_diff", "home_edge"]
    + CTX_FEATS
)


class PoissonGoalModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.rho_ = 0.0

    def fit(self, X: pd.DataFrame, goals_a: np.ndarray, goals_b: np.ndarray,
            confirmed: np.ndarray | None = None, estimate_rho: bool = False):
        rows_a, rows_b = _team_rows(X)
        mask = np.ones(len(X), dtype=bool) if confirmed is None else confirmed.astype(bool)
        Z = np.vstack([rows_a[mask], rows_b[mask]])
        y = np.concatenate([goals_a[mask], goals_b[mask]])
        self.pipe_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("poisson", PoissonRegressor(alpha=self.alpha, max_iter=500)),
            ]
        )
        self.pipe_.fit(Z, y)
        if estimate_rho:
            la, lb = self.predict_lambdas(X)
            self.rho_ = estimate_dc_rho(
                goals_a[mask], goals_b[mask], la[mask], lb[mask]
            )
        else:
            self.rho_ = 0.0
        return self

    def predict_lambdas(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        rows_a, rows_b = _team_rows(X)
        la = np.clip(self.pipe_.predict(rows_a), *LAMBDA_CLIP)
        lb = np.clip(self.pipe_.predict(rows_b), *LAMBDA_CLIP)
        return la, lb

    def predict_proba(self, X: pd.DataFrame, max_goals: int = 10) -> np.ndarray:
        """3-class outcome probabilities via the (DC-corrected) score matrix."""
        from src.prediction.score_matrix import apply_dixon_coles, build_matrix, outcome_probs

        rho = float(getattr(self, "rho_", 0.0))
        la, lb = self.predict_lambdas(X)
        out = np.empty((len(X), 3))
        for i, (a, b) in enumerate(zip(la, lb)):
            m, _ = build_matrix(a, b, max_goals)
            out[i] = outcome_probs(apply_dixon_coles(m, a, b, rho))
        return out
