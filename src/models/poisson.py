"""Poisson expected-goals model.

Every match becomes two team-rows (attacking perspective): the target is
the attacking team's 90-minute goals; features describe the attacker, the
defender and match context. A single sklearn PoissonRegressor is fit on the
stacked rows, then lambda_a / lambda_b are predicted per match.

Matches whose 90-minute score is unconfirmed (non-World-Cup shootout
matches where the recorded draw score may include extra time) are excluded
from training to avoid target contamination.
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

    def fit(self, X: pd.DataFrame, goals_a: np.ndarray, goals_b: np.ndarray,
            confirmed: np.ndarray | None = None):
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
        return self

    def predict_lambdas(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        rows_a, rows_b = _team_rows(X)
        la = np.clip(self.pipe_.predict(rows_a), *LAMBDA_CLIP)
        lb = np.clip(self.pipe_.predict(rows_b), *LAMBDA_CLIP)
        return la, lb

    def predict_proba(self, X: pd.DataFrame, max_goals: int = 10) -> np.ndarray:
        """3-class outcome probabilities via the exact-score matrix."""
        from src.prediction.score_matrix import build_matrix, outcome_probs

        la, lb = self.predict_lambdas(X)
        out = np.empty((len(X), 3))
        for i, (a, b) in enumerate(zip(la, lb)):
            m, _ = build_matrix(a, b, max_goals)
            out[i] = outcome_probs(m)
        return out
