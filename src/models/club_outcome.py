"""Three-class outcome models for club football.

Same symmetrization idea as :mod:`src.models.outcome` — every training row
also appears with the two clubs swapped, and prediction averages the two
views — but parameterised by feature list instead of importing the
international one.

Symmetrization matters slightly differently here. At an international
neutral venue the 'home' slot is arbitrary, and symmetrizing removes that
artifact. In a domestic league the home slot is never arbitrary: ``team_a``
is always the home side. Symmetrizing still helps, because it forces the
model to learn *home advantage* through the explicit ``home_edge`` feature
rather than absorbing it into every other coefficient, and it doubles the
effective sample. ``home_edge`` is encoded +1/-1 under mirroring so the two
views stay distinguishable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.club_features import CLUB_FEATURE_COLUMNS


def mirror_club(X: pd.DataFrame) -> pd.DataFrame:
    """Swap the two clubs' views of a fixture."""
    out = X.copy()
    for col in X.columns:
        if col.endswith("_a") and col not in ("elo_expected_a",):
            partner = col[:-2] + "_b"
            if partner in X.columns:
                out[col] = X[partner]
                out[partner] = X[col]
        elif col.endswith("_diff"):
            out[col] = -X[col]
    out["home_edge"] = -X["home_edge"]
    out["elo_expected_a"] = 1.0 - X["elo_expected_a"]
    return out


def mirror_labels(y: np.ndarray) -> np.ndarray:
    return 2 - y


class SymmetrizedClubModel:
    def __init__(self, base, features: list[str] | None = None):
        self.base = base
        self.features = list(features or CLUB_FEATURE_COLUMNS)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        Xa = X[self.features]
        Xs = pd.concat([Xa, mirror_club(Xa)], ignore_index=True)
        ys = np.concatenate([y, mirror_labels(y)])
        self.base.fit(Xs.to_numpy(dtype=float), ys)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xa = X[self.features]
        p1 = self.base.predict_proba(Xa.to_numpy(dtype=float))
        p2 = self.base.predict_proba(mirror_club(Xa).to_numpy(dtype=float))[:, ::-1]
        return (p1 + p2) / 2.0


def make_club_logistic(cfg: dict, features: list[str] | None = None) -> SymmetrizedClubModel:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=float(cfg["models"]["logit_C"]), max_iter=3000)),
    ])
    return SymmetrizedClubModel(pipe, features)


def make_club_gradient_boosting(cfg: dict, features: list[str] | None = None) -> SymmetrizedClubModel:
    m = cfg["models"]
    hgb = HistGradientBoostingClassifier(
        max_iter=int(m["hgb_max_iter"]),
        learning_rate=float(m["hgb_learning_rate"]),
        max_leaf_nodes=int(m["hgb_max_leaf_nodes"]),
        min_samples_leaf=int(m["hgb_min_samples_leaf"]),
        l2_regularization=float(m["hgb_l2"]),
        random_state=int(cfg.get("random_seed", 42)),
    )
    return SymmetrizedClubModel(hgb, features)
