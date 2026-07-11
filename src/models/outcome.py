"""Three-class outcome models on the full feature set.

Training rows are symmetrized: every match also appears with the two teams
swapped (suffix swap, diffs negated, home_edge sign flipped, elo_expected
mirrored, label mirrored). This removes any artifact from which side the
source lists first — important because at neutral venues the 'home' slot is
arbitrary. Prediction averages the original and mirrored views.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import FEATURE_COLUMNS


def mirror(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in X.columns:
        if col.endswith("_a") and col != "elo_expected_a":
            out[col] = X[col[:-2] + "_b"]
            out[col[:-2] + "_b"] = X[col]
        elif col.endswith("_diff"):
            out[col] = -X[col]
    out["home_edge"] = -X["home_edge"]
    out["elo_expected_a"] = 1.0 - X["elo_expected_a"]
    return out


def mirror_labels(y: np.ndarray) -> np.ndarray:
    return 2 - y  # 0<->2, draw stays 1


class SymmetrizedModel:
    """Wraps an sklearn classifier with symmetrized training + prediction."""

    def __init__(self, base):
        self.base = base

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        Xa = X[FEATURE_COLUMNS]
        Xs = pd.concat([Xa, mirror(Xa)], ignore_index=True)
        ys = np.concatenate([y, mirror_labels(y)])
        self.base.fit(Xs.to_numpy(), ys)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xa = X[FEATURE_COLUMNS]
        p1 = self.base.predict_proba(Xa.to_numpy())
        p2 = self.base.predict_proba(mirror(Xa).to_numpy())[:, ::-1]
        return (p1 + p2) / 2.0


def make_logistic(cfg: dict) -> SymmetrizedModel:
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=float(cfg["models"]["logit_C"]), max_iter=3000)),
        ]
    )
    return SymmetrizedModel(pipe)


def make_gradient_boosting(cfg: dict) -> SymmetrizedModel:
    m = cfg["models"]
    hgb = HistGradientBoostingClassifier(
        max_iter=int(m["hgb_max_iter"]),
        learning_rate=float(m["hgb_learning_rate"]),
        max_leaf_nodes=int(m["hgb_max_leaf_nodes"]),
        min_samples_leaf=int(m["hgb_min_samples_leaf"]),
        l2_regularization=float(m["hgb_l2"]),
        random_state=42,
    )
    return SymmetrizedModel(hgb)
