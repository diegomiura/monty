"""Baseline models. All expose fit(X_df, y) / predict_proba(X_df) with the
shared 3-class encoding (0=team_a win, 1=draw, 2=team_b win)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class FrequencyBaseline:
    """Historical class frequencies from the training period."""

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.p_ = np.bincount(y, minlength=3) / len(y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile(self.p_, (len(X), 1))


class HigherEloBaseline:
    """Empirical W/D/L frequencies conditional on which side has higher
    (home-adjusted) Elo. Serves as the 'pick the higher-Elo team' baseline
    while still emitting honest probabilities."""

    def _bucket(self, X: pd.DataFrame) -> np.ndarray:
        return (X["elo_expected_a"].to_numpy() >= 0.5).astype(int)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        b = self._bucket(X)
        self.p_ = np.zeros((2, 3))
        for k in (0, 1):
            m = b == k
            self.p_[k] = (np.bincount(y[m], minlength=3) + 1) / (m.sum() + 3)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.p_[self._bucket(X)]


class EloProbabilityModel:
    """Multinomial logistic regression on the Elo expected score alone —
    the canonical 'Elo probability' component of the ensemble."""

    FEATS = ["elo_expected_a"]

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.lr_ = LogisticRegression(C=10.0, max_iter=1000)
        self.lr_.fit(X[self.FEATS].to_numpy(), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # sklearn orders columns by classes_, which is sorted [0, 1, 2] here
        return self.lr_.predict_proba(X[self.FEATS].to_numpy())
