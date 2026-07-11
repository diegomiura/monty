"""Multinomial logistic recalibration.

A LogisticRegression is fit on the log-probabilities of the model being
calibrated, using a chronological calibration slice that precedes the test
period. With C large this nests temperature scaling (single shared scale)
while also allowing per-class intercept shifts. Calibration is kept only
when it improves validation log loss (decided in the training pipeline,
never on the final holdout tournament).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-12


class MultinomialRecalibrator:
    def fit(self, p: np.ndarray, y: np.ndarray):
        self.lr_ = LogisticRegression(C=1.0, max_iter=1000)
        self.lr_.fit(np.log(np.clip(p, EPS, 1.0)), y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.lr_.predict_proba(np.log(np.clip(p, EPS, 1.0)))


class IdentityCalibrator:
    def fit(self, p: np.ndarray, y: np.ndarray):
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return p
