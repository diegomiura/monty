"""Closing-odds market baseline.

Bookmaker odds are the reference this model is measured against, never an
input to it (see ``market.use_market_features`` in ``config/clubs.yaml``,
off by default). Published decimal odds are not probabilities: their implied
probabilities sum to more than one, the excess being the bookmaker's margin.
Over the 5,320 priced Premier League matches the mean overround is 2.51%.

Removing that margin requires an assumption about how it is distributed
across the three outcomes, and the choice is not cosmetic — it moves the
benchmark the model is judged against.

``proportional``
    Divide every implied probability by their sum, assuming the margin is
    spread in proportion to probability. Simple, standard, and the default
    here. In practice bookmakers load proportionally *more* margin onto
    longshots (the favourite-longshot bias), so dividing uniformly leaves
    longshots too high and favourites too low.

``shin``
    Shin's model, which derives the margin from an assumed fraction ``z`` of
    insider money and removes proportionally more from longshots. Relative
    to proportional it therefore shifts probability *toward* favourites —
    verified on real EPL lines, where a 1.50/4.20/6.50 market moves the
    favourite from 0.630 to 0.643 and the longshot from 0.145 to 0.137.
    Usually the better-calibrated of the two.

Neither is "correct"; they are different assumptions, so both are reported
and the gap between them bounds how much the choice matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ODDS_COLUMNS = ["odds_close_a", "odds_close_draw", "odds_close_b"]


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """Raw (booked) probabilities 1/odds — these sum to > 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(odds > 0, 1.0 / odds, np.nan)


def overround(odds: np.ndarray) -> np.ndarray:
    return np.nansum(implied_probabilities(odds), axis=1)


def devig_proportional(odds: np.ndarray) -> np.ndarray:
    q = implied_probabilities(odds)
    return q / q.sum(axis=1, keepdims=True)


def _shin_z(q: np.ndarray, tol: float = 1e-10) -> float:
    """Insider fraction z for one match's booked probabilities."""
    total = q.sum()
    if not np.isfinite(total) or total <= 1.0 + tol:
        return 0.0

    def implied_sum(z: float) -> float:
        root = np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / total)
        return float(((root - z) / (2.0 * (1.0 - z))).sum())

    lo, hi = 0.0, 0.99
    if (implied_sum(lo) - 1.0) * (implied_sum(hi) - 1.0) > 0:
        return 0.0                      # no sign change; fall back to proportional
    return brentq(lambda z: implied_sum(z) - 1.0, lo, hi, xtol=tol)


def devig_shin(odds: np.ndarray) -> np.ndarray:
    """Shin de-vigging, row by row (z is a per-match quantity)."""
    q_all = implied_probabilities(odds)
    out = np.full_like(q_all, np.nan, dtype=float)
    for i, q in enumerate(q_all):
        if not np.isfinite(q).all():
            continue
        z = _shin_z(q)
        if z <= 0.0:
            out[i] = q / q.sum()
            continue
        total = q.sum()
        root = np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / total)
        p = (root - z) / (2.0 * (1.0 - z))
        out[i] = p / p.sum()            # guard against residual drift
    return out


DEVIG_METHODS = {"proportional": devig_proportional, "shin": devig_shin}


def market_probabilities(df: pd.DataFrame, method: str = "proportional") -> np.ndarray:
    """De-vigged (home, draw, away) probabilities; NaN rows where unpriced."""
    if method not in DEVIG_METHODS:
        raise ValueError(f"unknown devig method {method!r}; expected {sorted(DEVIG_METHODS)}")
    odds = df[ODDS_COLUMNS].to_numpy(dtype=float)
    priced = np.isfinite(odds).all(axis=1)
    out = np.full((len(df), 3), np.nan)
    if priced.any():
        out[priced] = DEVIG_METHODS[method](odds[priced])
    return out


def priced_mask(df: pd.DataFrame, source: str | None = None) -> np.ndarray:
    """Rows with a complete price, optionally restricted to one bookmaker.

    Restricting matters: over the EPL history Pinnacle prices 5,150 matches
    at 2.40% overround while Bet365 prices 170 at 5.65%. Pooling them mixes
    a sharp market with a soft one in the reference the model is scored
    against.
    """
    ok = np.isfinite(df[ODDS_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    if source is not None and "odds_source" in df.columns:
        ok &= (df["odds_source"] == source).to_numpy()
    return ok


class MarketBaseline:
    """The closing line, exposed with the same interface as a model.

    It is not fitted — the market needs no training data — so ``fit`` is a
    no-op that exists only so it can be dropped into the same comparison
    loop as everything else.
    """

    def __init__(self, method: str = "proportional"):
        self.method = method

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return market_probabilities(X, self.method)
