"""Reliability/calibration analysis helpers."""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import reliability_curve

OUTCOMES = ["team_a_win", "draw", "team_b_win"]


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    """Reliability data for each of the three outcomes."""
    out = {}
    for c, name in enumerate(OUTCOMES):
        out[name] = reliability_curve((y == c).astype(float), p[:, c], bins)
    return out
