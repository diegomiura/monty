"""Probabilistic and goal metrics. Outcome encoding: 0 = team_a win,
1 = draw, 2 = team_b win (column order of every probability matrix)."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _onehot(y: np.ndarray, k: int = 3) -> np.ndarray:
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def brier_multiclass(y: np.ndarray, p: np.ndarray) -> float:
    """Mean over matches of the sum of squared errors across all 3 outcome
    probabilities (as required by the spec)."""
    return float(np.mean(np.sum((p - _onehot(y)) ** 2, axis=1)))


def rps(y: np.ndarray, p: np.ndarray) -> float:
    """Ranked probability score over the ordered outcomes (win/draw/loss)."""
    cp = np.cumsum(p, axis=1)
    cy = np.cumsum(_onehot(y), axis=1)
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1) / (p.shape[1] - 1)))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(p.argmax(axis=1) == y))


def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    pred = p.argmax(axis=1)
    accs = [np.mean(pred[y == c] == c) for c in np.unique(y)]
    return float(np.mean(accs))


def top_two_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    top2 = np.argsort(-p, axis=1)[:, :2]
    return float(np.mean([y[i] in top2[i] for i in range(len(y))]))


def confusion(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    pred = p.argmax(axis=1)
    cm = np.zeros((3, 3), dtype=int)
    for t, q in zip(y, pred):
        cm[t, q] += 1
    return cm


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE of the predicted (argmax) class probability."""
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece, n = 0.0, len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.sum() / n * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def reliability_curve(y_bin: np.ndarray, p_bin: np.ndarray, bins: int = 10):
    """(mean predicted, observed frequency, count) per bin for one outcome."""
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_bin > lo) & (p_bin <= hi)
        if m.sum():
            rows.append((float(p_bin[m].mean()), float(y_bin[m].mean()), int(m.sum())))
    return rows


def result_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "log_loss": round(log_loss(y, p), 4),
        "brier": round(brier_multiclass(y, p), 4),
        "rps": round(rps(y, p), 4),
        "accuracy": round(accuracy(y, p), 4),
        "balanced_accuracy": round(balanced_accuracy(y, p), 4),
        "top_two_accuracy": round(top_two_accuracy(y, p), 4),
        "ece": round(expected_calibration_error(y, p), 4),
    }


def goal_metrics(ga: np.ndarray, gb: np.ndarray, la: np.ndarray, lb: np.ndarray) -> dict:
    total_pred, total_obs = la + lb, ga + gb
    exact = (np.round(la) == ga) & (np.round(lb) == gb)
    within1 = (np.abs(la - ga) <= 1) & (np.abs(lb - gb) <= 1)
    return {
        "mae_goals_a": round(float(np.mean(np.abs(la - ga))), 4),
        "mae_goals_b": round(float(np.mean(np.abs(lb - gb))), 4),
        "mae_total_goals": round(float(np.mean(np.abs(total_pred - total_obs))), 4),
        "rmse_goals": round(float(np.sqrt(np.mean((la - ga) ** 2 + (lb - gb) ** 2))), 4),
        "exact_score_rate_rounded": round(float(np.mean(exact)), 4),
        "one_goal_tolerance_rate": round(float(np.mean(within1)), 4),
        "mean_pred_total": round(float(np.mean(total_pred)), 3),
        "mean_obs_total": round(float(np.mean(total_obs)), 3),
    }
