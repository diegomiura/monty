"""Ensemble: nonnegative weights over component probability vectors,
summing to 1, selected by grid search on validation log loss only."""
from __future__ import annotations

from itertools import product

import numpy as np

from src.evaluation.metrics import log_loss


def weight_grid(n_components: int, step: float = 0.05):
    ticks = int(round(1.0 / step))
    for combo in product(range(ticks + 1), repeat=n_components - 1):
        if sum(combo) <= ticks:
            yield np.array(list(combo) + [ticks - sum(combo)], dtype=float) / ticks


def select_weights(prob_list: list[np.ndarray], y_val: np.ndarray, step: float = 0.05):
    """Return (weights, val_log_loss)."""
    P = np.stack(prob_list)  # (k, n, 3)
    best_w, best_ll = None, np.inf
    for w in weight_grid(len(prob_list), step):
        ll = log_loss(y_val, np.tensordot(w, P, axes=1))
        if ll < best_ll - 1e-9:
            best_ll, best_w = ll, w
    return best_w, float(best_ll)


def blend(prob_list: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, np.stack(prob_list), axes=1)
