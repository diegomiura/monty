"""Extended World Cup backtest reporting: breakdowns + figures."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import result_metrics
from src.utils.config import resolve
from src.visualization.plots import plot_goal_diagnostics, plot_reliability


def breakdowns(wc: pd.DataFrame, y: np.ndarray, p: np.ndarray, X: pd.DataFrame) -> dict:
    """Metric breakdowns by Elo-difference bucket and favorite probability."""
    out = {}
    elo_diff = X["elo_diff"].to_numpy()
    for name, mask in {
        "elo_gap_small(<100)": np.abs(elo_diff) < 100,
        "elo_gap_medium(100-250)": (np.abs(elo_diff) >= 100) & (np.abs(elo_diff) < 250),
        "elo_gap_large(>=250)": np.abs(elo_diff) >= 250,
    }.items():
        if mask.sum() >= 5:
            out[name] = result_metrics(y[mask], p[mask])
    fav = p[:, [0, 2]].max(axis=1)
    for name, mask in {
        "favorite<50%": fav < 0.5,
        "favorite50-70%": (fav >= 0.5) & (fav < 0.7),
        "favorite>=70%": fav >= 0.7,
    }.items():
        if mask.sum() >= 5:
            out[name] = result_metrics(y[mask], p[mask])
    return out


def figures(year: int, mode: str, y: np.ndarray, p: np.ndarray,
            obs_total: np.ndarray, pred_total: np.ndarray) -> None:
    fig_dir = resolve("reports/figures")
    plot_reliability(y, p, f"World Cup {year} ({mode}) reliability", fig_dir / f"wc{year}_{mode}_reliability.png")
    plot_goal_diagnostics(obs_total, pred_total, f"World Cup {year} ({mode}) goals",
                          fig_dir / f"wc{year}_{mode}_goals.png")
