"""Report figures (matplotlib, saved to reports/figures)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.calibration import OUTCOMES, reliability_table


def plot_reliability(y: np.ndarray, p: np.ndarray, title: str, path: Path) -> None:
    rel = reliability_table(y, p)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, name in zip(axes, OUTCOMES):
        rows = rel[name]
        if rows:
            xs, ys, ns = zip(*rows)
            ax.plot(xs, ys, "o-", label="model")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
        ax.set_title(name)
        ax.set_xlabel("predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("observed frequency")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_importance(importance: list[dict], title: str, path: Path) -> None:
    names = [d["feature"] for d in importance][::-1]
    vals = [list(d.values())[1] for d in importance][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(names) + 1.5))
    ax.barh(names, vals)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_goal_diagnostics(obs_total: np.ndarray, pred_total: np.ndarray, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.arange(0, 11) - 0.5
    axes[0].hist(obs_total, bins=bins, alpha=0.6, label="observed", density=True)
    axes[0].hist(pred_total, bins=bins, alpha=0.6, label="predicted E[goals]", density=True)
    axes[0].set_xlabel("total goals")
    axes[0].legend()
    axes[1].scatter(pred_total, obs_total, s=8, alpha=0.4)
    axes[1].set_xlabel("predicted total")
    axes[1].set_ylabel("observed total")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
