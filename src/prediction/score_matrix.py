"""Exact-score probability matrix from expected-goal rates.

Base assumption: P(score a-b) = Pois(a; lambda_a) * Pois(b; lambda_b).
`apply_dixon_coles` adds the Dixon-Coles (1997) low-score dependence
correction; its rho parameter is estimated at train time and kept only when
it improves unseen chronological validation (see train_pipeline).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def build_matrix(lambda_a: float, lambda_b: float, max_goals: int = 10, max_tail: float = 1e-3):
    """Return (matrix, tail_mass). Grows the grid until tail mass is small."""
    for mg in (max_goals, max_goals + 5, max_goals + 10):
        g = np.arange(mg + 1)
        pa = stats.poisson.pmf(g, lambda_a)
        pb = stats.poisson.pmf(g, lambda_b)
        m = np.outer(pa, pb)
        tail = 1.0 - m.sum()
        if tail <= max_tail:
            return m, float(tail)
    return m, float(tail)


def apply_dixon_coles(m: np.ndarray, lambda_a: float, lambda_b: float, rho: float) -> np.ndarray:
    """Dixon-Coles tau correction on the four low-score cells.

    tau(0,0)=1-la*lb*rho, tau(0,1)=1+la*rho, tau(1,0)=1+lb*rho,
    tau(1,1)=1-rho. The four adjustments cancel exactly, so total mass is
    preserved (a rescale guards the rare case where a tau had to be clipped
    positive). rho < 0 shifts mass toward 0-0 and 1-1.
    """
    if rho == 0.0 or m.shape[0] < 2:
        return m
    out = m.copy()
    out[0, 0] *= max(1.0 - lambda_a * lambda_b * rho, 1e-9)
    out[0, 1] *= max(1.0 + lambda_a * rho, 1e-9)
    out[1, 0] *= max(1.0 + lambda_b * rho, 1e-9)
    out[1, 1] *= max(1.0 - rho, 1e-9)
    return out * (m.sum() / out.sum())


def outcome_probs(m: np.ndarray) -> np.ndarray:
    """[P(team_a win), P(draw), P(team_b win)], normalized over the grid."""
    win_a = np.tril(m, -1).sum()
    draw = np.trace(m)
    win_b = np.triu(m, 1).sum()
    tot = win_a + draw + win_b
    return np.array([win_a, draw, win_b]) / tot


def markets(m: np.ndarray, lambda_a: float, lambda_b: float) -> dict:
    grid = m / m.sum()
    n = grid.shape[0]
    idx_a, idx_b = np.indices((n, n))
    total = idx_a + idx_b
    p = {
        "both_teams_score": float(grid[1:, 1:].sum()),
        "over_2_5": float(grid[total > 2.5].sum()),
        "under_2_5": float(grid[total < 2.5].sum()),
        "team_a_clean_sheet": float(grid[:, 0].sum()),
        "team_b_clean_sheet": float(grid[0, :].sum()),
        "no_goal": float(grid[0, 0]),
    }
    # First scorer under the constant-intensity Poisson race approximation:
    # P(team_a first) = lambda_a/(lambda_a+lambda_b) * P(any goal). Documented
    # simplification: intensities assumed constant across the 90 minutes.
    p_any = 1.0 - p["no_goal"]
    lam_sum = max(lambda_a + lambda_b, 1e-9)
    p["team_a_scores_first"] = float(lambda_a / lam_sum * p_any)
    p["team_b_scores_first"] = float(lambda_b / lam_sum * p_any)
    return p


def top_scorelines(m: np.ndarray, k: int = 5) -> list[dict]:
    grid = m / m.sum()
    flat = [(f"{a}-{b}", float(grid[a, b])) for a in range(grid.shape[0]) for b in range(grid.shape[1])]
    flat.sort(key=lambda t: -t[1])
    return [{"score": s, "probability": round(p, 4)} for s, p in flat[:k]]
