"""Staged knockout calculation.

1. 90-minute score distribution from the Poisson matrix.
2. P(extra time) = P(draw after 90).
3. Extra-time goals: independent Poisson at per-minute regulation intensity
   scaled by a historically-calibrated factor (bundle.et_stats, estimated
   from World Cup matches with confirmed ET splits, pre-cutoff only).
4. P(penalties) = P(draw at 90) * P(ET level | draw at 90).
5. Shootout: kept close to 50/50 — a small Elo-based tilt, hard-clipped
   (per spec: no reliable structured shootout data, so estimates shrink
   toward a coin flip).
"""
from __future__ import annotations

import numpy as np

from src.prediction.score_matrix import build_matrix, outcome_probs


def knockout_probs(
    lambda_a: float,
    lambda_b: float,
    p90: np.ndarray,
    elo_expected_a: float,
    et_stats: dict,
    ko_cfg: dict,
) -> dict:
    p_draw_90 = float(p90[1])

    minutes = float(ko_cfg.get("et_minutes", 30))
    factor = float(et_stats.get("et_rate_factor", 1.0))
    la_et = lambda_a * (minutes / 90.0) * factor
    lb_et = lambda_b * (minutes / 90.0) * factor
    m_et, _ = build_matrix(la_et, lb_et, max_goals=6)
    p_et = outcome_probs(m_et)  # outcome of the 30 ET minutes alone

    p_pens = p_draw_90 * float(p_et[1])

    lo, hi = ko_cfg.get("shootout_clip", [0.45, 0.55])
    w = float(ko_cfg.get("shootout_elo_weight", 0.10))
    p_shootout_a = float(np.clip(0.5 + w * (elo_expected_a - 0.5), lo, hi))

    p_adv_a = float(p90[0]) + p_draw_90 * float(p_et[0]) + p_pens * p_shootout_a
    p_adv_b = float(p90[2]) + p_draw_90 * float(p_et[2]) + p_pens * (1.0 - p_shootout_a)
    total = p_adv_a + p_adv_b  # == 1 up to float error
    return {
        "team_a_advances": round(p_adv_a / total, 4),
        "team_b_advances": round(p_adv_b / total, 4),
        "extra_time": round(p_draw_90, 4),
        "penalty_shootout": round(p_pens, 4),
    }
