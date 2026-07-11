"""Per-team rolling form state, updated strictly chronologically."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MAX_HISTORY = 60  # matches kept per team; enough for windows + EW stats


class TeamState:
    __slots__ = ("history", "n_matches", "last_date")

    def __init__(self):
        # each entry: (date, gf, ga, points, competitive: bool)
        self.history: list[tuple] = []
        self.n_matches = 0
        self.last_date: pd.Timestamp | None = None

    def add(self, date, gf, ga, competitive):
        pts = 3.0 if gf > ga else (1.0 if gf == ga else 0.0)
        self.history.append((date, gf, ga, pts, competitive))
        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)
        self.n_matches += 1
        self.last_date = date


def window_stats(hist: list[tuple], n: int) -> dict[str, float]:
    """Stats over the last n matches; NaN when fewer than n played."""
    if len(hist) < n:
        return {
            "ppm": np.nan, "gf": np.nan, "ga": np.nan, "gd": np.nan,
            "win": np.nan, "draw": np.nan, "cs": np.nan, "fts": np.nan,
        }
    w = hist[-n:]
    gf = sum(m[1] for m in w) / n
    ga = sum(m[2] for m in w) / n
    return {
        "ppm": sum(m[3] for m in w) / n,
        "gf": gf,
        "ga": ga,
        "gd": gf - ga,
        "win": sum(m[3] == 3.0 for m in w) / n,
        "draw": sum(m[3] == 1.0 for m in w) / n,
        "cs": sum(m[2] == 0 for m in w) / n,
        "fts": sum(m[1] == 0 for m in w) / n,
    }


def competitive_window_stats(hist: list[tuple], n: int) -> dict[str, float]:
    comp = [m for m in hist if m[4]]
    if len(comp) < min(n, 5):  # need at least 5 competitive matches
        return {"ppm": np.nan, "gd": np.nan}
    w = comp[-n:]
    k = len(w)
    return {
        "ppm": sum(m[3] for m in w) / k,
        "gd": (sum(m[1] for m in w) - sum(m[2] for m in w)) / k,
    }


def ew_stats(hist: list[tuple], asof: pd.Timestamp, halflife_days: float) -> dict[str, float]:
    """Exponentially day-weighted goals for/against and points per match."""
    if len(hist) < 3:
        return {"gf": np.nan, "ga": np.nan, "ppm": np.nan}
    wsum = gf = ga = pts = 0.0
    for date, g_for, g_against, p, _ in hist[-40:]:
        age = (asof - date).days
        w = math.pow(0.5, age / halflife_days)
        wsum += w
        gf += w * g_for
        ga += w * g_against
        pts += w * p
    return {"gf": gf / wsum, "ga": ga / wsum, "ppm": pts / wsum}


def form_features(state: TeamState, asof: pd.Timestamp, windows: list[int], halflife: float) -> dict[str, float]:
    """All form features for one team as of (strictly before) `asof`."""
    h = state.history
    out: dict[str, float] = {}
    for n in windows:
        s = window_stats(h, n)
        out[f"ppm{n}"] = s["ppm"]
        out[f"gd{n}"] = s["gd"]
        if n == 10:
            out.update({f"{k}10": s[k] for k in ("gf", "ga", "win", "draw", "cs", "fts")})
    cw = competitive_window_stats(h, 10)
    out["comp_ppm10"] = cw["ppm"]
    out["comp_gd10"] = cw["gd"]
    ew = ew_stats(h, asof, halflife)
    out["ew_gf"] = ew["gf"]
    out["ew_ga"] = ew["ga"]
    out["ew_ppm"] = ew["ppm"]
    out["days_since_last"] = (
        min((asof - state.last_date).days, 365.0) if state.last_date is not None else np.nan
    )
    out["log_matches"] = math.log1p(state.n_matches)
    recent = h[-10:]
    out["comp_share10"] = (sum(m[4] for m in recent) / len(recent)) if len(recent) >= 5 else np.nan
    return out
