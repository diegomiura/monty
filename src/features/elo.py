"""Custom international Elo ratings.

Design (eloratings.net-inspired, parameters tunable via config and tuned
only on validation years <= 2013):

  expected_a = 1 / (1 + 10 ** (-(R_a - R_b + H * home_edge) / 400))
  R_a' = R_a + K * G * (S_a - expected_a)

  * K depends on competition class (World Cup highest, friendlies lowest).
  * G is a goal-margin multiplier: 1 (margin<=1), 1.5 (2), (11+margin)/8 (>=3).
  * H Elo points of home advantage for a non-neutral home side; host teams
    at their own tournament are by construction non-neutral in the data,
    so host advantage is carried by the same term.
  * Result S uses the 90-minute score only (extra time and shootouts never
    change S; a knockout match level after 90' counts as a draw = 0.5).
  * Time decay: a team inactive longer than `inactivity_gap_days` is shrunk
    toward the initial rating by `inactivity_shrink` before its next match.

Ratings are updated strictly chronologically; the rating stored in each
feature row is the value immediately BEFORE the match.
"""
from __future__ import annotations

import pandas as pd


class Elo:
    def __init__(self, cfg: dict):
        e = cfg["elo"]
        self.initial = float(e["initial_rating"])
        self.k_base = {k: float(v) for k, v in e["k_base"].items()}
        self.k_scale = float(e["k_scale"])
        self.home_adv = float(e["home_advantage"])
        self.margin = bool(e["margin_multiplier"])
        self.gap_days = int(e["inactivity_gap_days"])
        self.shrink = float(e["inactivity_shrink"])
        self.ratings: dict[str, float] = {}
        self.last_played: dict[str, pd.Timestamp] = {}

    def rating(self, team: str, date: pd.Timestamp | None = None) -> float:
        r = self.ratings.get(team, self.initial)
        if date is not None:
            last = self.last_played.get(team)
            if last is not None and (date - last).days > self.gap_days:
                r = self.initial + (r - self.initial) * (1.0 - self.shrink)
        return r

    def expected(self, r_a: float, r_b: float, home_edge: int) -> float:
        return 1.0 / (1.0 + 10.0 ** (-(r_a - r_b + self.home_adv * home_edge) / 400.0))

    @staticmethod
    def margin_multiplier(goal_diff: int) -> float:
        m = abs(goal_diff)
        if m <= 1:
            return 1.0
        if m == 2:
            return 1.5
        return (11.0 + m) / 8.0

    def update(
        self,
        date: pd.Timestamp,
        team_a: str,
        team_b: str,
        goals_a_90: int,
        goals_b_90: int,
        comp_class: str,
        home_edge: int,
    ) -> None:
        """Apply one completed match. home_edge: +1 team_a home, 0 neutral."""
        r_a = self.rating(team_a, date)
        r_b = self.rating(team_b, date)
        exp_a = self.expected(r_a, r_b, home_edge)
        s_a = 1.0 if goals_a_90 > goals_b_90 else (0.0 if goals_a_90 < goals_b_90 else 0.5)
        k = self.k_base.get(comp_class, self.k_base["other_tournament"]) * self.k_scale
        g = self.margin_multiplier(goals_a_90 - goals_b_90) if self.margin else 1.0
        delta = k * g * (s_a - exp_a)
        self.ratings[team_a] = r_a + delta
        self.ratings[team_b] = r_b - delta
        self.last_played[team_a] = date
        self.last_played[team_b] = date
