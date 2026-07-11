"""Chronological feature builder.

Leakage guarantee: matches are processed grouped by date in ascending order.
For every date D, feature rows for ALL matches on D are emitted from state
that contains only matches with date < D; only afterwards are D's results
folded into Elo/rolling state. Same-day matches therefore can never see
each other, and no match can ever see itself or the future. (The primary
dataset has no kickoff times; this whole-day rule is the conservative
ordering required by the spec.)

The same builder serves three uses:
  * full pass  -> training feature table (this is also "rolling mode" —
    every row reflects real-time pre-match knowledge);
  * advance(cutoff) + fixture_row(...) -> frozen-mode / CLI predictions
    where state is deliberately NOT advanced past a cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.elo import Elo
from src.features.rolling import TeamState, form_features
from src.features.tournament import competition_class, is_knockout_stage
from src.utils.logging import get_logger

log = get_logger(__name__)

TEAM_FEATS = [
    "elo", "ppm5", "gd5", "ppm10", "gd10", "gf10", "ga10", "win10", "draw10",
    "cs10", "fts10", "ppm20", "gd20", "comp_ppm10", "comp_gd10",
    "ew_gf", "ew_ga", "ew_ppm", "days_since_last", "log_matches", "comp_share10",
]
DIFF_FEATS = ["elo", "ppm10", "gd10", "ew_gf", "ew_ga", "ew_ppm", "comp_ppm10", "days_since_last"]
CONTEXT_FEATS = [
    "home_edge", "neutral", "elo_expected_a",
    "comp_friendly", "comp_qualifier", "comp_nations_league",
    "comp_continental_final", "comp_world_cup", "comp_other", "knockout",
]

FEATURE_COLUMNS = (
    [f"{f}_a" for f in TEAM_FEATS]
    + [f"{f}_b" for f in TEAM_FEATS]
    + [f"{f}_diff" for f in DIFF_FEATS]
    + CONTEXT_FEATS
)


class FeatureBuilder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.windows = list(cfg["features"]["rolling_windows"])
        self.halflife = float(cfg["features"]["ew_halflife_days"])
        self.elo = Elo(cfg)
        self.teams: dict[str, TeamState] = {}
        self.processed_through: pd.Timestamp | None = None

    def _state(self, team: str) -> TeamState:
        if team not in self.teams:
            self.teams[team] = TeamState()
        return self.teams[team]

    # ------------------------------------------------------------------ #
    def fixture_row(
        self,
        team_a: str,
        team_b: str,
        date: pd.Timestamp,
        competition: str,
        stage: str = "",
        neutral: bool = True,
        team_a_home: bool = False,
    ) -> dict[str, float]:
        """Pre-match features from current state (state must already be
        advanced to strictly before `date`)."""
        home_edge = 0 if neutral else (1 if team_a_home else 0)
        r_a = self.elo.rating(team_a, date)
        r_b = self.elo.rating(team_b, date)
        row: dict[str, float] = {}
        fa = form_features(self._state(team_a), date, self.windows, self.halflife)
        fb = form_features(self._state(team_b), date, self.windows, self.halflife)
        row.update({f"{k}_a": v for k, v in fa.items()})
        row.update({f"{k}_b": v for k, v in fb.items()})
        row["elo_a"], row["elo_b"] = r_a, r_b
        for f in DIFF_FEATS:
            row[f"{f}_diff"] = row[f"{f}_a"] - row[f"{f}_b"]
        row["home_edge"] = float(home_edge)
        row["neutral"] = float(neutral)
        row["elo_expected_a"] = self.elo.expected(r_a, r_b, home_edge)
        cc = competition_class(competition)
        for c in ("friendly", "qualifier", "nations_league", "continental_final", "world_cup"):
            row[f"comp_{c}"] = float(cc == c)
        row["comp_other"] = float(cc == "other_tournament")
        row["knockout"] = float(is_knockout_stage(stage) or stage == "knockout")
        return row

    def _apply_match(self, m) -> None:
        cc = competition_class(m.competition)
        home_edge = 0 if m.neutral else 1
        self.elo.update(m.date, m.team_a, m.team_b, m.goals_a_90, m.goals_b_90, cc, home_edge)
        competitive = cc != "friendly"
        self._state(m.team_a).add(m.date, m.goals_a_90, m.goals_b_90, competitive)
        self._state(m.team_b).add(m.date, m.goals_b_90, m.goals_a_90, competitive)

    def advance(self, matches: pd.DataFrame, through_date: pd.Timestamp) -> None:
        """Fold all matches with date < through_date into state (no rows)."""
        sub = matches[matches["date"] < through_date]
        if self.processed_through is not None:
            sub = sub[sub["date"] >= self.processed_through]
        for m in sub.itertuples():
            self._apply_match(m)
        self.processed_through = through_date

    def run(self, matches: pd.DataFrame, emit_from: pd.Timestamp | None = None) -> pd.DataFrame:
        """Full chronological pass; returns one feature row per match."""
        assert matches["date"].is_monotonic_increasing, "matches must be date-sorted"
        rows, ids = [], []
        for date, day in matches.groupby("date", sort=True):
            if emit_from is None or date >= emit_from:
                for m in day.itertuples():
                    row = self.fixture_row(
                        m.team_a, m.team_b, m.date, m.competition,
                        stage=getattr(m, "stage", "") or "",
                        neutral=bool(m.neutral),
                        team_a_home=not bool(m.neutral),
                    )
                    row["match_id"] = m.match_id
                    rows.append(row)
                    ids.append(m.Index)
            for m in day.itertuples():
                self._apply_match(m)
        self.processed_through = matches["date"].max() + pd.Timedelta(days=1)
        feats = pd.DataFrame(rows, index=ids)
        return feats


def build_feature_table(matches: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Feature table aligned with `matches` plus targets."""
    fb = FeatureBuilder(cfg)
    feats = fb.run(matches)
    feats["outcome"] = np.select(
        [matches["winner_90"] == "team_a", matches["winner_90"] == "draw"], [0, 1], default=2
    )
    feats["goals_a_90"] = matches["goals_a_90"]
    feats["goals_b_90"] = matches["goals_b_90"]
    feats["goals_90_confirmed"] = matches["goals_90_confirmed"]
    feats["date"] = matches["date"]
    feats["competition"] = matches["competition"]
    return feats
