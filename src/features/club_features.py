"""Chronological feature builder for club football.

Mirrors :mod:`src.features.build_features` and inherits its leakage
guarantee verbatim: matches are processed grouped by date in ascending
order, feature rows for every match on date D are emitted from state
containing only matches with date < D, and only then is D folded in. Two
matches on the same day therefore cannot see each other, and no match can
see itself. (Kickoff times exist from 2019-20 but not before; ordering
within a day is deliberately *not* used, so the rule is uniform across the
whole history rather than stricter in recent seasons only.)

Why this is a separate builder rather than a parameter on the international
one:

* the international feature set carries six competition one-hots and a
  knockout flag, all of which are constant for a single-league dataset —
  dead columns that would dilute permutation importance for no gain;
* clubs need features internationals have no analogue for (rolling shots
  and shots on target, promotion status);
* clubs need Elo *entry* logic — a newly promoted side has no top-flight
  history and must not start at the league average.

The genuinely leakage-critical machinery (:class:`~src.features.elo.Elo` and
:class:`~src.features.rolling.TeamState`) is imported and shared, not copied.

Post-match statistics
---------------------
``shots_a``/``shots_target_a`` and friends describe the match on whose row
they sit. They are read **only** inside :meth:`ClubFeatureBuilder._apply`,
which runs strictly after that match's feature row has been emitted, so they
can influence later matches and never their own. ``tests/test_club_features``
verifies this by corrupting a match's own statistics and asserting the whole
feature table is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.elo import Elo
from src.features.rolling import TeamState, form_features
from src.utils.logging import get_logger

log = get_logger(__name__)

#: Per-team form features carried over from the international builder.
TEAM_FEATS = [
    "elo", "ppm5", "gd5", "ppm10", "gd10", "gf10", "ga10", "win10", "draw10",
    "cs10", "fts10", "ppm20", "gd20", "ew_gf", "ew_ga", "ew_ppm",
    "days_since_last", "log_matches",
]
#: Club-only: rolling shot volume and accuracy, for and against.
SHOT_FEATS = ["shots_for10", "shots_against10", "sot_for10", "sot_against10", "sot_ratio10"]

DIFF_FEATS = ["elo", "ppm10", "gd10", "ew_gf", "ew_ga", "ew_ppm", "days_since_last"]
SHOT_DIFF_FEATS = ["shots_for10", "sot_for10"]

CONTEXT_FEATS = ["home_edge", "elo_expected_a", "is_new_a", "is_new_b"]

CLUB_FEATURE_COLUMNS = (
    [f"{f}_a" for f in TEAM_FEATS]
    + [f"{f}_b" for f in TEAM_FEATS]
    + [f"{f}_a" for f in SHOT_FEATS]
    + [f"{f}_b" for f in SHOT_FEATS]
    + [f"{f}_diff" for f in DIFF_FEATS]
    + [f"{f}_diff" for f in SHOT_DIFF_FEATS]
    + CONTEXT_FEATS
)

SHOT_WINDOW = 10


class ShotState:
    """Rolling post-match shot history for one club.

    Kept separate from :class:`TeamState` so the international path is
    untouched. Values may be NaN (seasons before 2000-01 have no shot data);
    missingness stays explicit rather than becoming a silent zero.
    """

    __slots__ = ("history",)

    def __init__(self):
        self.history: list[tuple[float, float, float, float]] = []

    def add(self, shots_for, shots_against, sot_for, sot_against) -> None:
        self.history.append((shots_for, shots_against, sot_for, sot_against))
        if len(self.history) > 40:
            self.history.pop(0)

    def features(self, window: int = SHOT_WINDOW) -> dict[str, float]:
        w = [h for h in self.history[-window:] if not any(pd.isna(v) for v in h)]
        if len(w) < window:
            return dict.fromkeys(SHOT_FEATS, np.nan)
        sf = float(np.mean([h[0] for h in w]))
        sa = float(np.mean([h[1] for h in w]))
        tf = float(np.mean([h[2] for h in w]))
        ta = float(np.mean([h[3] for h in w]))
        return {
            "shots_for10": sf,
            "shots_against10": sa,
            "sot_for10": tf,
            "sot_against10": ta,
            "sot_ratio10": tf / sf if sf > 0 else np.nan,
        }


class ClubFeatureBuilder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        fcfg = cfg.get("features", {})
        self.windows = list(fcfg.get("rolling_windows", [5, 10, 20]))
        self.halflife = float(fcfg.get("ew_halflife_days", 180))
        self.use_shots = bool(fcfg.get("use_shot_features", True))
        self.promoted_offset = float(cfg["elo"].get("promoted_elo_offset", 0.0))
        #: A club counts as currently in the competition if it played within
        #: this many days — one season plus a summer, so a side that sat out
        #: last year still anchors the mean but one gone for decades does not.
        self.active_window_days = int(cfg["elo"].get("active_window_days", 500))
        self.elo = Elo(cfg)
        self.teams: dict[str, TeamState] = {}
        self.shots: dict[str, ShotState] = {}
        self.seen: set[str] = set()
        self._matches_applied = 0

    # -- entry conditions -------------------------------------------------- #
    def initial_rating_for(self, club: str, date: pd.Timestamp | None = None) -> float:
        """Elo a club enters the competition with.

        A *promoted* side has no top-flight history. Starting it at the league
        average would rate it as a median team, which promoted clubs
        systematically are not, so it enters at the current mean of active
        ratings plus a negative offset.

        Two cases must not be treated as promotions:

        * **Founding members.** Every club in the dataset's first season is
          "new" only because the data starts there. Applying the promotion
          offset to them penalises each club in turn and, because every
          penalised entry drags the running mean down, compounds: on the real
          Premier League data this produced a 426-point spread across the 20
          clubs of 1993-08-14 purely from the order fixtures happened to be
          listed in. Before any match has been folded into state there is no
          league to be promoted into, so everyone starts level.
        * **Dormant clubs.** The mean must be taken over clubs currently in
          the competition. Sides relegated decades ago keep frozen ratings
          that would otherwise drag the anchor down; ``date`` restricts the
          mean to clubs seen recently.
        """
        if self._matches_applied == 0:
            return self.elo.initial
        ratings = self._active_ratings(date)
        if not ratings:
            return self.elo.initial
        return float(np.mean(ratings)) + self.promoted_offset

    def _active_ratings(self, date: pd.Timestamp | None) -> list[float]:
        """Ratings of clubs currently in the competition."""
        if date is None or not self.elo.last_played:
            return list(self.elo.ratings.values())
        cutoff = date - pd.Timedelta(days=self.active_window_days)
        active = [
            r for club, r in self.elo.ratings.items()
            if (last := self.elo.last_played.get(club)) is not None and last >= cutoff
        ]
        return active or list(self.elo.ratings.values())

    def _ensure(self, club: str, date: pd.Timestamp | None = None) -> None:
        if club not in self.seen:
            self.elo.ratings.setdefault(club, self.initial_rating_for(club, date))
            self.seen.add(club)
        self.teams.setdefault(club, TeamState())
        self.shots.setdefault(club, ShotState())

    # -- feature emission -------------------------------------------------- #
    def fixture_row(
        self,
        team_a: str,
        team_b: str,
        date: pd.Timestamp,
        team_a_home: bool = True,
    ) -> dict[str, float]:
        """Pre-match features from current state.

        State must already contain only matches strictly before ``date``.
        """
        new_a, new_b = team_a not in self.seen, team_b not in self.seen
        self._ensure(team_a, date)
        self._ensure(team_b, date)

        home_edge = 1 if team_a_home else 0
        r_a = self.elo.rating(team_a, date)
        r_b = self.elo.rating(team_b, date)

        row: dict[str, float] = {}
        fa = form_features(self.teams[team_a], date, self.windows, self.halflife)
        fb = form_features(self.teams[team_b], date, self.windows, self.halflife)
        for f in TEAM_FEATS:
            if f == "elo":
                continue
            row[f"{f}_a"] = fa.get(f, np.nan)
            row[f"{f}_b"] = fb.get(f, np.nan)
        row["elo_a"], row["elo_b"] = r_a, r_b

        sa = self.shots[team_a].features() if self.use_shots else dict.fromkeys(SHOT_FEATS, np.nan)
        sb = self.shots[team_b].features() if self.use_shots else dict.fromkeys(SHOT_FEATS, np.nan)
        for f in SHOT_FEATS:
            row[f"{f}_a"], row[f"{f}_b"] = sa[f], sb[f]

        for f in DIFF_FEATS + SHOT_DIFF_FEATS:
            row[f"{f}_diff"] = row[f"{f}_a"] - row[f"{f}_b"]

        row["home_edge"] = float(home_edge)
        row["elo_expected_a"] = self.elo.expected(r_a, r_b, home_edge)
        row["is_new_a"] = float(new_a)
        row["is_new_b"] = float(new_b)
        return row

    # -- state update ------------------------------------------------------ #
    def _apply(self, m) -> None:
        """Fold one completed match into state.

        The only place post-match statistics are read. Always called after
        the match's own feature row has been emitted.
        """
        self._ensure(m.team_a, m.date)
        self._ensure(m.team_b, m.date)
        self._matches_applied += 1
        self.elo.update(
            m.date, m.team_a, m.team_b, m.goals_a_90, m.goals_b_90,
            "league", 0 if bool(getattr(m, "neutral", False)) else 1,
        )
        self.teams[m.team_a].add(m.date, m.goals_a_90, m.goals_b_90, True)
        self.teams[m.team_b].add(m.date, m.goals_b_90, m.goals_a_90, True)
        if self.use_shots:
            sa = getattr(m, "shots_a", np.nan)
            sb = getattr(m, "shots_b", np.nan)
            ta = getattr(m, "shots_target_a", np.nan)
            tb = getattr(m, "shots_target_b", np.nan)
            self.shots[m.team_a].add(sa, sb, ta, tb)
            self.shots[m.team_b].add(sb, sa, tb, ta)

    def advance(self, matches: pd.DataFrame, through_date: pd.Timestamp) -> None:
        for m in matches[matches["date"] < through_date].itertuples():
            self._apply(m)

    def run(self, matches: pd.DataFrame, emit_from: pd.Timestamp | None = None) -> pd.DataFrame:
        """Full chronological pass; one feature row per match."""
        assert matches["date"].is_monotonic_increasing, "matches must be date-sorted"
        rows, ids = [], []
        for date, day in matches.groupby("date", sort=True):
            if emit_from is None or date >= emit_from:
                for m in day.itertuples():
                    row = self.fixture_row(
                        m.team_a, m.team_b, m.date,
                        team_a_home=not bool(getattr(m, "neutral", False)),
                    )
                    row["match_id"] = m.match_id
                    rows.append(row)
                    ids.append(m.Index)
            for m in day.itertuples():
                self._apply(m)
        return pd.DataFrame(rows, index=ids)


def build_club_feature_table(matches: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Feature table aligned with ``matches``, plus targets."""
    matches = matches.sort_values("date").reset_index(drop=True)
    feats = ClubFeatureBuilder(cfg).run(matches)
    feats["outcome"] = np.select(
        [matches["winner_90"] == "team_a", matches["winner_90"] == "draw"], [0, 1], default=2
    )
    for col in ("goals_a_90", "goals_b_90", "date", "season", "competition"):
        if col in matches.columns:
            feats[col] = matches[col]
    for col in ("odds_close_a", "odds_close_draw", "odds_close_b", "odds_source"):
        if col in matches.columns:
            feats[col] = matches[col]
    return feats
