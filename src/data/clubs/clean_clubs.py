"""Build the canonical club match table from football-data.co.uk seasons.

Schema deliberately mirrors the international canonical table
(``data/processed/matches.csv``) so the existing Elo, rolling-form and
feature machinery can consume it unchanged, plus club-only columns:

    match_id, date, kickoff_time, team_a, team_b, goals_a_90, goals_b_90,
    competition, season, division, neutral, stage, group,
    goals_a_extra_time, goals_b_extra_time, penalty_goals_a,
    penalty_goals_b, went_to_extra_time, went_to_penalties,
    goals_90_confirmed, winner_90, match_status, source, retrieved_at,
    ht_goals_a, ht_goals_b, shots_a, shots_b, shots_target_a,
    shots_target_b, corners_a, corners_b, referee,
    odds_close_a, odds_close_draw, odds_close_b

Conventions and why they hold here
----------------------------------
* ``team_a`` is the home side and ``team_b`` the away side. Unlike the
  international table, ``neutral`` is **always False**: domestic league
  fixtures are played at the home club's ground. (Rare real exceptions —
  a stadium ban or a ground share — are not flagged in this source and are
  a documented limitation.)
* League play has no knockout ties, so extra-time and shootout columns exist
  for schema compatibility and are always zero/False. Nothing in this
  pipeline can therefore mistake a shootout goal for a real one.
* ``goals_90_confirmed`` is True for every row: full-time goals in this
  source are 90-minute goals by construction.
* Match statistics (shots, corners, cards) describe the match itself and are
  **post-match** information. They are carried on the row for use as lagged
  features about *earlier* matches only, and must never enter the feature row
  for the match they came from.
* Closing odds are carried for the market benchmark. They are not features
  unless a validation gate explicitly enables them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.clubs.football_data_uk import load_seasons, season_codes
from src.data.clubs.team_names_clubs import check_club_drift, describe_drift
from src.utils.dates import utc_now_iso
from src.utils.logging import get_logger

log = get_logger(__name__)

COMPETITION_BY_DIV = {
    "E0": "Premier League",
    "E1": "Championship",
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}


def _kickoff(date: pd.Series, time: pd.Series) -> pd.Series:
    """Combine date and (often absent) local kickoff time.

    Kickoff times exist only from 2019-20. Where absent the value is NaT, and
    the feature builder falls back to the conservative whole-day ordering the
    international pipeline already uses. They are local UK times; no timezone
    is attached because the source does not state one, and inventing one
    would be a false precision.
    """
    t = time.astype(str).str.strip()
    valid = t.str.match(r"^\d{1,2}:\d{2}$").fillna(False)
    out = pd.Series(pd.NaT, index=date.index, dtype="datetime64[ns]")
    if valid.any():
        combined = date[valid].dt.strftime("%Y-%m-%d") + " " + t[valid]
        out[valid] = pd.to_datetime(combined, format="%Y-%m-%d %H:%M", errors="coerce")
    return out


#: Closing-odds books in preference order: (source label, H/D/A columns).
#: Pinnacle first — the sharpest of the three and the widest covered here.
ODDS_BOOKS = [
    ("pinnacle", ("PSCH", "PSCD", "PSCA")),
    ("bet365", ("B365CH", "B365CD", "B365CA")),
    ("average", ("AvgCH", "AvgCD", "AvgCA")),
]


def _closing_odds(raw: pd.DataFrame) -> pd.DataFrame:
    """Pick one book per row and record which.

    Selection is **per row, not per column**: a book supplies all three of
    home/draw/away or none of them. Filling each column independently could
    take the home price from Pinnacle and the draw price from Bet365, giving
    a three-way "market" that no bookmaker ever offered and whose overround
    is meaningless.

    ``odds_source`` preserves provenance so the benchmark can be restricted
    to a single consistent market rather than silently mixing books.
    """
    n = len(raw)
    out = pd.DataFrame(
        {
            "odds_close_a": pd.Series([np.nan] * n, index=raw.index, dtype=float),
            "odds_close_draw": pd.Series([np.nan] * n, index=raw.index, dtype=float),
            "odds_close_b": pd.Series([np.nan] * n, index=raw.index, dtype=float),
            "odds_source": pd.Series([None] * n, index=raw.index, dtype=object),
        }
    )
    unfilled = pd.Series(True, index=raw.index)
    for label, (h, d, a) in ODDS_BOOKS:
        if not all(c in raw.columns for c in (h, d, a)):
            continue
        complete = raw[h].notna() & raw[d].notna() & raw[a].notna()
        take = unfilled & complete
        if not take.any():
            continue
        out.loc[take, "odds_close_a"] = raw.loc[take, h].astype(float)
        out.loc[take, "odds_close_draw"] = raw.loc[take, d].astype(float)
        out.loc[take, "odds_close_b"] = raw.loc[take, a].astype(float)
        out.loc[take, "odds_source"] = label
        unfilled &= ~take
    return out


def build_club_canonical(
    raw: pd.DataFrame,
    division: str = "E0",
    retrieved_at: str | None = None,
    allow_new_clubs: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Map parsed football-data.co.uk rows onto the canonical schema.

    Raises on any club name absent from :data:`KNOWN_CLUBS` unless
    ``allow_new_clubs=True``. An unrecognised name is far more often a
    respelling of an existing club than a genuine promotion, and a respelling
    silently creates a phantom team that starts from a default rating with no
    history — a corruption that produces plausible-looking output and would
    not surface in any downstream check. Refusing is the only safe default;
    review the reported names, then either pin the new club in KNOWN_CLUBS or
    pass the flag deliberately.
    """
    retrieved_at = retrieved_at or utc_now_iso()
    competition = COMPETITION_BY_DIV.get(division, division)

    unknown = check_club_drift(raw)
    if unknown and not allow_new_clubs:
        raise ValueError(
            f"{describe_drift(unknown)}\n"
            "Refusing to build: an unrecognised spelling of an existing club would "
            "create a phantom team with no history. Add genuinely new clubs to "
            "KNOWN_CLUBS in src/data/clubs/team_names_clubs.py, or pass "
            "allow_new_clubs=True if you have reviewed the names above."
        )
    if unknown:
        log.warning("proceeding with new club names (allow_new_clubs=True):\n%s",
                    describe_drift(unknown))

    df = pd.DataFrame(
        {
            "date": raw["Date"],
            "kickoff_time": _kickoff(raw["Date"], raw["Time"]),
            "team_a": raw["HomeTeam"],
            "team_b": raw["AwayTeam"],
            "goals_a_90": raw["FTHG"],
            "goals_b_90": raw["FTAG"],
            "competition": competition,
            "season": raw["season"],
            "division": raw["division"],
            "ht_goals_a": raw["HTHG"],
            "ht_goals_b": raw["HTAG"],
            "shots_a": raw["HS"],
            "shots_b": raw["AS"],
            "shots_target_a": raw["HST"],
            "shots_target_b": raw["AST"],
            "corners_a": raw["HC"],
            "corners_b": raw["AC"],
            "referee": raw["Referee"],
        }
    )
    df = pd.concat([df, _closing_odds(raw)], axis=1)

    incomplete = df["goals_a_90"].isna() | df["goals_b_90"].isna()
    if incomplete.any():
        log.warning("dropping %d row(s) with no full-time score", int(incomplete.sum()))
        df = df[~incomplete]

    df["goals_a_90"] = df["goals_a_90"].astype(int)
    df["goals_b_90"] = df["goals_b_90"].astype(int)

    # League play: no neutral venues, no extra time, no shootouts.
    df["neutral"] = False
    df["stage"] = ""
    df["group"] = ""
    df["goals_a_extra_time"] = 0
    df["goals_b_extra_time"] = 0
    df["penalty_goals_a"] = np.nan
    df["penalty_goals_b"] = np.nan
    df["went_to_extra_time"] = False
    df["went_to_penalties"] = False
    df["goals_90_confirmed"] = True
    df["match_status"] = "completed"
    df["source"] = "football_data_uk"
    df["retrieved_at"] = retrieved_at

    df["winner_90"] = np.select(
        [df["goals_a_90"] > df["goals_b_90"], df["goals_a_90"] < df["goals_b_90"]],
        ["team_a", "team_b"],
        default="draw",
    )
    df["match_id"] = (
        df["date"].dt.strftime("%Y-%m-%d") + "_" + df["team_a"] + "_" + df["team_b"]
    ).str.replace(" ", "-", regex=False)

    df = df.sort_values(["date", "kickoff_time", "team_a"], na_position="last")
    df = df.reset_index(drop=True)

    dupes = int(df["match_id"].duplicated().sum())
    if dupes:
        raise ValueError(f"{dupes} duplicate match_id(s) — the same fixture on the same date")

    audit = {
        "division": division,
        "competition": competition,
        "n_matches": len(df),
        "first_date": str(df["date"].min().date()),
        "last_date": str(df["date"].max().date()),
        "n_clubs": int(pd.concat([df["team_a"], df["team_b"]]).nunique()),
        "n_seasons": int(df["season"].nunique()),
        "unknown_club_names": unknown,
        "kickoff_time_coverage": round(float(df["kickoff_time"].notna().mean()), 4),
        "shots_coverage": round(float(df["shots_a"].notna().mean()), 4),
        "closing_odds_coverage": round(float(df["odds_close_a"].notna().mean()), 4),
        "closing_odds_by_source": {
            k: int(v) for k, v in df["odds_source"].value_counts().items()
        },
        "home_win_rate": round(float((df["winner_90"] == "team_a").mean()), 4),
        "draw_rate": round(float((df["winner_90"] == "draw").mean()), 4),
        "away_win_rate": round(float((df["winner_90"] == "team_b").mean()), 4),
        "goals_per_match": round(float((df["goals_a_90"] + df["goals_b_90"]).mean()), 4),
        "retrieved_at": retrieved_at,
        "source": "football_data_uk",
    }
    return df, audit


def refresh_club_dataset(
    first_year: int = 1993,
    last_year: int = 2025,
    division: str = "E0",
    config: dict | None = None,
    offline: bool = False,
    incomplete_ok: tuple[str, ...] = (),
    schedule: str = "double_round_robin",
    allow_new_clubs: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Download (or reuse cache), parse, and build the canonical club table."""
    codes = season_codes(first_year, last_year)
    raw, season_reports = load_seasons(
        codes, division, config, offline,
        incomplete_ok=incomplete_ok, schedule=schedule,
    )
    df, audit = build_club_canonical(
        raw, division, allow_new_clubs=allow_new_clubs
    )
    audit["seasons"] = season_reports
    return df, audit
