"""Canonical club table: schema conventions, club-name drift detection,
and the guarantees the downstream model relies on."""
import numpy as np
import pandas as pd
import pytest

from src.data.clubs.clean_clubs import build_club_canonical
from src.data.clubs.team_names_clubs import (
    KNOWN_CLUBS,
    check_club_drift,
    describe_drift,
    resolve_club,
    validate_club,
)


def _raw(rows):
    """Minimal parsed-source frame (the shape select_columns produces)."""
    cols = ["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG",
            "HS", "AS", "HST", "AST", "HC", "AC", "Referee",
            "PSCH", "PSCD", "PSCA", "B365CH", "B365CD", "B365CA",
            "AvgCH", "AvgCD", "AvgCA", "season", "division"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df["Date"] = pd.to_datetime(df["Date"])
    df["season"] = df["season"].fillna("2025-26")
    df["division"] = df["division"].fillna("E0")
    return df[cols]


def test_league_rows_have_no_neutral_venue_or_extra_time():
    """League play: the ET/shootout columns exist for schema compatibility
    and must always be inert, so no club goal can ever be a shootout goal."""
    df, _ = build_club_canonical(_raw([
        {"Date": "2025-08-15", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 2, "FTAG": 1},
        {"Date": "2025-08-16", "HomeTeam": "Everton", "AwayTeam": "Fulham", "FTHG": 0, "FTAG": 0},
    ]))
    assert not df["neutral"].any()
    assert not df["went_to_extra_time"].any()
    assert not df["went_to_penalties"].any()
    assert (df["goals_a_extra_time"] == 0).all()
    assert (df["goals_b_extra_time"] == 0).all()
    assert df["penalty_goals_a"].isna().all()
    assert df["goals_90_confirmed"].all()


def test_winner_90_matches_the_goals():
    df, _ = build_club_canonical(_raw([
        {"Date": "2025-08-15", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 2, "FTAG": 1},
        {"Date": "2025-08-16", "HomeTeam": "Everton", "AwayTeam": "Fulham", "FTHG": 0, "FTAG": 0},
        {"Date": "2025-08-17", "HomeTeam": "Leeds", "AwayTeam": "Wolves", "FTHG": 1, "FTAG": 3},
    ]))
    assert list(df["winner_90"]) == ["team_a", "draw", "team_b"]


def test_home_side_is_team_a():
    df, _ = build_club_canonical(_raw([
        {"Date": "2025-08-15", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 2, "FTAG": 1},
    ]))
    assert df.loc[0, "team_a"] == "Arsenal"
    assert df.loc[0, "goals_a_90"] == 2


def test_kickoff_time_present_only_when_source_has_it():
    df, _ = build_club_canonical(_raw([
        {"Date": "2025-08-15", "Time": "15:00", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
         "FTHG": 1, "FTAG": 0},
        {"Date": "1995-08-15", "HomeTeam": "Everton", "AwayTeam": "Fulham", "FTHG": 1, "FTAG": 0},
    ]))
    got = df.set_index("team_a")["kickoff_time"]
    assert got["Arsenal"] == pd.Timestamp("2025-08-15 15:00")
    assert pd.isna(got["Everton"])


def test_rows_without_a_score_are_dropped():
    df, audit = build_club_canonical(_raw([
        {"Date": "2025-08-15", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 1, "FTAG": 0},
        {"Date": "2025-08-16", "HomeTeam": "Everton", "AwayTeam": "Fulham"},  # unplayed
    ]))
    assert len(df) == 1
    assert audit["n_matches"] == 1


def test_duplicate_fixture_raises():
    with pytest.raises(ValueError, match="duplicate match_id"):
        build_club_canonical(_raw([
            {"Date": "2025-08-15", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 1, "FTAG": 0},
            {"Date": "2025-08-15", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 2, "FTAG": 2},
        ]))


def test_closing_odds_fall_back_across_bookmakers():
    """Pinnacle preferred, then Bet365, then the market average."""
    df, _ = build_club_canonical(_raw([
        {"Date": "2025-08-15", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 1, "FTAG": 0,
         "PSCH": 1.5, "B365CH": 1.6, "AvgCH": 1.7},
        {"Date": "2025-08-16", "HomeTeam": "Everton", "AwayTeam": "Fulham", "FTHG": 1, "FTAG": 0,
         "B365CH": 2.6, "AvgCH": 2.7},
        {"Date": "2025-08-17", "HomeTeam": "Leeds", "AwayTeam": "Wolves", "FTHG": 1, "FTAG": 0,
         "AvgCH": 3.7},
        {"Date": "2025-08-18", "HomeTeam": "Burnley", "AwayTeam": "Brentford", "FTHG": 1, "FTAG": 0},
    ]))
    assert list(df["odds_close_a"])[:3] == [1.5, 2.6, 3.7]
    assert pd.isna(df["odds_close_a"].iloc[3])


def test_match_stats_are_carried_but_marked_post_match():
    """Shots describe the match itself; they must be present on the row (for
    later lagged use) and must never be confused with pre-match features."""
    df, _ = build_club_canonical(_raw([
        {"Date": "2025-08-15", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 1, "FTAG": 0,
         "HS": 14, "AS": 6, "HST": 7, "AST": 2},
    ]))
    assert df.loc[0, "shots_a"] == 14
    assert df.loc[0, "shots_target_b"] == 2


# --- club-name drift ------------------------------------------------------ #

def test_known_clubs_covers_the_verified_set():
    assert len(KNOWN_CLUBS) == 51
    for c in ("Arsenal", "Man United", "Nott'm Forest", "Sheffield Weds", "Luton"):
        assert c in KNOWN_CLUBS


def test_drift_is_empty_for_known_names():
    df = pd.DataFrame({"HomeTeam": ["Arsenal", "Man United"], "AwayTeam": ["Chelsea", "Leeds"]})
    assert check_club_drift(df) == []


def test_drift_flags_a_rename_distinctly_from_a_promotion():
    """A respelling of an existing club would silently create a phantom team
    with no history; a genuinely new club is fine. The report must separate
    the two."""
    df = pd.DataFrame({"HomeTeam": ["Man Utd", "Ipswich"], "AwayTeam": ["Chelsea", "Arsenal"]})
    unknown = check_club_drift(df)
    assert unknown == ["Man Utd"]                       # Ipswich is known
    assert "RENAME" in describe_drift(unknown)

    promoted = check_club_drift(
        pd.DataFrame({"HomeTeam": ["Wrexham"], "AwayTeam": ["Arsenal"]})
    )
    assert "newly promoted" in describe_drift(promoted)


def test_user_input_aliases_resolve():
    assert resolve_club("Manchester United") == "Man United"
    assert resolve_club("Spurs") == "Tottenham"
    assert resolve_club("Nottingham Forest") == "Nott'm Forest"
    assert resolve_club("Arsenal") == "Arsenal"
    assert validate_club("Man Utd") == "Man United"


def test_unknown_club_raises_with_suggestions():
    with pytest.raises(ValueError, match="Did you mean"):
        validate_club("Arsenl")
