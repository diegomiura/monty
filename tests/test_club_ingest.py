"""football-data.co.uk parsing: the three published malformations, the
double round-robin integrity check, and deterministic date parsing.

These use synthetic files mirroring the real defects (documented in
reports/CLUB_DATA_AUDIT.md), so the suite needs no network.
"""
import pandas as pd
import pytest

from src.data.clubs.football_data_uk import (
    expected_match_count,
    read_football_data_csv,
    season_code,
    season_codes,
    season_integrity,
    season_label,
    select_columns,
)

HEADER = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST"


def _round_robin_rows(teams, date="01/01/2020"):
    """Every ordered pair once — a complete season."""
    rows = []
    for h in teams:
        for a in teams:
            if h != a:
                rows.append(f"E0,{date},{h},{a},1,0,H,10,5,4,2")
    return rows


def _write(tmp_path, lines, name="E0.csv"):
    p = tmp_path / name
    p.write_text("\n".join(lines), encoding="latin-1")
    return p


def test_season_code_and_label_round_trip():
    assert season_code(1993) == "9394"
    assert season_code(2025) == "2526"
    assert season_code(1999) == "9900"
    assert season_label("9394") == "1993-94"
    assert season_label("2526") == "2025-26"
    assert season_codes(1993, 1995) == ["9394", "9495", "9596"]


def test_ragged_rows_are_kept_not_dropped(tmp_path):
    """The 2004-05 failure mode: trailing extra fields on some rows.

    pandas.read_csv raises here, and the usual escapes silently drop these
    rows. All four matches must survive.
    """
    lines = [HEADER] + [
        "E0,01/01/2020,Arsenal,Chelsea,1,0,H,10,5,4,2",
        "E0,02/01/2020,Everton,Fulham,2,2,D,8,9,3,3,,,,,",   # 5 extra fields
        "E0,03/01/2020,Leeds,Everton,0,1,A,7,11,2,5,,,,,",
        "E0,04/01/2020,Chelsea,Arsenal,3,1,H,14,6,6,1",
    ]
    df = read_football_data_csv(_write(tmp_path, lines))
    assert len(df) == 4
    assert list(df["HomeTeam"]) == ["Arsenal", "Everton", "Leeds", "Chelsea"]


def test_blank_padding_rows_are_dropped(tmp_path):
    """1993-94 ships 90 fully-empty trailing rows; 1999-00 ships 172."""
    lines = [HEADER, "E0,01/01/2020,Arsenal,Chelsea,1,0,H,10,5,4,2"] + [",,,,,,,,,,"] * 25
    df = read_football_data_csv(_write(tmp_path, lines))
    assert len(df) == 1


def test_utf8_bom_on_first_header_field(tmp_path):
    p = tmp_path / "bom.csv"
    p.write_bytes(("﻿" + HEADER + "\nE0,01/01/2020,Arsenal,Chelsea,1,0,H,10,5,4,2").encode("utf-8"))
    df = read_football_data_csv(p)
    assert "Div" in df.columns, f"BOM not stripped: {list(df.columns)[:2]}"


def test_dates_are_day_first_and_deterministic(tmp_path):
    """03/04/05 must be 3 April 2005, never 4 March. Both the 2- and
    4-digit year formats appear, sometimes in the same file."""
    lines = [HEADER,
             "E0,03/04/05,Arsenal,Chelsea,1,0,H,10,5,4,2",
             "E0,13/08/1993,Everton,Fulham,2,2,D,8,9,3,3",
             "E0,24/05/2026,Leeds,Chelsea,0,1,A,7,11,2,5"]
    out = select_columns(read_football_data_csv(_write(tmp_path, lines)))
    assert list(out["Date"]) == [
        pd.Timestamp("2005-04-03"), pd.Timestamp("1993-08-13"), pd.Timestamp("2026-05-24")
    ]


def test_unparseable_date_raises(tmp_path):
    lines = [HEADER, "E0,2020-01-01,Arsenal,Chelsea,1,0,H,10,5,4,2"]
    with pytest.raises(ValueError, match="matched neither"):
        select_columns(read_football_data_csv(_write(tmp_path, lines)))


def test_missing_era_columns_become_nan_not_zero(tmp_path):
    """Pre-2000 seasons have no shots; that must stay explicitly missing."""
    lines = ["Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR",
             "E0,01/01/1995,Arsenal,Chelsea,1,0,H"]
    out = select_columns(read_football_data_csv(_write(tmp_path, lines)))
    assert out["HS"].isna().all()
    assert out["PSCH"].isna().all()
    assert out["FTHG"].iloc[0] == 1


def test_integrity_accepts_a_complete_season(tmp_path):
    teams = ["A", "B", "C", "D"]
    df = select_columns(read_football_data_csv(_write(tmp_path, [HEADER] + _round_robin_rows(teams))))
    rep = season_integrity(df, "1920")
    assert rep["teams"] == 4
    assert rep["expected_matches"] == 12
    assert rep["actual_matches"] == 12
    assert rep["complete"]


def test_integrity_catches_silently_dropped_rows(tmp_path):
    """The failure this check exists for: a parser that quietly loses rows."""
    teams = ["A", "B", "C", "D"]
    rows = _round_robin_rows(teams)[:-3]        # lose 3 matches
    df = select_columns(read_football_data_csv(_write(tmp_path, [HEADER] + rows)))
    with pytest.raises(ValueError, match="3 missing"):
        season_integrity(df, "1920")


def test_integrity_catches_duplicate_fixtures(tmp_path):
    teams = ["A", "B", "C", "D"]
    rows = _round_robin_rows(teams)
    rows.append(rows[0])
    df = select_columns(read_football_data_csv(_write(tmp_path, [HEADER] + rows)))
    with pytest.raises(ValueError, match="duplicated fixture"):
        season_integrity(df, "1920")


def test_expected_count_only_defined_for_round_robin():
    assert expected_match_count(20) == 380
    assert expected_match_count(22) == 462
    assert expected_match_count(30, "unknown") is None
    with pytest.raises(ValueError, match="unknown schedule"):
        expected_match_count(20, "conference_playoff")


def test_non_round_robin_competition_skips_the_count_identity(tmp_path):
    """MLS 2025 played 540 matches among 30 teams; the round-robin identity
    would demand 870 and reject a perfectly good season. Such competitions
    must declare schedule='unknown' — duplicate checks still apply."""
    teams = [f"T{i}" for i in range(6)]
    rows = _round_robin_rows(teams)[:10]      # unbalanced, deliberately partial
    df = select_columns(read_football_data_csv(_write(tmp_path, [HEADER] + rows)))

    with pytest.raises(ValueError, match="missing"):
        season_integrity(df, "2526", schedule="double_round_robin")

    rep = season_integrity(df, "2526", schedule="unknown")
    assert rep["expected_matches"] is None
    assert rep["complete"] is None
    assert rep["actual_matches"] == 10


def test_unknown_schedule_still_rejects_duplicates(tmp_path):
    teams = ["A", "B", "C"]
    rows = _round_robin_rows(teams)
    rows.append(rows[0])
    df = select_columns(read_football_data_csv(_write(tmp_path, [HEADER] + rows)))
    with pytest.raises(ValueError, match="duplicated fixture"):
        season_integrity(df, "2526", schedule="unknown")


def test_incomplete_season_allowed_when_declared(tmp_path):
    """A season in progress is fine if the caller says so, but must still
    reject duplicates and overflow."""
    teams = ["A", "B", "C", "D"]
    df = select_columns(read_football_data_csv(
        _write(tmp_path, [HEADER] + _round_robin_rows(teams)[:6])))
    rep = season_integrity(df, "2526", complete=False)
    assert rep["actual_matches"] == 6
    assert not rep["complete"]
