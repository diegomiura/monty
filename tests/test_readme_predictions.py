"""README prediction section: fixture detection, rendering, splicing."""
import json

import pandas as pd

from src.visualization.readme_predictions import (
    _GENERATED_LINE,
    END_MARK,
    START_MARK,
    current_section,
    parse_kickoff_utc,
    render_section,
    section_equivalent,
    splice_readme,
    upcoming_fixtures,
)

WC_JSON = {
    "name": "World Cup 2026",
    "matches": [
        {  # played -> excluded
            "round": "Semi-final", "num": 101, "date": "2026-07-14",
            "time": "14:00 UTC-5", "team1": "France", "team2": "Spain",
            "score": {"ft": [1, 0]}, "ground": "Dallas",
        },
        {  # unplayed, both teams decided -> included
            "round": "Semi-final", "num": 102, "date": "2026-07-15",
            "time": "15:00 UTC-4", "team1": "England", "team2": "Argentina",
            "ground": "Atlanta",
        },
        {  # unresolved placeholder -> excluded
            "round": "Final", "num": 104, "date": "2026-07-19",
            "time": "15:00 UTC-4", "team1": "W101", "team2": "W102",
            "ground": "New York",
        },
    ],
}


def _fake_pred():
    return {
        "team_a": "England",
        "team_b": "Argentina",
        "expected_goals": {"team_a": 1.128, "team_b": 1.368},
        "outcome_90": {"team_a_win": 0.2371, "draw": 0.3508, "team_b_win": 0.4121},
        "most_likely_score": "1-1",
        "scorelines": [{"score": "1-1", "probability": 0.1667}],
        "additional_probabilities": {
            "both_teams_score": 0.5258, "over_2_5": 0.4303, "under_2_5": 0.5697,
            "team_a_clean_sheet": 0.2404, "team_b_clean_sheet": 0.343,
            "team_a_scores_first": 0.4026, "team_b_scores_first": 0.4882,
        },
        "knockout": {
            "team_a_advances": 0.4021, "team_b_advances": 0.5979,
            "extra_time": 0.3508, "penalty_shootout": 0.1927,
        },
        "confidence": {"label": "high", "score": 76},
        "model_information": {"model_version": "0.2.0", "latest_match_in_features": "2026-07-11"},
        "_fixture": {
            "round": "Semi-final", "ground": "Atlanta",
            "kickoff_utc": pd.Timestamp("2026-07-15T19:00:00Z"),
            "kickoff_known": True,
        },
    }


def test_upcoming_fixtures_filters_played_and_placeholders(tmp_path):
    path = tmp_path / "wc.json"
    path.write_text(json.dumps(WC_JSON))
    fx = upcoming_fixtures(path, resolver=lambda t: t)
    assert [(f["team_a"], f["team_b"]) for f in fx] == [("England", "Argentina")]


def test_kickoff_local_time_converts_to_utc():
    ts, known = parse_kickoff_utc("2026-07-14", "14:00 UTC-5")
    assert known and ts == pd.Timestamp("2026-07-14T19:00:00Z")
    ts, known = parse_kickoff_utc("2026-07-14", None)
    assert not known and ts == pd.Timestamp("2026-07-14T00:00:00Z")


def test_render_section_contains_all_stat_blocks():
    md = render_section([_fake_pred()], mode="rolling")
    for needle in (
        START_MARK, END_MARK, "England", "Argentina", "41.2%",
        "Expected goals", "Top scorelines", "Both teams to score",
        "Over 2.5 goals", "scores first", "Advancement", "Extra time",
        "Penalty shootout", "high", "1–1",
    ):
        assert needle in md, needle


def test_section_equivalent_ignores_only_the_timestamp():
    md = render_section([_fake_pred()], mode="rolling")
    a = current_section(md)
    stamp_only = _GENERATED_LINE.sub("Generated **some other time**.", a)
    assert section_equivalent(a, stamp_only)
    assert not section_equivalent(a, a.replace("41.2%", "50.0%"))
    assert not section_equivalent(a, None)


def test_splice_is_idempotent_and_preserves_rest():
    readme = "# Title\n\nintro\n\n---\n\n## Contents\n\n- stuff\n\n## Quick start\n"
    s1 = render_section([_fake_pred()], mode="rolling")
    once = splice_readme(readme, s1)
    assert once.count(START_MARK) == 1 and "## Quick start" in once
    # regenerating replaces the block instead of stacking a second copy
    s2 = s1.replace("41.2%", "43.0%")
    twice = splice_readme(once, s2)
    assert twice.count(START_MARK) == 1
    assert "43.0%" in twice and "41.2%" not in twice
    assert "# Title" in twice and "## Quick start" in twice
