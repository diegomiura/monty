"""Bracket parsing, winner resolution, and exact-enumeration propagation."""
import json

import pandas as pd
import pytest

from src.data.team_names import load_alias_table, make_resolver
from src.prediction.simulate_tournament import (
    forecast_bracket,
    load_bracket,
    node_result,
)

RESOLVER = make_resolver(load_alias_table())


def _node(num, date, rnd, s1, s2, score=None):
    return {"num": num, "date": pd.Timestamp(date), "round": rnd,
            "slot1": s1, "slot2": s2, "score": score}


def _toy_bracket():
    return [
        _node(1, "2026-07-14", "Semi-final", "Alpha", "Beta"),
        _node(2, "2026-07-15", "Semi-final", "Gamma", "Delta"),
        _node(3, "2026-07-18", "Match for third place", "L1", "L2"),
        _node(4, "2026-07-19", "Final", "W1", "W2"),
    ]


CUT = pd.Timestamp("2026-07-11")


def test_coin_flip_bracket_is_uniform():
    fc = forecast_bracket(_toy_bracket(), lambda a, b, n: 0.5, CUT)
    for row in fc["teams"]:
        assert row["p_champion"] == pytest.approx(0.25)
        assert row["p_reach_final"] == pytest.approx(0.5)
        assert row["p_third_place"] == pytest.approx(0.25)
        assert row["p_runner_up"] == pytest.approx(0.25)


def test_probabilities_are_coherent():
    def skewed(a, b, node):  # alphabetically earlier team is stronger
        return 0.7 if a < b else 0.3

    fc = forecast_bracket(_toy_bracket(), skewed, CUT)
    rows = {r["team"]: r for r in fc["teams"]}
    # reported values are rounded to 4 decimals, hence the tolerances
    assert sum(r["p_champion"] for r in fc["teams"]) == pytest.approx(1.0, abs=1e-3)
    assert sum(r["p_reach_final"] for r in fc["teams"]) == pytest.approx(2.0, abs=1e-3)
    assert sum(r["p_third_place"] for r in fc["teams"]) == pytest.approx(1.0, abs=1e-3)
    for r in fc["teams"]:
        assert r["p_champion"] <= r["p_reach_final"] + 2e-4
        assert r["p_champion"] + r["p_runner_up"] == pytest.approx(r["p_reach_final"], abs=5e-4)
    # Alpha beats everyone with 0.7; Gamma loses to everyone (alphabetical
    # order: Alpha < Beta < Delta < Gamma).
    assert rows["Alpha"]["p_champion"] == pytest.approx(0.7 * 0.7)
    assert rows["Gamma"]["p_champion"] == pytest.approx(0.3 * 0.3)


def test_deterministic_pair_prob_propagates():
    fc = forecast_bracket(_toy_bracket(), lambda a, b, n: 1.0, CUT)
    rows = {r["team"]: r for r in fc["teams"]}
    assert rows["Alpha"]["p_champion"] == 1.0
    assert rows["Gamma"]["p_runner_up"] == 1.0
    assert rows["Beta"]["p_third_place"] == 1.0


def test_completed_match_resolves_before_cutoff():
    nodes = _toy_bracket()
    nodes[0]["date"] = pd.Timestamp("2026-07-01")
    nodes[0]["score"] = {"ft": [0, 2]}  # Beta beat Alpha
    fc = forecast_bracket(nodes, lambda a, b, n: 0.5, CUT)
    rows = {r["team"]: r for r in fc["teams"]}
    assert rows["Alpha"]["p_champion"] == 0.0
    assert rows["Beta"]["p_reach_final"] == 1.0
    assert len(fc["pending"]) == 3


def test_completed_match_on_or_after_cutoff_is_repredicted():
    nodes = _toy_bracket()
    nodes[0]["score"] = {"ft": [0, 2]}  # result exists but is post-cutoff
    fc = forecast_bracket(nodes, lambda a, b, n: 0.5, CUT)
    rows = {r["team"]: r for r in fc["teams"]}
    assert rows["Alpha"]["p_champion"] == pytest.approx(0.25)
    assert len(fc["pending"]) == 4


def test_node_result_priority():
    # penalties override the level FT/ET scores
    n = _node(1, "2026-07-01", "Final", "A", "B",
              {"ft": [1, 1], "et": [1, 1], "p": [3, 4]})
    assert node_result(n) == ("B", "A")
    # extra time decides when unequal and no shootout
    n = _node(1, "2026-07-01", "Final", "A", "B", {"ft": [1, 1], "et": [2, 1]})
    assert node_result(n) == ("A", "B")
    # plain 90' result
    n = _node(1, "2026-07-01", "Final", "A", "B", {"ft": [3, 0]})
    assert node_result(n) == ("A", "B")
    # no score yet
    assert node_result(_node(1, "2026-07-01", "Final", "A", "B")) is None


def test_load_bracket_rejects_group_placeholders(tmp_path):
    doc = {"name": "test", "matches": [
        {"num": 1, "date": "2026-06-29", "round": "Round of 32", "team1": "1A", "team2": "2B"},
    ]}
    p = tmp_path / "wc.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="group placeholder"):
        load_bracket(p, RESOLVER)


def test_load_bracket_resolves_names_and_placeholders(tmp_path):
    doc = {"name": "test", "matches": [
        {"num": 1, "date": "2026-06-29", "round": "Round of 32", "team1": "USA",
         "team2": "Korea Republic", "score": {"ft": [1, 0]}},
        {"num": 2, "date": "2026-07-04", "round": "Round of 16", "team1": "W1", "team2": "France"},
        {"date": "2026-06-15", "round": "Matchday 1", "team1": "USA", "team2": "France",
         "group": "Group A"},  # group match: not a knockout node
    ]}
    p = tmp_path / "wc.json"
    p.write_text(json.dumps(doc))
    nodes = load_bracket(p, RESOLVER)
    assert len(nodes) == 2
    assert nodes[0]["slot1"] == "United States"
    assert nodes[0]["slot2"] == "South Korea"
    assert nodes[1]["slot1"] == "W1"  # placeholder untouched


def test_real_cached_bracket_parses():
    """The actual cached 2026 file: every slot is a known team or a W/L
    reference to an earlier match, and the bracket forecasts cleanly."""
    from src.utils.config import load_config, resolve

    cfg = load_config()
    path = resolve(cfg["data"]["raw_dir"]) / "worldcup_2026.json"
    if not path.exists():
        pytest.skip("no cached 2026 fixture file")
    aliases = load_alias_table(resolve(cfg["data"]["raw_dir"]) / "former_names.csv")
    nodes = load_bracket(path, make_resolver(aliases))
    assert [n["round"] for n in nodes].count("Final") == 1
    fc = forecast_bracket(nodes, lambda a, b, n: 0.5, pd.Timestamp("2026-07-11"))
    assert sum(r["p_champion"] for r in fc["teams"]) == pytest.approx(1.0, abs=1e-3)
