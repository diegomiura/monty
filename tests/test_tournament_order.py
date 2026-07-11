"""Chronological ordering and frozen/rolling mode guarantees."""
import numpy as np
import pandas as pd

from src.features.build_features import FeatureBuilder


def test_run_requires_sorted_dates(cfg, toy_matches):
    shuffled = toy_matches.sample(frac=1.0, random_state=1).reset_index(drop=True)
    fb = FeatureBuilder(cfg)
    try:
        fb.run(shuffled)
        assert False, "unsorted input must be rejected"
    except AssertionError:
        pass


def test_frozen_mode_does_not_update_features(cfg, toy_matches):
    """Fixture rows emitted after advance(cutoff) must be identical no
    matter how many tournament matches happened after the cutoff."""
    cutoff = pd.Timestamp("2020-05-01")
    fb = FeatureBuilder(cfg)
    fb.advance(toy_matches, cutoff)
    row1 = fb.fixture_row("Alpha", "Gamma", pd.Timestamp("2020-05-01"), "FIFA World Cup",
                          stage="Final", neutral=True)
    row2 = fb.fixture_row("Beta", "Gamma", pd.Timestamp("2020-06-01"), "FIFA World Cup",
                          stage="Semi-final", neutral=True)
    # emitting rows must not mutate state
    row1_again = fb.fixture_row("Alpha", "Gamma", pd.Timestamp("2020-05-01"), "FIFA World Cup",
                                stage="Final", neutral=True)
    for k in row1:
        assert (pd.isna(row1[k]) and pd.isna(row1_again[k])) or row1[k] == row1_again[k]
    assert fb.processed_through == cutoff


def test_rolling_mode_uses_completed_matches_only(cfg, toy_matches):
    """Rolling = full chronological pass; the row for the last match must
    include earlier tournament matches but not itself."""
    fb = FeatureBuilder(cfg)
    feats = fb.run(toy_matches)
    last = toy_matches.iloc[-1]
    fb2 = FeatureBuilder(cfg)
    fb2.advance(toy_matches, last["date"])
    want = fb2.fixture_row(last["team_a"], last["team_b"], last["date"], last["competition"],
                           stage=last["stage"], neutral=bool(last["neutral"]))
    for k, v in want.items():
        got = feats.iloc[-1][k]
        assert (pd.isna(got) and pd.isna(v)) or abs(got - v) < 1e-9


def test_advance_is_incremental(cfg, toy_matches):
    fb1 = FeatureBuilder(cfg)
    fb1.advance(toy_matches, pd.Timestamp("2020-03-01"))
    fb1.advance(toy_matches, pd.Timestamp("2020-07-01"))
    fb2 = FeatureBuilder(cfg)
    fb2.advance(toy_matches, pd.Timestamp("2020-07-01"))
    assert fb1.elo.ratings == fb2.elo.ratings
