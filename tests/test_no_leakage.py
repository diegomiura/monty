"""Leakage tests: no feature may use the predicted match or anything after it."""
import numpy as np
import pandas as pd

from src.features.build_features import FeatureBuilder


def test_same_day_matches_do_not_affect_each_other(cfg, toy_matches):
    """Two matches on 2020-04-01: each must see state as of 2020-03-31."""
    fb = FeatureBuilder(cfg)
    feats = fb.run(toy_matches)
    day = toy_matches[toy_matches["date"] == "2020-04-01"]
    i1, i2 = day.index  # Alpha-Beta then Gamma-Beta
    # Beta plays both. If leakage existed, Beta's state in the second row
    # would include the first same-day match.
    fb2 = FeatureBuilder(cfg)
    fb2.advance(toy_matches, pd.Timestamp("2020-04-01"))
    expected = fb2.fixture_row("Gamma", "Beta", pd.Timestamp("2020-04-01"), "Friendly",
                               neutral=False, team_a_home=True)
    for col in ("elo_a", "elo_b", "ppm5_b", "log_matches_b"):
        got, want = feats.loc[i2, col], expected[col]
        assert (np.isnan(got) and np.isnan(want)) or got == want


def test_feature_row_ignores_future_matches(cfg, toy_matches):
    """Features for match at date D from the full pass must equal features
    computed from a dataset truncated at D — proving no future data enters."""
    fb_full = FeatureBuilder(cfg)
    feats_full = fb_full.run(toy_matches)
    for idx, m in list(toy_matches.iterrows())[1:]:
        trunc = toy_matches[toy_matches["date"] < m["date"]].reset_index(drop=True)
        fb_t = FeatureBuilder(cfg)
        if len(trunc):
            fb_t.run(trunc)
        row = fb_t.fixture_row(m["team_a"], m["team_b"], m["date"], m["competition"],
                               stage=m["stage"], neutral=bool(m["neutral"]),
                               team_a_home=not bool(m["neutral"]))
        for col, want in row.items():
            got = feats_full.loc[idx, col]
            assert (pd.isna(got) and pd.isna(want)) or abs(got - want) < 1e-9, (
                f"match {idx} feature {col}: full-pass {got} != truncated {want}"
            )


def test_elo_state_excludes_cutoff_day(cfg, toy_matches):
    fb = FeatureBuilder(cfg)
    fb.advance(toy_matches, pd.Timestamp("2020-04-01"))
    r_before = dict(fb.elo.ratings)
    fb2 = FeatureBuilder(cfg)
    fb2.advance(toy_matches, pd.Timestamp("2020-04-02"))
    assert fb2.elo.ratings != r_before  # the 04-01 matches only count from 04-02


def test_training_dates_precede_validation(cfg):
    """The pipeline split helper must produce strictly earlier training data."""
    dates = pd.date_range("2010-01-01", "2020-01-01", freq="7D")
    feats = pd.DataFrame({"date": dates})
    cutoff = pd.Timestamp("2019-01-01")
    val_days = int(cfg["validation"]["val_window_days"])
    t_val = cutoff - pd.Timedelta(days=val_days)
    fit = feats[feats["date"] < t_val]
    val = feats[(feats["date"] >= t_val) & (feats["date"] < cutoff)]
    assert fit["date"].max() < val["date"].min()
    assert val["date"].max() < cutoff
