"""Extra-time and penalty separation guarantees."""
import numpy as np
import pandas as pd

from src.features.build_features import FeatureBuilder


def test_et_goals_not_in_90min_score(toy_matches):
    m = toy_matches.iloc[5]  # Final decided in ET: 1-1 (90'), 2-1 (ET)
    assert m["goals_a_90"] == 1 and m["goals_b_90"] == 1
    assert m["goals_a_extra_time"] == 1
    assert m["winner_90"] == "draw"


def test_penalty_goals_not_normal_goals(toy_matches):
    m = toy_matches.iloc[6]  # shootout 0-0, pens 3-4
    assert m["goals_a_90"] == 0 and m["goals_b_90"] == 0
    assert m["penalty_goals_a"] == 3.0 and m["penalty_goals_b"] == 4.0
    assert m["winner_90"] == "draw"


def test_elo_uses_90min_result_for_et_match(cfg, toy_matches):
    """The ET-decided final must count as a DRAW for Elo (1-1 at 90')."""
    fb = FeatureBuilder(cfg)
    fb.advance(toy_matches, pd.Timestamp("2020-05-01"))
    before_a = fb.elo.rating("Alpha")
    before_g = fb.elo.rating("Gamma")
    fb.advance(toy_matches, pd.Timestamp("2020-05-02"))
    after_a = fb.elo.rating("Alpha")
    after_g = fb.elo.rating("Gamma")
    exp_a = fb.elo.expected(before_a, before_g, 0)
    # draw: winner of expectation loses points
    if exp_a > 0.5:
        assert after_a < before_a and after_g > before_g
    else:
        assert after_a > before_a


def test_clean_build_separates_wc_et(cfg):
    """Integration check on the real canonical dataset when present."""
    from pathlib import Path

    from src.utils.config import resolve

    path = resolve(cfg["data"]["processed_dir"]) / "matches.csv"
    if not path.exists():
        import pytest

        pytest.skip("canonical dataset not built")
    df = pd.read_csv(path, low_memory=False)
    et = df[df["went_to_extra_time"] & df["goals_90_confirmed"]]
    if len(et):
        # every confirmed ET match must be a draw at 90 minutes
        assert (et["goals_a_90"] == et["goals_b_90"]).all()
    pens = df[df["went_to_penalties"] & df["goals_90_confirmed"] & df["went_to_extra_time"]]
    if len(pens):
        assert (pens["goals_a_90"] + pens["goals_a_extra_time"]
                == pens["goals_b_90"] + pens["goals_b_extra_time"]).all()
