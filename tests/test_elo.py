import pandas as pd

from src.features.elo import Elo


def make_elo(cfg):
    return Elo(cfg)


def test_elo_update_symmetric(cfg):
    elo = make_elo(cfg)
    d = pd.Timestamp("2020-01-01")
    elo.update(d, "A", "B", 2, 0, "friendly", 0)
    assert elo.ratings["A"] > cfg["elo"]["initial_rating"] > elo.ratings["B"]
    # zero-sum update
    assert abs((elo.ratings["A"] - 1500) + (elo.ratings["B"] - 1500)) < 1e-9


def test_elo_draw_moves_toward_stronger(cfg):
    elo = make_elo(cfg)
    elo.ratings.update({"A": 1700.0, "B": 1500.0})
    elo.update(pd.Timestamp("2020-01-01"), "A", "B", 1, 1, "friendly", 0)
    assert elo.ratings["A"] < 1700.0 and elo.ratings["B"] > 1500.0


def test_home_advantage_raises_expected(cfg):
    elo = make_elo(cfg)
    assert elo.expected(1500, 1500, 1) > 0.5 > elo.expected(1500, 1500, 0) - 1e-9


def test_margin_multiplier():
    assert Elo.margin_multiplier(1) == 1.0
    assert Elo.margin_multiplier(2) == 1.5
    assert Elo.margin_multiplier(3) == (11 + 3) / 8
    assert Elo.margin_multiplier(-3) == (11 + 3) / 8


def test_k_by_competition(cfg):
    e1, e2 = make_elo(cfg), make_elo(cfg)
    d = pd.Timestamp("2020-01-01")
    e1.update(d, "A", "B", 1, 0, "world_cup", 0)
    e2.update(d, "A", "B", 1, 0, "friendly", 0)
    assert e1.ratings["A"] > e2.ratings["A"]


def test_inactivity_shrink(cfg):
    elo = make_elo(cfg)
    elo.ratings["A"] = 1800.0
    elo.last_played["A"] = pd.Timestamp("2000-01-01")
    later = pd.Timestamp("2010-01-01")
    assert elo.rating("A", later) < 1800.0


def test_pre_match_rating_used_in_features(cfg, toy_matches):
    """The stored feature must be the rating BEFORE the match."""
    from src.features.build_features import FeatureBuilder

    fb = FeatureBuilder(cfg)
    feats = fb.run(toy_matches)
    # First ever match: both teams at the initial rating.
    assert feats.iloc[0]["elo_a"] == cfg["elo"]["initial_rating"]
    assert feats.iloc[0]["elo_b"] == cfg["elo"]["initial_rating"]
