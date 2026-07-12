import numpy as np
import pandas as pd

from src.features.build_features import FeatureBuilder, build_feature_table
from src.models.baselines import EloProbabilityModel, FrequencyBaseline, HigherEloBaseline
from src.models.outcome import make_gradient_boosting, make_logistic, mirror, mirror_labels


def _feats(cfg, toy_matches):
    return build_feature_table(toy_matches, cfg)


def test_baseline_probs_sum_to_one(cfg, toy_matches):
    feats = _feats(cfg, toy_matches)
    y = feats["outcome"].to_numpy()
    for model in (FrequencyBaseline(), HigherEloBaseline(), EloProbabilityModel()):
        p = model.fit(feats, y).predict_proba(feats)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
        assert (p >= 0).all()


def test_outcome_models_sum_to_one(cfg, rich_matches):
    # rich_matches, not toy_matches: HistGradientBoosting (sklearn >= 1.9
    # with numpy >= 2.x) cannot fit columns that are entirely NaN, which the
    # sparse toy fixture produces for the long rolling-form windows.
    feats = _feats(cfg, rich_matches)
    y = feats["outcome"].to_numpy()
    for maker in (make_logistic, make_gradient_boosting):
        m = maker(cfg).fit(feats, y)
        p = m.predict_proba(feats)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_mirror_swaps_correctly(cfg, toy_matches):
    feats = _feats(cfg, toy_matches)
    from src.features.build_features import FEATURE_COLUMNS

    X = feats[FEATURE_COLUMNS]
    M = mirror(X)
    assert np.allclose(M["elo_a"].to_numpy(), X["elo_b"].to_numpy(), equal_nan=True)
    assert np.allclose(M["elo_diff"].to_numpy(), -X["elo_diff"].to_numpy(), equal_nan=True)
    assert np.allclose(M["home_edge"].to_numpy(), -X["home_edge"].to_numpy())
    assert np.allclose(M["elo_expected_a"].to_numpy(), 1 - X["elo_expected_a"].to_numpy())
    assert (mirror_labels(np.array([0, 1, 2])) == np.array([2, 1, 0])).all()


def test_symmetry_of_prediction(cfg, toy_matches):
    """Swapping team A and B must mirror the probabilities exactly."""
    feats = _feats(cfg, toy_matches)
    y = feats["outcome"].to_numpy()
    m = make_logistic(cfg).fit(feats, y)
    fb = FeatureBuilder(cfg)
    fb.run(toy_matches)
    r1 = fb.fixture_row("Alpha", "Beta", pd.Timestamp("2021-01-01"), "Friendly", neutral=True)
    r2 = fb.fixture_row("Beta", "Alpha", pd.Timestamp("2021-01-01"), "Friendly", neutral=True)
    p1 = m.predict_proba(pd.DataFrame([r1]))[0]
    p2 = m.predict_proba(pd.DataFrame([r2]))[0]
    assert np.allclose(p1, p2[::-1], atol=1e-9)
