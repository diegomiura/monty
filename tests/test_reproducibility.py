"""Saved models must reproduce identical predictions."""
import numpy as np
import pandas as pd
import joblib

from src.features.build_features import build_feature_table
from src.models.outcome import make_logistic


def test_saved_model_reproduces_predictions(cfg, toy_matches, tmp_path):
    feats = build_feature_table(toy_matches, cfg)
    y = feats["outcome"].to_numpy()
    m = make_logistic(cfg).fit(feats, y)
    p1 = m.predict_proba(feats)
    path = tmp_path / "m.joblib"
    joblib.dump(m, path)
    m2 = joblib.load(path)
    p2 = m2.predict_proba(feats)
    assert np.array_equal(p1, p2)


def test_refresh_is_idempotent(cfg):
    """Rebuilding the canonical table from the same raw files must yield the
    same match set (no duplicates on refresh)."""
    from pathlib import Path

    from src.utils.config import resolve

    raw = resolve(cfg["data"]["raw_dir"])
    if not (raw / "results.csv").exists():
        import pytest

        pytest.skip("raw data not downloaded")
    from src.data.clean import build_canonical
    from src.data.download import download_all

    paths = download_all(cfg, offline=True)
    m1, _ = build_canonical(paths, "t1")
    m2, _ = build_canonical(paths, "t2")
    assert len(m1) == len(m2)
    assert m1["match_id"].is_unique
    assert (m1["match_id"] == m2["match_id"]).all()
