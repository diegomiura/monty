"""Parameter tuning on DEVELOPMENT folds only (validation years <= 2013).

Nothing in here ever sees 2014+ data, so the 2014/2018/2022 World Cup
holdouts and all later validation remain untouched by tuning. Results are
written to data/metadata/tuning.json; chosen values then live in
config/default.yaml (updated manually, documented).

Run:  python -m src.evaluation.tune
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data.merge import load_matches
from src.evaluation.metrics import log_loss
from src.features.build_features import build_feature_table
from src.models.outcome import make_gradient_boosting, make_logistic
from src.utils.config import load_config, resolve
from src.utils.logging import get_logger

log = get_logger(__name__)

DEV_TRAIN_START = "1990-01-01"


def _elo_fold_score(feats: pd.DataFrame, years: list[int]) -> float:
    """Mean fold log loss of an Elo-only logistic model."""
    scores = []
    for y in years:
        tr = feats[(feats["date"] >= DEV_TRAIN_START) & (feats["date"] < f"{y}-01-01")]
        va = feats[(feats["date"] >= f"{y}-01-01") & (feats["date"] < f"{y + 1}-01-01")]
        lr = LogisticRegression(C=10.0, max_iter=1000)
        lr.fit(tr[["elo_expected_a"]], tr["outcome"])
        scores.append(log_loss(va["outcome"].to_numpy(), lr.predict_proba(va[["elo_expected_a"]])))
    return float(np.mean(scores))


def tune_elo(matches: pd.DataFrame, cfg: dict) -> dict:
    years = cfg["validation"]["dev_val_years"]
    grid = [
        {"home_advantage": h, "k_scale": k, "margin_multiplier": m}
        for h in (60.0, 80.0, 100.0)
        for k in (0.8, 1.0, 1.2)
        for m in (True, False)
    ]
    results = []
    for params in grid:
        c = copy.deepcopy(cfg)
        c["elo"].update(params)
        feats = build_feature_table(matches[matches["date"] < f"{years[-1] + 1}-01-01"].reset_index(drop=True), c)
        score = _elo_fold_score(feats, years)
        results.append({**params, "log_loss": round(score, 5)})
        log.info("elo %s -> %.5f", params, score)
    results.sort(key=lambda r: r["log_loss"])
    return {"grid": results, "best": results[0]}


def tune_outcome_models(feats: pd.DataFrame, cfg: dict) -> dict:
    years = cfg["validation"]["dev_val_years"]
    out = {"logistic": [], "gradient_boosting": []}
    for C in (0.02, 0.05, 0.2):
        c = copy.deepcopy(cfg)
        c["models"]["logit_C"] = C
        scores = []
        for y in years[::2]:  # every other dev year for speed
            tr = feats[(feats["date"] >= DEV_TRAIN_START) & (feats["date"] < f"{y}-01-01")]
            va = feats[(feats["date"] >= f"{y}-01-01") & (feats["date"] < f"{y + 1}-01-01")]
            m = make_logistic(c).fit(tr, tr["outcome"].to_numpy())
            scores.append(log_loss(va["outcome"].to_numpy(), m.predict_proba(va)))
        out["logistic"].append({"C": C, "log_loss": round(float(np.mean(scores)), 5)})
        log.info("logit C=%s -> %.5f", C, np.mean(scores))
    for lr_rate, leaves in ((0.06, 31), (0.1, 31), (0.06, 63)):
        c = copy.deepcopy(cfg)
        c["models"]["hgb_learning_rate"] = lr_rate
        c["models"]["hgb_max_leaf_nodes"] = leaves
        scores = []
        for y in years[::2]:
            tr = feats[(feats["date"] >= DEV_TRAIN_START) & (feats["date"] < f"{y}-01-01")]
            va = feats[(feats["date"] >= f"{y}-01-01") & (feats["date"] < f"{y + 1}-01-01")]
            m = make_gradient_boosting(c).fit(tr, tr["outcome"].to_numpy())
            scores.append(log_loss(va["outcome"].to_numpy(), m.predict_proba(va)))
        out["gradient_boosting"].append(
            {"learning_rate": lr_rate, "max_leaf_nodes": leaves, "log_loss": round(float(np.mean(scores)), 5)}
        )
        log.info("hgb lr=%s leaves=%s -> %.5f", lr_rate, leaves, np.mean(scores))
    return out


def main() -> None:
    cfg = load_config()
    matches = load_matches(cfg)
    dev_cut = f"{cfg['validation']['dev_val_years'][-1] + 1}-01-01"
    report = {"note": f"all tuning on validation years <= {dev_cut} only"}
    report["elo"] = tune_elo(matches, cfg)

    best = report["elo"]["best"]
    c = copy.deepcopy(cfg)
    c["elo"].update({k: best[k] for k in ("home_advantage", "k_scale", "margin_multiplier")})
    feats = build_feature_table(matches[matches["date"] < dev_cut].reset_index(drop=True), c)
    report["outcome_models"] = tune_outcome_models(feats, c)

    out = resolve(cfg["data"]["metadata_dir"]) / "tuning.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"elo_best": report["elo"]["best"],
                      "logistic": report["outcome_models"]["logistic"],
                      "gradient_boosting": report["outcome_models"]["gradient_boosting"]}, indent=2))


if __name__ == "__main__":
    main()
