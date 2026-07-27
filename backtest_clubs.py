#!/usr/bin/env python3
"""Chronological validation of the club model against the closing line.

Expanding-window by season: train on every match before season S, predict S,
step forward. No random splitting, no future data — the same discipline the
international backtests use.

The market row is the point of the exercise. It is not a model we beat or
lose to casually: the closing line aggregates far more information than any
free-data model can see (team news, lineups, money). The honest question is
how close a goals-and-shots model gets, and whether it is *calibrated* over
the range where the market is.

Examples:
    python backtest_clubs.py
    python backtest_clubs.py --first-eval-season 2010-11 --devig shin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.clubs.clean_clubs import refresh_club_dataset
from src.evaluation.market import MarketBaseline, overround, priced_mask
from src.evaluation.metrics import result_metrics
from src.features.club_features import CLUB_FEATURE_COLUMNS, build_club_feature_table
from src.models.baselines import EloProbabilityModel, FrequencyBaseline, HigherEloBaseline
from src.models.club_outcome import make_club_gradient_boosting, make_club_logistic
from src.utils.config import resolve


def season_start_year(season: str) -> int:
    return int(season.split("-")[0])


def build_models(cfg):
    return {
        "frequency": FrequencyBaseline(),
        "higher_elo": HigherEloBaseline(),
        "elo": EloProbabilityModel(),
        "logistic": make_club_logistic(cfg),
        "gradient_boosting": make_club_gradient_boosting(cfg),
    }


def run_validation(feats: pd.DataFrame, cfg: dict, first_eval_season: str, devig: str) -> dict:
    seasons = sorted(feats["season"].unique(), key=season_start_year)
    evals = [s for s in seasons if season_start_year(s) >= season_start_year(first_eval_season)]

    per_fold, preds = [], []
    for season in evals:
        train = feats[feats["season"].map(season_start_year) < season_start_year(season)]
        test = feats[feats["season"] == season]
        if len(train) < 500 or test.empty:
            continue
        ytr = train["outcome"].to_numpy()
        yte = test["outcome"].to_numpy()

        fold = {"season": season, "n_train": len(train), "n_test": len(test)}
        probs = {}
        for name, model in build_models(cfg).items():
            model.fit(train, ytr)
            p = model.predict_proba(test)
            probs[name] = p
            fold[name] = result_metrics(yte, p)["log_loss"]
        probs["market"] = MarketBaseline(devig).predict_proba(test)

        rows = pd.DataFrame({"season": season, "outcome": yte}, index=test.index)
        for name, p in probs.items():
            rows[[f"{name}_home", f"{name}_draw", f"{name}_away"]] = p
        rows["odds_source"] = test.get("odds_source")
        preds.append(rows)
        per_fold.append(fold)
        print(f"  {season}: train {len(train):5d} test {len(test):3d} "
              f"logistic {fold['logistic']:.4f} gb {fold['gradient_boosting']:.4f}")

    return {"folds": per_fold, "predictions": pd.concat(preds) if preds else pd.DataFrame()}


def score_all(preds: pd.DataFrame, models: list[str], mask: np.ndarray | None = None) -> pd.DataFrame:
    sub = preds if mask is None else preds[mask]
    y = sub["outcome"].to_numpy()
    rows = []
    for name in models:
        cols = [f"{name}_home", f"{name}_draw", f"{name}_away"]
        p = sub[cols].to_numpy(dtype=float)
        ok = np.isfinite(p).all(axis=1)
        if ok.sum() == 0:
            continue
        m = result_metrics(y[ok], p[ok])
        rows.append({"model": name, **m})
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/clubs.yaml")
    ap.add_argument("--first-eval-season", default="2000-01")
    ap.add_argument("--devig", default=None, choices=["proportional", "shin"])
    ap.add_argument("--offline", action="store_true", default=True)
    ap.add_argument("--out", default="reports/backtests")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(resolve(args.config)))
    devig = args.devig or cfg["market"]["devig_method"]
    benchmark_source = cfg["market"].get("benchmark_source")

    d = cfg["data"]
    print("Building club dataset...")
    matches, audit = refresh_club_dataset(
        d["first_season_year"], d["last_season_year"], d["division"],
        offline=args.offline, schedule=d["schedule"],
    )
    print(f"  {audit['n_matches']} matches, {audit['n_clubs']} clubs, "
          f"{audit['first_date']} -> {audit['last_date']}")

    print("Building features...")
    feats = build_club_feature_table(matches, cfg)

    print(f"Chronological validation (expanding window, from {args.first_eval_season}):")
    res = run_validation(feats, cfg, args.first_eval_season, devig)
    preds = res["predictions"]
    if preds.empty:
        print("error: no folds evaluated", file=sys.stderr)
        return 2

    model_names = ["frequency", "higher_elo", "elo", "logistic", "gradient_boosting"]
    overall = score_all(preds, model_names)

    priced = priced_mask(matches.loc[preds.index], benchmark_source)
    head_to_head = score_all(preds, model_names + ["market"], priced)

    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "epl_validation_predictions.csv")
    summary = {
        "devig_method": devig,
        "benchmark_source": benchmark_source,
        "n_matches_scored": int(len(preds)),
        "n_matches_priced": int(priced.sum()),
        "all_matches": overall.to_dict("records"),
        "priced_subset": head_to_head.to_dict("records"),
        "folds": res["folds"],
    }
    (out_dir / "epl_validation.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n=== All {len(preds)} validated matches ===")
    print(overall.to_string(index=False))
    print(f"\n=== Head to head vs the closing line ({benchmark_source}, "
          f"{int(priced.sum())} matches, {devig} de-vig) ===")
    print(head_to_head.to_string(index=False))
    print(f"\nWritten to {out_dir}/epl_validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
