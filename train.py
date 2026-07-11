#!/usr/bin/env python3
"""Train the full model bundle through a cutoff date.

Examples:
    python train.py --cutoff "2026-06-11"          # pre-WC2026 (frozen 2026)
    python train.py --cutoff "2026-07-11"          # latest (rolling 2026)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.merge import load_matches
from src.data.validate import freshness_report, print_freshness, staleness_warning
from src.features.build_features import build_feature_table
from src.models.train_pipeline import train_bundle
from src.utils.config import load_config, resolve
from src.utils.dates import parse_cutoff

np.random.seed(42)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="canonical matches CSV (default: data/processed/matches.csv)")
    ap.add_argument("--cutoff", required=True, help="training cutoff date (exclusive), e.g. 2026-06-11")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="output bundle path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data:
        matches = pd.read_csv(args.data, parse_dates=["date"], low_memory=False)
        matches["stage"] = matches.get("stage", "").fillna("") if "stage" in matches else ""
    else:
        matches = load_matches(cfg)

    cutoff = parse_cutoff(args.cutoff)
    print_freshness(freshness_report(matches, cutoff, config=cfg))
    warn = staleness_warning(matches, cutoff)
    if warn:
        print(warn)

    print("building chronological features ...")
    feats = build_feature_table(matches, cfg)
    bundle = train_bundle(feats, matches, pd.Timestamp(cutoff.date()), cfg)

    out = Path(args.out) if args.out else resolve("models") / f"bundle_{cutoff.date()}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    joblib.dump(bundle, resolve("models") / "bundle_latest.joblib")
    print(f"saved {out} (and models/bundle_latest.joblib)")
    print(json.dumps({
        "training_cutoff": bundle.training_cutoff,
        "latest_match_used": bundle.latest_match_used,
        "n_train": bundle.n_train,
        "ensemble_weights": {n: round(float(w), 3) for n, w in
                             zip(["elo", "poisson", "logistic", "gradient_boosting"], bundle.weights)},
        "calibration_used": bundle.use_calibration,
        "validation": bundle.validation_report,
        "et_stats": bundle.et_stats,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
