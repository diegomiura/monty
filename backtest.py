#!/usr/bin/env python3
"""Backtest complete tournaments with leakage-safe training.

Examples:
    python backtest.py --competition "FIFA World Cup" --year 2022 --mode frozen
    python backtest.py --competition "FIFA World Cup" --year 2026 --mode rolling
    python backtest.py --validate                     # chronological folds
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.data.merge import load_matches
from src.evaluation.backtest import (
    chronological_validation,
    save_backtest,
    summarize_validation,
    world_cup_backtest,
)
from src.features.build_features import build_feature_table
from src.utils.config import load_config, resolve

np.random.seed(42)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--competition", default="FIFA World Cup")
    ap.add_argument("--year", type=int)
    ap.add_argument("--mode", choices=["frozen", "rolling", "both"], default="frozen")
    ap.add_argument("--validate", action="store_true", help="run expanding-window chronological validation")
    ap.add_argument("--val-years", default="2014-2025", help="year range for --validate")
    args = ap.parse_args()

    cfg = load_config()
    matches = load_matches(cfg)
    print("building chronological features ...")
    feats = build_feature_table(matches, cfg)

    if args.validate:
        lo, hi = (int(x) for x in args.val_years.split("-"))
        table = chronological_validation(feats, cfg, list(range(lo, hi + 1)))
        out = resolve("reports/backtests") / f"chronological_validation_{lo}_{hi}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print("\n=== Mean metrics across folds", f"{lo}-{hi} ===")
        print(summarize_validation(table).to_string())
        return 0

    if args.competition != "FIFA World Cup":
        print("Only FIFA World Cup backtests are supported in v1 (stage labels "
              "are only available for World Cups).")
        return 2
    if not args.year:
        print("--year is required for tournament backtests")
        return 2

    modes = ["frozen", "rolling"] if args.mode == "both" else [args.mode]
    bundle = None
    for mode in modes:
        result = world_cup_backtest(args.year, mode, matches, feats, cfg, bundle=bundle)
        bundle = result["bundle"]  # reuse the pre-tournament model across modes
        save_backtest(result, args.year, mode, cfg)
        rep = result["report"]
        print(f"\n=== World Cup {args.year} [{mode}] ===")
        print(json.dumps({k: rep[k] for k in ("training_cutoff", "n_matches", "ensemble",
                                              "goals", "ensemble_weights", "calibration_used")}, indent=2))
        print("components:", json.dumps({k: v["log_loss"] for k, v in rep["components"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
