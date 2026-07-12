#!/usr/bin/env python3
"""Forecast the remainder of the 2026 World Cup knockout bracket.

Propagates the trained model's pairwise advancement probabilities through
the remaining fixtures by exact enumeration, producing each team's
probability of reaching the semi-final/final, finishing third, and winning
the World Cup.

Examples:
    python simulate_tournament.py --mode rolling
    python simulate_tournament.py --mode frozen --json-only
    python simulate_tournament.py --cutoff "2026-07-11T00:00:00Z"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.data.merge import load_matches, refresh_dataset
from src.data.team_names import load_alias_table, make_resolver
from src.data.validate import freshness_report, print_freshness, staleness_warning
from src.data.world_cup_2026 import wc2026_start
from src.prediction.simulate_tournament import (
    PairPredictor,
    console_table,
    forecast_bracket,
    load_bracket,
)
from src.utils.config import load_config, resolve
from src.utils.dates import cutoff_date, parse_cutoff, utc_now, utc_now_iso

FORECAST_LOG = "reports/predictions/wc2026_tournament_forecasts.csv"
LOG_FIELDS = [
    "created_at", "prediction_cutoff", "mode", "team",
    "p_reach_semi_final", "p_reach_final", "p_third_place",
    "p_runner_up", "p_champion", "model_version", "latest_match_used",
]


def append_forecast_log(rows: list[dict], cutoff, mode: str, bundle, latest_match: str) -> None:
    """Append-only per-team snapshot log (forecasts are never rewritten)."""
    path = resolve(FORECAST_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    created = utc_now_iso()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            if r["p_champion"] <= 0 and not r["p_third_place"]:
                continue  # already eliminated
            w.writerow(
                {
                    "created_at": created,
                    "prediction_cutoff": str(cutoff),
                    "mode": mode,
                    "team": r["team"],
                    "p_reach_semi_final": r.get("p_reach_semi_final", ""),
                    "p_reach_final": r["p_reach_final"],
                    "p_third_place": r["p_third_place"],
                    "p_runner_up": r["p_runner_up"],
                    "p_champion": r["p_champion"],
                    "model_version": bundle.model_version,
                    "latest_match_used": latest_match,
                }
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["frozen", "rolling"], default="rolling",
                    help="frozen: features stop at WC2026 start; rolling: features through cutoff")
    ap.add_argument("--cutoff", default=None, help="prediction cutoff (default: now, UTC)")
    ap.add_argument("--model", default=None, help="bundle path (default: models/bundle_latest.joblib)")
    ap.add_argument("--update-data", action="store_true")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="do not append to the forecast log")
    args = ap.parse_args()

    cfg = load_config()
    if args.update_data:
        refresh_dataset(cfg)
    matches = load_matches(cfg)

    cutoff = parse_cutoff(args.cutoff) if args.cutoff else pd.Timestamp(utc_now())
    model_path = Path(args.model) if args.model else resolve("models") / "bundle_latest.joblib"
    if not model_path.exists():
        print(f"error: no trained model at {model_path}. Run: python train.py --cutoff <date>",
              file=sys.stderr)
        return 2
    bundle = joblib.load(model_path)

    bracket_path = resolve(cfg["data"]["raw_dir"]) / "worldcup_2026.json"
    if not bracket_path.exists():
        print("error: no cached 2026 fixture file. Run: python update_data.py --include-world-cup-2026",
              file=sys.stderr)
        return 2
    aliases = load_alias_table(resolve(cfg["data"]["raw_dir"]) / "former_names.csv")
    try:
        nodes = load_bracket(bracket_path, make_resolver(aliases))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not args.json_only:
        print_freshness(freshness_report(matches, cutoff, config=cfg))
        warn = staleness_warning(matches, cutoff)
        if warn:
            print(warn)

    feature_cutoff = pd.Timestamp(wc2026_start(cfg)) if args.mode == "frozen" else cutoff_date(cutoff)
    hosts = tuple(cfg["world_cup_2026"]["host_countries"])
    pp = PairPredictor(bundle, matches, feature_cutoff, hosts=hosts)
    fc = forecast_bracket(nodes, pp.advance_prob, cutoff_date(cutoff))

    if not fc["pending"]:
        print("error: no matches left to forecast before this cutoff", file=sys.stderr)
        return 2

    latest_match = str(matches[matches["date"] < feature_cutoff]["date"].max().date())
    out = {
        "tournament": "FIFA World Cup 2026",
        "mode": args.mode,
        "prediction_cutoff": str(cutoff),
        "created_at": utc_now_iso(),
        "method": (
            "exact bracket enumeration over pairwise staged-knockout advancement "
            "probabilities; features frozen at the cutoff (hypothetical results do "
            "not update Elo/form within a path)"
        ),
        "remaining_matches": [
            {"num": n["num"], "date": str(n["date"].date()), "round": n["round"],
             "slot1": n["slot1"], "slot2": n["slot2"]}
            for n in fc["pending"]
        ],
        "teams": fc["teams"],
        "evaluated_pairings": [
            {
                "date": key[2], "team_a": key[0], "team_b": key[1],
                "outcome_90": p["outcome_90"], "knockout": p["knockout"],
                "expected_goals": p["expected_goals"],
            }
            for key, p in sorted(pp.predictions.items(), key=lambda kv: kv[0][2])
        ],
        "model_information": {
            "model_version": bundle.model_version,
            "feature_version": bundle.feature_version,
            "training_cutoff": bundle.training_cutoff,
            "feature_cutoff": str(feature_cutoff.date()),
            "latest_match_in_features": latest_match,
        },
    }

    if not args.json_only:
        print(console_table(fc["teams"], len(fc["pending"])))
    print(json.dumps(out, indent=2))

    if not args.no_log:
        append_forecast_log(fc["teams"], cutoff, args.mode, bundle, latest_match)
        stamp = f"{args.mode}_{cutoff.date()}"
        path = resolve("reports/predictions") / f"wc2026_tournament_forecast_{stamp}.json"
        if path.exists():  # never overwrite an earlier saved forecast
            path = path.with_name(path.stem + "_" + utc_now().strftime("%H%M%S") + ".json")
        path.write_text(json.dumps(out, indent=2))
        print(f"\nsaved {path} and appended to {FORECAST_LOG}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
