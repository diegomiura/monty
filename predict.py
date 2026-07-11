#!/usr/bin/env python3
"""Predict a match between any two national teams.

Examples:
    python predict.py --team-a "Argentina" --team-b "France" \
        --date "2026-07-15T20:00:00Z" --competition "FIFA World Cup" \
        --stage knockout --neutral --mode rolling

    python predict.py --team-a "Spain" --team-b "Norway" --date 2026-07-14 \
        --competition "FIFA World Cup" --stage knockout --neutral --mode frozen
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
from src.data.validate import freshness_report, print_freshness, staleness_warning
from src.data.world_cup_2026 import wc2026_start
from src.prediction.predict_match import console_report, predict_fixture
from src.utils.config import load_config, resolve
from src.utils.dates import parse_cutoff, utc_now_iso
from src.utils.validation import validate_team
from src.data.team_names import load_alias_table, make_resolver

PREDICTION_LOG = "reports/predictions/world_cup_2026_predictions.csv"
LOG_FIELDS = [
    "match", "kickoff", "prediction_created_at", "prediction_cutoff",
    "predicted_team_a_win", "predicted_draw", "predicted_team_b_win",
    "predicted_score", "expected_goals_a", "expected_goals_b",
    "actual_score", "actual_result", "brier_score", "log_loss",
    "goal_absolute_error", "correct_result", "correct_exact_score",
    "model_version", "feature_version", "latest_match_used", "mode",
]


def append_prediction_log(pred: dict, cutoff, mode: str) -> None:
    """Append-only log for 2026 WC predictions. Result fields stay empty at
    prediction time and are only ever appended later, never rewritten."""
    path = resolve(PREDICTION_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(
            {
                "match": f"{pred['team_a']} v {pred['team_b']}",
                "kickoff": pred["kickoff_timestamp_utc"],
                "prediction_created_at": utc_now_iso(),
                "prediction_cutoff": str(cutoff),
                "predicted_team_a_win": pred["outcome_90"]["team_a_win"],
                "predicted_draw": pred["outcome_90"]["draw"],
                "predicted_team_b_win": pred["outcome_90"]["team_b_win"],
                "predicted_score": pred["most_likely_score"],
                "expected_goals_a": pred["expected_goals"]["team_a"],
                "expected_goals_b": pred["expected_goals"]["team_b"],
                "actual_score": "", "actual_result": "", "brier_score": "",
                "log_loss": "", "goal_absolute_error": "", "correct_result": "",
                "correct_exact_score": "",
                "model_version": pred["model_information"]["model_version"],
                "feature_version": pred["model_information"]["feature_version"],
                "latest_match_used": pred["model_information"]["latest_match_in_features"],
                "mode": mode,
            }
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team-a", required=True)
    ap.add_argument("--team-b", required=True)
    ap.add_argument("--date", required=True, help="kickoff date or ISO timestamp (UTC)")
    ap.add_argument("--competition", default="Friendly")
    ap.add_argument("--stage", default="", help="e.g. 'group' or 'knockout'")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--home-team-a", action="store_true", help="team A plays at home (non-neutral)")
    ap.add_argument("--mode", choices=["frozen", "rolling"], default="rolling",
                    help="frozen: features stop at WC2026 start; rolling: features through cutoff")
    ap.add_argument("--cutoff", default=None, help="prediction cutoff (default: kickoff time)")
    ap.add_argument("--update-data", action="store_true")
    ap.add_argument("--model", default=None, help="bundle path (default: models/bundle_latest.joblib)")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="do not append to the WC2026 prediction log")
    args = ap.parse_args()

    cfg = load_config()
    if args.update_data:
        refresh_dataset(cfg)
    matches = load_matches(cfg)

    aliases = load_alias_table(resolve(cfg["data"]["raw_dir"]) / "former_names.csv")
    resolver = make_resolver(aliases)
    known = set(matches["team_a"]) | set(matches["team_b"])
    try:
        team_a = validate_team(args.team_a, known, resolver)
        team_b = validate_team(args.team_b, known, resolver)
        kickoff = parse_cutoff(args.date)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if team_a == team_b:
        print("error: team A and team B must differ", file=sys.stderr)
        return 2

    cutoff = parse_cutoff(args.cutoff) if args.cutoff else kickoff
    if cutoff > kickoff:
        print("error: prediction cutoff cannot be after kickoff", file=sys.stderr)
        return 2

    model_path = Path(args.model) if args.model else resolve("models") / "bundle_latest.joblib"
    if not model_path.exists():
        print(f"error: no trained model at {model_path}. Run: python train.py --cutoff <date>",
              file=sys.stderr)
        return 2
    bundle = joblib.load(model_path)

    if args.mode == "frozen":
        feature_cutoff = pd.Timestamp(wc2026_start(cfg))
    else:
        feature_cutoff = pd.Timestamp(cutoff.date())

    if not args.json_only:
        rep = freshness_report(matches, cutoff, team_a, team_b, cfg)
        print_freshness(rep)
        warn = staleness_warning(matches, cutoff)
        if warn:
            print(warn)

    pred = predict_fixture(
        bundle, matches, team_a, team_b, kickoff,
        competition=args.competition, stage=args.stage,
        neutral=args.neutral or not args.home_team_a,
        team_a_home=args.home_team_a,
        feature_cutoff=feature_cutoff,
    )
    pred["mode"] = args.mode

    if not args.json_only:
        print(console_report(pred))
    print(json.dumps(pred, indent=2))

    is_wc26 = "world cup" in args.competition.lower() and kickoff.year == 2026
    if is_wc26 and not args.no_log:
        append_prediction_log(pred, cutoff, args.mode)
        print(f"\nlogged to {PREDICTION_LOG} (append-only)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
