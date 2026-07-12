#!/usr/bin/env python3
"""Regenerate the 'Next round' predictions section of README.md.

Detects the next World Cup round from the cached openfootball fixture list
(unplayed matches with both teams decided) and renders the trained model's
predictions between the WC2026-PREDICTIONS markers in README.md.

Examples:
    python update_readme.py                      # rolling mode, latest bundle
    python update_readme.py --update-data        # refresh results first
    python update_readme.py --dry-run            # print the section only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

from src.data.merge import load_matches, refresh_dataset
from src.data.validate import staleness_warning
from src.utils.config import load_config, resolve
from src.visualization.readme_predictions import (
    load_upcoming,
    predict_round,
    render_section,
    splice_readme,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["frozen", "rolling"], default="rolling")
    ap.add_argument("--model", default=None, help="bundle path (default: models/bundle_latest.joblib)")
    ap.add_argument("--readme", default=None, help="README path (default: repo README.md)")
    ap.add_argument("--update-data", action="store_true", help="refresh the dataset first")
    ap.add_argument("--dry-run", action="store_true", help="print the section, do not write README")
    args = ap.parse_args()

    cfg = load_config()
    if args.update_data:
        refresh_dataset(cfg)
    matches = load_matches(cfg)

    fixtures = load_upcoming(cfg)
    if not fixtures:
        print(
            "error: no unplayed fixtures with both teams decided in "
            "data/raw/worldcup_2026.json — run 'python update_data.py "
            "--include-world-cup-2026' after the current round finishes.",
            file=sys.stderr,
        )
        return 2

    model_path = Path(args.model) if args.model else resolve("models") / "bundle_latest.joblib"
    if not model_path.exists():
        print(f"error: no trained model at {model_path}. Run: python train.py --cutoff <date>", file=sys.stderr)
        return 2
    bundle = joblib.load(model_path)

    latest_kickoff = max(f["kickoff_utc"] for f in fixtures)
    warn = staleness_warning(matches, latest_kickoff)
    if warn:
        print(warn, file=sys.stderr)

    preds = predict_round(bundle, matches, fixtures, cfg, mode=args.mode)
    section = render_section(preds, args.mode)

    if args.dry_run:
        print(section)
        return 0

    readme = Path(args.readme) if args.readme else resolve("README.md")
    readme.write_text(splice_readme(readme.read_text(), section))
    names = ", ".join(f"{p['team_a']} v {p['team_b']}" for p in preds)
    print(f"README updated: {len(preds)} fixture(s) ({names}), mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
