#!/usr/bin/env python3
"""Grade the append-only prediction log and refresh the README track record.

Reads reports/predictions/world_cup_2026_predictions.csv, matches each
logged prediction to its completed match, fills the evaluation columns
(never the prediction columns), and renders the latest results into the
TRACK-RECORD section of README.md.

Examples:
    python update_track_record.py                 # grade + update README
    python update_track_record.py --update-data   # refresh results first
    python update_track_record.py --dry-run       # print, write nothing
    python update_track_record.py --limit 5       # show the last 5 matches
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.merge import load_matches, refresh_dataset
from src.evaluation.track_record import (
    current_section,
    grade,
    load_log,
    render_section,
    section_equivalent,
    splice_readme,
    summarize,
    write_graded_log,
)
from src.utils.config import load_config, resolve


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--limit", type=int, default=10, help="matches to show (default: 10)")
    ap.add_argument("--log", default=None, help="prediction log path")
    ap.add_argument("--readme", default=None, help="README path (default: repo README.md)")
    ap.add_argument("--update-data", action="store_true", help="refresh the dataset first")
    ap.add_argument("--dry-run", action="store_true", help="print the section, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="rewrite even when only the generation timestamp would change")
    args = ap.parse_args()

    cfg = load_config()
    if args.update_data:
        refresh_dataset(cfg)
    matches = load_matches(cfg)

    try:
        log = load_log(args.log)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    graded = grade(log, matches)
    s = summarize(graded)
    if s["n"] == 0:
        print(
            f"No logged prediction has a completed result yet "
            f"({s['n_ungraded']} ungraded). Latest match in dataset: "
            f"{matches['date'].max().date()}. Run with --update-data to refresh.",
            file=sys.stderr,
        )

    section = render_section(graded, limit=args.limit)

    if args.dry_run:
        print(section)
        return 0

    if s["n"]:
        path = write_graded_log(graded, args.log)
        print(f"graded {s['n']}/{len(graded)} logged predictions -> {path}")

    readme = Path(args.readme) if args.readme else resolve("README.md")
    text = readme.read_text()
    if not args.force and section_equivalent(current_section(text), current_section(section)):
        print("README track record already current; not rewriting. Use --force to rewrite anyway.")
        return 0
    readme.write_text(splice_readme(text, section))
    print(
        f"README updated: {s['n']} graded prediction(s), "
        f"{s.get('correct_results', 0)} correct, {s.get('exact_scores', 0)} exact."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
