#!/usr/bin/env python3
"""Refresh the canonical dataset from free public sources.

Examples:
    python update_data.py
    python update_data.py --through "2026-07-11T20:00:00Z" --include-world-cup-2026
    python update_data.py --offline          # rebuild from cached raw files
"""
from __future__ import annotations

import argparse
import sys

from src.data.merge import refresh_dataset
from src.data.validate import freshness_report, print_freshness
from src.data.world_cup_2026 import write_verification_report
from src.utils.config import load_config
from src.utils.dates import parse_cutoff, utc_now_iso


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--through", default=None, help="prediction cutoff the data should support (ISO timestamp)")
    ap.add_argument("--include-world-cup-2026", action="store_true", help="verify 2026 WC results across sources")
    ap.add_argument("--offline", action="store_true", help="use cached raw files, no network")
    args = ap.parse_args()

    cutoff = parse_cutoff(args.through) if args.through else parse_cutoff(utc_now_iso())

    matches = refresh_dataset(load_config(), offline=args.offline)
    print_freshness(freshness_report(matches, cutoff))

    if args.include_world_cup_2026:
        rep = write_verification_report(matches)
        print(f"2026 WC verification: {rep.get('agreed', 0)}/{rep.get('checked', 0)} "
              f"matches agree across sources; disagreements: {len(rep.get('disagreements', []))}")
        for d in rep.get("disagreements", []):
            print("  DISAGREE:", d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
