"""Date/timestamp helpers.

The primary dataset provides match dates without kickoff times. The
conservative rule (documented assumption): a match on date D may only use
information from matches strictly before D, so same-day matches can never
affect each other.  All cutoffs are handled as UTC timestamps.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_cutoff(value: str) -> pd.Timestamp:
    """Parse a user-supplied cutoff ('2026-07-11' or '2026-07-11T20:00:00Z')."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def cutoff_date(cutoff: pd.Timestamp) -> pd.Timestamp:
    """Naive date (00:00) below which whole match-days are usable.

    Because the dataset has dates but not kickoff times, a feature cutoff at
    time T admits only matches with date strictly before T's date.
    """
    return pd.Timestamp(cutoff.tz_convert("UTC").date())
