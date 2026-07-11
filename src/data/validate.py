"""Data freshness reporting and staleness checks."""
from __future__ import annotations

import json

import pandas as pd

from src.utils.config import load_config, resolve
from src.utils.dates import utc_now

FRIENDLY_LIKE = {"Friendly", "FIFA Series"}


def freshness_report(
    matches: pd.DataFrame,
    prediction_cutoff: pd.Timestamp,
    team_a: str | None = None,
    team_b: str | None = None,
    config: dict | None = None,
) -> dict:
    cfg = config or load_config()
    meta_path = resolve(cfg["data"]["metadata_dir"]) / "data_audit.json"
    audit = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    competitive = matches[~matches["competition"].isin(FRIENDLY_LIKE)]
    wc26 = matches[
        (matches["competition"] == "FIFA World Cup") & (matches["date"] >= "2026-01-01")
    ]

    def latest_for(team):
        if team is None:
            return None
        m = matches[(matches["team_a"] == team) | (matches["team_b"] == team)]
        return str(m["date"].max().date()) if len(m) else "never"

    rep = {
        "data_retrieval_timestamp": audit.get("retrieved_at", "unknown"),
        "prediction_cutoff": str(prediction_cutoff),
        "latest_match_in_dataset": str(matches["date"].max().date()),
        "latest_competitive_international": str(competitive["date"].max().date()),
        "latest_match_team_a": latest_for(team_a),
        "latest_match_team_b": latest_for(team_b),
        "latest_completed_wc2026_match": str(wc26["date"].max().date()) if len(wc26) else "none",
        "n_matches_added_latest_update": audit.get("n_matches_added_vs_previous_build"),
        "sources": [s.get("url") for s in audit.get("sources", [])][:3],
    }
    return rep


def print_freshness(rep: dict) -> None:
    print("--- Data freshness ---")
    for k, v in rep.items():
        print(f"{k:38s} {v}")
    print("----------------------")


def staleness_warning(matches: pd.DataFrame, prediction_cutoff: pd.Timestamp, max_gap_days: int = 30) -> str | None:
    """Warn when the dataset looks stale relative to the requested cutoff."""
    latest = matches["date"].max()
    ref = min(pd.Timestamp(prediction_cutoff.date()), pd.Timestamp(utc_now().date()))
    gap = (ref - latest).days
    if gap > max_gap_days:
        return (
            f"WARNING: latest match in dataset is {latest.date()}, {gap} days before "
            f"the prediction cutoff. Run 'python update_data.py' to refresh, or the "
            f"prediction will rely on stale form/Elo."
        )
    return None
