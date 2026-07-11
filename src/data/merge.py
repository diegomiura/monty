"""Orchestrates download -> clean -> canonical dataset on disk."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.clean import build_canonical
from src.data.download import download_all, source_audit
from src.utils.config import load_config, resolve
from src.utils.dates import utc_now_iso
from src.utils.logging import get_logger

log = get_logger(__name__)


def refresh_dataset(config: dict | None = None, offline: bool = False) -> pd.DataFrame:
    """Download (or reuse cache), rebuild the canonical table, write outputs.

    Rebuilding from raw is idempotent, so a data refresh can never create
    duplicate matches (the table is derived from scratch each time).
    """
    cfg = config or load_config()
    paths = download_all(cfg, offline=offline)
    retrieved_at = utc_now_iso()

    prev_path = resolve(cfg["data"]["processed_dir"]) / "matches.csv"
    n_prev = len(pd.read_csv(prev_path)) if prev_path.exists() else 0

    matches, audit = build_canonical(paths, retrieved_at)

    out_dir = resolve(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out_dir / "matches.csv", index=False)

    meta_dir = resolve(cfg["data"]["metadata_dir"])
    meta_dir.mkdir(parents=True, exist_ok=True)
    audit["retrieved_at"] = retrieved_at
    audit["n_matches_added_vs_previous_build"] = int(len(matches) - n_prev)
    audit["sources"] = source_audit(paths)
    (meta_dir / "data_audit.json").write_text(json.dumps(audit, indent=2))
    log.info("canonical dataset: %d matches through %s", len(matches), audit["last_date"])
    return matches


def load_matches(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    path = resolve(cfg["data"]["processed_dir"]) / "matches.csv"
    if not path.exists():
        raise FileNotFoundError("No canonical dataset. Run: python update_data.py")
    df = pd.read_csv(path, parse_dates=["date"], dtype={"stage": str, "group": str}, low_memory=False)
    df["stage"] = df["stage"].fillna("")
    df["group"] = df["group"].fillna("")
    return df
