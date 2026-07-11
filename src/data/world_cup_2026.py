"""2026 FIFA World Cup workflow: ingest, verify, and expose tournament data.

Completed 2026 matches already flow into the canonical table via clean.py
(martj42 results corrected with openfootball ft/et/p splits). This module
adds the dedicated tournament view: stage/group labels, cross-source
verification, and cutoff-aware retrieval for frozen/rolling modes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.clean import load_openfootball_wc
from src.data.team_names import load_alias_table, make_resolver
from src.utils.config import load_config, resolve
from src.utils.dates import cutoff_date
from src.utils.logging import get_logger

log = get_logger(__name__)


def wc2026_start(config: dict | None = None) -> pd.Timestamp:
    cfg = config or load_config()
    return pd.Timestamp(cfg["world_cup_2026"]["start_date"])


def wc2026_matches(matches: pd.DataFrame, prediction_cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Completed 2026 World Cup matches, optionally only those strictly
    before the prediction cutoff's date (rolling-mode rule)."""
    wc = matches[
        (matches["competition"] == "FIFA World Cup") & (matches["date"] >= "2026-06-01")
    ].copy()
    if prediction_cutoff is not None:
        wc = wc[wc["date"] < cutoff_date(prediction_cutoff)]
    return wc


def verify_against_openfootball(matches: pd.DataFrame, config: dict | None = None) -> dict:
    """Cross-check 2026 results between the two independent sources.

    Returns a verification report; disagreements are listed, not silently fixed.
    """
    cfg = config or load_config()
    raw = resolve(cfg["data"]["raw_dir"])
    path = raw / "worldcup_2026.json"
    report = {"checked": 0, "agreed": 0, "disagreements": [], "missing_in_primary": []}
    if not path.exists():
        report["note"] = "openfootball 2026 file not cached; verification skipped"
        return report
    aliases = load_alias_table(raw / "former_names.csv")
    of = load_openfootball_wc(path, make_resolver(aliases))
    wc = wc2026_matches(matches)
    idx = {(r.date, r.team_a, r.team_b): r for r in wc.itertuples()}
    for _, r in of.iterrows():
        m = idx.get((r["date"], r["team_a"], r["team_b"]))
        swapped = False
        if m is None:  # sources may disagree on nominal home side
            m = idx.get((r["date"], r["team_b"], r["team_a"]))
            swapped = m is not None
        if m is None:
            report["missing_in_primary"].append(f"{r['date'].date()} {r['team_a']} v {r['team_b']}")
            continue
        report["checked"] += 1
        of_ft = (r["of_ft_b"], r["of_ft_a"]) if swapped else (r["of_ft_a"], r["of_ft_b"])
        if (m.goals_a_90, m.goals_b_90) == of_ft:
            report["agreed"] += 1
        else:
            report["disagreements"].append(
                f"{r['date'].date()} {r['team_a']} v {r['team_b']}: "
                f"primary {m.goals_a_90}-{m.goals_b_90} vs openfootball ft {of_ft[0]}-{of_ft[1]}"
            )
    return report


def write_verification_report(matches: pd.DataFrame, config: dict | None = None) -> dict:
    cfg = config or load_config()
    rep = verify_against_openfootball(matches, cfg)
    out = resolve(cfg["data"]["metadata_dir"]) / "wc2026_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    if rep.get("disagreements"):
        log.warning("2026 WC source disagreement: %s", rep["disagreements"])
    return rep
