"""Download and cache the free public data sources.

Sources (both free, no API key required):

1. martj42/international_results — community-maintained CSV of senior men's
   international results 1872..present. Scores for knockout matches include
   extra time (never penalties). https://github.com/martj42/international_results
2. openfootball/worldcup.json — public-domain JSON for FIFA World Cups with
   separate regulation (ft), extra-time (et) and penalty (p) scores plus
   stage labels. https://github.com/openfootball/worldcup.json

Raw downloads are cached under data/raw with a JSON metadata sidecar
(retrieval timestamp, URL, size). Cached copies are reused when younger
than cache_max_age_hours, or always with offline=True.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.utils.config import load_config, resolve
from src.utils.logging import get_logger

log = get_logger(__name__)

USER_AGENT = "soccer-predictor/0.1 (research; free public data)"


def _meta_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".meta.json")


def _cache_age_hours(target: Path) -> float | None:
    meta = _meta_path(target)
    if not (target.exists() and meta.exists()):
        return None
    retrieved = json.loads(meta.read_text())["retrieved_at"]
    dt = datetime.strptime(retrieved, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0


def fetch(url: str, target: Path, max_age_hours: float = 6.0, offline: bool = False) -> Path:
    """Download url to target unless a fresh cached copy exists."""
    age = _cache_age_hours(target)
    if offline:
        if age is None:
            raise FileNotFoundError(
                f"Offline mode but no cached copy of {url} at {target}. "
                "Run update_data.py with network access first."
            )
        log.info("offline: using cache (%.1f h old): %s", age, target.name)
        return target
    if age is not None and age < max_age_hours:
        log.info("cache fresh (%.1f h): %s", age, target.name)
        return target
    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
    except requests.RequestException as exc:
        if age is not None:
            log.warning("download failed (%s); falling back to cache %.1f h old: %s", exc, age, target.name)
            return target
        raise RuntimeError(
            f"Could not download {url} and no cached copy exists: {exc}"
        ) from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(resp.content)
    _meta_path(target).write_text(
        json.dumps(
            {
                "url": url,
                "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bytes": len(resp.content),
            },
            indent=2,
        )
    )
    log.info("downloaded %s (%d bytes)", target.name, len(resp.content))
    return target


def download_all(config: dict | None = None, offline: bool = False) -> dict[str, Path]:
    """Fetch every raw source; returns {name: path}. Optional sources may fail
    without aborting the pipeline (a warning is logged)."""
    cfg = config or load_config()
    dcfg = cfg["data"]
    raw = resolve(dcfg["raw_dir"])
    max_age = float(dcfg.get("cache_max_age_hours", 6))
    out: dict[str, Path] = {}

    out["results"] = fetch(dcfg["results_url"], raw / "results.csv", max_age, offline)
    out["shootouts"] = fetch(dcfg["shootouts_url"], raw / "shootouts.csv", max_age, offline)
    out["former_names"] = fetch(dcfg["former_names_url"], raw / "former_names.csv", max_age, offline)

    for year in dcfg["openfootball_wc_years"]:
        url = dcfg["openfootball_wc_url_template"].format(year=year)
        try:
            out[f"worldcup_{year}"] = fetch(url, raw / f"worldcup_{year}.json", max_age, offline)
        except (RuntimeError, FileNotFoundError) as exc:
            # Optional enrichment source: pipeline continues without it.
            log.warning("optional source worldcup_%s unavailable: %s", year, exc)
    return out


def source_audit(paths: dict[str, Path]) -> list[dict]:
    """Machine-readable audit of every raw source used."""
    audit = []
    for name, path in paths.items():
        meta = json.loads(_meta_path(path).read_text()) if _meta_path(path).exists() else {}
        audit.append(
            {
                "source": name,
                "path": str(path),
                "url": meta.get("url"),
                "retrieved_at": meta.get("retrieved_at"),
                "bytes": meta.get("bytes"),
                "license": (
                    "CC0/public-domain-style community data; verify at source repo"
                    if "worldcup" in name
                    else "See https://github.com/martj42/international_results (free public dataset)"
                ),
            }
        )
    return audit
