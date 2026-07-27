"""Download and parse football-data.co.uk league CSVs.

Source: https://www.football-data.co.uk/mmz4281/{season}/{div}.csv
Audit:  reports/CLUB_DATA_AUDIT.md (coverage, fields per era, licensing)

The site publishes free season CSVs going back to 1993-94. Downloading and
analysing is the intended use; **redistribution is not granted**, so raw
files are cached under the gitignored ``data/raw`` tree exactly like the
international sources and are never committed.

Why this module exists instead of a bare ``pd.read_csv``
--------------------------------------------------------
Three malformations appear in the published files, verified during the audit:

1. **Ragged rows.** 2004-05 E0 has 45 rows carrying 62 fields against a
   57-field header. ``pd.read_csv`` raises; the usual escapes
   (``on_bad_lines='skip'``, ``engine='python'``) *silently drop those 45
   matches* — 12% of the season, all at the end of it. The extra fields are
   empty trailing commas.
2. **Blank padding rows.** 1993-94 reports 552 rows of which 90 are entirely
   empty (real count 462); 1999-00 reports 552 with 172 empty (real 380).
3. **UTF-8 BOM** on the first header field, which reads as ``ï»¿Div`` when
   the file is decoded as latin-1 (needed for accented club and referee
   names).

Silent row loss is the dangerous one, so :func:`season_integrity` checks the
parsed frame against a property the data must satisfy rather than against a
hardcoded count: in a completed double round-robin of ``n`` teams there are
exactly ``n * (n - 1)`` matches. That catches dropped rows, duplicated rows,
and truncated downloads with no per-season table to maintain.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from src.data.download import fetch
from src.utils.config import load_config, resolve
from src.utils.logging import get_logger

log = get_logger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

#: Columns we keep. Everything else (69 odds columns, other bookmakers) is
#: dropped at parse time; closing odds are pulled separately by name.
CORE_COLUMNS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HC", "AC",
    "HF", "AF", "HY", "AY", "HR", "AR", "Referee",
]

#: Closing-odds columns, kept for the market benchmark only — never features
#: unless a validation gate turns them on (see config ``use_market_features``).
CLOSING_ODDS_COLUMNS = ["PSCH", "PSCD", "PSCA", "B365CH", "B365CD", "B365CA",
                        "AvgCH", "AvgCD", "AvgCA"]

NUMERIC_COLUMNS = [
    "FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST",
    "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
] + CLOSING_ODDS_COLUMNS


def season_code(start_year: int) -> str:
    """1993 -> '9394', 2025 -> '2526' (the site's season directory name)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_codes(first_year: int, last_year: int) -> list[str]:
    """Codes for every season starting in [first_year, last_year]."""
    return [season_code(y) for y in range(first_year, last_year + 1)]


def season_label(code: str) -> str:
    """'2526' -> '2025-26' for human-readable reports."""
    a, b = int(code[:2]), int(code[2:])
    century = 1900 if a >= 90 else 2000
    return f"{century + a}-{b:02d}"


#: Columns that must exist in every published season. Everything else may be
#: legitimately absent for an era (shots pre-2000-01, odds pre-2015-16), but a
#: missing column from this set means the parse went wrong, not that the data
#: is old — so it must raise rather than be filled with NaN.
REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

#: A UTF-8 BOM decoded as latin-1 (which we need for accented names) appears
#: as these three characters, not as U+FEFF.
_BOM_LATIN1 = "ï»¿"


def _strip_bom(s: str) -> str:
    return s.lstrip("﻿").removeprefix(_BOM_LATIN1)


def read_football_data_csv(path: str | Path) -> pd.DataFrame:
    """Parse one season CSV, tolerating every malformation noted above.

    Rows are padded to the widest row rather than dropped, so a ragged file
    keeps all of its matches. Returns every column present in the file; use
    :func:`select_columns` to reduce to the ones we model on.
    """
    raw = Path(path).read_bytes().decode("latin-1")
    rows = list(csv.reader(raw.splitlines()))
    if not rows:
        raise ValueError(f"{path} is empty")

    header = [_strip_bom(h).strip() for h in rows[0]]
    width = max(len(r) for r in rows)
    header += [f"_unnamed{i}" for i in range(width - len(header))]
    body = [r + [None] * (width - len(r)) for r in rows[1:]]

    df = pd.DataFrame(body, columns=header)
    # Blank padding rows carry an empty Date; genuine rows never do.
    if "Date" in df.columns:
        date = df["Date"].astype(str).str.strip()
        df = df[date.ne("") & date.ne("None") & df["Date"].notna()]
    df = df.drop(columns=[c for c in df.columns if c.startswith("_unnamed")], errors="ignore")
    return df.reset_index(drop=True)


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep core + closing-odds columns that exist, coerce types.

    Early seasons genuinely lack shots (pre-2000-01), kickoff times
    (pre-2019-20) and closing odds (pre-2015-16); those columns are created
    as all-NaN so every season shares one schema and missingness stays
    explicit rather than becoming a silent zero.
    """
    absent = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if absent:
        raise ValueError(
            f"required column(s) {absent} missing from the parsed file — this "
            f"indicates a parse failure, not an old season. Columns found: "
            f"{list(df.columns)[:12]}"
        )
    out = pd.DataFrame(index=df.index)
    for col in CORE_COLUMNS + CLOSING_ODDS_COLUMNS:
        out[col] = df[col] if col in df.columns else pd.NA
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("Div", "HomeTeam", "AwayTeam", "FTR", "HTR", "Referee", "Time"):
        out[col] = out[col].astype(object).where(out[col].notna(), None)
    out["Date"] = _parse_dates(df["Date"])
    return out


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Parse the site's two date formats explicitly.

    Both ``dd/mm/yy`` (older seasons) and ``dd/mm/yyyy`` (newer) occur, often
    within one file. Letting pandas fall back to dateutil parses each element
    individually, which is slow and — far worse — can resolve an ambiguous
    value like ``03/04/05`` differently from its neighbours. Two explicit
    passes make the day-first reading deterministic; ``%y`` maps 69-99 to
    1969-1999 and 00-68 to 2000-2068, which covers 1993-2026 correctly.
    """
    s = raw.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    missing = out.isna()
    if missing.any():
        out[missing] = pd.to_datetime(s[missing], format="%d/%m/%y", errors="coerce")
    unparsed = out.isna() & s.ne("") & s.ne("nan") & s.ne("None")
    if unparsed.any():
        raise ValueError(
            f"{int(unparsed.sum())} date(s) matched neither %d/%m/%Y nor %d/%m/%y, "
            f"e.g. {s[unparsed].head(3).tolist()}"
        )
    return out


#: Schedule shapes we can verify a season's completeness against.
#: ``double_round_robin`` — every ordered pair meets exactly once, so a
#: completed season of n teams has exactly n*(n-1) matches. True of the
#: English, Spanish, Italian, German and French top flights.
#: ``unknown`` — no count identity available; only duplicate and score
#: checks run. Required for competitions that are not balanced round
#: robins: MLS (conferences plus an unbalanced schedule — 2025 played 540
#: matches among 30 teams where the identity would demand 870), the Scottish
#: Premiership (post-split 33 or 38 games), and any cup or playoff format.
SCHEDULES = ("double_round_robin", "unknown")


def expected_match_count(n_teams: int, schedule: str = "double_round_robin") -> int | None:
    """Matches in a completed season, or None when the shape can't imply one."""
    if schedule == "double_round_robin":
        return n_teams * (n_teams - 1)
    if schedule == "unknown":
        return None
    raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")


def season_integrity(
    df: pd.DataFrame,
    code: str,
    complete: bool = True,
    schedule: str = "double_round_robin",
) -> dict:
    """Check a parsed season for dropped, duplicated or truncated rows.

    The strongest available check is the schedule identity: in a balanced
    double round-robin, ``n`` teams must produce exactly ``n * (n - 1)``
    matches. That single assertion catches the 2004-05 ragged-CSV failure
    mode (rows silently dropped), duplicated fixtures, and truncated
    downloads, without maintaining a per-season count table.

    **This identity is specific to balanced round-robin leagues.** Pass
    ``schedule="unknown"`` for competitions that are not (see
    :data:`SCHEDULES`); duplicate and score checks still run, but a silent
    row loss would no longer be detectable from structure alone.

    ``complete=False`` for a season still in progress: the upper bound and
    duplicate checks are enforced, the exact count is not.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")

    teams = sorted(set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna()))
    n = len(teams)
    expected = expected_match_count(n, schedule)
    actual = len(df)
    pairs = list(zip(df["HomeTeam"], df["AwayTeam"]))
    duplicates = len(pairs) - len(set(pairs))
    missing_scores = int(df["FTHG"].isna().sum() + df["FTAG"].isna().sum())

    report = {
        "season": season_label(code),
        "schedule": schedule,
        "teams": n,
        "expected_matches": expected,
        "actual_matches": actual,
        "duplicate_fixtures": duplicates,
        "missing_scores": missing_scores,
        "complete": None if expected is None else actual == expected,
    }
    if duplicates:
        raise ValueError(
            f"{season_label(code)}: {duplicates} duplicated fixture(s) — "
            "the same ordered pair appears more than once."
        )
    if expected is None:
        return report
    if actual > expected:
        raise ValueError(
            f"{season_label(code)}: {actual} matches but {n} teams imply at most "
            f"{expected} under a {schedule} schedule. Either the parse is wrong or "
            "this competition is not a balanced round robin — pass schedule='unknown'."
        )
    if complete and actual < expected:
        raise ValueError(
            f"{season_label(code)}: parsed {actual} matches, expected {expected} "
            f"for {n} teams — {expected - actual} missing. This is the ragged-CSV "
            "failure mode; do not proceed with a partial season."
        )
    return report


def download_season(
    code: str, div: str = "E0", config: dict | None = None, offline: bool = False
) -> Path:
    cfg = config or load_config()
    raw_dir = resolve(cfg["data"]["raw_dir"]) / "clubs"
    url = BASE_URL.format(season=code, div=div)
    max_age = float(cfg["data"].get("cache_max_age_hours", 6))
    return fetch(url, raw_dir / f"{div}_{code}.csv", max_age, offline)


def load_seasons(
    codes: list[str],
    div: str = "E0",
    config: dict | None = None,
    offline: bool = False,
    incomplete_ok: tuple[str, ...] = (),
    schedule: str = "double_round_robin",
) -> tuple[pd.DataFrame, list[dict]]:
    """Download (or reuse cache) and parse several seasons.

    Returns the concatenated frame and one integrity report per season.
    ``incomplete_ok`` names season codes allowed to be mid-season.
    ``schedule`` selects the completeness identity — leave the default for
    balanced round-robin leagues, pass ``"unknown"`` otherwise.
    """
    frames, reports = [], []
    for code in codes:
        path = download_season(code, div, config, offline)
        df = select_columns(read_football_data_csv(path))
        df["season"] = season_label(code)
        df["division"] = div
        rep = season_integrity(
            df, code, complete=code not in incomplete_ok, schedule=schedule
        )
        reports.append(rep)
        frames.append(df)
        log.info(
            "%s %s: %d matches, %d teams%s",
            div, season_label(code), rep["actual_matches"], rep["teams"],
            "" if rep["complete"] else " (incomplete)",
        )
    out = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return out, reports
