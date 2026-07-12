"""Build the canonical match table from raw sources.

Output schema (data/processed/matches.csv) — one row per completed senior
men's international match, sorted chronologically:

    match_id, date, team_a, team_b, goals_a_90, goals_b_90,
    goals_a_extra_time, goals_b_extra_time, penalty_goals_a, penalty_goals_b,
    competition, stage, group, neutral, venue, host_team_a, host_team_b,
    went_to_extra_time, went_to_penalties, winner_90, goals_90_confirmed,
    match_status, source, retrieved_at

team_a is the (nominal) home side from the source; `neutral` marks neutral
venues. Regulation, extra-time and shootout goals are kept strictly separate.

Extra-time handling (documented):
  * The primary source records post-extra-time scores for knockout matches
    (never penalties). For FIFA World Cup matches we correct to true
    90-minute scores using openfootball's ft/et/p splits.
  * Matches identified in shootouts.csv (any competition) are draws after
    both 90' and extra time; the recorded (draw) score is kept as the
    90-minute score with goals_90_confirmed=False when we lack the ft split.
  * Non-World-Cup knockout matches decided in extra time WITHOUT a shootout
    cannot be identified from the primary source and remain recorded with
    their post-ET score. This is a documented limitation affecting a small
    fraction of matches (see reports/data_audit).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.team_names import load_alias_table, make_resolver
from src.utils.logging import get_logger

log = get_logger(__name__)

FRIENDLY_LIKE = {"Friendly", "FIFA Series"}


def _norm_team(t) -> str:
    return t["name"] if isinstance(t, dict) else str(t)


def load_openfootball_wc(path: Path, resolve) -> pd.DataFrame:
    """Flatten one openfootball worldcup.json file (schema: {name, matches})."""
    doc = json.loads(Path(path).read_text())
    rows = []
    for m in doc["matches"]:
        t1, t2 = _norm_team(m["team1"]), _norm_team(m["team2"])
        if any(len(t) <= 4 and (t[0] in "WL" and t[1:].isdigit()) for t in (t1, t2)):
            continue  # unresolved knockout placeholder (e.g. W99)
        score = m.get("score") or {}
        if "ft" not in score:
            continue  # not played yet
        ft, et, pen = score["ft"], score.get("et"), score.get("p")
        rows.append(
            {
                "date": pd.Timestamp(m["date"]),
                "team_a": resolve(t1),
                "team_b": resolve(t2),
                "of_ft_a": ft[0],
                "of_ft_b": ft[1],
                "of_et_a": (et[0] - ft[0]) if et else 0,
                "of_et_b": (et[1] - ft[1]) if et else 0,
                "of_pen_a": pen[0] if pen else np.nan,
                "of_pen_b": pen[1] if pen else np.nan,
                "of_went_et": et is not None,
                "of_went_pens": pen is not None,
                "stage": "Group" if m.get("group") else m.get("round", ""),
                "group": (m.get("group") or "").replace("Group ", ""),
                "of_venue": m.get("ground", ""),
            }
        )
    return pd.DataFrame(rows)


# Hosts per World Cup edition, used only for rows appended from openfootball
# when the primary source lags behind (see build_canonical). Limitation: a
# co-host playing inside another host's country would be mislabelled
# non-neutral; no such appended row can occur for 2026 (all three hosts were
# eliminated before the primary source started lagging).
WC_HOSTS = {
    2010: {"South Africa"},
    2014: {"Brazil"},
    2018: {"Russia"},
    2022: {"Qatar"},
    2026: {"United States", "Mexico", "Canada"},
}


def build_canonical(raw_paths: dict[str, Path], retrieved_at: str) -> tuple[pd.DataFrame, dict]:
    """Return (matches, audit_info)."""
    aliases = load_alias_table(raw_paths.get("former_names"))
    resolve = make_resolver(aliases)

    res = pd.read_csv(raw_paths["results"])
    res["date"] = pd.to_datetime(res["date"])
    res["home_team"] = res["home_team"].map(resolve)
    res["away_team"] = res["away_team"].map(resolve)

    completed = res[res["home_score"].notna() & res["away_score"].notna()].copy()
    fixtures = res[res["home_score"].isna()].copy()

    df = pd.DataFrame(
        {
            "date": completed["date"],
            "team_a": completed["home_team"],
            "team_b": completed["away_team"],
            "goals_a_90": completed["home_score"].astype(int),
            "goals_b_90": completed["away_score"].astype(int),
            "competition": completed["tournament"],
            "venue": completed["city"].fillna("") + ", " + completed["country"].fillna(""),
            "host_country": completed["country"],
            "neutral": completed["neutral"].astype(bool),
        }
    )
    df["stage"] = ""
    df["group"] = ""
    df["goals_a_extra_time"] = 0
    df["goals_b_extra_time"] = 0
    df["penalty_goals_a"] = np.nan
    df["penalty_goals_b"] = np.nan
    df["went_to_extra_time"] = False
    df["went_to_penalties"] = False
    df["goals_90_confirmed"] = True

    # --- shootout flags (all competitions) -------------------------------
    sh = pd.read_csv(raw_paths["shootouts"])
    sh["date"] = pd.to_datetime(sh["date"])
    sh["home_team"] = sh["home_team"].map(resolve)
    sh["away_team"] = sh["away_team"].map(resolve)
    sh_keys = set(zip(sh["date"], sh["home_team"], sh["away_team"]))
    key = list(zip(df["date"], df["team_a"], df["team_b"]))
    is_shootout = pd.Series([k in sh_keys for k in key], index=df.index)
    df.loc[is_shootout, "went_to_penalties"] = True
    # A shootout implies the match was level after 90' and extra time. The
    # recorded score is the post-ET (draw) score; exact 90' split unconfirmed.
    df.loc[is_shootout, "goals_90_confirmed"] = False

    # --- World Cup regulation-score corrections from openfootball --------
    of_frames = []
    for name, path in raw_paths.items():
        if name.startswith("worldcup_"):
            of_frames.append(load_openfootball_wc(path, resolve))
    n_corrected = 0
    unmatched_of = []
    extra_rows = []
    if of_frames:
        of = pd.concat(of_frames, ignore_index=True)
        idx = {}
        for i, (d, a, b) in enumerate(key):
            idx[(d, a, b)] = df.index[i]
            idx.setdefault((d, b, a), df.index[i])  # tolerate swapped order
        for _, r in of.iterrows():
            j = idx.get((r["date"], r["team_a"], r["team_b"]))
            if j is None:
                # Completed WC match the primary source does not have yet
                # (martj42 typically lags openfootball by a day or two during
                # a tournament). Append it as a full row so rolling-mode
                # features stay current; once the primary catches up the
                # match keys collide and this branch no longer fires, so a
                # refresh can never duplicate it.
                unmatched_of.append(f"{r['date'].date()} {r['team_a']} v {r['team_b']}")
                hosts = WC_HOSTS.get(r["date"].year, set())
                host = next((t for t in (r["team_a"], r["team_b"]) if t in hosts), None)
                extra_rows.append(
                    {
                        "date": r["date"],
                        "team_a": r["team_a"],
                        "team_b": r["team_b"],
                        "goals_a_90": int(r["of_ft_a"]),
                        "goals_b_90": int(r["of_ft_b"]),
                        "competition": "FIFA World Cup",
                        "venue": r["of_venue"],
                        "host_country": host or "",
                        "neutral": host is None,
                        "stage": r["stage"],
                        "group": r["group"],
                        "goals_a_extra_time": int(r["of_et_a"]),
                        "goals_b_extra_time": int(r["of_et_b"]),
                        "penalty_goals_a": r["of_pen_a"],
                        "penalty_goals_b": r["of_pen_b"],
                        "went_to_extra_time": bool(r["of_went_et"]),
                        "went_to_penalties": bool(r["of_went_pens"]),
                        "goals_90_confirmed": True,
                        "_of_only": True,
                    }
                )
                continue
            swapped = df.at[j, "team_a"] != r["team_a"]
            sfx = ("_b", "_a") if swapped else ("_a", "_b")
            df.at[j, "stage"] = r["stage"]
            df.at[j, "group"] = r["group"]
            if r["of_went_et"]:
                df.at[j, f"goals{sfx[0]}_90"] = r["of_ft_a"]
                df.at[j, f"goals{sfx[1]}_90"] = r["of_ft_b"]
                df.at[j, f"goals{sfx[0]}_extra_time"] = r["of_et_a"]
                df.at[j, f"goals{sfx[1]}_extra_time"] = r["of_et_b"]
                df.at[j, "went_to_extra_time"] = True
                df.at[j, "goals_90_confirmed"] = True
                n_corrected += 1
            if r["of_went_pens"]:
                df.at[j, "went_to_penalties"] = True
                df.at[j, f"penalty_goals{sfx[0]}"] = r["of_pen_a"]
                df.at[j, f"penalty_goals{sfx[1]}"] = r["of_pen_b"]
    # Any shootout match necessarily reached extra time in competitions that
    # play one (Copa América historically often went straight to penalties;
    # we therefore do NOT force went_to_extra_time for non-WC shootouts).

    if extra_rows:
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
        log.info(
            "appended %d completed WC match(es) missing from primary: %s",
            len(extra_rows),
            unmatched_of,
        )

    # --- derived fields ---------------------------------------------------
    df["winner_90"] = np.select(
        [df["goals_a_90"] > df["goals_b_90"], df["goals_a_90"] < df["goals_b_90"]],
        ["team_a", "team_b"],
        default="draw",
    )
    df["host_team_a"] = (~df["neutral"]) & (df["team_a"] == df["host_country"])
    df["host_team_b"] = (~df["neutral"]) & (df["team_b"] == df["host_country"])
    df["match_status"] = "completed"
    df["source"] = "martj42_international_results+openfootball_wc"
    if "_of_only" in df.columns:
        of_only = df["_of_only"].fillna(False).astype(bool)
        df.loc[of_only, "source"] = "openfootball_wc_only"
        df = df.drop(columns="_of_only")
    df["retrieved_at"] = retrieved_at

    df = df.sort_values(["date", "team_a", "team_b"], kind="mergesort").reset_index(drop=True)
    df.insert(
        0,
        "match_id",
        [
            f"{d.date()}_{a.replace(' ', '')}_{b.replace(' ', '')}"
            for d, a, b in zip(df["date"], df["team_a"], df["team_b"])
        ],
    )
    dup = df["match_id"].duplicated(keep=False)
    if dup.any():  # same pair twice on one day: disambiguate deterministically
        df.loc[dup, "match_id"] = (
            df.loc[dup, "match_id"] + "_" + df.loc[dup].groupby("match_id").cumcount().astype(str)
        )

    audit = {
        "n_matches": int(len(df)),
        "n_teams": int(len(set(df["team_a"]) | set(df["team_b"]))),
        "first_date": str(df["date"].min().date()),
        "last_date": str(df["date"].max().date()),
        "n_shootouts_flagged": int(df["went_to_penalties"].sum()),
        "n_wc_et_corrections": n_corrected,
        "n_unconfirmed_90min_scores": int((~df["goals_90_confirmed"]).sum()),
        "openfootball_rows_unmatched_in_primary": unmatched_of,
        "n_appended_from_openfootball": len(extra_rows),
        "n_scheduled_fixtures_excluded": int(len(fixtures)),
        "missing_rate": {
            c: float(df[c].isna().mean()) for c in df.columns if df[c].isna().any()
        },
    }
    return df, audit
