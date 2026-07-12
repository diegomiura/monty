"""Render model predictions for the next World Cup round into README.md.

The README section between these markers is machine-generated:

    <!-- WC2026-PREDICTIONS:START -->
    <!-- WC2026-PREDICTIONS:END -->

``python update_readme.py`` re-renders it from the trained bundle. Fixtures
are auto-detected from the cached openfootball ``worldcup_2026.json``:
matches with no full-time score and both team slots resolved (placeholders
like ``W101``/``L102`` are skipped until the previous round completes).

The numbers shown are the same deterministic model outputs that predict.py
produces from the same bundle and dataset; this module never writes to the
append-only prediction log.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.data.clean import WC_HOSTS
from src.data.team_names import load_alias_table, make_resolver
from src.data.world_cup_2026 import wc2026_start
from src.prediction.predict_match import predict_fixture
from src.utils.config import resolve
from src.utils.dates import utc_now_iso

START_MARK = "<!-- WC2026-PREDICTIONS:START -->"
END_MARK = "<!-- WC2026-PREDICTIONS:END -->"

_PLACEHOLDER = re.compile(r"[WL]\d{1,3}")
_KICKOFF = re.compile(r"(\d{1,2}):(\d{2})\s*UTC([+-]\d{1,2})")

FLAGS = {
    "Algeria": "🇩🇿", "Argentina": "🇦🇷", "Australia": "🇦🇺", "Austria": "🇦🇹",
    "Belgium": "🇧🇪", "Bosnia and Herzegovina": "🇧🇦", "Brazil": "🇧🇷",
    "Cameroon": "🇨🇲", "Canada": "🇨🇦", "Cape Verde": "🇨🇻", "Chile": "🇨🇱",
    "Colombia": "🇨🇴", "Costa Rica": "🇨🇷", "Croatia": "🇭🇷", "Curacao": "🇨🇼",
    "Czechia": "🇨🇿", "Denmark": "🇩🇰", "DR Congo": "🇨🇩", "Ecuador": "🇪🇨",
    "Egypt": "🇪🇬", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "France": "🇫🇷", "Germany": "🇩🇪",
    "Ghana": "🇬🇭", "Greece": "🇬🇷", "Haiti": "🇭🇹", "Iran": "🇮🇷",
    "Iraq": "🇮🇶", "Italy": "🇮🇹", "Ivory Coast": "🇨🇮", "Jamaica": "🇯🇲",
    "Japan": "🇯🇵", "Jordan": "🇯🇴", "Mexico": "🇲🇽", "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱", "New Zealand": "🇳🇿", "Nigeria": "🇳🇬",
    "Norway": "🇳🇴", "Panama": "🇵🇦", "Paraguay": "🇵🇾", "Peru": "🇵🇪",
    "Poland": "🇵🇱", "Portugal": "🇵🇹", "Qatar": "🇶🇦", "Saudi Arabia": "🇸🇦",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Senegal": "🇸🇳", "Serbia": "🇷🇸",
    "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Spain": "🇪🇸",
    "Sweden": "🇸🇪", "Switzerland": "🇨🇭", "Tunisia": "🇹🇳", "Turkey": "🇹🇷",
    "Ukraine": "🇺🇦", "United States": "🇺🇸", "Uruguay": "🇺🇾",
    "Uzbekistan": "🇺🇿", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
}


def _flag(team: str) -> str:
    f = FLAGS.get(team, "")
    return f + " " if f else ""


def _norm_team(t) -> str:
    return t["name"] if isinstance(t, dict) else str(t)


def parse_kickoff_utc(date_str: str, time_str: str | None) -> tuple[pd.Timestamp, bool]:
    """openfootball times are local ('14:00 UTC-5'). Returns (kickoff in
    UTC, whether the time was known). Unknown times fall back to midnight
    UTC on the local date, flagged so the renderer omits the hour."""
    base = pd.Timestamp(date_str)
    m = _KICKOFF.match(time_str or "")
    if not m:
        return base.tz_localize("UTC"), False
    hh, mm, offset = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return (base + pd.Timedelta(hours=hh - offset, minutes=mm)).tz_localize("UTC"), True


def upcoming_fixtures(wc_json_path: Path, resolver) -> list[dict]:
    """Next-round fixtures: unplayed matches whose both team slots are
    resolved. Sorted by match number so the round reads in schedule order."""
    doc = json.loads(Path(wc_json_path).read_text())
    out = []
    for m in doc["matches"]:
        t1, t2 = _norm_team(m["team1"]), _norm_team(m["team2"])
        if _PLACEHOLDER.fullmatch(t1) or _PLACEHOLDER.fullmatch(t2):
            continue
        if "ft" in (m.get("score") or {}):
            continue
        kickoff, known = parse_kickoff_utc(m["date"], m.get("time"))
        out.append(
            {
                "team_a": resolver(t1),
                "team_b": resolver(t2),
                "round": m.get("round", ""),
                "ground": m.get("ground", ""),
                "local_date": pd.Timestamp(m["date"]),
                "kickoff_utc": kickoff,
                "kickoff_known": known,
                "num": m.get("num", 0),
            }
        )
    return sorted(out, key=lambda f: (f["num"], f["local_date"]))


def load_upcoming(cfg) -> list[dict]:
    raw = resolve(cfg["data"]["raw_dir"])
    resolver = make_resolver(load_alias_table(raw / "former_names.csv"))
    return upcoming_fixtures(raw / "worldcup_2026.json", resolver)


def predict_round(bundle, matches: pd.DataFrame, fixtures: list[dict], cfg, mode: str = "rolling") -> list[dict]:
    """Predict each fixture with the same conventions as predict.py: rolling
    features stop strictly before the fixture's local date, frozen at the
    tournament start. A host team is treated as the (non-neutral) home side."""
    preds = []
    for f in fixtures:
        hosts = WC_HOSTS.get(f["local_date"].year, set())
        team_a, team_b = f["team_a"], f["team_b"]
        if team_b in hosts and team_a not in hosts:
            team_a, team_b = team_b, team_a
        neutral = team_a not in hosts
        stage = "group" if "group" in f["round"].lower() else "knockout"
        cutoff = pd.Timestamp(wc2026_start(cfg)) if mode == "frozen" else f["local_date"]
        pred = predict_fixture(
            bundle, matches, team_a, team_b, f["kickoff_utc"],
            competition="FIFA World Cup", stage=stage, neutral=neutral,
            team_a_home=not neutral, feature_cutoff=cutoff,
        )
        pred["_fixture"] = f
        preds.append(pred)
    return preds


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _bar(p: float, width: int = 22) -> str:
    n = round(p * width)
    return "`" + "█" * n + "░" * (width - n) + "`"


def _score(s: str) -> str:
    return s.replace("-", "–")


def _render_match(pred: dict) -> str:
    f = pred["_fixture"]
    a, b = pred["team_a"], pred["team_b"]
    o = pred["outcome_90"]
    xg = pred["expected_goals"]
    add = pred["additional_probabilities"]
    conf = pred["confidence"]

    when = f["kickoff_utc"].strftime("%a %d %b %Y")
    if f["kickoff_known"]:
        when += f["kickoff_utc"].strftime(" · %H:%M UTC")
    lines = [
        f"### {_flag(a)}{a} vs {_flag(b)}{b}",
        "",
        f"**{f['round']} · {when} · {f['ground']}**",
        "",
        "| Result after 90 minutes | Probability | |",
        "|:--|--:|:--|",
        f"| **{a}** win | **{_pct(o['team_a_win'])}** | {_bar(o['team_a_win'])} |",
        f"| Draw | **{_pct(o['draw'])}** | {_bar(o['draw'])} |",
        f"| **{b}** win | **{_pct(o['team_b_win'])}** | {_bar(o['team_b_win'])} |",
        "",
        f"**Expected goals:** {a} **{xg['team_a']:.2f}** · {b} **{xg['team_b']:.2f}**"
        f" &nbsp;&nbsp;|&nbsp;&nbsp; **Most likely score:** {_score(pred['most_likely_score'])}",
        "",
        "| Top scorelines | Probability |",
        "|:--|--:|",
    ]
    for s in pred["scorelines"]:
        lines.append(f"| {_score(s['score'])} | {_pct(s['probability'])} |")
    lines += [
        "",
        "| Goals market | Probability |",
        "|:--|--:|",
        f"| Both teams to score | {_pct(add['both_teams_score'])} |",
        f"| Over 2.5 goals | {_pct(add['over_2_5'])} |",
        f"| Under 2.5 goals | {_pct(add['under_2_5'])} |",
        f"| {a} clean sheet | {_pct(add['team_a_clean_sheet'])} |",
        f"| {b} clean sheet | {_pct(add['team_b_clean_sheet'])} |",
        f"| {a} scores first | {_pct(add['team_a_scores_first'])} |",
        f"| {b} scores first | {_pct(add['team_b_scores_first'])} |",
    ]
    ko = pred.get("knockout")
    if ko:
        lines += [
            "",
            "| Advancement (incl. extra time & penalties) | Probability | |",
            "|:--|--:|:--|",
            f"| **{a} advance** | **{_pct(ko['team_a_advances'])}** | {_bar(ko['team_a_advances'])} |",
            f"| **{b} advance** | **{_pct(ko['team_b_advances'])}** | {_bar(ko['team_b_advances'])} |",
            f"| Extra time | {_pct(ko['extra_time'])} | |",
            f"| Penalty shootout | {_pct(ko['penalty_shootout'])} | |",
        ]
    lines += [
        "",
        f"_Model confidence: **{conf['label']}** ({conf['score']}/100)_",
    ]
    return "\n".join(lines)


def render_section(preds: list[dict], mode: str) -> str:
    """The full README block, markers included, so splicing is one replace."""
    if not preds:
        body = "_No upcoming fixtures with both teams decided — regenerate after the current round._"
        return f"{START_MARK}\n{body}\n{END_MARK}"
    info = preds[0]["model_information"]
    rounds = list(dict.fromkeys(p["_fixture"]["round"] for p in preds))
    label = " & ".join(rounds)
    if len(rounds) == 1 and len(preds) > 1 and label.endswith("final"):
        label += "s"
    header = [
        START_MARK,
        '<a id="next-round"></a>',
        "",
        f"## 🔮 Next round: {label} — model predictions",
        "",
        "_Auto-generated by `python update_readme.py` — do not edit this section by hand._",
        "",
        f"Generated **{utc_now_iso()}** · model **v{info['model_version']}** ({mode} mode)"
        f" · features through **{info['latest_match_in_features']}**."
        " Win/draw/win refers to the score after 90 minutes and sums to 1;"
        " advancement includes extra time and penalties. Every prediction is"
        " also stored in the append-only"
        " [prediction log](reports/predictions/world_cup_2026_predictions.csv).",
    ]
    parts = [_render_match(p) for p in preds]
    return "\n".join(header) + "\n\n" + "\n\n---\n\n".join(parts) + "\n\n" + END_MARK


def splice_readme(readme_text: str, section: str) -> str:
    """Replace the marker block (idempotent). If the markers are missing,
    insert the block just above '## Contents', or append as a fallback."""
    if START_MARK in readme_text and END_MARK in readme_text:
        pre = readme_text.split(START_MARK)[0]
        post = readme_text.split(END_MARK, 1)[1]
        return pre + section + post
    anchor = "## Contents"
    if anchor in readme_text:
        pre, post = readme_text.split(anchor, 1)
        return pre + section + "\n\n---\n\n" + anchor + post
    return readme_text.rstrip() + "\n\n" + section + "\n"
