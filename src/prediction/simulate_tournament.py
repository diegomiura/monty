"""Remaining-bracket forecast for the knockout stage of a tournament.

Reads the openfootball fixture list (cached raw file), resolves each
knockout slot to either a concrete team or a W<num>/L<num> reference, and
propagates probability distributions through the bracket by exact
enumeration — every match node holds a distribution over the team occupying
each slot, and pairwise advancement probabilities come from the same staged
knockout calculation used by predict.py. No Monte Carlo, so results are
deterministic and reproducible.

Documented assumptions:

* Features are frozen at the forecast cutoff: a hypothetical result earlier
  in a simulated path does not update Elo/form for later rounds. Every path
  is treated identically, so paths remain comparable; this understates the
  momentum a surprise winner would gain.
* Pair orientation is canonicalized (host country first, else alphabetical)
  so that A-vs-B and B-vs-A yield the same forecast.
* Completed matches dated on/after the cutoff day are re-predicted rather
  than resolved from their result (same conservative same-day rule as the
  feature builder). Retrospective cutoffs are only as leak-safe as the
  cached bracket file: once the source fills a later-round slot with a
  concrete team, the placeholder reference is gone.
* The third-place match is forecast like any other knockout match; its
  winner distribution is reported as P(third place).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.features.build_features import FeatureBuilder
from src.prediction.predict_match import predict_fixture

KNOCKOUT_ROUNDS = [
    "Round of 32",
    "Round of 16",
    "Quarter-final",
    "Semi-final",
    "Match for third place",
    "Final",
]
_PLACEHOLDER = re.compile(r"^([WL])(\d+)$")


def _norm_team(t) -> str:
    return t["name"] if isinstance(t, dict) else str(t)


def load_bracket(path: str | Path, resolve) -> list[dict]:
    """Knockout nodes from an openfootball worldcup.json file, sorted by
    match number. Slots are canonical team names or W<num>/L<num> tokens."""
    doc = json.loads(Path(path).read_text())
    nodes = []
    for m in doc["matches"]:
        rnd = m.get("round", "")
        if rnd not in KNOCKOUT_ROUNDS:
            continue
        slots = []
        for raw in (m["team1"], m["team2"]):
            name = _norm_team(raw)
            slots.append(name if _PLACEHOLDER.match(name) else resolve(name))
        if m.get("num") is None:
            raise ValueError(f"knockout entry without match number: {m.get('date')} {slots}")
        nodes.append(
            {
                "num": int(m["num"]),
                "date": pd.Timestamp(m["date"]),
                "round": rnd,
                "slot1": slots[0],
                "slot2": slots[1],
                "score": m.get("score"),
            }
        )
    if not nodes:
        raise ValueError(f"no knockout matches found in {path}")
    nodes.sort(key=lambda n: n["num"])
    for n in nodes:
        for s in (n["slot1"], n["slot2"]):
            if not _PLACEHOLDER.match(s) and re.match(r"^\d[A-L]", s):
                raise ValueError(
                    f"unresolved group placeholder '{s}' in match {n['num']}: "
                    "group-stage simulation is not supported — rerun once the "
                    "group stage is complete (or update the cached fixture file)"
                )
    return nodes


def node_result(node: dict) -> tuple[str, str] | None:
    """(winner, loser) of a completed knockout node. Penalties decide when
    present, then the after-extra-time score, then the 90' score. Returns
    None when the match has no decisive score yet."""
    sc = node.get("score") or {}
    for key in ("p", "et", "ft"):
        s = sc.get(key)
        if s and s[0] != s[1]:
            w = (node["slot1"], node["slot2"])
            return w if s[0] > s[1] else (w[1], w[0])
    return None


# ------------------------------------------------------------------ #
class PairPredictor:
    """Cached pairwise advancement probabilities from the match predictor.

    One FeatureBuilder (advanced to the cutoff once) is shared across all
    pairings; orientation is canonicalized so enumeration order cannot
    change the forecast."""

    def __init__(self, bundle, matches: pd.DataFrame, feature_cutoff: pd.Timestamp,
                 competition: str = "FIFA World Cup", hosts: tuple[str, ...] = ()):
        self.bundle = bundle
        self.matches = matches
        self.cutoff_day = pd.Timestamp(feature_cutoff.date())
        self.competition = competition
        self.hosts = set(hosts)
        self.fb = FeatureBuilder(bundle.cfg)
        self.fb.advance(matches, self.cutoff_day)
        self._cache: dict = {}
        self.predictions: dict = {}  # canonical (a, b, date) -> full prediction

    def _canonical(self, a: str, b: str) -> tuple[str, str]:
        a_host, b_host = a in self.hosts, b in self.hosts
        if a_host != b_host:
            return (a, b) if a_host else (b, a)
        return (a, b) if a <= b else (b, a)

    def advance_prob(self, a: str, b: str, node: dict) -> float:
        x, y = self._canonical(a, b)
        key = (x, y, str(node["date"].date()))
        if key not in self._cache:
            x_home = x in self.hosts and y not in self.hosts
            pred = predict_fixture(
                self.bundle, self.matches, x, y,
                kickoff=node["date"], competition=self.competition,
                stage="knockout", neutral=not x_home, team_a_home=x_home,
                feature_cutoff=self.cutoff_day, fb=self.fb,
            )
            self._cache[key] = float(pred["knockout"]["team_a_advances"])
            self.predictions[key] = pred
        p_x = self._cache[key]
        return p_x if a == x else 1.0 - p_x


# ------------------------------------------------------------------ #
def forecast_bracket(nodes: list[dict], pair_prob, cutoff_day: pd.Timestamp) -> dict:
    """Exact bracket enumeration. `pair_prob(a, b, node)` returns
    P(a eliminates b) for that node. Returns per-node winner/loser
    distributions plus per-team aggregate probabilities."""
    win: dict[int, dict[str, float]] = {}
    lose: dict[int, dict[str, float]] = {}
    reach: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pending: list[dict] = []

    def slot_dist(token: str) -> dict[str, float]:
        m = _PLACEHOLDER.match(token)
        if not m:
            return {token: 1.0}
        ref = int(m.group(2))
        if ref not in win:
            raise ValueError(f"slot references match {ref}, which comes later in the bracket")
        return win[ref] if m.group(1) == "W" else lose[ref]

    for node in nodes:
        d1, d2 = slot_dist(node["slot1"]), slot_dist(node["slot2"])
        for team, p in list(d1.items()) + list(d2.items()):
            reach[team][node["round"]] += p

        res = node_result(node)
        if res is not None and node["date"] < cutoff_day:
            w_team, l_team = res
            win[node["num"]] = {w_team: 1.0}
            lose[node["num"]] = {l_team: 1.0}
            continue

        pending.append(node)
        w, l = defaultdict(float), defaultdict(float)
        for a, pa in d1.items():
            for b, pb in d2.items():
                if a == b:
                    raise ValueError(f"team {a} appears in both slots of match {node['num']}")
                weight = pa * pb
                if weight <= 0:
                    continue
                p = pair_prob(a, b, node)
                w[a] += weight * p
                w[b] += weight * (1.0 - p)
                l[a] += weight * (1.0 - p)
                l[b] += weight * p
        win[node["num"]] = dict(w)
        lose[node["num"]] = dict(l)

    final = next(n for n in nodes if n["round"] == "Final")
    third = next((n for n in nodes if n["round"] == "Match for third place"), None)
    teams = sorted(reach)
    rows = []
    for t in teams:
        rows.append(
            {
                "team": t,
                **{
                    f"p_reach_{r.lower().replace(' ', '_').replace('-', '_')}": round(reach[t].get(r, 0.0), 4)
                    for r in KNOCKOUT_ROUNDS
                    if r != "Match for third place" and any(n["round"] == r for n in nodes)
                },
                "p_third_place": round(win[third["num"]].get(t, 0.0), 4) if third else None,
                "p_runner_up": round(lose[final["num"]].get(t, 0.0), 4),
                "p_champion": round(win[final["num"]].get(t, 0.0), 4),
            }
        )
    rows.sort(key=lambda r: (-r["p_champion"], -r["p_reach_final"], r["team"]))
    return {"teams": rows, "win": win, "lose": lose, "pending": pending}


def console_table(rows: list[dict], pending_n: int) -> str:
    active = [r for r in rows if r["p_champion"] > 0 or r["p_third_place"]]
    lines = [
        "=" * 74,
        f"Remaining-bracket forecast ({pending_n} matches left)",
        "-" * 74,
        f"{'team':<16} {'reach SF':>9} {'reach final':>12} {'3rd place':>10} {'CHAMPION':>10}",
    ]
    for r in active:
        lines.append(
            f"{r['team']:<16} {r.get('p_reach_semi_final', 0):>9.1%} "
            f"{r['p_reach_final']:>12.1%} {(r['p_third_place'] or 0):>10.1%} {r['p_champion']:>10.1%}"
        )
    lines.append("=" * 74)
    return "\n".join(lines)
