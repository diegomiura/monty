"""Score the append-only prediction log against completed matches.

This is the project's live track record: every row in
``reports/predictions/world_cup_2026_predictions.csv`` was written *before*
kickoff by ``predict.py``, so grading it after the fact is the only honest
measure of forecasting performance. (The World Cup backtests in
``reports/backtests/`` are a much larger sample but are retrospective — the
model was fitted and locked with hindsight about which tournament it would
be evaluated on. The two must never be pooled.)

Append-only guarantee
---------------------
Grading fills the evaluation columns that ``predict.py`` deliberately leaves
empty (``actual_score``, ``brier_score``, ...). The prediction columns are
never rewritten; :func:`write_graded_log` asserts this before touching disk,
so a bug here fails loudly instead of quietly revising history.

90-minute convention
--------------------
Every probability in the log refers to the score after 90 minutes, so
grading uses ``goals_a_90``/``goals_b_90`` only. A knockout tie that was
settled in extra time or on penalties is graded as a **draw** — the team
that lifted the trophy may well be marked as "not our pick". That is
correct, and the renderer annotates those rows so the table cannot be
misread.

Metric definitions (per match, consistent with ``src.evaluation.metrics``):
  * ``brier_score``          sum of squared errors over all three outcomes
  * ``log_loss``             -log(probability assigned to the actual outcome)
  * ``goal_absolute_error``  |xG_a - goals_a| + |xG_b - goals_b|
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import resolve
from src.utils.dates import utc_now_iso

LOG_PATH = "reports/predictions/world_cup_2026_predictions.csv"

START_MARK = "<!-- TRACK-RECORD:START -->"
END_MARK = "<!-- TRACK-RECORD:END -->"

#: Written once by predict.py and never rewritten.
PREDICTION_COLUMNS = [
    "match", "kickoff", "prediction_created_at", "prediction_cutoff",
    "predicted_team_a_win", "predicted_draw", "predicted_team_b_win",
    "predicted_score", "expected_goals_a", "expected_goals_b",
    "model_version", "feature_version", "latest_match_used", "mode",
]
#: Left empty at prediction time, filled in here once the result is known.
EVALUATION_COLUMNS = [
    "actual_score", "actual_result", "brier_score", "log_loss",
    "goal_absolute_error", "correct_result", "correct_exact_score",
]

PROB_COLUMNS = ["predicted_team_a_win", "predicted_draw", "predicted_team_b_win"]
OUTCOMES = ["team_a", "draw", "team_b"]
EPS = 1e-12

_MATCH_RE = re.compile(r"^(?P<a>.+?) v (?P<b>.+)$")


def load_log(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else resolve(LOG_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"No prediction log at {p}. It is written by predict.py when you "
            "predict a 2026 World Cup fixture."
        )
    return pd.read_csv(p)


def _teams(match: str) -> tuple[str, str] | None:
    m = _MATCH_RE.match(str(match))
    return (m.group("a").strip(), m.group("b").strip()) if m else None


def _outcome_index(goals_a: int, goals_b: int) -> int:
    """0 = team_a win, 1 = draw, 2 = team_b win (metrics.py column order)."""
    if goals_a > goals_b:
        return 0
    return 1 if goals_a == goals_b else 2


def find_result(
    match: str, kickoff: str, matches: pd.DataFrame, tol_days: int = 1
) -> dict | None:
    """Locate the completed match for one log row, oriented like the log.

    The log stores a UTC kickoff while the canonical table stores a local
    match date, so dates are matched within ``tol_days``. If the canonical
    row has the opposite home/away orientation, goals are flipped so the
    returned values line up with the logged probabilities.
    """
    pair = _teams(match)
    if pair is None:
        return None
    a, b = pair
    when = pd.Timestamp(kickoff)
    if when.tzinfo is not None:
        when = when.tz_convert("UTC").tz_localize(None)

    same_pair = matches[
        ((matches["team_a"] == a) & (matches["team_b"] == b))
        | ((matches["team_a"] == b) & (matches["team_b"] == a))
    ]
    if same_pair.empty:
        return None
    near = same_pair[(same_pair["date"] - when).abs() <= pd.Timedelta(days=tol_days)]
    if near.empty:
        return None
    # Closest kickoff wins if a pair somehow met twice inside the window.
    row = near.loc[(near["date"] - when).abs().idxmin()]

    flipped = row["team_a"] != a
    goals_a = int(row["goals_b_90"] if flipped else row["goals_a_90"])
    goals_b = int(row["goals_a_90"] if flipped else row["goals_b_90"])
    return {
        "goals_a": goals_a,
        "goals_b": goals_b,
        "date": row["date"],
        "stage": row.get("stage", "") or "",
        "went_to_extra_time": bool(row.get("went_to_extra_time", False)),
        "went_to_penalties": bool(row.get("went_to_penalties", False)),
        "goals_90_confirmed": bool(row.get("goals_90_confirmed", True)),
    }


def grade(log: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Return the log with evaluation columns filled where the result is known.

    Rows whose fixture was never played (e.g. a speculative final that the
    bracket did not produce) stay ungraded with ``graded=False`` rather than
    being force-matched to some other match.

    Evaluation columns are built as whole columns with explicit dtypes rather
    than written cell-by-cell: an unpopulated log reads back with those
    columns typed as ``float64`` (all-NaN) or ``str`` (all-empty), and pandas
    >= 3 raises rather than silently widening the dtype on assignment.
    """
    out = log.copy()
    cols: dict[str, list] = {c: [] for c in EVALUATION_COLUMNS}
    graded_flags, et_flags, pen_flags, stages = [], [], [], []

    for _, row in out.iterrows():
        res = find_result(row["match"], row["kickoff"], matches)
        if res is None:
            for c in EVALUATION_COLUMNS:
                cols[c].append(np.nan)
            graded_flags.append(False)
            et_flags.append(False)
            pen_flags.append(False)
            stages.append("")
            continue

        p = np.array([float(row[c]) for c in PROB_COLUMNS], dtype=float)
        y = _outcome_index(res["goals_a"], res["goals_b"])
        onehot = np.zeros(3)
        onehot[y] = 1.0
        actual_score = f"{res['goals_a']}-{res['goals_b']}"

        cols["actual_score"].append(actual_score)
        cols["actual_result"].append(OUTCOMES[y])
        cols["brier_score"].append(round(float(np.sum((p - onehot) ** 2)), 6))
        cols["log_loss"].append(round(float(-np.log(max(p[y], EPS))), 6))
        cols["goal_absolute_error"].append(
            round(
                abs(float(row["expected_goals_a"]) - res["goals_a"])
                + abs(float(row["expected_goals_b"]) - res["goals_b"]),
                6,
            )
        )
        cols["correct_result"].append(bool(int(np.argmax(p)) == y))
        cols["correct_exact_score"].append(bool(str(row["predicted_score"]) == actual_score))
        graded_flags.append(True)
        et_flags.append(res["went_to_extra_time"])
        pen_flags.append(res["went_to_penalties"])
        stages.append(res["stage"])

    for c in ("actual_score", "actual_result", "correct_result", "correct_exact_score"):
        out[c] = pd.Series(cols[c], index=out.index, dtype=object)
    for c in ("brier_score", "log_loss", "goal_absolute_error"):
        out[c] = pd.Series(cols[c], index=out.index, dtype=float)
    out["graded"] = pd.Series(graded_flags, index=out.index, dtype=bool)
    out["went_to_extra_time"] = pd.Series(et_flags, index=out.index, dtype=bool)
    out["went_to_penalties"] = pd.Series(pen_flags, index=out.index, dtype=bool)
    out["actual_stage"] = pd.Series(stages, index=out.index, dtype=object)
    return out


def summarize(graded: pd.DataFrame) -> dict:
    """Aggregate the graded rows. Uniform-guess log loss (ln 3 = 1.0986) is
    reported alongside as the zero-skill anchor."""
    g = graded[graded["graded"]]
    n = len(g)
    if n == 0:
        return {"n": 0, "n_ungraded": int((~graded["graded"]).sum())}
    return {
        "n": n,
        "n_ungraded": int((~graded["graded"]).sum()),
        "correct_results": int(g["correct_result"].sum()),
        "accuracy": round(float(g["correct_result"].mean()), 4),
        "exact_scores": int(g["correct_exact_score"].sum()),
        "log_loss": round(float(g["log_loss"].mean()), 4),
        "brier": round(float(g["brier_score"].mean()), 4),
        "goal_abs_error": round(float(g["goal_absolute_error"].mean()), 4),
        "uniform_log_loss": round(float(np.log(3)), 4),
        "first_kickoff": str(pd.Timestamp(g["kickoff"].min()).date()),
        "last_kickoff": str(pd.Timestamp(g["kickoff"].max()).date()),
    }


def write_graded_log(graded: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Persist evaluation columns back into the log, in place.

    Raises if any prediction column would change — the append-only rule is
    enforced here rather than trusted.

    The on-disk column order is preserved exactly: ``predict.py`` appends new
    rows with a ``csv.DictWriter`` over a fixed field list, so reordering the
    header here would silently misalign every future appended prediction.
    """
    p = Path(path) if path else resolve(LOG_PATH)
    original = pd.read_csv(p)
    if len(original) != len(graded):
        raise ValueError(
            f"log has {len(original)} rows but {len(graded)} were graded; refusing to write"
        )
    for col in PREDICTION_COLUMNS:
        before = original[col].astype(str).fillna("")
        after = graded[col].astype(str).fillna("")
        if not before.equals(after):
            raise ValueError(
                f"refusing to write: prediction column '{col}' would change. "
                "Logged predictions are append-only."
            )
    columns = list(original.columns) + [
        c for c in EVALUATION_COLUMNS if c not in original.columns
    ]
    graded[columns].to_csv(p, index=False)
    return p


# --------------------------------------------------------------------- #
# README rendering
# --------------------------------------------------------------------- #
def _pct(p: float) -> str:
    return f"{p * 100:.0f}%"


def _score(s: str) -> str:
    return str(s).replace("-", "–")


def _pick(row) -> tuple[str, float]:
    """The model's headline call and the probability it carried."""
    pair = _teams(row["match"])
    probs = [float(row[c]) for c in PROB_COLUMNS]
    i = int(np.argmax(probs))
    label = "Draw" if i == 1 else (pair[i // 2] if pair else OUTCOMES[i])
    return label, probs[i]


def _actual_label(row) -> str:
    pair = _teams(row["match"])
    res = row["actual_result"]
    if res == "draw":
        return "Draw"
    if pair is None:
        return str(res)
    return pair[0] if res == "team_a" else pair[1]


def render_section(graded: pd.DataFrame, limit: int = 10) -> str:
    """The full README block, markers included, so splicing is one replace."""
    g = graded[graded["graded"]].copy()
    if g.empty:
        body = (
            "_No logged prediction has a completed result yet — "
            "run `python update_track_record.py` after the next match._"
        )
        return f"{START_MARK}\n{body}\n{END_MARK}"

    g["_k"] = pd.to_datetime(g["kickoff"], format="mixed", utc=True)
    g = g.sort_values("_k", ascending=False).head(limit)
    s = summarize(graded)

    lines = [
        START_MARK,
        '<a id="track-record"></a>',
        "",
        "## 📊 Track record — how the predictions actually did",
        "",
        "_Auto-generated by `python update_track_record.py` — do not edit this section by hand._",
        "",
        f"Every row was written to the append-only "
        f"[prediction log]({LOG_PATH}) **before kickoff**, then graded "
        f"against the official result. Predictions are never rewritten — only "
        f"the result columns are filled in. Generated **{utc_now_iso()}**.",
        "",
        f"**{s['correct_results']}/{s['n']} results correct** · "
        f"**{s['exact_scores']}/{s['n']} exact scores** · "
        f"log loss **{s['log_loss']}** (a uniform 33/33/33 guess scores "
        f"{s['uniform_log_loss']}) · Brier **{s['brier']}**",
        "",
        "| Match | Mode | Model's pick | Actual (90 min) | Result |",
        "|:--|:--|:--|:--|:--:|",
    ]

    for _, row in g.iterrows():
        pick, prob = _pick(row)
        note = ""
        if row["went_to_penalties"]:
            note = " _(pens)_"
        elif row["went_to_extra_time"]:
            note = " _(a.e.t.)_"
        hit = "✅" if row["correct_result"] else "❌"
        if row["correct_exact_score"]:
            hit += " 🎯"
        actual = f"{_actual_label(row)} {_score(row['actual_score'])}{note}"
        lines.append(
            f"| {row['match']} | {row['mode']} | {pick} {_pct(prob)} | {actual} | {hit} |"
        )

    lines += [
        "",
        "✅ correct result · ❌ wrong · 🎯 exact scoreline predicted · "
        "_a.e.t._ / _pens_ = level after 90 minutes and settled later; graded "
        "on the 90-minute score, which is what the model forecasts.",
        "",
        "A ❌ 🎯 row is not a contradiction: the most likely *single scoreline* "
        "can be 1–1 while the most likely *outcome* is still a win, because "
        "the win probability sums over many scorelines.",
    ]
    if s["log_loss"] > s["uniform_log_loss"]:
        misses = graded[graded["graded"] & ~graded["correct_result"].astype(bool)]
        drawn = int((misses["actual_result"] == "draw").sum())
        detail = ""
        if drawn:
            detail = (
                f" {drawn} of the {len(misses)} misses finished level after 90 "
                "minutes while the model favoured one side"
                + (" (all settled in extra time or on penalties)."
                   if bool(misses["went_to_extra_time"].all()) else ".")
            )
        lines += [
            "",
            f"**These predictions scored worse than a uniform guess** "
            f"({s['log_loss']} vs {s['uniform_log_loss']} log loss).{detail} "
            "Reported as-is.",
        ]
    if s["n_ungraded"]:
        lines.append(
            f"\n{s['n_ungraded']} logged prediction(s) not shown: the fixture "
            "was never played (a speculative matchup the bracket did not produce)."
        )
    lines += [
        "",
        f"> **Sample size: {s['n']} matches.** Far too small to judge "
        "calibration — one result swings every number in that summary. The "
        "meaningful evaluation is the 290-match World Cup backtest in "
        "[Results](#results); this table is a live, pre-registered honesty "
        "check, not a performance claim.",
        "",
        END_MARK,
    ]
    return "\n".join(lines)


_GENERATED_LINE = re.compile(r"^Every row was written.*$", re.MULTILINE)


def current_section(text: str) -> str | None:
    if START_MARK in text and END_MARK in text:
        return text.split(START_MARK, 1)[1].split(END_MARK, 1)[0]
    return None


def section_equivalent(a: str | None, b: str | None) -> bool:
    """True when two sections differ only by the generation timestamp."""
    if a is None or b is None:
        return False
    return _GENERATED_LINE.sub("", a).strip() == _GENERATED_LINE.sub("", b).strip()


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
