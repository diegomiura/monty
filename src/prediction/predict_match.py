"""Single-match prediction: features -> ensemble -> full JSON output."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS, FeatureBuilder
from src.models.train_pipeline import COMPONENT_NAMES
from src.prediction.score_matrix import build_matrix, markets, outcome_probs, top_scorelines
from src.prediction.simulate_knockout import knockout_probs


# ------------------------------------------------------------------ #
def confidence_score(comp_probs: dict, p: np.ndarray, X_row: pd.Series, n_a: float, n_b: float) -> dict:
    """Heuristic 0-100 confidence from model evidence (documented):
    * agreement: mean L1 distance between component probability vectors;
    * sharpness: 1 - normalized entropy of the ensemble distribution;
    * data volume: match-history depth of both teams;
    * feature completeness: share of non-missing features."""
    P = np.stack([comp_probs[n][0] for n in COMPONENT_NAMES])
    dis = np.mean([np.abs(P[i] - P[j]).sum() for i in range(len(P)) for j in range(i + 1, len(P))])
    agreement = max(0.0, 1.0 - dis)  # dis in [0, 2]
    ent = -np.sum(p * np.log(np.clip(p, 1e-12, 1)))
    sharpness = 1.0 - ent / np.log(3)
    volume = min(min(n_a, n_b) / 150.0, 1.0)
    completeness = 1.0 - float(X_row[FEATURE_COLUMNS].isna().mean())
    score = 100 * (0.4 * agreement + 0.2 * sharpness + 0.25 * volume + 0.15 * completeness)
    label = "low" if score < 40 else ("medium" if score < 65 else "high")
    notes = []
    if agreement < 0.8:
        notes.append("component models disagree noticeably")
    if volume < 0.5:
        notes.append("limited match history for at least one team")
    if completeness < 0.9:
        notes.append("some features missing")
    return {"label": label, "score": int(round(score)), "uncertainty_notes": notes}


def feature_contributions(bundle, X: pd.DataFrame, top_k: int = 5) -> list[dict]:
    """Per-match contributions from the logistic component: standardized
    feature value x (coef[win_a] - coef[win_b]) — positive favors team_a."""
    pipe = bundle.models["logistic"].base
    Z = pipe.named_steps["scaler"].transform(
        pipe.named_steps["imputer"].transform(X[FEATURE_COLUMNS].to_numpy())
    )
    lr = pipe.named_steps["lr"]
    contrib = Z[0] * (lr.coef_[0] - lr.coef_[2])
    order = np.argsort(-np.abs(contrib))
    out = []
    for i in order[:top_k]:
        out.append(
            {
                "feature": FEATURE_COLUMNS[i],
                "direction": "favors_team_a" if contrib[i] > 0 else "favors_team_b",
                "contribution": round(float(abs(contrib[i])), 4),
            }
        )
    return out


# ------------------------------------------------------------------ #
def predict_fixture(
    bundle,
    matches: pd.DataFrame,
    team_a: str,
    team_b: str,
    kickoff: pd.Timestamp,
    competition: str,
    stage: str = "",
    neutral: bool = True,
    team_a_home: bool = False,
    feature_cutoff: pd.Timestamp | None = None,
) -> dict:
    """Predict one fixture. `feature_cutoff` controls which completed
    matches feed Elo/form: frozen mode passes the tournament start; rolling
    mode passes the prediction cutoff. Only matches with date strictly
    before the cutoff's date are used (conservative same-day rule)."""
    cfg = bundle.cfg
    cutoff = feature_cutoff if feature_cutoff is not None else kickoff
    # Internal match dates are tz-naive day stamps; normalize both timestamps.
    cutoff_day = pd.Timestamp(cutoff.tz_convert("UTC").date()) if getattr(cutoff, "tzinfo", None) else pd.Timestamp(cutoff.date())
    kickoff_day = pd.Timestamp(kickoff.tz_convert("UTC").date()) if getattr(kickoff, "tzinfo", None) else pd.Timestamp(kickoff.date())
    fb = FeatureBuilder(cfg)
    fb.advance(matches, cutoff_day)
    row = fb.fixture_row(
        team_a, team_b, kickoff_day, competition, stage=stage, neutral=neutral, team_a_home=team_a_home
    )
    X = pd.DataFrame([row])

    comp_probs = bundle.component_probs(X)
    p = bundle.predict_proba(X)[0]
    la, lb = bundle.predict_lambdas(X)
    la, lb = float(la[0]), float(lb[0])

    sm_cfg = cfg["score_matrix"]
    m, tail = build_matrix(la, lb, int(sm_cfg["max_goals"]), float(sm_cfg["max_tail_mass"]))
    # Rescale exact scorelines to the calibrated W/D/L probabilities so the
    # displayed scores are consistent with the headline outcome probabilities.
    grid = m / m.sum()
    p_matrix = outcome_probs(m)
    scale = np.ones_like(grid)
    iu = np.triu_indices_from(grid, 1)
    il = np.tril_indices_from(grid, -1)
    di = np.diag_indices_from(grid)
    scale[il] = p[0] / max(p_matrix[0], 1e-12)
    scale[di] = p[1] / max(p_matrix[1], 1e-12)
    scale[iu] = p[2] / max(p_matrix[2], 1e-12)
    grid_cal = grid * scale
    grid_cal /= grid_cal.sum()

    mkts = markets(grid_cal, la, lb)
    scorelines = top_scorelines(grid_cal, 5)

    n_a = fb.teams[team_a].n_matches if team_a in fb.teams else 0
    n_b = fb.teams[team_b].n_matches if team_b in fb.teams else 0
    conf = confidence_score(comp_probs, p, X.iloc[0], n_a, n_b)

    result = {
        "team_a": team_a,
        "team_b": team_b,
        "match_date": str(kickoff.date()),
        "kickoff_timestamp_utc": kickoff.strftime("%Y-%m-%dT%H:%M:%SZ") if kickoff.tzinfo else str(kickoff),
        "competition": competition,
        "stage": stage or "unspecified",
        "neutral": neutral,
        "expected_goals": {"team_a": round(la, 3), "team_b": round(lb, 3)},
        "outcome_90": {
            "team_a_win": round(float(p[0]), 4),
            "draw": round(float(p[1]), 4),
            "team_b_win": round(float(p[2]), 4),
        },
        "most_likely_score": scorelines[0]["score"],
        "scorelines": scorelines,
        "additional_probabilities": {k: round(v, 4) for k, v in mkts.items()},
        "score_matrix_tail_mass": round(tail, 6),
        "confidence": conf,
        "important_features": feature_contributions(bundle, X),
        "component_probabilities": {
            n: [round(float(v), 4) for v in comp_probs[n][0]] for n in COMPONENT_NAMES
        },
        "model_information": {
            "model_version": bundle.model_version,
            "feature_version": bundle.feature_version,
            "training_cutoff": bundle.training_cutoff,
            "latest_match_used_in_training": bundle.latest_match_used,
            "feature_cutoff": str(cutoff_day.date()),
            "latest_match_in_features": str(matches[matches["date"] < cutoff_day]["date"].max().date()),
            "data_sources": ["martj42/international_results", "openfootball/worldcup.json"],
            "ensemble_weights": {n: round(float(w), 3) for n, w in zip(COMPONENT_NAMES, bundle.weights)},
            "calibration_used": bool(bundle.use_calibration),
        },
    }
    if (stage or "").lower() in {"knockout"} or row["knockout"] == 1.0:
        result["knockout"] = knockout_probs(
            la, lb, p, float(row["elo_expected_a"]), bundle.et_stats, cfg["knockout"]
        )
    # invariants
    s = sum(result["outcome_90"].values())
    assert abs(s - 1.0) < 1e-3, f"outcome probabilities sum to {s}"
    return result


def console_report(pred: dict) -> str:
    o = pred["outcome_90"]
    lines = [
        "=" * 62,
        f"{pred['team_a']}  vs  {pred['team_b']}",
        f"{pred['competition']} | {pred['stage']} | {pred['match_date']}"
        + (" | neutral venue" if pred["neutral"] else ""),
        "-" * 62,
        f"Win {pred['team_a']:<28} {o['team_a_win']:6.1%}",
        f"Draw{'':<28} {o['draw']:6.1%}",
        f"Win {pred['team_b']:<28} {o['team_b_win']:6.1%}",
        f"Expected goals: {pred['expected_goals']['team_a']:.2f} - {pred['expected_goals']['team_b']:.2f}"
        f"   Most likely score: {pred['most_likely_score']}",
        "Top scorelines: "
        + ", ".join(f"{s['score']} ({s['probability']:.1%})" for s in pred["scorelines"]),
    ]
    a = pred["additional_probabilities"]
    lines += [
        f"BTTS {a['both_teams_score']:.1%} | Over 2.5 {a['over_2_5']:.1%} | Under 2.5 {a['under_2_5']:.1%}",
        f"Clean sheet: {pred['team_a']} {a['team_a_clean_sheet']:.1%}, {pred['team_b']} {a['team_b_clean_sheet']:.1%}",
        f"Scores first: {pred['team_a']} {a['team_a_scores_first']:.1%}, {pred['team_b']} {a['team_b_scores_first']:.1%}",
    ]
    if "knockout" in pred:
        k = pred["knockout"]
        lines += [
            f"Advances: {pred['team_a']} {k['team_a_advances']:.1%}, {pred['team_b']} {k['team_b_advances']:.1%} "
            f"| Extra time {k['extra_time']:.1%} | Penalties {k['penalty_shootout']:.1%}",
        ]
    conf = pred["confidence"]
    lines += [
        f"Confidence: {conf['label']} ({conf['score']}/100)"
        + (f" — {'; '.join(conf['uncertainty_notes'])}" if conf["uncertainty_notes"] else ""),
        "Key features: "
        + ", ".join(f"{f['feature']} ({f['direction'].replace('favors_', '→')})" for f in pred["important_features"]),
        "=" * 62,
    ]
    return "\n".join(lines)
