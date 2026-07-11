"""Chronological validation and World Cup backtests (frozen & rolling)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.evaluation.metrics import goal_metrics, log_loss, result_metrics
from src.features.build_features import FeatureBuilder, build_feature_table
from src.models.baselines import FrequencyBaseline, HigherEloBaseline
from src.models.train_pipeline import COMPONENT_NAMES, fit_components, component_probs, train_bundle
from src.utils.config import load_config, resolve
from src.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------- #
# Expanding-window chronological validation
# --------------------------------------------------------------------- #
def chronological_validation(feats: pd.DataFrame, cfg: dict, val_years: list[int]) -> pd.DataFrame:
    """Train through Y-1, validate on year Y, for each Y. Returns a tidy
    metric table for baselines + components."""
    train_start = pd.Timestamp(cfg["features"]["train_start"])
    rows = []
    for year in val_years:
        t0, t1 = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
        tr = feats[(feats["date"] >= train_start) & (feats["date"] < t0)]
        va = feats[(feats["date"] >= t0) & (feats["date"] < t1)]
        if len(tr) < 2000 or len(va) < 100:
            log.warning("skipping fold %s (train=%d, val=%d)", year, len(tr), len(va))
            continue
        y_tr, y_va = tr["outcome"].to_numpy(), va["outcome"].to_numpy()
        models = {
            "frequency": FrequencyBaseline().fit(tr, y_tr),
            "higher_elo": HigherEloBaseline().fit(tr, y_tr),
        }
        models.update(fit_components(tr, cfg))
        for name, model in models.items():
            p = model.predict_proba(va)
            rows.append({"fold": year, "model": name, **result_metrics(y_va, p)})
        log.info("fold %d done (train=%d, val=%d)", year, len(tr), len(va))
    return pd.DataFrame(rows)


def summarize_validation(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby("model")[["log_loss", "brier", "rps", "accuracy", "ece"]]
        .mean()
        .round(4)
        .sort_values("log_loss")
    )


# --------------------------------------------------------------------- #
# World Cup backtests
# --------------------------------------------------------------------- #
def wc_matches_for_year(matches: pd.DataFrame, year: int) -> pd.DataFrame:
    wc = matches[matches["competition"] == "FIFA World Cup"].copy()
    wc = wc[wc["date"].dt.year == year]
    if wc.empty:
        raise ValueError(f"No FIFA World Cup matches found for {year}")
    return wc


def frozen_features(matches: pd.DataFrame, wc: pd.DataFrame, cutoff: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    """Fixture rows for every tournament match with team state frozen at the
    tournament start — tournament matches never update features."""
    fb = FeatureBuilder(cfg)
    fb.advance(matches, cutoff)
    rows = []
    for m in wc.itertuples():
        row = fb.fixture_row(
            m.team_a, m.team_b, m.date, m.competition,
            stage=m.stage or "", neutral=bool(m.neutral), team_a_home=not bool(m.neutral),
        )
        rows.append(row)
    out = pd.DataFrame(rows, index=wc.index)
    out["outcome"] = np.select([wc["winner_90"] == "team_a", wc["winner_90"] == "draw"], [0, 1], 2)
    out["date"] = wc["date"].values
    return out


def rolling_features(feats: pd.DataFrame, wc: pd.DataFrame) -> pd.DataFrame:
    """Rolling mode = the standard chronological pass: each tournament match
    sees all matches completed on earlier days (never same-day, never later)."""
    return feats.loc[wc.index]


def world_cup_backtest(
    year: int,
    mode: str,
    matches: pd.DataFrame,
    feats: pd.DataFrame,
    cfg: dict,
    bundle=None,
) -> dict:
    wc = wc_matches_for_year(matches, year)
    cutoff = wc["date"].min()  # first tournament match day; training uses < cutoff
    if bundle is None:
        bundle = train_bundle(feats, matches, cutoff, cfg)

    if mode == "frozen":
        X = frozen_features(matches, wc, cutoff, cfg)
    elif mode == "rolling":
        X = rolling_features(feats, wc)
    else:
        raise ValueError("mode must be frozen or rolling")

    y = np.select([wc["winner_90"] == "team_a", wc["winner_90"] == "draw"], [0, 1], 2)
    p = bundle.predict_proba(X)
    la, lb = bundle.predict_lambdas(X)

    comp = bundle.component_probs(X)
    per_component = {n: result_metrics(y, comp[n]) for n in COMPONENT_NAMES}

    ko_mask = wc["stage"].apply(
        lambda s: s in {"Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Third place", "Final"}
    ).to_numpy()

    from src.evaluation.world_cup_report import breakdowns, figures

    extra_breakdowns = breakdowns(wc, y, p, X)
    figures(year, mode, y, p,
            (wc["goals_a_90"] + wc["goals_b_90"]).to_numpy(float), la + lb)

    report = {
        "tournament": f"FIFA World Cup {year}",
        "mode": mode,
        "training_cutoff": bundle.training_cutoff,
        "latest_match_used": bundle.latest_match_used,
        "n_matches": int(len(wc)),
        "ensemble": result_metrics(y, p),
        "goals": goal_metrics(
            wc["goals_a_90"].to_numpy(float), wc["goals_b_90"].to_numpy(float), la, lb
        ),
        "by_stage": {
            "group": result_metrics(y[~ko_mask], p[~ko_mask]) if (~ko_mask).any() else None,
            "knockout": result_metrics(y[ko_mask], p[ko_mask]) if ko_mask.any() else None,
        },
        "components": per_component,
        "breakdowns": extra_breakdowns,
        "ensemble_weights": dict(zip(COMPONENT_NAMES, np.round(bundle.weights, 3).tolist())),
        "calibration_used": bool(bundle.use_calibration),
    }
    preds = pd.DataFrame(
        {
            "date": wc["date"].dt.date,
            "team_a": wc["team_a"],
            "team_b": wc["team_b"],
            "stage": wc["stage"],
            "p_team_a": p[:, 0].round(4),
            "p_draw": p[:, 1].round(4),
            "p_team_b": p[:, 2].round(4),
            "lambda_a": la.round(3),
            "lambda_b": lb.round(3),
            "goals_a_90": wc["goals_a_90"],
            "goals_b_90": wc["goals_b_90"],
            "winner_90": wc["winner_90"],
        }
    )
    return {"report": report, "predictions": preds, "bundle": bundle}


def save_backtest(result: dict, year: int, mode: str, cfg: dict) -> None:
    out_dir = resolve("reports/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"wc{year}_{mode}.json").write_text(json.dumps(result["report"], indent=2))
    result["predictions"].to_csv(out_dir / f"wc{year}_{mode}_predictions.csv", index=False)
    log.info("saved reports/backtests/wc%d_%s.json", year, mode)
