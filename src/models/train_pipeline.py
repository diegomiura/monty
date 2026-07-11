"""Shared training pipeline: fit components -> select ensemble weights and
calibration on a chronological validation window -> refit on all pre-cutoff
data -> package a reproducible bundle.

Split layout for a training cutoff T (all strictly chronological):

    [train_start ......... T-val ....... T-mid ...... T)
     component fitting     weights+cal   cal keep/drop
                           (val-A)       decision (val-B)

After selection, components are refit on everything < T with the selected
weights/calibrator frozen (documented: selection never sees data >= T, and
the final holdout tournament is never used for any selection).
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn

from src.evaluation.metrics import log_loss
from src.features.build_features import FEATURE_COLUMNS
from src.models.baselines import EloProbabilityModel
from src.models.calibrate import IdentityCalibrator, MultinomialRecalibrator
from src.models.ensemble import blend, select_weights
from src.models.outcome import make_gradient_boosting, make_logistic
from src.models.poisson import PoissonGoalModel
from src.utils.logging import get_logger

log = get_logger(__name__)

MODEL_VERSION = "0.1.0"
FEATURE_VERSION = "fv1-" + str(len(FEATURE_COLUMNS))
COMPONENT_NAMES = ["elo", "poisson", "logistic", "gradient_boosting"]


def fit_components(feats: pd.DataFrame, cfg: dict) -> dict:
    y = feats["outcome"].to_numpy()
    models = {
        "elo": EloProbabilityModel().fit(feats, y),
        "poisson": PoissonGoalModel(alpha=float(cfg["models"]["poisson_alpha"])).fit(
            feats,
            feats["goals_a_90"].to_numpy(float),
            feats["goals_b_90"].to_numpy(float),
            feats["goals_90_confirmed"].to_numpy(),
        ),
        "logistic": make_logistic(cfg).fit(feats, y),
        "gradient_boosting": make_gradient_boosting(cfg).fit(feats, y),
    }
    return models


def component_probs(models: dict, X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: models[name].predict_proba(X) for name in COMPONENT_NAMES}


@dataclass
class Bundle:
    models: dict
    weights: np.ndarray
    calibrator: object
    use_calibration: bool
    cfg: dict
    training_cutoff: str
    latest_match_used: str
    n_train: int
    et_stats: dict
    validation_report: dict
    global_importance: list = field(default_factory=list)
    model_version: str = MODEL_VERSION
    feature_version: str = FEATURE_VERSION
    created_at: str = ""
    package_versions: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = component_probs(self.models, X)
        p = blend([probs[n] for n in COMPONENT_NAMES], self.weights)
        return self.calibrator.transform(p) if self.use_calibration else p

    def predict_lambdas(self, X: pd.DataFrame):
        return self.models["poisson"].predict_lambdas(X)

    def component_probs(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        return component_probs(self.models, X)


def estimate_et_stats(matches: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Historical extra-time scoring stats from matches with confirmed ET
    splits (World Cups 2010+), computed strictly before the cutoff."""
    past = matches[matches["date"] < cutoff]
    et = past[past["went_to_extra_time"]]
    ko = past[past["stage"].isin(["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final", "Third place"])]
    out = {"n_et_matches": int(len(et))}
    if len(et) >= 5:
        out["mean_et_goals"] = float((et["goals_a_extra_time"] + et["goals_b_extra_time"]).mean())
        out["p_pens_given_et"] = float(et["went_to_penalties"].mean())
    else:
        out["mean_et_goals"] = 0.6  # documented fallback prior
        out["p_pens_given_et"] = 0.5
    out["mean_ko_goals_90"] = float((ko["goals_a_90"] + ko["goals_b_90"]).mean()) if len(ko) else 2.4
    # per-minute ET intensity relative to regulation intensity
    out["et_rate_factor"] = float(
        np.clip(out["mean_et_goals"] / max(out["mean_ko_goals_90"] / 3.0, 1e-6), 0.5, 1.5)
    )
    return out


def train_bundle(
    feats: pd.DataFrame,
    matches: pd.DataFrame,
    cutoff: pd.Timestamp,
    cfg: dict,
) -> Bundle:
    train_start = pd.Timestamp(cfg["features"]["train_start"])
    val_days = int(cfg["validation"]["val_window_days"])
    step = float(cfg["validation"]["ensemble_weight_step"])

    usable = feats[(feats["date"] >= train_start) & (feats["date"] < cutoff)]
    t_val = cutoff - pd.Timedelta(days=val_days)
    t_mid = cutoff - pd.Timedelta(days=val_days // 3)
    fit_df = usable[usable["date"] < t_val]
    val_a = usable[(usable["date"] >= t_val) & (usable["date"] < t_mid)]
    val_b = usable[usable["date"] >= t_mid]
    if len(fit_df) < 2000 or len(val_a) < 200 or len(val_b) < 100:
        raise ValueError(
            f"Not enough data before cutoff {cutoff.date()} "
            f"(fit={len(fit_df)}, valA={len(val_a)}, valB={len(val_b)})"
        )

    log.info("fit=%d  val_a=%d  val_b=%d  cutoff=%s", len(fit_df), len(val_a), len(val_b), cutoff.date())
    models = fit_components(fit_df, cfg)

    pa = component_probs(models, val_a)
    ya = val_a["outcome"].to_numpy()
    weights, ll_a = select_weights([pa[n] for n in COMPONENT_NAMES], ya, step)
    log.info("ensemble weights %s (val-A log loss %.4f)", dict(zip(COMPONENT_NAMES, weights.round(3))), ll_a)

    cal = MultinomialRecalibrator().fit(blend([pa[n] for n in COMPONENT_NAMES], weights), ya)
    pb = component_probs(models, val_b)
    yb = val_b["outcome"].to_numpy()
    blended_b = blend([pb[n] for n in COMPONENT_NAMES], weights)
    ll_raw = log_loss(yb, blended_b)
    ll_cal = log_loss(yb, cal.transform(blended_b))
    use_cal = ll_cal < ll_raw
    log.info("val-B log loss: raw %.4f vs calibrated %.4f -> %s calibration",
             ll_raw, ll_cal, "keep" if use_cal else "drop")

    # Permutation importance of the GB component on val-B (out-of-sample,
    # before refit) — model-derived interpretability for the tree model.
    perm_imp = permutation_importance_gb(models["gradient_boosting"], val_b, yb)

    # Refit components on all pre-cutoff data with selections frozen.
    models = fit_components(usable, cfg)

    imp = global_importance(models, usable, cfg)

    validation_report = {
        "val_a": {"n": int(len(val_a)), "log_loss": round(ll_a, 4)},
        "val_b": {
            "n": int(len(val_b)),
            "log_loss_uncalibrated": round(ll_raw, 4),
            "log_loss_calibrated": round(ll_cal, 4),
        },
        "component_val_a_log_loss": {n: round(log_loss(ya, pa[n]), 4) for n in COMPONENT_NAMES},
        "gb_permutation_importance_val_b": perm_imp[:15],
    }

    return Bundle(
        models=models,
        weights=weights,
        calibrator=cal if use_cal else IdentityCalibrator(),
        use_calibration=use_cal,
        cfg=cfg,
        training_cutoff=str(cutoff.date()),
        latest_match_used=str(usable["date"].max().date()),
        n_train=int(len(usable)),
        et_stats=estimate_et_stats(matches, cutoff),
        validation_report=validation_report,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        package_versions={
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    )


def permutation_importance_gb(model, val_df: pd.DataFrame, y: np.ndarray,
                              n_repeats: int = 3, seed: int = 42) -> list:
    """Log-loss increase when each feature is permuted on the validation
    window (out-of-sample permutation importance for the tree model)."""
    rng = np.random.default_rng(seed)
    base = log_loss(y, model.predict_proba(val_df))
    rows = []
    for col in FEATURE_COLUMNS:
        deltas = []
        for _ in range(n_repeats):
            shuffled = val_df.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            deltas.append(log_loss(y, model.predict_proba(shuffled)) - base)
        rows.append({"feature": col, "logloss_increase": round(float(np.mean(deltas)), 5)})
    rows.sort(key=lambda r: -r["logloss_increase"])
    return rows


def global_importance(models: dict, usable: pd.DataFrame, cfg: dict, top_k: int = 15) -> list:
    """Mean |standardized coefficient| across classes from the logistic
    component — model-derived, cheap, and stable. (Tree permutation
    importance is produced separately in the evaluation reports.)"""
    lr_pipe = models["logistic"].base
    coefs = np.abs(lr_pipe.named_steps["lr"].coef_).mean(axis=0)
    order = np.argsort(-coefs)
    return [
        {"feature": FEATURE_COLUMNS[i], "mean_abs_coef": round(float(coefs[i]), 4)}
        for i in order[:top_k]
    ]
