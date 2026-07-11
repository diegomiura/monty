# International Soccer Predictor — 2026 World Cup

A reproducible, leakage-safe statistical/ML pipeline that predicts senior
men's international matches: 90-minute win/draw/win probabilities, expected
goals, scorelines, goal markets, and knockout advancement — focused on the
2026 FIFA World Cup. Free data and open-source software only; no API keys
required; runs locally on a normal computer.

Probabilities come from trained models (Elo, Poisson goal model, multinomial
logistic regression, gradient boosting) combined in a validation-selected
ensemble — never from language-model intuition.

## Quick start

```bash
pip install -r requirements.txt        # or: conda env create -f environment.yml

# 1. Download/refresh data (free public sources, cached under data/raw)
python update_data.py --include-world-cup-2026

# 2. Train a model bundle (example: rolling 2026 cutoff)
python train.py --cutoff "2026-07-11"

# 3. Predict any two national teams
python predict.py --team-a "Argentina" --team-b "France" \
    --date "2026-07-15T20:00:00Z" --competition "FIFA World Cup" \
    --stage knockout --neutral --mode rolling

# 4. Backtests
python backtest.py --validate                                  # chronological folds
python backtest.py --competition "FIFA World Cup" --year 2022 --mode both

# 5. Tests
python -m pytest tests/ -q
```

## Data sources (all free, verified current at retrieval time)

| Source | Coverage | Role | License/terms |
|---|---|---|---|
| [martj42/international_results](https://github.com/martj42/international_results) | 49k+ senior men's internationals, 1872 → present (incl. completed 2026 WC matches) | primary results | free public dataset (see repo) |
| [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) | World Cups 2010–2026 with separate FT/ET/penalty scores and stage labels | 90-minute score corrections, stages, cross-verification | public domain (openfootball) |
| martj42 `shootouts.csv` | all recorded shootouts | penalty identification | as above |
| martj42 `former_names.csv` | country renames | team-name aliases | as above |

Retrieval metadata (URL, timestamp, bytes) is stored beside every raw file
and in `data/metadata/data_audit.json`. The 2026 results are cross-verified
between the two independent sources at update time
(`data/metadata/wc2026_verification.json`).

**No API keys are needed.** `football-data.org` support can be added via
`.env` later; the system is fully functional from the downloadable files.
Betting odds are not used.

## The canonical match table

`data/processed/matches.csv` — one row per completed match with regulation
(90-minute), extra-time, and penalty-shootout goals kept strictly separate
(`goals_*_90`, `goals_*_extra_time`, `penalty_goals_*`), plus competition,
stage (World Cups), neutrality, hosts, and provenance columns.

Extra-time policy (documented):
* World Cup matches 2010–2026: regulation scores confirmed from openfootball
  (`goals_90_confirmed = True`); ET-decided matches are corrected to their
  true 90' draw.
* Any shootout match (all competitions): the 90' result is a draw; the
  recorded draw score may include ET goals (flagged `goals_90_confirmed=False`,
  excluded from goal-model training).
* Non-WC knockout matches decided in ET *without* a shootout cannot be
  identified from the primary source and keep their post-ET score — a known
  limitation affecting a small fraction of historical matches.

Team names use the primary source's current-name convention (Soviet Union →
Russia, Zaïre → DR Congo, FR Yugoslavia → Serbia); genuinely distinct
historical teams (Czechoslovakia, Yugoslavia, German DR) stay separate.
Aliases (FIFA/openfootball spellings + former names) are documented in
`src/data/team_names.py`.

## Leakage policy

For a match on day D, every feature uses only matches with date **strictly
before D**. The dataset has no kickoff times, so this conservative whole-day
rule guarantees same-day matches never affect each other. The feature
builder processes matches in date groups: rows for day D are emitted before
day D's results update any state. Automated tests
(`tests/test_no_leakage.py`, `tests/test_tournament_order.py`) prove:

* full-pass features equal features computed from a dataset truncated at D;
* same-day matches see identical prior state;
* frozen mode never updates features from tournament matches;
* training/validation/calibration splits are strictly chronological.

## Features

* **Elo** (custom): K by competition class (WC 60 … friendlies 20),
  goal-margin multiplier, +100 home advantage (non-neutral only; hosts are
  non-neutral in the data), inactivity shrinkage. Updated with 90' results
  only. Parameters tuned exclusively on validation years ≤ 2013.
* **Rolling form**: windows 5/10/20 (points, goals, clean sheets, failure to
  score, win/draw rates), competitive-only windows, exponential day-decay
  (half-life 550 days ≈ the spec's suggested weights), days since last
  match, match-count depth.
* **Context**: neutrality, home edge, competition class one-hots, knockout
  stage flag.
* Team A/B blocks plus difference features; 60 features total. Rankings and
  advanced stats (xG, lineups) are optional extensions not required by the
  core system.

## Models

| Component | What it is |
|---|---|
| `frequency` | historical class frequencies (baseline) |
| `higher_elo` | empirical W/D/L given the higher-Elo side (baseline) |
| `elo` | multinomial LR on the Elo expected score |
| `poisson` | team-rows Poisson regression → λ_a, λ_b → exact-score matrix |
| `logistic` | multinomial LR on all features (symmetrized) |
| `gradient_boosting` | sklearn HistGradientBoosting (symmetrized; XGBoost/LightGBM optional, not required) |
| **ensemble** | nonnegative weights summing to 1, grid-searched on a chronological validation window; multinomial recalibration kept only when it improves held-out log loss |

Training splits for cutoff T: components fit on `< T−3y`; ensemble weights
on `[T−3y, T−1y)` (val-A); calibration keep/drop decided on `[T−1y, T)`
(val-B); components then refit on all `< T` with selections frozen. No
selection ever touches data ≥ T; World Cup holdouts are never used to tune
the model evaluated on them (all hyperparameters come from folds ≤ 2013).

## Modes

* **Frozen**: features stop at the tournament start; measures pure
  pre-tournament forecasting.
* **Rolling**: completed tournament matches update Elo/form for later
  predictions (model and weights stay locked). Saved predictions are
  append-only and never revised after results are known
  (`reports/predictions/world_cup_2026_predictions.csv`).

## Knockout handling

90-minute probabilities stay separate from advancement. Staged calculation:
P(ET) = P(draw at 90'); ET goals ~ Poisson at per-minute rates recalibrated
on historical WC ET matches; P(pens) = P(draw 90') × P(ET level); shootout
≈ 50/50 with a small Elo tilt hard-clipped to [0.45, 0.55].

## Results

See `reports/RESULTS.md` for the current validation and backtest tables
(chronological folds 2014–2025, frozen/rolling World Cup backtests
2014/2018/2022, and 2026 outputs), plus figures under `reports/figures/`.

## Reproducibility

Seeds fixed; every bundle (`models/*.joblib`) stores the trained components,
ensemble weights, calibrator, config snapshot, feature version, training
cutoff, latest match used, ET-model statistics, and package versions. Same
data + config ⇒ same predictions (tested).

## Limitations

See the Limitations section of `reports/RESULTS.md`.
