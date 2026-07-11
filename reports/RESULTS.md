# Results — v0.1.0 (generated 2026-07-11)

Data: 49,503 completed senior men's internationals, 1872-11-30 → 2026-07-10
(98 completed 2026 World Cup matches, cross-verified 98/98 between two
independent sources). All numbers below are on chronologically unseen data.

## Chronological expanding-window validation (folds 2014–2025, mean)

Train through year Y−1, validate on year Y. Hyperparameters and Elo
parameters were tuned only on folds ≤ 2013 (`data/metadata/tuning.json`).

| model             |   log_loss |   brier |    rps |   accuracy |    ece |
|:------------------|-----------:|--------:|-------:|-----------:|-------:|
| logistic          |     0.8825 |  0.5193 | 0.1732 |     0.5921 | 0.0338 |
| gradient_boosting |     0.8851 |  0.5208 | 0.1736 |     0.5910 | 0.0320 |
| elo               |     0.8974 |  0.5279 | 0.1763 |     0.5908 | 0.0457 |
| poisson           |     0.8991 |  0.5286 | 0.1774 |     0.5868 | 0.0473 |
| higher_elo        |     0.9555 |  0.5653 | 0.1930 |     0.5908 | 0.0358 |
| frequency         |     1.0542 |  0.6360 | 0.2284 |     0.4732 | 0.0182 |

Reading: the full-feature models add real value over Elo/Poisson, which add
substantial value over the naive baselines. The margin between logistic and
gradient boosting is small; the ensemble keeps both. Per-fold numbers:
`reports/backtests/chronological_validation_2014_2025.csv`.

## World Cup backtests

Model trained strictly before each tournament; ensemble weights and
calibration selected on pre-tournament validation windows only. Frozen =
features locked at tournament start; rolling = completed tournament matches
update Elo/form for later match-days (the model itself stays locked).

|   WC | mode    |   n |   log_loss |   brier |    rps |    acc |    ece |   total-goal MAE |   pred goals/m |   obs goals/m |
|-----:|:--------|----:|-----------:|--------:|-------:|-------:|-------:|-----------------:|---------------:|--------------:|
| 2014 | frozen  |  64 |     0.9996 |  0.6025 | 0.2037 | 0.5156 | 0.1075 |           1.4661 |          2.601 |         2.547 |
| 2014 | rolling |  64 |     0.9724 |  0.5862 | 0.1952 | 0.5469 | 0.0758 |           1.4602 |          2.592 |         2.547 |
| 2018 | frozen  |  64 |     0.9881 |  0.5884 | 0.2079 | 0.5469 | 0.1250 |           1.1467 |          2.475 |         2.594 |
| 2018 | rolling |  64 |     0.9763 |  0.5825 | 0.2055 | 0.5156 | 0.1832 |           1.1469 |          2.473 |         2.594 |
| 2022 | frozen  |  64 |     1.0599 |  0.6211 | 0.2234 | 0.5469 | 0.1509 |           1.4790 |          2.526 |         2.625 |
| 2022 | rolling |  64 |     1.0795 |  0.6326 | 0.2295 | 0.5156 | 0.1426 |           1.4832 |          2.500 |         2.625 |
| 2026 | frozen  |  98 |     0.8897 |  0.5300 | 0.1626 | 0.6531 | 0.1344 |           1.3947 |          2.693 |         2.867 |
| 2026 | rolling |  98 |     0.9014 |  0.5363 | 0.1662 | 0.6429 | 0.1275 |           1.3977 |          2.706 |         2.867 |

Pooled frozen (290 matches): log loss 0.9732, Brier 0.5790, RPS 0.1951,
accuracy 57.6%, ECE 0.054 (`reports/figures/wc_all_frozen_reliability.png`).

Honest observations, not hidden:

* World Cups are harder than ordinary internationals (log loss ≈ 0.97–1.06
  vs 0.88 in annual folds): more neutral venues, more balanced fields, and
  2022 in particular was upset-heavy for every public model.
* Rolling beat frozen in 2014 and 2018 but *hurt* in 2022 and 2026-to-date —
  in-tournament form is a noisy signal; we report both modes rather than
  cherry-picking.
* On 2026-to-date the pure Elo component (log loss 0.853) has outperformed
  the ensemble (0.890). Across all four tournaments the ensemble is more
  robust, but this is exactly the kind of result the decision rules require
  us to surface. ECE on single tournaments (~0.11–0.18) is dominated by
  small-sample bin noise (64–98 matches).
* Predicted goal totals track observed totals well (2.5–2.7 vs 2.5–2.9).
* Breakdowns by Elo gap and favorite strength are inside each
  `reports/backtests/wc*_*.json`; predictions per match in the
  `*_predictions.csv` files; reliability diagrams for all three outcomes
  and goal diagnostics per tournament under `reports/figures/`.

## Final 2026 configuration

* Ensemble weights (selected on pre-cutoff validation window):
  logistic 0.80, gradient boosting 0.15, Poisson 0.05, Elo 0.0
  (rolling bundle, cutoff 2026-07-11).
* Multinomial recalibration evaluated and **dropped** (val-B log loss
  0.8408 raw vs 0.8433 calibrated).
* Extra-time model: from 22 confirmed WC ET matches before the cutoff —
  0.77 ET goals/match, P(pens | ET) ≈ 0.68 empirically; per-minute ET rate
  factor 0.89 relative to regulation.
* Top features (GB permutation importance, out-of-sample):
  `elo_expected_a` dominates, then Elo levels/diff and medium-window form —
  see `reports/figures/importance_gb_permutation.png` and
  `importance_logistic.png`.

## Example current predictions (2026-07-11, quarterfinals)

| match | mode | P(A) | P(draw) | P(B) | xG | advance |
|---|---|---|---|---|---|---|
| Norway v England | rolling | 20.9% | 29.7% | 49.4% | 1.16–1.61 | ENG 65.9% |
| Argentina v Switzerland | rolling | 57.2% | 28.8% | 14.0% | 1.72–0.94 | ARG 74.3% |
| Argentina v Switzerland | frozen | 56.4% | 32.3% | 11.3% | 1.79–0.89 | ARG 76.1% |

All 2026 predictions are appended (never rewritten) to
`reports/predictions/world_cup_2026_predictions.csv`.

## Limitations

1. **No kickoff times in the primary dataset.** A conservative whole-day
   cutoff is used; a same-day earlier match can never inform a later one,
   which slightly under-uses information on multi-match days.
2. **Extra-time splits are only confirmed for World Cups 2010–2026.**
   Non-WC knockout matches decided in ET (without a shootout) keep their
   post-ET score and are mislabeled as 90' wins — a small share of
   historical matches, unquantified precisely because unidentifiable from
   the source; shootout matches (always 90' draws) are handled correctly
   everywhere.
3. **Stage labels exist only for World Cups**, so the knockout feature is 0
   for other competitions' knockout matches (the CLI accepts an explicit
   `--stage knockout` for any prediction).
4. **Independent Poisson scoreline model**: no Dixon-Coles low-score
   correlation correction yet (to be added only if it improves chronological
   validation); scorelines are rescaled to the calibrated W/D/L
   probabilities to keep outputs consistent.
5. **No player-level data** (injuries, lineups, xG). The system is
   results-only by design of v1; free advanced sources can be added behind
   the same cutoff rules.
6. **Confidence score is a documented heuristic** (model agreement, entropy,
   history depth, feature completeness), not a formally calibrated
   uncertainty estimate.
7. **First-scorer probabilities** assume constant scoring intensity.
8. **Single-tournament ECE is noisy** (64–98 matches); judge calibration on
   the pooled reliability plot.

## Recommended future improvements

* Dixon-Coles correction + bivariate Poisson, gated on validation gains.
* Free FIFA-ranking history as an optional feature block.
* ET-split backfill for Euros/Copa América from openfootball text files.
* Bootstrap ensembles for calibrated uncertainty intervals.
* Group-stage simulation to produce advancement probabilities from the
  group context (points, goal difference scenarios).
