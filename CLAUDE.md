You are a senior machine-learning engineer and soccer analytics researcher.

Your task is to build a complete, reproducible international soccer prediction system, focused on the 2026 FIFA World Cup.

This must be a real statistical and machine-learning pipeline. Do not create probabilities using language-model intuition. Probabilities must come from trained models, statistical calculations, or a documented ensemble.

The project must use only free tools, open-source software, free public datasets, and free API tiers.

The complete system should run locally on a normal computer.

# Main goal

Given two national teams and a match kickoff time, predict the following.

* Team A win probability after 90 minutes
* Draw probability after 90 minutes
* Team B win probability after 90 minutes
* Expected goals for Team A
* Expected goals for Team B
* Most likely scoreline
* At least five scorelines with probabilities
* Both teams to score probability
* Over 2.5 goals probability
* Under 2.5 goals probability
* Clean-sheet probability for each team
* Probability each team scores first
* Model confidence
* Most influential features
* For knockout matches, probability each team advances
* Probability of extra time
* Probability of a penalty shootout

The Team A win, draw, and Team B win probabilities must sum to 1.0.

All normal match-result probabilities must refer to the score after 90 minutes, including stoppage time but excluding extra time and penalties.

# Cost restrictions

Use only free resources.

Allowed software includes the following.

* Python
* pandas
* NumPy
* SciPy
* scikit-learn
* statsmodels
* XGBoost when available
* LightGBM when available
* matplotlib
* joblib
* requests
* Jupyter
* Git
* SQLite
* DuckDB

Do not require any of the following.

* Paid sports-data APIs
* Paid cloud computing
* Paid databases
* Paid betting feeds
* Paid analytics subscriptions
* Paid LLM APIs
* Proprietary modeling platforms
* Data collected in violation of a website’s terms

XGBoost and LightGBM must remain optional. The project must have a scikit-learn-only fallback.

Do not require an LLM to run the prediction system.

# Development behavior

Write executable code rather than pseudocode.

Create actual project files.

Run the code, tests, and backtests whenever the environment permits.

Never claim that code ran successfully unless it was actually executed.

When something fails, inspect the error and attempt to fix it.

Do not stop the entire project because one optional data source or package is unavailable.

Build the smallest complete working system first. Improve it only when validation results justify the added complexity.

Keep core logic in reusable Python modules rather than placing everything inside notebooks.

Document every important assumption.

# Data coverage

The model must use senior men’s international match data through the requested prediction cutoff.

A dataset ending in 2024 or 2025 is not sufficient for a current 2026 World Cup prediction.

Use historical international results for long-term team strength and recent international results for current form.

When available before the prediction cutoff, include the following.

* FIFA World Cup matches
* FIFA World Cup qualifying matches
* UEFA Nations League matches
* CONCACAF Nations League matches
* Copa América matches
* UEFA European Championship matches
* Africa Cup of Nations matches
* AFC Asian Cup matches
* CONCACAF Gold Cup matches
* Other recognized continental championships
* International friendlies
* Pre-World Cup preparation matches
* Completed 2026 World Cup matches

Exclude the following from the main senior model.

* Youth internationals
* Under-23 matches
* Olympic matches
* B-team matches
* Club matches
* Unofficial representative teams

# Free data sources

Search for and evaluate free, legally accessible data sources.

Preferred source types include the following.

1. Official FIFA fixtures and results pages for verification
2. Official confederation competition results
3. Football-data.org free API coverage
4. OpenFootball public repositories
5. StatsBomb Open Data
6. Public international match-result CSV files
7. Historical FIFA ranking datasets
8. Historical Elo datasets
9. User-provided CSV, JSON, or database files

Do not assume a source is current because it has a familiar name.

For every source, record the following.

* Source name
* Source URL or repository
* Retrieval date and time
* Data coverage period
* Most recent match date
* License or stated usage conditions
* Fields available
* Missing-data rate
* Known limitations
* Whether the source includes extra-time or penalty scores
* Whether it contains information published after matches

Prefer downloadable CSV or JSON files when possible.

Cache raw downloads locally.

Do not repeatedly call an API when a cached response can be used.

Store API keys through environment variables.

The project must still work without an API key by using downloadable data or user-supplied files.

Do not use betting odds as a required feature.

Betting odds may be added later as an optional benchmark only when the data is free and legally accessible.

# Data freshness

Before training or predicting, print a freshness report containing the following.

```text
Data retrieval timestamp
Prediction cutoff
Latest match in dataset
Latest competitive international
Latest match for Team A
Latest match for Team B
Latest completed 2026 World Cup match
Number of matches added during the latest update
Sources used
```

If the dataset appears outdated, warn the user and attempt to update it using free sources.

Do not silently make a current prediction from stale data.

When live retrieval fails, either use the latest cached data while clearly displaying its cutoff or stop with an actionable error.

Never fabricate recent results.

# Canonical match table

Create a standardized match-level dataset.

Include these fields whenever available.

```text
match_id
date
kickoff_time
kickoff_timestamp_utc
team_a
team_b
goals_a_90
goals_b_90
goals_a_extra_time
goals_b_extra_time
penalty_goals_a
penalty_goals_b
competition
stage
group
neutral
venue
host_team_a
host_team_b
went_to_extra_time
went_to_penalties
winner_90
team_advanced
match_status
source
retrieved_at
```

Regulation goals, extra-time goals, and penalty-shootout scores must remain separate.

Never count penalty-shootout goals as normal match goals.

If a source reports only a final score that includes extra time, do not treat it as a 90-minute score unless the regulation score can be confirmed.

Standardize all team names.

Create a documented alias table.

Do not automatically merge historically distinct teams without documenting the decision.

Handle renamed countries carefully.

# Prediction cutoff rules

For a match beginning at time T, every feature must use information that was available strictly before T.

Use exact timestamps when possible.

Two matches played on the same date must be ordered by kickoff time.

If kickoff times are unavailable, use a conservative approach and prevent same-day matches from affecting one another.

Every saved prediction must include the following.

```text
prediction_created_at
prediction_cutoff
data_retrieved_at
latest_match_used
training_cutoff
model_version
feature_version
source_versions
```

Save pre-match predictions in an append-only file.

Do not overwrite old predictions after the result becomes known.

# Leakage prevention

Prevent all forms of future-data leakage.

Never use the following.

* The result of the match being predicted
* Matches played after the prediction cutoff
* Final tournament statistics when predicting earlier tournament matches
* Rankings released after the match
* Lineups confirmed after the prediction cutoff
* Season-wide or tournament-wide aggregates containing future matches
* Random train-test splits as the main validation method
* Imputers fitted using test data
* Scalers fitted using test data
* Feature selection performed using the final test set
* Calibration fitted on the final test set
* Ensemble weights chosen using World Cup test results
* Team encodings calculated using future matches
* Hyperparameters selected using the final tournament holdout

Build chronological features one match at a time.

Create automated tests proving that every feature date is earlier than the predicted match.

# Feature engineering

Build pre-match features for both teams and difference features between them.

The system must remain functional using only match results, dates, teams, competition types, and neutral-site status.

Optional features may improve the model when free and reliable data exists.

## Elo ratings

Implement a custom international Elo system.

Account for the following.

* Win, draw, or loss
* Opponent strength
* Goal margin
* Match importance
* Neutral venue
* Home advantage
* Time decay
* Host-country advantage

Store each team’s Elo rating immediately before each match.

Tune Elo parameters using training and validation periods only.

Do not tune Elo using the final World Cup holdout.

## Recent form

For rolling windows such as 5, 10, and 20 matches, calculate the following.

* Win rate
* Draw rate
* Loss rate
* Points per match
* Goals scored per match
* Goals conceded per match
* Goal difference per match
* Clean-sheet rate
* Failed-to-score rate
* Days since previous match
* Competitive-match share
* Opponent-adjusted performance

Create separate recent-form features for all matches and competitive matches.

Apply exponential recency weighting where appropriate.

Suggested starting weights are shown below.

```text
Previous 90 days       1.00
91 to 180 days         0.90
181 to 365 days        0.75
366 to 730 days        0.50
Older than 730 days    handled mainly through long-term ratings
```

Treat these as starting values rather than proven optimal settings.

Tune them through chronological validation.

Friendlies should generally receive less importance than competitive matches.

The final competition weights must be selected through validation.

## Rankings

When free historical ranking data is available, use only the latest ranking published before each match.

Possible features include the following.

* Ranking difference
* Ranking-points difference
* Change since previous ranking
* Elo and ranking interaction
* Ranking age in days

The system must continue to work when ranking data is unavailable.

## Tournament context

When known before kickoff, include the following.

* Competition
* Tournament stage
* Group-stage indicator
* Knockout-stage indicator
* Neutral-site indicator
* Host-country status
* Confederation matchup
* Days of rest
* Matches already played in the current tournament
* Current group points
* Current group goal difference
* Goals scored earlier in the tournament
* Goals conceded earlier in the tournament
* Whether the team can advance with a draw
* Whether the team has already qualified
* Travel between venues when free reliable venue data is available

Tournament-state variables must be calculated using completed matches only.

Do not guess motivation from narrative descriptions.

## Advanced statistics

Use optional advanced features only when they are available from a free and reliable source.

Possible fields include the following.

* Expected goals
* Expected goals allowed
* Shots
* Shots on target
* Set-piece shots
* Possession
* Pressing statistics
* Player availability
* Starting lineup strength
* Goalkeeper changes

Never invent advanced statistics.

Never replace unavailable data with language-model estimates.

The core model must still run without these fields.

## Missing values

Use scikit-learn pipelines.

Fit imputers on training data only.

Add missingness indicators when useful.

Do not calculate global replacement values using the full dataset.

# Required models

Build and compare several models.

## Baselines

Implement the following.

1. Historical class-frequency baseline
2. Higher-Elo-team baseline
3. Elo probability model
4. Independent Poisson goal model
5. Multinomial logistic-regression model

A more complicated model should not be selected unless it improves chronological validation performance over meaningful baselines.

## Poisson goal model

Create a model that predicts expected goals for each team.

Start with Poisson regression.

A practical design may convert every match into two team-level rows.

Each row should contain the attacking team, defending team, pre-match features, and goals scored after 90 minutes.

Predict the following.

```text
lambda_a
lambda_b
```

Use the expected goal rates to build an exact-score probability matrix.

Calculate score probabilities from 0 to at least 8 goals for each team.

Track the probability mass beyond the cutoff.

Increase the score cutoff when remaining probability mass is too large.

Calculate the following from the score matrix.

* Team A win probability
* Draw probability
* Team B win probability
* Exact-score probabilities
* Both teams to score
* Over 2.5 goals
* Under 2.5 goals
* Team A clean sheet
* Team B clean sheet

First-team-to-score probability may be approximated using scoring intensities. Clearly document the method.

If validation results support it, test a Dixon-Coles correction or another low-score adjustment.

Do not add the correction unless it improves unseen chronological results.

## Outcome classifiers

Train a three-class model for Team A win, draw, and Team B win.

Test at least the following.

* Multinomial logistic regression
* HistGradientBoostingClassifier
* Random Forest or Extra Trees
* XGBoost when installed
* LightGBM when installed

Use predicted probabilities rather than only predicted labels.

The primary optimization target is probabilistic performance.

Raw accuracy is secondary.

## Ensemble

Create an ensemble combining some or all of the following.

* Elo probabilities
* Poisson probabilities
* Logistic-regression probabilities
* Gradient-boosting probabilities

Select ensemble weights using validation data only.

Use a transparent method such as constrained grid search, nonnegative optimization, or logistic stacking.

Weights must be nonnegative and sum to 1.

Do not select weights using the final World Cup holdout.

# Probability calibration

Evaluate calibrated and uncalibrated models.

Possible methods include the following.

* Sigmoid calibration
* Isotonic calibration
* Multinomial logistic recalibration
* Temperature scaling when implemented correctly

Use a separate chronological calibration period.

Do not fit calibration on the final test tournament.

Only keep calibration when it improves validation log loss, Brier score, ranked probability score, or reliability plots.

Generate reliability diagrams for all three outcomes.

# Validation design

Do not use random splitting as the main evaluation method.

## Expanding-window validation

Use chronological folds.

An example follows.

```text
Train through 2012, validate on 2013
Train through 2013, validate on 2014
Train through 2014, validate on 2015
Continue forward through available data
```

The exact years should depend on data coverage.

Use early folds for development and later folds for realistic model selection.

## World Cup holdouts

Backtest complete World Cups when supported by the data.

Prioritize the following.

* 2014 World Cup
* 2018 World Cup
* 2022 World Cup

Use earlier data only when predicting each tournament.

Do not use a tournament to tune the same model being evaluated on that tournament.

## Frozen tournament mode

In frozen mode, train using matches completed before the tournament begins.

Lock the following before the first tournament match.

* Model architecture
* Hyperparameters
* Feature definitions
* Elo parameters
* Calibration method
* Ensemble weights
* Data-cleaning rules

Predict the entire tournament without allowing tournament matches to update features.

This measures pure pre-tournament forecasting performance.

## Rolling tournament mode

In rolling mode, lock the model and hyperparameters before the tournament.

Predict matches in chronological order.

After a match finishes, allow it to update the following for later matches.

* Elo ratings
* Rolling form
* Tournament points
* Tournament goal difference
* Current-tournament goals
* Current-tournament advanced statistics when available

Do not retrain hyperparameters or change ensemble weights during the tournament.

A completed match may affect later matches but must never change an earlier saved prediction.

Report frozen and rolling results separately.

# 2026 World Cup mode

Create a dedicated 2026 World Cup workflow.

The workflow should ingest completed 2026 World Cup matches available before the prediction cutoff.

Suggested module name follows.

```text
src/data/world_cup_2026.py
```

Collect these fields when available.

```text
match_id
date
kickoff_time
kickoff_timestamp_utc
team_a
team_b
group
stage
venue
neutral
status
goals_a_90
goals_b_90
goals_a_extra_time
goals_b_extra_time
penalty_goals_a
penalty_goals_b
winner_90
team_advanced
source
retrieved_at
```

Verify official results against an official FIFA or competition source when practical.

The 2026 system must support both frozen and rolling modes.

## Frozen 2026 mode

Use matches completed before the tournament began.

Do not update current-tournament features.

Do not train on 2026 World Cup matches.

## Rolling 2026 mode

Use all matches completed before the prediction cutoff.

Allow completed 2026 World Cup matches to update Elo, recent form, group standings, tournament statistics, and rest-day calculations.

Do not use unfinished or future matches.

Do not retune the model during the tournament.

# Knockout matches

Keep 90-minute probabilities separate from advancement probabilities.

Use a staged calculation.

1. Predict the 90-minute score distribution
2. Calculate the probability of a draw after 90 minutes
3. Model extra-time goals using reduced scoring rates or historically validated extra-time parameters
4. Calculate the probability of reaching penalties
5. Estimate shootout probabilities

Penalty predictions should remain close to 50 percent unless reliable structured data supports a meaningful difference.

Shrink extreme shootout estimates toward 50 percent.

Clearly report the following.

```text
Team A wins in 90 minutes
Draw after 90 minutes
Team B wins in 90 minutes
Team A advances
Team B advances
Extra time occurs
Penalty shootout occurs
```

# Evaluation metrics

Calculate the following.

## Result metrics

* Multiclass log loss
* Multiclass Brier score
* Ranked probability score
* Accuracy
* Balanced accuracy
* Confusion matrix
* Top-two accuracy
* Expected calibration error
* Reliability curves

For multiclass Brier score, calculate the mean sum of squared errors across all three result probabilities.

## Goal metrics

* Mean absolute error for Team A goals
* Mean absolute error for Team B goals
* Total-goal mean absolute error
* Root mean squared goal error
* Exact-score accuracy
* One-goal-tolerance accuracy
* Poisson deviance when appropriate
* Predicted total goals compared with observed total goals

## Breakdown reports

Report results by the following.

* Tournament
* Group stage or knockout stage
* Elo-difference bucket
* Favorite-probability bucket
* Confederation matchup
* Neutral or non-neutral venue
* Low-scoring or high-scoring match
* Frozen or rolling mode

Do not judge the model only by winner accuracy.

Probabilistic calibration is more important than generating confident picks.

# Confidence and uncertainty

Create a confidence score based on model evidence.

Possible inputs include the following.

* Agreement among component models
* Prediction entropy
* Amount of historical data for both teams
* Missing-feature rate
* Distance from the training distribution
* Age of recent team data
* Bootstrap uncertainty
* Ensemble dispersion

Return the following.

```text
confidence_label
confidence_score
uncertainty_notes
```

The label should be low, medium, or high.

The score should range from 0 to 100.

A high win probability does not automatically mean high confidence.

# Interpretability

Use model-derived explanations.

For linear models, use coefficients or feature contributions.

For tree models, use permutation importance.

Use SHAP only when the free package is available and stable in the environment.

Provide global feature importance and match-level feature contributions.

For each match, identify the five most influential available features.

Do not invent tactical narratives that are not represented by the data.

Do not allow the written explanation to contradict the model output.

# Prediction output

Produce machine-readable JSON and a readable console report.

Use a structure similar to the following.

```json
{
  "team_a": "Argentina",
  "team_b": "France",
  "match_date": "2026-07-15",
  "prediction_cutoff": "2026-07-15T18:00:00Z",
  "expected_goals": {
    "team_a": 1.42,
    "team_b": 1.31
  },
  "outcome_90": {
    "team_a_win": 0.397,
    "draw": 0.281,
    "team_b_win": 0.322
  },
  "most_likely_score": "1-1",
  "scorelines": [
    {
      "score": "1-1",
      "probability": 0.118
    },
    {
      "score": "1-0",
      "probability": 0.090
    },
    {
      "score": "0-1",
      "probability": 0.083
    },
    {
      "score": "2-1",
      "probability": 0.064
    },
    {
      "score": "1-2",
      "probability": 0.059
    }
  ],
  "additional_probabilities": {
    "both_teams_score": 0.537,
    "over_2_5": 0.491,
    "under_2_5": 0.509,
    "team_a_clean_sheet": 0.270,
    "team_b_clean_sheet": 0.242,
    "team_a_scores_first": 0.462,
    "team_b_scores_first": 0.415,
    "no_goal": 0.123
  },
  "knockout": {
    "team_a_advances": 0.535,
    "team_b_advances": 0.465,
    "extra_time": 0.281,
    "penalty_shootout": 0.117
  },
  "confidence": {
    "label": "medium",
    "score": 67
  },
  "important_features": [
    {
      "feature": "elo_difference",
      "direction": "favors_team_a",
      "contribution": 0.081
    }
  ],
  "model_information": {
    "model_version": "string",
    "training_cutoff": "date",
    "latest_match_used": "date",
    "data_sources": [],
    "ensemble_weights": {
      "elo": 0.2,
      "poisson": 0.4,
      "machine_learning": 0.4
    }
  }
}
```

The values above are examples only.

Never hard-code example probabilities.

# Project structure

Create a repository similar to the following.

```text
soccer-predictor/
    README.md
    requirements.txt
    environment.yml
    .env.example
    .gitignore
    config/
        default.yaml
    data/
        raw/
        interim/
        processed/
        metadata/
    models/
    reports/
        figures/
        predictions/
        backtests/
    notebooks/
        01_data_audit.ipynb
        02_feature_analysis.ipynb
        03_model_comparison.ipynb
        04_world_cup_backtest.ipynb
        05_world_cup_2026.ipynb
    src/
        data/
            download.py
            update_recent.py
            world_cup_2026.py
            clean.py
            merge.py
            team_names.py
            validate.py
        features/
            elo.py
            rolling.py
            rankings.py
            tournament.py
            build_features.py
        models/
            baselines.py
            poisson.py
            outcome.py
            calibrate.py
            ensemble.py
        evaluation/
            metrics.py
            calibration.py
            backtest.py
            world_cup_report.py
        prediction/
            predict_match.py
            score_matrix.py
            simulate_knockout.py
        visualization/
            plots.py
        utils/
            config.py
            dates.py
            logging.py
            validation.py
    tests/
        test_elo.py
        test_no_leakage.py
        test_probabilities.py
        test_score_matrix.py
        test_team_names.py
        test_tournament_order.py
        test_extra_time.py
    train.py
    predict.py
    backtest.py
    update_data.py
```

Adjust this structure only when there is a clear technical reason.

# Command-line interface

Support commands similar to the following.

```bash
python update_data.py \
  --through "2026-07-11T20:00:00Z" \
  --include-world-cup-2026
```

```bash
python train.py \
  --data data/processed/matches.csv \
  --cutoff "2026-06-01"
```

```bash
python predict.py \
  --team-a "Argentina" \
  --team-b "France" \
  --date "2026-07-15T20:00:00Z" \
  --competition "FIFA World Cup 2026" \
  --stage knockout \
  --neutral \
  --mode rolling \
  --update-data
```

```bash
python backtest.py \
  --competition "FIFA World Cup" \
  --year 2022 \
  --mode frozen
```

Validate user inputs and return useful error messages.

# Saved prediction log

Save all 2026 World Cup predictions in an append-only file.

Suggested path follows.

```text
reports/predictions/world_cup_2026_predictions.csv
```

Include these fields.

```text
match
kickoff
prediction_created_at
prediction_cutoff
predicted_team_a_win
predicted_draw
predicted_team_b_win
predicted_score
expected_goals_a
expected_goals_b
actual_score
actual_result
brier_score
log_loss
goal_absolute_error
correct_result
correct_exact_score
model_version
feature_version
latest_match_used
mode
```

Do not modify the original prediction probabilities after the result becomes available.

Results may be appended to separate evaluation fields.

# Reproducibility

Set random seeds.

Save the following.

* Trained preprocessing pipeline
* Trained models
* Calibration models
* Ensemble weights
* Feature definitions
* Elo parameters
* Training cutoff
* Data-source metadata
* Package versions
* Configuration file
* Backtest results
* Model comparison table

Use joblib or another free open-source serialization method.

A saved model should reproduce the same prediction when given the same data and configuration.

# Testing

Create automated tests covering the following.

* Elo updates
* Probabilities sum to one
* Score-matrix probabilities sum approximately to one
* No future matches enter features
* Training dates precede validation dates
* Calibration does not see test data
* Team aliases resolve correctly
* Neutral venues are handled correctly
* Same-day matches are ordered safely
* Extra-time goals are separate from 90-minute goals
* Penalty-shootout scores are not treated as normal goals
* Frozen tournament mode does not update features
* Rolling mode uses completed matches only
* Saved models reproduce predictions
* Data refresh does not create duplicate matches

Run the tests after major changes.

# Required reports

Produce the following.

* Data-source audit
* Data-freshness report
* Missing-data report
* Model-comparison table
* Chronological validation results
* 2014 World Cup backtest when supported
* 2018 World Cup backtest when supported
* 2022 World Cup backtest when supported
* Frozen and rolling comparison
* Calibration plots
* Reliability diagrams
* Feature-importance plots
* Goal-prediction diagnostics
* 2026 World Cup prediction log
* Example current prediction
* README with complete setup instructions

# Development sequence

Follow this order.

1. Inspect the environment and available files
2. Identify and verify free data sources
3. Download or import historical international results
4. Update recent international and 2026 data
5. Build the canonical dataset
6. Audit team names and match timestamps
7. Add leakage tests
8. Implement Elo ratings
9. Implement rolling form features
10. Implement the baseline models
11. Implement the Poisson goal model
12. Implement the outcome classifiers
13. Run chronological validation
14. Add probability calibration
15. Build the ensemble
16. Backtest past World Cups
17. Lock the final model using pre-2026 validation
18. Add frozen and rolling 2026 modes
19. Build the prediction CLI
20. Generate reports and plots
21. Run all tests
22. Document limitations

At the end of every major stage, report what was created, what was run, and the actual results.

# Decision rules

Prefer the simplest model that performs well on unseen chronological data.

Do not select a complex model because it looks more advanced.

Do not hide poor results.

If the machine-learning model does not outperform Elo or Poisson, report that honestly.

Do not use accuracy alone to choose the final model.

Prioritize log loss, Brier score, ranked probability score, and calibration.

Do not produce artificially extreme probabilities.

Soccer contains substantial randomness.

# Final deliverables

Deliver the following.

* Complete source code
* Requirements file
* Environment file
* Data-download scripts
* Data-source documentation
* Training commands
* Prediction commands
* Backtesting commands
* Automated tests
* Trained model format
* Model comparison
* Calibration analysis
* World Cup backtests
* 2026 frozen and rolling workflows
* Example prediction
* README
* Limitations
* Recommended future improvements

The goal is not to generate impressive-looking predictions.

The goal is to produce honest, reproducible, leakage-safe, and well-calibrated probabilities for international soccer matches.

