## Plan: Group Stage Result Prediction Model

Build a match-level predictive model from the 2026 World Cup group stage schedule, FIFA ranking data, and historical international match results, then aggregate match predictions into group standings.

**Steps**
1. Define the prediction target: match result (win/draw/loss) and scoreline.
2. Load data from `data/worldcup.json`, `data/rankings.json`, and `data/results.csv`.
3. Normalize team names across datasets using the existing mapping logic in `exploration.ipynb` and extend it for any unmatched teams.
4. Build a training dataset from historical results: pair each match with team features at the match date. Give more weight to recent matches and those between similarly ranked teams. Weight should be zero after 15 years. World cup matches should be weighted more than friendlies.
5. Engineer features such as ranking points, ranking difference, recent form, goal averages, confederation pair, and match context
6. Choose a model approach:
   - Poisson goal model for predicted scores, or
   - multiclass outcome classifier for win/draw/loss.
7. Train and validate using historical World Cup or international match splits; measure outcome accuracy and group ranking quality.
8. Predict the 2026 group stage matches and simulate group tables to determine expected standings.

**Relevant files**
- `/Users/federicodambrosio/Code/fifa26/exploration.ipynb` — existing data loading, ranking mapping, and prototype analysis.
- `/Users/federicodambrosio/Code/fifa26/main.py` — candidate place for a reusable prediction pipeline.
- `/Users/federicodambrosio/Code/fifa26/data/worldcup.json` — 2026 group stage schedule.
- `/Users/federicodambrosio/Code/fifa26/data/rankings.json` — FIFA ranking values.
- `/Users/federicodambrosio/Code/fifa26/data/results.csv` — historical match outcomes.

**Verification**
1. Confirm the training data includes valid team name matches and ranking features.
2. Validate model predictions against historical group stage outcomes and confirm they produce sensible probability distributions.
3. Verify the final group simulation orders teams correctly by points, goal difference, and goals scored.
4. Check that the model does not result in outlier results, compared with historical data (e.g., very high scorelines).

**Decisions**
- Focus on match-level prediction first, then use aggregation to predict group stage outcomes.
- Use ranking points as the primary strength signal and historical match data for form/goal expectations.
- Keep the scope to group stage prediction, not knockouts.
