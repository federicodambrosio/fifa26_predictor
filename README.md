# FIFA 26 World Cup Predictor

This is a vibecoded project built for fun for a company prediction league. It predicts FIFA World Cup 2026 match outcomes and group-stage tables with a lightweight, transparent ranking-based model. It combines public FIFA ranking data, historical international match results, and tournament structure to produce deterministic, stochastic, and Monte Carlo forecasts for a league-style prediction pool.

## Overview

The project was built to support a company league prediction challenge with a repeatable workflow that can be run from the command line. It reads public football data, estimates team strength from historical results, and exports predictions as CSV files for review or sharing.

## What the model does

The current implementation:

- predicts group-stage matches and simulated group standings
- supports deterministic, stochastic, and Monte Carlo prediction modes
- predicts knockout rounds such as Round of 32, Quarter-final, Semi-final, and Final
- can backtest the approach against recent historical results
- exports CSV files that can be consumed in spreadsheets or downstream analysis tools

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The project metadata in pyproject declares Python 3.14+.

## Quick start

Run the default deterministic group-stage forecast:

```bash
python main.py --mode deterministic
```

Other useful commands:

```bash
# Stochastic predictions with a fixed seed
python main.py --mode stochastic --seed 42

# Monte Carlo simulation with more iterations
python main.py --mode montecarlo --simulations 2000

# Backtest the method against recent historical matches
python main.py --mode backtest

# Predict one knockout round and export it to CSV
python main.py --predict-round "Round of 32"
```

## Outputs

The script writes several CSV files into the repository root, including:

- predicted_group_stage_matches_deterministic.csv
- predicted_group_stage_matches_stochastic.csv
- montecarlo_group_distribution.csv
- backtest_last_year_results.csv
- predicted_round_of_32.csv and other knockout-round exports

## Methodology

The model is a heuristics-based forecast system rather than a trained machine-learning model. It uses:

- FIFA ranking points from the ranking dataset
- historical international results weighted by recency, tournament importance, and ranking similarity
- a home advantage adjustment and draw tendency calibration
- optional injection of completed group-stage results into historical team-strength estimates

This makes the approach transparent and easy to inspect, but it also means it is sensitive to the chosen weights and assumptions.

## Data sources

The repository expects the following data files under the data directory:

- rankings.json: FIFA ranking points
- results.csv: historical international match results
- worldcup.json: match schedule and tournament round structure
- supporting metadata files for aliases and related tournament information

## Limitations

This is a practical forecasting prototype rather than a production-grade sports model. Important limitations include:

- it does not model injuries, squad rotation, or late roster changes
- it uses simplified probability logic and manual weight tuning
- it is not calibrated against a large out-of-sample benchmark set
- results should be interpreted as a fun, analytical forecast rather than a certainty

## Development

To run tests:

```bash
pytest
```

To lint the code:

```bash
ruff check .
```

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0-only). See the LICENSE file for details.
