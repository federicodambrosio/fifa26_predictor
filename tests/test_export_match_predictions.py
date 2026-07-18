import json
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import export_match_predictions, load_matches


def test_load_matches_includes_third_place_when_requesting_final_round(
    tmp_path: Path,
) -> None:
    worldcup_path = tmp_path / "worldcup.json"
    worldcup_data = {
        "matches": [
            {
                "round": "Final",
                "date": "2026-07-18",
                "team1": "Spain",
                "team2": "Argentina",
                "ground": "New York/New Jersey (East Rutherford)",
            },
            {
                "round": "Match for third place",
                "date": "2026-07-17",
                "team1": "France",
                "team2": "England",
                "ground": "Miami",
            },
        ]
    }
    worldcup_path.write_text(json.dumps(worldcup_data), encoding="utf-8")

    matches = load_matches(worldcup_path, round_contains="Final")

    assert len(matches) == 2
    assert {row["round"] for _, row in matches.iterrows()} == {
        "Final",
        "Match for third place",
    }


def test_export_match_predictions_sorts_by_match_datetime(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        [
            {
                "group": "Group A",
                "round": "Matchday 1",
                "team1": "Brazil",
                "team2": "Argentina",
                "team1_goals": 1,
                "team2_goals": 0,
                "prob_team1": 0.5,
                "prob_draw": 0.2,
                "prob_team2": 0.3,
                "result": "team1",
                "rating1": 1.0,
                "rating2": 2.0,
                "host_country": "USA",
                "points1": 3,
                "points2": 0,
                "date": pd.Timestamp("2026-06-12 15:00:00"),
            },
            {
                "group": "Group A",
                "round": "Matchday 1",
                "team1": "France",
                "team2": "Germany",
                "team1_goals": 2,
                "team2_goals": 1,
                "prob_team1": 0.6,
                "prob_draw": 0.1,
                "prob_team2": 0.3,
                "result": "team1",
                "rating1": 1.5,
                "rating2": 1.8,
                "host_country": "USA",
                "points1": 3,
                "points2": 0,
                "date": pd.Timestamp("2026-06-10 18:00:00"),
            },
            {
                "group": "Group A",
                "round": "Matchday 1",
                "team1": "Mexico",
                "team2": "Uruguay",
                "team1_goals": 0,
                "team2_goals": 1,
                "prob_team1": 0.4,
                "prob_draw": 0.1,
                "prob_team2": 0.5,
                "result": "team2",
                "rating1": 1.3,
                "rating2": 1.7,
                "host_country": "USA",
                "points1": 0,
                "points2": 3,
                "date": pd.Timestamp("2026-06-14 12:00:00"),
            },
        ]
    )

    out_path = tmp_path / "predictions.csv"
    export_match_predictions(predictions, out_path)

    exported = pd.read_csv(out_path, parse_dates=["date"])

    assert list(exported["team1"]) == ["France", "Brazil", "Mexico"]
    assert exported["date"].tolist() == sorted(exported["date"].tolist())
