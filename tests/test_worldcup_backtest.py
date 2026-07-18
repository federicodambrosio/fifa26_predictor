import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import backtest_worldcup


@pytest.fixture
def sample_worldcup_json(tmp_path: Path) -> Path:
    """Create a sample worldcup.json file for testing."""
    worldcup_data = {
        "name": "World Cup 2026",
        "matches": [
            {
                "round": "Matchday 1",
                "date": "2026-06-11",
                "time": "13:00",
                "team1": "Mexico",
                "team2": "South Africa",
                "score": {"ft": [2, 0], "ht": [1, 0]},
                "goals1": [],
                "goals2": [],
                "group": "Group A",
                "ground": "Mexico City",
            },
            {
                "round": "Matchday 1",
                "date": "2026-06-11",
                "time": "14:00",
                "team1": "France",
                "team2": "Germany",
                "score": {"ft": [1, 1], "ht": [0, 0]},
                "goals1": [],
                "goals2": [],
                "group": "Group X",
                "ground": "Seattle",
            },
            {
                "round": "Round of 32",
                "date": "2026-07-04",
                "time": "18:00",
                "team1": "Brazil",
                "team2": "Argentina",
                "score": {"ft": [2, 1], "ht": [1, 1]},
                "goals1": [],
                "goals2": [],
                "group": None,
                "ground": "Los Angeles",
            },
        ],
    }
    
    worldcup_path = tmp_path / "test_worldcup.json"
    with open(worldcup_path, "w", encoding="utf-8") as f:
        json.dump(worldcup_data, f)
    
    return worldcup_path


def test_backtest_worldcup_returns_results_and_metrics(tmp_path: Path) -> None:
    """Test that backtest_worldcup returns results DataFrame and metrics dict."""
    from main import load_rankings, build_rank_map
    
    worldcup_path = tmp_path / "test_worldcup.json"
    worldcup_data = {
        "name": "World Cup 2026",
        "matches": [
            {
                "round": "Matchday 1",
                "date": "2026-06-11",
                "time": "13:00",
                "team1": "Mexico",
                "team2": "South Africa",
                "score": {"ft": [2, 0], "ht": [1, 0]},
                "goals1": [],
                "goals2": [],
                "group": "Group A",
                "ground": "Mexico City",
            },
        ],
    }
    
    with open(worldcup_path, "w", encoding="utf-8") as f:
        json.dump(worldcup_data, f)
    
    rankings = load_rankings()
    rank_map = build_rank_map(rankings)
    
    results, metrics = backtest_worldcup(rank_map, worldcup_path)
    
    # Check results DataFrame structure
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert "team1" in results.columns
    assert "actual_team1_goals" in results.columns
    assert "predicted_team1_goals" in results.columns
    assert "correct_outcome" in results.columns
    
    # Check metrics dictionary structure
    assert isinstance(metrics, dict)
    assert "matches" in metrics
    assert "outcome_accuracy" in metrics
    assert "exact_score_accuracy" in metrics
    assert "mean_absolute_goal_diff_error" in metrics
    assert "group_stage_accuracy" in metrics
    assert "knockout_accuracy" in metrics
    
    # Check metric values are reasonable
    assert 0 <= metrics["outcome_accuracy"] <= 1
    assert 0 <= metrics["exact_score_accuracy"] <= 1
    assert metrics["matches"] > 0


def test_backtest_worldcup_separates_group_and_knockout_stages(tmp_path: Path) -> None:
    """Test that backtest correctly identifies group vs knockout stages."""
    from main import load_rankings, build_rank_map
    
    worldcup_path = tmp_path / "test_worldcup.json"
    worldcup_data = {
        "name": "World Cup 2026",
        "matches": [
            {
                "round": "Matchday 1",
                "date": "2026-06-11",
                "team1": "Mexico",
                "team2": "South Africa",
                "score": {"ft": [2, 0], "ht": [1, 0]},
                "group": "Group A",
                "ground": "Mexico City",
            },
            {
                "round": "Round of 32",
                "date": "2026-07-04",
                "team1": "Brazil",
                "team2": "Argentina",
                "score": {"ft": [2, 1], "ht": [1, 1]},
                "group": None,
                "ground": "Los Angeles",
            },
        ],
    }
    
    with open(worldcup_path, "w", encoding="utf-8") as f:
        json.dump(worldcup_data, f)
    
    rankings = load_rankings()
    rank_map = build_rank_map(rankings)
    
    results, metrics = backtest_worldcup(rank_map, worldcup_path)
    
    # Should have results from both group and knockout stages
    group_stage = results[results["round"].str.contains("Matchday", na=False)]
    knockout_stage = results[~results["round"].str.contains("Matchday", na=False)]
    
    assert len(group_stage) > 0
    assert len(knockout_stage) > 0
    assert metrics["group_stage_accuracy"] >= 0
    assert metrics["knockout_accuracy"] >= 0
