import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

TEAM_NAME_MAP = {
    "South Korea": "Korea Republic",
    "Czech Republic": "Czechia",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Ivory Coast": "Côte d'Ivoire",
    "Turkey": "Türkiye",
    "Iran": "IR Iran",
    "Cape Verde": "Cabo Verde",
}

HOST_COUNTRY_CITIES = {
    "Mexico": [
        "Mexico City",
        "Guadalajara",
        "Monterrey",
        "Zapopan",
        "Guadalajara (Zapopan)",
    ],
    "Canada": [
        "Toronto",
        "Vancouver",
        "Montreal",
        "Edmonton",
        "Winnipeg",
        "Ottawa",
        "Calgary",
    ],
    "USA": [
        "New York",
        "East Rutherford",
        "Boston",
        "Foxborough",
        "Los Angeles",
        "San Francisco",
        "Santa Clara",
        "Seattle",
        "Atlanta",
        "Philadelphia",
        "Houston",
        "Dallas",
        "Kansas City",
        "Miami",
        "Denver",
        "Cincinnati",
        "Charlotte",
        "Minneapolis",
        "Jacksonville",
        "Orlando",
        "Salt Lake",
    ],
}

DEFAULT_RANK_POINTS = 1500.0
RATING_SCALE = 420.0
DRAW_BASE = 0.18
DRAW_DECAY = 0.12
GOAL_SCALING_FACTOR = 1.47
HOME_ADVANTAGE = 0.08


def normalize_team_name(team: str) -> str:
    """Return a canonical ranking name for a team."""
    return TEAM_NAME_MAP.get(team, team)


def load_rankings(path: Path | str = DATA_DIR / "rankings.json") -> pd.DataFrame:
    """Load FIFA ranking data and normalize team names."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    rankings = pd.json_normalize(payload["Results"])
    rankings["Team"] = rankings["TeamName"].apply(
        lambda teams: next(
            (entry["Description"] for entry in teams if entry.get("Locale") == "en-GB"),
            teams[0]["Description"],
        )
    )
    rankings["TotalPoints"] = rankings["TotalPoints"].astype(float)
    return rankings[["Team", "TotalPoints", "ConfederationName"]]


def infer_host_country(ground: str) -> str | None:
    """Infer the host country from a stadium or city name."""
    for country, city_list in HOST_COUNTRY_CITIES.items():
        for city in city_list:
            if city in ground:
                return country
    return None


def load_group_matches(path: Path | str = DATA_DIR / "worldcup.json") -> pd.DataFrame:
    """Load the World Cup group-stage schedule."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    matches = pd.DataFrame(payload["matches"])
    matches = matches[matches["round"].str.contains("Matchday", na=False)].copy()
    matches["host_country"] = matches["ground"].apply(infer_host_country)
    return matches


def build_rank_map(rankings: pd.DataFrame) -> dict[str, float]:
    """Build a simple mapping from team name to ranking points."""
    rank_map = rankings.set_index("Team")["TotalPoints"].to_dict()
    normalized_rank_map = {}
    for team, points in rank_map.items():
        normalized_rank_map[team] = points
    return normalized_rank_map


def rating_points_for(team: str, rank_map: dict[str, float]) -> float:
    """Look up the ranking points for a team, with fallback defaults."""
    normalized = normalize_team_name(team)
    if normalized in rank_map:
        return rank_map[normalized]
    if team in rank_map:
        return rank_map[team]
    return DEFAULT_RANK_POINTS


def predict_match(
    team1: str,
    team2: str,
    rank_map: dict[str, float],
    ground: str,
    host_country: str | None,
    base_goal: float,
) -> dict:
    """Predict the scoreline and outcome probabilities for one match."""
    rating1 = rating_points_for(team1, rank_map)
    rating2 = rating_points_for(team2, rank_map)
    rating_diff = rating1 - rating2

    host_adjustment = 0.0
    if host_country is not None:
        if team1 == host_country:
            host_adjustment = HOME_ADVANTAGE
        elif team2 == host_country:
            host_adjustment = -HOME_ADVANTAGE

    p_draw = DRAW_BASE + DRAW_DECAY * np.exp(-abs(rating_diff) / 200.0)
    p_draw = float(np.clip(p_draw, 0.08, 0.40))
    raw_win = 1.0 / (1.0 + 10.0 ** (-rating_diff / 300.0))
    p_team1 = (1.0 - p_draw) * raw_win
    p_team2 = (1.0 - p_draw) * (1.0 - raw_win)

    expected_goals1 = base_goal * np.exp((rating_diff / RATING_SCALE) + host_adjustment)
    expected_goals2 = base_goal * np.exp(
        (-rating_diff / RATING_SCALE) - host_adjustment
    )

    if p_draw >= 0.30:
        score = max(0, round((expected_goals1 + expected_goals2) / 2.0))
        team1_goals = team2_goals = score
    elif p_team1 > p_team2:
        team1_goals = max(1, round(expected_goals1))
        team2_goals = max(0, round(expected_goals2 * 0.65))
    else:
        team2_goals = max(1, round(expected_goals2))
        team1_goals = max(0, round(expected_goals1 * 0.65))

    if team1_goals > team2_goals:
        result = "team1"
    elif team1_goals < team2_goals:
        result = "team2"
    else:
        result = "draw"

    return {
        "team1": team1,
        "team2": team2,
        "group": None,
        "ground": ground,
        "team1_goals": int(team1_goals),
        "team2_goals": int(team2_goals),
        "prob_team1": float(p_team1),
        "prob_draw": float(p_draw),
        "prob_team2": float(p_team2),
        "result": result,
        "rating1": float(rating1),
        "rating2": float(rating2),
        "host_country": host_country,
    }


def simulate_group_stage(
    matches: pd.DataFrame, rank_map: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict every group-stage match and simulate the resulting tables."""
    historical = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    average_goals = (
        historical["home_score"].mean() + historical["away_score"].mean()
    ) / 2.0
    base_goal = float(np.clip(average_goals, 1.20, 1.70))

    predictions = []
    for _, row in matches.iterrows():
        prediction = predict_match(
            row["team1"],
            row["team2"],
            rank_map,
            row.get("ground", ""),
            row.get("host_country"),
            base_goal,
        )
        prediction["group"] = row["group"]
        prediction["round"] = row["round"]
        predictions.append(prediction)

    prediction_frame = pd.DataFrame(predictions)
    prediction_frame["points1"] = np.where(
        prediction_frame["team1_goals"] > prediction_frame["team2_goals"],
        3,
        np.where(
            prediction_frame["team1_goals"] == prediction_frame["team2_goals"], 1, 0
        ),
    )
    prediction_frame["points2"] = np.where(
        prediction_frame["team2_goals"] > prediction_frame["team1_goals"],
        3,
        np.where(
            prediction_frame["team1_goals"] == prediction_frame["team2_goals"], 1, 0
        ),
    )

    rows = []
    for _, row in prediction_frame.iterrows():
        rows.append(
            {
                "group": row["group"],
                "team": row["team1"],
                "played": 1,
                "wins": int(row["team1_goals"] > row["team2_goals"]),
                "draws": int(row["team1_goals"] == row["team2_goals"]),
                "losses": int(row["team1_goals"] < row["team2_goals"]),
                "goals_for": int(row["team1_goals"]),
                "goals_against": int(row["team2_goals"]),
                "points": int(row["points1"]),
            }
        )
        rows.append(
            {
                "group": row["group"],
                "team": row["team2"],
                "played": 1,
                "wins": int(row["team2_goals"] > row["team1_goals"]),
                "draws": int(row["team2_goals"] == row["team1_goals"]),
                "losses": int(row["team2_goals"] < row["team1_goals"]),
                "goals_for": int(row["team2_goals"]),
                "goals_against": int(row["team1_goals"]),
                "points": int(row["points2"]),
            }
        )

    table = pd.DataFrame(rows)
    grouped = table.groupby(["group", "team"], as_index=False).sum()
    grouped["goal_difference"] = grouped["goals_for"] - grouped["goals_against"]
    grouped["points"] = grouped["points"].astype(int)
    grouped["goal_difference"] = grouped["goal_difference"].astype(int)

    grouped = grouped.sort_values(
        ["group", "points", "goal_difference", "goals_for"],
        ascending=[True, False, False, False],
    )

    return prediction_frame, grouped


def export_match_predictions(predictions: pd.DataFrame, path: Path) -> None:
    """Export all match predictions to a CSV file."""
    sorted_predictions = predictions.sort_values(["group", "round", "team1"])
    sorted_predictions.to_csv(path, index=False)


def print_group_predictions(grouped: pd.DataFrame) -> None:
    """Print group table predictions in readable format."""
    for group, entries in grouped.groupby("group"):
        print(f"\n{group}")
        for _idx, row in entries.iterrows():
            print(
                f"{row['team']}: {row['points']} pts, {row['goal_difference']} GD, {row['goals_for']} GF, "
                f"{row['goals_against']} GA, {row['wins']}-{row['draws']}-{row['losses']}"
            )


def main() -> None:
    """Run the group-stage prediction pipeline and print results."""
    rankings = load_rankings()
    rank_map = build_rank_map(rankings)
    matches = load_group_matches()
    predictions, group_table = simulate_group_stage(matches, rank_map)

    csv_path = Path(__file__).resolve().parent / "predicted_group_stage_matches.csv"
    export_match_predictions(predictions, csv_path)
    print(f"Wrote all match predictions to {csv_path}")

    print("\nPredicted group standings")
    print_group_predictions(group_table)

    print("\nSample match predictions")
    for _, row in predictions.sort_values(["group", "round"]).head(12).iterrows():
        print(
            f"{row['group']} {row['round']}: {row['team1']} {row['team1_goals']}-{row['team2_goals']} {row['team2']} "
            f"(P1={row['prob_team1']:.2f}, D={row['prob_draw']:.2f}, P2={row['prob_team2']:.2f})"
        )


if __name__ == "__main__":
    main()
