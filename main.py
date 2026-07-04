import argparse
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
HISTORICAL_HOME_GOAL_ADJUSTMENT = 0.18
HISTORY_STRENGTH_SCALE = 50.0
GOAL_CAP = 5


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


def load_matches(
    path: Path | str = DATA_DIR / "worldcup.json", round_contains: str | None = None
) -> pd.DataFrame:
    """Load matches from the World Cup JSON; optionally filter by substring in the `round` field."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    matches = pd.DataFrame(payload.get("matches", []))
    if matches.empty:
        return matches

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    if round_contains:
        matches = matches[
            matches["round"].str.contains(round_contains, na=False)
        ].copy()
    else:
        matches = matches.copy()
    matches["host_country"] = matches["ground"].apply(infer_host_country)
    return matches


def load_group_matches(path: Path | str = DATA_DIR / "worldcup.json") -> pd.DataFrame:
    """Backward-compatible wrapper to load group-stage matches."""
    return load_matches(path=path, round_contains="Matchday")


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


def tournament_importance_weight(tournament: str) -> float:
    """Return a relative importance weight for a historical tournament."""
    normalized = str(tournament).strip().lower()
    if "world cup" in normalized and "qualif" not in normalized:
        return 2.0
    if "friendly" in normalized:
        return 0.7
    if "qualif" in normalized or "qualification" in normalized:
        return 1.1
    if "cup" in normalized or "championship" in normalized:
        return 1.2
    return 1.0


def match_recency_weight(
    match_date: pd.Timestamp, reference_date: pd.Timestamp
) -> float:
    """Weight recent matches higher, decaying rapidly around 4 years and zero after 15 years."""
    if pd.isna(match_date):
        return 0.0

    age_years = float((reference_date - match_date).days) / 365.25
    if age_years <= 0.0:
        return 1.0
    if age_years >= 15.0:
        return 0.0
    return float(np.exp(-age_years / 4.0))


def ranking_similarity_weight(
    team1: str, team2: str, rank_map: dict[str, float]
) -> float:
    """Weight matches between similarly ranked teams higher."""
    diff = abs(rating_points_for(team1, rank_map) - rating_points_for(team2, rank_map))
    return float(np.exp(-diff / 200.0))


def historical_match_weight(
    row: pd.Series, rank_map: dict[str, float], reference_date: pd.Timestamp
) -> float:
    """Compute a combined weight for a historical match."""
    base = (
        tournament_importance_weight(row.get("tournament", ""))
        * match_recency_weight(row.get("date", pd.NaT), reference_date)
        * ranking_similarity_weight(
            row.get("home_team", ""), row.get("away_team", ""), rank_map
        )
    )
    # Allow injected rows (e.g. boosted group-stage results) to increase their influence
    boost = float(row.get("injected_group_boost", 1.0))
    return base * boost


def build_historical_team_strengths(
    historical: pd.DataFrame, rank_map: dict[str, float], reference_date: pd.Timestamp
) -> dict[str, float]:
    """Build weighted team strength offsets from historical international results."""
    historical = historical.copy()
    historical["weight"] = historical.apply(
        lambda row: historical_match_weight(row, rank_map, reference_date), axis=1
    )

    records: list[dict[str, float]] = []
    for _, row in historical.iterrows():
        home_diff = float(row["home_score"] - row["away_score"])
        away_diff = float(row["away_score"] - row["home_score"])
        records.append(
            {
                "team": row["home_team"],
                "weighted_diff": (home_diff - HISTORICAL_HOME_GOAL_ADJUSTMENT)
                * row["weight"],
                "weight": row["weight"],
            }
        )
        records.append(
            {
                "team": row["away_team"],
                "weighted_diff": (away_diff + HISTORICAL_HOME_GOAL_ADJUSTMENT)
                * row["weight"],
                "weight": row["weight"],
            }
        )

    strength_frame = pd.DataFrame.from_records(records)
    if strength_frame.empty:
        return {}

    grouped = strength_frame.groupby("team", as_index=False).agg(
        total_diff=("weighted_diff", "sum"),
        total_weight=("weight", "sum"),
    )
    grouped["strength"] = grouped["total_diff"] / grouped["total_weight"].clip(
        lower=1e-8
    )
    strengths = grouped.set_index("team")["strength"].astype(float).to_dict()
    return {str(team): float(value) for team, value in strengths.items()}


def compute_weighted_average_goals(
    historical: pd.DataFrame,
    rank_map: dict[str, float],
    reference_date: pd.Timestamp,
) -> float:
    """Compute the weighted average goals per team from historical matches."""
    if historical.empty:
        return 1.45

    weighted = historical.copy()
    weighted["weight"] = weighted.apply(
        lambda row: historical_match_weight(row, rank_map, reference_date), axis=1
    )
    total_weight = weighted["weight"].sum()
    if total_weight > 0.0:
        average_goals = ((weighted["home_score"] + weighted["away_score"]) / 2.0).mul(
            weighted["weight"]
        ).sum() / total_weight
    else:
        average_goals = (
            weighted["home_score"].mean() + weighted["away_score"].mean()
        ) / 2.0

    return float(np.clip(average_goals, 1.20, 1.70))


def apply_goal_dampening(team1_goals: float, team2_goals: float) -> tuple[int, int]:
    """Dampen extreme goal predictions to be more realistic."""
    # Reduce extreme scores to avoid unrealistic blowouts
    # Shift extreme predictions toward more realistic outcomes
    max_for_winner = 3.5
    max_for_loser = 1.5

    if team1_goals > team2_goals:
        # team1 is winning, dampen excessive goals
        if team1_goals > max_for_winner:
            team1_goals = max(2, team1_goals - 1.0)
        if team2_goals > max_for_loser:
            team2_goals = max(0, team2_goals - 0.5)
    elif team2_goals > team1_goals:
        # team2 is winning
        if team2_goals > max_for_winner:
            team2_goals = max(2, team2_goals - 1.0)
        if team1_goals > max_for_loser:
            team1_goals = max(0, team1_goals - 0.5)

    return round(team1_goals), round(team2_goals)


def backtest_last_year(
    rank_map: dict[str, float],
    days: int = 365,
    reference_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backtest the model on the most recent year of international matches."""
    historical = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    if historical.empty:
        raise ValueError("Historical results are empty; cannot run backtest.")

    if reference_date is None:
        reference_date = historical["date"].max()
    if pd.isna(reference_date):
        raise ValueError("Historical results contain no valid dates.")

    cutoff_date = reference_date - pd.Timedelta(days=days)
    training = historical[historical["date"] < cutoff_date].copy()
    test = historical[historical["date"] >= cutoff_date].copy()

    if training.empty:
        training = historical[historical["date"] < reference_date].copy()

    test = test.dropna(subset=["home_score", "away_score", "home_team", "away_team"])
    if test.empty:
        raise ValueError("No complete matches found in the last year for backtesting.")

    historical_strength = build_historical_team_strengths(
        training, rank_map, reference_date
    )
    base_goal = compute_weighted_average_goals(training, rank_map, reference_date)

    rows = []
    for _, row in test.iterrows():
        host_country = (
            None if bool(row.get("neutral", False)) else str(row.get("country", ""))
        )
        prediction = predict_match(
            row["home_team"],
            row["away_team"],
            rank_map,
            "",
            host_country,
            base_goal,
            historical_strength,
        )
        actual_result = (
            "team1"
            if float(row["home_score"]) > float(row["away_score"])
            else "team2"
            if float(row["home_score"]) < float(row["away_score"])
            else "draw"
        )
        goal_diff_error = abs(
            (prediction["team1_goals"] - prediction["team2_goals"])
            - (float(row["home_score"]) - float(row["away_score"]))
        )
        rows.append(
            {
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": int(row["home_score"]),
                "away_score": int(row["away_score"]),
                "predicted_home_goals": prediction["team1_goals"],
                "predicted_away_goals": prediction["team2_goals"],
                "predicted_result": prediction["result"],
                "actual_result": actual_result,
                "correct_outcome": prediction["result"] == actual_result,
                "exact_score": (
                    prediction["team1_goals"] == int(row["home_score"])
                    and prediction["team2_goals"] == int(row["away_score"])
                ),
                "goal_diff_error": float(goal_diff_error),
            }
        )

    results = pd.DataFrame.from_records(rows)
    metrics = {
        "matches": len(results),
        "outcome_accuracy": float(results["correct_outcome"].mean()),
        "exact_score_accuracy": float(results["exact_score"].mean()),
        "mean_absolute_goal_diff_error": float(results["goal_diff_error"].mean()),
    }
    return results, metrics


def predict_match(
    team1: str,
    team2: str,
    rank_map: dict[str, float],
    ground: str,
    host_country: str | None,
    base_goal: float,
    historical_strength: dict[str, float] | None = None,
) -> dict:
    """Predict the scoreline and outcome probabilities for one match."""
    rating1 = rating_points_for(team1, rank_map)
    rating2 = rating_points_for(team2, rank_map)
    rating_diff = rating1 - rating2

    history_diff = 0.0
    if historical_strength is not None:
        history_diff = historical_strength.get(team1, 0.0) - historical_strength.get(
            team2, 0.0
        )

    adjusted_diff = rating_diff + history_diff * HISTORY_STRENGTH_SCALE

    host_adjustment = 0.0
    if host_country is not None:
        if team1 == host_country:
            host_adjustment = HOME_ADVANTAGE
        elif team2 == host_country:
            host_adjustment = -HOME_ADVANTAGE

    p_draw = DRAW_BASE + DRAW_DECAY * np.exp(-abs(adjusted_diff) / 200.0)
    p_draw = float(np.clip(p_draw, 0.08, 0.40))
    raw_win = 1.0 / (1.0 + 10.0 ** (-adjusted_diff / 300.0))
    p_team1 = (1.0 - p_draw) * raw_win
    p_team2 = (1.0 - p_draw) * (1.0 - raw_win)

    expected_goals1 = base_goal * np.exp(
        (adjusted_diff / RATING_SCALE) + host_adjustment
    )
    expected_goals2 = base_goal * np.exp(
        (-adjusted_diff / RATING_SCALE) - host_adjustment
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

    # Apply goal cap to avoid extreme deterministic blowouts
    team1_goals = int(min(team1_goals, GOAL_CAP))
    team2_goals = int(min(team2_goals, GOAL_CAP))

    # Apply dampening to reduce unrealistic extreme scorelines
    team1_goals, team2_goals = apply_goal_dampening(team1_goals, team2_goals)

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


def predict_match_stochastic(
    team1: str,
    team2: str,
    rank_map: dict[str, float],
    ground: str,
    host_country: str | None,
    base_goal: float,
    rng: np.random.Generator | None,
    historical_strength: dict[str, float] | None = None,
) -> dict:
    """Predict one match by sampling from probabilistic outcomes."""
    if rng is None:
        rng = np.random.default_rng()

    rating1 = rating_points_for(team1, rank_map)
    rating2 = rating_points_for(team2, rank_map)
    rating_diff = rating1 - rating2

    history_diff = 0.0
    if historical_strength is not None:
        history_diff = historical_strength.get(team1, 0.0) - historical_strength.get(
            team2, 0.0
        )

    adjusted_diff = rating_diff + history_diff * HISTORY_STRENGTH_SCALE

    host_adjustment = 0.0
    if host_country is not None:
        if team1 == host_country:
            host_adjustment = HOME_ADVANTAGE
        elif team2 == host_country:
            host_adjustment = -HOME_ADVANTAGE

    p_draw = DRAW_BASE + DRAW_DECAY * np.exp(-abs(adjusted_diff) / 200.0)
    p_draw = float(np.clip(p_draw, 0.08, 0.40))
    raw_win = 1.0 / (1.0 + 10.0 ** (-adjusted_diff / 300.0))
    p_team1 = (1.0 - p_draw) * raw_win
    p_team2 = (1.0 - p_draw) * (1.0 - raw_win)

    expected_goals1 = base_goal * np.exp(
        (adjusted_diff / RATING_SCALE) + host_adjustment
    )
    expected_goals2 = base_goal * np.exp(
        (-adjusted_diff / RATING_SCALE) - host_adjustment
    )

    result = rng.choice(["team1", "draw", "team2"], p=[p_team1, p_draw, p_team2])

    if result == "draw":
        score = max(0, rng.poisson((expected_goals1 + expected_goals2) / 2.0))
        team1_goals = team2_goals = int(score)
    elif result == "team1":
        team1_goals = max(1, int(rng.poisson(expected_goals1)))
        team2_goals = max(0, int(rng.poisson(expected_goals2 * 0.65)))
        if team1_goals <= team2_goals:
            team1_goals = team2_goals + 1
    else:
        team2_goals = max(1, int(rng.poisson(expected_goals2)))
        team1_goals = max(0, int(rng.poisson(expected_goals1 * 0.65)))
        if team2_goals <= team1_goals:
            team2_goals = team1_goals + 1

    # Apply goal cap to sampled results and recompute final result
    team1_goals = int(min(team1_goals, GOAL_CAP))
    team2_goals = int(min(team2_goals, GOAL_CAP))

    # Apply dampening to reduce unrealistic extreme scorelines
    team1_goals, team2_goals = apply_goal_dampening(team1_goals, team2_goals)

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
    matches: pd.DataFrame,
    rank_map: dict[str, float],
    stochastic: bool = False,
    random_state: np.random.Generator | None = None,
    use_group_results: bool = False,
    group_weight_factor: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict every group-stage match and simulate the resulting tables."""
    if stochastic and random_state is None:
        random_state = np.random.default_rng()

    historical = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    reference_date = (
        matches["date"].max()
        if "date" in matches.columns and pd.notna(matches["date"].max())
        else pd.Timestamp("2026-06-01")
    )
    # Optionally inject completed group-stage results (from worldcup.json)
    if use_group_results:
        group_matches = load_group_matches()

        # Keep only matches that have a final score
        def has_ft_score(s):
            try:
                return bool(s and isinstance(s.get("ft", None), list))
            except Exception:
                return False

        injected = []
        for _, gm in group_matches.iterrows():
            if has_ft_score(gm.get("score", {})):
                ft = gm["score"]["ft"]
                injected.append(
                    {
                        "date": gm["date"],
                        "tournament": "World Cup 2026",
                        "home_team": gm["team1"],
                        "away_team": gm["team2"],
                        "home_score": int(ft[0]),
                        "away_score": int(ft[1]),
                        "injected_group_boost": float(group_weight_factor),
                    }
                )
        if injected:
            injected_df = pd.DataFrame.from_records(injected)
            # Avoid duplicating matches that already exist in historical by date+teams
            if not historical.empty:
                merged = historical.merge(
                    injected_df,
                    left_on=["date", "home_team", "away_team"],
                    right_on=["date", "home_team", "away_team"],
                    how="left",
                    indicator=True,
                )
                # Keep only historical rows that are not exact matches (to avoid duplicates)
                historical = (
                    merged[merged["_merge"] == "left_only"]
                    .drop(columns=["_merge"])
                    .reindex(columns=historical.columns)
                )
            historical = pd.concat(
                [historical, injected_df], ignore_index=True, sort=False
            )

    historical["weight"] = historical.apply(
        lambda row: historical_match_weight(row, rank_map, reference_date), axis=1
    )
    historical_strength = build_historical_team_strengths(
        historical, rank_map, reference_date
    )
    base_goal = compute_weighted_average_goals(historical, rank_map, reference_date)

    predictions = []
    for _, row in matches.iterrows():
        if stochastic:
            prediction = predict_match_stochastic(
                row["team1"],
                row["team2"],
                rank_map,
                row.get("ground", ""),
                row.get("host_country"),
                base_goal,
                random_state,
                historical_strength,
            )
        else:
            prediction = predict_match(
                row["team1"],
                row["team2"],
                rank_map,
                row.get("ground", ""),
                row.get("host_country"),
                base_goal,
                historical_strength,
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


def predict_knockout_round(
    round_name: str,
    rank_map: dict[str, float],
    deterministic: bool = True,
    use_group_results: bool = False,
    group_weight_factor: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Predict matches for a knockout round (e.g., 'Round of 32').

    Completed matches present in the JSON with a final `score.ft` are used as ground truth.
    """
    matches = load_matches(round_contains=round_name)
    if matches.empty:
        return pd.DataFrame()

    # Prepare historical strengths and base goals (optionally injecting group results)
    historical = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    reference_date = (
        matches["date"].max()
        if pd.notna(matches["date"].max())
        else pd.Timestamp("2026-06-01")
    )

    if use_group_results:
        group_matches = load_group_matches()
        injected = []

        def has_ft_score(s):
            try:
                return bool(s and isinstance(s.get("ft", None), list))
            except Exception:
                return False

        for _, gm in group_matches.iterrows():
            if has_ft_score(gm.get("score", {})):
                ft = gm["score"]["ft"]
                injected.append(
                    {
                        "date": gm["date"],
                        "tournament": "World Cup 2026",
                        "home_team": gm["team1"],
                        "away_team": gm["team2"],
                        "home_score": int(ft[0]),
                        "away_score": int(ft[1]),
                        "injected_group_boost": float(group_weight_factor),
                    }
                )
        if injected:
            injected_df = pd.DataFrame.from_records(injected)
            if not historical.empty:
                merged = historical.merge(
                    injected_df,
                    left_on=["date", "home_team", "away_team"],
                    right_on=["date", "home_team", "away_team"],
                    how="left",
                    indicator=True,
                )
                historical = (
                    merged[merged["_merge"] == "left_only"]
                    .drop(columns=["_merge"])
                    .reindex(columns=historical.columns)
                )
            historical = pd.concat(
                [historical, injected_df], ignore_index=True, sort=False
            )

    historical["weight"] = historical.apply(
        lambda row: historical_match_weight(row, rank_map, reference_date), axis=1
    )
    historical_strength = build_historical_team_strengths(
        historical, rank_map, reference_date
    )
    base_goal = compute_weighted_average_goals(historical, rank_map, reference_date)

    rng = np.random.default_rng(seed) if seed is not None else None
    predictions = []

    for _, row in matches.iterrows():
        actual_score = None
        if isinstance(row.get("score"), dict) and isinstance(
            row["score"].get("ft"), list
        ):
            ft = row["score"]["ft"]
            actual_score = (int(ft[0]), int(ft[1]))

        if actual_score is not None:
            # compute probabilities but use actual goals
            probs = predict_match(
                row["team1"],
                row["team2"],
                rank_map,
                row.get("ground", ""),
                row.get("host_country"),
                base_goal,
                historical_strength,
            )
            probs["team1_goals"] = int(actual_score[0])
            probs["team2_goals"] = int(actual_score[1])
            if probs["team1_goals"] > probs["team2_goals"]:
                probs["result"] = "team1"
            elif probs["team1_goals"] < probs["team2_goals"]:
                probs["result"] = "team2"
            else:
                probs["result"] = "draw"
            prediction = probs
        else:
            if deterministic:
                prediction = predict_match(
                    row["team1"],
                    row["team2"],
                    rank_map,
                    row.get("ground", ""),
                    row.get("host_country"),
                    base_goal,
                    historical_strength,
                )
            else:
                prediction = predict_match_stochastic(
                    row["team1"],
                    row["team2"],
                    rank_map,
                    row.get("ground", ""),
                    row.get("host_country"),
                    base_goal,
                    rng,
                    historical_strength,
                )

        prediction["group"] = row.get("group")
        prediction["round"] = row.get("round")
        predictions.append(prediction)

    prediction_frame = pd.DataFrame.from_records(predictions)
    return prediction_frame


def summarize_monte_carlo(
    matches: pd.DataFrame,
    rank_map: dict[str, float],
    simulations: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run multiple stochastic simulations and summarize team distribution statistics."""
    rng = np.random.default_rng(seed)
    records = []

    for simulation in range(1, simulations + 1):
        _, grouped = simulate_group_stage(
            matches, rank_map, stochastic=True, random_state=rng
        )
        for group, entries in grouped.groupby("group"):
            sorted_entries = entries.sort_values(
                ["points", "goal_difference", "goals_for"],
                ascending=[False, False, False],
            )
            for position, row in enumerate(
                sorted_entries.itertuples(index=False), start=1
            ):
                records.append(
                    {
                        "simulation": simulation,
                        "group": group,
                        "team": row.team,
                        "position": position,
                        "points": row.points,
                        "goal_difference": row.goal_difference,
                        "goals_for": row.goals_for,
                    }
                )

    all_results = pd.DataFrame.from_records(records)
    summary = (
        all_results.groupby(["group", "team"], as_index=False)
        .agg(
            simulations=("position", "size"),
            first_count=("position", lambda x: (x == 1).sum()),
            second_count=("position", lambda x: (x == 2).sum()),
            third_count=("position", lambda x: (x == 3).sum()),
            fourth_count=("position", lambda x: (x == 4).sum()),
            avg_points=("points", "mean"),
            avg_goal_difference=("goal_difference", "mean"),
            avg_goals_for=("goals_for", "mean"),
        )
        .assign(
            prob_first=lambda df: df["first_count"] / simulations,
            prob_second=lambda df: df["second_count"] / simulations,
            prob_third=lambda df: df["third_count"] / simulations,
            prob_fourth=lambda df: df["fourth_count"] / simulations,
            prob_top2=lambda df: (df["first_count"] + df["second_count"]) / simulations,
        )
    )

    return summary.sort_values(
        ["group", "prob_top2", "prob_first"], ascending=[True, False, False]
    )


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
    parser = argparse.ArgumentParser(
        description=(
            "Predict 2026 World Cup group-stage results deterministically, "
            "stochastically, or with Monte Carlo simulations."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["deterministic", "stochastic", "montecarlo", "backtest"],
        default="deterministic",
        help="Choose deterministic, stochastic, Monte Carlo, or backtest mode.",
    )
    parser.add_argument(
        "--use-group-results",
        action="store_true",
        help="Use completed group-stage results from data/worldcup.json as ground truth and inject them into historical strengths.",
    )
    parser.add_argument(
        "--group-weight-factor",
        type=float,
        default=1.0,
        help="Multiplier to increase influence of injected group-stage results when building historical strengths.",
    )
    parser.add_argument(
        "--predict-round",
        type=str,
        default=None,
        help="If set, predict the specified knockout round (e.g., 'Round of 32') and export to a separate CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for stochastic predictions and Monte Carlo simulations.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations when mode is montecarlo.",
    )
    args = parser.parse_args()

    rankings = load_rankings()
    rank_map = build_rank_map(rankings)
    matches = load_group_matches()

    # If the user asked to predict a knockout round, do that first and exit
    if args.predict_round:
        deterministic = args.mode == "deterministic"
        predictions = predict_knockout_round(
            args.predict_round,
            rank_map,
            deterministic=deterministic,
            use_group_results=args.use_group_results,
            group_weight_factor=args.group_weight_factor,
            seed=args.seed,
        )
        out_path = (
            Path(__file__).resolve().parent
            / f"predicted_{args.predict_round.replace(' ', '_').lower()}.csv"
        )
        if not predictions.empty:
            export_match_predictions(predictions, out_path)
            print(f"Wrote knockout predictions to {out_path}")
        else:
            print(f"No matches found for round '{args.predict_round}' in worldcup.json")
        return

    if args.mode == "montecarlo":
        summary = summarize_monte_carlo(
            matches, rank_map, args.simulations, seed=args.seed
        )
        csv_path = Path(__file__).resolve().parent / "montecarlo_group_distribution.csv"
        summary.to_csv(csv_path, index=False)
        print(f"Wrote Monte Carlo summary to {csv_path}")
        print(f"Ran {args.simulations} Monte Carlo simulations")
        for group, entries in summary.groupby("group"):
            print(f"\n{group}")
            for row in (
                entries.sort_values("prob_top2", ascending=False).head(4).itertuples()
            ):
                print(
                    f"{row.team}: top2={row.prob_top2:.3f}, first={row.prob_first:.3f}, "
                    f"avg_pts={row.avg_points:.2f}, avg_gd={row.avg_goal_difference:.2f}"
                )
    elif args.mode == "backtest":
        results, metrics = backtest_last_year(rank_map)
        csv_path = Path(__file__).resolve().parent / "backtest_last_year_results.csv"
        results.to_csv(csv_path, index=False)
        print(f"Wrote backtest results to {csv_path}")
        print("\nBacktest metrics")
        print(f"Matches: {metrics['matches']}")
        print(f"Outcome accuracy: {metrics['outcome_accuracy']:.3f}")
        print(f"Exact score accuracy: {metrics['exact_score_accuracy']:.3f}")
        print(
            f"Mean absolute goal difference error: {metrics['mean_absolute_goal_diff_error']:.3f}"
        )
        print("\nSample backtest rows")
        for _, row in results.sort_values(["date"]).head(10).iterrows():
            print(
                f"{row['date'].date()}: {row['home_team']} {row['home_score']}-{row['away_score']} "
                f"predicted {row['predicted_home_goals']}-{row['predicted_away_goals']} "
                f"(correct_outcome={row['correct_outcome']})"
            )
    else:
        predictions, group_table = simulate_group_stage(
            matches,
            rank_map,
            stochastic=args.mode == "stochastic",
            random_state=(
                np.random.default_rng(args.seed) if args.seed is not None else None
            ),
            use_group_results=args.use_group_results,
            group_weight_factor=args.group_weight_factor,
        )

        csv_path = (
            Path(__file__).resolve().parent
            / f"predicted_group_stage_matches_{args.mode}.csv"
        )
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
