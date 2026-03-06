"""
Synthetic Data Generation for FPL ML Testing

Generates realistic test data for ML system testing including:
- Players with realistic stats and distributions
- Teams with balanced strengths
- Fixtures with proper constraints
- Edge cases and corner scenarios
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def generate_teams(n_teams: int = 20) -> pd.DataFrame:
    """
    Generate Premier League teams with balanced strengths.

    Args:
        n_teams: Number of teams (default 20 for PL)

    Returns:
        DataFrame with teams and strength ratings
    """
    np.random.seed(42)

    teams = []
    for i in range(1, n_teams + 1):
        # Generate strengths around 1000 ± 100
        attack_home = np.random.normal(1000, 80)
        attack_away = np.random.normal(950, 80)
        defense_home = np.random.normal(1000, 80)
        defense_away = np.random.normal(950, 80)

        teams.append(
            {
                "id": i,
                "name": f"Team {i}",
                "short_name": f"T{i}",
                "strength_attack_home": max(800, min(1200, attack_home)),
                "strength_attack_away": max(800, min(1200, attack_away)),
                "strength_defence_home": max(800, min(1200, defense_home)),
                "strength_defence_away": max(800, min(1200, defense_away)),
            }
        )

    df = pd.DataFrame(teams)

    # Ensure some variation: top 5 teams, mid-table, relegation zone
    # Sort by attack home strength
    df = df.sort_values("strength_attack_home", ascending=False).reset_index(drop=True)

    return df


def generate_players(
    teams_df: pd.DataFrame, players_per_position: Dict[int, int] = None
) -> pd.DataFrame:
    """
    Generate realistic player data.

    Args:
        teams_df: Teams DataFrame
        players_per_position: Dict of {position: count} for player generation
            Default: {1: 20, 2: 80, 3: 140, 4: 100} (total 340)

    Returns:
        DataFrame with players
    """
    np.random.seed(42)

    if players_per_position is None:
        players_per_position = {
            1: 20,  # GK: 2 per team avg
            2: 80,  # DEF: ~4 per team avg
            3: 140,  # MID: ~7 per team avg
            4: 100,  # FWD: ~5 per team avg
        }

    players = []
    player_id = 1

    position_weights = {
        1: {
            "cost_range": (40, 60),
            "minutes_range": (1500, 2700),
            "goal_range": (0, 2),
            "assist_range": (0, 3),
        },
        2: {
            "cost_range": (45, 90),
            "minutes_range": (1000, 2700),
            "goal_range": (0, 8),
            "assist_range": (0, 10),
        },
        3: {
            "cost_range": (55, 130),
            "minutes_range": (500, 2700),
            "goal_range": (2, 15),
            "assist_range": (1, 15),
        },
        4: {
            "cost_range": (60, 150),
            "minutes_range": (500, 2700),
            "goal_range": (5, 25),
            "assist_range": (1, 12),
        },
    }

    # Name pools for variety
    first_names = [
        "James",
        "John",
        "Mohamed",
        "Raheem",
        "Kevin",
        "Bruno",
        "Jadon",
        "Phil",
        "Erling",
        "Kylian",
        "Lionel",
        "Cristiano",
        "Harry",
        "Son",
        "Jude",
        "Declan",
        "Jack",
        "Bukayo",
        "Martin",
        "Aaron",
        "Cole",
        "Callum",
    ]
    last_names = [
        "Smith",
        "Jones",
        "Wilson",
        "Taylor",
        "Brown",
        "Davies",
        "Evans",
        "Johnson",
        "Martinez",
        "Grealish",
        "Salah",
        "Haaland",
        "De Bruyne",
        "Foden",
        "Sancho",
        "Mount",
        "Kane",
        "Heung-min",
        "Bellingham",
        "Rice",
        "Grealish",
        "Palmer",
    ]

    for pos_type, count in players_per_position.items():
        for _ in range(count):
            team = np.random.choice(teams_df["id"].values)

            # Get position defaults
            pos_defaults = position_weights[pos_type]

            # Cost in tenths (FPL format: £4.0m = 40)
            base_cost = np.random.randint(
                pos_defaults["cost_range"][0], pos_defaults["cost_range"][1] + 1
            )
            cost = base_cost * 10

            minutes = np.random.randint(
                pos_defaults["minutes_range"][0], pos_defaults["minutes_range"][1] + 1
            )
            expected_goals = round(
                np.random.uniform(
                    pos_defaults["goal_range"][0] / 38 * (minutes / 90),
                    pos_defaults["goal_range"][1] / 38 * (minutes / 90),
                ),
                1,
            )
            expected_assists = round(
                np.random.uniform(
                    pos_defaults["assist_range"][0] / 38 * (minutes / 90),
                    pos_defaults["assist_range"][1] / 38 * (minutes / 90),
                ),
                1,
            )

            # ICT index correlates with attacking stats
            ict_base = 20 + (expected_goals + expected_assists) * 5
            ict_index = round(np.random.normal(ict_base, 30), 1)
            ict_index = max(0, ict_index)

            # Form correlated with ICT and recent performance
            form = round(np.random.normal(5.0, 1.5), 1)
            form = max(0, min(10, form))

            # ep_next correlated with form and expected points
            ep_next = round(form * 0.7 + np.random.normal(0, 1), 1)
            ep_next = max(0, min(12, ep_next))

            # Recent minutes (last 4 games)
            recent_4_minutes = np.random.randint(
                max(0, minutes - 360), min(360, minutes + 45)
            )

            player = {
                "id": player_id,
                "web_name": f"{np.random.choice(first_names)} {np.random.choice(last_names)}",
                "element_type": pos_type,
                "team": team,
                "now_cost": cost,
                "minutes": minutes,
                "starts": int(minutes / 90 * np.random.uniform(0.7, 0.95)),
                "goals_scored": np.random.poisson(expected_goals),
                "assists": np.random.poisson(expected_assists),
                "expected_goals": expected_goals,
                "expected_assists": expected_assists,
                "ict_index": ict_index,
                "form": form,
                "ep_next": ep_next,
                "chance_of_playing_next_round": 100,
                "status": np.random.choice(
                    ["a", "a", "a", "a", "d", "i"],
                    p=[0.85, 0.05, 0.05, 0.03, 0.01, 0.01],
                ),
                "influence": ict_index * np.random.uniform(0.8, 1.2),
                "recent_4_minutes": recent_4_minutes,
                # Understat enrichment flag (random subset)
                "understat_matched": np.random.random() > 0.2,
                "understat_xG_per_90": round(np.random.uniform(0.05, 1.0), 2)
                if np.random.random() > 0.2
                else None,
                "understat_xA_per_90": round(np.random.uniform(0.02, 0.5), 2)
                if np.random.random() > 0.2
                else None,
                "understat_minutes": minutes if np.random.random() > 0.2 else 0,
            }
            players.append(player)
            player_id += 1

    return pd.DataFrame(players)


def generate_fixtures(teams_df: pd.DataFrame, num_gameweeks: int = 10) -> List[Dict]:
    """
    Generate fixture list with balanced home/away distribution.

    Args:
        teams_df: Teams DataFrame
        num_gameweeks: Number of gameweeks to generate

    Returns:
        List of fixture dictionaries
    """
    np.random.seed(42)

    n_teams = len(teams_df)
    assert n_teams % 2 == 0, "Must have even number of teams"

    fixtures = []
    fixture_id = 1

    for gw in range(1, num_gameweeks + 1):
        # Create round-robin schedule
        teams = teams_df["id"].tolist()
        np.random.shuffle(teams)

        # Pair teams
        for i in range(0, n_teams, 2):
            if i + 1 < n_teams:
                home = teams[i]
                away = teams[i + 1]

                fixture = {
                    "id": fixture_id,
                    "event": gw,
                    "team_h": int(home),
                    "team_a": int(away),
                    "team_h_difficulty": np.random.randint(2, 5),
                    "team_a_difficulty": np.random.randint(2, 5),
                    "finished": gw < 2,  # First GW is "finished"
                    "kickoff_time": f"2025-01-{(gw % 28) + 1:02d}T{15 + i // 2:02d}:00:00Z",
                }
                fixtures.append(fixture)
                fixture_id += 1

    return fixtures


def generate_history_for_player(player: Dict, num_games: int = 10) -> List[Dict]:
    """
    Generate game-by-game history for a single player.

    Args:
        player: Player dictionary
        num_games: Number of past games to generate

    Returns:
        List of game history entries
    """
    history = []

    # Base rate from player stats
    base_xg_per_90 = player["expected_goals"] / max(1, player["minutes"]) * 90
    base_xa_per_90 = player["expected_assists"] / max(1, player["minutes"]) * 90

    avg_minutes = player["minutes"] / max(1, num_games)

    for game_num in range(num_games):
        # Random variation
        minutes_played = np.random.randint(
            max(0, avg_minutes - 30), min(90, avg_minutes + 30)
        )

        # Points vary based on minutes, xg, xa
        if minutes_played >= 60:
            appearance_points = 2
        elif minutes_played > 0:
            appearance_points = 1
        else:
            appearance_points = 0

        # Random scoring events (Poisson)
        goals = np.random.poisson(max(0, base_xg_per_90 * minutes_played / 90))
        assists = np.random.poisson(max(0, base_xa_per_90 * minutes_played / 90))

        # FPL points
        pos = player["element_type"]
        goal_points = goals * {1: 6, 2: 6, 3: 5, 4: 4}[pos]
        assist_points = assists * 3

        total_points = (
            appearance_points + goal_points + assist_points + np.random.randint(0, 4)
        )  # Bonus + other

        game = {
            "event": game_num + 1,
            "minutes": minutes_played,
            "total_points": total_points,
            "goals_scored": goals,
            "assists": assists,
            "clean_sheets": int(np.random.random() < 0.2) if pos in [1, 2] else 0,
            "bps": np.random.randint(0, 30),
        }
        history.append(game)

    return history


def generate_history_cache(players_df: pd.DataFrame, max_players: int = 100) -> Dict:
    """
    Generate history cache for subset of players.

    Args:
        players_df: Players DataFrame
        max_players: Maximum number of players to generate history for

    Returns:
        Dictionary mapping player_id -> {'history': [...]}
    """
    cache = {}

    for _, player in players_df.head(max_players).iterrows():
        num_games = np.random.randint(5, 20)
        history = generate_history_for_player(player.to_dict(), num_games)
        cache[str(player["id"])] = {"history": history}

    return cache


def save_fixtures_json(fixtures: List[Dict], path: str):
    """Save fixtures to JSON."""
    ensure_dir(os.path.dirname(path))

    with open(path, "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"✅ Saved {len(fixtures)} fixtures to {path}")


def save_dataframe_json(df: pd.DataFrame, path: str):
    """Save DataFrame to JSON (records format)."""
    ensure_dir(os.path.dirname(path))

    df.to_json(path, orient="records", indent=2)
    print(f"✅ Saved {len(df)} records to {path}")


def save_dataframe_csv(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV."""
    ensure_dir(os.path.dirname(path))

    df.to_csv(path, index=False)
    print(f"✅ Saved {len(df)} records to {path}")


def generate_full_test_suite(
    output_dir: str = "backend/tests/fixtures",
    n_teams: int = 20,
    num_gameweeks: int = 12,
    include_history: bool = True,
    include_golden: bool = True,
):
    """
    Generate complete test data suite.

    Args:
        output_dir: Output directory for fixtures
        n_teams: Number of teams to generate
        num_gameweeks: Number of gameweeks
        include_history: Whether to generate player history cache
        include_golden: Whether to generate golden dataset
    """
    print("=" * 60)
    print("FPL ML Test Data Generator")
    print("=" * 60)

    # Generate teams
    print("\n1. Generating teams...")
    teams_df = generate_teams(n_teams)
    save_dataframe_json(teams_df, os.path.join(output_dir, "teams_sample.json"))

    # Generate players
    print("\n2. Generating players...")
    players_df = generate_players(teams_df)
    save_dataframe_json(players_df, os.path.join(output_dir, "players_sample.json"))
    save_dataframe_csv(players_df, os.path.join(output_dir, "players_sample.csv"))

    # Generate fixtures
    print("\n3. Generating fixtures...")
    fixtures = generate_fixtures(teams_df, num_gameweeks)
    save_fixtures_json(fixtures, os.path.join(output_dir, "fixtures_sample.json"))

    # Generate player history
    if include_history:
        print("\n4. Generating player history cache...")
        history_cache = generate_history_cache(players_df, max_players=100)

        history_path = os.path.join(output_dir, "history_cache.json")
        with open(history_path, "w") as f:
            json.dump(history_cache, f, indent=2)
        print(f"✅ Saved history for {len(history_cache)} players to {history_path}")

    # Generate golden dataset (small, fixed sample for regression)
    if include_golden:
        print("\n5. Generating golden dataset...")
        golden_players = players_df.head(20).copy()

        # Ensure golden data is deterministic
        golden_players["expected_goals"] = golden_players["expected_goals"].round(2)
        golden_players["expected_assists"] = golden_players["expected_assists"].round(2)
        golden_players["ict_index"] = golden_players["ict_index"].round(2)
        golden_players["form"] = golden_players["form"].round(1)

        save_dataframe_json(
            golden_players, os.path.join(output_dir, "golden_players.json")
        )

    # Generate metadata
    print("\n6. Generating metadata...")
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "n_teams": len(teams_df),
        "n_players": len(players_df),
        "n_fixtures": len(fixtures),
        "num_gameweeks": num_gameweeks,
        "has_history": include_history,
        "positions": {
            "GK": len(players_df[players_df["element_type"] == 1]),
            "DEF": len(players_df[players_df["element_type"] == 2]),
            "MID": len(players_df[players_df["element_type"] == 3]),
            "FWD": len(players_df[players_df["element_type"] == 4]),
        },
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {metadata_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Data Generation Complete!")
    print("=" * 60)
    print(f"📊 Teams: {len(teams_df)}")
    print(f"👥 Players: {len(players_df)}")
    print(f"📅 Fixtures: {len(fixtures)}")
    if include_history:
        print(
            f"📜 History entries: {sum(len(h['history']) for h in history_cache.values())}"
        )
    if include_golden:
        print(f"✨ Golden players: {len(golden_players)}")
    print(f"\n📁 Output directory: {os.path.abspath(output_dir)}")
    print("=" * 60)


def generate_edge_cases(output_dir: str = "backend/tests/fixtures"):
    """
    Generate edge case test data for specific scenarios.

    Edge cases:
    - Player with 0 minutes
    - Player with 100% injury rate
    - Team with no fixtures
    - Duplicate player IDs (should be caught by validation)
    - Extremely high/low values
    """
    print("\nGenerating edge case fixtures...")

    edge_cases = []

    # 1. Player with 0 minutes
    edge_cases.append(
        {
            "id": 9999,
            "web_name": "NoMinutesMan",
            "element_type": 3,
            "team": 1,
            "now_cost": 50,
            "minutes": 0,
            "starts": 0,
            "goals_scored": 0,
            "assists": 0,
            "expected_goals": 0.0,
            "expected_assists": 0.0,
            "ict_index": 0.0,
            "form": 0.0,
            "ep_next": 0.0,
            "status": "a",
        }
    )

    # 2. Player with extremely high minutes (fatigue test)
    edge_cases.append(
        {
            "id": 9998,
            "web_name": "IronMan",
            "element_type": 2,
            "team": 1,
            "now_cost": 70,
            "minutes": 5000,  # Unrealistic
            "starts": 60,
            "goals_scored": 50,
            "assists": 40,
            "expected_goals": 45.0,
            "expected_assists": 35.0,
            "ict_index": 500.0,
            "form": 9.5,
            "ep_next": 12.0,
            "status": "a",
        }
    )

    # 3. Extremely cheap player
    edge_cases.append(
        {
            "id": 9997,
            "web_name": "BudgetKing",
            "element_type": 4,
            "team": 20,
            "now_cost": 35,  # £3.5m minimum
            "minutes": 1800,
            "starts": 25,
            "goals_scored": 5,
            "assists": 3,
            "expected_goals": 4.5,
            "expected_assists": 2.5,
            "ict_index": 120.0,
            "form": 6.0,
            "ep_next": 4.2,
            "status": "a",
        }
    )

    df = pd.DataFrame(edge_cases)
    save_dataframe_json(df, os.path.join(output_dir, "edge_cases.json"))

    print(f"✅ Generated {len(edge_cases)} edge case players")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic FPL test data")
    parser.add_argument(
        "--output",
        default="backend/tests/fixtures",
        help="Output directory (default: backend/tests/fixtures)",
    )
    parser.add_argument(
        "--teams", type=int, default=20, help="Number of teams (default: 20)"
    )
    parser.add_argument(
        "--gameweeks", type=int, default=12, help="Number of gameweeks (default: 12)"
    )
    parser.add_argument(
        "--no-history", action="store_true", help="Skip generating player history"
    )
    parser.add_argument(
        "--no-golden", action="store_true", help="Skip generating golden dataset"
    )
    parser.add_argument(
        "--edge-cases", action="store_true", help="Generate edge case fixtures"
    )

    args = parser.parse_args()

    # Generate main suite
    generate_full_test_suite(
        output_dir=args.output,
        n_teams=args.teams,
        num_gameweeks=args.gameweeks,
        include_history=not args.no_history,
        include_golden=not args.no_golden,
    )

    # Generate edge cases if requested
    if args.edge_cases:
        generate_edge_cases(args.output)

    print("\n✅ All test data generated successfully!")
    print("💡 To run tests: cd backend && pytest tests/ -v")
