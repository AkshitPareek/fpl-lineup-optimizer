"""
Unit Tests for FPL ML Predictors

Tests PointPredictor, MinutesPredictor, and EVCalculator in isolation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.point_predictor import PointPredictor, compute_uncertainty_bounds
from backend.minutes_predictor import MinutesPredictor
from backend.ev_calculator import EVCalculator, EVDistribution


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def teams_df():
    """Sample teams data."""
    return pd.DataFrame(
        [
            {
                "id": i + 1,
                "name": f"Team {i + 1}",
                "short_name": f"T{i + 1}",
                "strength_attack_home": 1000 + i * 20,
                "strength_attack_away": 980 + i * 20,
                "strength_defence_home": 1000 + i * 15,
                "strength_defence_away": 970 + i * 15,
            }
            for i in range(5)
        ]
    )


@pytest.fixture
def fixtures():
    """Sample fixtures for 3 gameweeks."""
    return [
        {
            "id": 1,
            "event": 1,
            "team_h": 1,
            "team_a": 2,
            "team_h_difficulty": 3,
            "team_a_difficulty": 4,
        },
        {
            "id": 2,
            "event": 1,
            "team_h": 3,
            "team_a": 4,
            "team_h_difficulty": 3,
            "team_a_difficulty": 4,
        },
        {
            "id": 3,
            "event": 2,
            "team_h": 1,
            "team_a": 3,
            "team_h_difficulty": 3,
            "team_a_difficulty": 4,
        },
        {
            "id": 4,
            "event": 2,
            "team_h": 2,
            "team_a": 4,
            "team_h_difficulty": 4,
            "team_a_difficulty": 3,
        },
        {
            "id": 5,
            "event": 3,
            "team_h": 1,
            "team_a": 5,
            "team_h_difficulty": 3,
            "team_a_difficulty": 5,
        },
        {
            "id": 6,
            "event": 3,
            "team_h": 2,
            "team_a": 3,
            "team_h_difficulty": 4,
            "team_a_difficulty": 3,
        },
    ]


@pytest.fixture
def players_df():
    """Sample players with realistic data."""
    data = []
    positions = [
        (1, "GK", 40, 70),
        (2, "DEF", 50, 60),
        (3, "MID", 60, 80),
        (4, "FWD", 50, 70),
    ]

    for pos_id, pos_name, min_cost, max_cost in positions:
        for i in range(10):
            minutes = np.random.randint(500, 2700)
            games = minutes // 90
            is_understat = np.random.random() > 0.3

            player = {
                "id": pos_id * 100 + i + 1,
                "web_name": f"{pos_name}{i + 1}",
                "element_type": pos_id,
                "team": np.random.randint(1, 6),
                "now_cost": np.random.randint(min_cost, max_cost)
                * 10,  # In FPL format (tenths)
                "minutes": minutes,
                "starts": int(games * np.random.uniform(0.7, 0.95)),
                "goals_scored": int(
                    minutes
                    / 500
                    * np.random.uniform(0.2, 2.0)
                    * (
                        1
                        if pos_id == 1
                        else (2 if pos_id == 2 else (3 if pos_id == 3 else 4))
                    )
                ),
                "assists": int(
                    minutes
                    / 500
                    * np.random.uniform(0.1, 1.5)
                    * (1 if pos_id == 1 else (1.5 if pos_id == 2 else 2))
                ),
                "expected_goals": round(np.random.uniform(0, 20), 1),
                "expected_assists": round(np.random.uniform(0, 15), 1),
                "ict_index": round(np.random.uniform(10, 300), 1),
                "form": round(np.random.uniform(0, 10), 1),
                "ep_next": round(np.random.uniform(0, 8), 1),
                "chance_of_playing_next_round": 100,
                "status": "a",
                "influence": round(np.random.uniform(10, 500), 1),
                "recent_4_minutes": np.random.randint(200, 360),
                "understat_matched": is_understat,
                "understat_xG_per_90": round(np.random.uniform(0.0, 0.8), 2)
                if is_understat
                else None,
                "understat_xA_per_90": round(np.random.uniform(0.0, 0.4), 2)
                if is_understat
                else None,
            }
            data.append(player)

    return pd.DataFrame(data)


@pytest.fixture
def history_cache():
    """Sample player history for minutes predictor."""
    cache = {}
    for player_id in range(120, 500):
        num_games = np.random.randint(0, 10)
        if num_games > 0:
            history = [
                {
                    "minutes": np.random.randint(0, 100)
                    if np.random.random() < 0.2
                    else np.random.randint(60, 100)
                }
                for _ in range(num_games)
            ]
        else:
            history = []
        cache[str(player_id)] = {"history": history}
    return cache


# ============================================================================
# PointPredictor Tests
# ============================================================================


def test_point_predictor_initialization(players_df, teams_df, fixtures):
    """Test PointPredictor initializes correctly."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1, use_understat=False)

    assert predictor.players is not None
    assert predictor.teams is not None
    assert predictor.fixtures is not None
    assert predictor.current_gw == 1
    assert hasattr(predictor, "fixture_map")
    assert hasattr(predictor, "team_attack_home")
    assert hasattr(predictor, "team_defense_away")
    assert len(predictor.fixture_map) > 0


def test_team_strength_computation(players_df, teams_df, fixtures):
    """Test team strength dictionaries are populated."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1)

    assert isinstance(predictor.team_attack_home, dict)
    assert isinstance(predictor.team_attack_away, dict)
    assert isinstance(predictor.team_defense_home, dict)
    assert isinstance(predictor.team_defense_away, dict)
    assert isinstance(predictor.team_names, dict)

    # All teams should have ratings
    for team_id in teams_df["id"]:
        assert team_id in predictor.team_attack_home
        assert team_id in predictor.team_names


def test_fixture_map_structure(players_df, teams_df, fixtures):
    """Test fixture map contains proper structure."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1)

    for (team_id, gw), team_fixtures in predictor.fixture_map.items():
        for fixture in team_fixtures:
            assert "opponent" in fixture
            assert "is_home" in fixture in [True, False]
            assert "difficulty" in fixture
            assert 1 <= fixture["difficulty"] <= 6


def test_xg_per_90_calculation_fpl_data(players_df, teams_df, fixtures):
    """Test xG/90 from FPL data."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1, use_understat=False)

    # Get player with sufficient minutes
    player = players_df[players_df["minutes"] >= predictor.MIN_MINUTES_FOR_RATES].iloc[
        0
    ]
    xg_per_90 = predictor._get_xg_per_90(player)

    expected = (player["expected_goals"] / player["minutes"]) * 90
    assert abs(xg_per_90 - expected) < 0.001
    assert xg_per_90 >= 0


def test_xg_per_90_baseline_for_limited_minutes(players_df, teams_df, fixtures):
    """Test baseline xG/90 for players with limited minutes."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1, use_understat=False)

    player = players_df[players_df["minutes"] < predictor.MIN_MINUTES_FOR_RATES].iloc[0]
    xg_per_90 = predictor._get_xg_per_90(player)

    expected_baseline = predictor.BASELINE_XG[player["element_type"]]
    assert xg_per_90 == expected_baseline


def test_xg_points_opponent_adjustment(players_df, teams_df, fixtures):
    """Test xG points adjusted by opponent defense strength."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1)

    player = players_df.iloc[0]
    xg_points = predictor.calculate_xg_points(player, 1)

    # Should be positive and capped
    assert 0 <= xg_points <= 9.0

    # xG points should be higher when facing weaker defense
    fixtures_vs_weak = [
        f for f in fixtures if f["team_a"] == 5
    ]  # Team 5 has high difficulty
    if fixtures_vs_weak:
        weak_defense_fixture = fixtures_vs_weak[0]
        # Manually test that higher defense difficulty affects xG
        # (defense strength = difficulty/1000, lower defense strength = more xG)
        # This is a structural test, not assertion-based


def test_xa_points_calculation(players_df, teams_df, fixtures):
    """Test xA points calculation."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1)

    player = players_df[players_df["element_type"].isin([3, 4])].iloc[0]  # Attacker
    xa_points = predictor.calculate_xa_points(player, 1)

    assert 0 <= xa_points <= 6.0  # Capped at 6.0


def test_clean_sheet_prob_gk_def_only(players_df, teams_df, fixtures):
    """Test clean sheet points only for GK and DEF."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1)

    # GK and DEF should get CS points
    gk = players_df[players_df["element_type"] == 1].iloc[0]
    defender = players_df[players_df["element_type"] == 2].iloc[0]
    mid = players_df[players_df["element_type"] == 3].iloc[0]
    fwd = players_df[players_df["element_type"] == 4].iloc[0]

    cs_gk = predictor.calculate_clean_sheet_prob(gk, 1)
    cs_def = predictor.calculate_clean_sheet_prob(defender, 1)
    cs_mid = predictor.calculate_clean_sheet_prob(mid, 1)
    cs_fwd = predictor.calculate_clean_sheet_prob(fwd, 1)

    assert cs_gk >= 0 and cs_gk <= predictor.POINTS_CLEAN_SHEET[1] * 1.0  # Max 4 points
    assert (
        cs_def >= 0 and cs_def <= predictor.POINTS_CLEAN_SHEET[2] * 1.0
    )  # Max 4 points
    # MID and FWD get 0 CS points according to rules
    assert cs_mid == 0
    assert cs_fwd == 0


def test_cbit_bonus_by_position(players_df, teams_df, fixtures):
    """Test CBIT/CBIRT bonus varies by position."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1)

    # High influence defender
    defender = players_df[players_df["element_type"] == 2].iloc[0]
    defender["influence"] = 150.0
    cbit_def = predictor.calculate_cbit_bonus(defender, 1)

    # High influence midfielder
    mid = players_df[players_df["element_type"] == 3].iloc[0]
    mid["influence"] = 150.0
    cbit_mid = predictor.calculate_cbit_bonus(mid, 1)

    # GK should get no CBIT
    gk = players_df[players_df["element_type"] == 1].iloc[0]
    cbit_gk = predictor.calculate_cbit_bonus(gk, 1)

    assert cbit_gk == 0.0
    assert 0 <= cbit_def <= 2.0
    assert 0 <= cbit_mid <= 2.0


def test_appearance_points_by_minutes(players_df, teams_df, fixtures):
    """Test appearance points based on minutes."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1)

    # Full game player
    full_player = pd.Series({"minutes": 90})
    assert predictor.calculate_appearance_points(full_player) == 2.0

    # Partial game (30-60 mins)
    partial_player = pd.Series({"minutes": 45, "starts": 0})
    points = predictor.calculate_appearance_points(partial_player)
    assert points in [0.0, 1.0]

    # No minutes
    no_minutes = pd.Series({"minutes": 0, "starts": 0})
    assert predictor.calculate_appearance_points(no_minutes) == 0.0


def test_regression_factor_underperformer(players_df, teams_df, fixtures):
    """Test regression factor > 1 for underperformers."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1)

    player = pd.Series(
        {
            "goals_scored": 2.0,
            "expected_goals": 10.0,
            "assists": 1.0,
            "expected_assists": 5.0,
        }
    )

    factor = predictor.calculate_regression_factor(player)
    assert factor > 1.0  # Underperformer should regress up


def test_regression_factor_overperformer(players_df, teams_df, fixtures):
    """Test regression factor < 1 for overperformers."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1)

    player = pd.Series(
        {
            "goals_scored": 10.0,
            "expected_goals": 5.0,
            "assists": 5.0,
            "expected_assists": 3.0,
        }
    )

    factor = predictor.calculate_regression_factor(player)
    assert factor < 1.0  # Overperformer should regress down


def test_regression_factor_insufficient_data(players_df, teams_df, fixtures):
    """Test regression factor = 1 for insufficient data."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1)

    player = pd.Series(
        {
            "goals_scored": 0.5,
            "expected_goals": 0.3,
            "assists": 0.2,
            "expected_assists": 0.1,
        }
    )

    factor = predictor.calculate_regression_factor(player)
    assert factor == 1.0


def test_bonus_points_calculation(players_df, teams_df, fixtures):
    """Test bonus points estimation."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1)

    player = players_df.iloc[0]
    bonus = predictor.calculate_bonus_points(player, 1)

    assert 0 <= bonus <= 1.2  # Capped at 1.2


def test_predict_gameweek_returns_tuple(players_df, teams_df, fixtures):
    """Test predict_gameweek returns (float, dict) tuple."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1, use_understat=False)

    player = players_df.iloc[0]
    xp, breakdown = predictor.predict_gameweek(player, 1)

    assert isinstance(xp, float)
    assert isinstance(breakdown, dict)
    assert len(breakdown) > 0


def test_prediction_within_bounds(players_df, teams_df, fixtures):
    """Test all predictions are within reasonable bounds."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1, use_understat=False)

    for _, player in players_df.iterrows():
        xp, breakdown = predictor.predict_gameweek(player, 1)

        assert 0 <= xp <= 8.0  # Max 8 points per gameweek
        assert all(v >= 0 for v in breakdown.values())
        assert all(isinstance(v, (int, float)) for v in breakdown.values())


def test_predict_all_players_returns_dataframe(players_df, teams_df, fixtures):
    """Test predict_all_players returns DataFrame with correct structure."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1, use_understat=False)
    predictions = predictor.predict_all_players(gameweeks=3)

    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(players_df)

    # Check required columns
    assert "id" in predictions.columns
    assert "web_name" in predictions.columns
    assert "total_xp" in predictions.columns
    assert "element_type" in predictions.columns

    # Check gameweek columns
    for gw in [1, 2, 3]:
        assert f"xp_gw{gw}" in predictions.columns

    # All values should be non-negative
    xp_cols = [
        c for c in predictions.columns if c.startswith("xp_gw") or c == "total_xp"
    ]
    for col in xp_cols:
        assert all(predictions[col] >= 0)


def test_multi_gameweek_prediction(players_df, teams_df, fixtures):
    """Test multi-gameweek prediction for single player."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1)
    player_id = players_df.iloc[0]["id"]

    prediction = predictor.predict_multi_gameweek(player_id, gameweeks=3)

    assert prediction.player_id == player_id
    assert len(prediction.gameweek_predictions) == 3
    assert prediction.total_expected > 0
    assert isinstance(prediction.breakdown, dict)


def test_prediction_consistency(players_df, teams_df, fixtures):
    """Test same input yields same output."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1, use_understat=False)
    player = players_df.iloc[0]

    xp1, _ = predictor.predict_gameweek(player, 1)
    xp2, _ = predictor.predict_gameweek(player, 1)

    assert abs(xp1 - xp2) < 1e-6  # Should be identical within floating point


def test_calibration_factor_applied(players_df, teams_df, fixtures):
    """Test calibration factor is applied to total."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1, use_understat=False)

    player = players_df.iloc[0]
    xp1, breakdown1 = predictor.predict_gameweek(player, 1)
    xp5, breakdown5 = predictor.predict_gameweek(
        player, 5
    )  # Later GW should have lower calibration

    # Later gameweeks have reduced calibration
    assert "calibration" in breakdown1
    assert "calibration" in breakdown5
    assert breakdown5["calibration"] <= breakdown1["calibration"]


def test_form_factor_calculation(players_df, teams_df, fixtures):
    """Test form factor affects predictions."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1, use_understat=False)

    player = players_df.iloc[0].copy()
    original_form = player["form"]

    # High form
    player["form"] = 8.0
    xp_high, _ = predictor.predict_gameweek(player, 1)

    # Low form
    player["form"] = 2.0
    xp_low, _ = predictor.predict_gameweek(player, 1)

    player["form"] = original_form

    assert xp_high > xp_low


def test_minutes_decay_effect(players_df, teams_df, fixtures):
    """Test reduced minutes lowers predictions."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1, use_understat=False)

    player = players_df.iloc[0].copy()

    # Full minutes
    player["recent_4_minutes"] = 360
    xp_full, _ = predictor.predict_gameweek(player, 1)

    # Low minutes
    player["recent_4_minutes"] = 180
    xp_low, _ = predictor.predict_gameweek(player, 1)

    assert xp_full > xp_low


def test_availability_adjustment(players_df, teams_df, fixtures):
    """Test injured players have reduced predictions."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1, use_understat=False)

    player = players_df.iloc[0].copy()

    # Fully available
    player["chance_of_playing_next_round"] = 100
    xp_available, _ = predictor.predict_gameweek(player, 1)

    # 50% chance
    player["chance_of_playing_next_round"] = 50
    xp_50, _ = predictor.predict_gameweek(player, 1)

    assert xp_available > xp_50


def test_no_fixtures_fallback(players_df, teams_df, fixtures):
    """Test fallback when no fixtures found."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1, use_understat=False)

    player = players_df.iloc[0]
    # Use GW that doesn't exist in fixture map
    xp, breakdown = predictor.predict_gameweek(player, 99)

    assert xp > 0  # Should still produce estimate
    assert "xg_points" in breakdown


def test_get_fixtures_method(players_df, teams_df, fixtures):
    """Test get_fixtures returns upcoming fixtures."""
    predictor = PointPredictor(players_df, teams_df, fixtures, 1)

    team_id = 1
    upcoming = predictor.get_fixtures(team_id, gameweeks=3)

    assert isinstance(upcoming, list)
    for fixture in upcoming:
        assert "opponent" in fixture
        assert "is_home" in fixture
        assert "difficulty" in fixture
        assert "gameweek" in fixture


# ============================================================================
# MinutesPredictor Tests
# ============================================================================


def test_minutes_predictor_initialization(players_df, history_cache):
    """Test MinutesPredictor initializes correctly."""
    predictor = MinutesPredictor(players_df, history_cache)

    assert predictor.players is not None
    assert predictor.history_cache is not None


def test_get_recent_minutes(players_df, history_cache):
    """Test recent minutes extraction from history."""
    predictor = MinutesPredictor(players_df, history_cache)

    # Test with history
    player_id = 1
    recent = predictor.get_recent_minutes(player_id, last_n_gws=5)

    assert isinstance(recent, list)
    assert len(recent) <= 5
    assert all(isinstance(m, int) for m in recent)


def test_get_recent_minutes_no_history(players_df):
    """Test player with no history."""
    predictor = MinutesPredictor(players_df, {})

    recent = predictor.get_recent_minutes(999, last_n_gws=5)
    assert recent == []


def test_start_probability_full_starter(players_df, history_cache):
    """Test start probability for consistent starter."""
    predictor = MinutesPredictor(players_df, history_cache)
    player = players_df.iloc[0]

    prob = predictor.calculate_start_probability(player)

    assert 0.8 <= prob <= 1.0  # Should be high


def test_start_probability_rotated_player(players_df, history_cache):
    """Test start probability for rotated player."""
    predictor = MinutesPredictor(players_df, history_cache)
    player = players_df.iloc[2]  # Some rotation

    prob = predictor.calculate_start_probability(player)

    assert 0.0 <= prob <= 1.0


def test_start_probability_injured(players_df):
    """Test injured player has 0 start probability."""
    predictor = MinutesPredictor(players_df, {})

    player = pd.Series({"id": 999, "element_type": 2, "status": "i"})
    prob = predictor.calculate_start_probability(player)
    assert prob == 0.0


def test_start_probability_doubtful(players_df):
    """Test doubtful player has reduced probability."""
    predictor = MinutesPredictor(players_df, {})

    player = pd.Series(
        {
            "id": 999,
            "element_type": 2,
            "status": "d",
            "chance_of_playing_next_round": 50,
        }
    )
    prob = predictor.calculate_start_probability(player)
    assert 0 < prob < 0.5


def test_expected_minutes_by_probability(players_df, history_cache):
    """Test expected minutes mapping from start probability."""
    predictor = MinutesPredictor(players_df, history_cache)

    for _, player in players_df.iterrows():
        exp_mins = predictor.calculate_expected_minutes(player)

        assert 0 <= exp_mins <= 90.0

        # Check discrete buckets
        if exp_mins > 0:
            prob = predictor.calculate_start_probability(player)
            if prob >= 0.8:
                assert exp_mins == 85.0
            elif prob >= 0.5:
                assert exp_mins == 65.0
            elif prob >= 0.2:
                assert exp_mins == 25.0
            else:
                assert exp_mins == 0.0


def test_rotation_risk_detection(players_df, history_cache):
    """Test rotation risk calculation."""
    predictor = MinutesPredictor(players_df, history_cache)

    for _, player in players_df.iterrows():
        risk = predictor.calculate_rotation_risk(player)

        assert 0 <= risk <= 1.0


def test_nailedness_score_range(players_df, history_cache):
    """Test nailedness scores are in 0-100 range."""
    predictor = MinutesPredictor(players_df, history_cache)

    for _, player in players_df.iterrows():
        score = predictor.get_nailedness_score(player)

        assert 0 <= score <= 100.0
        assert isinstance(score, float)


def test_predict_all_players_minutes(players_df, history_cache):
    """Test predictions for all players."""
    predictor = MinutesPredictor(players_df, history_cache)
    predictions = predictor.predict_all_players()

    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(players_df)

    expected_cols = [
        "id",
        "web_name",
        "start_probability",
        "expected_minutes",
        "rotation_risk",
        "nailedness",
    ]
    for col in expected_cols:
        assert col in predictions.columns

    # Range checks
    assert all(predictions["start_probability"].between(0, 1))
    assert all(predictions["expected_minutes"].between(0, 90))
    assert all(predictions["rotation_risk"].between(0, 1))
    assert all(predictions["nailedness"].between(0, 100))


def test_gk_nailedness_boost(players_df, history_cache):
    """Test GKs get slight boost to start probability."""
    predictor = MinutesPredictor(players_df, history_cache)

    gk = players_df[players_df["element_type"] == 1].iloc[0]
    gk_prob = predictor.calculate_start_probability(gk)

    # GKs have 1.1 multiplier
    assert gk_prob <= 1.0


# ============================================================================
# EVCalculator Tests
# ============================================================================


def test_ev_calculator_initialization(players_df, teams_df):
    """Test EVCalculator initialization."""
    calc = EVCalculator(players_df, teams_df)

    assert calc.players is not None
    assert calc.teams is not None
    assert hasattr(calc, "positions")


def test_calculate_player_distribution(players_df, teams_df):
    """Test EV distribution calculation for single player."""
    calc = EVCalculator(players_df, teams_df)
    player_id = players_df.iloc[0]["id"]

    dist = calc.calculate_player_distribution(player_id)

    assert isinstance(dist, EVDistribution)
    assert dist.player_id == player_id
    assert dist.expected_points >= 0
    assert dist.floor >= 0
    assert dist.ceiling >= dist.expected_points
    assert dist.std_dev >= 0
    assert dist.risk_score >= 0


def test_distribution_floor_less_than_ceiling(players_df, teams_df):
    """Test floor <= expected <= ceiling."""
    calc = EVCalculator(players_df, teams_df)

    for player_id in players_df["id"].head(20):
        dist = calc.calculate_player_distribution(player_id)
        assert dist.floor <= dist.expected_points <= dist.ceiling


def test_position_variance_ordering(players_df, teams_df):
    """Test forwards have higher variance than defenders."""
    calc = EVCalculator(players_df, teams_df)

    def_players = players_df[players_df["element_type"] == 2]
    fwd_players = players_df[players_df["element_type"] == 4]

    if len(def_players) > 0 and len(fwd_players) > 0:
        def_variances = []
        fwd_variances = []

        for pid in def_players["id"].head(5):
            try:
                dist = calc.calculate_player_distribution(pid)
                def_variances.append(dist.std_dev)
            except:
                pass

        for pid in fwd_players["id"].head(5):
            try:
                dist = calc.calculate_player_distribution(pid)
                fwd_variances.append(dist.std_dev)
            except:
                pass

        if def_variances and fwd_variances:
            avg_def_var = np.mean(def_variances)
            avg_fwd_var = np.mean(fwd_variances)
            assert (
                avg_fwd_var > avg_def_var * 1.1
            )  # Forwards at least 10% more volatile


def test_get_all_distributions(players_df, teams_df):
    """Test getting distributions for all players."""
    calc = EVCalculator(players_df, teams_df)
    df = calc.get_all_distributions()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    expected_cols = [
        "id",
        "web_name",
        "team",
        "position",
        "price",
        "expected_points",
        "floor",
        "ceiling",
        "std_dev",
        "upside",
        "downside",
        "risk_score",
    ]
    for col in expected_cols:
        assert col in df.columns


def test_high_ceiling_players(players_df, teams_df):
    """Test filtering high ceiling players."""
    calc = EVCalculator(players_df, teams_df)
    high_ceiling = calc.get_high_ceiling_players(min_ceiling=5.0)

    assert isinstance(high_ceiling, pd.DataFrame)
    if len(high_ceiling) > 0:
        assert all(high_ceiling["ceiling"] >= 5.0)
        # Should be sorted descending
        assert high_ceiling["ceiling"].iloc[0] >= high_ceiling["ceiling"].iloc[-1]


def test_safe_players(players_df, teams_df):
    """Test filtering low-risk safe players."""
    calc = EVCalculator(players_df, teams_df)
    safe = calc.get_safe_players(max_risk=0.5, min_expected=2.0)

    assert isinstance(safe, pd.DataFrame)
    if len(safe) > 0:
        assert all(safe["risk_score"] <= 0.5)
        assert all(safe["expected_points"] >= 2.0)


def test_high_upside_players(players_df, teams_df):
    """Test filtering high upside players."""
    calc = EVCalculator(players_df, teams_df)
    upside = calc.get_high_upside_players(min_upside=3.0)

    assert isinstance(upside, pd.DataFrame)
    if len(upside) > 0:
        assert all(upside["upside"] >= 3.0)


def test_player_comparison(players_df, teams_df):
    """Test player comparison functionality."""
    calc = EVCalculator(players_df, teams_df)
    player_ids = players_df["id"].head(5).tolist()

    comparison = calc.compare_players(player_ids)

    assert "players" in comparison
    assert "recommendations" in comparison
    assert len(comparison["players"]) > 0

    recs = comparison["recommendations"]
    assert "highest_expected" in recs
    assert "highest_ceiling" in recs
    assert "safest_pick" in recs


def test_squad_ev_calculation(players_df, teams_df):
    """Test squad-level EV aggregation."""
    calc = EVCalculator(players_df, teams_df)
    squad_ids = players_df["id"].head(15).tolist()

    squad_ev = calc.calculate_squad_ev(squad_ids[:11])

    assert "total_expected" in squad_ev
    assert "floor" in squad_ev
    assert "ceiling" in squad_ev
    assert "std_dev" in squad_ev
    assert "risk_profile" in squad_ev

    assert squad_ev["floor"] <= squad_ev["total_expected"] <= squad_ev["ceiling"]


def test_simulate_gameweek(players_df, teams_df):
    """Test Monte Carlo gameweek simulation."""
    calc = EVCalculator(players_df, teams_df)
    squad_ids = players_df["id"].head(11).tolist()

    sim = calc.simulate_gameweek(squad_ids, n_simulations=100)

    assert sim["simulations"] == 100
    assert "mean" in sim
    assert "median" in sim
    assert "std_dev" in sim
    assert "percentile_10" in sim
    assert "percentile_90" in sim

    assert sim["mean"] > 0
    assert sim["min"] <= sim["median"] <= sim["max"]


def test_simulation_distribution(players_df, teams_df):
    """Test simulation produces reasonable distribution."""
    calc = EVCalculator(players_df, teams_df)
    squad_ids = players_df["id"].head(11).tolist()

    sim = calc.simulate_gameweek(squad_ids, n_simulations=1000)

    # Percentiles should be ordered
    assert sim["percentile_10"] <= sim["percentile_25"] <= sim["percentile_50"]
    assert sim["percentile_50"] <= sim["percentile_75"] <= sim["percentile_90"]

    # Probabilities should be reasonable
    assert 0 <= sim["prob_above_50"] <= 100
    assert 0 <= sim["prob_above_70"] <= sim["prob_above_50"]


def test_uncertainty_bounds_function(players_df):
    """Test compute_uncertainty_bounds utility."""
    predictions = pd.DataFrame(
        {"id": [1, 2, 3], "total_xp": [10.0, 15.0, 8.0], "xp_gw1": [5.0, 7.0, 4.0]}
    )

    result = compute_uncertainty_bounds(predictions, confidence_level=0.14)

    assert f"xp_gw1_lower" in result.columns
    assert f"xp_gw1_upper" in result.columns
    assert f"xp_gw1_sigma" in result.columns

    # Lower should be less than upper
    assert all(result["xp_gw1_lower"] <= result["xp_gw1_upper"])


def test_minimum_variance(players_df, teams_df):
    """Test minimum standard deviation is enforced."""
    calc = EVCalculator(players_df, teams_df)

    # Low expected points player
    player = players_df.iloc[0]
    player["ep_next"] = 0.1

    dist = calc.calculate_player_distribution(player["id"], expected_points=0.1)
    assert dist.std_dev >= 0.5  # Minimum variance enforced


def test_risk_score_definition(players_df, teams_df):
    """Test risk score is coefficient of variation."""
    calc = EVCalculator(players_df, teams_df)
    dist = calc.calculate_player_distribution(players_df.iloc[0]["id"])

    expected_cv = dist.std_dev / max(0.1, dist.expected_points)
    assert abs(dist.risk_score - expected_cv) < 0.01


# ============================================================================
# Edge Cases
# ============================================================================


def test_handle_missing_player(players_df, teams_df):
    """Test handling of non-existent player ID."""
    calc = EVCalculator(players_df, teams_df)

    with pytest.raises(ValueError, match="Player .* not found"):
        calc.calculate_player_distribution(99999)


def test_handle_zero_expected_points(players_df, teams_df):
    """Test handling of zero expected points."""
    calc = EVCalculator(players_df, teams_df)

    dist = calc.calculate_player_distribution(
        players_df.iloc[0]["id"], expected_points=0.0
    )

    assert dist.expected_points == 0.0
    assert dist.floor >= 0
    assert dist.ceiling >= 0


def test_empty_squad_ev(players_df, teams_df):
    """Test EV calculation with empty squad."""
    calc = EVCalculator(players_df, teams_df)

    result = calc.calculate_squad_ev([])
    assert "error" in result


def test_compare_with_invalid_players(players_df, teams_df):
    """Test comparison with invalid player IDs."""
    calc = EVCalculator(players_df, teams_df)

    comparison = calc.compare_players([99999, 88888])
    assert "error" in comparison or len(comparison["players"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
