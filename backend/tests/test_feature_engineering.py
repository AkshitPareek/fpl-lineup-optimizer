"""
Feature Engineering Tests for FPL ML System

Tests for feature calculation, data preprocessing, and enrichment.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

from backend.point_predictor import PointPredictor
from backend.minutes_predictor import MinutesPredictor


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_teams_df():
    """Create sample teams DataFrame."""
    return pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Arsenal",
                "short_name": "ARS",
                "strength_attack_home": 1050,
                "strength_attack_away": 1020,
                "strength_defence_home": 1030,
                "strength_defence_away": 1010,
            },
            {
                "id": 2,
                "name": "Aston Villa",
                "short_name": "AVL",
                "strength_attack_home": 1000,
                "strength_attack_away": 980,
                "strength_defence_home": 990,
                "strength_defence_away": 970,
            },
            {
                "id": 3,
                "name": "Chelsea",
                "short_name": "CHE",
                "strength_attack_home": 1020,
                "strength_attack_away": 1000,
                "strength_defence_home": 1010,
                "strength_defence_away": 990,
            },
        ]
    )


@pytest.fixture
def sample_fixtures():
    """Create sample fixtures for 5 gameweeks."""
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
            "team_a": 1,
            "team_h_difficulty": 4,
            "team_a_difficulty": 3,
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
            "event": 3,
            "team_h": 2,
            "team_a": 1,
            "team_h_difficulty": 4,
            "team_a_difficulty": 3,
        },
        {
            "id": 5,
            "event": 4,
            "team_h": 1,
            "team_a": 2,
            "team_h_difficulty": 3,
            "team_a_difficulty": 4,
        },
        {
            "id": 6,
            "event": 5,
            "team_h": 3,
            "team_a": 2,
            "team_h_difficulty": 4,
            "team_a_difficulty": 3,
        },
    ]


@pytest.fixture
def sample_players_df():
    """Create sample players DataFrame."""
    return pd.DataFrame(
        [
            # GK
            {
                "id": 1,
                "web_name": "GK1",
                "element_type": 1,
                "team": 1,
                "now_cost": 55,
                "minutes": 1800,
                "starts": 20,
                "goals_scored": 0,
                "assists": 0,
                "expected_goals": 0,
                "expected_assists": 0,
                "ict_index": 10.0,
                "form": 4.5,
                "ep_next": 4.2,
                "chance_of_playing_next_round": 100,
                "status": "a",
                "influence": 15.0,
                "recent_4_minutes": 360,
            },
            # DEF
            {
                "id": 2,
                "web_name": "DEF1",
                "element_type": 2,
                "team": 1,
                "now_cost": 60,
                "minutes": 2000,
                "starts": 22,
                "goals_scored": 3,
                "assists": 4,
                "expected_goals": 2.5,
                "expected_assists": 3.2,
                "ict_index": 45.0,
                "form": 6.1,
                "ep_next": 5.8,
                "chance_of_playing_next_round": 100,
                "status": "a",
                "influence": 120.0,
                "recent_4_minutes": 360,
            },
            # MID
            {
                "id": 3,
                "web_name": "MID1",
                "element_type": 3,
                "team": 2,
                "now_cost": 95,
                "minutes": 2200,
                "starts": 24,
                "goals_scored": 12,
                "assists": 8,
                "expected_goals": 10.5,
                "expected_assists": 7.8,
                "ict_index": 150.0,
                "form": 8.2,
                "ep_next": 7.9,
                "chance_of_playing_next_round": 100,
                "status": "a",
                "influence": 280.0,
                "recent_4_minutes": 360,
            },
            # FWD
            {
                "id": 4,
                "web_name": "FWD1",
                "element_type": 4,
                "team": 3,
                "now_cost": 120,
                "minutes": 1800,
                "starts": 20,
                "goals_scored": 15,
                "assists": 5,
                "expected_goals": 14.2,
                "expected_assists": 4.5,
                "ict_index": 200.0,
                "form": 9.1,
                "ep_next": 8.5,
                "chance_of_playing_next_round": 100,
                "status": "a",
                "influence": 320.0,
                "recent_4_minutes": 360,
            },
            # Player with limited minutes (baseline)
            {
                "id": 5,
                "web_name": "ROOKIE",
                "element_type": 4,
                "team": 1,
                "now_cost": 50,
                "minutes": 90,
                "starts": 1,
                "goals_scored": 0,
                "assists": 0,
                "expected_goals": 0.1,
                "expected_assists": 0.05,
                "ict_index": 5.0,
                "form": 3.0,
                "ep_next": 1.5,
                "chance_of_playing_next_round": 100,
                "status": "a",
                "influence": 8.0,
                "recent_4_minutes": 45,
            },
        ]
    )


@pytest.fixture
def sample_history_cache():
    """Create sample player history for minutes predictor."""
    return {
        "1": {  # GK with consistent starts
            "history": [
                {"minutes": 90},
                {"minutes": 90},
                {"minutes": 90},
                {"minutes": 90},
                {"minutes": 90},
            ]
        },
        "2": {  # DEF with some rotation
            "history": [
                {"minutes": 90},
                {"minutes": 0},
                {"minutes": 85},
                {"minutes": 90},
                {"minutes": 75},
            ]
        },
        "3": {  # MID heavily rotated
            "history": [
                {"minutes": 90},
                {"minutes": 45},
                {"minutes": 90},
                {"minutes": 30},
                {"minutes": 90},
            ]
        },
        "4": {  # FWD injured
            "history": [
                {"minutes": 90},
                {"minutes": 90},
                {"minutes": 0},
                {"minutes": 0},
                {"minutes": 0},
            ]
        },
        "5": {  # Rookie no history
            "history": []
        },
    }


@pytest.fixture
def point_predictor(sample_players_df, sample_teams_df, sample_fixtures):
    """Create PointPredictor instance."""
    return PointPredictor(
        players_df=sample_players_df,
        teams_df=sample_teams_df,
        fixtures=sample_fixtures,
        current_gameweek=1,
        use_understat=False,  # Disable for tests
    )


@pytest.fixture
def minutes_predictor(sample_players_df, sample_history_cache):
    """Create MinutesPredictor instance."""
    return MinutesPredictor(
        players_df=sample_players_df, history_cache=sample_history_cache
    )


# ============================================================================
# Team Strength Computation Tests
# ============================================================================


def test_team_strength_computation(point_predictor, sample_teams_df):
    """Test team strength dictionaries are correctly built."""
    assert hasattr(point_predictor, "team_attack_home")
    assert hasattr(point_predictor, "team_attack_away")
    assert hasattr(point_predictor, "team_defense_home")
    assert hasattr(point_predictor, "team_defense_away")
    assert hasattr(point_predictor, "team_names")

    # Verify values match input
    assert point_predictor.team_attack_home[1] == 1050
    assert point_predictor.team_defense_away[2] == 970
    assert point_predictor.team_names[3] == "Chelsea"


def test_fixture_map_building(point_predictor, sample_fixtures):
    """Test fixture map is correctly built."""
    assert hasattr(point_predictor, "fixture_map")

    # Check Arsenal (team 1) fixtures in GW1
    gw1_home = point_predictor.fixture_map.get((1, 1), [])
    gw1_away = point_predictor.fixture_map.get((1, 1), [])

    # Should have both home and away fixtures
    assert len(gw1_home) + len(gw1_away) >= 1

    # Verify opponent and difficulty
    for fixture_list in [gw1_home, gw1_away]:
        for fix in fixture_list:
            assert "opponent" in fix
            assert "is_home" in fix
            assert "difficulty" in fix
            assert 2 <= fix["difficulty"] <= 5


# ============================================================================
# xG/xA Feature Tests
# ============================================================================


def test_xg_per_90_with_minutes(point_predictor):
    """Test xG/90 calculation for player with sufficient minutes."""
    player = point_predictor.players[point_predictor.players["minutes"] >= 270].iloc[0]
    xg_per_90 = point_predictor._get_xg_per_90(player)

    assert xg_per_90 >= 0
    expected = (player["expected_goals"] / player["minutes"]) * 90
    assert abs(xg_per_90 - expected) < 0.001


def test_xg_per_90_fallback_for_limited_minutes(point_predictor):
    """Test baseline xG used for players with limited minutes."""
    player = point_predictor.players[point_predictor["minutes"] < 270].iloc[0]
    xg_per_90 = point_predictor._get_xg_per_90(player)

    expected_baseline = point_predictor.BASELINE_XG[player["element_type"]]
    assert xg_per_90 == expected_baseline


def test_xa_per_90_calculation(point_predictor):
    """Test xA/90 calculation."""
    player = point_predictor.players[point_predictor["minutes"] >= 270].iloc[0]
    xa_per_90 = point_predictor._get_xa_per_90(player)

    assert xa_per_90 >= 0
    expected = (player["expected_assists"] / player["minutes"]) * 90
    assert abs(xa_per_90 - expected) < 0.001


def test_position_baselines(point_predictor):
    """Test position-based baseline values are reasonable."""
    assert point_predictor.BASELINE_XG[1] == 0.0  # GK
    assert point_predictor.BASELINE_XG[2] == 0.03  # DEF
    assert point_predictor.BASELINE_XG[3] == 0.08  # MID
    assert point_predictor.BASELINE_XG[4] == 0.35  # FWD

    assert point_predictor.BASELINE_XA[1] == 0.0  # GK
    assert point_predictor.BASELINE_XA[2] == 0.04  # DEF
    assert point_predictor.BASELINE_XA[3] == 0.08  # MID
    assert point_predictor.BASELINE_XA[4] == 0.12  # FWD


# ============================================================================
# Points Calculation Tests
# ============================================================================


def test_calculate_xg_points(point_predictor):
    """Test xG points calculation."""
    player = point_predictor.players.iloc[3]  # FWD
    xg_points = point_predictor.calculate_xg_points(player, 1)

    assert 0 <= xg_points <= 9.0  # Capped at 9.0
    assert isinstance(xg_points, float)


def test_calculate_xa_points(point_predictor):
    """Test xA points calculation."""
    player = point_predictor.players.iloc[2]  # MID
    xa_points = point_predictor.calculate_xa_points(player, 1)

    assert 0 <= xa_points <= 6.0  # Capped at 6.0
    assert isinstance(xa_points, float)


def test_calculate_clean_sheet_prob(point_predictor):
    """Test clean sheet probability calculation."""
    player = point_predictor.players.iloc[1]  # DEF
    cs_points = point_predictor.calculate_clean_sheet_prob(player, 1)

    assert cs_points >= 0
    # Max CS points for defender is 4 * probability (max 1.0) = 4
    assert cs_points <= 4.0


def test_calculate_cbit_bonus(point_predictor):
    """Test CBIT/CBIRT bonus calculation."""
    # Defender with high influence
    defender = point_predictor.players[point_predictor["element_type"] == 2].iloc[0]
    cbit = point_predictor.calculate_cbit_bonus(defender, 1)
    assert 0 <= cbit <= 2.0  # Max bonus is 2 points

    # Forward with high influence
    forward = point_predictor.players[point_predictor["element_type"] == 4].iloc[0]
    cbit_forward = point_predictor.calculate_cbit_bonus(forward, 1)
    assert 0 <= cbit_forward <= 2.0

    # GK should get 0
    gk = point_predictor.players[point_predictor["element_type"] == 1].iloc[0]
    cbit_gk = point_predictor.calculate_cbit_bonus(gk, 1)
    assert cbit_gk == 0.0


def test_calculate_appearance_points(point_predictor):
    """Test appearance points estimation."""
    # Player with high minutes
    regular = point_predictor.players.iloc[3]
    points = point_predictor.calculate_appearance_points(regular)
    assert points == 2.0  # 60+ mins

    # Player with some minutes but <60
    sub = point_predictor.players.iloc[0]  # GK with 1800 mins
    points_sub = point_predictor.calculate_appearance_points(sub)
    # GK should also get 2.0 based on avg mins logic
    assert points_sub in [0.0, 1.0, 2.0]


# ============================================================================
# Prediction Tests
# ============================================================================


def test_predict_gameweek_returns_valid(point_predictor):
    """Test gameweek prediction returns valid tuple."""
    player = point_predictor.players.iloc[0]
    xp, breakdown = point_predictor.predict_gameweek(player, 1)

    assert isinstance(xp, float)
    assert 0 <= xp <= 8.0  # Reasonable cap
    assert isinstance(breakdown, dict)

    required_keys = [
        "xg_points",
        "xa_points",
        "cs_points",
        "cbit_bonus",
        "appearance",
        "bonus",
        "regression_factor",
        "calibration",
        "form_factor",
    ]
    for key in required_keys:
        assert key in breakdown
        assert isinstance(breakdown[key], (int, float))


def test_predict_all_players(point_predictor):
    """Test predictions for all players."""
    predictions = point_predictor.predict_all_players(gameweeks=5)

    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(point_predictor.players)

    expected_cols = ["id", "web_name", "team", "element_type", "total_xp"]
    for col in expected_cols:
        assert col in predictions.columns

    # Check GW prediction columns
    for gw in [1, 2, 3, 4, 5]:
        assert f"xp_gw{gw}" in predictions.columns

    # All predictions should be non-negative
    xp_cols = [
        c for c in predictions.columns if c.startswith("xp_gw") or c == "total_xp"
    ]
    for col in xp_cols:
        assert all(predictions[col] >= 0)


def test_multi_gameweek_prediction(point_predictor):
    """Test multi-gameweek prediction for single player."""
    player_id = point_predictor.players.iloc[0]["id"]
    prediction = point_predictor.predict_multi_gameweek(player_id, gameweeks=5)

    assert prediction.player_id == player_id
    assert isinstance(prediction.gameweek_predictions, dict)
    assert len(prediction.gameweek_predictions) == 5
    assert prediction.total_expected > 0
    assert isinstance(prediction.breakdown, dict)


def test_regression_factor_calculation(point_predictor):
    """Test regression factor computation."""
    # Underperformer (goals < xG)
    under_performer = pd.Series(
        {
            "goals_scored": 2.0,
            "expected_goals": 10.0,
            "assists": 1.0,
            "expected_assists": 5.0,
        }
    )
    factor = point_predictor.calculate_regression_factor(under_performer)
    # Underperformers should get factor > 1.0 (positive regression)
    assert factor > 1.0

    # Overperformer (goals > xG)
    over_performer = pd.Series(
        {
            "goals_scored": 10.0,
            "expected_goals": 5.0,
            "assists": 5.0,
            "expected_assists": 3.0,
        }
    )
    factor = point_predictor.calculate_regression_factor(over_performer)
    # Overperformers should get factor < 1.0 (negative regression)
    assert factor < 1.0

    # Insufficient data
    insufficient = pd.Series(
        {
            "goals_scored": 0.5,
            "expected_goals": 0.3,
            "assists": 0.2,
            "expected_assists": 0.1,
        }
    )
    factor = point_predictor.calculate_regression_factor(insufficient)
    assert factor == 1.0


def test_bonus_points_calculation(point_predictor):
    """Test bonus points estimation."""
    player = point_predictor.players.iloc[2]  # MID with good ICT
    bonus = point_predictor.calculate_bonus_points(player, 1)

    assert 0 <= bonus <= 1.2  # Capped at 1.2


def test_form_factor_application(point_predictor):
    """Test form factor in predictions."""
    # High form player (form=8)
    high_form = point_predictor.players.iloc[3]
    xp_high, _ = point_predictor.predict_gameweek(high_form, 1)

    # Temporarily reduce form
    original_form = high_form["form"]
    high_form["form"] = 2.0
    xp_low, _ = point_predictor.predict_gameweek(high_form, 1)

    # Restore form
    high_form["form"] = original_form

    # High form should yield higher predictions
    assert xp_high >= xp_low


def test_minutes_decay(point_predictor):
    """Test minutes decay for players losing playing time."""
    # Player with full minutes
    full_minutes = pd.Series({"recent_4_minutes": 360, "form": 6.0, "element_type": 3})
    player_row = point_predictor.players.iloc[0]

    xp_full, _ = point_predictor.predict_gameweek(player_row, 1)

    # Simulate reduced minutes
    player_row["recent_4_minutes"] = 180  # Average 45 mins
    xp_low, _ = point_predictor.predict_gameweek(player_row, 1)

    player_row["recent_4_minutes"] = 360  # Reset

    assert xp_full >= xp_low


# ============================================================================
# Minutes Predictor Tests
# ============================================================================


def test_start_probability_with_history(minutes_predictor):
    """Test start probability from recent minutes."""
    player_id = 1  # GK with 5 consecutive 90-min starts
    player = minutes_predictor.players[
        minutes_predictor.players["id"] == player_id
    ].iloc[0]
    prob = minutes_predictor.calculate_start_probability(player)

    assert 0.8 <= prob <= 1.0  # Should be very high


def test_start_probability_with_rotation(minutes_predictor):
    """Test start probability for rotated player."""
    player_id = 3  # MID with variable minutes
    player = minutes_predictor.players[minutes_predictor["id"] == player_id].iloc[0]
    prob = minutes_predictor.calculate_start_probability(player)

    # Should be moderate (0.3-0.7)
    assert 0.3 <= prob <= 0.7


def test_start_probability_injured_player(minutes_predictor):
    """Test injured player has near-zero start probability."""
    player = pd.Series(
        {
            "id": 999,
            "element_type": 2,
            "status": "i",  # Injured
            "chance_of_playing_next_round": 0,
        }
    )
    prob = minutes_predictor.calculate_start_probability(player)
    assert prob == 0.0


def test_start_probability_doubtful_player(minutes_predictor):
    """Test doubtful player has reduced start probability."""
    player = pd.Series(
        {
            "id": 999,
            "element_type": 2,
            "status": "d",  # Doubtful
            "chance_of_playing_next_round": 50,
        }
    )
    prob = minutes_predictor.calculate_start_probability(player)
    assert 0 < prob < 0.5


def test_expected_minutes_calculation(minutes_predictor):
    """Test expected minutes based on start probability."""
    player = minutes_predictor.players.iloc[0]
    exp_mins = minutes_predictor.calculate_expected_minutes(player)

    assert 0 <= exp_mins <= 90.0
    assert isinstance(exp_mins, float)


def test_rotation_risk_detection(minutes_predictor):
    """Test rotation pattern detection."""
    player_id = 3  # Variable minutes
    player = minutes_predictor.players[minutes_predictor["id"] == player_id].iloc[0]
    risk = minutes_predictor.calculate_rotation_risk(player)

    assert 0 <= risk <= 1.0  # Risk in 0-1 range


def test_nailedness_score(minutes_predictor):
    """Test overall nailedness score calculation."""
    for _, player in minutes_predictor.players.iterrows():
        score = minutes_predictor.get_nailedness_score(player)
        assert 0 <= score <= 100.0
        assert isinstance(score, float)


def test_predict_all_players_minutes(minutes_predictor):
    """Test minutes predictions for all players."""
    predictions = minutes_predictor.predict_all_players()

    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(minutes_predictor.players)

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

    # Validate ranges
    assert all(predictions["start_probability"].between(0, 1))
    assert all(predictions["expected_minutes"].between(0, 90))
    assert all(predictions["rotation_risk"].between(0, 1))
    assert all(predictions["nailedness"].between(0, 100))


# ============================================================================
# Edge Cases
# ============================================================================


def test_player_with_no_data(point_predictor):
    """Test handling of player with minimal data."""
    no_data_player = pd.Series(
        {
            "id": 999,
            "element_type": 3,
            "team": 1,
            "web_name": "Unknown",
            "minutes": 0,
            "expected_goals": 0,
            "expected_assists": 0,
            "ict_index": 0,
            "form": 0,
            "starts": 0,
            "goals_scored": 0,
            "assists": 0,
            "influence": 0,
            "recent_4_minutes": 0,
            "chance_of_playing_next_round": None,
        }
    )
    xp, breakdown = point_predictor.predict_gameweek(no_data_player, 1)

    assert xp >= 0
    assert all(v >= 0 for v in breakdown.values())


def test_player_no_fixtures(point_predictor):
    """Test player with no fixtures in target gameweek."""
    player = point_predictor.players.iloc[0]
    # Use gameweek with no fixtures
    xp, breakdown = point_predictor.predict_gameweek(player, 99)

    assert xp > 0  # Should still produce estimate
    assert "xg_points" in breakdown


def test_calculation_consistency(point_predictor):
    """Test same player same gameweek yields same prediction."""
    player = point_predictor.players.iloc[0]
    xp1, _ = point_predictor.predict_gameweek(player, 1)
    xp2, _ = point_predictor.predict_gameweek(player, 1)

    assert abs(xp1 - xp2) < 0.0001  # Should be identical


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
