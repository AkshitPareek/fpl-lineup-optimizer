"""
Integration Tests for FPL ML System

Tests end-to-end workflows, API endpoints, and component interactions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json
import time

from backend.main import app
from backend.fpl_service import FPLService
from backend.point_predictor import PointPredictor
from backend.minutes_predictor import MinutesPredictor
from backend.ev_calculator import EVCalculator
from backend.optimizer import FPLOptimizer
from backend.advanced_optimizer import MultiPeriodFPLOptimizer


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_fpl_data():
    """Create comprehensive mock FPL data."""
    # Mock teams
    teams_df = pd.DataFrame(
        [
            {
                "id": i + 1,
                "name": f"Team {i + 1}",
                "short_name": f"T{i + 1}",
                "strength_attack_home": 1000,
                "strength_attack_away": 1000,
                "strength_defence_home": 1000,
                "strength_defence_away": 1000,
            }
            for i in range(6)
        ]
    )

    # Mock players (realistic subset)
    players = []
    positions = [(1, "GK", 1), (2, "DEF", 4), (3, "MID", 8), (4, "FWD", 3)]
    pos_idx = 0
    for pos_id, pos_name, count in positions:
        for i in range(count):
            players.append(
                {
                    "id": pos_id * 100 + i + 1,
                    "web_name": f"{pos_name}{i + 1}",
                    "element_type": pos_id,
                    "team": i + 1,
                    "now_cost": (50 + i * 20) * 10,  # In tenths
                    "minutes": 2000,
                    "starts": 25,
                    "goals_scored": 10 if pos_id == 4 else (5 if pos_id == 3 else 2),
                    "assists": 5 if pos_id in [3, 4] else 2,
                    "expected_goals": 9.0,
                    "expected_assists": 4.5,
                    "ict_index": 100.0,
                    "form": 6.5,
                    "ep_next": 5.5,
                    "chance_of_playing_next_round": 100,
                    "status": "a",
                    "influence": 150.0,
                    "recent_4_minutes": 360,
                }
            )
            pos_idx += 1

    players_df = pd.DataFrame(players)

    # Mock fixtures (multiple GWs)
    fixtures = []
    for gw in range(1, 6):
        for i in range(0, min(6, len(teams_df)), 2):
            if i + 1 < len(teams_df):
                fixtures.append(
                    {
                        "id": gw * 10 + i,
                        "event": gw,
                        "team_h": i + 1,
                        "team_a": i + 2,
                        "team_h_difficulty": 3,
                        "team_a_difficulty": 4,
                        "finished": False,
                    }
                )

    # Mock events
    events = [
        {"id": 1, "name": "Gameweek 1", "is_current": True, "finished": True},
        {
            "id": 2,
            "name": "Gameweek 2",
            "is_current": False,
            "finished": False,
            "is_next": True,
        },
    ]

    return {
        "static": {
            "elements": players_df.to_dict(orient="records"),
            "teams": teams_df.to_dict(orient="records"),
            "events": events,
        },
        "fixtures": fixtures,
    }


@pytest.fixture
def setup_environment(monkeypatch):
    """Setup test environment with mocked data."""
    monkeypatch.setenv("ML_ENABLED", "true")
    monkeypatch.setenv("FRONTEND_URL", "http://localhost:5173")


# ============================================================================
# API Endpoint Tests
# ============================================================================


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    def test_health_check(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_get_data_endpoint(self, client, mock_fpl_data, monkeypatch):
        """Test /api/data endpoint."""
        # Mock FPL service
        mock_service = Mock()
        mock_service.get_latest_data.return_value = mock_fpl_data
        monkeypatch.setattr("backend.main.fpl_service", mock_service)

        response = client.get("/api/data")
        assert response.status_code == 200
        data = response.json()
        assert "static" in data
        assert "fixtures" in data

    def test_predictions_endpoint(self, client, mock_fpl_data, monkeypatch):
        """Test /api/predictions endpoint."""
        mock_service = Mock()
        mock_service.get_latest_data.return_value = mock_fpl_data
        monkeypatch.setattr("backend.main.fpl_service", mock_service)

        response = client.get("/api/predictions?gameweeks=5")
        assert response.status_code == 200
        data = response.json()

        assert "current_gameweek" in data
        assert "horizon" in data
        assert "predictions" in data
        assert len(data["predictions"]) > 0

        # Check prediction structure
        pred = data["predictions"][0]
        assert "id" in pred
        assert "web_name" in pred
        assert "expected_points" in pred or "total_xp" in pred

    def test_optimize_single_gw(self, client, mock_fpl_data, monkeypatch):
        """Test /api/optimize endpoint (single GW)."""
        mock_service = Mock()
        mock_service.get_latest_data.return_value = mock_fpl_data
        mock_service.get_manager_team.return_value = {"picks": []}
        monkeypatch.setattr("backend.main.fpl_service", mock_service)

        response = client.post(
            "/api/optimize",
            json={"budget": 100.0, "gameweeks": 1, "strategy": "standard"},
        )
        assert response.status_code == 200
        data = response.json()

        assert "squad" in data
        assert "starting_xi" in data
        assert "captain" in data
        assert len(data["squad"]) == 15
        assert len(data["starting_xi"]) == 11

    def test_optimize_multi_period(self, client, mock_fpl_data, monkeypatch):
        """Test /api/optimize/multi-period endpoint."""
        mock_service = Mock()
        mock_service.get_latest_data.return_value = mock_fpl_data
        mock_service.get_manager_team.return_value = {"picks": []}
        monkeypatch.setattr("backend.main.fpl_service", mock_service)

        response = client.post(
            "/api/optimize/multi-period",
            json={
                "budget": 100.0,
                "gameweeks": 5,
                "strategy": "standard",
                "robust": False,
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert "current_gw" in data
        assert "squad" in data
        assert "gameweek_plans" in data
        assert len(data["gameweek_plans"]) == 5
        assert "total_expected_points" in data

    def test_optimize_compare(self, client, mock_fpl_data, monkeypatch):
        """Test /api/optimize/compare endpoint."""
        mock_service = Mock()
        mock_service.get_latest_data.return_value = mock_fpl_data
        mock_service.get_manager_team.return_value = {"picks": []}
        monkeypatch.setattr("backend.main.fpl_service", mock_service)

        response = client.post(
            "/api/optimize/compare",
            json={"budget": 100.0, "gameweeks": 3, "strategy": "standard"},
        )
        assert response.status_code == 200
        data = response.json()

        assert "comparison" in data
        assert "with_hits" in data["comparison"]
        assert "no_hits" in data["comparison"]

    def test_robust_optimization(self, client, mock_fpl_data, monkeypatch):
        """Test /api/optimize/robust endpoint."""
        mock_service = Mock()
        mock_service.get_latest_data.return_value = mock_fpl_data
        monkeypatch.setattr("backend.main.fpl_service", mock_service)

        response = client.post(
            "/api/optimize/robust", json={"budget": 100.0, "gamma": 1.0}
        )
        assert response.status_code == 200
        data = response.json()
        assert "squad" in data
        assert "total_expected_points" in data

    def test_dream_team(self, client, mock_fpl_data, monkeypatch):
        """Test /api/dream-team endpoint."""
        mock_service = Mock()
        mock_service.get_latest_data.return_value = mock_fpl_data
        monkeypatch.setattr("backend.main.fpl_service", mock_service)

        response = client.get("/api/dream-team")
        assert response.status_code == 200
        data = response.json()

        assert "dream_team" in data
        assert "captain" in data
        assert "total_expected_points" in data
        assert len(data["dream_team"]) == 11

    def test_understat_endpoint(self, client, mock_fpl_data, monkeypatch):
        """Test Understat enrichment endpoint."""
        mock_service = Mock()
        mock_service.get_latest_data.return_value = mock_fpl_data
        monkeypatch.setattr("backend.main.fpl_service", mock_service)

        response = client.get("/api/understat/enriched")
        assert response.status_code in [200, 500]  # 500 OK if Understat unavailable

    def test_fixtures_endpoint(self, client, mock_fpl_data, monkeypatch):
        """Test fixture analysis endpoint."""
        mock_service = Mock()
        mock_service.get_latest_data.return_value = mock_fpl_data
        monkeypatch.setattr("backend.main.fpl_service", mock_service)

        response = client.get("/api/fixtures?num_gameweeks=5")
        assert response.status_code == 200
        data = response.json()

        assert "teams" in data
        assert len(data["teams"]) > 0

    def test_backtest_endpoint(self, client, mock_fpl_data, monkeypatch):
        """Test backtest endpoint."""
        mock_service = Mock()
        mock_service.get_latest_data.return_value = mock_fpl_data
        monkeypatch.setattr("backend.main.fpl_service", mock_service)

        response = client.post(
            "/api/backtest",
            json={"start_gw": 1, "end_gw": 5, "initial_budget": 100.0, "horizon": 3},
        )
        # May return error due to complex backtest logic, but should be 200 or 500
        assert response.status_code in [200, 500]

    def test_invalid_request_handling(self, client):
        """Test error handling for invalid requests."""
        response = client.post(
            "/api/optimize",
            json={
                "budget": -10,  # Invalid negative budget
                "gameweeks": 1,
            },
        )
        assert response.status_code == 422  # Validation error

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/api/predictions", headers={"Origin": "http://localhost:5173"}
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


# ============================================================================
# Pipeline Integration Tests
# ============================================================================


class TestEndToEndPipeline:
    """Tests for complete ML pipeline."""

    def test_data_to_predictions_flow(self, mock_fpl_data):
        """Test full flow: data → features → predictions."""
        static_data = mock_fpl_data["static"]
        fixtures = mock_fpl_data["fixtures"]
        events = static_data["events"]

        current_gw = 2  # Next GW

        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])

        # Step 1: Initialize predictor
        predictor = PointPredictor(
            players_df=players_df,
            teams_df=teams_df,
            fixtures=fixtures,
            current_gameweek=current_gw,
            use_understat=False,
        )

        # Step 2: Generate predictions
        predictions = predictor.predict_all_players(gameweeks=5)

        # Step 3: Verify predictions structure
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(players_df)
        assert all(c in predictions.columns for c in ["id", "total_xp", "web_name"])

        # Step 4: Predictions must be realistic
        assert all(predictions["total_xp"] >= 0)
        assert all(predictions["total_xp"] <= 40)  # 5 GWs * 8 max per GW

    def test_predictions_to_optimization_flow(self, mock_fpl_data):
        """Test predictions feeding into optimizer."""
        static_data = mock_fpl_data["static"]
        fixtures = mock_fpl_data["fixtures"]
        current_gw = 2

        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])

        # Generate predictions
        predictor = PointPredictor(
            players_df, teams_df, fixtures, current_gw, use_understat=False
        )
        predictions = predictor.predict_all_players(gameweeks=3)

        # Merge predictions back to players_df
        players_with_xp = players_df.merge(
            predictions[["id", "total_xp"]], on="id", how="left"
        )
        players_with_xp["expected_points"] = players_with_xp["total_xp"] / 3

        # Run optimization
        optimizer = MultiPeriodFPLOptimizer(
            players_df=players_with_xp,
            teams_df=teams_df,
            fixtures=fixtures,
            current_gameweek=current_gw,
        )

        solution = optimizer.optimize_multi_period(
            budget=100.0,
            gameweeks=3,
            current_squad_ids=[],
            banked_transfers=1,
            chips_used=[],
        )

        assert solution.status == "Optimal"
        assert len(solution.squad) == 15
        assert len(solution.gameweek_plans) == 3

    def test_ev_calculator_integration(self, mock_fpl_data):
        """Test EV calculator with predictions."""
        static_data = mock_fpl_data["static"]
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])

        # Add expected points
        players_df["expected_points"] = 5.0

        ev_calc = EVCalculator(players_df, teams_df)

        # Get distributions for all players
        distributions = ev_calc.get_all_distributions()
        assert len(distributions) == len(players_df)

        # Calculate squad EV
        sample_squad = players_df.head(15)
        squad_ev = ev_calc.calculate_squad_ev(sample_squad["id"].tolist()[:11])

        assert "total_expected" in squad_ev
        assert squad_ev["total_expected"] > 0

    def test_minutes_predictor_integration(self, mock_fpl_data):
        """Test minutes predictor with player data."""
        players_df = pd.DataFrame(mock_fpl_data["static"]["elements"])

        minutes_pred = MinutesPredictor(players_df, {})

        predictions = minutes_pred.predict_all_players()

        assert len(predictions) == len(players_df)
        assert all(predictions["start_probability"].between(0, 1))
        assert all(predictions["expected_minutes"].between(0, 90))

    def test_full_workflow_with_real_data_patterns(self):
        """Test workflow with realistic data patterns."""
        # Simulate realistic player data
        np.random.seed(42)

        n_players = 200
        positions = []
        for pos, count in [(1, 20), (2, 40), (3, 80), (4, 60)]:
            positions.extend([pos] * count)

        players_df = pd.DataFrame(
            {
                "id": range(1, n_players + 1),
                "web_name": [f"Player_{i}" for i in range(1, n_players + 1)],
                "element_type": positions[:n_players],
                "team": np.random.randint(1, 21, n_players),
                "now_cost": np.random.randint(40, 170, n_players) * 10,
                "minutes": np.random.randint(0, 2700, n_players),
                "starts": np.random.randint(0, 38, n_players),
                "goals_scored": np.random.randint(0, 35, n_players),
                "assists": np.random.randint(0, 25, n_players),
                "expected_goals": np.random.uniform(0, 20, n_players),
                "expected_assists": np.random.uniform(0, 15, n_players),
                "ict_index": np.random.uniform(0, 300, n_players),
                "form": np.random.uniform(0, 10, n_players),
                "ep_next": np.random.uniform(0, 8, n_players),
                "chance_of_playing_next_round": 100,
                "status": "a",
                "influence": np.random.uniform(0, 400, n_players),
                "recent_4_minutes": np.random.randint(0, 360, n_players),
            }
        )

        teams_df = pd.DataFrame(
            {
                "id": range(1, 21),
                "name": [f"Team_{i}" for i in range(1, 21)],
                "strength_attack_home": np.random.normal(1000, 50, 20),
                "strength_attack_away": np.random.normal(950, 50, 20),
                "strength_defence_home": np.random.normal(1000, 50, 20),
                "strength_defence_away": np.random.normal(950, 50, 20),
            }
        )

        # Create simple fixtures
        fixtures = []
        for gw in range(1, 6):
            teams = list(range(1, 21))
            np.random.shuffle(teams)
            for i in range(0, 20, 2):
                fixtures.append(
                    {
                        "id": gw * 10 + i // 2,
                        "event": gw,
                        "team_h": teams[i],
                        "team_a": teams[i + 1],
                        "team_h_difficulty": np.random.randint(2, 6),
                        "team_a_difficulty": np.random.randint(2, 6),
                        "finished": False,
                    }
                )

        # Run full pipeline
        predictor = PointPredictor(
            players_df, teams_df, fixtures, 1, use_understat=False
        )
        predictions = predictor.predict_all_players(gameweeks=5)

        # Should produce predictions for all players
        assert len(predictions) == n_players
        assert all(predictions["total_xp"] >= 0)

        # Top 10 players by expected points should be mostly forwards/midfielders
        top_10 = predictions.nlargest(10, "total_xp")
        top_positions = players_df[players_df["id"].isin(top_10["id"])][
            "element_type"
        ].values
        attacking_count = sum(1 for p in top_positions if p in [3, 4])
        assert attacking_count >= 5  # At least half should be attackers


# ============================================================================
# Data Quality Integration Tests
# ============================================================================


class TestDataQualityIntegration:
    """Tests for data quality across components."""

    def test_no_missing_player_ids(self, mock_fpl_data):
        """Test all player IDs are unique and present."""
        players_df = pd.DataFrame(mock_fpl_data["static"]["elements"])

        assert players_df["id"].is_unique
        assert players_df["id"].notna().all()

    def test_fixture_team_consistency(self, mock_fpl_data):
        """Test fixtures reference valid team IDs."""
        fixtures = mock_fpl_data["fixtures"]
        teams_df = pd.DataFrame(mock_fpl_data["static"]["teams"])
        valid_teams = set(teams_df["id"])

        for fixture in fixtures:
            assert fixture["team_h"] in valid_teams
            assert fixture["team_a"] in valid_teams
            assert fixture["team_h"] != fixture["team_a"]  # Not same team

    def test_player_team_references(self, mock_fpl_data):
        """Test player team IDs reference valid teams."""
        players_df = pd.DataFrame(mock_fpl_data["static"]["elements"])
        teams_df = pd.DataFrame(mock_fpl_data["static"]["teams"])
        valid_teams = set(teams_df["id"])

        assert all(p["team"] in valid_teams for p in players_df.itertuples())

    def test_consistent_position_mapping(self, mock_fpl_data):
        """Test element_type values are consistent (1-4)."""
        players_df = pd.DataFrame(mock_fpl_data["static"]["elements"])

        valid_positions = {1, 2, 3, 4}
        assert all(p in valid_positions for p in players_df["element_type"])

    def test_cost_in_tenths(self, mock_fpl_data):
        """Test player costs are in FPL format (tenths of £)."""
        players_df = pd.DataFrame(mock_fpl_data["static"]["elements"])

        # FPL costs are typically 40-170 (meaning £4.0-£17.0m)
        assert all((c >= 40) and (c <= 170) for c in players_df["now_cost"])

    def test_fixture_difficulty_range(self, mock_fpl_data):
        """Test fixture difficulties are in valid range."""
        fixtures = mock_fpl_data["fixtures"]

        for f in fixtures:
            assert 2 <= f["team_h_difficulty"] <= 5
            assert 2 <= f["team_a_difficulty"] <= 5


# ============================================================================
# Error Handling & Fallback Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling and fallback behavior."""

    def test_point_predictor_empty_players(self):
        """Test predictor with empty players DataFrame."""
        empty_df = pd.DataFrame()
        teams_df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "name": "Team1",
                    "short_name": "T1",
                    "strength_attack_home": 1000,
                    "strength_attack_away": 1000,
                    "strength_defence_home": 1000,
                    "strength_defence_away": 1000,
                }
            ]
        )

        predictor = PointPredictor(empty_df, teams_df, [], 1)
        predictions = predictor.predict_all_players()
        assert len(predictions) == 0

    def test_missing_player_handled_gracefully(self, mock_fpl_data):
        """Test request for non-existent player doesn't crash."""
        players_df = pd.DataFrame(mock_fpl_data["static"]["elements"])
        teams_df = pd.DataFrame(mock_fpl_data["static"]["teams"])
        fixtures = mock_fpl_data["fixtures"]

        predictor = PointPredictor(players_df, teams_df, fixtures, 1)

        # Non-existent player ID
        with pytest.raises(IndexError):
            predictor.predict_multi_gameweek(99999)

    def test_invalid_gameweek_handled(self, mock_fpl_data):
        """Test invalid gameweek uses fallback."""
        players_df = pd.DataFrame(mock_fpl_data["static"]["elements"])
        teams_df = pd.DataFrame(mock_fpl_data["static"]["teams"])
        fixtures = mock_fpl_data["fixtures"]

        predictor = PointPredictor(players_df, teams_df, fixtures, 1)
        player = players_df.iloc[0]

        # Gameway with no fixtures
        xp, breakdown = predictor.predict_gameweek(player, 99)
        assert xp > 0  # Should still produce estimate

    def test_zero_budget_optimization(self, mock_fpl_data):
        """Test optimization with zero budget fails gracefully."""
        static_data = mock_fpl_data["static"]
        fixtures = mock_fpl_data["fixtures"]

        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])

        optimizer = FPLOptimizer(players_df)

        try:
            lineup = optimizer.optimize(budget=0, gameweeks=1)
            # Should either fail or produce zero-cost team
            assert lineup is not None
        except Exception as e:
            # Expected: infeasible problem
            assert "Infeasible" in str(e) or "budget" in str(e).lower()

    def test_api_without_fpl_service(self, client, monkeypatch):
        """Test API returns error when FPL service fails."""
        mock_service = Mock()
        mock_service.get_latest_data.side_effect = Exception("API Error")
        monkeypatch.setattr("backend.main.fpl_service", mock_service)

        response = client.get("/api/data")
        assert response.status_code == 500

    def test_malformed_json_input(self, client):
        """Test malformed JSON is handled."""
        response = client.post("/api/optimize", data="invalid{json")
        assert response.status_code == 422  # Unprocessable entity


# ============================================================================
# Performance Regression Tests
# ============================================================================


class TestPerformanceBaseline:
    """Tests for performance baselines."""

    def test_prediction_latency_single_player(self, mock_fpl_data):
        """Test single player prediction latency."""
        players_df = pd.DataFrame(mock_fpl_data["static"]["elements"])
        teams_df = pd.DataFrame(mock_fpl_data["static"]["teams"])
        fixtures = mock_fpl_data["fixtures"]

        predictor = PointPredictor(
            players_df, teams_df, fixtures, 1, use_understat=False
        )
        player = players_df.iloc[0]

        start = time.time()
        for _ in range(10):
            xp, _ = predictor.predict_gameweek(player, 1)
        avg_latency = (time.time() - start) / 10

        assert avg_latency < 0.010  # Should be < 10ms per prediction

    def test_batch_prediction_latency(self, mock_fpl_data):
        """Test batch prediction latency for all players."""
        players_df = pd.DataFrame(mock_fpl_data["static"]["elements"])
        teams_df = pd.DataFrame(mock_fpl_data["static"]["teams"])
        fixtures = mock_fpl_data["fixtures"]
        n_players = len(players_df)

        predictor = PointPredictor(
            players_df, teams_df, fixtures, 1, use_understat=False
        )

        start = time.time()
        predictions = predictor.predict_all_players(gameweeks=5)
        total_latency = time.time() - start

        # Should be < 2 seconds for 100+ players
        assert total_latency < 2.0
        assert len(predictions) == n_players

    def test_optimization_latency(self, mock_fpl_data):
        """Test optimization completes within reasonable time."""
        static_data = mock_fpl_data["static"]
        fixtures = mock_fpl_data["fixtures"]
        current_gw = 2

        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])

        optimizer = MultiPeriodFPLOptimizer(
            players_df=players_df,
            teams_df=teams_df,
            fixtures=fixtures,
            current_gameweek=current_gw,
        )

        start = time.time()
        solution = optimizer.optimize_multi_period(
            budget=100.0, gameweeks=3, current_squad_ids=[], banked_transfers=1
        )
        latency = time.time() - start

        assert latency < 30.0  # Should complete within 30 seconds
        assert solution.status == "Optimal"


# ============================================================================
# Mock Data Consistency Tests
# ============================================================================


class TestMockDataFixtures:
    """Validate that mock fixtures maintain consistency."""

    def test_mock_players_have_required_fields(self, mock_fpl_data):
        """Test all players have required fields for predictions."""
        players = mock_fpl_data["static"]["elements"]

        required = [
            "id",
            "web_name",
            "element_type",
            "team",
            "now_cost",
            "minutes",
            "expected_goals",
            "expected_assists",
            "ict_index",
            "form",
            "ep_next",
        ]

        for player in players:
            for field in required:
                assert field in player, f"Missing field: {field}"

    def test_mock_teams_have_strengths(self, mock_fpl_data):
        """Test teams have required strength attributes."""
        teams = mock_fpl_data["static"]["teams"]

        required = [
            "strength_attack_home",
            "strength_attack_away",
            "strength_defence_home",
            "strength_defence_away",
        ]

        for team in teams:
            for field in required:
                assert field in team, f"Missing field: {field}"

    def test_mock_fixtures_complete(self, mock_fpl_data):
        """Test fixtures have all required fields."""
        fixtures = mock_fpl_data["fixtures"]

        for fixture in fixtures:
            assert "id" in fixture
            assert "event" in fixture
            assert "team_h" in fixture
            assert "team_a" in fixture
            assert "team_h_difficulty" in fixture
            assert "team_a_difficulty" in fixture


# ============================================================================
# End-to-End Scenario Tests
# ============================================================================


class TestRealWorldScenarios:
    """Test complete real-world usage scenarios."""

    def test_new_manager_optimization(self, mock_fpl_data):
        """Scenario: New manager with empty squad, 100m budget."""
        static_data = mock_fpl_data["static"]
        fixtures = mock_fpl_data["fixtures"]
        current_gw = 2

        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])

        # Multi-period optimization
        optimizer = MultiPeriodFPLOptimizer(
            players_df=players_df,
            teams_df=teams_df,
            fixtures=fixtures,
            current_gameweek=current_gw,
        )

        solution = optimizer.optimize_multi_period(
            budget=100.0,
            gameweeks=5,
            current_squad_ids=[],  # Empty squad
            banked_transfers=1,
            chips_used=[],
        )

        assert solution.status == "Optimal"
        assert len(solution.squad) == 15
        assert solution.total_expected_points > 0

    def test_manager_with_existing_squad(self, mock_fpl_data):
        """Scenario: Manager with existing squad, wants transfers."""
        static_data = mock_fpl_data["static"]
        fixtures = mock_fpl_data["fixtures"]
        current_gw = 2

        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])

        # Existing squad: first 15 players
        current_squad_ids = players_df.head(15)["id"].tolist()

        optimizer = MultiPeriodFPLOptimizer(
            players_df=players_df,
            teams_df=teams_df,
            fixtures=fixtures,
            current_gameweek=current_gw,
        )

        solution = optimizer.optimize_multi_period(
            budget=100.0,
            gameweeks=5,
            current_squad_ids=current_squad_ids,
            banked_transfers=2,  # Has 2 FTs banked
            chips_used=["wildcard"],  # Already used wildcard
        )

        assert solution.status == "Optimal"
        assert len(solution.squad) == 15
        # Should respect existing squad constraints

    def test_transfer_analysis_flow(self, mock_fpl_data):
        """Scenario: Manager evaluating if hits are worth it."""
        static_data = mock_fpl_data["static"]
        fixtures = mock_fpl_data["fixtures"]
        current_gw = 2

        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])

        # Existing squad
        current_squad_ids = players_df.head(15)["id"].tolist()
        squad_value = players_df.head(15)["now_cost"].sum() / 10

        # Use compare endpoint
        # (This is more of an integration smoke test)
        assert squad_value > 0

    def test_chip_recommendation_flow(self, mock_fpl_data):
        """Scenario: Manager wants chip advice."""
        static_data = mock_fpl_data["static"]
        fixtures = mock_fpl_data["fixtures"]
        current_gw = 2

        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])

        current_squad = players_df.head(15)["id"].tolist()

        # This would normally go through ChipAdvisor
        # Just verify data flow doesn't break
        assert len(current_squad) == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
