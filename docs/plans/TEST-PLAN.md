# FPL ML Integration Test Plan

**Version:** 1.0  
**Last Updated:** 2025-03-04  
**Team:** FPL ML Integration  
**Purpose:** Comprehensive testing strategy for ML system reliability, accuracy, and safety

---

## 1. Test Strategy Overview

The FPL ML system requires rigorous testing across multiple dimensions to ensure production readiness. The system integrates:
- **PointPredictor**: Expected points calculation with xG/xA, ICT, CBIT/CBIRT, fixtures
- **MinutesPredictor**: Start probability and expected minutes
- **EVCalculator**: Risk analysis and Monte Carlo simulations
- **UnderstatService**: External data integration for advanced metrics
- **Optimizers**: Multi-period and robust optimization consuming predictions

Testing follows the **Testing Pyramid**:
- 70% Unit Tests (fast, isolated)
- 20% Integration Tests (component interactions)
- 10% Performance & E2E Tests (slow, comprehensive)

---

## 2. Unit Tests

### 2.1 Feature Engineering Tests

**File:** `backend/tests/test_feature_engineering.py`

**Coverage:**
- xG/xA per 90 calculations
- Understat data enrichment fallback logic
- Team strength computation
- Fixture map building
- CBIT/CBIRT bonus estimation
- Regression factor calculation
- Form and minutes decay factors

**Sample Tests:**
```python
def test_xg_per_90_with_understat():
    """Test xG/90 uses Understat data when available."""
    predictor = PointPredictor(mock_players, mock_teams, mock_fixtures, 1, use_understat=True)
    player = mock_players[mock_players['understat_matched'] == True].iloc[0]
    xg_per_90 = predictor._get_xg_per_90(player)
    assert xg_per_90 == player['understat_xG_per_90']
    assert xg_per_90 > 0

def test_xg_per_90_fallback_to_fpl():
    """Test xG/90 falls back to FPL data when Understat unavailable."""
    predictor = PointPredictor(mock_players, mock_teams, mock_fixtures, 1, use_understat=False)
    player = mock_players[mock_players['minutes'] > 270].iloc[0]
    xg_per_90 = predictor._get_xg_per_90(player)
    expected = (player['expected_goals'] / player['minutes']) * 90
    assert abs(xg_per_90 - expected) < 0.001

def test_baseline_xg_for_limited_data():
    """Test position-based baseline for players with limited minutes."""
    predictor = PointPredictor(mock_players, mock_teams, mock_fixtures, 1)
    player = create_player(minutes=100, element_type=4)  # Forward
    xg_per_90 = predictor._get_xg_per_90(player)
    assert xg_per_90 == predictor.BASELINE_XG[4]  # 0.35 for forwards
```

### 2.2 Point Predictor Tests

**File:** `backend/tests/test_ml_predictor.py`

**Coverage:**
- Model initialization and data validation
- Single gameweek prediction
- Multi-gameweek prediction aggregation
- Component breakdown accuracy
- Prediction distribution sanity (min/max bounds)
- Edge cases: injured players, no fixtures, missing data

**Sample Tests:**
```python
def test_predict_gameweek_returns_valid_points():
    """Test prediction returns finite, positive points."""
    predictor = PointPredictor(players_df, teams_df, fixtures, current_gw)
    player = players_df.iloc[0]
    xp, breakdown = predictor.predict_gameweek(player, current_gw)
    assert 0 <= xp <= 8.0  # Cap at 8.0 per gameweek
    assert all(0 <= v for v in breakdown.values())
    assert 'xg_points' in breakdown
    assert 'xa_points' in breakdown

def test_xg_points_calculation():
    """Test xG points calculation with fixture adjustment."""
    predictor = PointPredictor(players_df, teams_df, fixtures, current_gw)
    player = create_test_player(element_type=4, team=1)  # Forward
    xg_points = predictor.calculate_xg_points(player, current_gw)
    # Verify opponent defense adjustment applied
    assert xg_points >= 0
    # Should be capped at 9.0
    assert xg_points <= 9.0

def test_cbit_bonus_for_defenders():
    """Test CBIT bonus calculation for defenders."""
    predictor = PointPredictor(players_df, teams_df, fixtures, current_gw)
    high_influence_def = create_player(
        element_type=2,  # Defender
        influence=80,
        minutes=2000
    )
    cbit = predictor.calculate_cbit_bonus(high_influence_def, current_gw)
    assert cbit >= 0
    assert cbit <= predictor.POINTS_DEFENSIVE_BONUS  # Cap at 2.0

def test_prediction_with_no_fixtures():
    """Test fallback when player has no upcoming fixtures."""
    predictor = PointPredictor(players_df, teams_df, fixtures, current_gw)
    player = create_player(team=999)  # Team not in fixtures
    xp, breakdown = predictor.predict_gameweek(player, current_gw)
    # Should use default estimates
    assert xp > 0 and xp < 4.0
```

### 2.3 Minutes Predictor Tests

**File:** `backend/tests/test_ml_predictor.py` (same file)

**Coverage:**
- Start probability calculation
- Expected minutes estimation
- Rotation risk detection
- Nailedness score computation
- Availability status handling

**Sample Tests:**
```python
def test_start_probability_with_history():
    """Test start prob from recent minutes."""
    predictor = MinutesPredictor(players_df, history_cache)
    player = create_player_with_history(
        recent_minutes=[90, 90, 85, 0, 90]  # 80% starts
    )
    prob = predictor.calculate_start_probability(player)
    assert 0.7 <= prob <= 0.9  # High start probability

def test_rotation_risk_detection():
    """Test rotation pattern detection."""
    predictor = MinutesPredictor(players_df, history_cache)
    player = create_player_with_history(
        recent_minutes=[90, 0, 90, 0, 90]  # Alternating pattern
    )
    risk = predictor.calculate_rotation_risk(player)
    assert risk > 0.5  # High rotation variance

def test_injured_player_zero_start_prob():
    """Test injured players have 0 start probability."""
    predictor = MinutesPredictor(players_df, history_cache)
    player = create_player(status='i')  # Injured
    prob = predictor.calculate_start_probability(player)
    assert prob == 0.0
```

### 2.4 EV Calculator Tests

**File:** `backend/tests/test_ml_predictor.py` (same file)

**Coverage:**
- Distribution percentiles (10th/90th)
- Risk score calculation
- Squad-level EV aggregation
- Monte Carlo simulation convergence
- Position variance application

**Sample Tests:**
```python
def test_ev_distribution_floor_ceiling():
    """Test floor is less than expected, ceiling is greater."""
    ev_calc = EVCalculator(players_df, teams_df)
    dist = ev_calc.calculate_player_distribution(player_id)
    assert dist.floor <= dist.expected_points <= dist.ceiling
    assert dist.ceiling - dist.expected_points >= 0
    assert dist.expected_points - dist.floor >= 0

def test_position_variance_consistency():
    """Test forwards have higher variance than defenders."""
    ev_calc = EVCalculator(players_df, teams_df)
    def_dist = ev_calc.calculate_player_distribution(defender_id)
    fwd_dist = ev_calc.calculate_player_distribution(forward_id)
    assert fwd_dist.std_dev > def_dist.std_dev * 1.2  # Forwards at least 20% more volatile

def test_squad_ev_aggregation():
    """Test squad EV metrics calculation."""
    ev_calc = EVCalculator(players_df, teams_df)
    squad_ev = ev_calc.calculate_squad_ev(starting_xi_ids)
    assert 'total_expected' in squad_ev
    assert 'floor' in squad_ev
    assert 'ceiling' in squad_ev
    assert squad_ev['floor'] <= squad_ev['total_expected'] <= squad_ev['ceiling']
```

### 2.5 Model Loading & Configuration

**Tests:**
```python
def test_point_predictor_initialization():
    """Test predictor initializes correctly with all required attributes."""
    predictor = PointPredictor(players_df, teams_df, fixtures, current_gw)
    assert hasattr(predictor, 'players')
    assert hasattr(predictor, 'teams')
    assert hasattr(predictor, 'fixture_map')
    assert hasattr(predictor, 'team_attack_home')
    assert hasattr(predictor, 'team_defense_away')

def test_understat_service_import():
    """Test UnderstatService can be imported."""
    try:
        from understat_service import UnderstatService
        assert True
    except ImportError:
        pytest.skip("Understat not installed")
```

---

## 3. Integration Tests

**File:** `backend/tests/test_integration_ml.py`

### 3.1 End-to-End ML Pipeline

**Test: Data → Features → Prediction → Optimization**
```python
def test_full_pipeline_optimization():
    """Test complete pipeline from raw data to optimized team."""
    # 1. Fetch data
    data = fpl_service.get_latest_data()
    players_df = pd.DataFrame(data['static']['elements'])
    teams_df = pd.DataFrame(data['static']['teams'])
    fixtures = data['fixtures']
    current_gw = get_active_gameweek(data['static']['events'])

    # 2. Generate predictions
    predictor = PointPredictor(players_df, teams_df, fixtures, current_gw)
    predictions = predictor.predict_all_players(gameweeks=5)
    assert len(predictions) == len(players_df)
    assert 'total_xp' in predictions.columns
    assert all(predictions['total_xp'] >= 0)

    # 3. Run optimization
    optimizer = MultiPeriodFPLOptimizer(
        players_df=players_df,
        teams_df=teams_df,
        fixtures=fixtures,
        current_gameweek=current_gw
    )
    solution = optimizer.optimize_multi_period(
        budget=100.0,
        gameweeks=5,
        current_squad_ids=[],
        banked_transfers=1,
        chips_used=[]
    )
    assert solution.status == "Optimal"
    assert len(solution.squad) == 15
    assert len(solution.gameweek_plans) == 5

def test_predictions_influence_optimization():
    """Test that changing predictions affects optimization results."""
    # Get baseline
    predictor1 = PointPredictor(players_df, teams_df, fixtures, current_gw)
    preds1 = predictor1.predict_all_players(gameweeks=3)
    baseline_xp = preds1['total_xp'].sum()

    # Artificially boost a player's predictions
    preds2 = preds1.copy()
    star_idx = preds2['total_xp'].idxmax()
    preds2.loc[star_idx, 'total_xp'] *= 1.5

    # Re-run optimization with modified predictions (by manipulating expected_points)
    players_modified = players_df.copy()
    players_modified.loc[players_modified['id'] == preds2.loc[star_idx, 'id'], 'ep_next'] = \
        preds2.loc[star_idx, 'total_xp'] / 3

    optimizer = MultiPeriodFPLOptimizer(players_modified, teams_df, fixtures, current_gw)
    solution = optimizer.optimize_multi_period(budget=100.0, gameweeks=3)

    # Optimal team should now include the boosted player
    squad_ids = [p.element for p in solution.squad]
    assert preds2.loc[star_idx, 'id'] in squad_ids
```

### 3.2 API Integration Tests

**Test: ML-enabled vs ML-disabled endpoints**
```python
def test_predictions_endpoint():
    """Test /api/predictions endpoint returns valid data."""
    response = client.get("/api/predictions?gameweeks=5")
    assert response.status_code == 200
    data = response.json()
    assert 'predictions' in data
    assert len(data['predictions']) > 0
    for p in data['predictions']:
        assert 'expected_points' in p or 'total_xp' in p
        assert p['expected_points'] >= 0

def test_optimization_uses_ml_predictions():
    """Test that optimization endpoint uses ML predictions."""
    response = client.post("/api/optimize/multi-period", json={
        "budget": 100.0,
        "gameweeks": 5,
        "strategy": "standard",
        "robust": False
    })
    assert response.status_code == 200
    data = response.json()
    assert 'total_expected_points' in data
    assert data['total_expected_points'] > 30  # Reasonable minimum

def test_health_check():
    """Test health endpoint returns OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
```

### 3.3 Database & State Tests

**Tests:**
```python
def test_prediction_caching():
    """Test that predictions can be cached and retrieved."""
    cache_key = f"predictions:{current_gw}:5"
    predictions = predictor.predict_all_players(gameweeks=5)
    cache.set(cache_key, predictions, ttl=3600)
    cached = cache.get(cache_key)
    assert cached.equals(predictions)

def test_data_freshness():
    """Test that stale data (>24h) triggers refresh."""
    old_timestamp = datetime.utcnow() - timedelta(hours=25)
    cache_data(old_timestamp)
    assert should_refresh_data() is True
```

---

## 4. Data Quality Tests

**File:** `backend/tests/test_data_quality.py`

### 4.1 Historical Data Validation

```python
def test_no_gaps_in_historical_data():
    """Test that historical gameweek data has no missing weeks."""
    history = get_historical_data(player_id, season="2025")
    gw_numbers = [entry['event'] for entry in history]
    expected_gws = range(min(gw_numbers), max(gw_numbers) + 1)
    assert set(gw_numbers) >= set(expected_gws)  # No gaps

def test_consistent_data_types():
    """Test DataFrame columns have correct dtypes."""
    players_df = pd.DataFrame(static_data['elements'])
    assert players_df['id'].dtype in [np.int64, np.int32]
    assert players_df['now_cost'].dtype in [np.float64, np.float32]
    assert players_df['element_type'].dtype in [np.int64, np.int32]

def test_valid_range_for_metrics():
    """Test metrics are within valid ranges."""
    players_df = pd.DataFrame(static_data['elements'])
    assert all(players_df['now_cost'] >= 40)  # Min player cost
    assert all(players_df['now_cost'] <= 170)  # Max player cost
    assert all(players_df['minutes'] >= 0)
```

### 4.2 Feature Distribution Checks

```python
def test_xg_distribution_not_normalized():
    """Test xG per 90 is not all identical (would indicate data issue)."""
    valid_players = players_df[players_df['minutes'] >= 270]
    if len(valid_players) > 10:
        xg_values = (valid_players['expected_goals'] / valid_players['minutes']) * 90
        assert xg_values.std() > 0.01  # Non-zero variance

def test_no_negative_values_in_predictions():
    """Test all prediction outputs are non-negative."""
    predictions = predictor.predict_all_players()
    numeric_cols = predictions.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col.startswith('xp_gw') or col == 'total_xp':
            assert all(predictions[col] >= 0)
```

### 4.3 Outlier Detection

```python
def test_extreme_predictions_are_flagged():
    """Test extremely high predictions (>15pts) are identified."""
    predictions = predictor.predict_all_players()
    outliers = predictions[predictions['total_xp'] > 15]
    if len(outliers) > 0:
        # Log warning for manual review
        logger.warning(f"Found {len(outliers)} players with extreme predictions")
```

---

## Germany> 5. Model Validation Tests

### 5.1 Accuracy on Holdout Set

**File:** `backend/tests/test_model_validation.py`

```python
def test_mae_within_threshold():
    """Test Mean Absolute Error on holdout set meets threshold."""
    # Load holdout data (past gameweek)
    holdout_data = load_holdout_data(gw=10)
    actual_points = holdout_data['actual_points']

    predictions = []
    for _, player in holdout_data.iterrows():
        xp, _ = predictor.predict_gameweek(player, 10)
        predictions.append(xp)

    mae = np.mean(np.abs(np.array(predictions) - actual_points))
    # Historical FPL prediction MAE: ~2.5-3.5 points
    assert mae < 4.0, f"MAE {mae:.2f} exceeds threshold of 4.0"

def test_r2_not_negative():
    """Test R² score is above 0 (better than mean prediction)."""
    r2 = r2_score(actual_points, predictions)
    assert r2 > 0.0, f"R² is negative: {r2:.3f}"
    # Realistic expectation: R² ~ 0.10-0.20 for FPL
    assert r2 > -0.5  # Extreme check, should be positive
```

### 5.2 Feature Importance Sanity

```python
def test_top_xg_players_are_attackers():
    """Test players with highest xG are forwards/midfielders."""
    top_xg = predictions.nlargest(20, 'xg_points')
    positions = top_xg['element_type'].values
    forward_count = sum(1 for p in positions if p == 4)
    mid_count = sum(1 for p in positions if p == 3)
    # At least 80% should be attackers
    assert (forward_count + mid_count) / len(positions) >= 0.8

def test_goalkeepers_have_low_xg():
    """Test GKs have negligible xG scores."""
    gks = predictions[predictions['element_type'] == 1]
    assert all(gks['xg_points'] < 0.5)
```

### 5.3 Bias Detection

```python
def test_no_position_bias():
    """Test predictions are not systematically biased by position."""
    predictions = predictor.predict_all_players()
    for pos in [1, 2, 3, 4]:
        pos_preds = predictions[predictions['element_type'] == pos]['total_xp']
        mean_pred = pos_preds.mean()
        overall_mean = predictions['total_xp'].mean()
        # Position mean should be within ±50% of overall
        assert 0.5 * overall_mean <= mean_pred <= 1.5 * overall_mean

def test_no_team_bias():
    """Test predictions not systematically biased by team."""
    predictions = predictor.predict_all_players()
    team_means = predictions.groupby('team')['total_xp'].mean()
    overall_mean = predictions['total_xp'].mean()
    team_deviation = (team_means - overall_mean).abs() / overall_mean
    # No team should deviate by >100%
    assert all(team_deviation < 1.0)
```

---

## 6. Performance Tests

**File:** `backend/tests/test_performance.py`

### 6.1 Inference Speed

```python
def test_single_player_prediction_latency():
    """Test single prediction completes in <50ms."""
    import time
    player = players_df.iloc[0]
    start = time.time()
    xp, breakdown = predictor.predict_gameweek(player, current_gw)
    latency = (time.time() - start) * 1000  # ms
    assert latency < 50, f"Single prediction took {latency:.1f}ms"

def test_all_players_batch_latency():
    """Test full player predictions complete in <2s."""
    import time
    start = time.time()
    predictions = predictor.predict_all_players(gameweeks=5)
    latency = (time.time() - start)
    assert latency < 2.0, f"All players prediction took {latency:.1f}s"

def test_multi_period_optimization_latency():
    """Test multi-period optimization completes in <30s."""
    import time
    start = time.time()
    solution = optimizer.optimize_multi_period(
        budget=100.0,
        gameweeks=5,
        current_squad_ids=[],
        banked_transfers=1
    )
    latency = (time.time() - start)
    assert latency < 30.0, f"Multi-period optimization took {latency:.1f}s"
```

### 6.2 Memory Usage

```python
def test_memory_usage_under_limit():
    """Test memory usage stays under 500MB."""
    import psutil, os
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / 1024 / 1024  # MB

    # Run heavy operation
    predictions = predictor.predict_all_players(gameweeks=8)
    ev_calc = EVCalculator(players_df, teams_df)
    dists = ev_calc.get_all_distributions()

    after = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = after - before
    assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"
```

### 6.3 API Latency Benchmarks

**File:** `backend/tests/test_api_performance.py`

```python
def test_predictions_endpoint_latency():
    """Test /api/predictions returns in <3s."""
    start = time.time()
    response = client.get("/api/predictions?gameweeks=5")
    latency = time.time() - start
    assert response.status_code == 200
    assert latency < 3.0, f"API latency: {latency:.1f}s"

def test_optimization_endpoint_latency():
    """Test /api/optimize/multi-period returns in <45s."""
    start = time.time()
    response = client.post("/api/optimize/multi-period", json={
        "budget": 100.0,
        "gameweeks": 5,
        "strategy": "standard"
    })
    latency = time.time() - start
    assert response.status_code == 200
    assert latency < 45.0, f"Optimization took {latency:.1f}s"
```

---

## 7. A/B Testing Framework

**File:** `backend/tests/test_ab_testing.py`

### 7.1 Comparison Framework

```python
class ABTest:
    """Framework for comparing ML predictions vs rule-based baseline."""

    def __init__(self, predictor_ml, predictor_baseline):
        self.predictor_ml = predictor_ml
        self.predictor_baseline = predictor_baseline

    def run_experiment(self, test_period_gws: List[int], sample_size: int = 100):
        """Compare predictions on historical periods."""
        results = []
        for gw in test_period_gws:
            # Get actual outcomes for gw
            actual_data = load_gameweek_actuals(gw)

            ml_preds = self.predictor_ml.predict_all_players(gameweeks=1)
            base_preds = self.predictor_baseline.predict_all_players(gameweeks=1)

            # Evaluate accuracy
            ml_mae = mean_absolute_error(actual_data, ml_preds)
            base_mae = mean_absolute_error(actual_data, base_preds)

            results.append({
                'gameweek': gw,
                'ml_mae': ml_mae,
                'baseline_mae': base_mae,
                'improvement': base_mae - ml_mae
            })

        return pd.DataFrame(results)

    def test_statistical_significance(self, results_df: pd.DataFrame) -> bool:
        """Paired t-test on per-gameweek errors."""
        from scipy import stats
        _, p_value = stats.ttest_rel(
            results_df['baseline_mae'],
            results_df['ml_mae']
        )
        return p_value < 0.05  # 95% confidence

def test_ml_beats_baseline():
    """Test ML predictions are statistically better than baseline."""
    ab_test = ABTest(ml_predictor, baseline_predictor)
    results = ab_test.run_experiment(test_period_gws=[10, 11, 12, 13, 14])
    assert ab_test.test_statistical_significance(results)
    avg_improvement = results['improvement'].mean()
    assert avg_improvement > 0.1  # At least 0.1pt MAE improvement
```

### 7.2 Canary Deployment Strategy

**Criteria for ML Enablement:**
- Pass all unit tests (>80% coverage)
- MAE on last 3 GWS < 3.5 points
- No position bias (p-value > 0.05)
- API latency < 3s p95
- No crashes in 24h shadow mode

**Rollback Triggers:**
- Error rate > 1% for 10 minutes
- Prediction MAE degrades > 0.5pts vs baseline
- P50 latency > 5s
- Memory usage > 800MB

---

## 8. Monitoring Tests

**File:** `backend/tests/test_monitoring.py`

```python
def test_health_check_endpoint():
    """Test /health returns correct status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'
    assert 'version' in data

def test_metrics_collection():
    """Test Prometheus-style metrics are collected."""
    from prometheus_client import REGISTRY

    # Check ML-specific metrics exist
    metrics_to_check = [
        'fpl_predictions_total',
        'fpl_prediction_latency_seconds',
        'fpl_optimization_duration_seconds',
        'fpl_understat_match_rate'
    ]

    for metric in metrics_to_check:
        try:
            metric_instance = REGISTRY.get_sample_value(metric)
            assert metric_instance is not None
        except KeyError:
            pytest.fail(f"Metric {metric} not registered")

def test_alert_thresholds():
    """Test alert thresholds are properly configured."""
    # Simulate high latency
    record_high_latency_metric(5.0)
    # Verify alert would fire
    assert check_alert('high_prediction_latency') is True
```

---

## 9. Test Data Management

### 9.1 Fixtures & Mock Data

**Directory:** `backend/tests/fixtures/`

- `players_sample.json` - 50 representative players
- `teams_sample.json` - All 20 teams
- `fixtures_sample.json` - 6 GWs of fixtures
- `understat_sample.json` - Mock Understat API responses

### 9.2 Synthetic Data Generation

**File:** `backend/scripts/generate_test_data.py`

```python
#!/usr/bin/env python3
"""
Generate synthetic test data for FPL ML system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random

def generate_players(n_players: int = 500) -> pd.DataFrame:
    """Generate synthetic player data."""
    positions = [
        (1, 'GK', 1),
        (2, 'DEF', 4),
        (3, 'MID', 8),
        (4, 'FWD', 3)
    ]

    players = []
    for i in range(n_players):
        pos_id, pos_name, pos_multiplier = random.choice(positions)
        team = random.randint(1, 20)

        # Generate realistic stats
        minutes = random.randint(0, 2700)
        games = minutes // 90 + random.randint(0, 5)

        player = {
            'id': i + 1,
            'web_name': f"Player_{i+1}",
            'element_type': pos_id,
            'position': pos_name,
            'team': team,
            'now_cost': random.randint(40, 170) + 10,  # In tenths
            'minutes': minutes,
            'starts': int(games * random.uniform(0.5, 0.95)),
            'goals_scored': int(minutes / 500 * random.uniform(0.3, 2.0) * pos_multiplier),
            'assists': int(minutes / 500 * random.uniform(0.1, 1.5) * pos_multiplier),
            'expected_goals': round(random.uniform(0, 20), 1),
            'expected_assists': round(random.uniform(0, 15), 1),
            'ict_index': round(random.uniform(0, 100), 1),
            'form': round(random.uniform(0, 10), 1),
            'ep_next': round(random.uniform(0, 8), 1),
            'chance_of_playing_next_round': random.choice([100, 100, 100, 100, 75, 50, 0]),
            'status': random.choice(['a', 'a', 'a', 'a', 'd', 'i', 'u']),
        }
        players.append(player)

    return pd.DataFrame(players)

def generate_teams() -> pd.DataFrame:
    """Generate 20 Premier League teams."""
    teams = [
        {'id': i+1, 'name': f"Team_{i+1}", 'short_name': f"T{i+1}"}
        for i in range(20)
    ]
    df = pd.DataFrame(teams)

    # Add strength ratings
    df['strength_attack_home'] = np.random.normal(1000, 100, 20)
    df['strength_attack_away'] = np.random.normal(950, 100, 20)
    df['strength_defence_home'] = np.random.normal(1000, 100, 20)
    df['strength_defence_away'] = np.random.normal(950, 100, 20)

    return df

def generate_fixtures(num_gameweeks: int = 10) -> List[Dict]:
    """Generate fixture list for all teams."""
    fixtures = []
    fixture_id = 1

    for gw in range(1, num_gameweeks + 1):
        # Shuffle teams each GW
        teams = list(range(1, 21))
        random.shuffle(teams)

        # Create pairwise matches
        for i in range(0, 20, 2):
            home = teams[i]
            away = teams[i + 1]

            # Double GW handling
            for is_double in [False]:
                fixture = {
                    'id': fixture_id,
                    'event': gw,
                    'team_h': home,
                    'team_a': away,
                    'team_h_difficulty': random.randint(2, 5),
                    'team_a_difficulty': random.randint(2, 5),
                    'finished': False,
                    'kickoff_time': f"2025-0{(gw%12)+1:02d}-{(i//2)%28+1:02d}T{15 + i//2}:00:00Z"
                }
                fixtures.append(fixture)
                fixture_id += 1

    return fixtures

def generate_test_suite():
    """Generate complete test data suite."""
    print("Generating synthetic test data...")

    players = generate_players(500)
    teams = generate_teams()
    fixtures = generate_fixtures(12)

    # Save to JSON fixtures
    os.makedirs('backend/tests/fixtures', exist_ok=True)

    players.to_json('backend/tests/fixtures/players_sample.json', orient='records')
    teams.to_json('backend/tests/fixtures/teams_sample.json', orient='records')

    with open('backend/tests/fixtures/fixtures_sample.json', 'w') as f:
        json.dump(fixtures, f, indent=2)

    # Generate golden dataset for regression testing
    golden_players = players.head(10)
    golden_players.to_json('backend/tests/fixtures/golden_players.json', orient='records')

    print(f"✅ Generated {len(players)} players")
    print(f"✅ Generated {len(teams)} teams")
    print(f"✅ Generated {len(fixtures)} fixtures")
    print("Test data saved to backend/tests/fixtures/")

if __name__ == "__main__":
    generate_test_suite()
```

### 9.3 Golden Datasets

**Golden datasets** are immutable reference outputs:
- Sample input → predictions stored as JSON
- Used to detect regression when code changes
- Version-controlled in git

---

## 10. Test Automation

### 10.1 CI/CD Configuration

**File:** `.github/workflows/test.yml`

```yaml
name: ML Tests

on:
  push:
    paths:
      - 'backend/**/*.py'
      - 'backend/tests/**'
  pull_request:
    paths:
      - 'backend/**/*.py'

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist pytest-benchmark
        pip install httpx  # For FastAPI TestClient

    - name: Generate test data
      run: |
        cd backend/scripts
        python generate_test_data.py

    - name: Run unit tests with coverage
      run: |
        cd backend
        pytest tests/ -v --cov=. --cov-report=xml --cov-report=html --cov-fail-under=80

    - name: Run integration tests
      run: |
        cd backend
        pytest tests/test_integration*.py -v --timeout=60

    - name: Run performance benchmarks
      run: |
        cd backend
        pytest tests/test_performance.py -v --benchmark-only

    - name: Upload coverage
      uses: actions/upload-artifact@v3
      with:
        name: coverage-report
        path: htmlcov/
```

### 10.2 Pre-commit Hooks

**File:** `.pre-commit-config.yaml`

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: bash -c 'cd backend && pytest tests/unit/ -x'
        language: system
        pass_filenames: false
        always_run: true
```

### 10.3 Nightly Regression Tests

**Schedule:** Daily at 2 AM UTC

```yaml
on:
  schedule:
    - cron: '0 2 * * *'

jobs:
  nightly-regression:
    runs-on: ubuntu-latest
    steps:
      - ... (checkout, install, generate data)
      - name: Run full test suite
        run: |
          cd backend
          pytest tests/ -v --benchmark-save=nightly
      - name: Compare benchmarks
        run: |
          pytest tests/ --benchmark-compare=nightly
```

---

## 11. Acceptance Criteria

### 11.1 Pre-Production Checklist

Before deploying ML features to production:

- [ ] **Unit Tests**: All pass, >80% code coverage
- [ ] **Integration Tests**: Full E2E pipeline passes
- [ ] **Validation**: Holdout MAE < 3.5, R² > 0.05
- [ ] **Bias Checks**: No significant position/team bias (p > 0.05)
- [ ] **Performance**: Single prediction <50ms, batch <2s, optimization <30s
- [ ] **Memory**: Peak usage <500MB
- [ ] **A/B Test**: ML outperforms baseline with statistical significance
- [ ] **Monitoring**: All metrics and alerts configured
- [ ] **Data Quality**: No gaps/errors in last 7 days of data
- [ ] **Documentation**: Updated API docs and model cards

### 11.2 Feature Flag Enablement

ML predictions gated behind `ML_ENABLED` feature flag:

```python
# In main.py
if os.getenv("ML_ENABLED", "false").lower() == "true":
    predictor = PointPredictor(...)
else:
    predictor = RuleBasedPredictor()  # Fallback
```

**Canary Rollout:**
1. Phase 1 (1% traffic): Shadow mode, log comparisons only
2. Phase 2 (10% traffic): Serve ML to subset, monitor errors
3. Phase 3 (50% traffic): Full comparison with rollback trigger
4. Phase 4 (100% traffic): All traffic on ML

### 11.3 Rollback Triggers

Automatic rollback to rule-based system when:

| Metric | Threshold | Duration |
|--------|-----------|----------|
| Error Rate | > 1% | 5 minutes |
| Prediction MAE degrades | > 0.5pts vs baseline | 30 minutes |
| P95 Latency | > 5 seconds | 10 minutes |
| Memory Usage | > 800MB | 5 minutes |
| Crash Rate | > 0.5% | 5 minutes |

---

## 12. Test Execution Strategy

### 12.1 Test Levels

```bash
# 1. Unit tests (fast)
pytest backend/tests/unit/ -v --cov=backend

# 2. Integration tests (medium)
pytest backend/tests/integration/ -v

# 3. Performance tests (slow)
pytest backend/tests/test_performance.py -v --benchmark-only

# 4. Full suite
pytest backend/tests/ -v --cov=. --cov-report=html

# 5. Specific module
pytest backend/tests/test_ml_predictor.py::test_predict_gameweek_returns_valid_points -v
```

### 12.2 Parallel Execution

```bash
# Use pytest-xdist for parallel test execution
pytest backend/tests/ -n auto  # Auto-detect CPU cores
pytest backend/tests/ -n 4     # Use 4 workers
```

### 12.3 Selective Testing

```bash
# Run only tests that passed previously (pytest --last-failed)
pytest --last-failed

# Run tests matching keyword
pytest -k "test_xg" -v

# Run tests excluding slow ones
pytest -m "not slow"
```

---

## 13. Coverage Requirements

| Component | Coverage Target | Critical Paths |
|-----------|----------------|----------------|
| point_predictor.py | >90% | All prediction methods |
| minutes_predictor.py | >85% | Start probability, rotation risk |
| ev_calculator.py | >90% | Distribution, simulation |
| understat_service.py | >80% | API integration, fallback |
| optimizer*.py | >75% | Constraint handling |
| main.py (API) | >70% | Endpoint coverage |

**Strict coverage enforcement:** CI fails if overall <80% or any critical module <70%

---

## 14. Test Maintenance

### 14.1 Golden Dataset Updates

Golden datasets must be updated only when:
- Model algorithm changes (not just hyperparameter tuning)
- Feature engineering changes
- Data schema changes

Procedure:
1. Run gold standard test case
2. Review changes manually
3. Commit with message: "Update golden dataset for [reason]"
4. Annotate in changelog

### 14.2 Test Data Refresh

- **Unit tests**: Synthetic data, regenerated daily via cron
- **Integration tests**: Pull fresh FPL data weekly
- **Golden datasets**: Version-controlled, only change intentionally

---

## Appendix A: Test Matrix Summary

| Test Type | Target | Tools | Frequency |
|-----------|--------|-------|-----------|
| Unit | 500+ tests | pytest, hypothesis | On every commit |
| Integration | 50+ scenarios | TestClient, fixtures | On every PR |
| Model Validation | Holdout sets | scikit-learn metrics | Daily |
| Performance | Benchmarks | pytest-benchmark | On merge to main |
| A/B Testing | ML vs Baseline | scipy.stats | Weekly |
| Monitoring | Health checks | prometheus-client | Every minute |

### Test Execution Timeline

| Test Category | Duration | Parallelizable |
|---------------|----------|----------------|
| Unit | ~1-2 minutes | Yes |
| Integration | ~5-10 minutes | Partially |
| Performance | ~3-5 minutes | Yes |
| Validation | ~10-15 minutes | Yes |
| **Total CI Time** | **~15-25 minutes** | - |

---

## Appendix B: Sample Test Scenarios

### Scenario 1: Model Degradation Detection

```
1. Nightly job runs holdout test on last 5 gameweeks
2. Computes MAE for each GW, tracks rolling average
3. If 3-day moving average MAE > 3.5pts:
   → Create GitHub issue
   → Send Slack alert to #ml-alerts
   → Disable ML feature flag automatically
```

### Scenario 2: Understat Outage

```
1. Understat API returns 500 error
2. PointPredictor logs warning, falls back to FPL xG
3. Metric 'understat_fallback_count' increments
4. If fallback rate > 20% for 1 hour:
   → Alert engineering team
   → Consider disabling Understat integration
```

### Scenario 3: Optimization Timeout

```
1. Multi-period optimization takes >45 seconds
2. Partial solution returned with 'partial_result': true
3. Metric 'optimization_timeouts' incremented
4. Alert: "Optimization performance degraded"
5. Consider reducing gameweek horizon or simplifying model
```

---

**End of Test Plan**

*This document is maintained by the FPL ML Integration team. Last reviewed: 2025-03-04.*
