# FPL ML Integration Technical Specification

**Version:** 1.0  
**Date:** 2025-03-04  
**Status:** Draft  
**Author:** Architect (FPL ML Integration Project)  

---

## 1. System Architecture Overview

### Architecture Diagram (Text Description)

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
│  (backend/main.py - Existing + New ML Endpoints)           │
├─────────────┬─────────────┬────────────┬───────────────────┤
│   FPL Data  │ Understat    │ Historical │   ML Services     │
│   Service   │   Service   │   Service  │   (NEW)           │
└─────────────┴─────────────┴────────────┴───────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   PostgreSQL     │
                    │   Database       │
                    │  (Features,      │
                    │   Predictions,   │
                    │   Model Registry)│
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  ML Training     │
                    │  Pipeline        │
                    │  (Batch)         │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   ML Models      │
                    │  ┌─────────────┐ │
                    │  │ XGBoost     │ │
                    │  ├─────────────┤ │
                    │  │  Neural Net  │ │
                    │  ├─────────────┤ │
                    │  │  Ensemble    │ │
                    │  └─────────────┘ │
                    └──────────────────┘
```

### Component Interactions

1. **Data Ingestion Layer** (Existing):
   - `FPLService`: Fetches real-time data from FPL API
   - `UnderstatService`: Fetches xG/xA statistics
   - `HistoricalDataService`: Reconstructs historical states for training

2. **Feature Engineering Layer** (NEW):
   - `FeatureEngineer`: Transforms raw data into ML features
   - Stores features in PostgreSQL for reuse
   - Caches computed features to avoid redundant computation

3. **ML Prediction Layer** (NEW):
   - `MLPredictionService`: Main service for ML-based predictions
   - `ModelManager`: Loads and manages multiple model versions
   - Provides fallback to rule-based `PointPredictor` when models unavailable

4. **Training Pipeline** (NEW):
   - `DataLoader`: Loads historical data, builds training dataset
   - `ModelTrainer`: Trains XGBoost and Neural Network models
   - `ModelEvaluator`: Validates models, computes metrics, compares to baseline

5. **Optimizer Integration**:
   - Existing optimizers will call ML services for predictions
   - Can switch between rule-based and ML-based predictions via config

---

## 2. Detailed Component Specifications

### 2.1 `backend/ml_predictor.py`

#### Classes

**`FeatureEngineer`**
- **Purpose**: Transform raw FPL/Understat data into ML features
- **Key Methods**:
  - `__init__(players_df, teams_df, fixtures, historical_data=None)`
  - `create_features(player_id, gameweek)` → `Dict[str, float]`
  - `create_training_dataset(start_gw, end_gw)` → `pd.DataFrame`
  - `_compute_rolling_stats(player_id, metric, window)` → `float`
  - `_get_fixture_difficulty(player_id, gw)` → `float`
  - `_compute_form_metrics(player_id, gw)` → `Dict`

- **Feature Categories**:
  1. **Player Profile**: `age`, `position`, `team`, `now_cost`
  2. **Recent Form** (rolling windows: 1, 3, 6 GWs):
     - `total_points_rolling_N`
     - `minutes_rolling_N`
     - `bps_rolling_N`
     - `goals_rolling_N`, `assists_rolling_N`
  3. **Expected Metrics** (from Understat, if available):
     - `xG_rolling_N`, `xA_rolling_N`, `xGI_rolling_N`
     - `xG_per_90`, `xA_per_90`
  4. **Fixture Analysis**:
     - `avg_fdr_next_N` (next 1, 3, 6 fixtures)
     - `home_game_next_N` (boolean)
     - `blank_gw_next_N`, `double_gw_next_N`
  5. **Team Context**:
     - `team_strength_attack`, `team_strength_defense`
     - `team_form_rolling_N`
  6. **Advanced Metrics**:
     - `ict_index_rolling_N`
     - `cbirt_eligible` (boolean for MIDs/FWDs)
     - `is_captain_candidate` (based on ownership trends)
  7. **Interaction Features**:
     - `cost_normalized_form` = `form_rolling_3 / (now_cost / 10)`
     - `xG_vs_price` = `xG_per_90 / (now_cost / 10)`
     - `fixture_weighted_form` = `form_rolling_3 * (6 - avg_fdr_next_3) / 6`

- **Output**: Feature vector of ~50-80 features per player per gameweek

**`ModelManager`**
- **Purpose**: Load, version, and manage ML models
- **Key Methods**:
  - `__init__(models_dir="models/production")`
  - `load_model(model_type, version="latest")` → `Model`
  - `list_available_models()` → `Dict[str, List[str]]`
  - `predict(features_dict, model_type="ensemble", gw=None)` → `Dict[int, float]`
  - `get_model_metrics(model_type, version)` → `Dict`
  - `reload_models()` (for hot-reloading after retrain)

- **Model Storage Structure**:
  ```
  models/production/
  ├── xgboost/
  │   ├── v1.0.0/
  │   │   ├── model.pkl
  │   │   ├── feature_names.json
  │   │   ├── metadata.json (hyperparams, train date, metrics)
  │   │   └── scaler.pkl (if needed)
  │   └── latest -> v1.0.0
  ├── neural_network/
  │   ├── v1.0.0/
  │   │   ├── model.pt (PyTorch) or model.h5 (Keras)
  │   │   ├── feature_names.json
  │   │   ├── metadata.json
  │   │   └── scaler.pkl
  │   └── latest -> v1.0.0
  └── ensemble/
      ├── v1.0.0/
      │   ├── weights.json ({"xgboost": 0.6, "neural_network": 0.4})
      │   └── metadata.json
      └── latest -> v1.0.0
  ```

- **Model Loading**: Lazy loading; only loads when first requested
- **Hot Reload**: Checks timestamp on `latest` symlink; reloads if changed

**`MLPredictionService`**
- **Purpose**: Main service interface for ML predictions (used by optimizers)
- **Key Methods**:
  - `__init__(use_ml=True, fallback_to_rule_based=True)`
  - `predict_all_players(players_df, teams_df, fixtures, current_gw, horizon=5)` → `pd.DataFrame`
  - `predict_player(player_id, gw)` → `Tuple[float, Dict]` (points, breakdown)
  - `predict_squad(squad_ids, gameweeks)` → `Dict`
  - `is_healthy()` → `bool` (check if models loaded)
  - `get_model_info()` → `Dict` (what models are active, versions)

- **Fallback Strategy**:
  - If ML models fail to load: log warning, fall back to `PointPredictor`
  - If prediction fails for a player: use rule-based prediction for that player
  - Monitor fallback frequency; alert if > 5%

- **Integration with Optimizers**:
  - `MultiPeriodFPLOptimizer` will call `MLPredictionService.predict_all_players()` 
  - Replace direct `PointPredictor` instantiation in optimizers

### 2.2 `backend/ml_training.py`

#### Classes

**`DataLoader`**
- **Purpose**: Prepare training data from historical FPL data
- **Key Methods**:
  - `__init__(historical_service, feature_engineer)`
  - `load_historical_dataset(start_season, end_season)` → `pd.DataFrame`
  - `get_features_labels(data)` → `Tuple[pd.DataFrame, pd.Series]`
  - `split_train_val_test(data, ratios=(0.7, 0.15, 0.15))` → `Dict[str, pd.DataFrame]`
  - `save_dataset(df, path)` / `load_dataset(path)`

- **Data Sources**:
  - `HistoricalDataService`: Per-player per-gameweek data
  - `UnderstatService`: xG/xA data (backfill from available seasons)
  - Static FPL data (player attributes, teams)

- **Target Variable**: `actual_points` (scored in that gameweek)
- **Dataset Construction**:
  - For each player-gameweek combination from historical seasons
  - Compute features using `FeatureEngineer` with historical context
  - Label = actual points scored
  - Filter out players with < 30 minutes (to reduce noise) OR explicitly mark as "partial"

- **Caching**: Store precomputed feature datasets to avoid recomputation
  - Location: `data/cache/training_datasets/`
  - Format: Parquet (fast, compressed)

- **Season Handling**:
  - Train on seasons 2023/24, 2024/25, validate on 2025/26 (current)
  - Or time-based split: train on GW 1-25, validate on GW 26-38, test on current

**`ModelTrainer`**
- **Purpose**: Train XGBoost and Neural Network models
- **Key Methods**:
  - `__init__(config)`
  - `train_xgboost(train_df, val_df, params=None)` → `Tuple[Model, Dict]`
  - `train_neural_network(train_df, val_df, params=None)` → `Tuple[Model, Dict]`
  - `cross_validation(df, n_folds=5)` → `Dict[str, List[float]]`
  - `save_model(model, path, metadata)` / `load_model(path)`

- **XGBoost Configuration**:
  ```python
  DEFAULT_XGB_PARAMS = {
      "objective": "reg:squarederror",
      "eval_metric": "rmse",
      "n_estimators": 500,
      "max_depth": 6,
      "learning_rate": 0.05,
      "subsample": 0.8,
      "colsample_bytree": 0.8,
      "min_child_weight": 1,
      "gamma": 0,
      "reg_alpha": 0,
      "reg_lambda": 1,
      "random_state": 42,
      "n_jobs": -1,
      "tree_method": "hist"  # For faster training
  }
  ```

- **Neural Network Architecture**:
  ```python
  class FPLPointPredictor(nn.Module):
      def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
          super().__init__()
          layers = []
          prev_dim = input_dim
          for hidden_dim in hidden_dims:
              layers.append(nn.Linear(prev_dim, hidden_dim))
              layers.append(nn.ReLU())
              layers.append(nn.BatchNorm1d(hidden_dim))
              layers.append(nn.Dropout(dropout))
              prev_dim = hidden_dim
          layers.append(nn.Linear(prev_dim, 1))
          self.network = nn.Sequential(*layers)
      
      def forward(self, x):
          return self.network(x)
  ```

- **Neural Network Training**:
  - Loss: `MSELoss()` (mean squared error)
  - Optimizer: `AdamW` (learning rate 1e-3, weight decay 1e-4)
  - Scheduler: `ReduceLROnPlateau` (factor 0.5, patience 10)
  - Early Stopping: patience 20 epochs on validation RMSE
  - Batch Size: 256
  - Max Epochs: 200

- **Training Process**:
  1. Load/prepare datasets
  2. Scale features (StandardScaler fit on train, transform all splits)
  3. Train XGBoost with early stopping (50 rounds)
  4. Train Neural Network with early stopping (20 epochs)
  5. Compute validation metrics for both
  6. Select better model OR prepare for ensemble

**`ModelEvaluator`**
- **Purpose**: Comprehensive evaluation and comparison
- **Key Methods**:
  - `__init__(metrics_dir="models/metrics")`
  - `evaluate_model(model, test_df)` → `Dict[str, float]`
  - `compare_to_baseline(predictions_df, baseline_predictions)` → `Dict`
  - `compute_feature_importance(model, feature_names)` → `pd.DataFrame`
  - `plot_residuals(predictions, actual)` → `matplotlib.Figure`
  - `generate_evaluation_report(model_metrics, baseline_metrics)` → `Dict`
  - `log_metrics(model_type, version, metrics)` → `None`

- **Evaluation Metrics**:
  - **Primary**: RMSE (Root Mean Squared Error)
  - **Secondary**: MAE, R², MAPE (Mean Absolute Percentage Error)
  - **Business-specific**: 
    - `top_10_accuracy`: % of top 10 predicted players in actual top 10
    - `position_accuracy`: Accuracy per position (GK, DEF, MID, FWD)
    - `value_above_threshold`: % of players with > 3pt error (anomalies)
    - `captain_accuracy`: Was the captain in top 5 predictions?

- **Baseline Comparison**:
  - Baseline 1: FPL's `ep_next` (Vancouver method)
  - Baseline 2: Rule-based `PointPredictor` from codebase
  - Target: ML model must beat baseline RMSE by > 5%

- **Residual Analysis**:
  - Plot predicted vs actual
  - Identify systematic biases (over/under-predicting certain positions)
  - Flag players with consistently high error for review

- **Feature Importance**:
  - XGBoost: Use `feature_importances_`
  - Neural Network: Use permutation importance or SHAP values
  - Report top 20 most important features

### 2.3 Database Schema (PostgreSQL)

#### Tables

**`players`** (static reference data)
```sql
CREATE TABLE players (
    id INTEGER PRIMARY KEY,              -- FPL element ID
    first_name VARCHAR(100),
    second_name VARCHAR(100),
    web_name VARCHAR(100) UNIQUE,
    team INTEGER REFERENCES teams(id),
    element_type INTEGER,                -- 1=GK, 2=DEF, 3=MID, 4=FWD
    now_cost DECIMAL(4,1),              -- In 10ths (e.g., 50 = £5.0m)
    total_points INTEGER,
    minutes INTEGER,
   goals_scored INTEGER,
    assists INTEGER,
    clean_sheets INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_players_team ON players(team);
CREATE INDEX idx_players_position ON players(element_type);
CREATE INDEX idx_players_cost ON players(now_cost);
```

**`teams`** (static reference data)
```sql
CREATE TABLE teams (
    id INTEGER PRIMARY KEY,             -- FPL team ID
    name VARCHAR(100),
    short_name VARCHAR(10) UNIQUE,
    strength_attack_home INTEGER,
    strength_attack_away INTEGER,
    strength_defence_home INTEGER,
    strength_defence_away INTEGER,
    code INTEGER,
    draw INTEGER,
    form VARCHAR(20),
    loss INTEGER,
    played INTEGER,
    points INTEGER,
    position INTEGER,
    win INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**`gameweeks`** (static reference data)
```sql
CREATE TABLE gameweeks (
    id INTEGER PRIMARY KEY,             -- GW number (1-38)
    is_current BOOLEAN DEFAULT FALSE,
    is_next BOOLEAN DEFAULT FALSE,
    deadline_time TIMESTAMP,
    finished BOOLEAN DEFAULT FALSE,
    finished_provisional BOOLEAN DEFAULT FALSE,
    name VARCHAR(50),
    chip_plays JSONB,                   -- {"freehit": {"num_played": 1000}, ...}
    element_stat JSONB,                 -- {"most_selected": {}, ...}
    top_element_info JSONB,             -- {"id": 123, "points": 100}
    most_captained INTEGER,
    most_vice_captained INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**`fixtures`** (dynamic, updated regularly)
```sql
CREATE TABLE fixtures (
    id INTEGER PRIMARY KEY,             -- FPL fixture ID
    event INTEGER REFERENCES gameweeks(id),  -- Gameweek
    team_h INTEGER REFERENCES teams(id),
    team_a INTEGER REFERENCES teams(id),
    team_h_score INTEGER,
    team_a_score INTEGER,
    team_h_difficulty FLOAT,
    team_a_difficulty FLOAT,
    fixture_type VARCHAR(20),           -- 'HOME', 'AWAY', 'NEUTRAL'
    is_home BOOLEAN,
    kickoff_time TIMESTAMP,
    finished BOOLEAN DEFAULT FALSE,
    started BOOLEAN DEFAULT FALSE,
    minutes INTEGER DEFAULT 0,
    provisional_minimum INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_fixtures_event ON fixtures(event);
CREATE INDEX idx_fixtures_team_h ON fixtures(team_h);
CREATE INDEX idx_fixtures_team_a ON fixtures(team_a);
CREATE INDEX idx_fixtures_finished ON fixtures(finished);
```

**`player_features`** (ML features - computed, stored for caching)
```sql
CREATE TABLE player_features (
    player_id INTEGER REFERENCES players(id),
    gameweek INTEGER REFERENCES gameweeks(id),
    feature_name VARCHAR(100),
    feature_value DECIMAL(12,6),
    feature_type VARCHAR(50),            -- 'profile', 'form', 'fixture', 'team', 'advanced'
    computation_timestamp TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (player_id, gameweek, feature_name)
);

CREATE INDEX idx_player_features_player_gw ON player_features(player_id, gameweek);
CREATE INDEX idx_player_features_feature ON player_features(feature_name);
```

**`ml_predictions`** (Stored predictions for audit/reproducibility)
```sql
CREATE TABLE ml_predictions (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    gameweek INTEGER REFERENCES gameweeks(id),
    model_type VARCHAR(50),              -- 'xgboost', 'neural_network', 'ensemble'
    model_version VARCHAR(50),
    predicted_points DECIMAL(6,2),
    confidence_interval_lower DECIMAL(6,2),  -- For ensemble uncertainty
    confidence_interval_upper DECIMAL(6,2),
    feature_importance JSONB,            -- Top features for this prediction
    prediction_timestamp TIMESTAMP DEFAULT NOW(),
    UNIQUE(player_id, gameweek, model_type, model_version)
);

CREATE INDEX idx_ml_predictions_player_gw ON ml_predictions(player_id, gameweek);
CREATE INDEX idx_ml_predictions_timestamp ON ml_predictions(prediction_timestamp);
CREATE INDEX idx_ml_predictions_model ON ml_predictions(model_type, model_version);
```

**`model_performance`** (Track model performance over time)
```sql
CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50),
    model_version VARCHAR(50),
    gameweek INTEGER REFERENCES gameweeks(id),
    rmse DECIMAL(6,4),
    mae DECIMAL(6,4),
    r2 DECIMAL(6,4),
    mape DECIMAL(6,4),
    top_10_accuracy DECIMAL(6,4),
    position_accuracy JSONB,             -- {"1": 0.72, "2": 0.68, ...}
    num_samples INTEGER,
    evaluation_timestamp TIMESTAMP DEFAULT NOW(),
    UNIQUE(model_type, model_version, gameweek)
);

CREATE INDEX idx_model_perf_model_version ON model_performance(model_type, model_version);
CREATE INDEX idx_model_perf_timestamp ON model_performance(evaluation_timestamp);
```

**`training_runs`** (Log training jobs)
```sql
CREATE TABLE training_runs (
    id SERIAL PRIMARY KEY,
    run_timestamp TIMESTAMP DEFAULT NOW(),
    model_type VARCHAR(50),
    model_version VARCHAR(50),
    start_season INTEGER,
    end_season INTEGER,
    num_samples INTEGER,
    hyperparameters JSONB,
    training_metrics JSONB,              -- {"train_rmse": 3.21, "val_rmse": 3.45, ...}
    val_metrics JSONB,
    test_metrics JSONB,
    feature_importance JSONB,
    training_duration_seconds INTEGER,
    status VARCHAR(20),                  -- 'completed', 'failed', 'partial'
    error_message TEXT,
    git_commit VARCHAR(40)
);
```

**`feature_importance`** (Persisted feature importance for analysis)
```sql
CREATE TABLE feature_importance (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50),
    model_version VARCHAR(50),
    feature_name VARCHAR(100),
    importance DECIMAL(12,6),
    importance_rank INTEGER,             -- 1 = most important
    computed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(model_type, model_version, feature_name)
);

CREATE INDEX idx_feature_imp_model_version ON feature_importance(model_type, model_version);
```

**`prediction_logs`** (For monitoring prediction drift)
```sql
CREATE TABLE prediction_logs (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    gameweek INTEGER REFERENCES gameweeks(id),
    model_type VARCHAR(50),
    model_version VARCHAR(50),
    predicted_points DECIMAL(6,2),
    fallback_used BOOLEAN DEFAULT FALSE,  -- Did we fall back to rule-based?
    fallback_reason VARCHAR(200),
    request_timestamp TIMESTAMP DEFAULT NOW(),
    response_time_ms INTEGER
);

CREATE INDEX idx_prediction_logs_timestamp ON prediction_logs(request_timestamp);
CREATE INDEX idx_prediction_logs_model_gw ON prediction_logs(model_type, gameweek);
-- Partition by month for large tables:
-- CREATE INDEX ... ON prediction_logs (DATE_TRUNC('month', request_timestamp));
```

### 2.4 Configuration & Environment Variables

**Config File: `config/ml_config.yaml`** (NEW)
```yaml
database:
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}
  name: ${DB_NAME:fpl_optimizer}
  user: ${DB_USER:fpl_user}
  password: ${DB_PASSWORD}
  pool_size: 20
  max_overflow: 30

models:
  paths:
    production: "models/production"
    training: "models/training"
    archive: "models/archive"
  refresh_interval_seconds: 3600  # Check for new models every hour
  fallback_to_rule_based: true
  prediction_timeout_seconds: 10   # Fail if prediction takes > 10s

training:
  enabled: ${TRAINING_ENABLED:false}
  schedule: "0 2 * * *"  # Daily at 2 AM UTC (cron)
  seasons: [2023, 2024, 2025]
  validation_split: 0.15
  test_split: 0.15
  xgboost_params:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
  neural_network:
    hidden_dims: [128, 64, 32]
    dropout: 0.3
    batch_size: 256
    max_epochs: 200
    early_stopping_patience: 20

features:
  compute_rolling_windows: [1, 3, 6]
  fixture_lookahead: 6
  min_minutes_for_training: 30
  use_understat: true
  use_historical_cache: true

monitoring:
  metrics_retention_days: 90
  alert_on_fallback_rate: 0.05  # Alert if > 5% of predictions use fallback
  alert_on_prediction_drift: true
  drift_threshold_rmse_increase: 0.5  # Alert if RMSE increases by 0.5+ pts
```

**Environment Variables**:
```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fpl_optimizer
DB_USER=fpl_user
DB_PASSWORD=secure_password_here

# ML Configuration
MODEL_ENV=production  # or 'staging', 'development'
ML_MODEL_VERSION=latest  # or specific version
TRAINING_ENABLED=false
FEATURE_COMPUTATION_STRATEGY=async  # 'async' or 'sync'

# Redis Cache (optional, for performance)
REDIS_URL=redis://localhost:6379/0
PREDICTION_CACHE_TTL=300  # 5 minutes

# Logging
LOG_LEVEL=INFO
ML_LOG_FILE=logs/ml_predictions.log

# Performance
ML_MAX_WORKERS=4  # Thread pool size for parallel predictions
PREDICTION_BATCH_SIZE=100  # Batch predictions for efficiency

# API
ENABLE_ML_ENDPOINTS=true
EXPOSE_MODEL_METRICS=true
```

---

## 3. API Design

### 3.1 New Endpoints

**`GET /api/ml/models`**  
**Purpose**: List available ML models and their versions  
**Response**:
```json
{
  "status": "ok",
  "models": {
    "xgboost": ["v1.0.0", "v0.9.0", "latest"],
    "neural_network": ["v1.0.0", "latest"],
    "ensemble": ["v1.0.0", "latest"]
  },
  "active": {
    "default": "ensemble",
    "versions": {
      "xgboost": "v1.0.0",
      "neural_network": "v1.0.0",
      "ensemble": "v1.0.0"
    }
  },
  "health": {
    "xgboost": "loaded",
    "neural_network": "loaded",
    "ensemble": "loaded"
  }
}
```

**`GET /api/ml/predictions`**  
**Purpose**: Get ML predictions for all players (replace/enhance existing `/api/predictions`)  
**Query Parameters**:
- `gameweeks` (default: 5)
- `model` (default: "ensemble", options: "xgboost", "neural_network", "ensemble")
- `version` (default: "latest")  
**Response**:
```json
{
  "current_gameweek": 25,
  "horizon": 5,
  "model": {
    "type": "ensemble",
    "version": "v1.0.0",
    "components": {
      "xgboost_weight": 0.6,
      "neural_network_weight": 0.4
    }
  },
  "predictions": [
    {
      "id": 123,
      "web_name": "Haaland",
      "position": "FWD",
      "team": "MCI",
      "expected_points": 8.7,
      "confidence_low": 7.2,
      "confidence_high": 10.1,
      "feature_breakdown": {
        "form_rolling_3": 2.1,
        "fixture_difficulty": 1.8,
        "xG_contribution": 2.5,
        "minutes_expected": 1.3,
        "total": 8.7
      },
      "model_contributions": {
        "xgboost": 8.9,
        "neural_network": 8.5
      }
    }
  ],
  "metadata": {
    "computation_time_ms": 2450,
    "num_players": 587,
    "fallback_count": 0
  }
}
```

**`GET /api/ml/player/{player_id}/prediction`**  
**Purpose**: Get prediction for specific player  
**Query Parameters**: `gameweek`, `model`, `version`  
**Response**:
```json
{
  "player_id": 123,
  "player_name": "Haaland",
  "gameweek": 25,
  "model": {
    "type": "ensemble",
    "version": "v1.0.0"
  },
  "prediction": {
    "expected_points": 8.7,
    "confidence_interval": [7.2, 10.1],
    "breakdown": {...},
    "feature_importance": {
      "xG_rolling_3": 0.25,
      "minutes_rolling_1": 0.18,
      "fixture_difficulty": 0.12,
      ...
    }
  }
}
```

**`GET /api/ml/metrics`** (Optional - for debugging)  
**Purpose**: Get recent model performance metrics  
**Query Parameters**: `model`, `version`, `days` (default: 7)  
**Response**:
```json
{
  "model": "xgboost",
  "version": "v1.0.0",
  "metrics": [
    {
      "gameweek": 24,
      "rmse": 3.21,
      "mae": 2.45,
      "r2": 0.185,
      "top_10_accuracy": 0.42,
      "samples": 587,
      "compared_to_baseline": {
        "rmse_improvement": 0.15,
        "r2_improvement": 0.025
      }
    }
  ]
}
```

**`POST /api/ml/retrain`** (Admin/CI only)  
**Purpose**: Trigger model retraining  
**Request Body**:
```json
{
  "model_types": ["xgboost", "neural_network"],
  "training_seasons": [2023, 2024],
  "validation_season": 2025,
  "hyperparameter_tuning": false,
  "notify_on_completion": true
}
```
**Response**:
```json
{
  "training_id": "run_20250304_020001",
  "status": "started",
  "estimated_duration_minutes": 120,
  "webhook_url": "/api/ml/training/run_20250304_020001/status"
}
```

**`GET /api/ml/training/{run_id}/status`** (Admin/CI only)  
**Purpose**: Check training job status  
**Response**:
```json
{
  "run_id": "run_20250304_020001",
  "status": "running",  // or 'completed', 'failed'
  "progress": {
    "stage": "training_neural_network",
    "percent_complete": 65,
    "current_epoch": 78,
    "best_val_rmse": 3.18
  },
  "started_at": "2025-03-04T02:00:01Z",
  "elapsed_seconds": 3540,
  "logs_url": "/api/ml/training/run_20250304_020001/logs"
}
```

### 3.2 Modified Existing Endpoints

**`/api/optimize` and `/api/optimize/multi-period`**  
**Change**: Internally use `MLPredictionService` instead of `PointPredictor`  
**Behavior**:  
- By default, use ML predictions if available  
- If ML service fails, log and fall back to `PointPredictor`  
- Add response header: `X-Model-Used: xgboost|neural_network|ensemble|fallback`  

**`/api/predictions`** (EXISTING)  
**Change**: Now wraps `MLPredictionService` but maintains backward compatibility  
- Response format enriched with ML-specific fields  
- If ML unavailable, transparently falls back to old method  
- Add query param: `?force_fallback=true` for testing  

**`/api/backtest`**  
**Change**: Option to backtest with ML predictions vs rule-based  
**New Query Params**:
- `use_ml_predictions` (default: true)
- `ml_model_version` (default: "latest")

### 3.3 Error Handling & Status Codes

**ML Service Errors**:
- `503 Service Unavailable`: ML models not loaded, falling back to rule-based (include `X-Fallback-Used: true` header)
- `504 Gateway Timeout`: Prediction took > 10s (configurable)
- `500 Internal Server Error`: Model inference failed (log details)
- `400 Bad Request`: Invalid model version specified

**Training Errors**:
- `409 Conflict`: Training already in progress for same model type
- `500 Internal Server Error`: Training failed (include error details in response)

**Monitoring Endpoints**:
- `200 OK`: Models healthy and loaded
- `503 Service Unavailable`: One or more models failed to load

---

## 4. Complete Data Flow

### 4.1 Training Pipeline Data Flow

```
[FPL API + Understat API]
         │
         ▼
┌──────────────────────────┐
│  HistoricalDataService   │  (Fetch per-GW history for all players)
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│   FeatureEngineer        │  (Compute features for each player-GW)
│   ────────────────────   │
│   • Rolling stats       │
│   • Fixture analysis    │
│   • Team context        │
│   • Understat enrich    │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│   Training Dataset (Parquet)     │
│   ────────────────────────────   │
│   columns: [features...]         │
│   rows: player_id, gw, label     │
└──────────────────────────────────┘
         │
         ├──────────────┬─────────────┐
         ▼              ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  XGBoost    │ │   Neural    │ │  Baseline   │
│  Trainer    │ │   Network   │ │  Comparison │
└─────────────┘ └─────────────┘ └─────────────┘
         │              │             │
         └──────────────┴─────────────┘
                        ▼
          ┌──────────────────────────┐
          │  ModelEvaluator          │
          │  • Compute metrics       │
          │  • Feature importance    │
          │  • Compare to baseline   │
          └──────────────────────────┘
                        │
                        ▼
          ┌──────────────────────────┐
          │  Model Registry          │
          │  • Save models           │
          │  • Update latest symlink │
          │  • Log to DB             │
          └──────────────────────────┘
                        │
                        ▼
          ┌──────────────────────────┐
          │  ModelManager Auto-Reload│
          │  (Hot reload)            │
          └──────────────────────────┘
```

### 4.2 Inference Pipeline Data Flow

```
┌─────────────────────────────────────────────┐
│  API Request: /api/optimize or /predictions│
└─────────────────────────────────────────────┘
                      │
                      ▼
          ┌──────────────────────────┐
          │  FPLService.get_latest_data() │
          └──────────────────────────┘
                      │
                      ▼
          ┌──────────────────────────┐
          │  FPL + Understat + Fixtures│
          └──────────────────────────┘
                      │
                      ▼
          ┌──────────────────────────┐
          │  FeatureEngineer (Lazy)  │
          │  • Check cached features │
          │  • Compute if missing   │
          │  • Cache to DB/Redis    │
          └──────────────────────────┘
                      │
                      ▼
          ┌──────────────────────────┐
          │  ModelManager            │
          │  • Load models (if not  │
          │    already loaded)      │
          │  • Select model type    │
          └──────────────────────────┘
                      │
                      ├─────────────┐
                      ▼             ▼
          ┌──────────────────────────┐
          │    XGBoost Inference     │
          │  (predict on feature DF) │
          └──────────────────────────┘
                      │
                      ├─────────────┐
                      ▼             ▼
          ┌──────────────────────────┐
          │   Neural Network Inference│
          │  (predict on feature DF)  │
          └──────────────────────────┘
                      │
                      └─────────────┘
                      ▼
          ┌──────────────────────────┐
          │    Ensemble Weights      │
          │   (0.6 * XGB + 0.4 * NN) │
          └──────────────────────────┘
                      │
                      ▼
          ┌──────────────────────────┐
          │  Fallback Check          │
          │  • NaN check             │
          │  • Outlier detection     │
          │  (Optional) fallback to  │
          │  rule-based if needed    │
          └──────────────────────────┘
                      │
                      ▼
          ┌──────────────────────────┐
          │   Optimizer Consumption  │
          │   (MultiPeriodFPNOptimizer)│
          └──────────────────────────┘
                      │
                      ▼
          ┌──────────────────────────┐
          │   API Response           │
          │   • Expected points      │
          │   • Confidence intervals │
          │   • Feature breakdown    │
          └──────────────────────────┘
```

---

## 5. Model Specifications

### 5.1 XGBoost Model

**Algorithm**: Gradient Boosted Decision Trees  
**Library**: `xgboost>=2.0.0`

**Hyperparameters** (Tuned via Optuna or Random Search):
```yaml
objective: reg:squarederror
eval_metric: rmse
n_estimators: 500 (range: 300-1000)
max_depth: 6 (range: 4-10)
learning_rate: 0.05 (range: 0.01-0.1)
subsample: 0.8 (range: 0.6-0.9)
colsample_bytree: 0.8 (range: 0.6-0.9)
min_child_weight: 1 (range: 1-5)
gamma: 0 (range: 0-0.2)
reg_alpha: 0 (L1 regularization)
reg_lambda: 1 (L2 regularization)
random_state: 42
n_jobs: -1
tree_method: hist  # Faster training
```

**Training Procedure**:
1. Prepare dataset: ~1M rows (players × gameweeks across seasons)
2. Train/val/test split: 70%/15%/15% (time-based)
3. Early stopping: 50 rounds, patience on validation RMSE
4. Feature scaling: Not needed for tree-based models
5. DMatrix construction for efficiency
6. Save model with `model.save_model('model.pkl')`

**Feature Importance Analysis**:
- Gain-based importance (default)
- Permutation importance (for final report)
- SHAP values for interpretability (monthly compute)

**Expected Performance**:
- RMSE: 3.0-3.5 points (0-plateau)
- Training time: ~10 minutes on 8-core CPU
- Inference time: <1ms per player (vectorized: <50ms for all)

**Model File Size**: ~50-100 MB

**Advantages**:
- Fast inference
- Robust to outliers
- Interpretable feature importance
- Handles mixed data types well

**Disadvantages**:
- May struggle with high-degree interactions
- Limited extrapolation

### 5.2 Neural Network Model

**Framework**: PyTorch >= 2.0 (or TensorFlow/Keras as alternative)  
**Architecture**:

```
Input Layer: [batch_size, num_features]
    │
    ▼
Linear( num_features → 128) + ReLU + BatchNorm + Dropout(0.3)
    │
    ▼
Linear( 128 → 64) + ReLU + BatchNorm + Dropout(0.3)
    │
    ▼
Linear( 64 → 32) + ReLU + BatchNorm + Dropout(0.3)
    │
    ▼
Linear( 32 → 1)  # Output: expected points
```

**Activation Functions**:
- Hidden layers: ReLU (faster training, sparse activations)
- Output layer: Linear (regression)

**Loss Function**: `MSELoss()` (Mean Squared Error)  
**Optimizer**: `AdamW` (lr=0.001, weight_decay=0.0001)  
**Learning Rate Scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=10)

**Training Configuration**:
```python
{
  "batch_size": 256,
  "max_epochs": 200,
  "early_stopping_patience": 20,
  "validation_fraction": 0.15,
  "shuffle": True,
  "num_workers": 4,
  "pin_memory": True,
  "gradient_clip_val": 1.0
}
```

**Regularization**:
- Dropout: 30% after each hidden layer
- Batch Normalization: Stabilizes training, faster convergence
- Weight decay (AdamW): 1e-4
- Early stopping: Prevents overfitting

**Feature Scaling**: Required  
- StandardScaler: `(x - mean) / std`
- Fit on training set only, apply to all splits
- Save scaler for inference

**Expected Performance**:
- RMSE: 2.9-3.4 points (similar to XGBoost, may beat slightly)
- Training time: ~30 minutes on GPU (RTX 3090), ~2h on CPU
- Inference time: <5ms per player (vectorized: ~200ms for all)

**Model File Size**: ~10-20 MB (weights only)

**Advantages**:
- Captures complex nonlinear interactions
- Can learn from feature embeddings
- Flexible architecture

**Disadvantages**:
- Slower training
- Sensitive to hyperparameters
- Less interpretable
- Requires careful regularization

### 5.3 Ensemble Model

**Strategy**: Weighted Average  
**Weights**: Determined by validation performance (RMSE inverse weighting)

Weights calculation:
```python
if val_rmse_xgb < val_rmse_nn:
    weight_xgb = 0.7
    weight_nn = 0.3
else:
    weight_xgb = 0.5
    weight_nn = 0.5
```

**Confidence Intervals**:
- Compute standard deviation between model predictions
- CI = `[ensemble - 1.96 * std, ensemble + 1.96 * std]` (95% CI)
- Wider CI indicates disagreement between models (higher uncertainty)

**Ensemble Weights Persistence**:
- Stored in `models/production/ensemble/v1.0.0/weights.json`
- Updated after each model retrain
- Dynamically adjusted based on recent performance (rolling window)

**Fallback Logic**:
- If XGBoost fails: Use Neural Network only (reweight to 100%)
- If Neural Network fails: Use XGBoost only
- If both fail: Fall back to `PointPredictor` (rule-based)

---

## 6. Performance Requirements

### 6.1 Inference Latency SLAs

**P50 (median)**:
- Single player prediction: < 5ms
- Full squad prediction (15 players): < 50ms
- All players prediction (587 players): < 500ms

**P95**:
- All players prediction: < 2 seconds

**P99** (including cold start model loading):
- All players prediction: < 5 seconds

**Cold Start** (first call after app restart):
- Model loading: < 2 seconds
- Contains model deserialization overhead

### 6.2 Memory Requirements

**ML Prediction Service**:
- Model in memory: XGBoost 100MB + Neural Network 20MB = 120MB
- Feature cache (PostgreSQL/Redis): ~100MB for all features
- Total: < 250MB per instance

**Training Pipeline**:
- Dataset in memory: ~5GB (1M rows × 80 features × float64)
- Model training: Additional 2-4GB
- Total: ~10GB RAM

### 6.3 Accuracy Targets

**Baseline**: Current `PointPredictor` (rule-based)  
**Target Improvement** (vs Baseline):

| Metric | Baseline Target | ML Target | Improvement |
|--------|-----------------|-----------|-------------|
| RMSE | 3.4 | 3.1 | 8.8% |
| MAE | 2.5 | 2.3 | 8% |
| R² | 0.15 | 0.22 | 46.7% |
| Top-10 Accuracy | 38% | 45% | 18.4% |
| Captain Accuracy (top-5) | 52% | 60% | 15.4% |

**Validation**:
- Test on held-out season (e.g., 2024/25) before deployment
- Backtest with historical predictions to estimate real-world ROI

### 6.4 Availability & Reliability

**Service Availability**: 99.9% uptime  
**Prediction Success Rate**: > 99.5% (exclude fallback)  
**Fallback Rate**: < 5% (alert if higher)  
**Mean Time Between Failures (MTBF)**: > 30 days  
**Mean Time To Recovery (MTTR)**: < 5 minutes (with auto-restart)

**Degraded Mode**:
- If ML service fails, automatically use `PointPredictor`
- Log fallback events for monitoring
- Alert on sustained high fallback rate

---

## 7. Deployment Plan

### 7.1 Deployment Strategy: Blue-Green with Canary

**Phase 0: Preparation** (Week 1)
- [ ] Set up PostgreSQL database (migration scripts prepared)
- [ ] Deploy `FeatureEngineer` as separate microservice (optional, or inline)
- [ ] Deploy model training pipeline in CI/CD
- [ ] Train initial models on historical data (2023-2024 seasons)
- [ ] Validate models against baseline (must beat by >5% RMSE)
- [ ] Prepare rollback scripts

**Phase 1: Shadow Deployment** (Week 2)
- Deploy ML models to production environment
- **Do NOT enable in API yet**
- Run predictions in parallel with existing system (log to separate file)
- Compare ML predictions vs rule-based predictions
- Monitor: inference latency, memory, prediction distribution
- **No user impact** - just logging

**Phase 2: Canary Release** (Week 3)
- Enable ML predictions for 1% of users (random selection)
- Add response header `X-Model-Version: xgboost-v1.0.0` for tracking
- Monitor: error rates, fallback rate, latency
- Compare optimization results between ML and rule-based users
- **Rollback triggers**:
  - Fallback rate > 10%
  - P95 latency > 2 seconds
  - User complaints about quality drop

**Phase 3: Gradual Ramp** (Week 4)
- If canary passes, increase to 10%, then 25%, then 50%, then 100%
- At each step, monitor for 24-48 hours before proceeding
- Use feature flag `ENABLE_ML_PREDICTIONS` for quick rollback

**Phase 4: Full Rollout** (Week 5)
- 100% traffic to ML predictions
- Decommission or deprecate `PointPredictor` (keep as fallback)

**Phase 5: Optimization** (Week 6+)
- Monitor drift: compare model predictions vs actual results weekly
- Retrain monthly with new data
- Automated retraining pipeline (CI)

### 7.2 Rollback Procedures

**Instant Rollback (Feature Flag)**:
```bash
# Via environment variable (takes effect on next request)
heroku config:set ENABLE_ML_PREDICTIONS=false
# Or in Kubernetes:
kubectl set env deployment/fpl-api ENABLE_ML_PREDICTIONS=false
```

**Database Rollback** (if schema changes needed):
```bash
# Apply down migration
alembic downgrade -1
```

**Model Rollback** (hot reload):
```bash
# Change symlink to previous version
ln -sfn models/production/xgboost/v0.9.0 models/production/xgboost/latest
# ModelManager will auto-reload on next request
```

**Full Rollback** (emergency):
1. Disable ML endpoints: `EXPOSE_ML_ENDPOINTS=false`
2. Restore code to previous commit (before ML integration)
3. Deploy old version

### 7.3 Database Migration

**Migration Strategy**: Zero-downtime, backward-compatible

**Step 1**: Add new tables first (non-blocking):
```sql
-- All CREATE TABLE statements are non-blocking
-- No data needed for startup
```

**Step 2**: Deploy code that writes to new tables but reads from old sources
**Step 3**: Backfill data (if needed) via one-time script
**Step 4**: Update read paths to new tables
**Step 5**: After validation, deprecate old tables (if any)

**Rollback**: All changes are additive; dropping new tables is safe.

### 7.4 Monitoring During Deployment

**Key Metrics to Watch**:
- Request latency P50/P95-P99
- Error rate (5xx responses)
- Fallback rate header count
- Model prediction distribution (mean, std, should be stable)
- Database connection pool usage
- Memory usage per pod/container

**Dashboard**: Grafana dashboard with:
- Real-time latency
- Error rates by endpoint
- Model version distribution
- Prediction confidence distribution
- Feature importance drift (compare week-over-week)

---

## 8. Monitoring & Observability

### 8.1 Application Metrics (Prometheus/StatsD)

**Counter Metrics**:
- `ml_predictions_total{model, version}`: Count of predictions served
- `ml_fallbacks_total{reason}`: Count of fallbacks (rule-based used instead)
- `ml_errors_total{type}`: Count of prediction errors
- `ml_requests_duration_seconds{model, quantile}`: Request latency histogram
- `model_versions_active{type}`: Currently loaded model versions

**Gauge Metrics**:
- `ml_models_loaded{type}`: 1 if loaded, 0 if not
- `ml_prediction_cache_hit_rate`: % of predictions served from cache
- `ml_feature_computation_time_ms`: Time to compute features
- `ml_inference_time_ms{model}`: Model inference latency

**Histogram Metrics**:
- `ml_prediction_value_range{model}`: Distribution of predicted points
- `ml_confidence_interval_width`: Width of ensemble CI

**Business Metrics** (for ROI tracking):
- `optimization_results_total{strategy}`: Count of optimizations run
- `optimization_duration_seconds`: Time to run optimization
- `feature_importance_top{feature_name}`: Track importance changes

### 8.2 Logging

**Structured JSON Logs** (via `structlog` or `python-json-logger`):
```json
{
  "timestamp": "2025-03-04T10:30:00Z",
  "level": "INFO",
  "component": "ml_prediction",
  "message": "Prediction completed",
  "model": "ensemble",
  "version": "v1.0.0",
  "player_id": 123,
  "gameweek": 25,
  "predicted_points": 8.7,
  "inference_time_ms": 3,
  "total_time_ms": 25,
  "fallback_used": false,
  "request_id": "uuid"
}
```

**Log Levels**:
- `DEBUG`: Feature values for individual players (sampled)
- `INFO`: High-level operations, errors, fallbacks
- `WARNING`: High fallback rate, prediction anomalies
- `ERROR`: Model loading failures, training crashes

**Log Aggregation**: Fluentd → Elasticsearch/Splunk → Kibana/Graylog

### 8.3 Tracing

**Distributed Tracing** (OpenTelemetry):
- Trace from API request through prediction pipeline
- Span for: feature engineering, model loading, inference, ensemble
- Export to Jaeger/Tempo for latency analysis

**SLA Tracking**:
- Alert if P95 inference > 2 seconds
- Alert if fallback rate > 5%
- Alert if model health check fails

### 8.4 Alerts (PagerDuty/Opsgenie)

**Critical** (Page):
- `ml_models_loaded{type} == 0` for > 5 minutes
- Error rate > 1% for 5 minutes
- Fallback rate > 20% for 10 minutes

**Warning** (Notify):
- Fallback rate > 5% for 30 minutes
- P95 latency > 2 seconds for 1 hour
- Prediction value distribution shift > 2σ from baseline
- Model performance drift (RMSE increase > 0.5)

**Info**:
- New model deployed
- Retraining job completed
- Feature importance changes detected

### 8.5 Model Monitoring (Evidently AI or Custom)

**Data Drift Detection**:
- Monitor feature distributions (mean, std, percentiles)
- Compare training data distribution vs current inference data
- Alert if KL divergence > threshold (e.g., 0.1)

**Performance Degradation**:
- Track actual vs predicted after each gameweek
- Compute rolling RMSE over last 10 gameweeks
- Alert if rolling RMSE exceeds baseline by > 10%

**Prediction Calibration**:
- Ensure predictions are well-calibrated (higher predicted → higher actual on average)
- Plot calibration curve weekly
- Recalibrate models if bias detected (systematic under/over-prediction)

**Feature Importance Drift**:
- Compute SHAP values on recent predictions
- Compare to historical average feature importance
- Alert if top features change significantly (e.g., `xG` replaced by `minutes`)

---

## 9. Risk Analysis & Mitigation

### 9.1 Technical Risks

**Risk 1: Model Predictions Are Worse Than Rule-Based**  
**Probability**: Medium  
**Impact**: High (user dissatisfaction, worse optimization results)  
**Mitigation**:
- Rigorous A/B testing before full rollout (canary phase)
- Continue using rule-based as fallback
- Monitor performance metrics daily
- Quick rollback mechanism via feature flag

**Risk 2: High Inference Latency**  
**Probability**: Low-Medium  
**Impact**: Medium (slow API responses, poor UX)  
**Mitigation**:
- Set strict latency SLAs with timeouts
- Batch predictions (predict all players at once instead of per-player)
- Cache feature vectors (avoid recomputation)
- Use optimized model formats (ONNX for neural net)
- Horizontal scaling: add more API pods

**Risk 3: Database Bottleneck** (feature cache, prediction logs)  
**Probability**: Medium  
**Impact**: Medium-High  
**Mitigation**:
- Add Redis cache in front of PostgreSQL for feature reads
- Use connection pooling (20-30 connections)
- Partition `prediction_logs` by month
- Implement async writes for logging
- Set up read replicas if needed

**Risk 4: Model Training Fails or Overfits**  
**Probability**: Medium  
**Impact**: Medium (delays, wasted effort)  
**Mitigation**:
- Start with simple models (XGBoost) before neural networks
- Use proper validation splits (time-based, not random)
- Early stopping and regularization
- Cross-validation to detect overfitting
- Have baseline model as fallback

**Risk 5: Drift (Player Meta Changes, Rule Changes)**  
**Probability**: High  
**Impact**: High (model degradation over time)  
**Mitigation**:
- Monthly retraining with latest data
- Monitor prediction distributions
- Implement automated drift detection
- Keep `PointPredictor` as fallback indefinitely

**Risk 6: Data Quality Issues** (missing Understat, corrupted cache)  
**Probability**: Medium  
**Impact**: Medium  
**Mitigation**:
- Validate data at ingestion
- Fallback to FPL data if Understat unavailable
- Checksums for cached files
- Alert on high fallback rate

**Risk 7: Memory Leaks or Resource Exhaustion**  
**Probability**: Low  
**Impact**: High (service crash)  
**Mitigation**:
- Set memory limits in containerization (Docker/K8s)
- Implement periodic model cache clearing (optional)
- Monitor memory usage with alerts
- Use async/await properly to avoid blocking

**Risk 8: Incompatible Model Versions** (after retraining)  
**Probability**: Low  
**Impact**: Medium  
**Mitigation**:
- Version all models with semantic versioning
- Test new models in staging before prod
- Keep old versions available for quick rollback
- Use symlinks and atomic updates for `latest`

**Risk 9: Cold Start Performance**  
**Probability**: High  
**Impact**: Medium  
**Mitigation**:
- Pre-warm models on startup (load at container init)
- Use lazy loading but trigger on first request with background loading
- Keep models in memory (don't unload)

**Risk 10: Security Vulnerabilities** (pickle exploits, code injection)  
**Probability**: Low  
**Impact**: Critical  
**Mitigation**:
- Never load untrusted model files
- Use signed model files (checksum verification)
- Scan dependencies for CVEs regularly
- Restrict file system permissions
- Use ONNX format (safer than pickle) for neural networks

### 9.2 Operational Risks

**Risk 1: Training Pipeline Complexity**  
**Mitigation**: Start with batch training (manual), automate later

**Risk 2: Team Knowledge Gap** (ML expertise)  
**Mitigation**: Train team, involve ML specialist, use simple models initially

**Risk 3: Vendor Lock-in** (Understat API, FPL API changes)  
**Mitigation**: Abstract data sources behind interfaces, monitor API changes

---

## 10. Implementation Phases & Milestones

### Phase 1: Foundation (Weeks 1-2)
- [ ] Database schema design and migrations
- [ ] `FeatureEngineer` implementation (core features)
- [ ] Feature caching layer (DB + optional Redis)
- [ ] XGBoost baseline model training
- [ ] Validation framework (metrics, comparison to baseline)

**Milestone 1**: Can reproduce baseline `PointPredictor` with ML features, train XGBoost to >5% improvement.

### Phase 2: Model Development (Weeks 3-4)
- [ ] XGBoost hyperparameter tuning (Optuna)
- [ ] Neural Network implementation and training
- [ ] Ensemble strategy development
- [ ] Feature importance analysis
- [ ] Model serialization and versioning

**Milestone 2**: ML ensemble beats baseline by >8% RMSE, load times <500ms for all players.

### Phase 3: Service Integration (Weeks 5-6)
- [ ] `MLPredictionService` and `ModelManager` implementation
- [ ] Feature cache warm-up (precompute for all players)
- [ ] Integration with `MultiPeriodFPLOptimizer`
- [ ] Fallback mechanism implementation
- [ ] Shadow deployment (log ML predictions alongside)

**Milestone 3**: ML predictions ready for canary, latency SLAs met, fallback mechanism verified.

### Phase 4: Deployment & Monitoring (Weeks 7-8)
- [ ] Canary deployment (1% → 25% → 100%)
- [ ] Monitoring dashboards (Grafana, logs)
- [ ] Alert rules setup
- [ ] Training pipeline automation (CI/CD)
- [ ] Documentation and handoff to ops

**Milestone 4**: 100% traffic on ML predictions, stable for 7 days, automated retraining pipeline working.

### Phase 5: Optimization (Weeks 9+)
- [ ] Model retraining (monthly cadence)
- [ ] A/B testing different architectures
- [ ] Feature engineering enhancements
- [ ] Performance tuning (batch inference, quantization)
- [ ] Advanced ensemble methods (stacking, blending)

---

## 11. Success Criteria

### Definition of Done (DoD)

1. **Model Performance**:
   - RMSE improvement: > 8% vs baseline `PointPredictor`
   - Top-10 accuracy: > 42%
   - Validation on out-of-sample data (held-out season)

2. **Performance**:
   - P95 prediction latency: < 2 seconds for all players
   - Memory usage: < 300MB per pod
   - Cold start: < 3 seconds

3. **Reliability**:
   - Prediction success rate: > 99.5%
   - Fallback rate: < 5%
   - Zero-downtime deployment verified

4. **Observability**:
   - Key metrics dashboard operational
   - Alerts configured and tested
   - Structured logging in place

5. **Code Quality**:
   - Unit tests: > 80% coverage for ML modules
   - Integration tests: full pipeline from features to predictions
   - Type hints and documentation complete

6. **Documentation**:
   - Technical spec (this document)
   - API documentation updated
   - Deployment runbook
   - Monitoring guide

7. **Rollback Ready**:
   - Feature flag tested
   - Database rollback script validated
   - Model rollback procedure documented

### Product Success Metrics

- **User Engagement**: Increase in active users (track through anonymized IDs)
- **Satisfaction**: Positive user feedback on prediction quality
- **Business Impact**: Increased conversions (paid tier upsells)
- **Stability**: No increase in support tickets related to predictions

---

## 12. Appendices

### Appendix A: Feature Dictionary (Detailed)

| Feature Name | Type | Description | Computation |
|-------------|------|-------------|-------------|
| `player_id` | int | FPL element ID | Static |
| `gameweek` | int | Target gameweek | Static |
| `age` | int | Player age | Static |
| `position` | cat | 1=GK, 2=DEF, 3=MID, 4=FWD | Static |
| `now_cost` | float | Current price in £m | Static (from that GW) |
| `form_rolling_1` | float | Total points previous GW | `total_points` from previous GW |
| `form_rolling_3` | float | Mean total points last 3 GWs | Rolling mean |
| `form_rolling_6` | float | Mean total points last 6 GWs | Rolling mean |
| `minutes_rolling_1` | float | Minutes played previous GW | `minutes` from previous GW |
| `minutes_rolling_3` | float | Mean minutes last 3 GWs | Rolling mean |
| `bps_rolling_3` | float | Mean BPS last 3 GWs | Rolling mean of `bps` |
| `goals_rolling_3` | float | Mean goals last 3 GWs | Rolling mean |
| `assists_rolling_3` | float | Mean assists last 3 GWs | Rolling mean |
| `xG_rolling_3` | float | Mean xG last 3 matches | Understat + rolling |
| `xA_rolling_3` | float | Mean xA last 3 matches | Understat + rolling |
| `xGI_rolling_3` | float | Mean xGI = xG+xA | Derived |
| `xG_per_90` | float | xG per 90 minutes | Understat aggregate |
| `xA_per_90` | float | xA per 90 minutes | Understat aggregate |
| `avg_fdr_next_1` | float | Avg fixture difficulty next GW | FDR from fixtures table |
| `avg_fdr_next_3` | float | Avg FDR next 3 GWs | Weighted sum |
| `home_game_next` | bool | Is next game at home? | Fixture lookup |
| `blank_gw_next_3` | int | Count of blank GWs in next 3 | Fixture analysis |
| `double_gw_next_3` | int | Count of double GWs in next 3 | Fixture analysis |
| `team_strength_attack` | int | Attack strength rating | Teams table |
| `team_strength_defense` | int | Defense strength rating | Teams table |
| `ict_index_rolling_3` | float | Mean ICT index last 3 GWs | Rolling mean of `ict_index` |
| `cbirt_eligible` | bool | MID/FWD with CBIRT ≥ 12? | `cbirt` >= threshold |
| `is_captain_candidate` | bool | In top 5 ownership? | Ownership data |
| `cost_normalized_form` | float | `form_3 / (cost/10)` | Interaction |
| `xG_vs_price` | float | `xG_per_90 / (cost/10)` | Interaction |
| `fixture_weighted_form` | float | `form_3 * (6-avg_fdr_3)/6` | Interaction |
| `is_home_next` | bool | Binary home advantage | Fixture |
| `team_form_rolling_3` | float | Team's recent form | Team-level rolling |
| `opponent_strength` | float | Avg opponent strength | From fixtures |

### Appendix B: Training Dataset Example

```csv
player_id,gameweek,age,position,now_cost,form_rolling_3,minutes_rolling_3,xG_rolling_3,avg_fdr_next_3,home_game_next,...,actual_points
123,25,28,3,6.5,4.2,75,0.8,4.2,1,...,8.7
123,26,28,3,6.5,5.1,82,1.2,3.8,0,...,9.2
456,25,24,2,5.0,3.1,80,0.4,4.5,1,...,4.8
...
```

### Appendix C: Model Comparison Framework

After each model train, populate:

```python
{
  "model_type": "xgboost",
  "version": "v1.0.0",
  "train_date": "2025-03-01",
  "hyperparameters": {...},
  "dataset": {
    "num_samples": 987654,
    "seasons": [2023, 2024],
    "features": 78
  },
  "metrics": {
    "train": {"rmse": 2.95, "mae": 2.28, "r2": 0.22},
    "val": {"rmse": 3.12, "mae": 2.41, "r2": 0.19},
    "test": {"rmse": 3.08, "mae": 2.38, "r2": 0.20}
  },
  "baseline_comparison": {
    "baseline_rmse": 3.34,
    "improvement_pct": 7.8,
    "is_deployable": true
  },
  "feature_importance": [
    {"feature": "minutes_rolling_3", "importance": 0.15},
    {"feature": "xG_rolling_3", "importance": 0.12},
    ...
  ]
}
```

### Appendix D: Database Connection Pool Configuration

```python
# SQLAlchemy config
SQLALCHEMY_DATABASE_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
SQLALCHEMY_ENGINE_OPTIONS = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_pre_ping": True,  # Check connection health
    "pool_recycle": 3600,   # Recycle connections after 1h
    "echo_pool": False,
    "future": True
}
```

### Appendix E: Docker/K8s Configuration Snippet

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY models/production/ ./models/production/
COPY config/ml_config.yaml ./config/

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# K8s deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fpl-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: fpl-optimizer:latest
        resources:
          limits:
            memory: "1Gi"
            cpu: "1000m"
        env:
        - name: ENABLE_ML_PREDICTIONS
          value: "true"
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
```

---

## 13. References & Further Reading

- FPL API Documentation: https://fantasy.premierleague.com/api/
- XGBoost Parameters: https://xgboost.readthedocs.io/en/stable/parameter.html
- PyTorch Neural Networks: https://pytorch.org/docs/stable/nn.html
- Understat API: https://github.com/amosbastian/understatapi
- Evidently AI for drift detection: https://evidently.ai/
- Prometheus metrics: https://prometheus.io/docs/practices/instrumentation/

---

**End of Technical Specification Document v1.0**

*For questions or clarifications, contact the architecture team.*
