-- ML Integration Database Schema
-- Version: 1.0
-- Created: 2025-03-04

-- Enable UUID extension if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- REFERENCE DATA TABLES
-- ============================================================================

-- Teams table (from FPL API)
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    short_name VARCHAR(10) NOT NULL,
    code INTEGER,
    strength INTEGER,
    strength_overall_home INTEGER,
    strength_overall_away INTEGER,
    strength_attack_home INTEGER,
    strength_attack_away INTEGER,
    strength_defence_home INTEGER,
    strength_defence_away INTEGER,
    pulse_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_teams_name ON teams(name);
CREATE INDEX idx_teams_short_name ON teams(short_name);

-- Players table (static FPL player data)
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,              -- FPL element ID
    first_name VARCHAR(100) NOT NULL,
    second_name VARCHAR(100) NOT NULL,
    web_name VARCHAR(100) NOT NULL,
    team INTEGER REFERENCES teams(id),
    element_type INTEGER NOT NULL,       -- 1=GK, 2=DEF, 3=MID, 4=FWD
    now_cost INTEGER NOT NULL,           -- In 10ths (e.g., 50 = £5.0m)
    total_points INTEGER DEFAULT 0,
    minutes INTEGER DEFAULT 0,
    goals_scored INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    clean_sheets INTEGER DEFAULT 0,
    goals_conceded INTEGER DEFAULT 0,
    own_goals INTEGER DEFAULT 0,
    penalties_saved INTEGER DEFAULT 0,
    penalties_missed INTEGER DEFAULT 0,
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,
    saves INTEGER DEFAULT 0,
    bonus INTEGER DEFAULT 0,
    bps INTEGER DEFAULT 0,
    selected_by_percent DECIMAL(5,2) DEFAULT 0.0,
    form DECIMAL(5,2) DEFAULT 0.0,
    ep_next DECIMAL(5,2) DEFAULT 0.0,    -- Expected points next GW
    ep_this DECIMAL(5,2) DEFAULT 0.0,    -- Expected points current GW
    status VARCHAR(10) DEFAULT 'a',      -- 'a'=available, 'u'=unavailable, 'i'=injured, 's'=suspended
    fixture_difficulty DECIMAL(3,2),     -- Avg FDR over next N fixtures
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_players_id ON players(id);
CREATE INDEX idx_players_team ON players(team);
CREATE INDEX idx_players_position ON players(element_type);
CREATE INDEX idx_players_cost ON players(now_cost);
CREATE INDEX idx_players_web_name ON players(web_name);
CREATE INDEX idx_players_status ON players(status);

-- ============================================================================
-- FEATURE STORE TABLES
-- ============================================================================

-- Player features for ML training/serving
-- One row per player per gameweek with computed features
CREATE TABLE IF NOT EXISTS players_features (
    id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    gameweek INTEGER NOT NULL,
    season VARCHAR(10) NOT NULL,         -- e.g., '2025-26'
    feature_version INTEGER DEFAULT 1,
    
    -- Basic features
    now_cost INTEGER NOT NULL,
    element_type INTEGER NOT NULL,
    team INTEGER NOT NULL,
    
    -- Rolling statistics (last N gameweeks)
    total_points_rolling_1 DECIMAL(5,2),
    total_points_rolling_3 DECIMAL(5,2),
    total_points_rolling_6 DECIMAL(5,2),
    minutes_rolling_1 DECIMAL(5,2),
    minutes_rolling_3 DECIMAL(5,2),
    minutes_rolling_6 DECIMAL(5,2),
    goals_scored_rolling_1 DECIMAL(5,2),
    goals_scored_rolling_3 DECIMAL(5,2),
    assists_rolling_1 DECIMAL(5,2),
    assists_rolling_3 DECIMAL(5,2),
    clean_sheets_rolling_1 DECIMAL(5,2),
    clean_sheets_rolling_3 DECIMAL(5,2),
    
    -- Fixture difficulty
    avg_fdr_next_3 DECIMAL(3,2),         -- Average fixture difficulty rating for next 3 GWs
    avg_fdr_next_6 DECIMAL(3,2),
    dgw_next_3 INTEGER DEFAULT 0,        -- Count of double gameweeks in next 3
    dgw_next_6 INTEGER DEFAULT 0,
    bgw_next_3 INTEGER DEFAULT 0,        -- Count of blank gameweeks in next 3
    
    -- Team form
    team_points_rolling_3 INTEGER,
    team_goals_rolling_3 INTEGER,
    team_conceded_rolling_3 INTEGER,
    
    -- Player availability
    is_injured BOOLEAN DEFAULT FALSE,
    is_suspended BOOLEAN DEFAULT FALSE,
    injury_risk_score DECIMAL(3,2),       -- 0-1 probability of missing
    
    -- Advanced features
    form_trend DECIMAL(5,2),              -- Slope of points over last 5 GWs
    points_per_minute DECIMAL(5,3),
    value_ratio DECIMAL(5,2),             -- Points per £10m over last 5 GWs
    ownership_factor DECIMAL(5,2),        -- Selected by % (0-100)
    
    -- Interaction features
    home_advantage BOOLEAN,               -- Is next fixture home?
    opp_strength DECIMAL(3,2),            -- Opponent defensive strength
    
    -- Target variable (actual points scored in that GW)
    actual_points INTEGER,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(player_id, gameweek, season)
);

CREATE INDEX idx_features_player_gw ON players_features(player_id, gameweek);
CREATE INDEX idx_features_season ON players_features(season);
CREATE INDEX idx_features_gw ON players_features(gameweek);
CREATE INDEX idx_features_actual_points ON players_features(actual_points);
CREATE INDEX idx_features_feature_version ON players_features(feature_version);

-- Feature version tracking
CREATE TABLE IF NOT EXISTS feature_versions (
    version INTEGER PRIMARY KEY,
    description TEXT,
    feature_list JSONB,                   -- List of feature columns in this version
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

INSERT INTO feature_versions (version, description, feature_list) VALUES 
(1, 'Initial feature set', '{"features": ["now_cost", "element_type", "team", "total_points_rolling_1", "total_points_rolling_3", "total_points_rolling_6", "minutes_rolling_1", "minutes_rolling_3", "minutes_rolling_6", "goals_scored_rolling_1", "goals_scored_rolling_3", "assists_rolling_1", "assists_rolling_3", "clean_sheets_rolling_1", "clean_sheets_rolling_3", "avg_fdr_next_3", "avg_fdr_next_6", "dgw_next_3", "dgw_next_6", "bgw_next_3", "team_points_rolling_3", "team_goals_rolling_3", "team_conceded_rolling_3", "is_injured", "is_suspended", "injury_risk_score", "form_trend", "points_per_minute", "value_ratio", "ownership_factor", "home_advantage", "opp_strength"]}')
ON CONFLICT (version) DO NOTHING;

-- ============================================================================
-- MODEL MANAGEMENT TABLES
-- ============================================================================

-- Model registry (tracks trained models)
CREATE TABLE IF NOT EXISTS model_registry (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,     -- e.g., 'xgboost_points', 'neural_net_points'
    model_version VARCHAR(20) NOT NULL,   -- Semantic version: '1.0.0'
    model_type VARCHAR(50) NOT NULL,      -- 'xgboost', 'neural_network', 'ensemble'
    
    -- Storage
    artifact_path TEXT NOT NULL,          -- Path to model artifact (S3 or local)
    feature_version INTEGER REFERENCES feature_versions(version),
    
    -- Metadata
    hyperparameters JSONB,
    training_dataset_hash CHAR(64),       -- SHA256 of training data
    training_start_time TIMESTAMP,
    training_end_time TIMESTAMP,
    
    -- Performance metrics
    mae DECIMAL(8,4),
    rmse DECIMAL(8,4),
    r2 DECIMAL(8,4),
    validation_score DECIMAL(8,4),
    
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE,      -- Is this the production model?
    decommissioned_at TIMESTAMP,
    
    UNIQUE(model_name, model_version)
);

CREATE INDEX idx_model_registry_name ON model_registry(model_name);
CREATE INDEX idx_model_registry_active ON model_registry(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_model_registry_type ON model_registry(model_type);

-- Model predictions (cached predictions for players/gameweeks)
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    model_id UUID REFERENCES model_registry(model_id) ON DELETE SET NULL,
    player_id INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    gameweek INTEGER NOT NULL,
    season VARCHAR(10) NOT NULL,
    predicted_points DECIMAL(8,4) NOT NULL,
    confidence DECIMAL(5,4),              -- 0-1 confidence score
    prediction_interval_lower DECIMAL(8,4), -- For uncertainty quantification
    prediction_interval_upper DECIMAL(8,4),
    input_features JSONB,                 -- Feature vector used for this prediction
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(model_id, player_id, gameweek, season)
);

CREATE INDEX idx_predictions_player_gw ON model_predictions(player_id, gameweek);
CREATE INDEX idx_predictions_model ON model_predictions(model_id);
CREATE INDEX idx_predictions_created ON model_predictions(created_at);

-- Training metadata (track training runs)
CREATE TABLE IF NOT EXISTS training_metadata (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    
    -- Data info
    training_data_start_gameweek INTEGER,
    training_data_end_gameweek INTEGER,
    training_samples INTEGER,
    validation_samples INTEGER,
    
    -- Training configuration
    hyperparameters JSONB NOT NULL,
    feature_version INTEGER REFERENCES feature_versions(version),
    data_preprocessing_steps JSONB,
    
    -- Results
    training_duration_seconds INTEGER,
    final_metrics JSONB,                  -- {'mae': X, 'rmse': Y, 'r2': Z}
    feature_importance JSONB,             -- Feature → importance mapping
    
    -- Git/Metadata
    git_commit_hash CHAR(40),
    training_environment JSONB,           -- Python, library versions
    
    trained_at TIMESTAMP DEFAULT NOW(),
    trained_by VARCHAR(100) DEFAULT 'ml-system'
);

CREATE INDEX idx_training_metadata_model ON training_metadata(model_name, model_version);
CREATE INDEX idx_training_metadata_date ON training_metadata(trained_at);

-- ============================================================================
-- TRAINING DATA TABLES (for prepared datasets)
-- ============================================================================

-- Prepared training datasets (time-split for ML)
CREATE TABLE IF NOT EXISTS training_datasets (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(100) NOT NULL,   -- e.g., 'train_2024', 'val_2024', 'test_2024'
    season VARCHAR(10) NOT NULL,
    split_type VARCHAR(20) NOT NULL,      -- 'train', 'validation', 'test', 'holdout'
    start_gameweek INTEGER NOT NULL,
    end_gameweek INTEGER NOT NULL,
    
    dataset_hash CHAR(64) NOT NULL,       -- SHA256 of the dataset rows
    num_samples INTEGER NOT NULL,
    num_features INTEGER NOT NULL,
    
    file_path TEXT,                       -- Path to parquet/csv file if stored externally
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(dataset_name, season, split_type)
);

CREATE INDEX idx_training_datasets_name ON training_datasets(dataset_name);
CREATE INDEX idx_training_datasets_season ON training_datasets(season);
CREATE INDEX idx_training_datasets_split ON training_datasets(split_type);

-- ============================================================================
-- MONITORING & LOGGING TABLES
-- ============================================================================

-- Prediction logs (for monitoring drift/performance)
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    model_id UUID REFERENCES model_registry(model_id),
    player_id INTEGER REFERENCES players(id),
    gameweek INTEGER NOT NULL,
    predicted_points DECIMAL(8,4) NOT NULL,
    actual_points INTEGER,                -- NULL until gameweek completes
    error DECIMAL(8,4),                   -- prediction error (actual - predicted)
    
    -- Request metadata
    endpoint_used VARCHAR(100),           -- Which API endpoint triggered prediction
    user_id INTEGER,                      -- Manager ID if relevant
    latency_ms INTEGER,
    
    -- System health
    cache_hit BOOLEAN DEFAULT FALSE,
    fallback_used BOOLEAN DEFAULT FALSE,  -- Used rule-based fallback?
    confidence DECIMAL(5,4)
);

CREATE INDEX idx_prediction_logs_timestamp ON prediction_logs(timestamp);
CREATE INDEX idx_prediction_logs_gameweek ON prediction_logs(gameweek);
CREATE INDEX idx_prediction_logs_model ON prediction_logs(model_id);
CREATE INDEX idx_prediction_logs_error ON prediction_logs(error) WHERE error IS NOT NULL;

-- System metrics (for Grafana/Prometheus)
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    labels JSONB,                         -- Additional context (model, endpoint, etc.)
    recorded_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_system_metrics_name_time ON system_metrics(metric_name, recorded_at);

-- ============================================================================
-- INITIAL DATA VERSION TRACKING
-- ============================================================================

INSERT INTO feature_versions (version, description, feature_list) VALUES 
(1, 'Initial feature set with rolling stats, fixture difficulty, team form', 
 '{"features": ["now_cost", "element_type", "team", "total_points_rolling_1", "total_points_rolling_3", "total_points_rolling_6", "minutes_rolling_1", "minutes_rolling_3", "minutes_rolling_6", "goals_scored_rolling_1", "goals_scored_rolling_3", "assists_rolling_1", "assists_rolling_3", "clean_sheets_rolling_1", "clean_sheets_rolling_3", "avg_fdr_next_3", "avg_fdr_next_6", "dgw_next_3", "dgw_next_6", "bgw_next_3", "team_points_rolling_3", "team_goals_rolling_3", "team_conceded_rolling_3", "is_injured", "is_suspended", "injury_risk_score", "form_trend", "points_per_minute", "value_ratio", "ownership_factor", "home_advantage", "opp_strength"]}')
ON CONFLICT (version) DO NOTHING;

-- ============================================================================
-- FUNCTIONS & TRIGGERS
-- ============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables with updated_at
CREATE TRIGGER update_teams_updated_at BEFORE UPDATE ON teams
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_players_updated_at BEFORE UPDATE ON players
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_players_features_updated_at BEFORE UPDATE ON players_features
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE players_features IS 'Feature store for ML training and serving. One row per player per gameweek.';
COMMENT ON COLUMN players_features.total_points_rolling_1 IS 'Total points from the previous 1 gameweek';
COMMENT ON COLUMN players_features.total_points_rolling_3 IS 'Average total points over the last 3 gameweeks';
COMMENT ON COLUMN players_features.avg_fdr_next_3 IS 'Average fixture difficulty rating for the next 3 fixtures (lower is easier)';
COMMENT ON TABLE model_registry IS 'Tracks all trained models and their metadata. Active model is used for predictions.';
COMMENT ON TABLE prediction_logs IS 'Logs all predictions for monitoring drift and calculating actual performance';

-- ============================================================================
-- ROW LEVEL SECURITY (Optional - enable if needed)
-- ============================================================================

-- Enable RLS if needed for multi-tenancy
-- ALTER TABLE players_features ENABLE ROW LEVEL SECURITY;
