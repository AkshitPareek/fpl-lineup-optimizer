-- Add missing columns to players_features table
-- Migration: 002_add_missing_feature_columns.sql

-- Add rolling columns for 6 gameweeks that were missing
ALTER TABLE players_features ADD COLUMN IF NOT EXISTS goals_scored_rolling_6 DECIMAL(5,2);
ALTER TABLE players_features ADD COLUMN IF NOT EXISTS assists_rolling_6 DECIMAL(5,2);
ALTER TABLE players_features ADD COLUMN IF NOT EXISTS clean_sheets_rolling_6 DECIMAL(5,2);

-- Rename team_points_rolling to match expected column name
ALTER TABLE players_features RENAME COLUMN team_points_rolling TO team_points_rolling_3;

-- Verify columns exist
SELECT column_name FROM information_schema.columns 
WHERE table_name = 'players_features' 
ORDER BY ordinal_position;
