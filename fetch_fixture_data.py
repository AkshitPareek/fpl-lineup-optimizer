#!/usr/bin/env python3
"""
Fetch real fixture data for feature engineering.

This script fetches:
- Upcoming fixtures for all teams
- Team strength ratings
- Historical fixture results
"""

import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def fetch_fpl_data():
    """Fetch all FPL data."""
    print("📡 Fetching FPL data...")
    
    # Static data (teams, players)
    static_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    static = requests.get(static_url).json()
    
    # Fixtures
    fixtures_url = 'https://fantasy.premierleague.com/api/fixtures/'
    fixtures = requests.get(fixtures_url).json()
    
    print(f"   Loaded {len(static['elements'])} players")
    print(f"   Loaded {len(fixtures)} fixtures")
    
    return static, fixtures


def build_team_mappings(static):
    """Build team ID to name/strength mappings."""
    teams = {}
    for team in static['teams']:
        teams[team['id']] = {
            'name': team['name'],
            'short_name': team['short_name'],
            'strength': team['strength'],
            'strength_overall_home': team['strength_overall_home'],
            'strength_overall_away': team['strength_overall_away'],
            'strength_attack_home': team['strength_attack_home'],
            'strength_attack_away': team['strength_attack_away'],
            'strength_defence_home': team['strength_defence_home'],
            'strength_defence_away': team['strength_defence_away'],
        }
    return teams


def get_upcoming_fixtures(fixtures, team_id, current_gw, n_gws=5):
    """Get upcoming fixtures for a team."""
    upcoming = []
    
    for f in fixtures:
        if f['event'] and f['event'] > current_gw and f['event'] <= current_gw + n_gws:
            if f['team_h'] == team_id or f['team_a'] == team_id:
                is_home = f['team_h'] == team_id
                opponent = f['team_a'] if is_home else f['team_h']
                
                upcoming.append({
                    'gw': f['event'],
                    'is_home': is_home,
                    'opponent_id': opponent,
                    'difficulty': f['team_h_difficulty'] if is_home else f['team_a_difficulty'],
                    'date': f.get('kickoff_time', '')
                })
    
    return sorted(upcoming, key=lambda x: x['gw'])


def calculate_fdr_features(upcoming_fixtures, teams, n_gws=5):
    """Calculate fixture difficulty rating features."""
    if not upcoming_fixtures:
        return {
            'fdr_next': 3.0,
            'fdr_avg_5': 3.0,
            'fdr_min': 3.0,
            'fdr_max': 3.0,
            'fdr_variance': 0.0,
            'n_home': 2,
            'n_away': 3
        }
    
    # Next GW difficulty
    fdr_next = upcoming_fixtures[0]['difficulty'] if upcoming_fixtures else 3.0
    
    # Average over next 5 GWs
    difficulties = [f['difficulty'] for f in upcoming_fixtures[:n_gws]]
    fdr_avg = sum(difficulties) / len(difficulties) if difficulties else 3.0
    
    # Min/Max
    fdr_min = min(difficulties) if difficulties else 3.0
    fdr_max = max(difficulties) if difficulties else 3.0
    
    # Variance (how much fixture difficulty varies)
    if len(difficulties) > 1:
        mean_diff = sum(difficulties) / len(difficulties)
        variance = sum((d - mean_diff) ** 2 for d in difficulties) / len(difficulties)
    else:
        variance = 0.0
    
    # Home/Away balance
    home_count = sum(1 for f in upcoming_fixtures[:n_gws] if f['is_home'])
    away_count = len(upcoming_fixtures[:n_gws]) - home_count
    
    return {
        'fdr_next': float(fdr_next),
        'fdr_avg_5': float(fdr_avg),
        'fdr_min': float(fdr_min),
        'fdr_max': float(fdr_max),
        'fdr_variance': float(variance),
        'n_home': home_count,
        'n_away': away_count
    }


def get_player_fixture_features(player, fixtures, teams, current_gw):
    """Get all fixture-related features for a player."""
    team_id = player.get('team', 0)
    
    # Get upcoming fixtures
    upcoming = get_upcoming_fixtures(fixtures, team_id, current_gw)
    
    # Calculate FDR features
    fdr_features = calculate_fdr_features(upcoming, teams)
    
    # Add opponent info for next GW
    if upcoming:
        next_f = upcoming[0]
        opponent_team = teams.get(next_f['opponent_id'], {})
        fdr_features['opponent_name'] = opponent_team.get('short_name', 'UNK')
        fdr_features['is_home_next'] = 1.0 if next_f['is_home'] else 0.0
        fdr_features['opponent_strength'] = opponent_team.get('strength', 3)
    else:
        fdr_features['opponent_name'] = 'UNK'
        fdr_features['is_home_next'] = 0.5
        fdr_features['opponent_strength'] = 3
    
    return fdr_features


def save_fixture_data():
    """Fetch and save all fixture data."""
    print("="*70)
    print("FETCHING FIXTURE DATA FOR FEATURE ENGINEERING")
    print("="*70)
    
    static, fixtures = fetch_fpl_data()
    teams = build_team_mappings(static)
    
    # Find current gameweek
    current_gw = 1
    for event in static['events']:
        if event.get('is_current'):
            current_gw = event['id']
            break
    
    print(f"\nCurrent GW: {current_gw}")
    
    # Build player fixture features
    print("\nBuilding player fixture features...")
    
    player_features = {}
    for player in static['elements']:
        player_id = player['id']
        features = get_player_fixture_features(player, fixtures, teams, current_gw)
        player_features[player_id] = features
    
    # Save to file
    output_dir = Path('data/fixtures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'player_fixture_features_gw{current_gw}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'gw': current_gw,
            'timestamp': datetime.now().isoformat(),
            'player_features': player_features,
            'teams': teams
        }, f, indent=2)
    
    print(f"\n✅ Saved to {output_file}")
    
    # Print sample
    print("\nSample features (Player 1):")
    sample = player_features.get(1, {})
    for key, val in sample.items():
        print(f"  {key}: {val}")
    
    return output_file


if __name__ == "__main__":
    try:
        save_fixture_data()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
