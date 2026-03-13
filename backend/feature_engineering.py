"""
Feature Engineering Module for FPL Predictor

Adds advanced predictive features:
- Fixture difficulty rating (detailed opponent analysis)
- Rest days / fatigue (fixture congestion)
- Momentum indicators (3/5/10 game rolling averages)
- Team chemistry (assists between players)
- Weather data integration

Usage:
    from feature_engineering import FeatureEngineer
    
    engineer = FeatureEngineer()
    enhanced_features = engineer.enhance_player_features(player_data, fixtures)
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for FPL prediction.
    
    Adds contextual features beyond basic player stats to capture:
    - Situational difficulty
    - Physical/mental fatigue
    - Recent form trajectories
    - Team dynamics
    - Environmental factors
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Cache for expensive computations
        self._fixture_difficulty_cache = {}
        self._team_stats_cache = {}
        self._player_history_cache = {}
        
        # Load team strength data
        self.team_strength = self._load_team_strength()
        
    def _load_team_strength(self) -> Dict:
        """Load team strength ratings from FPL API or file."""
        cache_file = self.data_dir / "team_strength.json"
        
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        
        # Default strengths (1-5 scale)
        # These should be updated from FPL API
        default_strength = {
            'LIV': {'overall': 5, 'attack': 5, 'defence': 4, 'home': 5, 'away': 4},
            'MCI': {'overall': 5, 'attack': 5, 'defence': 5, 'home': 5, 'away': 5},
            'ARS': {'overall': 5, 'attack': 4, 'defence': 5, 'home': 5, 'away': 4},
            'CHE': {'overall': 4, 'attack': 4, 'defence': 4, 'home': 4, 'away': 4},
            'MUN': {'overall': 4, 'attack': 4, 'defence': 3, 'home': 4, 'away': 3},
            'TOT': {'overall': 4, 'attack': 4, 'defence': 3, 'home': 4, 'away': 3},
            'NEW': {'overall': 4, 'attack': 4, 'defence': 4, 'home': 4, 'away': 3},
            'AVL': {'overall': 4, 'attack': 4, 'defence': 3, 'home': 4, 'away': 3},
            'BRI': {'overall': 3, 'attack': 3, 'defence': 3, 'home': 3, 'away': 3},
            'WHU': {'overall': 3, 'attack': 3, 'defence': 3, 'home': 3, 'away': 3},
            'CRY': {'overall': 3, 'attack': 3, 'defence': 3, 'home': 3, 'away': 3},
            'FUL': {'overall': 3, 'attack': 3, 'defence': 3, 'home': 3, 'away': 2},
            'EVE': {'overall': 3, 'attack': 2, 'defence': 3, 'home': 3, 'away': 2},
            'BRE': {'overall': 3, 'attack': 3, 'defence': 3, 'home': 3, 'away': 2},
            'NFO': {'overall': 3, 'attack': 3, 'defence': 3, 'home': 3, 'away': 2},
            'BOU': {'overall': 2, 'attack': 3, 'defence': 2, 'home': 3, 'away': 2},
            'WOL': {'overall': 2, 'attack': 2, 'defence': 3, 'home': 3, 'away': 2},
            'IPS': {'overall': 2, 'attack': 2, 'defence': 2, 'home': 2, 'away': 2},
            'LEI': {'overall': 2, 'attack': 2, 'defence': 2, 'home': 2, 'away': 2},
            'SOU': {'overall': 2, 'attack': 2, 'defence': 2, 'home': 2, 'away': 1},
        }
        
        return default_strength
    
    def calculate_fixture_difficulty_rating(
        self, 
        player_team: str, 
        opponent_team: str, 
        is_home: bool,
        player_position: int
    ) -> Dict[str, float]:
        """
        Calculate detailed fixture difficulty rating.
        
        Args:
            player_team: 3-letter team code (e.g., 'LIV')
            opponent_team: 3-letter opponent code
            is_home: Whether player is at home
            player_position: 1=GK, 2=DEF, 3=MID, 4=FWD
            
        Returns:
            Dict with difficulty metrics
        """
        cache_key = f"{player_team}_{opponent_team}_{is_home}_{player_position}"
        if cache_key in self._fixture_difficulty_cache:
            return self._fixture_difficulty_cache[cache_key]
        
        # Get team strengths
        player_strength = self.team_strength.get(player_team, {'overall': 3, 'attack': 3, 'defence': 3})
        opponent_strength = self.team_strength.get(opponent_team, {'overall': 3, 'attack': 3, 'defence': 3})
        
        # Calculate base difficulty
        overall_diff = opponent_strength['overall'] - player_strength['overall']
        
        # Position-specific difficulty
        if player_position == 1:  # GK - cares about opponent attack
            attack_threat = opponent_strength['attack']
            difficulty = attack_threat * 0.6 + overall_diff * 0.4
        elif player_position == 2:  # DEF - cares about opponent attack and their defence
            attack_threat = opponent_strength['attack']
            own_defence = player_strength['defence']
            difficulty = (attack_threat - own_defence) * 0.5 + overall_diff * 0.5
        elif player_position in [3, 4]:  # MID/FWD - cares about opponent defence
            defence_strength = opponent_strength['defence']
            own_attack = player_strength['attack']
            difficulty = (defence_strength - own_attack) * 0.5 + overall_diff * 0.5
        else:
            difficulty = overall_diff
        
        # Home advantage adjustment
        home_bonus = 0.5 if is_home else -0.5
        player_home_adv = player_strength.get('home', 3) - 3
        opponent_away_dis = opponent_strength.get('away', 3) - 3
        home_adjustment = home_bonus + (player_home_adv - opponent_away_dis) * 0.3
        
        # Normalize to 1-5 scale
        base_fdr = 3.0 + difficulty * 0.5 + home_adjustment * 0.3
        fdr = max(1.0, min(5.0, base_fdr))
        
        result = {
            'fdr': round(fdr, 2),
            'fdr_defensive': round(max(1.0, min(5.0, opponent_strength['attack'] + home_adjustment * 0.2)), 2),
            'fdr_offensive': round(max(1.0, min(5.0, 6 - opponent_strength['defence'] + home_adjustment * 0.2)), 2),
            'home_advantage': round(home_adjustment, 2),
            'opponent_attack': opponent_strength['attack'],
            'opponent_defence': opponent_strength['defence'],
            'relative_strength': round(overall_diff, 2)
        }
        
        self._fixture_difficulty_cache[cache_key] = result
        return result
    
    def calculate_rest_days_and_fatigue(
        self, 
        player_history: List[Dict],
        current_gw: int
    ) -> Dict[str, float]:
        """
        Calculate rest days and fatigue metrics.
        
        Args:
            player_history: List of gameweek performance dicts
            current_gw: Current gameweek number
            
        Returns:
            Dict with fatigue metrics
        """
        if not player_history or len(player_history) < 2:
            return {
                'rest_days': 7.0,
                'fatigue_score': 0.0,
                'matches_7d': 1,
                'matches_30d': 4,
                'minutes_7d': 90,
                'minutes_30d': 360
            }
        
        # Sort by gameweek
        sorted_history = sorted(player_history, key=lambda x: x.get('round', 0), reverse=True)
        
        # Calculate days since last match (assume 7 days per GW)
        last_gw = sorted_history[0].get('round', current_gw)
        rest_days = (current_gw - last_gw) * 7
        
        # Calculate matches in last 7/14/30 days
        matches_7d = sum(1 for h in sorted_history if h.get('minutes', 0) > 0 and (current_gw - h.get('round', 0)) <= 1)
        matches_14d = sum(1 for h in sorted_history if h.get('minutes', 0) > 0 and (current_gw - h.get('round', 0)) <= 2)
        matches_30d = sum(1 for h in sorted_history if h.get('minutes', 0) > 0 and (current_gw - h.get('round', 0)) <= 4)
        
        # Calculate minutes played
        minutes_7d = sum(h.get('minutes', 0) for h in sorted_history if (current_gw - h.get('round', 0)) <= 1)
        minutes_14d = sum(h.get('minutes', 0) for h in sorted_history if (current_gw - h.get('round', 0)) <= 2)
        minutes_30d = sum(h.get('minutes', 0) for h in sorted_history if (current_gw - h.get('round', 0)) <= 4)
        
        # Calculate fatigue score (0-10)
        # High minutes + short rest = high fatigue
        fatigue_from_minutes = min(10, minutes_14d / 180)  # ~2 full matches = max fatigue
        fatigue_from_frequency = min(10, matches_14d * 2.5)  # 4 matches in 14 days = max fatigue
        fatigue_recovery = max(0, (rest_days - 3) * 0.5)  # 3+ days rest reduces fatigue
        
        fatigue_score = min(10, (fatigue_from_minutes + fatigue_from_frequency) / 2 - fatigue_recovery)
        fatigue_score = max(0, fatigue_score)
        
        # Fixture congestion (matches per week)
        congestion = matches_14d / 2.0  # matches per week
        
        return {
            'rest_days': float(rest_days),
            'fatigue_score': round(fatigue_score, 2),
            'matches_7d': matches_7d,
            'matches_14d': matches_14d,
            'matches_30d': matches_30d,
            'minutes_7d': minutes_7d,
            'minutes_14d': minutes_14d,
            'minutes_30d': minutes_30d,
            'fixture_congestion': round(congestion, 2)
        }
    
    def calculate_momentum_indicators(
        self, 
        player_history: List[Dict],
        current_gw: int
    ) -> Dict[str, float]:
        """
        Calculate momentum indicators (rolling averages and trends).
        
        Args:
            player_history: List of gameweek performance dicts
            current_gw: Current gameweek number
            
        Returns:
            Dict with momentum metrics
        """
        if not player_history:
            return {
                'form_3gw': 0.0,
                'form_5gw': 0.0,
                'form_10gw': 0.0,
                'trend': 0.0,
                'consistency': 0.0,
                'points_per_minute': 0.0
            }
        
        # Sort by gameweek
        sorted_history = sorted(player_history, key=lambda x: x.get('round', 0), reverse=True)
        
        # Get recent performances
        def get_recent_gws(n_gws):
            recent = [h for h in sorted_history if (current_gw - h.get('round', 0)) < n_gws]
            return recent[:n_gws]
        
        recent_3 = get_recent_gws(3)
        recent_5 = get_recent_gws(5)
        recent_10 = get_recent_gws(10)
        
        # Calculate form (points per game)
        def calc_form(matches):
            if not matches:
                return 0.0
            total_points = sum(h.get('total_points', 0) for h in matches)
            return total_points / len(matches)
        
        form_3gw = calc_form(recent_3)
        form_5gw = calc_form(recent_5)
        form_10gw = calc_form(recent_10)
        
        # Calculate trend (improving or declining)
        if len(recent_5) >= 3:
            early = sum(h.get('total_points', 0) for h in recent_5[-3:]) / 3
            late = sum(h.get('total_points', 0) for h in recent_5[:3]) / 3 if len(recent_5) >= 3 else early
            trend = late - early
        else:
            trend = 0.0
        
        # Calculate consistency (coefficient of variation)
        if len(recent_5) >= 3:
            points = [h.get('total_points', 0) for h in recent_5]
            mean_pts = np.mean(points)
            std_pts = np.std(points)
            consistency = 1.0 - min(1.0, std_pts / (mean_pts + 1))  # Higher = more consistent
        else:
            consistency = 0.5
        
        # Points per minute (efficiency)
        total_mins = sum(h.get('minutes', 0) for h in recent_5)
        total_pts = sum(h.get('total_points', 0) for h in recent_5)
        ppm = total_pts / total_mins * 90 if total_mins > 0 else 0  # Points per 90 mins
        
        # Goal involvement trend
        goals_3gw = sum(h.get('goals_scored', 0) for h in recent_3)
        assists_3gw = sum(h.get('assists', 0) for h in recent_3)
        
        # xG trend
        xg_3gw = sum(float(h.get('expected_goals', 0) or 0) for h in recent_3)
        xa_3gw = sum(float(h.get('expected_assists', 0) or 0) for h in recent_3)
        
        return {
            'form_3gw': round(form_3gw, 2),
            'form_5gw': round(form_5gw, 2),
            'form_10gw': round(form_10gw, 2),
            'trend': round(trend, 2),
            'consistency': round(consistency, 2),
            'points_per_90': round(ppm, 3),
            'goals_3gw': goals_3gw,
            'assists_3gw': assists_3gw,
            'xg_3gw': round(xg_3gw, 2),
            'xa_3gw': round(xa_3gw, 2),
            'goal_involvement_3gw': goals_3gw + assists_3gw
        }
    
    def calculate_team_chemistry(
        self, 
        player_id: int,
        player_team: str,
        player_history: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate team chemistry metrics (assists between players).
        
        Args:
            player_id: Player ID
            player_team: Team code
            player_history: Player's gameweek history
            
        Returns:
            Dict with chemistry metrics
        """
        # This would require detailed event data (who assisted whom)
        # For now, use proxy metrics
        
        if not player_history:
            return {
                'assist_consistency': 0.0,
                'key_passes_per_game': 0.0,
                'team_attack_strength': 3.0,
                'creativity_index': 0.0
            }
        
        # Calculate assist-related metrics
        total_assists = sum(h.get('assists', 0) for h in player_history[-10:])
        total_key_passes = sum(float(h.get('creativity', 0) or 0) for h in player_history[-10:])
        
        # Assist consistency
        if len(player_history) >= 5:
            assist_matches = sum(1 for h in player_history[-10:] if h.get('assists', 0) > 0)
            assist_consistency = assist_matches / min(10, len(player_history))
        else:
            assist_consistency = 0.0
        
        # Creativity index (from ICT)
        avg_creativity = np.mean([float(h.get('creativity', 0) or 0) for h in player_history[-5:]]) if player_history else 0
        
        # Team attack strength (from team_strength dict)
        team_attack = self.team_strength.get(player_team, {}).get('attack', 3)
        
        return {
            'assist_consistency': round(assist_consistency, 2),
            'key_passes_per_game': round(total_key_passes / 10, 1),
            'team_attack_strength': team_attack,
            'creativity_index': round(avg_creativity / 100, 2),  # Normalize
            'recent_assists': total_assists
        }
    
    def get_weather_adjustment(self, match_date: str, venue: str) -> Dict[str, float]:
        """
        Get weather-based adjustment factors.
        
        Args:
            match_date: Date string (YYYY-MM-DD)
            venue: Stadium/location
            
        Returns:
            Dict with weather adjustments
        """
        # This would integrate with weather API
        # For now, return neutral values with placeholders
        
        # Season-based adjustments
        month = datetime.strptime(match_date, '%Y-%m-%d').month if match_date else 1
        
        # Winter months (Nov-Feb) see more low-scoring games
        is_winter = month in [11, 12, 1, 2]
        
        # Rain/wind affects passing teams more
        weather_factor = 0.9 if is_winter else 1.0
        
        return {
            'weather_factor': weather_factor,
            'is_winter': 1.0 if is_winter else 0.0,
            'rain_adjustment': 0.0,  # Would come from weather API
            'wind_adjustment': 0.0,
            'temp_adjustment': 0.0
        }
    
    def enhance_player_features(
        self, 
        player_data: Dict,
        fixtures: List[Dict],
        player_history: List[Dict],
        current_gw: int
    ) -> Dict:
        """
        Enhance player data with all engineered features.
        
        Args:
            player_data: Base player dict from FPL API
            fixtures: Upcoming fixtures for player
            player_history: Historical performance
            current_gw: Current gameweek
            
        Returns:
            Enhanced player dict with new features
        """
        enhanced = player_data.copy()
        
        # Get next fixture
        next_fixture = fixtures[0] if fixtures else None
        
        if next_fixture:
            # 1. Fixture Difficulty
            fdr_data = self.calculate_fixture_difficulty_rating(
                player_team=player_data.get('team', 'UNK'),
                opponent_team=next_fixture.get('opponent', 'UNK'),
                is_home=next_fixture.get('is_home', True),
                player_position=player_data.get('position', 3)
            )
            enhanced.update({f'fdr_{k}': v for k, v in fdr_data.items()})
            
            # 2. Weather
            match_date = next_fixture.get('date', datetime.now().strftime('%Y-%m-%d'))
            weather_data = self.get_weather_adjustment(match_date, next_fixture.get('venue', ''))
            enhanced.update({f'weather_{k}': v for k, v in weather_data.items()})
        
        # 3. Fatigue
        fatigue_data = self.calculate_rest_days_and_fatigue(player_history, current_gw)
        enhanced.update({f'fatigue_{k}': v for k, v in fatigue_data.items()})
        
        # 4. Momentum
        momentum_data = self.calculate_momentum_indicators(player_history, current_gw)
        enhanced.update({f'momentum_{k}': v for k, v in momentum_data.items()})
        
        # 5. Team Chemistry
        chemistry_data = self.calculate_team_chemistry(
            player_data.get('id'),
            player_data.get('team', 'UNK'),
            player_history
        )
        enhanced.update({f'chemistry_{k}': v for k, v in chemistry_data.items()})
        
        return enhanced
    
    def get_feature_vector(self, enhanced_player: Dict) -> np.ndarray:
        """
        Convert enhanced player dict to feature vector for model input.
        
        Returns:
            numpy array of features
        """
        # Define feature order (must match training!)
        feature_keys = [
            # Base features
            'form', 'total_points', 'points_per_game', 'minutes',
            'goals_scored', 'assists', 'clean_sheets',
            'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
            
            # Fixture difficulty
            'fdr_fdr', 'fdr_fdr_defensive', 'fdr_fdr_offensive', 'fdr_home_advantage',
            'fdr_opponent_attack', 'fdr_opponent_defence', 'fdr_relative_strength',
            
            # Fatigue
            'fatigue_rest_days', 'fatigue_fatigue_score', 'fatigue_matches_7d',
            'fatigue_matches_14d', 'fatigue_minutes_14d', 'fatigue_fixture_congestion',
            
            # Momentum
            'momentum_form_3gw', 'momentum_form_5gw', 'momentum_trend',
            'momentum_consistency', 'momentum_points_per_90',
            'momentum_goals_3gw', 'momentum_xg_3gw', 'momentum_xa_3gw',
            
            # Chemistry
            'chemistry_assist_consistency', 'chemistry_key_passes_per_game',
            'chemistry_team_attack_strength', 'chemistry_creativity_index',
            
            # Weather
            'weather_weather_factor', 'weather_is_winter'
        ]
        
        features = []
        for key in feature_keys:
            val = enhanced_player.get(key, 0.0)
            if val is None:
                val = 0.0
            features.append(float(val))
        
        return np.array(features)


def test_feature_engineering():
    """Test the feature engineering module."""
    print("="*70)
    print("TESTING FEATURE ENGINEERING")
    print("="*70)
    
    engineer = FeatureEngineer()
    
    # Test fixture difficulty
    print("\n1. Fixture Difficulty Rating:")
    fdr = engineer.calculate_fixture_difficulty_rating('LIV', 'MCI', True, 4)
    print(f"   LIV (FWD) vs MCI at home: {fdr}")
    
    fdr2 = engineer.calculate_fixture_difficulty_rating('MCI', 'SOU', True, 2)
    print(f"   MCI (DEF) vs SOU at home: {fdr2}")
    
    # Test fatigue
    print("\n2. Fatigue Metrics:")
    sample_history = [
        {'round': 28, 'minutes': 90, 'total_points': 6},
        {'round': 27, 'minutes': 85, 'total_points': 4},
        {'round': 26, 'minutes': 90, 'total_points': 8},
        {'round': 25, 'minutes': 78, 'total_points': 2},
    ]
    fatigue = engineer.calculate_rest_days_and_fatigue(sample_history, 29)
    print(f"   Sample player: {fatigue}")
    
    # Test momentum
    print("\n3. Momentum Indicators:")
    momentum = engineer.calculate_momentum_indicators(sample_history, 29)
    print(f"   Sample player: {momentum}")
    
    # Test chemistry
    print("\n4. Team Chemistry:")
    chemistry = engineer.calculate_team_chemistry(123, 'LIV', sample_history)
    print(f"   Sample player: {chemistry}")
    
    # Test full enhancement
    print("\n5. Full Enhancement:")
    player = {
        'id': 123,
        'name': 'Test Player',
        'team': 'LIV',
        'position': 4,
        'form': 8.5,
        'total_points': 150
    }
    fixtures = [{'opponent': 'MCI', 'is_home': True, 'date': '2024-03-15'}]
    enhanced = engineer.enhance_player_features(player, fixtures, sample_history, 29)
    
    print(f"   Enhanced features count: {len(enhanced)}")
    print(f"   Sample enhanced values:")
    print(f"     fdr_fdr: {enhanced.get('fdr_fdr')}")
    print(f"     fatigue_fatigue_score: {enhanced.get('fatigue_fatigue_score')}")
    print(f"     momentum_form_3gw: {enhanced.get('momentum_form_3gw')}")
    print(f"     chemistry_creativity_index: {enhanced.get('chemistry_creativity_index')}")
    
    # Test feature vector
    print("\n6. Feature Vector:")
    vector = engineer.get_feature_vector(enhanced)
    print(f"   Vector shape: {vector.shape}")
    print(f"   First 10 values: {vector[:10]}")
    
    print("\n" + "="*70)
    print("✅ FEATURE ENGINEERING TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_feature_engineering()
