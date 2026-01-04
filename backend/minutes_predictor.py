"""
Minutes Predictor Module for FPL 2025/26

Predicts the probability of a player starting/playing based on:
- Recent starts and minutes history
- Injury flags and availability
- Rotation patterns
- Fixture congestion
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class MinutesPredictor:
    """
    Predicts expected minutes and start probability for FPL players.
    
    Uses historical patterns to estimate:
    - Probability of starting (0-1)
    - Expected minutes (0-90)
    - Rotation risk factor
    """
    
    # Minutes thresholds
    FULL_GAME = 90
    STARTER_THRESHOLD = 60
    
    def __init__(self, players_df: pd.DataFrame, history_cache: Dict = None):
        """
        Initialize with player data and optional history cache.
        
        Args:
            players_df: DataFrame with player data
            history_cache: Dict of player_id -> history data
        """
        self.players = players_df.copy()
        self.history_cache = history_cache or {}
    
    def get_recent_minutes(self, player_id: int, last_n_gws: int = 5) -> List[int]:
        """Get minutes played in last N gameweeks."""
        history = self.history_cache.get(str(player_id), {}).get('history', [])
        
        if not history:
            return []
        
        # Get last N entries
        recent = history[-last_n_gws:] if len(history) >= last_n_gws else history
        return [h.get('minutes', 0) for h in recent]
    
    def calculate_start_probability(self, player: pd.Series, gw: int = None) -> float:
        """
        Calculate probability of starting based on recent history.
        
        Factors:
        - Recent starts % (last 5 GWs)
        - Average minutes consistency
        - Injury/availability status
        - Position (GKs more nailed than outfield)
        
        Returns:
            Float between 0-1 representing start probability
        """
        player_id = player['id']
        
        # Get recent minutes
        recent_mins = self.get_recent_minutes(player_id, last_n_gws=5)
        
        if not recent_mins:
            # No history - use availability flag
            status = player.get('status', 'a')
            if status == 'a':  # Available
                return 0.7  # Default moderate probability
            elif status == 'd':  # Doubtful
                return 0.3
            else:  # Injured, unavailable, suspended
                return 0.0
        
        # Calculate starts percentage
        starts = sum(1 for m in recent_mins if m >= self.STARTER_THRESHOLD)
        start_pct = starts / len(recent_mins)
        
        # Calculate average minutes
        avg_mins = sum(recent_mins) / len(recent_mins)
        mins_factor = min(1.0, avg_mins / self.FULL_GAME)
        
        # Combine factors
        base_prob = (start_pct * 0.6 + mins_factor * 0.4)
        
        # Adjust for availability status
        status = player.get('status', 'a')
        chance = player.get('chance_of_playing_next_round')
        
        if status != 'a':
            if status == 'd':  # Doubtful
                base_prob *= 0.5
            elif status in ['i', 'u', 's']:  # Injured, unavailable, suspended
                base_prob = 0.0
        
        if chance is not None and not pd.isna(chance):
            base_prob *= (float(chance) / 100.0)
        
        # GKs tend to be more consistent starters
        if player.get('element_type') == 1:  # Goalkeeper
            base_prob = min(1.0, base_prob * 1.1)
        
        return min(1.0, max(0.0, base_prob))
    
    def calculate_expected_minutes(self, player: pd.Series, gw: int = None) -> float:
        """
        Calculate expected minutes for a player.
        
        Returns:
            Expected minutes (0-90)
        """
        start_prob = self.calculate_start_probability(player, gw)
        
        # If likely to start, estimate full 90 or 60+ minutes
        if start_prob >= 0.8:
            return 85.0  # Likely full game
        elif start_prob >= 0.5:
            return 65.0  # Likely starter
        elif start_prob >= 0.2:
            return 25.0  # Possible sub appearance
        else:
            return 0.0
    
    def calculate_rotation_risk(self, player: pd.Series, gw: int = None) -> float:
        """
        Calculate rotation risk based on patterns.
        
        Returns:
            Risk factor 0-1 (higher = more rotation risk)
        """
        recent_mins = self.get_recent_minutes(player['id'], last_n_gws=5)
        
        if len(recent_mins) < 3:
            return 0.3  # Default moderate risk with limited data
        
        # Check for alternating starts pattern
        starts = [1 if m >= self.STARTER_THRESHOLD else 0 for m in recent_mins]
        
        # High variance in starts = rotation risk
        if len(starts) >= 2:
            variance = np.var(starts)
            rotation_risk = min(1.0, variance * 4)  # Scale variance to 0-1
        else:
            rotation_risk = 0.0
        
        return rotation_risk
    
    def get_nailedness_score(self, player: pd.Series) -> float:
        """
        Calculate overall 'nailedness' score (0-100).
        
        Higher score = more likely to play consistently.
        """
        start_prob = self.calculate_start_probability(player)
        rotation_risk = self.calculate_rotation_risk(player)
        
        # Nailed score: high start prob, low rotation risk
        nailedness = (start_prob * 0.7 + (1 - rotation_risk) * 0.3) * 100
        
        return round(nailedness, 1)
    
    def predict_all_players(self) -> pd.DataFrame:
        """
        Generate predictions for all players.
        
        Returns:
            DataFrame with start_prob, expected_mins, nailedness columns
        """
        predictions = []
        
        for _, player in self.players.iterrows():
            predictions.append({
                'id': player['id'],
                'web_name': player['web_name'],
                'start_probability': self.calculate_start_probability(player),
                'expected_minutes': self.calculate_expected_minutes(player),
                'rotation_risk': self.calculate_rotation_risk(player),
                'nailedness': self.get_nailedness_score(player)
            })
        
        return pd.DataFrame(predictions)
