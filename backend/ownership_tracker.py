"""
Ownership Tracker Module for FPL Analytics

Tracks and analyzes player ownership data:
- Current ownership percentages from FPL API
- Effective Ownership (EO) calculations including captaincy
- Ownership trends over time
- Differential player identification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging


@dataclass
class OwnershipData:
    """Ownership information for a player."""
    player_id: int
    player_name: str
    team: str
    position: str
    ownership_pct: float
    captain_pct: float  # Estimated
    effective_ownership: float
    is_template: bool  # High ownership (>20%)
    is_differential: bool  # Low ownership (<5%)
    price: float


class OwnershipTracker:
    """
    Tracks player ownership and calculates Effective Ownership (EO).
    
    Effective Ownership = Ownership% × (1 + Captain%)
    
    Features:
    - Identifies template players (high ownership must-haves)
    - Identifies differentials (low ownership punts)
    - Compares user's squad to overall ownership
    """
    
    # Thresholds
    TEMPLATE_THRESHOLD = 20.0  # >20% ownership = template
    DIFFERENTIAL_THRESHOLD = 5.0  # <5% ownership = differential
    PREMIUM_CAPTAIN_BOOST = 1.5  # Premiums get captained more
    
    def __init__(self, players_df: pd.DataFrame, teams_df: pd.DataFrame):
        """
        Initialize with FPL player data.
        
        Args:
            players_df: DataFrame with player data (including selected_by_percent)
            teams_df: DataFrame with team data
        """
        self.players = players_df.copy()
        self.teams = teams_df.copy()
        
        # Build lookups
        self.team_names = dict(zip(self.teams['id'], self.teams['name']))
        self.team_short_names = dict(zip(self.teams['id'], self.teams['short_name']))
        
        # Position map
        self.positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    def get_ownership_data(self, player_id: int) -> Optional[OwnershipData]:
        """
        Get ownership information for a single player.
        
        Args:
            player_id: FPL player ID
            
        Returns:
            OwnershipData or None if not found
        """
        player = self.players[self.players['id'] == player_id]
        if len(player) == 0:
            return None
        
        player = player.iloc[0]
        ownership = float(player.get('selected_by_percent', 0) or 0)
        
        # Estimate captain percentage based on ownership and price
        captain_pct = self._estimate_captain_pct(player)
        
        # Calculate effective ownership
        eo = self._calculate_eo(ownership, captain_pct)
        
        return OwnershipData(
            player_id=player_id,
            player_name=player['web_name'],
            team=self.team_short_names.get(player['team'], 'UNK'),
            position=self.positions.get(player['element_type'], 'UNK'),
            ownership_pct=ownership,
            captain_pct=captain_pct,
            effective_ownership=eo,
            is_template=ownership >= self.TEMPLATE_THRESHOLD,
            is_differential=ownership < self.DIFFERENTIAL_THRESHOLD,
            price=player['now_cost'] / 10
        )
    
    def _estimate_captain_pct(self, player: pd.Series) -> float:
        """
        Estimate captain percentage based on ownership and player profile.
        
        Higher ownership players tend to be captained more.
        Premium players (>£10m) get additional captain boost.
        """
        ownership = float(player.get('selected_by_percent', 0) or 0)
        price = player['now_cost'] / 10
        element_type = player['element_type']
        
        # Base captain rate proportional to ownership
        # Top-owned players get captained by ~30-50% of their owners
        base_captain_rate = min(0.5, ownership / 100 * 0.5)
        
        # Premium boost for expensive attackers
        if price >= 12.0 and element_type in [3, 4]:  # MID or FWD
            base_captain_rate *= self.PREMIUM_CAPTAIN_BOOST
        elif price >= 10.0:
            base_captain_rate *= 1.2
        
        # GK and DEF rarely captained
        if element_type in [1, 2]:
            base_captain_rate *= 0.1
        
        # Form boost
        form = float(player.get('form', 0) or 0)
        if form >= 7:
            base_captain_rate *= 1.3
        
        # Return as percentage of total managers
        return min(50.0, base_captain_rate * ownership)
    
    def _calculate_eo(self, ownership: float, captain_pct: float) -> float:
        """
        Calculate Effective Ownership.
        
        EO = Ownership × (1 + Captain%/Ownership)
           = Ownership + Captain%
        
        This represents the expected point swing per point scored.
        """
        return ownership + captain_pct
    
    def get_all_ownership(self) -> pd.DataFrame:
        """
        Get ownership data for all players.
        
        Returns:
            DataFrame with ownership metrics
        """
        data = []
        
        for _, player in self.players.iterrows():
            ownership = float(player.get('selected_by_percent', 0) or 0)
            captain_pct = self._estimate_captain_pct(player)
            eo = self._calculate_eo(ownership, captain_pct)
            
            data.append({
                'id': player['id'],
                'web_name': player['web_name'],
                'team': self.team_short_names.get(player['team'], 'UNK'),
                'team_name': self.team_names.get(player['team'], 'Unknown'),
                'position': self.positions.get(player['element_type'], 'UNK'),
                'element_type': player['element_type'],
                'price': player['now_cost'] / 10,
                'ownership_pct': ownership,
                'captain_pct': round(captain_pct, 2),
                'effective_ownership': round(eo, 2),
                'is_template': ownership >= self.TEMPLATE_THRESHOLD,
                'is_differential': ownership < self.DIFFERENTIAL_THRESHOLD,
                'form': float(player.get('form', 0) or 0),
                'total_points': int(player.get('total_points', 0) or 0),
                'ep_next': float(player.get('ep_next', 0) or 0)
            })
        
        return pd.DataFrame(data)
    
    def get_template_players(self, min_ownership: float = None) -> pd.DataFrame:
        """
        Get highly-owned 'template' players.
        
        These are must-have players that most managers own.
        
        Args:
            min_ownership: Minimum ownership threshold (default: 20%)
            
        Returns:
            DataFrame of template players
        """
        threshold = min_ownership or self.TEMPLATE_THRESHOLD
        df = self.get_all_ownership()
        return df[df['ownership_pct'] >= threshold].sort_values(
            'ownership_pct', ascending=False
        )
    
    def get_differentials(self, max_ownership: float = None) -> pd.DataFrame:
        """
        Get low-ownership differential players.
        
        These are potential punts that could help climb ranks.
        
        Args:
            max_ownership: Maximum ownership threshold (default: 5%)
            
        Returns:
            DataFrame of differential players (filtered by form)
        """
        threshold = max_ownership or self.DIFFERENTIAL_THRESHOLD
        df = self.get_all_ownership()
        
        # Filter to differentials with decent form/points
        differentials = df[
            (df['ownership_pct'] < threshold) & 
            (df['form'] >= 3.0) &
            (df['price'] >= 4.5)
        ]
        
        return differentials.sort_values('form', ascending=False)
    
    def get_captaincy_rankings(self, gameweek: int = None) -> pd.DataFrame:
        """
        Get player rankings by estimated captaincy.
        
        Returns:
            DataFrame of top captain picks
        """
        df = self.get_all_ownership()
        
        # Filter to realistic captain options (MID/FWD with good ownership)
        captains = df[
            (df['element_type'].isin([3, 4])) &  # MID or FWD
            (df['ownership_pct'] >= 5.0)  # At least 5% owned
        ]
        
        return captains.sort_values('captain_pct', ascending=False).head(20)
    
    def compare_squad_to_template(self, squad_ids: List[int]) -> Dict:
        """
        Compare a user's squad to the template.
        
        Args:
            squad_ids: List of player IDs in user's squad
            
        Returns:
            Dict with template coverage and differentials
        """
        df = self.get_all_ownership()
        template = self.get_template_players()
        
        # Check which template players user has
        template_ids = set(template['id'].tolist())
        user_template = [pid for pid in squad_ids if pid in template_ids]
        missing_template = template_ids - set(squad_ids)
        
        # Check user's differentials
        user_players = df[df['id'].isin(squad_ids)]
        user_differentials = user_players[user_players['is_differential']]
        
        # Calculate effective ownership of squad
        squad_eo = user_players['effective_ownership'].mean()
        
        return {
            'template_coverage': len(user_template) / len(template_ids) * 100 if template_ids else 0,
            'template_players_owned': len(user_template),
            'template_players_missing': len(missing_template),
            'missing_template': [
                {
                    'id': row['id'],
                    'name': row['web_name'],
                    'ownership': row['ownership_pct'],
                    'position': row['position']
                }
                for _, row in template[template['id'].isin(missing_template)].iterrows()
            ],
            'differentials_owned': len(user_differentials),
            'differential_players': [
                {
                    'id': row['id'],
                    'name': row['web_name'],
                    'ownership': row['ownership_pct'],
                    'form': row['form']
                }
                for _, row in user_differentials.iterrows()
            ],
            'avg_effective_ownership': round(squad_eo, 2),
            'squad_style': 'Template' if len(user_differentials) <= 2 else 'Differential'
        }
    
    def get_ownership_by_position(self) -> Dict:
        """
        Get ownership breakdown by position.
        
        Returns:
            Dict with position-wise ownership stats
        """
        df = self.get_all_ownership()
        
        result = {}
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            pos_df = df[df['position'] == pos]
            result[pos] = {
                'most_owned': pos_df.nlargest(5, 'ownership_pct')[
                    ['id', 'web_name', 'team', 'price', 'ownership_pct', 'form']
                ].to_dict('records'),
                'top_differentials': pos_df[pos_df['is_differential']].nlargest(5, 'form')[
                    ['id', 'web_name', 'team', 'price', 'ownership_pct', 'form']
                ].to_dict('records')
            }
        
        return result
    
    def calculate_rank_impact(self, player_id: int, points_scored: int) -> Dict:
        """
        Calculate how a player's score affects rank.
        
        If you own a highly-owned player who scores, everyone benefits.
        If you own a differential who scores, you gain rank.
        
        Args:
            player_id: Player who scored
            points_scored: Points they scored
            
        Returns:
            Dict with rank impact analysis
        """
        data = self.get_ownership_data(player_id)
        if not data:
            return {'error': 'Player not found'}
        
        eo = data.effective_ownership
        
        # If you own, you gain relative to (100 - EO)%
        # If you don't own, you lose relative to EO%
        
        owns_you_gain = points_scored * (100 - eo) / 100
        owns_you_lose = points_scored * eo / 100
        
        return {
            'player_name': data.player_name,
            'ownership': data.ownership_pct,
            'effective_ownership': eo,
            'points_scored': points_scored,
            'if_you_own': {
                'relative_gain': round(owns_you_gain, 2),
                'impact': 'Good' if eo < 50 else 'Neutral' if eo < 80 else 'Low Impact'
            },
            'if_you_dont_own': {
                'relative_loss': round(owns_you_lose, 2),
                'impact': 'Risky' if eo > 50 else 'Acceptable' if eo > 20 else 'Safe Avoid'
            }
        }


def get_ownership_summary(players_df: pd.DataFrame, teams_df: pd.DataFrame) -> Dict:
    """
    Quick function to get ownership summary.
    """
    tracker = OwnershipTracker(players_df, teams_df)
    
    template = tracker.get_template_players()
    differentials = tracker.get_differentials()
    captains = tracker.get_captaincy_rankings()
    
    return {
        'template_count': len(template),
        'top_template': template.head(10)[['web_name', 'team', 'ownership_pct', 'price']].to_dict('records'),
        'differential_count': len(differentials),
        'top_differentials': differentials.head(10)[['web_name', 'team', 'ownership_pct', 'form']].to_dict('records'),
        'top_captains': captains.head(10)[['web_name', 'team', 'captain_pct', 'ownership_pct']].to_dict('records')
    }
