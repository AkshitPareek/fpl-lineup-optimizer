"""
Expected Value (EV) Distribution Calculator

Analyzes the probability distribution of expected points:
- Monte Carlo simulation for point variance
- Ceiling/floor analysis for players
- Squad-level EV distributions
- Risk assessment tools
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging


@dataclass 
class EVDistribution:
    """Expected value distribution for a player."""
    player_id: int
    player_name: str
    expected_points: float
    floor: float  # 10th percentile
    ceiling: float  # 90th percentile
    std_dev: float
    upside: float  # ceiling - expected
    downside: float  # expected - floor
    risk_score: float  # Coefficient of variation


class EVCalculator:
    """
    Calculates Expected Value distributions for FPL players.
    
    Uses simplified Monte Carlo approach based on:
    - Base expected points (from predictions)
    - Position-specific variance
    - Form volatility
    - Fixture difficulty impact
    """
    
    # Position-specific variance multipliers
    # GKs/DEFs more consistent, FWDs more volatile
    POSITION_VARIANCE = {
        1: 0.4,   # GK - Low variance
        2: 0.5,   # DEF - Low-medium variance
        3: 0.7,   # MID - Medium variance
        4: 0.9,   # FWD - High variance
    }
    
    # Base standard deviation as fraction of expected points
    BASE_STD_FRACTION = 0.6
    
    def __init__(self, players_df: pd.DataFrame, teams_df: pd.DataFrame):
        """
        Initialize with player data.
        
        Args:
            players_df: FPL players DataFrame
            teams_df: FPL teams DataFrame
        """
        self.players = players_df.copy()
        self.teams = teams_df.copy()
        
        self.team_names = dict(zip(self.teams['id'], self.teams['name']))
        self.positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    def calculate_player_distribution(self, player_id: int, 
                                       expected_points: float = None) -> EVDistribution:
        """
        Calculate the EV distribution for a single player.
        
        Args:
            player_id: Player's FPL ID
            expected_points: Override expected points (uses ep_next if None)
            
        Returns:
            EVDistribution with floor, ceiling, and risk metrics
        """
        player = self.players[self.players['id'] == player_id]
        if len(player) == 0:
            raise ValueError(f"Player {player_id} not found")
        
        player = player.iloc[0]
        
        # Get expected points
        if expected_points is None:
            expected_points = float(player.get('ep_next', 0) or 0)
            if expected_points == 0:
                expected_points = float(player.get('form', 0) or 0)
        
        # Calculate variance based on position and player characteristics
        element_type = player['element_type']
        position_var = self.POSITION_VARIANCE.get(element_type, 0.6)
        
        # Form volatility - high form players may be streaky
        form = float(player.get('form', 0) or 0)
        form_volatility = 1.0 + (abs(form - 4.0) / 10)  # Higher deviation from avg = more volatile
        
        # Calculate standard deviation
        std_dev = expected_points * self.BASE_STD_FRACTION * position_var * form_volatility
        std_dev = max(0.5, std_dev)  # Minimum variance
        
        # Calculate percentiles (assuming roughly normal distribution)
        floor = max(0, expected_points - 1.28 * std_dev)  # 10th percentile
        ceiling = expected_points + 1.28 * std_dev  # 90th percentile
        
        # Risk score (coefficient of variation)
        risk_score = std_dev / max(0.1, expected_points)
        
        return EVDistribution(
            player_id=player_id,
            player_name=player['web_name'],
            expected_points=round(expected_points, 2),
            floor=round(floor, 2),
            ceiling=round(ceiling, 2),
            std_dev=round(std_dev, 2),
            upside=round(ceiling - expected_points, 2),
            downside=round(expected_points - floor, 2),
            risk_score=round(risk_score, 2)
        )
    
    def get_all_distributions(self) -> pd.DataFrame:
        """
        Calculate EV distributions for all players.
        
        Returns:
            DataFrame with EV distribution metrics
        """
        data = []
        
        for _, player in self.players.iterrows():
            try:
                dist = self.calculate_player_distribution(player['id'])
                data.append({
                    'id': player['id'],
                    'web_name': player['web_name'],
                    'team': self.team_names.get(player['team'], 'Unknown'),
                    'position': self.positions.get(player['element_type'], 'UNK'),
                    'price': player['now_cost'] / 10,
                    'expected_points': dist.expected_points,
                    'floor': dist.floor,
                    'ceiling': dist.ceiling,
                    'std_dev': dist.std_dev,
                    'upside': dist.upside,
                    'downside': dist.downside,
                    'risk_score': dist.risk_score
                })
            except Exception:
                continue
        
        return pd.DataFrame(data)
    
    def get_high_ceiling_players(self, min_ceiling: float = 8.0, 
                                  position: int = None) -> pd.DataFrame:
        """
        Get players with high ceiling (explosive potential).
        
        Args:
            min_ceiling: Minimum ceiling threshold
            position: Filter by position (optional)
            
        Returns:
            DataFrame of high-ceiling players
        """
        df = self.get_all_distributions()
        
        if position:
            pos_name = self.positions.get(position, '')
            df = df[df['position'] == pos_name]
        
        high_ceiling = df[df['ceiling'] >= min_ceiling]
        return high_ceiling.sort_values('ceiling', ascending=False)
    
    def get_safe_players(self, max_risk: float = 0.5,
                         min_expected: float = 3.0) -> pd.DataFrame:
        """
        Get low-risk players with consistent returns.
        
        Args:
            max_risk: Maximum risk score
            min_expected: Minimum expected points
            
        Returns:
            DataFrame of safe, consistent players
        """
        df = self.get_all_distributions()
        
        safe = df[
            (df['risk_score'] <= max_risk) &
            (df['expected_points'] >= min_expected)
        ]
        
        return safe.sort_values('expected_points', ascending=False)
    
    def get_high_upside_players(self, min_upside: float = 4.0) -> pd.DataFrame:
        """
        Get players with most upside potential.
        
        Upside = Ceiling - Expected
        High upside = Room to exceed expectations
        
        Args:
            min_upside: Minimum upside value
            
        Returns:
            DataFrame of high-upside players
        """
        df = self.get_all_distributions()
        high_upside = df[df['upside'] >= min_upside]
        return high_upside.sort_values('upside', ascending=False)
    
    def calculate_squad_ev(self, squad_ids: List[int], 
                           starting_xi: List[int] = None) -> Dict:
        """
        Calculate aggregate EV distribution for a squad.
        
        Args:
            squad_ids: List of 15 player IDs
            starting_xi: List of 11 player IDs for starting lineup (optional)
            
        Returns:
            Dict with squad-level EV metrics
        """
        if starting_xi is None:
            starting_xi = squad_ids[:11]
        
        # Calculate for starting XI
        xi_dists = []
        for pid in starting_xi[:11]:
            try:
                dist = self.calculate_player_distribution(pid)
                xi_dists.append(dist)
            except Exception:
                continue
        
        if not xi_dists:
            return {'error': 'No valid players in squad'}
        
        # Aggregate metrics
        total_expected = sum(d.expected_points for d in xi_dists)
        
        # Floor: sum of individual floors (pessimistic)
        total_floor = sum(d.floor for d in xi_dists)
        
        # Ceiling: sum of individual ceilings (optimistic)
        total_ceiling = sum(d.ceiling for d in xi_dists)
        
        # Squad variance (assuming some correlation)
        # Not perfectly independent, so multiply std by sqrt(n) * 0.7
        avg_std = np.mean([d.std_dev for d in xi_dists])
        squad_std = avg_std * np.sqrt(len(xi_dists)) * 0.7
        
        # Risk profile
        avg_risk = np.mean([d.risk_score for d in xi_dists])
        
        # Player contributions
        player_contributions = [
            {
                'id': d.player_id,
                'name': d.player_name,
                'expected': d.expected_points,
                'floor': d.floor,
                'ceiling': d.ceiling,
                'pct_of_total': round(d.expected_points / max(1, total_expected) * 100, 1)
            }
            for d in xi_dists
        ]
        
        return {
            'total_expected': round(total_expected, 1),
            'floor': round(total_floor, 1),
            'ceiling': round(total_ceiling, 1),
            'range': round(total_ceiling - total_floor, 1),
            'std_dev': round(squad_std, 1),
            'avg_risk_score': round(avg_risk, 2),
            'risk_profile': 'Conservative' if avg_risk < 0.4 else 'Balanced' if avg_risk < 0.6 else 'Aggressive',
            'player_contributions': sorted(
                player_contributions, 
                key=lambda x: x['expected'], 
                reverse=True
            )
        }
    
    def compare_players(self, player_ids: List[int]) -> Dict:
        """
        Compare EV distributions of multiple players.
        
        Useful for transfer decisions.
        
        Args:
            player_ids: List of player IDs to compare
            
        Returns:
            Dict with comparison metrics
        """
        comparisons = []
        
        for pid in player_ids:
            try:
                dist = self.calculate_player_distribution(pid)
                player = self.players[self.players['id'] == pid].iloc[0]
                
                comparisons.append({
                    'id': pid,
                    'name': dist.player_name,
                    'team': self.team_names.get(player['team'], 'Unknown'),
                    'position': self.positions.get(player['element_type'], 'UNK'),
                    'price': player['now_cost'] / 10,
                    'expected': dist.expected_points,
                    'floor': dist.floor,
                    'ceiling': dist.ceiling,
                    'upside': dist.upside,
                    'risk': dist.risk_score,
                    'value': round(dist.expected_points / (player['now_cost'] / 10), 2)
                })
            except Exception:
                continue
        
        if not comparisons:
            return {'error': 'No valid players'}
        
        # Sort by expected points
        comparisons.sort(key=lambda x: x['expected'], reverse=True)
        
        # Recommendations
        best_expected = max(comparisons, key=lambda x: x['expected'])
        best_ceiling = max(comparisons, key=lambda x: x['ceiling'])
        best_floor = max(comparisons, key=lambda x: x['floor'])
        best_value = max(comparisons, key=lambda x: x['value'])
        safest = min(comparisons, key=lambda x: x['risk'])
        
        return {
            'players': comparisons,
            'recommendations': {
                'highest_expected': {'id': best_expected['id'], 'name': best_expected['name']},
                'highest_ceiling': {'id': best_ceiling['id'], 'name': best_ceiling['name']},
                'highest_floor': {'id': best_floor['id'], 'name': best_floor['name']},
                'best_value': {'id': best_value['id'], 'name': best_value['name']},
                'safest_pick': {'id': safest['id'], 'name': safest['name']}
            }
        }
    
    def simulate_gameweek(self, squad_ids: List[int], n_simulations: int = 1000) -> Dict:
        """
        Monte Carlo simulation of gameweek outcomes.
        
        Args:
            squad_ids: Squad player IDs
            n_simulations: Number of simulations
            
        Returns:
            Dict with simulation results
        """
        starting_xi = squad_ids[:11]
        
        # Get distributions for starting XI
        distributions = []
        for pid in starting_xi:
            try:
                dist = self.calculate_player_distribution(pid)
                distributions.append(dist)
            except Exception:
                continue
        
        if not distributions:
            return {'error': 'No valid players'}
        
        # Run simulations
        results = []
        for _ in range(n_simulations):
            total = 0
            for dist in distributions:
                # Sample from truncated normal distribution
                sampled = np.random.normal(dist.expected_points, dist.std_dev)
                sampled = max(0, sampled)  # No negative points
                total += sampled
            results.append(total)
        
        results = np.array(results)
        
        return {
            'simulations': n_simulations,
            'mean': round(np.mean(results), 1),
            'median': round(np.median(results), 1),
            'std_dev': round(np.std(results), 1),
            'min': round(np.min(results), 1),
            'max': round(np.max(results), 1),
            'percentile_10': round(np.percentile(results, 10), 1),
            'percentile_25': round(np.percentile(results, 25), 1),
            'percentile_50': round(np.percentile(results, 50), 1),
            'percentile_75': round(np.percentile(results, 75), 1),
            'percentile_90': round(np.percentile(results, 90), 1),
            'prob_above_50': round(np.mean(results >= 50) * 100, 1),
            'prob_above_60': round(np.mean(results >= 60) * 100, 1),
            'prob_above_70': round(np.mean(results >= 70) * 100, 1),
        }


def get_ev_summary(players_df: pd.DataFrame, teams_df: pd.DataFrame) -> Dict:
    """Quick function for EV summary."""
    calc = EVCalculator(players_df, teams_df)
    
    high_ceiling = calc.get_high_ceiling_players(min_ceiling=10.0)
    safe = calc.get_safe_players(max_risk=0.4, min_expected=4.0)
    high_upside = calc.get_high_upside_players(min_upside=5.0)
    
    return {
        'high_ceiling_count': len(high_ceiling),
        'top_ceiling': high_ceiling.head(10)[['web_name', 'team', 'expected_points', 'ceiling', 'upside']].to_dict('records'),
        'safe_players_count': len(safe),
        'top_safe': safe.head(10)[['web_name', 'team', 'expected_points', 'risk_score']].to_dict('records'),
        'high_upside_count': len(high_upside),
        'top_upside': high_upside.head(10)[['web_name', 'team', 'expected_points', 'ceiling', 'upside']].to_dict('records')
    }
