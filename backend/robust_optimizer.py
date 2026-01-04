"""
Robust Optimizer Module for FPL 2025/26

Implements robust optimization to handle prediction uncertainty.
Uses box uncertainty sets to maximize worst-case expected points,
hedging against the high variance (R² ≈ 0.14) of player points.
"""

import pulp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class RobustSolution:
    """Solution from robust optimization."""
    status: str
    squad: List[Dict[str, Any]]
    starters: List[Dict[str, Any]]
    bench: List[Dict[str, Any]]
    captain_id: int
    expected_points_nominal: float  # Without uncertainty
    expected_points_worst_case: float  # With uncertainty
    protection_level: float  # Gamma parameter used
    budget_used: float


class RobustOptimizer:
    """
    Robust optimization for FPL using uncertainty sets.
    
    Implements a min-max formulation to maximize the worst-case
    total score given prediction uncertainty:
    
        max_x min_ξ Σ (xp_j + ξ_j) * x_j
        s.t. |ξ_j| ≤ σ_j * Γ  (box uncertainty set)
    
    Where:
    - ξ_j represents the uncertainty in player j's predicted points
    - σ_j is the standard deviation of player j's prediction
    - Γ is the protection level (higher = more conservative)
    
    The inner minimization has a closed-form solution, leading to:
        max_x Σ (xp_j - Γ * σ_j) * x_j
    """
    
    POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    SQUAD_REQUIREMENTS = {1: 2, 2: 5, 3: 5, 4: 3}
    STARTER_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
    STARTER_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
    MAX_PER_TEAM = 3
    
    # Prediction uncertainty based on R² ≈ 0.14
    # This means ~86% of variance is unexplained
    BASE_R_SQUARED = 0.14
    
    def __init__(self, 
                 players_df: pd.DataFrame,
                 teams_df: pd.DataFrame):
        """
        Initialize robust optimizer.
        
        Args:
            players_df: Player data with expected points
            teams_df: Team data
        """
        self.players = players_df.copy()
        self.teams = teams_df.copy()
        
        self.team_names = dict(zip(self.teams['id'], self.teams['name']))
        
        # Add position names
        self.players['position'] = self.players['element_type'].map(self.POSITION_MAP)
        self.players['team_name'] = self.players['team'].map(self.team_names)
        
        # Compute uncertainty estimates
        self._compute_uncertainty()
    
    def _compute_uncertainty(self):
        """
        Compute prediction uncertainty (standard deviation) for each player.
        
        Uses the relationship: Var(Y|X) = Var(Y) * (1 - R²)
        
        We estimate σ proportional to expected points, adjusted by
        player-specific factors (minutes, consistency).
        """
        # Base unexplained variance ratio
        unexplained_var = 1 - self.BASE_R_SQUARED
        
        # Get expected points column
        xp_col = 'expected_points' if 'expected_points' in self.players.columns else 'ep_next'
        self.players['xp'] = pd.to_numeric(self.players.get(xp_col, 0), errors='coerce').fillna(0)
        
        # Base sigma = xp * sqrt(unexplained_variance)
        self.players['sigma'] = self.players['xp'] * np.sqrt(unexplained_var)
        
        # Adjust for minutes played (more minutes = more consistent = lower uncertainty)
        minutes = pd.to_numeric(self.players.get('minutes', 0), errors='coerce').fillna(0)
        max_minutes = minutes.max() if minutes.max() > 0 else 1
        minutes_factor = 1 + 0.3 * (1 - minutes / max_minutes)  # Higher for fewer minutes
        self.players['sigma'] *= minutes_factor
        
        # Adjust for ownership (heavily owned = more predictable)
        selected = pd.to_numeric(self.players.get('selected_by_percent', 0), errors='coerce').fillna(0)
        ownership_factor = 1 + 0.1 * (1 - selected / 100)
        self.players['sigma'] *= ownership_factor
        
        # Minimum sigma for meaningful uncertainty
        self.players['sigma'] = self.players['sigma'].clip(lower=0.5)
    
    def optimize(self,
                 budget: float = 100.0,
                 gamma: float = 1.0,
                 excluded_players: List[int] = [],
                 current_squad_ids: List[int] = []) -> RobustSolution:
        """
        Perform robust optimization.
        
        Args:
            budget: Budget in millions
            gamma: Protection level (0 = nominal, higher = more conservative)
                   Recommended: 0.5-1.5 for balanced approach
            excluded_players: Players to exclude
            current_squad_ids: Current squad for transfer mode
            
        Returns:
            RobustSolution with worst-case optimal team
        """
        # Filter available players
        available = self.players[
            ~self.players['id'].isin(excluded_players) &
            ~self.players['status'].isin(['u', 'i'])
        ].copy()
        
        player_ids = available['id'].tolist()
        
        # Helper function
        def get_attr(pid, attr):
            val = available.loc[available['id'] == pid, attr].values
            return val[0] if len(val) > 0 else 0
        
        # Create problem
        prob = pulp.LpProblem("FPL_Robust", pulp.LpMaximize)
        
        # Decision variables
        squad = pulp.LpVariable.dicts("squad", player_ids, cat='Binary')
        starter = pulp.LpVariable.dicts("starter", player_ids, cat='Binary')
        captain = pulp.LpVariable.dicts("captain", player_ids, cat='Binary')
        
        # Robust objective: max Σ (xp - γ*σ) * starter + Σ (xp - γ*σ) * captain
        # This maximizes worst-case points under box uncertainty
        objective_terms = []
        for j in player_ids:
            xp = get_attr(j, 'xp')
            sigma = get_attr(j, 'sigma')
            
            # Worst-case expected points
            robust_xp = xp - gamma * sigma
            
            # Starter contributes full robust points
            objective_terms.append(robust_xp * starter[j])
            # Captain adds another robust_xp (2x total)
            objective_terms.append(robust_xp * captain[j])
            # Bench contributes small amount
            objective_terms.append(0.1 * robust_xp * (squad[j] - starter[j]))
        
        prob += pulp.lpSum(objective_terms)
        
        # Constraints
        for j in player_ids:
            prob += starter[j] <= squad[j]
            prob += captain[j] <= starter[j]
        
        # Budget
        prob += pulp.lpSum([get_attr(j, 'now_cost') / 10.0 * squad[j] for j in player_ids]) <= budget
        
        # Squad = 15
        prob += pulp.lpSum([squad[j] for j in player_ids]) == 15
        
        # Starters = 11
        prob += pulp.lpSum([starter[j] for j in player_ids]) == 11
        
        # Captain = 1
        prob += pulp.lpSum([captain[j] for j in player_ids]) == 1
        
        # Position constraints for squad
        for pos, count in self.SQUAD_REQUIREMENTS.items():
            prob += pulp.lpSum([
                squad[j] for j in player_ids 
                if get_attr(j, 'element_type') == pos
            ]) == count
        
        # Position constraints for starters
        for pos in [1, 2, 3, 4]:
            pos_players = [j for j in player_ids if get_attr(j, 'element_type') == pos]
            prob += pulp.lpSum([starter[j] for j in pos_players]) >= self.STARTER_MIN[pos]
            prob += pulp.lpSum([starter[j] for j in pos_players]) <= self.STARTER_MAX[pos]
        
        # Max 3 per team
        for team_id in self.teams['id']:
            team_players = [j for j in player_ids if get_attr(j, 'team') == team_id]
            prob += pulp.lpSum([squad[j] for j in team_players]) <= self.MAX_PER_TEAM
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
        
        if prob.status != 1:
            return RobustSolution(
                status=pulp.LpStatus[prob.status],
                squad=[],
                starters=[],
                bench=[],
                captain_id=0,
                expected_points_nominal=0,
                expected_points_worst_case=0,
                protection_level=gamma,
                budget_used=0
            )
        
        # Extract solution
        result_squad = []
        result_starters = []
        result_bench = []
        captain_id = None
        
        for j in player_ids:
            if squad[j].value() == 1:
                player_data = available[available['id'] == j].to_dict('records')[0]
                player_data['is_starter'] = starter[j].value() == 1
                player_data['is_captain'] = captain[j].value() == 1
                player_data['robust_xp'] = get_attr(j, 'xp') - gamma * get_attr(j, 'sigma')
                
                # Clean up for JSON
                cleaned = {}
                for k, v in player_data.items():
                    if isinstance(v, float) and (v != v):  # NaN check
                        cleaned[k] = None
                    else:
                        cleaned[k] = v
                
                result_squad.append(cleaned)
                
                if cleaned['is_captain']:
                    captain_id = j
                if cleaned['is_starter']:
                    result_starters.append(cleaned)
                else:
                    result_bench.append(cleaned)
        
        # Sort by position
        result_starters.sort(key=lambda p: (p['element_type'], -(p.get('xp', 0) or 0)))
        result_bench.sort(key=lambda p: -(p.get('xp', 0) or 0))
        
        # Calculate totals
        nominal_xp = sum(
            (p.get('xp', 0) or 0) * (2 if p['is_captain'] else 1) 
            for p in result_starters
        )
        worst_case_xp = sum(
            (p.get('robust_xp', 0) or 0) * (2 if p['is_captain'] else 1) 
            for p in result_starters
        )
        
        budget_used = sum(p['now_cost'] for p in result_squad) / 10.0
        
        return RobustSolution(
            status='Optimal',
            squad=result_squad,
            starters=result_starters,
            bench=result_bench,
            captain_id=captain_id,
            expected_points_nominal=nominal_xp,
            expected_points_worst_case=worst_case_xp,
            protection_level=gamma,
            budget_used=budget_used
        )
    
    def sensitivity_analysis(self,
                              budget: float = 100.0,
                              gamma_range: List[float] = [0, 0.5, 1.0, 1.5, 2.0],
                              excluded_players: List[int] = []) -> pd.DataFrame:
        """
        Perform sensitivity analysis across different protection levels.
        
        Args:
            budget: Budget in millions
            gamma_range: List of gamma values to test
            excluded_players: Players to exclude
            
        Returns:
            DataFrame with results for each gamma level
        """
        results = []
        
        for gamma in gamma_range:
            solution = self.optimize(
                budget=budget,
                gamma=gamma,
                excluded_players=excluded_players
            )
            
            if solution.status == 'Optimal':
                # Get captain name
                captain_name = next(
                    (p['web_name'] for p in solution.starters if p['is_captain']),
                    'Unknown'
                )
                
                results.append({
                    'gamma': gamma,
                    'nominal_xp': solution.expected_points_nominal,
                    'worst_case_xp': solution.expected_points_worst_case,
                    'budget_used': solution.budget_used,
                    'captain': captain_name,
                    'key_players': ', '.join(
                        p['web_name'] for p in sorted(
                            solution.starters, 
                            key=lambda x: -(x.get('xp', 0) or 0)
                        )[:3]
                    )
                })
        
        return pd.DataFrame(results)
    
    def to_dict(self, solution: RobustSolution) -> Dict[str, Any]:
        """Convert solution to JSON-serializable dict."""
        return {
            'status': solution.status,
            'squad': solution.squad,
            'starters': solution.starters,
            'bench': solution.bench,
            'captain_id': solution.captain_id,
            'expected_points_nominal': solution.expected_points_nominal,
            'expected_points_worst_case': solution.expected_points_worst_case,
            'protection_level': solution.protection_level,
            'budget_used': solution.budget_used,
            'uncertainty_metrics': {
                'r_squared': self.BASE_R_SQUARED,
                'unexplained_variance': 1 - self.BASE_R_SQUARED,
                'gamma_interpretation': (
                    'Conservative' if solution.protection_level >= 1.5 else
                    'Balanced' if solution.protection_level >= 0.75 else
                    'Aggressive' if solution.protection_level > 0 else
                    'Nominal (no robustness)'
                )
            }
        }
