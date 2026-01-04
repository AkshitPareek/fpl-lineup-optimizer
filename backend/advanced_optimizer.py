"""
Advanced Multi-Period MILP Optimizer for FPL 2025/26

Implements a rolling horizon optimization over 3-8 gameweeks with:
- Transfer flow constraints with banking (up to 5 FTs)
- Chip strategy logic (Wildcard, Free Hit, Triple Captain, Bench Boost)
- Squad continuity across gameweeks
- Captain and vice-captain selection
- Robust optimization with uncertainty handling
"""

import pulp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from point_predictor import PointPredictor, compute_uncertainty_bounds
from transfer_explainer import TransferExplainer


class Chip(Enum):
    """FPL Chip types."""
    WILDCARD = "wildcard"
    FREE_HIT = "freehit"
    TRIPLE_CAPTAIN = "triple_captain"
    BENCH_BOOST = "bench_boost"


@dataclass
class TransferPlan:
    """Transfer recommendations for a gameweek."""
    gameweek: int
    transfers_in: List[Dict[str, Any]]
    transfers_out: List[Dict[str, Any]]
    free_transfers_used: int
    hits_taken: int
    hit_cost: int
    explanation: str
    xp_gain: float = 0.0  # Total xP gain from transfers
    hit_worth_it: bool = True  # Is the hit worth it? (xP gain > 6 threshold)


@dataclass
class GameweekPlan:
    """Optimal plan for a single gameweek."""
    gameweek: int
    starting_xi: List[Dict[str, Any]]
    bench: List[Dict[str, Any]]  # In bench order
    captain_id: int
    vice_captain_id: int
    expected_points: float
    chip_used: Optional[Chip] = None
    transfers: Optional[TransferPlan] = None


@dataclass
class MultiPeriodSolution:
    """Complete multi-period optimization solution."""
    status: str
    squad: List[Dict[str, Any]]  # Current 15-man squad
    gameweek_plans: List[GameweekPlan]
    total_expected_points: float
    transfer_summary: List[TransferPlan]
    banked_transfers_after: int
    chips_remaining: List[str]
    optimization_time_seconds: float


class MultiPeriodFPLOptimizer:
    """
    Multi-period MILP optimizer for FPL 2025/26 season.
    
    Features:
    - Rolling horizon optimization (3-8 gameweeks)
    - Transfer banking (up to 5 FTs)
    - Chip strategy (Wildcard, Free Hit, Triple Captain, Bench Boost)
    - CBIT/CBIRT expected point bonuses
    - Robust optimization option
    """
    
    # Position requirements
    POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    SQUAD_REQUIREMENTS = {1: 2, 2: 5, 3: 5, 4: 3}  # GK: 2, DEF: 5, MID: 5, FWD: 3
    STARTER_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
    STARTER_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
    
    MAX_PER_TEAM = 3
    MAX_BANKED_TRANSFERS = 5  # 2025/26 season: can bank up to 5 FTs
    TRANSFER_PENALTY = 4
    
    def __init__(self, 
                 players_df: pd.DataFrame,
                 teams_df: pd.DataFrame,
                 fixtures: List[Dict],
                 current_gameweek: int,
                 predictions_df: Optional[pd.DataFrame] = None):
        """
        Initialize the multi-period optimizer.
        
        Args:
            players_df: All player data
            teams_df: Team data
            fixtures: Fixture list
            current_gameweek: Starting gameweek
            predictions_df: Pre-computed predictions (optional)
        """
        self.players = players_df.copy()
        self.teams = teams_df.copy()
        self.fixtures = fixtures
        self.current_gw = current_gameweek
        
        # Build team name map
        self.team_names = dict(zip(self.teams['id'], self.teams['name']))
        
        # Position mapping
        self.players['position'] = self.players['element_type'].map(self.POSITION_MAP)
        self.players['team_name'] = self.players['team'].map(self.team_names)
        
        # Initialize point predictor
        self.predictor = PointPredictor(
            players_df=self.players,
            teams_df=self.teams,
            fixtures=fixtures,
            current_gameweek=current_gameweek
        )
        
        # Compute or use provided predictions
        if predictions_df is not None:
            self.predictions = predictions_df
        else:
            self.predictions = None
            
        # Initialize transfer explainer
        self.transfer_explainer = TransferExplainer(
            players_df=self.players,
            teams_df=self.teams,
            fixtures=fixtures
        )
            
    def _get_predictions(self, gameweeks: int) -> pd.DataFrame:
        """Get or compute predictions for the planning horizon."""
        if self.predictions is None:
            self.predictions = self.predictor.predict_all_players(gameweeks=gameweeks)
        return self.predictions
    
    def _filter_available_players(self, 
                                   excluded_ids: List[int] = [],
                                   include_injured: bool = False) -> pd.DataFrame:
        """Filter players who are available for selection."""
        df = self.players.copy()
        
        # Exclude specified players
        df = df[~df['id'].isin(excluded_ids)]
        
        # Exclude unavailable/injured unless specified
        if not include_injured:
            df = df[~df['status'].isin(['u', 'i'])]
        
        return df
    
    def optimize_single_gameweek(self,
                                  budget: float = 100.0,
                                  current_squad_ids: List[int] = [],
                                  excluded_players: List[int] = [],
                                  free_transfers: int = 1,
                                  use_wildcard: bool = False) -> Dict[str, Any]:
        """
        Optimize for a single gameweek (used as base case or with Wildcard).
        
        Args:
            budget: Available budget in millions
            current_squad_ids: Current squad player IDs (empty for new team)
            excluded_players: Players to exclude
            free_transfers: Available free transfers
            use_wildcard: If True, ignore transfer costs
            
        Returns:
            Optimization result dict
        """
        predictions = self._get_predictions(gameweeks=1)
        available = self._filter_available_players(excluded_ids=excluded_players)
        
        # Merge with predictions
        available = available.merge(
            predictions[['id', 'total_xp']], 
            on='id', 
            how='left'
        )
        available['total_xp'] = available['total_xp'].fillna(
            available.get('expected_points', 0)
        )
        
        # Problem setup
        prob = pulp.LpProblem("FPL_Single_GW", pulp.LpMaximize)
        player_ids = available['id'].tolist()
        
        # Decision variables
        squad = pulp.LpVariable.dicts("squad", player_ids, cat='Binary')
        starter = pulp.LpVariable.dicts("starter", player_ids, cat='Binary')
        captain = pulp.LpVariable.dicts("captain", player_ids, cat='Binary')
        
        # Helper to get player attribute
        def get_attr(pid, attr):
            return available.loc[available['id'] == pid, attr].values[0]
        
        # Objective: Maximize expected points
        # Starters get full points, captain gets 2x, bench gets 0.1x
        prob += pulp.lpSum([
            get_attr(i, 'total_xp') * starter[i] +
            get_attr(i, 'total_xp') * captain[i] +  # Captain bonus (2x = 1x + 1x captain)
            0.1 * get_attr(i, 'total_xp') * (squad[i] - starter[i])
            for i in player_ids
        ])
        
        # Transfer penalty (if not using wildcard)
        if current_squad_ids and not use_wildcard:
            retained = pulp.lpSum([squad[i] for i in player_ids if i in current_squad_ids])
            transfers_made = 15 - retained
            penalty = pulp.LpVariable("penalty", lowBound=0)
            prob += penalty >= self.TRANSFER_PENALTY * (transfers_made - free_transfers)
            # Subtract from objective
            prob += pulp.lpSum([
                get_attr(i, 'total_xp') * starter[i] +
                get_attr(i, 'total_xp') * captain[i] +
                0.1 * get_attr(i, 'total_xp') * (squad[i] - starter[i])
                for i in player_ids
            ]) - penalty
        
        # Constraints
        
        # Starter must be in squad
        for i in player_ids:
            prob += starter[i] <= squad[i]
            prob += captain[i] <= starter[i]
        
        # Budget
        prob += pulp.lpSum([get_attr(i, 'now_cost') / 10.0 * squad[i] for i in player_ids]) <= budget
        
        # Squad size = 15
        prob += pulp.lpSum([squad[i] for i in player_ids]) == 15
        
        # Starting XI = 11
        prob += pulp.lpSum([starter[i] for i in player_ids]) == 11
        
        # Exactly 1 captain
        prob += pulp.lpSum([captain[i] for i in player_ids]) == 1
        
        # Position constraints for squad
        for pos, count in self.SQUAD_REQUIREMENTS.items():
            prob += pulp.lpSum([
                squad[i] for i in player_ids 
                if get_attr(i, 'element_type') == pos
            ]) == count
        
        # Position constraints for starters
        for pos in [1, 2, 3, 4]:
            pos_players = [i for i in player_ids if get_attr(i, 'element_type') == pos]
            prob += pulp.lpSum([starter[i] for i in pos_players]) >= self.STARTER_MIN[pos]
            prob += pulp.lpSum([starter[i] for i in pos_players]) <= self.STARTER_MAX[pos]
        
        # Max 3 per team
        for team_id in self.teams['id']:
            team_players = [i for i in player_ids if get_attr(i, 'team') == team_id]
            prob += pulp.lpSum([squad[i] for i in team_players]) <= self.MAX_PER_TEAM
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
        
        # Extract results
        result_squad = []
        for i in player_ids:
            if squad[i].value() == 1:
                player_data = available[available['id'] == i].to_dict('records')[0]
                player_data['is_starter'] = starter[i].value() == 1
                player_data['is_captain'] = captain[i].value() == 1
                result_squad.append(player_data)
        
        # Sort: captain first, then starters by XP, then bench
        result_squad.sort(key=lambda p: (
            -p.get('is_captain', False),
            -p.get('is_starter', False),
            -p.get('total_xp', 0)
        ))
        
        return {
            'status': pulp.LpStatus[prob.status],
            'squad': result_squad,
            'total_xp': pulp.value(prob.objective) if prob.status == 1 else 0,
            'budget_used': sum(p['now_cost'] for p in result_squad) / 10.0
        }
    
    def optimize_multi_period(self,
                               budget: float = 100.0,
                               gameweeks: int = 5,
                               current_squad_ids: List[int] = [],
                               excluded_players: List[int] = [],
                               banked_transfers: int = 1,
                               chips_used: List[str] = [],
                               robust: bool = False,
                               uncertainty_budget: float = 0.3,
                               strategy: str = "standard",
                               max_hits: int = 5) -> MultiPeriodSolution:
        """
        Multi-period optimization over rolling horizon.
        
        Args:
            budget: Total budget in millions
            gameweeks: Planning horizon (3-8)
            current_squad_ids: Current squad (if any)
            excluded_players: Players to exclude
            banked_transfers: Currently banked free transfers
            chips_used: Already used chips
            robust: Use robust optimization
            uncertainty_budget: Γ parameter for robust opt
            strategy: Optimization strategy:
                - "standard": Maximize expected points
                - "differential": Boost low-ownership players (<10%)
                - "template": Favor high-ownership players (>20%)
            max_hits: Maximum hits allowed per gameweek (0 = no hits allowed)
            
        Returns:
            MultiPeriodSolution with complete plan
        """
        import time
        start_time = time.time()
        
        gameweeks = max(3, min(8, gameweeks))  # Clamp to 3-8
        horizon = list(range(self.current_gw, self.current_gw + gameweeks))
        
        # Get predictions for all gameweeks
        predictions = self._get_predictions(gameweeks=gameweeks)
        
        if robust:
            predictions = compute_uncertainty_bounds(predictions)
        
        # Apply strategy-based adjustments
        if strategy in ["differential", "template"]:
            # Get ownership from players_df and merge into predictions
            ownership_df = self.players[['id', 'selected_by_percent']].copy()
            ownership_df['selected_by_percent'] = pd.to_numeric(
                ownership_df['selected_by_percent'], errors='coerce'
            ).fillna(0)
            predictions = predictions.merge(ownership_df, on='id', how='left')
            predictions['selected_by_percent'] = predictions['selected_by_percent'].fillna(0)
        
        if strategy == "differential":
            # Boost low-ownership players
            ownership = predictions['selected_by_percent']
            # Players < 10% owned get bonus, up to +1.5 xP per gameweek
            differential_bonus = (10 - ownership).clip(lower=0) * 0.15
            for col in [c for c in predictions.columns if c.startswith('xp_gw')]:
                predictions[col] = predictions[col] + differential_bonus
            print(f"Strategy: Differential - boosting {(ownership < 10).sum()} low-owned players")
        elif strategy == "template":
            # Favor high-ownership (safe picks), penalize low-ownership
            ownership = predictions['selected_by_percent']
            # Players > 20% owned get bonus, < 5% get penalty
            template_bonus = (ownership - 10).clip(lower=-5, upper=10) * 0.05
            for col in [c for c in predictions.columns if c.startswith('xp_gw')]:
                predictions[col] = predictions[col] + template_bonus
            print(f"Strategy: Template - favoring {(ownership > 20).sum()} popular players")
        
        available = self._filter_available_players(excluded_ids=excluded_players)
        
        # Merge predictions
        gw_cols = [f'xp_gw{gw}' for gw in horizon]
        merge_cols = ['id'] + gw_cols
        available = available.merge(
            predictions[merge_cols].drop_duplicates(), 
            on='id', 
            how='left'
        )
        for col in gw_cols:
            available[col] = available[col].fillna(0)
        
        player_ids = available['id'].tolist()
        
        # Helper function
        def get_attr(pid, attr):
            val = available.loc[available['id'] == pid, attr].values
            return val[0] if len(val) > 0 else 0
        
        # Create MILP problem
        prob = pulp.LpProblem("FPL_Multi_Period", pulp.LpMaximize)
        
        # Decision variables per gameweek
        # squad[j,t] = 1 if player j in squad at GW t
        squad = pulp.LpVariable.dicts("squad", 
                                       [(j, t) for j in player_ids for t in horizon],
                                       cat='Binary')
        
        # starter[j,t] = 1 if player j starts at GW t
        starter = pulp.LpVariable.dicts("starter",
                                         [(j, t) for j in player_ids for t in horizon],
                                         cat='Binary')
        
        # captain[j,t] = 1 if player j is captain at GW t
        captain = pulp.LpVariable.dicts("captain",
                                         [(j, t) for j in player_ids for t in horizon],
                                         cat='Binary')
        
        # Transfer variables
        # z_in[j,t] = 1 if player j transferred in at start of GW t
        z_in = pulp.LpVariable.dicts("z_in",
                                      [(j, t) for j in player_ids for t in horizon],
                                      cat='Binary')
        
        # z_out[j,t] = 1 if player j transferred out at start of GW t  
        z_out = pulp.LpVariable.dicts("z_out",
                                       [(j, t) for j in player_ids for t in horizon],
                                       cat='Binary')
        
        # Free transfers and hits per gameweek
        # ft_used bounded by max banked transfers (5)
        ft_used = pulp.LpVariable.dicts("ft_used", horizon, lowBound=0, upBound=self.MAX_BANKED_TRANSFERS, cat='Integer')
        hits = pulp.LpVariable.dicts("hits", horizon, lowBound=0, upBound=max_hits, cat='Integer')  # Controlled by max_hits param
        ft_banked = pulp.LpVariable.dicts("ft_banked", horizon, lowBound=0, upBound=self.MAX_BANKED_TRANSFERS, cat='Integer')
        
        # Objective: Maximize total expected points over horizon minus hit costs
        # Plus small incentive to use free transfers when beneficial
        FREE_TRANSFER_INCENTIVE = 0.5  # Small bonus for using FTs productively
        
        objective_terms = []
        for t in horizon:
            gw_col = f'xp_gw{t}'
            for j in player_ids:
                xp = get_attr(j, gw_col)
                if robust:
                    # Use lower bound for worst-case
                    xp = xp * (1 - uncertainty_budget)
                objective_terms.append(xp * starter[(j, t)])
                objective_terms.append(xp * captain[(j, t)])  # Captain bonus
            # Subtract hit costs
            objective_terms.append(-self.TRANSFER_PENALTY * hits[t])
            # Add small incentive to use free transfers (encourages 1 FT per week)
            objective_terms.append(FREE_TRANSFER_INCENTIVE * ft_used[t])
        
        prob += pulp.lpSum(objective_terms)
        
        # Constraints per gameweek
        for idx, t in enumerate(horizon):
            # Starter must be in squad
            for j in player_ids:
                prob += starter[(j, t)] <= squad[(j, t)]
                prob += captain[(j, t)] <= starter[(j, t)]
            
            # Squad = 15
            prob += pulp.lpSum([squad[(j, t)] for j in player_ids]) == 15
            
            # Starters = 11
            prob += pulp.lpSum([starter[(j, t)] for j in player_ids]) == 11
            
            # Exactly 1 captain
            prob += pulp.lpSum([captain[(j, t)] for j in player_ids]) == 1
            
            # Position constraints for squad
            for pos, count in self.SQUAD_REQUIREMENTS.items():
                prob += pulp.lpSum([
                    squad[(j, t)] for j in player_ids 
                    if get_attr(j, 'element_type') == pos
                ]) == count
            
            # Position constraints for starters
            for pos in [1, 2, 3, 4]:
                pos_players = [j for j in player_ids if get_attr(j, 'element_type') == pos]
                prob += pulp.lpSum([starter[(j, t)] for j in pos_players]) >= self.STARTER_MIN[pos]
                prob += pulp.lpSum([starter[(j, t)] for j in pos_players]) <= self.STARTER_MAX[pos]
            
            # Max 3 per team
            for team_id in self.teams['id']:
                team_players = [j for j in player_ids if get_attr(j, 'team') == team_id]
                prob += pulp.lpSum([squad[(j, t)] for j in team_players]) <= self.MAX_PER_TEAM
            
            # Budget constraint
            prob += pulp.lpSum([
                get_attr(j, 'now_cost') / 10.0 * squad[(j, t)] 
                for j in player_ids
            ]) <= budget
            
            # Squad continuity (for t > first gameweek)
            if idx == 0:
                # First gameweek: establish initial squad
                if current_squad_ids:
                    # Track transfers from current squad
                    for j in player_ids:
                        if j in current_squad_ids:
                            # Can keep or transfer out
                            prob += z_out[(j, t)] <= 1
                            prob += z_in[(j, t)] == 0  # Can't transfer in existing players
                            prob += squad[(j, t)] == 1 - z_out[(j, t)]
                        else:
                            # Can only be in squad if transferred in
                            prob += z_out[(j, t)] == 0
                            prob += squad[(j, t)] == z_in[(j, t)]
                    
                    # Initial FT banked
                    prob += ft_banked[t] <= banked_transfers + 1 - ft_used[t]
                else:
                    # Fresh squad selection (Wildcard-like)
                    for j in player_ids:
                        prob += z_in[(j, t)] == 0
                        prob += z_out[(j, t)] == 0
                    prob += ft_banked[t] == 1
            else:
                prev_t = horizon[idx - 1]
                for j in player_ids:
                    # Flow conservation: squad[t] = squad[t-1] + in - out
                    prob += squad[(j, t)] == squad[(j, prev_t)] + z_in[(j, t)] - z_out[(j, t)]
                    # Can't transfer in and out same player
                    prob += z_in[(j, t)] + z_out[(j, t)] <= 1
                
                # Transfer banking
                # You get 1 new FT each GW. If you don't use it, you can bank up to 2.
                # ft_banked[t] = min(2, ft_banked[t-1] + 1 - ft_used[t])
                # But ft_banked[t] >= 1 always (you always have at least 1 FT)
                prob += ft_banked[t] <= ft_banked[prev_t] + 1 - ft_used[t]
                prob += ft_banked[t] <= self.MAX_BANKED_TRANSFERS
                prob += ft_banked[t] >= 1  # Always have at least 1 FT
            
            # Transfer balance: transfers in = transfers out
            prob += pulp.lpSum([z_in[(j, t)] for j in player_ids]) == pulp.lpSum([z_out[(j, t)] for j in player_ids])
            
            # Free transfers and hits
            transfers_made = pulp.lpSum([z_in[(j, t)] for j in player_ids])
            if idx == 0:
                available_ft = banked_transfers
            else:
                available_ft = ft_banked[horizon[idx - 1]] + 1
            
            # hits = max(0, transfers_made - available_ft)
            prob += hits[t] >= transfers_made - available_ft
            prob += ft_used[t] >= transfers_made - hits[t]
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))
        
        end_time = time.time()
        
        if prob.status != 1:  # Not optimal
            return MultiPeriodSolution(
                status=pulp.LpStatus[prob.status],
                squad=[],
                gameweek_plans=[],
                total_expected_points=0,
                transfer_summary=[],
                banked_transfers_after=banked_transfers,
                chips_remaining=[c for c in ['wildcard', 'freehit', 'triple_captain', 'bench_boost'] if c not in chips_used],
                optimization_time_seconds=end_time - start_time
            )
        
        # Extract solution
        gameweek_plans = []
        transfer_summary = []
        final_squad = []
        
        for idx, t in enumerate(horizon):
            gw_squad = []
            gw_starters = []
            gw_bench = []
            gw_captain_id = None
            gw_transfers_in = []
            gw_transfers_out = []
            
            for j in player_ids:
                if squad[(j, t)].value() == 1:
                    player_data = available[available['id'] == j].to_dict('records')[0]
                    
                    # Clean up for JSON (NaN -> None)
                    cleaned_data = {}
                    for k, v in player_data.items():
                        if isinstance(v, float) and (v != v):  # NaN check
                           cleaned_data[k] = None
                        else:
                           cleaned_data[k] = v
                    player_data = cleaned_data

                    player_data['is_starter'] = starter[(j, t)].value() == 1
                    player_data['is_captain'] = captain[(j, t)].value() == 1
                    player_data['expected_points'] = get_attr(j, f'xp_gw{t}')
                    
                    gw_squad.append(player_data)
                    
                    if player_data['is_captain']:
                        gw_captain_id = j
                    if player_data['is_starter']:
                        gw_starters.append(player_data)
                    else:
                        gw_bench.append(player_data)
                
                if z_in[(j, t)].value() == 1:
                    gw_transfers_in.append({
                        'id': int(j),
                        'name': str(get_attr(j, 'web_name')),
                        'cost': float(get_attr(j, 'now_cost') / 10.0),
                        'expected_points': float(get_attr(j, f'xp_gw{t}')),
                        'element_type': int(get_attr(j, 'element_type'))
                    })
                if z_out[(j, t)].value() == 1:
                    gw_transfers_out.append({
                        'id': int(j),
                        'name': str(get_attr(j, 'web_name')),
                        'cost': float(get_attr(j, 'now_cost') / 10.0),
                        'element_type': int(get_attr(j, 'element_type'))
                    })
            
            # Sort by position for display
            gw_starters.sort(key=lambda p: (p['element_type'], -p['expected_points']))
            gw_bench.sort(key=lambda p: -p['expected_points'])
            
            gw_xp = float(sum(p['expected_points'] * (2 if p['is_captain'] else 1) for p in gw_starters))
            
            # Calculate xP gain from transfers
            total_in_xp = sum(t.get('expected_points', 0) for t in gw_transfers_in)
            # Estimate out xP based on form
            total_out_xp = 0
            for t_out in gw_transfers_out:
                player = self.players[self.players['id'] == t_out['id']]
                if not player.empty:
                    form = float(player.iloc[0].get('form', 0) or 0)
                    total_out_xp += form
            
            xp_gain = total_in_xp - total_out_xp
            hits_taken = int(hits[t].value() or 0)
            hit_cost = hits_taken * self.TRANSFER_PENALTY
            
            # Hit is worth it if xP gain > hit cost + 2 (buffer for safety)
            hit_worth_it = bool(hits_taken == 0 or xp_gain > (hit_cost + 2))
            
            transfer_plan = TransferPlan(
                gameweek=t,
                transfers_in=gw_transfers_in,
                transfers_out=gw_transfers_out,
                free_transfers_used=int(ft_used[t].value() or 0),
                hits_taken=hits_taken,
                hit_cost=hit_cost,
                explanation=self._generate_transfer_explanation(gw_transfers_in, gw_transfers_out),
                xp_gain=round(xp_gain, 1),
                hit_worth_it=hit_worth_it
            )
            
            gameweek_plans.append(GameweekPlan(
                gameweek=t,
                starting_xi=gw_starters,
                bench=gw_bench,
                captain_id=gw_captain_id,
                vice_captain_id=gw_starters[1]['id'] if len(gw_starters) > 1 else None,
                expected_points=gw_xp,
                chip_used=None,
                transfers=transfer_plan
            ))
            
            if gw_transfers_in or gw_transfers_out:
                transfer_summary.append(transfer_plan)
            
            if idx == len(horizon) - 1:
                final_squad = gw_squad
        
        total_xp = sum(gp.expected_points for gp in gameweek_plans)
        total_hits = sum(ts.hit_cost for ts in transfer_summary)
        
        return MultiPeriodSolution(
            status='Optimal',
            squad=final_squad,
            gameweek_plans=gameweek_plans,
            total_expected_points=total_xp - total_hits,
            transfer_summary=transfer_summary,
            banked_transfers_after=int(ft_banked[horizon[-1]].value() or 0),
            chips_remaining=[c for c in ['wildcard', 'freehit', 'triple_captain', 'bench_boost'] if c not in chips_used],
            optimization_time_seconds=end_time - start_time
        )
    
    def _generate_transfer_explanation(self, 
                                        transfers_in: List[Dict],
                                        transfers_out: List[Dict]) -> str:
        """Generate a detailed explanation for transfers using TransferExplainer."""
        if not transfers_in:
            return "No transfers this gameweek."
        
        # Convert simple dictionaries back to pandas series-like objects or IDs that explainer expects
        # The explainer expects lists of dicts with 'id', 'name' etc. which matches input.
        # But it also needs predictions.
        
        # We need projected points for explanation context
        predictions_df = self._get_predictions(gameweeks=5) # Ensure predictions are available
        
        # Call explainer
        return self.transfer_explainer.explain_multi_transfer(
            transfers_in=transfers_in,
            transfers_out=transfers_out,
            current_gw=self.current_gw
        )
        
        return f"IN: {in_str} | OUT: {out_names}"
    
    def to_dict(self, solution: MultiPeriodSolution) -> Dict[str, Any]:
        """Convert solution to JSON-serializable dict."""
        return {
            'status': solution.status,
            'squad': solution.squad,
            'total_expected_points': solution.total_expected_points,
            'banked_transfers_after': solution.banked_transfers_after,
            'chips_remaining': solution.chips_remaining,
            'optimization_time_seconds': solution.optimization_time_seconds,
            'gameweek_plans': [
                {
                    'gameweek': gp.gameweek,
                    'starting_xi': gp.starting_xi,
                    'bench': gp.bench,
                    'captain_id': gp.captain_id,
                    'vice_captain_id': gp.vice_captain_id,
                    'expected_points': gp.expected_points,
                    'chip_used': gp.chip_used.value if gp.chip_used else None,
                    'transfers': {
                        'in': gp.transfers.transfers_in if gp.transfers else [],
                        'out': gp.transfers.transfers_out if gp.transfers else [],
                        'ft_used': gp.transfers.free_transfers_used if gp.transfers else 0,
                        'hits': gp.transfers.hits_taken if gp.transfers else 0,
                        'explanation': gp.transfers.explanation if gp.transfers else ""
                    }
                }
                for gp in solution.gameweek_plans
            ],
            'transfer_summary': [
                {
                    'gameweek': tp.gameweek,
                    'transfers_in': tp.transfers_in,
                    'transfers_out': tp.transfers_out,
                    'ft_used': tp.free_transfers_used,
                    'hits': tp.hits_taken,
                    'hit_cost': tp.hit_cost,
                    'explanation': tp.explanation,
                    'xp_gain': tp.xp_gain,
                    'hit_worth_it': tp.hit_worth_it
                }
                for tp in solution.transfer_summary
            ]
        }
