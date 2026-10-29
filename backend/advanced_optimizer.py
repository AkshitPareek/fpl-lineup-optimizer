"""
Advanced Multi-Period MILP Optimizer for FPL 2025/26

Implements a rolling horizon optimization over 3-15 gameweeks with:
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

from point_predictor import PointPredictor
from transfer_explainer import TransferExplainer
from chip_advisor import ChipAdvisor


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
    alternatives: List[Dict[str, Any]] = field(default_factory=list)  # Top N alternatives per position


@dataclass
class CaptainRecommendation:
    """Captain pick with reasoning."""
    player_id: int
    player_name: str
    expected_points: float
    has_dgw: bool
    fixtures: str
    confidence: str  # "high", "medium", "low"
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


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
    captain_recommendation: Optional[CaptainRecommendation] = None
    bank: Optional[float] = None  # In the bank after this week's transfers, in millions


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
    chip_suggestion: Optional[Dict[str, Any]] = None  # Legacy: single best chip
    chip_hold: Optional[List[Dict[str, Any]]] = None  # Chips saved, with the reason
    chip_recommendations: Optional[List[Dict[str, Any]]] = None  # All chip recommendations
    captain_recommendations: Optional[List[Dict[str, Any]]] = None  # Per-GW captain suggestions
    # False when the solver stopped at its time limit with its best plan so far.
    proven_optimal: bool = True


def ownership_adjustment(ownership: float, strategy: str) -> float:
    """Points added to one gameweek for Differential or Template. Standard adds none."""
    try:
        owned = float(ownership or 0)
    except (TypeError, ValueError):
        owned = 0.0
    if strategy == "differential":
        return max(0.0, 10.0 - owned) * 0.15
    if strategy == "template":
        if owned >= 20:
            return min(1.2, (owned - 20.0) * 0.04)
        if owned < 10:
            return -min(0.6, (10.0 - owned) * 0.06)
        return 0.0
    return 0.0


def robust_points(xp: float, sigma: float, uncertainty_budget: float) -> float:
    """Expected points after paying for this player's own doubt, not a flat haircut."""
    return max(0.0, float(xp) - float(uncertainty_budget) * float(sigma or 0))


def prediction_pool(players: pd.DataFrame, keep_ids: Optional[List[int]], per_position: int) -> pd.DataFrame:
    """Top points-per-game players in each position, plus anyone already in the squad."""
    ppg = pd.to_numeric(players.get('points_per_game'), errors='coerce').fillna(0)
    frame = players.assign(_ppg=ppg)
    keep = set(keep_ids or [])
    chosen = []
    for pos in (1, 2, 3, 4):
        pos_df = frame[frame['element_type'] == pos]
        if pos_df.empty:
            continue
        chosen.append(pos_df.nlargest(per_position, '_ppg'))
        chosen.append(pos_df[pos_df['id'].isin(keep)])
    if not chosen:
        return players.iloc[0:0].copy()
    return pd.concat(chosen, ignore_index=True).drop_duplicates('id').drop(columns='_ppg')


class MultiPeriodFPLOptimizer:
    """
    Multi-period MILP optimizer for FPL 2025/26 season.
    
    Features:
    - Rolling horizon optimization (3-15 gameweeks)
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
    LAST_GAMEWEEK = 38
    MAX_HORIZON = 15
    POOL_PER_POSITION = 30
    SOLVER_TIME_LIMIT = 15
    SOLVER_GAP = 0.01
    # Week k of the plan counts HORIZON_DECAY ** k. Replaying 2024/25 and 2025/26,
    # 0.9 and 0.8 both scored fewer real points than weighting every week equally.
    HORIZON_DECAY = 1.0
    # Share of a substitute's projection counted outside Bench Boost, roughly how
    # often each one comes on: the backup goalkeeper, then outfield substitutes in
    # order. Replaying 2024/25 and 2025/26 this beat a flat 0.1 by 40 points.
    BENCH_GK_WEIGHT = 0.03
    BENCH_WEIGHTS = (0.25, 0.08, 0.03)
    
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
            
    @classmethod
    def _multi_period_solver(cls):
        """HiGHS proves 12-15 week plans in seconds where CBC runs out of time."""
        options = dict(timeLimit=cls.SOLVER_TIME_LIMIT, gapRel=cls.SOLVER_GAP, threads=1)
        highs = pulp.HiGHS(msg=False, **options)
        if highs.available():
            return highs
        return pulp.PULP_CBC_CMD(msg=0, **options)

    @staticmethod
    def _chosen(var) -> bool:
        # HiGHS returns binaries within tolerance, e.g. 0.9999999.
        return (var.value() or 0) > 0.5

    @staticmethod
    def _count(var) -> int:
        return int(round(var.value() or 0))

    @staticmethod
    def _unlimited_transfers(active_chip, chip_gw, gameweek) -> bool:
        """Wildcard and Free Hit weeks do not spend free transfers or take hits."""
        return active_chip in ("wildcard", "freehit", "free_hit") and chip_gw == gameweek

    def _set_bank_after(self, prob, ft_banked_var, available, ft_used_var, cap_binary) -> None:
        """Set the next deadline's free transfers to min(5, available - used + 1)."""
        raw_next = available + 1 - ft_used_var
        cap = self.MAX_BANKED_TRANSFERS
        big_m = cap + 1
        prob += ft_banked_var <= cap
        prob += ft_banked_var <= raw_next
        prob += ft_banked_var >= raw_next - big_m * cap_binary
        prob += ft_banked_var >= cap - big_m * (1 - cap_binary)

    def _get_predictions(self, gameweeks: int) -> pd.DataFrame:
        """Get or compute predictions for the planning horizon."""
        if self.predictions is None:
            self.predictions = self.predictor.predict_all_players(gameweeks=gameweeks)
        return self.predictions
    
    def _filter_available_players(self, 
                                   excluded_ids: List[int] = [],
                                   include_injured: bool = False,
                                   keep_ids: List[int] = ()) -> pd.DataFrame:
        """Filter players who are available for selection. Players in keep_ids stay in the pool even when injured."""
        df = self.players.copy()
        
        # Exclude specified players
        df = df[~df['id'].isin(excluded_ids)]
        
        # An injured squad player can be benched; dropping him would force a transfer.
        if not include_injured:
            df = df[~df['status'].isin(['u', 'i']) | df['id'].isin(list(keep_ids))]
        
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
                               chip_to_use: Optional[Tuple[str, int]] = None,  # (chip_name, gameweek)
                               robust: bool = False,
                               uncertainty_budget: float = 0.3,
                               strategy: str = "standard",
                               max_hits: int = 5,
                               selling_prices: Optional[Dict[int, float]] = None) -> MultiPeriodSolution:
        """
        Multi-period optimization over rolling horizon.
        
        Args:
            budget: Total budget in millions: the current squad's selling value plus the bank
            selling_prices: What each current squad player sells for, in millions.
                Players missing here sell at their current price.
            gameweeks: Planning horizon (3-15, stopping at the last gameweek)
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
        banked_transfers = max(0, min(self.MAX_BANKED_TRANSFERS, int(banked_transfers)))
        
        remaining = max(1, self.LAST_GAMEWEEK + 1 - self.current_gw)
        gameweeks = min(max(3, min(self.MAX_HORIZON, gameweeks)), remaining)
        horizon = list(range(self.current_gw, self.current_gw + gameweeks))

        predictions = self._get_predictions(gameweeks=gameweeks)

        if strategy in ("differential", "template"):
            ownership_df = self.players[['id', 'selected_by_percent']].copy()
            ownership_df['selected_by_percent'] = pd.to_numeric(
                ownership_df['selected_by_percent'], errors='coerce'
            ).fillna(0)
            predictions = predictions.merge(ownership_df, on='id', how='left')
            predictions['selected_by_percent'] = predictions['selected_by_percent'].fillna(0)
            adjustment = predictions['selected_by_percent'].map(lambda own: ownership_adjustment(own, strategy))
            for col in [c for c in predictions.columns if c.startswith('xp_gw')]:
                predictions[col] = predictions[col] + adjustment * (predictions[col] > 0)
        
        available = self._filter_available_players(excluded_ids=excluded_players,
                                                   keep_ids=current_squad_ids or [])
        
        # Merge predictions
        gw_cols = [f'xp_gw{gw}' for gw in horizon]
        sigma_cols = [f'sigma_gw{gw}' for gw in horizon if f'sigma_gw{gw}' in predictions.columns]
        available = available.merge(
            predictions[['id'] + gw_cols + sigma_cols].drop_duplicates(),
            on='id',
            how='left'
        )
        for col in gw_cols + sigma_cols:
            available[col] = available[col].fillna(0)
        
        # Calculate total expected points for filtering
        available['total_horizon_xp'] = available[gw_cols].sum(axis=1)
        
        # ===== OPTIMIZATION: Pre-filter to top players per position =====
        # This reduces problem size from ~700 players to ~120, dramatically speeding up solve
        
        # Always include current squad players
        current_squad_set = set(current_squad_ids) if current_squad_ids else set()
        
        # Select top players per position
        filtered_players = []
        for pos in [1, 2, 3, 4]:
            pos_players = available[available['element_type'] == pos].copy()
            
            # Sort by total expected points over horizon
            pos_players = pos_players.sort_values('total_horizon_xp', ascending=False)
            
            # Take top N, but always include current squad players of this position
            top_pos = pos_players.head(self.POOL_PER_POSITION)
            current_pos = pos_players[pos_players['id'].isin(current_squad_set)]
            
            # Combine and deduplicate
            combined = pd.concat([top_pos, current_pos]).drop_duplicates(subset='id')
            filtered_players.append(combined)
        
        available = pd.concat(filtered_players, ignore_index=True)
        print(f"Optimization: Reduced player pool from {len(self.players)} to {len(available)} candidates")
        
        player_ids = available['id'].tolist()
        
        # Constraint building looks attributes up per player per gameweek.
        attrs_by_id = available.set_index('id').to_dict('index')

        def get_attr(pid, attr):
            return attrs_by_id.get(pid, {}).get(attr, 0)
        
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
        bank_cap = pulp.LpVariable.dicts("bank_cap", horizon, cat='Binary')
        # Money in the bank after each week's transfers, in tenths of a million.
        itb = pulp.LpVariable.dicts("itb", horizon, lowBound=0)
        
        # Objective: Maximize total expected points over horizon minus hit costs.
        # Transfers earn nothing by themselves; unused ones carry over in the bank.

        # Parse chip to use
        active_chip = None
        chip_gw = None
        if chip_to_use:
            active_chip, chip_gw = chip_to_use
        
        # bench_slot[j,t,k] = 1 if outfield player j is the k-th substitute at GW t.
        # Continuous is enough: with the squad and XI fixed this is an assignment
        # problem, and decreasing weights put the best substitute first.
        outfield_ids = [j for j in player_ids if get_attr(j, 'element_type') != 1]
        bench_slots = range(len(self.BENCH_WEIGHTS))
        bench_slot = pulp.LpVariable.dicts("bench_slot",
                                           [(j, t, k) for j in outfield_ids for t in horizon for k in bench_slots],
                                           lowBound=0, upBound=1)

        objective_terms = []
        for idx, t in enumerate(horizon):
            gw_col = f'xp_gw{t}'
            weight = self.HORIZON_DECAY ** idx
            
            # Check if this GW has active chip
            is_triple_captain = active_chip == 'triple_captain' and t == chip_gw
            is_bench_boost = active_chip == 'bench_boost' and t == chip_gw
            is_wildcard = active_chip == 'wildcard' and t == chip_gw
            is_free_hit = active_chip == 'free_hit' and t == chip_gw
            
            for j in player_ids:
                xp = float(get_attr(j, gw_col) or 0)
                if robust:
                    xp = robust_points(xp, get_attr(j, f'sigma_gw{t}') or 0, uncertainty_budget)
                
                xp *= weight

                # Starter points
                objective_terms.append(xp * starter[(j, t)])

                # Captain bonus (2x for normal, 3x total for Triple Captain)
                if is_triple_captain:
                    objective_terms.append(2 * xp * captain[(j, t)])  # 3x total (1x starter + 2x TC bonus)
                else:
                    objective_terms.append(xp * captain[(j, t)])  # 2x total (1x starter + 1x captain bonus)

                # Bench points
                if is_bench_boost:
                    # Full points for bench (squad - starter)
                    objective_terms.append(xp * (squad[(j, t)] - starter[(j, t)]))
                elif get_attr(j, 'element_type') == 1:
                    objective_terms.append(self.BENCH_GK_WEIGHT * xp * (squad[(j, t)] - starter[(j, t)]))
                else:
                    objective_terms.append(pulp.lpSum(
                        w * xp * bench_slot[(j, t, k)] for k, w in enumerate(self.BENCH_WEIGHTS)
                    ))

            # Hit costs (waived for Wildcard and Free Hit)
            if is_wildcard or is_free_hit:
                pass  # No penalty for unlimited transfers
            else:
                objective_terms.append(-weight * self.TRANSFER_PENALTY * hits[t])
        
        prob += pulp.lpSum(objective_terms)

        # Prices in tenths. Squad players sell at their selling price; anyone bought
        # in the plan sells for what he cost, since price changes are not forecast.
        current_set = set(current_squad_ids or [])
        selling_prices = selling_prices or {}
        buy_price = {j: int(get_attr(j, 'now_cost')) for j in player_ids}
        sell_price = {
            j: int(round(10 * selling_prices[j])) if j in current_set and j in selling_prices else buy_price[j]
            for j in player_ids
        }
        if current_set:
            start_bank = int(round(10 * budget)) - sum(sell_price[j] for j in current_set if j in sell_price)
        else:
            start_bank = int(round(10 * budget))
        free_hit_gw = chip_gw if active_chip in ('free_hit', 'freehit') else None

        def base_index(idx):
            """The week a squad is built from; a Free Hit squad is dropped at the next deadline. -1 is today's squad."""
            prev = idx - 1
            if prev >= 0 and horizon[prev] == free_hit_gw:
                prev -= 1
            return prev

        def owned_before(j, idx):
            prev = base_index(idx)
            if prev < 0:
                return 1 if j in current_set else 0
            return squad[(j, horizon[prev])]

        def bank_before(idx):
            prev = base_index(idx)
            return start_bank if prev < 0 else itb[horizon[prev]]
        
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

            # Each outfield substitute fills one bench slot, and each slot holds one player
            for j in outfield_ids:
                prob += pulp.lpSum(bench_slot[(j, t, k)] for k in bench_slots) == squad[(j, t)] - starter[(j, t)]
            for k in bench_slots:
                prob += pulp.lpSum(bench_slot[(j, t, k)] for j in outfield_ids) == 1
            
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
            
            fresh_squad = idx == 0 and not current_set
            available_ft = banked_transfers if idx == 0 else ft_banked[horizon[idx - 1]]
            if fresh_squad:
                # Picking a squad from scratch is not a transfer.
                for j in player_ids:
                    prob += z_in[(j, t)] == 0
                    prob += z_out[(j, t)] == 0
                prob += itb[t] == start_bank - pulp.lpSum(buy_price[j] * squad[(j, t)] for j in player_ids)
                prob += ft_banked[t] == 1
            else:
                for j in player_ids:
                    prob += squad[(j, t)] == owned_before(j, idx) + z_in[(j, t)] - z_out[(j, t)]
                    prob += z_in[(j, t)] + z_out[(j, t)] <= 1
                prob += itb[t] == bank_before(idx) + pulp.lpSum(
                    sell_price[j] * z_out[(j, t)] - buy_price[j] * z_in[(j, t)] for j in player_ids
                )
                if self._unlimited_transfers(active_chip, chip_gw, t):
                    # Saved transfers are kept through a Wildcard or Free Hit, but the week grants none.
                    prob += ft_used[t] == 0
                    prob += ft_banked[t] == available_ft
                else:
                    self._set_bank_after(prob, ft_banked[t], available_ft, ft_used[t], bank_cap[t])
            
            # Transfer balance: transfers in = transfers out
            # Note: This is removed because it is implied by the squad size (15) 
            # and the flow conservation equations. Keeping it causes infeasibility
            # if some current players are excluded (e.g. injured).
            # prob += pulp.lpSum([z_in[(j, t)] for j in player_ids]) == pulp.lpSum([z_out[(j, t)] for j in player_ids])
            
            # Free transfers and hits.
            # ft_banked[t] is the number available at the next deadline, so later
            # weeks spend that balance directly instead of granting another FT.
            transfers_made = pulp.lpSum([z_in[(j, t)] for j in player_ids])
            
            # Wildcard and Free Hit allow every transfer without a hit.
            if self._unlimited_transfers(active_chip, chip_gw, t):
                prob += hits[t] == 0
            else:
                prob += ft_used[t] <= transfers_made
                prob += ft_used[t] <= available_ft
                prob += hits[t] >= transfers_made - available_ft
                prob += ft_used[t] >= transfers_made - hits[t]
        
        solve_start = time.time()
        prob.solve(self._multi_period_solver())
        solve_time = time.time() - solve_start
        proven = prob.sol_status == pulp.LpSolutionOptimal
        print(f"MILP Solve time: {solve_time:.1f}s, Status: {pulp.LpStatus[prob.status]}, proven optimal: {proven}")
        
        end_time = time.time()
        
        if prob.status != 1:  # Not optimal
            return MultiPeriodSolution(
                status=pulp.LpStatus[prob.status],
                squad=[],
                gameweek_plans=[],
                total_expected_points=0,
                transfer_summary=[],
                banked_transfers_after=banked_transfers,
                chips_remaining=[c for c in ChipAdvisor.CHIPS if c not in chips_used],
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
                if self._chosen(squad[(j, t)]):
                    player_data = available[available['id'] == j].to_dict('records')[0]
                    
                    # Clean up for JSON (NaN -> None)
                    cleaned_data = {}
                    for k, v in player_data.items():
                        if isinstance(v, float) and (v != v):  # NaN check
                           cleaned_data[k] = None
                        else:
                           cleaned_data[k] = v
                    player_data = cleaned_data

                    player_data['is_starter'] = self._chosen(starter[(j, t)])
                    player_data['is_captain'] = self._chosen(captain[(j, t)])
                    player_data['expected_points'] = get_attr(j, f'xp_gw{t}')
                    
                    gw_squad.append(player_data)
                    
                    if player_data['is_captain']:
                        gw_captain_id = j
                    if player_data['is_starter']:
                        gw_starters.append(player_data)
                    else:
                        gw_bench.append(player_data)
                
                if self._chosen(z_in[(j, t)]):
                    gw_transfers_in.append({
                        'id': int(j),
                        'name': str(get_attr(j, 'web_name')),
                        'cost': float(get_attr(j, 'now_cost') / 10.0),
                        'expected_points': float(get_attr(j, f'xp_gw{t}')),
                        'element_type': int(get_attr(j, 'element_type'))
                    })
                if self._chosen(z_out[(j, t)]):
                    gw_transfers_out.append({
                        'id': int(j),
                        'name': str(get_attr(j, 'web_name')),
                        'cost': sell_price[j] / 10.0,
                        'expected_points': float(get_attr(j, f'xp_gw{t}')),
                        'element_type': int(get_attr(j, 'element_type'))
                    })
            
            # Sort by position for display. The bench is the goalkeeper, then
            # outfield substitutes in the order they come on.
            gw_starters.sort(key=lambda p: (p['element_type'], -p['expected_points']))
            def bench_position(p, t=t):
                if p['element_type'] == 1:
                    return -1
                return max(bench_slots, key=lambda k: bench_slot[(p['id'], t, k)].value() or 0)
            gw_bench.sort(key=lambda p: (p['element_type'] != 1, bench_position(p), -p['expected_points']))
            vice_id = max(
                (p for p in gw_starters if p['id'] != gw_captain_id),
                key=lambda p: p['expected_points'],
                default={'id': None},
            )['id']
            for player in gw_starters:
                player['is_vice_captain'] = player['id'] == vice_id
            
            gw_xp = float(sum(p['expected_points'] * (2 if p['is_captain'] else 1) for p in gw_starters))
            
            # Calculate xP gain from transfers
            total_in_xp = sum(t.get('expected_points', 0) for t in gw_transfers_in)
            total_out_xp = sum(t.get('expected_points', 0) for t in gw_transfers_out)
            
            xp_gain = total_in_xp - total_out_xp
            hits_taken = self._count(hits[t])
            hit_cost = hits_taken * self.TRANSFER_PENALTY
            
            # Hit is worth it if xP gain > hit cost + 2 (buffer for safety)
            hit_worth_it = bool(hits_taken == 0 or xp_gain > (hit_cost + 2))
            
            # Generate alternatives for this gameweek
            gw_squad_ids = [p['id'] for p in gw_squad]
            gw_alternatives = self.generate_transfer_alternatives(
                gw=t,
                current_squad_ids=gw_squad_ids,
                budget=budget,
                predictions_df=predictions,
                top_n=3
            )
            
            transfer_plan = TransferPlan(
                gameweek=t,
                transfers_in=gw_transfers_in,
                transfers_out=gw_transfers_out,
                free_transfers_used=self._count(ft_used[t]),
                hits_taken=hits_taken,
                hit_cost=hit_cost,
                explanation=self._generate_transfer_explanation(gw_transfers_in, gw_transfers_out),
                xp_gain=round(xp_gain, 1),
                hit_worth_it=hit_worth_it,
                alternatives=gw_alternatives
            )
            
            # Determine if chip was used this gameweek
            gw_chip_used = None
            if chip_to_use and chip_to_use[1] == t:
                chip_name = chip_to_use[0]
                chip_map = {
                    'triple_captain': Chip.TRIPLE_CAPTAIN,
                    'bench_boost': Chip.BENCH_BOOST,
                    'free_hit': Chip.FREE_HIT,
                    'wildcard': Chip.WILDCARD
                }
                gw_chip_used = chip_map.get(chip_name)
            
            gameweek_plans.append(GameweekPlan(
                gameweek=t,
                starting_xi=gw_starters,
                bench=gw_bench,
                captain_id=gw_captain_id,
                vice_captain_id=vice_id,
                expected_points=gw_xp,
                chip_used=gw_chip_used,
                transfers=transfer_plan,
                bank=round((itb[t].value() or 0) / 10.0, 1)
            ))
            
            if gw_transfers_in or gw_transfers_out:
                transfer_summary.append(transfer_plan)
            
            if t != free_hit_gw or not final_squad:
                final_squad = gw_squad
        
        total_xp = sum(gp.expected_points for gp in gameweek_plans)
        total_hits = sum(ts.hit_cost for ts in transfer_summary)
        
        # === Enhanced Chip & Captain Analysis ===
        chip_suggestion = None
        chip_hold = []
        chip_recommendations = []
        captain_recommendations = []
        
        if not chip_to_use:  # Only suggest if no chip already being used
            try:
                # Initialize ChipAdvisor
                chip_advisor = ChipAdvisor(
                    players_df=self.players,
                    teams_df=self.teams,
                    fixtures=self.fixtures,
                    current_gw=self.current_gw
                )
                
                # Convert gameweek_plans to dict format for analysis
                gw_plans_dicts = [
                    {
                        'gameweek': gp.gameweek,
                        'starting_xi': gp.starting_xi,
                        'bench': gp.bench
                    }
                    for gp in gameweek_plans
                ]
                
                # Get comprehensive chip and captain analysis
                analysis = chip_advisor.analyze_chip_opportunities_for_horizon(
                    gameweek_plans=gw_plans_dicts,
                    chips_used=chips_used,
                    predictions_df=predictions
                )
                
                chip_recommendations = analysis.get('chip_recommendations', [])
                captain_recommendations = analysis.get('captain_recommendations', [])
                chip_suggestion = analysis.get('best_chip')
                chip_hold = analysis.get('chip_hold', [])
                
                # Update gameweek_plans with captain recommendations
                for i, gp in enumerate(gameweek_plans):
                    if i < len(captain_recommendations):
                        cap_rec = captain_recommendations[i]
                        recommended = cap_rec.get('recommended_captain')
                        if recommended:
                            gp.captain_recommendation = CaptainRecommendation(
                                player_id=recommended.get('player_id', gp.captain_id),
                                player_name=recommended.get('player_name', ''),
                                expected_points=recommended.get('expected_points', 0),
                                has_dgw=recommended.get('has_dgw', False),
                                fixtures=recommended.get('fixtures', ''),
                                confidence=cap_rec.get('confidence', 'medium'),
                                alternatives=cap_rec.get('alternatives', [])
                            )
            except Exception as e:
                print(f"ChipAdvisor analysis error: {e}")
                # Fallback to basic analysis if ChipAdvisor fails
                chip_recommendations = []
                captain_recommendations = []
                chip_hold = []
        
        return MultiPeriodSolution(
            status='Optimal',
            squad=final_squad,
            gameweek_plans=gameweek_plans,
            total_expected_points=total_xp - total_hits,
            transfer_summary=transfer_summary,
            banked_transfers_after=self._count(ft_banked[horizon[-1]]),
            chips_remaining=[c for c in ChipAdvisor.CHIPS if c not in chips_used],
            optimization_time_seconds=end_time - start_time,
            chip_suggestion=chip_suggestion,
            chip_hold=chip_hold,
            chip_recommendations=chip_recommendations,
            captain_recommendations=captain_recommendations,
            proven_optimal=proven
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
    
    def generate_transfer_alternatives(self, 
                                        gw: int,
                                        current_squad_ids: List[int],
                                        budget: float,
                                        predictions_df: pd.DataFrame,
                                        top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Generate top N transfer alternatives per position for a given gameweek.
        
        Args:
            gw: Gameweek number
            current_squad_ids: IDs of players currently in squad
            budget: Available budget
            predictions_df: Predictions dataframe with xp_gw columns
            top_n: Number of alternatives per position (default 3)
            
        Returns:
            List of dicts, one per position, each containing ranked alternatives
        """
        alternatives = []
        position_names = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        xp_col = f'xp_gw{gw}'
        
        for pos in [1, 2, 3, 4]:
            # Find affordable players not in current squad
            candidates = predictions_df[
                (predictions_df['element_type'] == pos) &
                (~predictions_df['id'].isin(current_squad_ids)) &
                (predictions_df['now_cost'] / 10.0 <= budget)
            ].copy()
            
            if candidates.empty:
                continue
                
            # Rank by expected points for this GW
            if xp_col in candidates.columns:
                top_candidates = candidates.nlargest(top_n, xp_col)
            else:
                # Fallback to total_xp if GW-specific column missing
                top_candidates = candidates.nlargest(top_n, 'total_xp') if 'total_xp' in candidates.columns else candidates.head(top_n)
            
            options = []
            for _, player in top_candidates.iterrows():
                xp_value = float(player.get(xp_col, player.get('total_xp', 0)))
                options.append({
                    'id': int(player['id']),
                    'name': player['web_name'],
                    'team': player.get('team_name', ''),
                    'price': float(player['now_cost'] / 10.0),
                    'expected_points': round(xp_value, 1)
                })
            
            alternatives.append({
                'position': position_names[pos],
                'position_id': pos,
                'options': options
            })
        
        return alternatives
    
    def to_dict(self, solution: MultiPeriodSolution) -> Dict[str, Any]:
        """Convert solution to JSON-serializable dict."""
        return {
            'status': solution.status,
            'proven_optimal': solution.proven_optimal,
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
                    'bank': gp.bank,
                    'transfers': {
                        'in': gp.transfers.transfers_in if gp.transfers else [],
                        'out': gp.transfers.transfers_out if gp.transfers else [],
                        'ft_used': gp.transfers.free_transfers_used if gp.transfers else 0,
                        'hits': gp.transfers.hits_taken if gp.transfers else 0,
                        'explanation': gp.transfers.explanation if gp.transfers else "",
                        'alternatives': gp.transfers.alternatives if gp.transfers else []
                    },
                    'captain_recommendation': {
                        'player_id': gp.captain_recommendation.player_id,
                        'player_name': gp.captain_recommendation.player_name,
                        'expected_points': gp.captain_recommendation.expected_points,
                        'has_dgw': gp.captain_recommendation.has_dgw,
                        'fixtures': gp.captain_recommendation.fixtures,
                        'confidence': gp.captain_recommendation.confidence,
                        'alternatives': gp.captain_recommendation.alternatives
                    } if gp.captain_recommendation else None
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
                    'hit_worth_it': tp.hit_worth_it,
                    'alternatives': tp.alternatives
                }
                for tp in solution.transfer_summary
            ],
            'chip_suggestion': solution.chip_suggestion,
            'chip_hold': solution.chip_hold or [],
            'chip_recommendations': solution.chip_recommendations or [],
            'captain_recommendations': solution.captain_recommendations or []
        }

