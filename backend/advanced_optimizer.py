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

from point_predictor import PointPredictor, compute_uncertainty_bounds
from transfer_explainer import TransferExplainer
from chip_advisor import ChipAdvisor


class _HiGHS(pulp.HiGHS):
    """HiGHS that can start from the variables' current values, which PuLP's wrapper does not pass on."""
    warm_start = False

    def callSolver(self, lp):
        if self.warm_start:
            import highspy
            start = highspy.HighsSolution()
            start.col_value = [v.varValue if v.varValue is not None else 0.0 for v in lp.variables()]
            start.value_valid = True
            lp.solverModel.setSolution(start)
        lp.solverModel.run()


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
    chip_recommendations: Optional[List[Dict[str, Any]]] = None  # All chip recommendations
    captain_recommendations: Optional[List[Dict[str, Any]]] = None  # Per-GW captain suggestions
    # False when the solver stopped at its time limit with its best plan so far.
    proven_optimal: bool = True
    # Most the best possible plan could beat this one by, in objective points; None if unknown.
    optimality_gap_points: Optional[float] = None
    # Chips the plan plays, and chips it keeps for a better week after the horizon.
    chip_plan: List[Dict[str, Any]] = field(default_factory=list)
    chips_held: List[Dict[str, Any]] = field(default_factory=list)


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
    # Per position and week, so a Free Hit can reach one-week picks such as double-gameweek players.
    POOL_PER_POSITION_PER_WEEK = 8
    # Per position, so cheap bench fillers that free money for upgrades are always on offer.
    POOL_CHEAPEST_PER_POSITION = 3
    SOLVER_TIME_LIMIT = 15
    # The second solve, which also times the chips, starts from the chip-free plan.
    CHIP_SOLVER_TIME_LIMIT = 25
    # Chip timing often turns on fractions of a point, so the chip solve searches closer.
    CHIP_SOLVER_GAP = 0.002
    SOLVER_GAP = 0.01

    CHIP_NAMES = ('wildcard', 'free_hit', 'bench_boost', 'triple_captain')
    CHIP_ALIASES = {'freehit': 'free_hit', 'bboost': 'bench_boost', '3xc': 'triple_captain'}
    LAST_FIRST_HALF_GAMEWEEK = 19
    # Bench players count a tenth: roughly what they add through automatic substitutions.
    BENCH_WEIGHT = 0.1
    # A Wildcard squad is valued over this many weeks when deciding whether to keep the chip.
    WILDCARD_VALUE_WEEKS = 4
    # The Free Hit is considered in at most this many weeks, those where it gains most.
    FREE_HIT_CANDIDATE_WEEKS = 3
    # A chip is played only when it beats keeping it by this many points: a chip kept can
    # still be played once the projections update, so near ties go to keeping it.
    CHIP_KEEP_MARGIN = 0.5
    
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
        highs = _HiGHS(msg=False, **options)
        if highs.available():
            return highs
        return pulp.PULP_CBC_CMD(msg=0, **options)

    @classmethod
    def _solve_from_current_values(cls, prob, solver, time_limit=None) -> None:
        """Re-solve starting from the variables' current values."""
        solver.timeLimit = cls.CHIP_SOLVER_TIME_LIMIT if time_limit is None else time_limit
        if isinstance(solver, _HiGHS):
            solver.warm_start = True
            solver.gapRel = cls.CHIP_SOLVER_GAP
        else:
            solver.optionsDict['warmStart'] = True
            solver.optionsDict['gapRel'] = cls.CHIP_SOLVER_GAP
        prob.solve(solver)

    @staticmethod
    def _chosen(var) -> bool:
        # HiGHS returns binaries within tolerance, e.g. 0.9999999.
        return (var.value() or 0) > 0.5

    @staticmethod
    def _gap_points(prob) -> Optional[float]:
        """How far the solver's bound is above the plan it returned, from HiGHS; None for other solvers."""
        model = getattr(prob, 'solverModel', None)
        if model is None or not hasattr(model, 'getInfo'):
            return None
        info = model.getInfo()
        return round(max(0.0, abs(info.mip_dual_bound - info.objective_function_value)), 2)

    @staticmethod
    def _snapped(var) -> float:
        """The variable's value rounded if integer and clipped to its bounds, safe as a warm start."""
        value = var.varValue or 0
        if var.cat == pulp.LpInteger:
            value = round(value)
        if var.lowBound is not None:
            value = max(value, var.lowBound)
        if var.upBound is not None:
            value = min(value, var.upBound)
        return value

    @staticmethod
    def _count(var) -> int:
        return int(round(var.value() or 0))

    def _set_bank_after(self, prob, ft_banked_var, available, ft_used_var, cap_binary, chip_week=0) -> None:
        """
        Set the next deadline's free transfers to min(5, available - used + 1).
        A Wildcard or Free Hit week (chip_week = 1) keeps the saved transfers but adds none.
        """
        raw_next = available + 1 - ft_used_var - chip_week
        cap = self.MAX_BANKED_TRANSFERS
        big_m = cap + 1
        prob += ft_banked_var <= cap
        prob += ft_banked_var <= raw_next
        prob += ft_banked_var >= raw_next - big_m * cap_binary
        prob += ft_banked_var >= cap - big_m * (1 - cap_binary)

    @classmethod
    def _canonical_chip(cls, name: str) -> str:
        return cls.CHIP_ALIASES.get(name, name)

    @classmethod
    def _half(cls, gameweek: int) -> int:
        return 1 if gameweek <= cls.LAST_FIRST_HALF_GAMEWEEK else 2

    @classmethod
    def _half_end(cls, half: int) -> int:
        return cls.LAST_FIRST_HALF_GAMEWEEK if half == 1 else cls.LAST_GAMEWEEK

    def _chip_allowance(self, chips_used: List[str]) -> Dict[Tuple[str, int], int]:
        """
        Uses left of each chip in each half still to come. Every chip comes once
        up to GW19 and once more from GW20; chips_used covers the current half.
        """
        used = {self._canonical_chip(c) for c in chips_used}
        current = self._half(self.current_gw)
        return {
            (chip, half): int(half > current or chip not in used)
            for chip in self.CHIP_NAMES for half in (1, 2) if half >= current
        }

    def _best_squad(self, candidates: pd.DataFrame, weeks: List[int], budget_tenths: float,
                    price: Dict[int, float]) -> Tuple[float, Dict[str, Any]]:
        """
        The most a single 15-man squad, bought within budget and held unchanged, scores over
        these weeks: starters, captain and a tenth of the bench. Used to value Free Hit and
        Wildcard squads, and to seed the plan's chip search. Returns the score and the picks.
        """
        ids = candidates['id'].tolist()
        attrs = candidates.set_index('id').to_dict('index')
        prob = pulp.LpProblem("best_squad", pulp.LpMaximize)
        squad = pulp.LpVariable.dicts("s", ids, cat='Binary')
        start = pulp.LpVariable.dicts("x", [(j, w) for j in ids for w in weeks], cat='Binary')
        cap = pulp.LpVariable.dicts("c", [(j, w) for j in ids for w in weeks], cat='Binary')
        prob += pulp.lpSum(
            attrs[j][f'xp_gw{w}'] * (start[(j, w)] + cap[(j, w)] + self.BENCH_WEIGHT * (squad[j] - start[(j, w)]))
            for j in ids for w in weeks
        )
        prob += pulp.lpSum(price[j] * squad[j] for j in ids) <= budget_tenths
        prob += pulp.lpSum(squad[j] for j in ids) == 15
        for pos, count in self.SQUAD_REQUIREMENTS.items():
            prob += pulp.lpSum(squad[j] for j in ids if attrs[j]['element_type'] == pos) == count
        for team in {attrs[j]['team'] for j in ids}:
            prob += pulp.lpSum(squad[j] for j in ids if attrs[j]['team'] == team) <= self.MAX_PER_TEAM
        for w in weeks:
            prob += pulp.lpSum(start[(j, w)] for j in ids) == 11
            prob += pulp.lpSum(cap[(j, w)] for j in ids) == 1
            for j in ids:
                prob += start[(j, w)] <= squad[j]
                prob += cap[(j, w)] <= start[(j, w)]
            for pos in (1, 2, 3, 4):
                starters = pulp.lpSum(start[(j, w)] for j in ids if attrs[j]['element_type'] == pos)
                prob += starters >= self.STARTER_MIN[pos]
                prob += starters <= self.STARTER_MAX[pos]
        # One thread, like the plan's solver: HiGHS shares a thread pool sized by its first user.
        solver = pulp.HiGHS(msg=False, timeLimit=10, threads=1)
        prob.solve(solver if solver.available() else pulp.PULP_CBC_CMD(msg=0, timeLimit=10))
        picks = {
            'squad': [j for j in ids if self._chosen(squad[j])],
            'starters': {w: [j for j in ids if self._chosen(start[(j, w)])] for w in weeks},
            'captain': {w: next((j for j in ids if self._chosen(cap[(j, w)])), None) for w in weeks},
        }
        return (pulp.value(prob.objective) or 0.0) if prob.status == 1 else 0.0, picks

    def _xi_value(self, squad: pd.DataFrame, week: int) -> Tuple[float, float]:
        """Best starting XI plus captain for a fixed squad in one week, and the rest of the squad's points."""
        xp = squad[['element_type', f'xp_gw{week}']].rename(columns={f'xp_gw{week}': 'xp'})
        xp = xp.sort_values('xp', ascending=False)
        chosen = []
        for pos in (1, 2, 3, 4):
            chosen += xp[xp.element_type == pos].index[:self.STARTER_MIN[pos]].tolist()
        rest = xp.drop(chosen)
        rest = rest[rest.element_type != 1]
        counts = xp.loc[chosen].element_type.value_counts().to_dict()
        for idx, row in rest.iterrows():
            if len(chosen) == 11:
                break
            if counts.get(row.element_type, 0) < self.STARTER_MAX[row.element_type]:
                chosen.append(idx)
                counts[row.element_type] = counts.get(row.element_type, 0) + 1
        xi = xp.loc[chosen, 'xp']
        return float(xi.sum() + xi.max()), float(xp['xp'].sum() - xi.sum())

    def _new_squad_gain(self, pool, reference, window, budget_tenths, price, with_picks=False):
        """How much more the best squad for these weeks scores than the reference squad held unchanged."""
        by_window = pool.assign(_xp=pool[[f'xp_gw{g}' for g in window]].sum(axis=1))
        candidates = pd.concat([
            by_window[by_window.element_type == pos].nlargest(15, '_xp') for pos in (1, 2, 3, 4)
        ] + [by_window[by_window['id'].isin(reference.index)]]).drop_duplicates('id')
        best, picks = self._best_squad(candidates, window, budget_tenths, price)
        held = sum(xi + self.BENCH_WEIGHT * bench for xi, bench in (self._xi_value(reference, g) for g in window)) \
            if len(reference) == 15 else 0.0
        gain = max(0.0, best - held)
        return (gain, picks) if with_picks else gain

    def _chip_hold_values(self, allowance, horizon, pool, reference_ids, budget_tenths, price):
        """
        What each chip left unplayed in the horizon should still be worth: its gain in the best
        week left in its half after the horizon, giving each chip a different week. The plan then
        plays a chip only when a week in the horizon beats that.

        Gains beyond the horizon are measured against the squad held now, since later
        transfers are not planned that far ahead.
        """
        last = horizon[-1]
        later = {}
        for (chip, half), left in allowance.items():
            weeks = [w for w in range(last + 1, self._half_end(half) + 1) if self._half(w) == half]
            if left and weeks and any(self._half(t) == half for t in horizon):
                later[(chip, half)] = weeks
        if not later:
            return {}, {}

        reference = pool[pool['id'].isin(reference_ids)].set_index('id')
        gains = {}
        for (chip, half), weeks in later.items():
            for w in weeks:
                col = f'xp_gw{w}'
                if col not in pool.columns:
                    continue
                if (chip, w) in gains:
                    continue
                ref_bench = self._xi_value(reference, w)[1] if len(reference) == 15 else 0.0
                if chip == 'triple_captain':
                    gains[(chip, w)] = float(pool[col].max())
                elif chip == 'bench_boost':
                    gains[(chip, w)] = (1 - self.BENCH_WEIGHT) * ref_bench
                else:
                    window = [w] if chip == 'free_hit' else [
                        g for g in range(w, w + self.WILDCARD_VALUE_WEEKS) if f'xp_gw{g}' in pool.columns]
                    gains[(chip, w)] = self._new_squad_gain(pool, reference, window, budget_tenths, price)

        # Give each chip its own week, maximizing the total, as only one chip can be played a week.
        prob = pulp.LpProblem("chip_weeks", pulp.LpMaximize)
        keys = [(c, h, w) for (c, h), weeks in later.items() for w in weeks if (c, w) in gains]
        pick = pulp.LpVariable.dicts("p", keys, cat='Binary')
        prob += pulp.lpSum(gains[(c, w)] * pick[(c, h, w)] for c, h, w in keys)
        for (c, h) in later:
            prob += pulp.lpSum(pick[k] for k in keys if k[:2] == (c, h)) <= 1
        for w in {k[2] for k in keys}:
            prob += pulp.lpSum(pick[k] for k in keys if k[2] == w) <= 1
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        hold, best_week = {}, {}
        for c, h, w in keys:
            if self._chosen(pick[(c, h, w)]):
                hold[(c, h)] = gains[(c, w)]
                best_week[(c, h)] = w
        return hold, best_week

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
                               selling_prices: Optional[Dict[int, float]] = None,
                               optimize_chips: bool = False) -> MultiPeriodSolution:
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
            chip_to_use: Play this chip in this gameweek
            optimize_chips: Also choose which chips to play and when, among those not in
                chips_used. A chip is kept when a week after the horizon is worth more.
            
        Returns:
            MultiPeriodSolution with complete plan
        """
        import time
        start_time = time.time()
        banked_transfers = max(0, min(self.MAX_BANKED_TRANSFERS, int(banked_transfers)))
        
        remaining = max(1, self.LAST_GAMEWEEK + 1 - self.current_gw)
        gameweeks = min(max(3, min(self.MAX_HORIZON, gameweeks)), remaining)
        horizon = list(range(self.current_gw, self.current_gw + gameweeks))
        
        # Keeping a chip is weighed against the rest of its half, so project that far.
        chip_allowance = self._chip_allowance(chips_used) if optimize_chips else {}
        lookahead_end = max([horizon[-1]] + [
            self._half_end(half) for (_, half), left in chip_allowance.items()
            if left and any(self._half(t) == half for t in horizon)
        ])
        predictions = self._get_predictions(gameweeks=lookahead_end - self.current_gw + 1)
        
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
                # A blank gameweek stays at zero.
                predictions[col] = predictions[col] + differential_bonus * (predictions[col] > 0)
            print(f"Strategy: Differential - boosting {(ownership < 10).sum()} low-owned players")
        elif strategy == "template":
            # Favor high-ownership (safe picks), penalize low-ownership
            ownership = predictions['selected_by_percent']
            # Players > 20% owned get bonus, < 5% get penalty
            template_bonus = (ownership - 10).clip(lower=-5, upper=10) * 0.05
            for col in [c for c in predictions.columns if c.startswith('xp_gw')]:
                predictions[col] = predictions[col] + template_bonus * (predictions[col] > 0)
            print(f"Strategy: Template - favoring {(ownership > 20).sum()} popular players")
        
        available = self._filter_available_players(excluded_ids=excluded_players,
                                                   keep_ids=current_squad_ids or [])
        
        # Merge predictions
        gw_cols = [f'xp_gw{gw}' for gw in horizon]
        later_cols = [f'xp_gw{gw}' for gw in range(horizon[-1] + 1, lookahead_end + 1)
                      if f'xp_gw{gw}' in predictions.columns]
        merge_cols = ['id'] + gw_cols + later_cols
        available = available.merge(
            predictions[merge_cols].drop_duplicates(), 
            on='id', 
            how='left'
        )
        for col in gw_cols + later_cols:
            available[col] = available[col].fillna(0)
        
        # Calculate total expected points for filtering
        available['total_horizon_xp'] = available[gw_cols].sum(axis=1)
        
        # ===== OPTIMIZATION: Pre-filter the player pool =====
        # This reduces problem size from ~700 players to under 200, dramatically speeding up solve.
        # Kept per position: the best over the horizon, the best in each single week, the
        # cheapest, and the current squad.
        current_squad_set = set(current_squad_ids) if current_squad_ids else set()
        keep = set(current_squad_set)
        for pos in [1, 2, 3, 4]:
            pos_players = available[available['element_type'] == pos]
            keep.update(pos_players.nlargest(self.POOL_PER_POSITION, 'total_horizon_xp')['id'])
            for col in gw_cols:
                keep.update(pos_players.nlargest(self.POOL_PER_POSITION_PER_WEEK, col)['id'])
            fit = pos_players[pos_players['total_horizon_xp'] > 0]
            keep.update(fit.nsmallest(self.POOL_CHEAPEST_PER_POSITION, 'now_cost')['id'])
        available = available[available['id'].isin(keep)].reset_index(drop=True)
        print(f"Optimization: Reduced player pool from {len(self.players)} to {len(available)} candidates")
        
        player_ids = available['id'].tolist()
        
        # Constraint building looks attributes up per player per gameweek.
        attrs_by_id = available.set_index('id').to_dict('index')

        def get_attr(pid, attr):
            return attrs_by_id.get(pid, {}).get(attr, 0)
        
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
        fresh_start = not current_set

        # ===== Chips =====
        # use[(chip, t)] = 1 if the chip is played in GW t. A chip in chip_to_use is played in
        # its week; with optimize_chips the plan also chooses when to play the others.
        forced = (self._canonical_chip(chip_to_use[0]), int(chip_to_use[1])) if chip_to_use else None
        allowance = dict(chip_allowance)
        if forced and forced[1] in horizon:
            allowance[(forced[0], self._half(forced[1]))] = 1
        optional = {key: left for key, left in allowance.items()
                    if left and optimize_chips and not (forced and key == (forced[0], self._half(forced[1])))}
        hold, hold_week, free_hit_gain_now = {}, {}, {}
        if optional:
            price = {j: sell_price[j] if j in current_set else buy_price[j] for j in player_ids}
            hold, hold_week = self._chip_hold_values(optional, horizon, available, current_set,
                                                     int(round(10 * budget)), price)
            if current_set:
                # Each week's Free Hit gain over the squad held now: an upper bound on its gain
                # over the improved squad of the plan, used to pick the weeks it is considered in.
                reference = available[available['id'].isin(current_set)].set_index('id')
                for t in horizon:
                    if ('free_hit', self._half(t)) in optional:
                        free_hit_gain_now[t] = self._new_squad_gain(
                            available, reference, [t], int(round(10 * budget)), price)
        # What playing each optional chip gives up: its value later, plus the margin for keeping it.
        chip_cost = {key: hold.get(key, 0.0) + self.CHIP_KEEP_MARGIN for key in optional}

        def build(with_chips: bool, free_hit_weeks: set) -> Dict[str, Any]:
            """The plan's MILP. Without chips only a requested chip is played; with them the plan
            also times the others, with Free Hit squads considered in free_hit_weeks."""
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
        
            use = {}
            for idx, t in enumerate(horizon):
                for chip in self.CHIP_NAMES:
                    if forced == (chip, t):
                        pass
                    elif not with_chips or (chip, self._half(t)) not in optional:
                        continue
                    elif chip == 'free_hit' and current_set and t not in free_hit_weeks:
                        continue
                    if chip in ('wildcard', 'free_hit') and idx == 0 and fresh_start:
                        continue  # A squad picked from scratch already makes unlimited moves.
                    use[(chip, t)] = pulp.LpVariable(f"use_{chip}_{t}", cat='Binary')

            def chip_on(chip, t):
                return use.get((chip, t), 0)

            for t in horizon:
                prob += pulp.lpSum(use[(c, t)] for c in self.CHIP_NAMES if (c, t) in use) <= 1
            for (chip, half), left in allowance.items():
                in_half = [use[(chip, t)] for t in horizon if (chip, t) in use and self._half(t) == half]
                if in_half:
                    prob += pulp.lpSum(in_half) <= left
            if forced in use:
                prob += use[forced] == 1

            tc_weeks = [t for t in horizon if ('triple_captain', t) in use]
            bb_weeks = [t for t in horizon if ('bench_boost', t) in use]
            fh_weeks = [t for t in horizon if ('free_hit', t) in use]
            # Triple Captain adds the captain's points once more; Bench Boost counts the bench in full.
            tc = pulp.LpVariable.dicts("tc", [(j, t) for j in player_ids for t in tc_weeks], lowBound=0, upBound=1)
            bb = pulp.LpVariable.dicts("bb", [(j, t) for j in player_ids for t in bb_weeks], lowBound=0, upBound=1)
            # A Free Hit week is played by the squad with one-week swaps, fh_in for fh_out; the
            # squad carried between weeks stays put. Swaps rather than a second squad keep the
            # relaxation tight: a fractional chip can only make fractional swaps.
            fh_in = pulp.LpVariable.dicts("fh_in", [(j, t) for j in player_ids for t in fh_weeks], cat='Binary')
            fh_out = pulp.LpVariable.dicts("fh_out", [(j, t) for j in player_ids for t in fh_weeks], cat='Binary')

            def on_pitch(j, t):
                """The squad that plays GW t, after any Free Hit swaps."""
                if t in fh_weeks:
                    return squad[(j, t)] + fh_in[(j, t)] - fh_out[(j, t)]
                return squad[(j, t)]

            # Objective: expected points over the horizon minus hit costs, less the value of any
            # chip played now that would be worth more later in its half. Transfers earn nothing
            # by themselves; unused ones carry over in the bank.
            objective_terms = []
            for t in horizon:
                gw_col = f'xp_gw{t}'
                for j in player_ids:
                    xp = get_attr(j, gw_col)
                    if robust:
                        # Use lower bound for worst-case
                        xp = xp * (1 - uncertainty_budget)
                    # Starters, and the captain once more.
                    objective_terms.append(xp * (starter[(j, t)] + captain[(j, t)]))
                    objective_terms.append(self.BENCH_WEIGHT * xp * (on_pitch(j, t) - starter[(j, t)]))
                    if t in tc_weeks:
                        objective_terms.append(xp * tc[(j, t)])
                    if t in bb_weeks:
                        objective_terms.append((1 - self.BENCH_WEIGHT) * xp * bb[(j, t)])
                objective_terms.append(-self.TRANSFER_PENALTY * hits[t])
            for (chip, half), value in chip_cost.items():
                objective_terms.extend(-value * use[(chip, t)] for t in horizon
                                       if (chip, t) in use and self._half(t) == half)
            prob += pulp.lpSum(objective_terms)

            def squad_rules(member, t):
                prob.addConstraint(pulp.lpSum(member(j, t) for j in player_ids) == 15)
                for pos, count in self.SQUAD_REQUIREMENTS.items():
                    prob.addConstraint(pulp.lpSum(
                        member(j, t) for j in player_ids if get_attr(j, 'element_type') == pos) == count)
                for team_id in self.teams['id']:
                    team_players = [j for j in player_ids if get_attr(j, 'team') == team_id]
                    if team_players:
                        prob.addConstraint(pulp.lpSum(member(j, t) for j in team_players) <= self.MAX_PER_TEAM)

            # Constraints per gameweek
            for idx, t in enumerate(horizon):
                for j in player_ids:
                    prob += starter[(j, t)] <= on_pitch(j, t)
                    prob += captain[(j, t)] <= starter[(j, t)]
            
                squad_rules(lambda j, w: squad[(j, w)], t)
                prob += pulp.lpSum([starter[(j, t)] for j in player_ids]) == 11
                prob += pulp.lpSum([captain[(j, t)] for j in player_ids]) == 1
                for pos in [1, 2, 3, 4]:
                    pos_players = [j for j in player_ids if get_attr(j, 'element_type') == pos]
                    prob += pulp.lpSum([starter[(j, t)] for j in pos_players]) >= self.STARTER_MIN[pos]
                    prob += pulp.lpSum([starter[(j, t)] for j in pos_players]) <= self.STARTER_MAX[pos]

                if t in tc_weeks:
                    for j in player_ids:
                        prob += tc[(j, t)] <= captain[(j, t)]
                    prob += pulp.lpSum(tc[(j, t)] for j in player_ids) == use[('triple_captain', t)]
                if t in bb_weeks:
                    for j in player_ids:
                        prob += bb[(j, t)] <= on_pitch(j, t) - starter[(j, t)]
                        prob += bb[(j, t)] <= use[('bench_boost', t)]
                    # Implied by the integer squad, but without it a fractional bench spread over
                    # many players collects a whole Bench Boost from a fraction of the chip.
                    prob += pulp.lpSum(bb[(j, t)] for j in player_ids) <= 4 * use[('bench_boost', t)]
                if t in fh_weeks:
                    played = use[('free_hit', t)]
                    for j in player_ids:
                        prob += fh_in[(j, t)] <= played
                        prob += fh_out[(j, t)] <= played
                        prob += fh_in[(j, t)] <= 1 - squad[(j, t)]
                        prob += fh_out[(j, t)] <= squad[(j, t)]
                        prob += z_in[(j, t)] <= 1 - played
                        prob += z_out[(j, t)] <= 1 - played
                    prob += pulp.lpSum(fh_in[(j, t)] for j in player_ids) <= 15 * played
                    prob += pulp.lpSum(fh_out[(j, t)] for j in player_ids) <= 15 * played
                    squad_rules(on_pitch, t)
                    # The Free Hit squad costs the squad's selling value plus the bank, with kept
                    # players at their selling price: the swaps must fit in the bank.
                    prob += pulp.lpSum(buy_price[j] * fh_in[(j, t)] - sell_price[j] * fh_out[(j, t)]
                                       for j in player_ids) <= itb[t]

                unlimited = chip_on('wildcard', t) + chip_on('free_hit', t)
                available_ft = banked_transfers if idx == 0 else ft_banked[horizon[idx - 1]]
                if idx == 0 and fresh_start:
                    # Picking a squad from scratch is not a transfer.
                    for j in player_ids:
                        prob += z_in[(j, t)] == 0
                        prob += z_out[(j, t)] == 0
                    prob += itb[t] == start_bank - pulp.lpSum(buy_price[j] * squad[(j, t)] for j in player_ids)
                    prob += ft_banked[t] == 1
                else:
                    for j in player_ids:
                        owned = (1 if j in current_set else 0) if idx == 0 else squad[(j, horizon[idx - 1])]
                        prob += squad[(j, t)] == owned + z_in[(j, t)] - z_out[(j, t)]
                        prob += z_in[(j, t)] + z_out[(j, t)] <= 1
                    prob += itb[t] == (start_bank if idx == 0 else itb[horizon[idx - 1]]) + pulp.lpSum(
                        sell_price[j] * z_out[(j, t)] - buy_price[j] * z_in[(j, t)] for j in player_ids
                    )
                    # Saved transfers are kept through a Wildcard or Free Hit, but the week grants none.
                    self._set_bank_after(prob, ft_banked[t], available_ft, ft_used[t], bank_cap[t],
                                         chip_week=unlimited)

                # Free transfers and hits. ft_banked[t] is the number available at the next
                # deadline, so later weeks spend that balance directly. A Wildcard makes every
                # move free and a Free Hit leaves the squad alone, so neither spends any.
                transfers_made = pulp.lpSum([z_in[(j, t)] for j in player_ids])
                prob += ft_used[t] <= transfers_made
                prob += ft_used[t] <= available_ft
                prob += ft_used[t] <= self.MAX_BANKED_TRANSFERS * (1 - unlimited)
                prob += hits[t] >= transfers_made - available_ft - 15 * chip_on('wildcard', t)
                prob += ft_used[t] >= transfers_made - hits[t] - 15 * chip_on('wildcard', t)
            return dict(prob=prob, squad=squad, starter=starter, captain=captain, z_in=z_in, z_out=z_out, ft_used=ft_used, hits=hits, ft_banked=ft_banked, bank_cap=bank_cap, itb=itb, use=use, tc=tc, bb=bb, fh_in=fh_in, fh_out=fh_out)

        def seed_chips(m):
            """
            Play the optional chips as well as possible in the chip-free plan just found: each
            week's gain from each chip less the chip's value later, one chip a week. The joint
            solve starts from this, so it can only improve on it.
            """
            use, captain, squad, starter, z_in, itb, tc, bb, fh_in, fh_out = (
                m[k] for k in ("use", "captain", "squad", "starter", "z_in", "itb", "tc", "bb", "fh_in", "fh_out"))
            forced_week = forced[1] if forced in use else None
            seed_price = {j: sell_price[j] for j in player_ids}
            moves = {t: sum(1 for j in player_ids if self._chosen(z_in[(j, t)])) for t in horizon}
            base_chip_weeks = {t for (chip, t), var in use.items()
                               if chip in ('wildcard', 'free_hit') and (var.varValue or 0) > 0.5}

            def transfer_chain(chip_weeks):
                """
                Free transfers used, hits, transfers banked and whether the bank is full, per week,
                for the plan's moves when chip_weeks are Wildcard or Free Hit weeks. None if a
                week would need more than max_hits.
                """
                chain, bank = {}, banked_transfers
                for idx, t in enumerate(horizon):
                    if idx == 0 and fresh_start:
                        chain[t], bank = (0, 0, 1, 0), 1
                        continue
                    chip_week = int(t in chip_weeks)
                    used = 0 if chip_week else min(moves[t], bank)
                    hit = 0 if chip_week else moves[t] - used
                    if hit > max_hits:
                        return None
                    raw = bank + 1 - used - chip_week
                    bank = min(raw, self.MAX_BANKED_TRANSFERS)
                    chain[t] = (used, hit, bank, int(raw >= self.MAX_BANKED_TRANSFERS))
                return chain

            def hit_count(chain):
                return sum(row[1] for row in chain.values())

            base_hits = hit_count(transfer_chain(base_chip_weeks))
            gains, detail = {}, {}
            for (chip, t) in use:
                if (chip, t) == forced or t == forced_week or chip == 'wildcard':
                    continue
                if chip == 'triple_captain':
                    cap = next((j for j in player_ids if self._chosen(captain[(j, t)])), None)
                    gains[(chip, t)] = get_attr(cap, f'xp_gw{t}')
                    detail[(chip, t)] = cap
                elif chip == 'bench_boost':
                    bench = [j for j in player_ids if self._chosen(squad[(j, t)]) and not self._chosen(starter[(j, t)])]
                    gains[(chip, t)] = (1 - self.BENCH_WEIGHT) * sum(get_attr(j, f'xp_gw{t}') for j in bench)
                    detail[(chip, t)] = bench
                elif chip == 'free_hit':
                    if moves[t]:
                        continue  # The squad cannot also change in a Free Hit week.
                    # The week adds no free transfer, which can turn a later move into a hit.
                    chain = transfer_chain(base_chip_weeks | {t})
                    if chain is None:
                        continue
                    extra_hits = hit_count(chain) - base_hits
                    held_ids = [j for j in player_ids if self._chosen(squad[(j, t)])]
                    held = available[available['id'].isin(held_ids)].set_index('id')
                    budget_t = (itb[t].value() or 0) + sum(sell_price[j] for j in held_ids)
                    gains[(chip, t)], detail[(chip, t)] = self._new_squad_gain(
                        available, held, [t], budget_t, seed_price, with_picks=True)
                    gains[(chip, t)] -= self.TRANSFER_PENALTY * extra_hits
            net = {key: gain - chip_cost[(key[0], self._half(key[1]))] for key, gain in gains.items()}
            keys = [key for key, value in net.items() if value > 0]
            if not keys:
                return
            assign = pulp.LpProblem("chip_seed", pulp.LpMaximize)
            pick = pulp.LpVariable.dicts("p", keys, cat='Binary')
            assign += pulp.lpSum(net[k] * pick[k] for k in keys)
            for t in {k[1] for k in keys}:
                assign += pulp.lpSum(pick[k] for k in keys if k[1] == t) <= 1
            for (chip, half), left in optional.items():
                assign += pulp.lpSum(pick[k] for k in keys if k[0] == chip and self._half(k[1]) == half) <= left
            assign.solve(pulp.PULP_CBC_CMD(msg=0))
            chosen = sorted((k for k in keys if self._chosen(pick[k])), key=lambda k: -net[k])
            # Each Free Hit was costed alone; drop the weakest if together they need too many hits.
            while True:
                chain = transfer_chain(base_chip_weeks | {t for chip, t in chosen if chip == 'free_hit'})
                if chain is not None:
                    break
                weakest = max(i for i, k in enumerate(chosen) if k[0] == 'free_hit')
                chosen.pop(weakest)
            for t, (used, hit, banked, full) in chain.items():
                m['ft_used'][t].setInitialValue(used)
                m['hits'][t].setInitialValue(hit)
                m['ft_banked'][t].setInitialValue(banked)
                m['bank_cap'][t].setInitialValue(full)
            for (chip, t) in chosen:
                use[(chip, t)].setInitialValue(1)
                if chip == 'triple_captain':
                    tc[(detail[(chip, t)], t)].setInitialValue(1)
                elif chip == 'bench_boost':
                    for j in detail[(chip, t)]:
                        bb[(j, t)].setInitialValue(1)
                else:
                    picks = detail[(chip, t)]
                    for j in player_ids:
                        member = 1 if j in picks['squad'] else 0
                        owned = 1 if self._chosen(squad[(j, t)]) else 0
                        fh_in[(j, t)].setInitialValue(member * (1 - owned))
                        fh_out[(j, t)].setInitialValue(owned * (1 - member))
                        starter[(j, t)].setInitialValue(1 if j in picks['starters'][t] else 0)
                        captain[(j, t)].setInitialValue(1 if j == picks['captain'][t] else 0)

        solve_start = time.time()
        solver = self._multi_period_solver()
        model = build(with_chips=False, free_hit_weeks=set())
        model['prob'].solve(solver)
        def plant(target, values):
            for var in target['prob'].variables():
                var.setInitialValue(values.get(var.name) or 0)

        def solved_value(candidate):
            return pulp.value(candidate['prob'].objective) if candidate['prob'].status == 1 else None

        if optional and model['prob'].status == 1:
            # Plan without the optional chips first. Chips other than the Free Hit are settled
            # next, and the Free Hit search starts from that plan. It is kept only when it scores
            # higher: a Free Hit search that runs out of time cannot replace a better plan.
            chip_free = {var.name: self._snapped(var) for var in model['prob'].variables()}
            no_moves = {t for t in horizon
                        if not any(self._chosen(model['z_in'][(j, t)]) for j in player_ids)}
            free_hit_weeks = set()
            for half in (1, 2):
                weeks = [t for t in horizon if self._half(t) == half and t in free_hit_gain_now
                         and free_hit_gain_now[t] > chip_cost.get(('free_hit', half), 0.0)]
                weeks.sort(key=lambda t: -free_hit_gain_now[t])
                # The best weeks, and the best weeks the chip-free plan leaves without transfers,
                # where a Free Hit can be seeded without moving any.
                free_hit_weeks.update(weeks[:self.FREE_HIT_CANDIDATE_WEEKS])
                free_hit_weeks.update([t for t in weeks if t in no_moves][:self.FREE_HIT_CANDIDATE_WEEKS])
            base_values = chip_free
            if free_hit_weeks:
                no_fh = build(with_chips=True, free_hit_weeks=set())
                plant(no_fh, chip_free)
                seed_chips(no_fh)
                self._solve_from_current_values(no_fh['prob'], solver, self.SOLVER_TIME_LIMIT)
                if solved_value(no_fh) is not None:
                    model = no_fh
                    base_values = {var.name: self._snapped(var) for var in no_fh['prob'].variables()}
            chip_model = build(with_chips=True, free_hit_weeks=free_hit_weeks)
            plant(chip_model, base_values)
            seed_chips(chip_model)
            self._solve_from_current_values(chip_model['prob'], solver)
            chip_value = solved_value(chip_model)
            base_value = pulp.value(model['prob'].objective) or 0
            if chip_value is not None and chip_value >= base_value - 0.05:
                model = chip_model
            else:
                print(f"Free Hit search ({pulp.LpStatus[chip_model['prob'].status]}, {chip_value}) "
                      f"did not beat the plan without it ({base_value:.1f}); keeping that plan")
        (prob, squad, starter, captain, z_in, z_out, ft_used, hits, ft_banked, itb,
         use, tc, bb, fh_in, fh_out) = (model[k] for k in ('prob', 'squad', 'starter', 'captain', 'z_in', 'z_out', 'ft_used', 'hits', 'ft_banked', 'itb', 'use', 'tc', 'bb', 'fh_in', 'fh_out'))
        solve_time = time.time() - solve_start
        proven = prob.sol_status == pulp.LpSolutionOptimal
        gap_points = self._gap_points(prob)
        print(f"MILP Solve time: {solve_time:.1f}s, Status: {pulp.LpStatus[prob.status]}, proven optimal: {proven}, "
              f"gap: {gap_points} pts")
        
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
        
        chip_enum = {'triple_captain': Chip.TRIPLE_CAPTAIN, 'bench_boost': Chip.BENCH_BOOST,
                     'free_hit': Chip.FREE_HIT, 'wildcard': Chip.WILDCARD}
        hidden_cols = set(later_cols)
        planned_chips = []
        for idx, t in enumerate(horizon):
            gw_squad = []
            gw_starters = []
            gw_bench = []
            gw_captain_id = None
            gw_transfers_in = []
            gw_transfers_out = []
            chip_played = next((c for c in self.CHIP_NAMES if (c, t) in use and self._chosen(use[(c, t)])), None)
            free_hit = chip_played == 'free_hit'
            
            for j in player_ids:
                in_squad = self._chosen(squad[(j, t)])
                if free_hit:
                    plays = (in_squad and not self._chosen(fh_out[(j, t)])) or self._chosen(fh_in[(j, t)])
                else:
                    plays = in_squad
                if plays:
                    player_data = {k: v for k, v in attrs_by_id[j].items() if k not in hidden_cols}
                    player_data['id'] = j
                    
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
                
                # A Free Hit week lists its one-week swaps as transfers.
                if self._chosen(z_in[(j, t)]) or (free_hit and plays and not in_squad):
                    gw_transfers_in.append({
                        'id': int(j),
                        'name': str(get_attr(j, 'web_name')),
                        'cost': float(get_attr(j, 'now_cost') / 10.0),
                        'expected_points': float(get_attr(j, f'xp_gw{t}')),
                        'element_type': int(get_attr(j, 'element_type'))
                    })
                if self._chosen(z_out[(j, t)]) or (free_hit and in_squad and not plays):
                    gw_transfers_out.append({
                        'id': int(j),
                        'name': str(get_attr(j, 'web_name')),
                        'cost': sell_price[j] / 10.0,
                        'expected_points': float(get_attr(j, f'xp_gw{t}')),
                        'element_type': int(get_attr(j, 'element_type'))
                    })
            
            # Sort by position for display
            gw_starters.sort(key=lambda p: (p['element_type'], -p['expected_points']))
            gw_bench.sort(key=lambda p: -p['expected_points'])
            
            captain_multiplier = 3 if chip_played == 'triple_captain' else 2
            gw_xp = float(sum(p['expected_points'] * (captain_multiplier if p['is_captain'] else 1) for p in gw_starters))
            if chip_played == 'bench_boost':
                gw_xp += float(sum(p['expected_points'] for p in gw_bench))
            
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
            
            gw_chip_used = chip_enum.get(chip_played)
            if chip_played:
                planned_chips.append(self._planned_chip(chip_played, t, gw_starters, gw_bench, gw_captain_id,
                                                        squad, available, attrs_by_id, forced))
            
            gameweek_plans.append(GameweekPlan(
                gameweek=t,
                starting_xi=gw_starters,
                bench=gw_bench,
                captain_id=gw_captain_id,
                vice_captain_id=max(
                    (p for p in gw_starters if p['id'] != gw_captain_id),
                    key=lambda p: p['expected_points'],
                    default={'id': None},
                )['id'],
                expected_points=gw_xp,
                chip_used=gw_chip_used,
                transfers=transfer_plan,
                bank=round((itb[t].value() or 0) / 10.0, 1)
            ))
            
            if gw_transfers_in or gw_transfers_out:
                transfer_summary.append(transfer_plan)
            
            if not free_hit or not final_squad:
                final_squad = gw_squad
        
        total_xp = sum(gp.expected_points for gp in gameweek_plans)
        total_hits = sum(ts.hit_cost for ts in transfer_summary)
        
        # === Enhanced Chip & Captain Analysis ===
        chip_suggestion = None
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

        chips_held = [
            {'chip': chip, 'best_gameweek': hold_week[(chip, half)], 'expected_gain': round(value, 1)}
            for (chip, half), value in sorted(hold.items(), key=lambda kv: hold_week[kv[0]])
            if not any(p['chip'] == chip and self._half(p['gameweek']) == half for p in planned_chips)
        ]
        if optimize_chips:
            # The plan has already decided the chips, so suggest those rather than the rule-based advice.
            chip_recommendations = [
                {'chip': p['chip'], 'recommended_gameweek': p['gameweek'], 'estimated_gain': p['expected_gain'],
                 'reason': p['reason'], 'confidence': 'high', 'is_dgw': False}
                for p in planned_chips
            ] + [
                {'chip': h['chip'], 'recommended_gameweek': h['best_gameweek'], 'estimated_gain': h['expected_gain'],
                 'reason': f"Keep it: worth about {h['expected_gain']:.1f} pts in GW{h['best_gameweek']}, "
                           f"more than in any week of this plan",
                 'confidence': 'medium', 'is_dgw': False}
                for h in chips_held
            ]
            chip_suggestion = chip_recommendations[0] if planned_chips else None
        
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
            chip_recommendations=chip_recommendations,
            captain_recommendations=captain_recommendations,
            proven_optimal=proven,
            optimality_gap_points=gap_points,
            chip_plan=planned_chips,
            chips_held=chips_held
        )
    
    def _planned_chip(self, chip, t, starters, bench, captain_id, squad, available, attrs_by_id, forced):
        """A chip the plan plays, with what it adds that week."""
        captain_xp = next((p['expected_points'] for p in starters if p['id'] == captain_id), 0.0)
        gain, reason = None, ''
        if chip == 'triple_captain':
            gain = captain_xp
            name = next((p['web_name'] for p in starters if p['id'] == captain_id), 'captain')
            reason = f"{name} projected {captain_xp:.1f} pts"
        elif chip == 'bench_boost':
            gain = sum(p['expected_points'] for p in bench)
            reason = f"Bench projected {gain:.1f} pts"
        elif chip == 'free_hit':
            base = available[available['id'].isin(
                [j for j in attrs_by_id if self._chosen(squad[(j, t)])])].set_index('id')
            base_xi, base_bench = self._xi_value(base, t)
            played = sum(p['expected_points'] for p in starters) + captain_xp
            played_bench = sum(p['expected_points'] for p in bench)
            gain = played + self.BENCH_WEIGHT * played_bench - base_xi - self.BENCH_WEIGHT * base_bench
            reason = f"One-week squad projected {gain:.1f} pts more than the squad"
        else:
            reason = "Rebuilds the squad without hits"
        if forced == (chip, t):
            reason = f"Requested. {reason}".strip()
        return {'chip': chip, 'gameweek': t, 'expected_gain': None if gain is None else round(gain, 1),
                'reason': reason}

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
            'optimality_gap_points': solution.optimality_gap_points,
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
            'chip_plan': solution.chip_plan,
            'chips_held': solution.chips_held,
            'chip_suggestion': solution.chip_suggestion,
            'chip_recommendations': solution.chip_recommendations or [],
            'captain_recommendations': solution.captain_recommendations or []
        }

