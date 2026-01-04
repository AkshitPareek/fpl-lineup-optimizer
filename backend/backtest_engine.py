"""
Backtest Engine for FPL Optimizer

Simulates the performance of the optimization algorithm over past gameweeks.
Implements a rolling horizon simulation (Receding Horizon Control).

Loop:
1. Set 'current time' to start of GW X.
2. Reconstruct player data/state for GW X.
3. Run Multi-Period Optimizer (looking ahead 3-5 GWs).
4. Implement the *immediate* decisions (transfers, captain) for GW X.
5. Calculate 'Actual Points' using historical results for GW X.
6. Advance state (squad, banked transfers) to GW X+1.
7. Repeat.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

from fpl_service import FPLService
from historical_data_service import HistoricalDataService
from advanced_optimizer import MultiPeriodFPLOptimizer

class BacktestEngine:
    def __init__(self):
        self.fpl_service = FPLService()
        self.history_service = HistoricalDataService()
        
        # Load static data once
        self.full_data = self.fpl_service.get_latest_data()
        self.teams_df = pd.DataFrame(self.full_data['static']['teams'])
        self.fixtures = self.full_data['fixtures']
        
        # Ensure we have history cached
        self.history_service.fetch_all_player_history(self.full_data['static']['elements'])
    
    def calculate_hindsight_optimal(self, gw: int, budget: float = 100.0) -> Tuple[float, List[Dict]]:
        """
        Calculate maximum achievable points for a gameweek with perfect knowledge.
        
        Uses MILP optimizer with actual points as objective function.
        
        Args:
            gw: Gameweek to calculate optimal for
            budget: Available budget
            
        Returns:
            Tuple of (max_points, optimal_squad_with_captain)
        """
        import pulp
        
        elements = self.full_data['static']['elements']
        
        # Build player data with actual scores
        players = []
        for p in elements:
            pid = p['id']
            actual_pts = self.history_service.get_actual_score(pid, gw)
            players.append({
                'id': pid,
                'web_name': p['web_name'],
                'team': p['team'],
                'element_type': p['element_type'],
                'now_cost': p['now_cost'],
                'actual_points': actual_pts
            })
        
        players_df = pd.DataFrame(players)
        player_ids = players_df['id'].tolist()
        
        # MILP Problem
        prob = pulp.LpProblem("Hindsight_Optimal", pulp.LpMaximize)
        
        # Variables
        squad = pulp.LpVariable.dicts("squad", player_ids, cat='Binary')
        starter = pulp.LpVariable.dicts("starter", player_ids, cat='Binary')
        captain = pulp.LpVariable.dicts("captain", player_ids, cat='Binary')
        
        def get_attr(pid, attr):
            return players_df.loc[players_df['id'] == pid, attr].values[0]
        
        # Objective: Maximize actual points (starters + captain bonus)
        prob += pulp.lpSum([
            get_attr(i, 'actual_points') * starter[i] +
            get_attr(i, 'actual_points') * captain[i]  # Captain gets double (1x + 1x)
            for i in player_ids
        ])
        
        # Constraints
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
        
        # Position constraints for squad (GK:2, DEF:5, MID:5, FWD:3)
        SQUAD_REQ = {1: 2, 2: 5, 3: 5, 4: 3}
        STARTER_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
        STARTER_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
        
        for pos, count in SQUAD_REQ.items():
            prob += pulp.lpSum([squad[i] for i in player_ids if get_attr(i, 'element_type') == pos]) == count
        
        # Position constraints for starters
        for pos in [1, 2, 3, 4]:
            pos_players = [i for i in player_ids if get_attr(i, 'element_type') == pos]
            prob += pulp.lpSum([starter[i] for i in pos_players]) >= STARTER_MIN[pos]
            prob += pulp.lpSum([starter[i] for i in pos_players]) <= STARTER_MAX[pos]
        
        # Max 3 per team
        for team_id in self.teams_df['id']:
            team_players = [i for i in player_ids if get_attr(i, 'team') == team_id]
            prob += pulp.lpSum([squad[i] for i in team_players]) <= 3
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
        
        if prob.status != 1:
            return 0.0, []
        
        # Extract solution
        max_points = pulp.value(prob.objective)
        optimal_squad = []
        for i in player_ids:
            if squad[i].value() == 1:
                optimal_squad.append({
                    'id': i,
                    'name': get_attr(i, 'web_name'),
                    'points': get_attr(i, 'actual_points'),
                    'is_starter': starter[i].value() == 1,
                    'is_captain': captain[i].value() == 1
                })
        
        return max_points, optimal_squad
    
    def run_backtest(self, 
                     start_gw: int, 
                     end_gw: int, 
                     initial_budget: float = 100.0,
                     horizon: int = 3) -> Dict[str, Any]:
        """
        Run backtest simulation.
        
        Args:
            start_gw: First gameweek to simulate
            end_gw: Last gameweek to simulate
            initial_budget: Starting budget
            horizon: Optimization lookahead
            
        Returns:
            Dictionary with results and metrics
        """
        print(f"Starting backtest from GW {start_gw} to {end_gw}...")
        
        results = []
        current_squad_ids = []  # Start empty -> imply Wildcard
        banked_transfers = 1
        budget = initial_budget
        
        # Track cumulative performance
        total_actual_points = 0
        total_predicted_points = 0
        
        range_gws = range(start_gw, end_gw + 1)
        
        for gw in range_gws:
            print(f"Simulating GW {gw}...")
            
            # 1. Reconstruct State
            players_df = self.history_service.get_gameweek_state(gw, self.full_data['static'])
            
            # 2. Run Optimization
            # If start (squad empty), use Wildcard logic implicitly or explicitly?
            # advanced_optimizer handles empty current_squad as "Free Selection" (Wildcard behavior)
            
            optimizer = MultiPeriodFPLOptimizer(
                players_df=players_df,
                teams_df=self.teams_df,
                fixtures=self.fixtures,
                current_gameweek=gw
            )
            
            # Predict future? 
            # Note: PointPredictor inside optimizer uses 'current_gameweek' to filter fixtures.
            # It uses 'ep_next' etc from players_df which we corrected.
            # But it also calculates xG points based on fixtures.
            # It uses 'fixture_map' which has all season fixtures. 
            # It will correcty see fixtures from gw, gw+1...
            
            solution = optimizer.optimize_multi_period(
                budget=budget,
                gameweeks=horizon,
                current_squad_ids=current_squad_ids,
                banked_transfers=banked_transfers,
                chips_used=[] # Assume no chips for simplicity in base test
            )
            
            if solution.status != "Optimal":
                print(f"Warning: GW {gw} optimization failed ({solution.status})")
                continue
                
            # 3. Implement Immediate Decision (GW 0 of the plan)
            plan = solution.gameweek_plans[0] # Plan for 'gw'
            
            # Extract decisions
            starters = plan.starting_xi
            bench = plan.bench
            captain_id = plan.captain_id
            vice_captain_id = plan.vice_captain_id
            transfers_made = plan.transfers.transfers_in
            hits_cost = plan.transfers.hits_taken * 4
            
            # 4. Calculate ACTUAL Score
            gw_actual_score = 0
            captain_played = False
            
            # Calculate starters points
            for p in starters:
                pid = p['id']
                points = self.history_service.get_actual_score(pid, gw)
                
                # Check minutes (simplified: assuming if points > 0 or minute > 0 they played)
                # Ideally get minutes from history.
                # Points can be 0 even if played (yellow card, goals conceded etc). 
                # Better to check minutes explicitly if possible, but get_actual_score returns total.
                # Assuming auto-sub logic is complex, for now we sum starters.
                
                if pid == captain_id:
                    gw_actual_score += points * 2
                    captain_played = True # Need better check for captain playing
                else:
                    gw_actual_score += points
            
            # Auto-sub logic (Simplified)
            # If captain didn't play (0 points?), VC gets double.
            # If starter didn't play, bench player comes on.
            # Implementing full auto-sub is complex; usually backtests use simplifications.
            # Or we check minutes. 
            
            # Let's subtract hits
            gw_net_score = gw_actual_score - hits_cost
            
            total_actual_points += gw_net_score
            total_predicted_points += plan.expected_points
            
            # 5. Update State for Next Turn
            # New squad is the squad from the plan
            current_squad_ids = [p['id'] for p in starters + bench]
            
            # Update banked transfers
            # Logic: min(5, prev_banked + 1 - used)
            ft_used = plan.transfers.free_transfers_used
            banked_transfers = min(5, banked_transfers + 1 - ft_used)
            
            # Budget update?
            # We should recalculate budget based on new squad value.
            # Simplified: Used budget from plan?
            # Or just assume we have the players and next step uses their 'now_cost' from history.
            # But we need to know 'bank'.
            # Budget = Squad Value + Bank.
            # We can track 'Bank'.
            # Initial: 100.0. Spent: X. Bank = 100 - X.
            # Next week: Budget = New Squad Value (at new prices) + Bank.
            # Wait, selling prices logic (sell at avg of buy/current) is hard without transaction history.
            # Simplification: Assume fixed 100m budget always available? Or track bank?
            # Let's track Bank.
            
            cost_of_squad = sum(p['now_cost'] for p in starters + bench) / 10.0
            bank = budget - cost_of_squad
            
            # Next week available budget = bank + squad value (at next week's prices).
            # We will recalculate 'budget' (total liquid funds) at start of next loop implicitly
            # by strictly tracking Bank and assuming standard prices.
            # Actually, `optimize_multi_period` takes `budget` as TOTAL (Squad + Bank).
            # So `budget` variable should remain relatively constant or grow/shrink with price changes.
            # We can just keep `budget` roughly constant or update it by `budget = cost_of_squad + bank`.
            # We use `budget` for next iter.
            budget = cost_of_squad + bank
            
            results.append({
                'gameweek': gw,
                'predicted_points': plan.expected_points,
                'actual_points': gw_actual_score,
                'hits_cost': hits_cost,
                'net_score': gw_net_score,
                'transfers_in': len(plan.transfers.transfers_in),
                'banked_transfers_next': banked_transfers,
                'squad_value': cost_of_squad
            })
            
        metrics = {
            'total_actual_points': total_actual_points,
            'total_predicted_points': total_predicted_points,
            'prediction_error': total_predicted_points - total_actual_points,
            'avg_points_per_gw': total_actual_points / len(range_gws)
        }
        
    def run_backtest_generator(self, 
                             start_gw: int, 
                             end_gw: int, 
                             initial_budget: float = 100.0,
                             horizon: int = 3):
        """
        Generator version of run_backtest for streaming progress.
        Yields: {"progress": int, "message": str, "result": dict}
        
        Now includes hindsight optimal (maximum achievable) points comparison.
        """
        yield {"progress": 1, "message": f"Initializing backtest (GW {start_gw}-{end_gw})..."}
        
        results = []
        current_squad_ids = []
        banked_transfers = 1
        budget = initial_budget
        
        total_actual_points = 0
        total_predicted_points = 0
        total_max_possible = 0
        
        range_gws = range(start_gw, end_gw + 1)
        total_steps = len(range_gws)
        
        for idx, gw in enumerate(range_gws):
            progress_pct = int(5 + (idx / total_steps) * 90)
            yield {"progress": progress_pct, "message": f"Simulating Gameweek {gw}..."}
            
            # 1. Reconstruct State
            players_df = self.history_service.get_gameweek_state(gw, self.full_data['static'])
            
            # 2. Run Optimization
            yield {"progress": progress_pct, "message": f"Optimizing GW {gw} strategy..."}
            
            optimizer = MultiPeriodFPLOptimizer(
                players_df=players_df,
                teams_df=self.teams_df,
                fixtures=self.fixtures,
                current_gameweek=gw
            )
            
            solution = optimizer.optimize_multi_period(
                budget=budget,
                gameweeks=horizon,
                current_squad_ids=current_squad_ids,
                banked_transfers=banked_transfers,
                chips_used=[]
            )
            
            if solution.status != "Optimal":
                continue
                
            # 3. Implement Immediate Decision
            plan = solution.gameweek_plans[0]
            starters = plan.starting_xi
            bench = plan.bench
            captain_id = plan.captain_id
            vice_captain_id = plan.vice_captain_id
            hits_cost = plan.transfers.hits_taken * 4
            
            # 4. Calculate ACTUAL Score
            gw_actual_score = 0
            for p in starters:
                pid = p['id']
                points = self.history_service.get_actual_score(pid, gw)
                if pid == captain_id:
                    gw_actual_score += points * 2
                else:
                    gw_actual_score += points
            
            gw_net_score = gw_actual_score - hits_cost
            total_actual_points += gw_net_score
            total_predicted_points += plan.expected_points
            
            # 5. Calculate HINDSIGHT OPTIMAL (max possible with perfect knowledge)
            gw_max_possible, _ = self.calculate_hindsight_optimal(gw, budget=initial_budget)
            total_max_possible += gw_max_possible
            
            # 6. Update State
            current_squad_ids = [p['id'] for p in starters + bench]
            ft_used = plan.transfers.free_transfers_used
            
            # For first GW (fresh squad), reset to 1 FT as baseline
            # For subsequent GWs, calculate properly with minimum of 1
            if idx == 0:  # First GW in backtest
                banked_transfers = 1  # Reset to baseline after initial squad pick
            else:
                # Normal rolling: bank = min(5, current + 1 - used)
                # But ensure at least 1 FT always available
                banked_transfers = max(1, min(5, banked_transfers + 1 - ft_used))
            
            cost_of_squad = sum(p['now_cost'] for p in starters + bench) / 10.0
            bank = budget - cost_of_squad
            budget = cost_of_squad + bank
            
            results.append({
                'gameweek': gw,
                'predicted_points': plan.expected_points,
                'actual_points': gw_actual_score,
                'max_possible': gw_max_possible,
                'hits_cost': hits_cost,
                'net_score': gw_net_score,
                'efficiency': (gw_actual_score / gw_max_possible * 100) if gw_max_possible > 0 else 0,
                'transfers_in': len(plan.transfers.transfers_in),
                'banked_transfers_next': banked_transfers,
                'squad_value': cost_of_squad,
                'squad': starters + bench
            })
            
        # Calculate efficiency metrics
        efficiency_pct = (total_actual_points / total_max_possible * 100) if total_max_possible > 0 else 0
        
        metrics = {
            'total_actual_points': total_actual_points,
            'total_predicted_points': total_predicted_points,
            'total_max_possible': total_max_possible,
            'prediction_error': total_predicted_points - total_actual_points,
            'avg_points_per_gw': total_actual_points / len(range_gws) if range_gws else 0,
            'efficiency_pct': efficiency_pct,
            'points_left_on_table': total_max_possible - total_actual_points
        }
        
        final_result = {
            'metrics': metrics,
            'weekly_results': results
        }
        
        yield {"progress": 100, "message": "Simulation Complete", "result": final_result}

if __name__ == "__main__":
    engine = BacktestEngine()
    # Test run on recent GWs
    # Note: Ensure you pick GWs where history_cache is populated and fixtures correct
    start = 1
    end = 5 # Short run
    
    summary = engine.run_backtest(start, end)
    print("\nBacktest Complete!")
    print(f"Total Points: {summary['metrics']['total_actual_points']}")
    
    df = pd.DataFrame(summary['weekly_results'])
    print(df[['gameweek', 'actual_points', 'predicted_points', 'net_score']])
