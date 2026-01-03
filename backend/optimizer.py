import pulp
import pandas as pd

class FPLOptimizer:
    def __init__(self, data):
        self.static_data = data["static"]
        self.fixtures_data = data["fixtures"]
        self.players = pd.DataFrame(self.static_data["elements"])
        self.teams = pd.DataFrame(self.static_data["teams"])
        self.positions = pd.DataFrame(self.static_data["element_types"])
        
        # Preprocess Data
        self._preprocess_data()

    def _preprocess_data(self):
        # Map team names
        team_map = dict(zip(self.teams["id"], self.teams["name"]))
        self.players["team_name"] = self.players["team"].map(team_map)
        
        # Map position names
        pos_map = dict(zip(self.positions["id"], self.positions["singular_name_short"]))
        self.players["position"] = self.players["element_type"].map(pos_map)

        # Calculate expected points
        # Use 'ep_next' from FPL API as the primary source of truth
        # Fallback to form/ppg if ep_next is missing
        self.players["expected_points"] = pd.to_numeric(self.players["ep_next"], errors="coerce").fillna(
             pd.to_numeric(self.players["form"], errors="coerce").fillna(0) * 0.5 + 
             pd.to_numeric(self.players["points_per_game"], errors="coerce").fillna(0) * 0.5
        )

        # Replace all NaNs with None for JSON compliance
        self.players = self.players.where(pd.notnull(self.players), None)

    def optimize(self, budget=100.0, gameweeks=1, strategy="standard", excluded_players=[], current_team=None, free_transfers=1):
        """
        Solves the Linear Programming problem to find the best lineup.
        """
        # Filter available players
        current_player_ids = []
        if strategy == "my_squad" and current_team:
            # Optimize only within the current squad (Best XI)
            my_player_ids = [p["element"] for p in current_team["picks"]]
            available_players = self.players[self.players["id"].isin(my_player_ids)].copy()
            budget = 999.0 
        elif strategy == "transfers" and current_team:
            # Optimization for Transfers:
            # We want to pick a team that maximizes (Points - Transfer Cost)
            # We start with the full player pool
            available_players = self.players[
                ~self.players["id"].isin(excluded_players) & 
                (self.players["status"] != "u") & 
                (self.players["status"] != "i")
            ].copy()
            # We need to know who is in the current team to calculate transfers
            current_player_ids = [p["element"] for p in current_team["picks"]]
            
            # Smart Budget Calculation:
            # If we have entry_history, use the actual team value + bank
            # value = total value of squad, bank = money in bank. Both in 10ths.
            if "entry_history" in current_team:
                 total_budget_10ths = current_team["entry_history"]["value"] # Includes bank? No, value is squad value. Bank is separate.
                 # Actually FPL API: value = value of the team coordinates. bank = stored in bank.
                 # Total available = value + bank (approx, selling prices vary).
                 # Safer: value + bank.
                 budget = (current_team["entry_history"]["value"] + current_team["entry_history"]["bank"]) / 10.0
            else:
                 # Fallback if no history (shouldnt happen with real API)
                 budget = 100.0
        else:
            # Wildcard / Standard optimization
            available_players = self.players[
                ~self.players["id"].isin(excluded_players) & 
                (self.players["status"] != "u") & # Not unavailable
                (self.players["status"] != "i")   # Not injured (simplified)
            ].copy()
        
        # Adjust expected points based on strategy
        if strategy == "safe":
             # Boost high ownership
            available_players["expected_points"] *= (1 + pd.to_numeric(available_players["selected_by_percent"]) / 100)
        elif strategy == "differential":
             # Boost low ownership
            available_players["expected_points"] *= (1 + (100 - pd.to_numeric(available_players["selected_by_percent"])) / 500)

        # Setup LP Problem
        prob = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)
        
        # Decision Variables
        player_ids = available_players["id"].tolist()
        x = pulp.LpVariable.dicts("squad", player_ids, cat="Binary")   # Is in Squad (15 players)
        y = pulp.LpVariable.dicts("starter", player_ids, cat="Binary") # Is in Starting 11

        # Objective Function
        # We want to maximize points of the Starting 11 mostly, but also have a decent bench.
        # Objective = Sum(0.9 * XP * y) + Sum(0.1 * XP * x)
        
        obj_xp = pulp.lpSum([
            (0.9 * available_players.loc[available_players["id"]==i, "expected_points"].values[0] * y[i]) +
            (0.1 * available_players.loc[available_players["id"]==i, "expected_points"].values[0] * x[i]) 
            for i in player_ids
        ])
        
        if strategy == "transfers" and current_team:
            # Transfer Cost Logic (Same as before, based on x - squad changes)
            retained_vars = [x[i] for i in player_ids if i in current_player_ids]
            retained_count = pulp.lpSum(retained_vars)
            transfers_made = 15 - retained_count
            
            penalty = pulp.LpVariable("transfer_penalty", lowBound=0)
            prob += penalty >= 4 * (transfers_made - free_transfers)
            
            prob += obj_xp - penalty
        else:
            prob += obj_xp
        
        # Constraints
        
        # 0. Link x and y
        # If a player is a starter (y=1), they must be in the squad (x=1)
        for i in player_ids:
            prob += y[i] <= x[i]

        # Constraint 1: Budget (Based on SQUAD x)
        # Logic: Total Cost of Final Team <= Total Sellable Value of Current Team + Bank
        
        # 1. Create a map of ID -> Selling Price for current players
        selling_price_map = {}
        if current_team and "picks" in current_team:
            for pick in current_team["picks"]:
                if "selling_price" in pick:
                    selling_price_map[pick["element"]] = pick["selling_price"]

        # 2. Define Cost Function for Solver
        def get_player_cost(pid):
            if pid in selling_price_map:
                return selling_price_map[pid]
            else:
                return available_players.loc[available_players["id"]==pid, "now_cost"].values[0]

        # 3. Apply Constraint on x
        prob += pulp.lpSum([get_player_cost(i) / 10.0 * x[i] for i in player_ids]) <= budget

        # Constraint 2: Total Players (15 in squad x)
        prob += pulp.lpSum([x[i] for i in player_ids]) == 15

        # Constraint 3: Starting 11 (y)
        prob += pulp.lpSum([y[i] for i in player_ids]) == 11
        
        # Constraint 4: Position Constraints for SQUAD (x)
        # 2 GKs
        prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "element_type"].values[0] == 1]) == 2
        # 5 DEFs
        prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "element_type"].values[0] == 2]) == 5
        # 5 MIDs
        prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "element_type"].values[0] == 3]) == 5
        # 3 FWDs
        prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "element_type"].values[0] == 4]) == 3

        # Constraint 5: Position Constraints for STARTERS (y)
        # 1 GK
        prob += pulp.lpSum([y[i] for i in player_ids if available_players.loc[available_players["id"]==i, "element_type"].values[0] == 1]) == 1
        # Min 3 DEFs
        prob += pulp.lpSum([y[i] for i in player_ids if available_players.loc[available_players["id"]==i, "element_type"].values[0] == 2]) >= 3
        # Min 1 FWD
        prob += pulp.lpSum([y[i] for i in player_ids if available_players.loc[available_players["id"]==i, "element_type"].values[0] == 4]) >= 1
        
        # Constraint 6: Max 3 players per team on Squad
        for team_name in self.teams["name"]:
            prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "team_name"].values[0] == team_name]) <= 3
        # If 'my_squad', we still have 15 players, but we want to maximize the points of the starting 11.
        # However, the LP below optimizes the SUM of all selected players. 
        # For 'my_squad', we want to pick 11 starters.
        
        if strategy == "my_squad":
            # Select exactly 11 starters
            prob += pulp.lpSum([x[i] for i in player_ids]) == 11
             # Positional constraints for starting XI
            prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "position"].values[0] == "GKP"]) == 1
            prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "position"].values[0] == "DEF"]) >= 3
            prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "position"].values[0] == "MID"]) >= 2
            prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "position"].values[0] == "FWD"]) >= 1
            # Note: No max per position needed strictly as sum is 11, but standard rules apply (max 5 def, 5 mid, 3 fwd)
            # Implied by >= constraints and total 11? No, 1+3+2+1 = 7. need max constraints too.
            prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "position"].values[0] == "DEF"]) <= 5
            prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "position"].values[0] == "MID"]) <= 5
            prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "position"].values[0] == "FWD"]) <= 3

        else:
            # Standard Mode: Pick best 15 (Squad)
            prob += pulp.lpSum([x[i] for i in player_ids]) == 15
            
            # 3. Position Constraints
            # GK: 2
            prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "position"].values[0] == "GKP"]) == 2
            # DEF: 5
            prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "position"].values[0] == "DEF"]) == 5
            # MID: 5
            prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "position"].values[0] == "MID"]) == 5
            # FWD: 3
            prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "position"].values[0] == "FWD"]) == 3
            
            # 4. Max 3 players per team
            for team_name in self.teams["name"]:
                prob += pulp.lpSum([x[i] for i in player_ids if available_players.loc[available_players["id"]==i, "team_name"].values[0] == team_name]) <= 3

        # Solve
        # Use simple CBC solver
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract Results
        selected_players = []
        transfers_out = []
        transfers_in = []
        
        if pulp.LpStatus[prob.status] == "Optimal":
            for i in player_ids:
                if x[i].value() == 1:
                    player_data = available_players.loc[available_players["id"]==i].to_dict("records")[0]
                    # Specific cleanup for JSON serialization
                    cleaned_data = {}
                    for k, v in player_data.items():
                        if isinstance(v, float) and (v != v): # check for NaN
                            cleaned_data[k] = None
                        else:
                            cleaned_data[k] = v
                    # Add starter status
                    cleaned_data["is_starter"] = (y[i].value() == 1)
                    
                    selected_players.append(cleaned_data)
                    
                    if strategy == "transfers" and current_team and i not in current_player_ids:
                        transfers_in.append(cleaned_data["web_name"])
            
            if strategy == "transfers" and current_team:
                # Find transfers out
                selected_ids = [p["id"] for p in selected_players]
                transfers_out_data = []
                money_out_buys = 0
                money_in_sales = 0
                
                # Calculate Money Out (Cost of new players)
                for p in selected_players:
                     if p["id"] not in current_player_ids:
                         money_out_buys += p["now_cost"]

                # Calculate Money In (Value of sold players)
                for pid in current_player_ids:
                    if pid not in selected_ids:
                        # Fetch player details for report
                        player_row = self.players.loc[self.players["id"]==pid]
                        if not player_row.empty:
                            p_name = player_row.iloc[0]["web_name"]
                            transfers_out.append(p_name)
                            # Get selling price
                            s_price = selling_price_map.get(pid, player_row.iloc[0]["now_cost"])
                            money_in_sales += s_price
                            transfers_out_data.append({"name": p_name, "price": s_price})

        # Sort: Starters first, then Bench. Within each, by expected points.
        selected_players.sort(key=lambda p: (not p["is_starter"], -p["expected_points"]))
        
        # Calculate Bank
        # Total Budget Available was passed as `budget` (Team Value + Bank)
        # Budget Used is cost of final team.
        # Bank = Budget Available - Budget Used
        final_team_cost = sum(p["now_cost"] if p["id"] not in current_player_ids else (selling_price_map.get(p["id"], p["now_cost"])) for p in selected_players) / 10.0
        # Wait, budget constraint used mixed costs (selling price for kept, now_cost for new).
        # So `final_team_cost` logic above matches the constraint.
        bank = budget - final_team_cost

        return {
            "status": pulp.LpStatus[prob.status],
            "total_expected_points": pulp.value(prob.objective) if strategy != "transfers" else sum(p["expected_points"] for p in selected_players if p["is_starter"]), # Show Starter Points mostly? Or Squad? Let's show Squad total or Starter Total? Usually users care about Gameweek projected points (Starting 11).
            # ACTUALLY: Objective was weighted. `pulp.value(prob.objective)` is 0.9*Start + 0.1*Bench.
            # We should probably return the "Starting 11 Expected Points" as the headline number.
            "lineup": selected_players,
            "budget_used": final_team_cost,
            "bank": round(bank, 1),
            "transfers": {
                "in": transfers_in,
                "out": transfers_out,
                "financials": {
                    "money_in": money_in_sales / 10.0,
                    "money_out": money_out_buys / 10.0,
                    "net": (money_in_sales - money_out_buys) / 10.0
                } if strategy == "transfers" and current_team else None,
                "cost": (len(transfers_in) - free_transfers) * 4 if len(transfers_in) > free_transfers else 0
            } if strategy == "transfers" else None
        }
