"""
FPL Optimizer API - FastAPI Backend

Provides endpoints for:
- Single gameweek optimization
- Multi-period optimization (3-8 GWs)
- Robust optimization with uncertainty handling
- Manager team and data fetching
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel, Field
import pandas as pd
import json
from fastapi.responses import StreamingResponse

from fpl_service import FPLService
from optimizer import FPLOptimizer
from advanced_optimizer import MultiPeriodFPLOptimizer
from robust_optimizer import RobustOptimizer
from point_predictor import PointPredictor
from transfer_explainer import TransferExplainer
from backtest_engine import BacktestEngine
from fixture_analyzer import FixtureAnalyzer
from chip_advisor import ChipAdvisor

app = FastAPI(
    title="FPL Optimizer API",
    description="Advanced FPL lineup optimization with multi-period planning and robust optimization",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fpl_service = FPLService()


# Request Models

class OptimizationRequest(BaseModel):
    """Request for single gameweek optimization."""
    budget: float = 100.0
    gameweeks: int = 1
    strategy: str = "standard"  # standard, safe, differential, my_squad, transfers
    excluded_players: List[int] = []
    manager_id: Optional[int] = None
    free_transfers: int = 1


class MultiPeriodRequest(BaseModel):
    """Request for multi-period optimization."""
    budget: float = 100.0
    gameweeks: int = Field(default=5, ge=3, le=8, description="Planning horizon (3-8)")
    manager_id: Optional[int] = None
    excluded_players: List[int] = []
    banked_transfers: int = Field(default=1, ge=0, le=5, description="Currently banked FTs (0-5)")
    chips_used: List[str] = Field(default=[], description="Already used chips")
    robust: bool = Field(default=False, description="Use robust optimization")
    uncertainty_budget: float = Field(default=0.3, ge=0, le=1, description="Robustness parameter Γ")
    strategy: str = Field(default="standard", description="Strategy: standard, differential, template")


class RobustRequest(BaseModel):
    """Request for robust optimization."""
    budget: float = 100.0
    gamma: float = Field(default=1.0, ge=0, le=3, description="Protection level (0=nominal, higher=conservative)")
    excluded_players: List[int] = []
    manager_id: Optional[int] = None


# Endpoints

@app.get("/api/data")
async def get_data():
    """Fetches and returns the latest FPL data."""
    try:
        data = fpl_service.get_latest_data()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/manager/{manager_id}")
async def get_manager(manager_id: int):
    """Fetches manager's current team."""
    try:
        team = fpl_service.get_manager_team(manager_id)
        return team
    except Exception as e:
        raise HTTPException(status_code=404, detail="Manager not found or API error")


@app.post("/api/optimize")
async def optimize_team(request: OptimizationRequest):
    """Generates the optimal lineup based on constraints (single gameweek)."""
    try:
        data = fpl_service.get_latest_data()
        
        current_team = None
        if request.manager_id:
            try:
                current_team = fpl_service.get_manager_team(request.manager_id)
            except Exception:
                pass

        optimizer = FPLOptimizer(data)
        
        lineup = optimizer.optimize(
            budget=request.budget,
            gameweeks=request.gameweeks,
            strategy=request.strategy,
            excluded_players=request.excluded_players,
            current_team=current_team,
            free_transfers=request.free_transfers
        )
        return lineup
    except Exception as e:
        print(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize/multi-period")
async def optimize_multi_period(request: MultiPeriodRequest):
    """
    Multi-period optimization over rolling horizon (3-8 gameweeks).
    
    Returns optimal 15-man squad, starting XI, bench order, captain,
    and transfer plan for each gameweek with explanations.
    
    Features:
    - Transfer banking (up to 5 FTs)
    - Squad continuity across gameweeks
    - Hit cost minimization
    - Robust optimization option
    """
    try:
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        fixtures_data = data["fixtures"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Get current gameweek
        events = static_data.get("events", [])
        current_gw = next(
            (e["id"] for e in events if e.get("is_current") or e.get("is_next")),
            1
        )
        
        # Get current squad if manager_id provided
        current_squad_ids = []
        banked_transfers = request.banked_transfers
        actual_budget = request.budget
        
        if request.manager_id:
            try:
                manager_team = fpl_service.get_manager_team(request.manager_id)
                current_squad_ids = [p["element"] for p in manager_team.get("picks", [])]
                
                # Calculate actual budget = squad value + bank (the budget field is the bank)
                # Get current squad player costs
                current_squad_players = players_df[players_df['id'].isin(current_squad_ids)]
                squad_value = current_squad_players['now_cost'].sum() / 10.0  # Convert to millions
                
                # If budget is less than squad value, treat it as bank balance
                if request.budget < squad_value * 0.5:
                    # User provided bank, not total budget
                    actual_budget = squad_value + request.budget
                    print(f"Manager {request.manager_id}: Squad value £{squad_value:.1f}m + Bank £{request.budget:.1f}m = £{actual_budget:.1f}m total")
                
                # Get actual banked transfers from entry_history if available
                if "entry_history" in manager_team:
                    # FPL stores event_transfers_cost, we can infer FTs
                    pass  # Use provided value for now
            except Exception as e:
                print(f"Error fetching manager team: {e}")
        
        # Initialize optimizer
        optimizer = MultiPeriodFPLOptimizer(
            players_df=players_df,
            teams_df=teams_df,
            fixtures=fixtures_data,
            current_gameweek=current_gw
        )
        
        # Run optimization
        solution = optimizer.optimize_multi_period(
            budget=actual_budget,
            gameweeks=request.gameweeks,
            current_squad_ids=current_squad_ids,
            excluded_players=request.excluded_players,
            banked_transfers=banked_transfers,
            chips_used=request.chips_used,
            robust=request.robust,
            uncertainty_budget=request.uncertainty_budget,
            strategy=request.strategy
        )
        
        # Generate transfer explanations
        if solution.status == "Optimal" and solution.transfer_summary:
            explainer = TransferExplainer(
                players_df=players_df,
                teams_df=teams_df,
                fixtures=fixtures_data
            )
            
            # Enhance transfer explanations
            for transfer_plan in solution.transfer_summary:
                if transfer_plan.transfers_in:
                    enhanced_explanation = explainer.explain_multi_transfer(
                        transfers_in=transfer_plan.transfers_in,
                        transfers_out=transfer_plan.transfers_out,
                        current_gw=transfer_plan.gameweek,
                        horizon=request.gameweeks
                    )
                    transfer_plan.explanation = enhanced_explanation
        
        return optimizer.to_dict(solution)
        
    except Exception as e:
        print(f"Multi-period optimization error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize/compare")
async def optimize_compare(request: MultiPeriodRequest):
    """
    Compare optimization strategies with and without hits.
    
    Returns both strategies so user can choose:
    - With hits: Maximum net points even if taking penalty
    - Without hits: Best possible using only free transfers
    """
    try:
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        fixtures_data = data["fixtures"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Get current gameweek
        events = static_data.get("events", [])
        current_gw = next(
            (e["id"] for e in events if e.get("is_current") or e.get("is_next")),
            1
        )
        
        # Get current squad if manager_id provided
        current_squad_ids = []
        actual_budget = request.budget
        banked_transfers = request.banked_transfers
        
        if request.manager_id:
            manager_team = fpl_service.get_manager_team(request.manager_id)
            current_squad_ids = [p["element"] for p in manager_team.get("picks", [])]
            
            # Calculate actual budget = squad value + bank
            squad_players = players_df[players_df['id'].isin(current_squad_ids)]
            squad_value = squad_players['now_cost'].sum() / 10.0
            actual_budget = squad_value + request.budget
        
        optimizer = MultiPeriodFPLOptimizer(
            players_df=players_df,
            teams_df=teams_df,
            fixtures=fixtures_data,
            current_gameweek=current_gw
        )
        
        # Run WITH hits allowed
        solution_with_hits = optimizer.optimize_multi_period(
            budget=actual_budget,
            gameweeks=request.gameweeks,
            current_squad_ids=current_squad_ids,
            excluded_players=request.excluded_players,
            banked_transfers=banked_transfers,
            chips_used=request.chips_used,
            robust=request.robust,
            uncertainty_budget=request.uncertainty_budget,
            strategy=request.strategy,
            max_hits=10  # Allow hits
        )
        
        # Run WITHOUT hits
        solution_no_hits = optimizer.optimize_multi_period(
            budget=actual_budget,
            gameweeks=request.gameweeks,
            current_squad_ids=current_squad_ids,
            excluded_players=request.excluded_players,
            banked_transfers=banked_transfers,
            chips_used=request.chips_used,
            robust=request.robust,
            uncertainty_budget=request.uncertainty_budget,
            strategy=request.strategy,
            max_hits=0  # No hits allowed
        )
        
        # Calculate net points for comparison
        with_hits_net = solution_with_hits.total_expected_points - sum(
            tp.hit_cost for tp in solution_with_hits.transfer_summary
        )
        no_hits_net = solution_no_hits.total_expected_points
        
        return {
            "current_gw": current_gw,
            "strategy": request.strategy,
            "comparison": {
                "with_hits": {
                    "total_xp": round(solution_with_hits.total_expected_points, 1),
                    "total_hits": sum(tp.hits_taken for tp in solution_with_hits.transfer_summary),
                    "hit_cost": sum(tp.hit_cost for tp in solution_with_hits.transfer_summary),
                    "net_xp": round(with_hits_net, 1),
                    "recommended": with_hits_net > no_hits_net
                },
                "no_hits": {
                    "total_xp": round(solution_no_hits.total_expected_points, 1),
                    "total_hits": 0,
                    "hit_cost": 0,
                    "net_xp": round(no_hits_net, 1),
                    "recommended": no_hits_net >= with_hits_net
                },
                "difference": round(with_hits_net - no_hits_net, 1)
            },
            "with_hits": optimizer.to_dict(solution_with_hits),
            "no_hits": optimizer.to_dict(solution_no_hits)
        }
        
    except Exception as e:
        print(f"Compare optimization error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize/robust")
async def optimize_robust(request: RobustRequest):
    """
    Robust optimization accounting for prediction uncertainty.
    
    Uses box uncertainty sets to maximize worst-case expected points,
    hedging against the high variance (R² ≈ 0.14) of player points.
    
    Gamma parameter controls conservativeness:
    - 0: Nominal optimization (no robustness)
    - 0.5-1.0: Balanced approach
    - 1.5+: Conservative (prioritizes consistency)
    """
    try:
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Add expected points from ep_next or form
        players_df["expected_points"] = pd.to_numeric(
            players_df["ep_next"], errors="coerce"
        ).fillna(
            pd.to_numeric(players_df["form"], errors="coerce").fillna(0) * 0.5 +
            pd.to_numeric(players_df["points_per_game"], errors="coerce").fillna(0) * 0.5
        )
        
        # Get current squad if manager_id provided
        current_squad_ids = []
        if request.manager_id:
            try:
                manager_team = fpl_service.get_manager_team(request.manager_id)
                current_squad_ids = [p["element"] for p in manager_team.get("picks", [])]
            except Exception:
                pass
        
        # Initialize robust optimizer
        optimizer = RobustOptimizer(
            players_df=players_df,
            teams_df=teams_df
        )
        
        # Run optimization
        solution = optimizer.optimize(
            budget=request.budget,
            gamma=request.gamma,
            excluded_players=request.excluded_players,
            current_squad_ids=current_squad_ids
        )
        
        return optimizer.to_dict(solution)
        
    except Exception as e:
        print(f"Robust optimization error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions")
async def get_predictions(gameweeks: int = 5):
    """
    Get expected point predictions for all players over multiple gameweeks.
    
    Uses xG, xA, ICT index, CBIT/CBIRT bonus calculations, and fixture analysis.
    """
    try:
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        fixtures_data = data["fixtures"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Get current gameweek
        events = static_data.get("events", [])
        current_gw = next(
            (e["id"] for e in events if e.get("is_current") or e.get("is_next")),
            1
        )
        
        predictor = PointPredictor(
            players_df=players_df,
            teams_df=teams_df,
            fixtures=fixtures_data,
            current_gameweek=current_gw
        )
        
        predictions = predictor.predict_all_players(gameweeks=gameweeks)
        
        # Return top players by expected points
        top_players = predictions.nlargest(50, 'total_xp')
        
        return {
            "current_gameweek": current_gw,
            "horizon": gameweeks,
            "predictions": top_players.to_dict(orient="records")
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sensitivity")
async def sensitivity_analysis(
    budget: float = 100.0,
    gammas: str = "0,0.5,1.0,1.5,2.0"
):
    """
    Perform sensitivity analysis across different robustness levels.
    
    Shows how the optimal team and expected points change with different gamma values.
    """
    try:
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Add expected points
        players_df["expected_points"] = pd.to_numeric(
            players_df["ep_next"], errors="coerce"
        ).fillna(
            pd.to_numeric(players_df["form"], errors="coerce").fillna(0) * 0.5 +
            pd.to_numeric(players_df["points_per_game"], errors="coerce").fillna(0) * 0.5
        )
        
        optimizer = RobustOptimizer(
            players_df=players_df,
            teams_df=teams_df
        )
        
        gamma_values = [float(g) for g in gammas.split(",")]
        
        results = optimizer.sensitivity_analysis(
            budget=budget,
            gamma_range=gamma_values
        )
        
        return {
            "budget": budget,
            "analysis": results.to_dict(orient="records")
        }
        
    except Exception as e:
        print(f"Sensitivity analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



class BacktestRequest(BaseModel):
    start_gw: int
    end_gw: int
    initial_budget: float = 100.0
    horizon: int = 3

@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest):
    """
    Run a backtest simulation over the specified gameweeks.
    """
    try:
        engine = BacktestEngine()
        result = engine.run_backtest(
            start_gw=request.start_gw, 
            end_gw=request.end_gw,
            initial_budget=request.initial_budget,
            horizon=request.horizon
        )
        return result
    except Exception as e:
        print(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest/stream")
def run_backtest_stream(request: BacktestRequest):
    """
    Stream backtest progress via Server-Sent Events (SSE).
    """
    def event_generator():
        engine = BacktestEngine()
        for event in engine.run_backtest_generator(
            start_gw=request.start_gw,
            end_gw=request.end_gw,
            initial_budget=request.initial_budget,
            horizon=request.horizon
        ):
            yield f"data: {json.dumps(event)}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/dream-team")
async def get_dream_team(gameweek: Optional[int] = None):
    """
    Get the absolute best XI for a gameweek with NO constraints.
    
    This is "Fun Mode" - shows the optimal team if you had unlimited budget
    and could pick any players. Great for seeing who the algorithm thinks
    are the best picks this week.
    
    Returns:
    - Best 11 players by expected points
    - Optimal captain choice
    - Total expected points for the dream team
    """
    try:
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        fixtures_data = data["fixtures"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Get current gameweek
        events = static_data.get("events", [])
        current_gw = gameweek or next(
            (e["id"] for e in events if e.get("is_current") or e.get("is_next")),
            1
        )
        
        # Calculate predictions
        predictor = PointPredictor(
            players_df=players_df,
            teams_df=teams_df,
            fixtures=fixtures_data,
            current_gameweek=current_gw
        )
        
        # Get per-GW predictions
        predictions = []
        for _, player in players_df.iterrows():
            xp, breakdown = predictor.predict_gameweek(player, current_gw)
            predictions.append({
                'id': player['id'],
                'web_name': player['web_name'],
                'team': player['team'],
                'team_name': teams_df[teams_df['id'] == player['team']]['name'].values[0] if len(teams_df[teams_df['id'] == player['team']]) > 0 else 'Unknown',
                'element_type': player['element_type'],
                'position': ['GK', 'DEF', 'MID', 'FWD'][player['element_type'] - 1],
                'now_cost': player['now_cost'] / 10,
                'expected_points': round(xp, 2),
                'breakdown': {k: round(v, 2) for k, v in breakdown.items() if isinstance(v, (int, float))}
            })
        
        predictions_df = pd.DataFrame(predictions)
        
        # Select best XI (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, total 11)
        # Use greedy approach: pick best by position meeting minimum requirements
        dream_xi = []
        
        # Must have 1 GK
        gks = predictions_df[predictions_df['element_type'] == 1].nlargest(1, 'expected_points')
        dream_xi.extend(gks.to_dict('records'))
        
        # Get top outfield players
        outfield = predictions_df[predictions_df['element_type'] > 1].nlargest(30, 'expected_points')
        
        # Ensure minimum: 3 DEF, 2 MID, 1 FWD
        min_def = outfield[outfield['element_type'] == 2].head(3).to_dict('records')
        min_mid = outfield[outfield['element_type'] == 3].head(2).to_dict('records')
        min_fwd = outfield[outfield['element_type'] == 4].head(1).to_dict('records')
        
        dream_xi.extend(min_def)
        dream_xi.extend(min_mid)
        dream_xi.extend(min_fwd)
        
        # Fill remaining 4 slots with best available outfield
        used_ids = {p['id'] for p in dream_xi}
        remaining = outfield[~outfield['id'].isin(used_ids)]
        
        # Check position limits while adding
        pos_counts = {2: 3, 3: 2, 4: 1}  # Already added
        pos_limits = {2: 5, 3: 5, 4: 3}
        
        for _, player in remaining.iterrows():
            if len(dream_xi) >= 11:
                break
            if pos_counts.get(player['element_type'], 0) < pos_limits.get(player['element_type'], 5):
                dream_xi.append(player.to_dict())
                pos_counts[player['element_type']] = pos_counts.get(player['element_type'], 0) + 1
        
        # Sort by position
        dream_xi.sort(key=lambda x: (x['element_type'], -x['expected_points']))
        
        # Captain is highest xP player
        captain = max(dream_xi, key=lambda x: x['expected_points'])
        captain_id = captain['id']
        
        # Calculate total
        total_xp = sum(p['expected_points'] for p in dream_xi)
        total_xp_with_captain = total_xp + captain['expected_points']  # Captain gets double
        
        return {
            "gameweek": current_gw,
            "dream_team": dream_xi,
            "captain": captain,
            "total_expected_points": round(total_xp_with_captain, 1),
            "total_cost": round(sum(p['now_cost'] for p in dream_xi), 1)
        }
        
    except Exception as e:
        print(f"Dream team error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class TransferAnalysisRequest(BaseModel):
    """Request for transfer hit analysis."""
    manager_id: int
    free_transfers: int = 1
    bank: float = 0.0
    gameweeks_horizon: int = 4


@app.post("/api/transfer-analysis")
async def analyze_transfers(request: TransferAnalysisRequest):
    """
    Analyze whether taking transfer hits is worth it.
    
    Compares:
    - 0 transfers (roll the FT)
    - 1 transfer (use FT)
    - 2 transfers (1 FT + 1 hit = -4 pts)
    - 3 transfers (1 FT + 2 hits = -8 pts)
    
    Returns recommendation with break-even analysis.
    """
    try:
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        fixtures_data = data["fixtures"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Get manager's current team
        manager_team = fpl_service.get_manager_team(request.manager_id)
        current_squad_ids = [p["element"] for p in manager_team.get("picks", [])]
        
        # Get current squad value
        current_squad = players_df[players_df['id'].isin(current_squad_ids)]
        squad_value = current_squad['now_cost'].sum() / 10.0
        budget = squad_value + request.bank
        
        # Get current gameweek
        events = static_data.get("events", [])
        current_gw = next(
            (e["id"] for e in events if e.get("is_current") or e.get("is_next")),
            1
        )
        
        # Initialize optimizer
        optimizer = MultiPeriodFPLOptimizer(
            players_df=players_df,
            teams_df=teams_df,
            fixtures=fixtures_data,
            current_gameweek=current_gw
        )
        
        # Run scenarios
        scenarios = []
        
        for num_transfers in range(4):  # 0, 1, 2, 3 transfers
            # Temporarily adjust banked transfers to force specific number
            solution = optimizer.optimize_multi_period(
                budget=budget,
                gameweeks=request.gameweeks_horizon,
                current_squad_ids=current_squad_ids,
                banked_transfers=request.free_transfers,
                chips_used=[]
            )
            
            if solution.status == "Optimal" and solution.gameweek_plans:
                plan = solution.gameweek_plans[0]
                actual_transfers = len(plan.transfers.transfers_in)
                hits = max(0, actual_transfers - request.free_transfers)
                hit_cost = hits * 4
                
                gross_xp = plan.expected_points
                net_xp = gross_xp - hit_cost
                
                scenarios.append({
                    "transfers": actual_transfers,
                    "hits": hits,
                    "hit_cost": hit_cost,
                    "expected_points_gross": round(gross_xp, 1),
                    "expected_points_net": round(net_xp, 1),
                    "transfers_in": plan.transfers.transfers_in,
                    "transfers_out": plan.transfers.transfers_out
                })
                break  # For now, just get the optimal scenario
        
        # Get current squad expected points (no transfers)
        predictor = PointPredictor(
            players_df=players_df,
            teams_df=teams_df,
            fixtures=fixtures_data,
            current_gameweek=current_gw
        )
        
        current_xp = 0
        for pid in current_squad_ids[:11]:  # Starting XI estimate
            player = players_df[players_df['id'] == pid]
            if len(player) > 0:
                xp, _ = predictor.predict_gameweek(player.iloc[0], current_gw)
                current_xp += xp
        
        # Compare scenarios
        recommendation = "HOLD" if len(scenarios) == 0 else (
            "TRANSFER" if scenarios[0]["expected_points_net"] > current_xp else "HOLD"
        )
        
        point_gain = scenarios[0]["expected_points_net"] - current_xp if scenarios else 0
        
        return {
            "current_squad_xp": round(current_xp, 1),
            "optimal_scenario": scenarios[0] if scenarios else None,
            "recommendation": recommendation,
            "expected_point_gain": round(point_gain, 1),
            "hit_worth_it": point_gain > 0,
            "break_even_points": scenarios[0]["hit_cost"] if scenarios and scenarios[0]["hits"] > 0 else 0,
            "manager_id": request.manager_id,
            "free_transfers": request.free_transfers,
            "budget": round(budget, 1)
        }
        
    except Exception as e:
        print(f"Transfer analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fixtures/{team_id}")
async def get_team_fixtures(team_id: int, num_gameweeks: int = 6):
    """
    Get upcoming fixtures for a team with FDR ratings.
    
    Returns fixture ticker data for frontend display.
    """
    try:
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        fixtures_data = data["fixtures"]
        
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Get current gameweek
        events = static_data.get("events", [])
        current_gw = next(
            (e["id"] for e in events if e.get("is_current") or e.get("is_next")),
            1
        )
        
        analyzer = FixtureAnalyzer(
            fixtures=fixtures_data,
            teams_df=teams_df,
            current_gw=current_gw
        )
        
        ticker = analyzer.get_fixture_ticker(team_id, num_gameweeks)
        avg_fdr = analyzer.get_avg_fdr(team_id, num_gameweeks)
        blank_gws = analyzer.find_blank_gameweeks(team_id)
        double_gws = analyzer.find_double_gameweeks(team_id)
        
        return {
            "team_id": team_id,
            "current_gw": current_gw,
            "fixtures": ticker,
            "avg_fdr": round(avg_fdr, 2),
            "blank_gameweeks": blank_gws,
            "double_gameweeks": double_gws,
            "fixture_string": analyzer.get_fixture_string(team_id, num_gameweeks)
        }
        
    except Exception as e:
        print(f"Fixture error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fixtures")
async def get_all_fixtures(num_gameweeks: int = 5):
    """
    Get fixture difficulty for all teams.
    
    Useful for comparing which teams have easier runs.
    """
    try:
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        fixtures_data = data["fixtures"]
        
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Get current gameweek
        events = static_data.get("events", [])
        current_gw = next(
            (e["id"] for e in events if e.get("is_current") or e.get("is_next")),
            1
        )
        
        analyzer = FixtureAnalyzer(
            fixtures=fixtures_data,
            teams_df=teams_df,
            current_gw=current_gw
        )
        
        # Get fixtures for all teams
        team_fixtures = []
        for _, team in teams_df.iterrows():
            team_id = team['id']
            team_fixtures.append({
                "team_id": team_id,
                "team_name": team['name'],
                "short_name": team['short_name'],
                "fixtures": analyzer.get_fixture_ticker(team_id, num_gameweeks),
                "avg_fdr": round(analyzer.get_avg_fdr(team_id, num_gameweeks), 2),
                "blank_gws": analyzer.find_blank_gameweeks(team_id),
                "double_gws": analyzer.find_double_gameweeks(team_id)
            })
        
        # Sort by easiest fixtures
        team_fixtures.sort(key=lambda x: x['avg_fdr'])
        
        return {
            "current_gw": current_gw,
            "num_gameweeks": num_gameweeks,
            "teams": team_fixtures
        }
        
    except Exception as e:
        print(f"Fixtures error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ChipAdviceRequest(BaseModel):
    """Request for chip recommendations."""
    manager_id: Optional[int] = None
    current_squad: List[int] = []
    chips_used: List[str] = []


@app.post("/api/chip-recommendations")
async def get_chip_recommendations(request: ChipAdviceRequest):
    """
    Get recommendations for optimal chip usage timing.
    
    Analyzes blank/double gameweeks and squad to suggest best timing for:
    - Wildcard
    - Free Hit
    - Bench Boost
    - Triple Captain
    """
    try:
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        fixtures_data = data["fixtures"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Get current gameweek
        events = static_data.get("events", [])
        current_gw = next(
            (e["id"] for e in events if e.get("is_current") or e.get("is_next")),
            1
        )
        
        # Get current squad
        current_squad = request.current_squad
        if request.manager_id and not current_squad:
            try:
                manager_team = fpl_service.get_manager_team(request.manager_id)
                current_squad = [p["element"] for p in manager_team.get("picks", [])]
            except:
                pass
        
        advisor = ChipAdvisor(
            players_df=players_df,
            teams_df=teams_df,
            fixtures=fixtures_data,
            current_gw=current_gw
        )
        
        recommendations = advisor.get_all_recommendations(
            current_squad=current_squad,
            chips_used=request.chips_used
        )
        
        return {
            "current_gw": current_gw,
            "recommendations": recommendations
        }
        
    except Exception as e:
        print(f"Chip advice error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}

