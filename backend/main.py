"""
FPL Optimizer API - FastAPI Backend

Provides endpoints for:
- Single gameweek optimization
- Multi-period optimization (3-8 GWs)
- Robust optimization with uncertainty handling
- Manager team and data fetching
"""

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import pandas as pd
import json

from fpl_service import FPLService
from optimizer import FPLOptimizer
from advanced_optimizer import MultiPeriodFPLOptimizer
from robust_optimizer import RobustOptimizer
from point_predictor import PointPredictor
from transfer_explainer import TransferExplainer
from backtest_engine import BacktestEngine
from fixture_analyzer import FixtureAnalyzer
from chip_advisor import ChipAdvisor


def get_active_gameweek(events: list) -> int:
    """
    Get the active gameweek for predictions.
    
    If the current gameweek is finished, returns the next gameweek.
    This ensures predictions are always for upcoming matches.
    """
    current_gw = None
    next_gw = None
    
    for e in events:
        if e.get("is_current"):
            current_gw = e
        if e.get("is_next"):
            next_gw = e
    
    # If current GW is finished, use next GW
    if current_gw and current_gw.get("finished") and next_gw:
        return next_gw["id"]
    
    # Otherwise use current (or next if no current)
    if current_gw:
        return current_gw["id"]
    if next_gw:
        return next_gw["id"]
    
    return 1  # Fallback

app = FastAPI(
    title="FPL Optimizer API",
    description="Advanced FPL lineup optimization with multi-period planning and robust optimization",
    version="2.0.0"
)

import os

# Configure CORS - Must be configured before routes
# For production, explicitly list allowed origins
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://fploptimize.netlify.app",
    "https://www.fploptimize.netlify.app",
]

# Add environment variable origin if set
env_frontend_url = os.getenv("FRONTEND_URL", "")
if env_frontend_url:
    ALLOWED_ORIGINS.append(env_frontend_url.rstrip("/"))

# Remove duplicates and empty strings
ALLOWED_ORIGINS = list(set(o.rstrip("/") for o in ALLOWED_ORIGINS if o))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)


# Startup event to log CORS configuration
@app.on_event("startup")
async def startup_event():
    print(f"🚀 FPL Optimizer API starting...")
    print(f"📋 CORS allowed origins: {ALLOWED_ORIGINS}")


# Custom exception handler to ensure CORS headers on errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions and ensure CORS headers are included."""
    origin = request.headers.get("origin", "")
    
    # Build response
    response = JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )
    
    # Add CORS headers if origin is allowed
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    
    print(f"Exception handler: {exc}")
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions and ensure CORS headers are included."""
    origin = request.headers.get("origin", "")
    
    response = JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
    
    # Add CORS headers if origin is allowed
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response


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
    chip_to_use: Optional[List] = Field(default=None, description="Chip to activate: [chip_name, gameweek]")
    robust: bool = Field(default=False, description="Use robust optimization")
    uncertainty_budget: float = Field(default=0.3, ge=0, le=1, description="Robustness parameter Γ")
    strategy: str = Field(default="standard", description="Strategy: standard, differential, template")


class RobustRequest(BaseModel):
    """Request for robust optimization."""
    budget: float = 100.0
    gamma: float = Field(default=1.0, ge=0, le=3, description="Protection level (0=nominal, higher=conservative)")
    excluded_players: List[int] = []
    manager_id: Optional[int] = None



# Helper for analytics enrichment
def _enrich_with_analytics(response_dict: Dict, players_df: pd.DataFrame, teams_df: pd.DataFrame):
    """Enrich optimization result with EV and ownership analytics."""
    try:
        from ev_calculator import EVCalculator
        from ownership_tracker import OwnershipTracker
        
        ev_calc = EVCalculator(players_df, teams_df)
        own_tracker = OwnershipTracker(players_df, teams_df)
        
        # Get all distributions once
        ev_dists = ev_calc.get_all_distributions()
        ev_map = ev_dists.set_index('player_id').to_dict(orient='index')
        
        # Get all ownership
        own_data = own_tracker.get_all_ownership().set_index('player_id').to_dict(orient='index')
        
        def enrich_list(player_list):
            for p in player_list:
                pid = p.get('id')
                if not pid: continue
                
                analytics = {}
                
                # EV Data
                if pid in ev_map:
                    ev = ev_map[pid]
                    analytics.update({
                        'risk_score': ev.get('risk_score'),
                        'ceiling': ev.get('ceiling'),
                        'floor': ev.get('floor'),
                        'upside': ev.get('upside')
                    })
                
                # Ownership Data
                if pid in own_data:
                    own = own_data[pid]
                    analytics.update({
                        'ownership_pct': own.get('ownership_pct'),
                        'is_differential': own.get('is_differential'),
                        'is_template': own.get('is_template')
                    })
                    
                p['analytics'] = analytics

        # Enrich gameweek plans
        for plan in response_dict.get('gameweek_plans', []):
            enrich_list(plan.get('starting_xi', []))
            enrich_list(plan.get('bench', []))
            
        # Enrich top-level squad
        enrich_list(response_dict.get('squad', []))
        
        return response_dict
        
    except Exception as e:
        print(f"Analytics enrichment error: {e}")
        return response_dict


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
        current_gw = get_active_gameweek(events)
        
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
        
        response = optimizer.to_dict(solution)
        return _enrich_with_analytics(response, players_df, teams_df)
        
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
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Limit gameweeks to prevent timeout on free tier
    effective_gameweeks = min(request.gameweeks, 5)
    
    try:
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        fixtures_data = data["fixtures"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Get current gameweek
        events = static_data.get("events", [])
        current_gw = get_active_gameweek(events)
        
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
        
        # Parse chip_to_use from request
        chip_to_use = None
        if request.chip_to_use:
            chip_to_use = tuple(request.chip_to_use)  # [name, gw] -> (name, gw)
        
        # Define optimization functions for concurrent execution
        def run_with_hits():
            return optimizer.optimize_multi_period(
                budget=actual_budget,
                gameweeks=effective_gameweeks,
                current_squad_ids=current_squad_ids,
                excluded_players=request.excluded_players,
                banked_transfers=banked_transfers,
                chips_used=request.chips_used,
                chip_to_use=chip_to_use,
                robust=request.robust,
                uncertainty_budget=request.uncertainty_budget,
                strategy=request.strategy,
                max_hits=10  # Allow hits
            )
        
        def run_no_hits():
            return optimizer.optimize_multi_period(
                budget=actual_budget,
                gameweeks=effective_gameweeks,
                current_squad_ids=current_squad_ids,
                excluded_players=request.excluded_players,
                banked_transfers=banked_transfers,
                chips_used=request.chips_used,
                chip_to_use=chip_to_use,
                robust=request.robust,
                uncertainty_budget=request.uncertainty_budget,
                strategy=request.strategy,
                max_hits=0  # No hits allowed
            )
        
        # Run optimizations - for now run sequentially due to GIL
        # but wrap in executor to allow async timeout
        loop = asyncio.get_event_loop()
        
        try:
            # Set a 30 second timeout (solver is optimized with 15s limit)
            solution_with_hits = await asyncio.wait_for(
                loop.run_in_executor(None, run_with_hits),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            print("With-hits optimization timed out, returning simplified response")
            raise HTTPException(
                status_code=504,
                detail="Optimization timed out. Try reducing the gameweek horizon or using fewer constraints."
            )
        
        try:
            solution_no_hits = await asyncio.wait_for(
                loop.run_in_executor(None, run_no_hits),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            # If no-hits times out, return with-hits only
            print("No-hits optimization timed out, returning with-hits only")
            return {
                "current_gw": current_gw,
                "strategy": request.strategy,
                "comparison": {
                    "with_hits": {
                        "total_xp": round(solution_with_hits.total_expected_points, 1),
                        "total_hits": sum(tp.hits_taken for tp in solution_with_hits.transfer_summary),
                        "hit_cost": sum(tp.hit_cost for tp in solution_with_hits.transfer_summary),
                        "net_xp": round(solution_with_hits.total_expected_points - sum(tp.hit_cost for tp in solution_with_hits.transfer_summary), 1),
                        "recommended": True
                    },
                    "no_hits": {"error": "Timed out"},
                    "difference": 0
                },
                "with_hits": _enrich_with_analytics(optimizer.to_dict(solution_with_hits), players_df, teams_df),
                "no_hits": None,
                "partial_result": True
            }
        
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
            "with_hits": _enrich_with_analytics(optimizer.to_dict(solution_with_hits), players_df, teams_df),
            "no_hits": _enrich_with_analytics(optimizer.to_dict(solution_no_hits), players_df, teams_df)
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Optimization timed out. Try reducing the gameweek horizon."
        )
    except HTTPException:
        raise
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
        current_gw = get_active_gameweek(events)
        
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
        current_gw = get_active_gameweek(events)
        
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
        current_gw = get_active_gameweek(events)
        
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
        current_gw = get_active_gameweek(events)
        
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
        current_gw = get_active_gameweek(events)
        
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
        
        # Determine best chip to use based on urgency/value
        best_chip = None
        best_score = 0
        chip_priority = ['triple_captain', 'bench_boost', 'free_hit', 'wildcard']
        
        for chip_name in chip_priority:
            rec = recommendations.get(chip_name, {})
            if rec.get('recommended_gw') == current_gw:
                # Chip recommended for THIS gameweek - high priority
                score = rec.get('estimated_gain', 0) + 10
                if score > best_score:
                    best_score = score
                    best_chip = {'chip': chip_name, 'gw': current_gw, 'reason': rec.get('reason', '')}
        
        return {
            "current_gw": current_gw,
            "recommendations": recommendations,
            "best_chip": best_chip  # Auto-suggested chip for this GW, or None
        }
        
    except Exception as e:
        print(f"Chip advice error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/manager-chips/{manager_id}")
async def get_manager_chips(manager_id: int):
    """
    Get a manager's chip availability from FPL API.
    Returns used chips and available chips.
    """
    try:
        chips_info = fpl_service.get_manager_chips(manager_id)
        return chips_info
    except Exception as e:
        print(f"Error fetching manager chips: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}


# =============================================================================
# UNDERSTAT INTEGRATION ENDPOINTS
# =============================================================================

@app.get("/api/understat/players")
async def get_understat_players(season: str = "2025"):
    """
    Get all EPL player xG/xA statistics from Understat.
    
    Returns per-90 xG/xA rates, total season xG/xA, and other metrics.
    """
    try:
        from understat_service import UnderstatService
        
        service = UnderstatService()
        stats_df = service.get_all_stats_summary(season)
        
        if stats_df.empty:
            return {
                "season": season,
                "available": False,
                "message": "Understat data not available. Install understatapi: pip install understatapi",
                "players": []
            }
        
        # Sort by xGI (most productive players)
        stats_df = stats_df.sort_values('xGI', ascending=False)
        
        return {
            "season": season,
            "available": True,
            "total_players": len(stats_df),
            "players": stats_df.to_dict(orient='records')
        }
        
    except ImportError:
        return {
            "season": season,
            "available": False,
            "message": "Understat service not available. Install: pip install understatapi rapidfuzz",
            "players": []
        }
    except Exception as e:
        print(f"Understat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/understat/enriched")
async def get_enriched_players(season: str = "2025"):
    """
    Get FPL player data enriched with Understat xG/xA metrics.
    
    Matches FPL players to their Understat records and adds xG/xA per 90 data.
    This is the primary endpoint for enhanced predictions.
    """
    try:
        from understat_service import UnderstatService
        
        # Get FPL data
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Enrich with Understat
        service = UnderstatService()
        enriched_df = service.sync_with_fpl(players_df, teams_df, season)
        
        # Get match rate
        matched = enriched_df['understat_matched'].sum() if 'understat_matched' in enriched_df.columns else 0
        total = len(enriched_df)
        
        # Select relevant columns
        columns_to_include = [
            'id', 'web_name', 'team', 'element_type',
            'now_cost', 'total_points', 'form', 'ep_next',
            'understat_matched', 'understat_xG', 'understat_xA', 'understat_xGI',
            'understat_xG_per_90', 'understat_xA_per_90', 'understat_games', 'understat_minutes'
        ]
        
        available_cols = [c for c in columns_to_include if c in enriched_df.columns]
        result_df = enriched_df[available_cols].copy()
        
        # Sort by xGI (highest first)
        if 'understat_xGI' in result_df.columns:
            result_df = result_df.sort_values('understat_xGI', ascending=False)
        
        return {
            "season": season,
            "total_players": total,
            "matched_players": int(matched),
            "match_rate": round(matched / total * 100, 1) if total > 0 else 0,
            "players": result_df.to_dict(orient='records')
        }
        
    except ImportError:
        return {
            "season": season,
            "available": False,
            "message": "Understat service not available. Install: pip install understatapi rapidfuzz",
            "players": []
        }
    except Exception as e:
        print(f"Enriched data error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/understat/top-xg")
async def get_top_xg_players(limit: int = 20, position: Optional[int] = None, season: str = "2025"):
    """
    Get top players by xG or xG per 90.
    
    Args:
        limit: Number of players to return
        position: Filter by position (1=GK, 2=DEF, 3=MID, 4=FWD)
        season: Season year (e.g., "2025" for 2025/26)
    """
    try:
        from understat_service import UnderstatService
        
        # Get FPL data
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Enrich with Understat
        service = UnderstatService()
        enriched_df = service.sync_with_fpl(players_df, teams_df, season)
        
        # Filter by position if specified
        if position:
            enriched_df = enriched_df[enriched_df['element_type'] == position]
        
        # Filter to matched players with xG data
        if 'understat_matched' in enriched_df.columns:
            enriched_df = enriched_df[enriched_df['understat_matched'] == True]
        
        # Sort by xG per 90 (minimum 180 minutes for per-90 rate)
        if 'understat_xG_per_90' in enriched_df.columns:
            min_mins = enriched_df[enriched_df.get('understat_minutes', 0) >= 180]
            top_per_90 = min_mins.nlargest(limit, 'understat_xG_per_90')
        else:
            top_per_90 = pd.DataFrame()
        
        # Also get top by total xG
        if 'understat_xG' in enriched_df.columns:
            top_total = enriched_df.nlargest(limit, 'understat_xG')
        else:
            top_total = pd.DataFrame()
        
        # Build team name lookup
        team_names = {row['id']: row['name'] for _, row in teams_df.iterrows()}
        
        def format_player(row):
            return {
                'id': int(row['id']),
                'name': row['web_name'],
                'team': team_names.get(row['team'], 'Unknown'),
                'position': ['GK', 'DEF', 'MID', 'FWD'][int(row['element_type']) - 1],
                'price': row['now_cost'] / 10,
                'xG': round(row.get('understat_xG', 0), 2),
                'xA': round(row.get('understat_xA', 0), 2),
                'xGI': round(row.get('understat_xGI', 0), 2),
                'xG_per_90': round(row.get('understat_xG_per_90', 0), 3),
                'xA_per_90': round(row.get('understat_xA_per_90', 0), 3),
                'minutes': int(row.get('understat_minutes', 0))
            }
        
        return {
            "season": season,
            "position_filter": position,
            "top_by_xG_per_90": [format_player(row) for _, row in top_per_90.iterrows()],
            "top_by_total_xG": [format_player(row) for _, row in top_total.iterrows()]
        }
        
    except ImportError:
        return {
            "available": False,
            "message": "Understat service not available"
        }
    except Exception as e:
        print(f"Top xG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/understat/regression-candidates")
async def get_regression_candidates(season: str = "2025"):
    """
    Find players who are over/under-performing their xG.
    
    Useful for identifying:
    - Underperformers (goals < xG) who may improve
    - Overperformers (goals > xG) who may regress
    """
    try:
        from understat_service import UnderstatService
        
        service = UnderstatService()
        stats_df = service.get_all_stats_summary(season)
        
        if stats_df.empty:
            return {"available": False, "message": "Understat data not available"}
        
        # Filter players with enough data
        min_xG = 1.0
        candidates = stats_df[stats_df['xG'] >= min_xG].copy()
        
        # Calculate regression potential
        candidates['goal_diff'] = candidates['goals'] - candidates['xG']
        candidates['regression_factor'] = candidates['xG'] / candidates['goals'].replace(0, 0.1)
        
        # Underperformers (low goals vs xG)
        underperformers = candidates.nsmallest(10, 'goal_diff')
        
        # Overperformers (high goals vs xG)
        overperformers = candidates.nlargest(10, 'goal_diff')
        
        def format_candidate(row, is_underperformer):
            return {
                'name': row['player_name'],
                'team': row['team'],
                'goals': int(row['goals']),
                'xG': round(row['xG'], 2),
                'difference': round(row['goal_diff'], 2),
                'regression_factor': round(row['regression_factor'], 2),
                'assessment': 'BUY CANDIDATE - Underperforming xG' if is_underperformer else 'SELL CANDIDATE - Overperforming xG'
            }
        
        return {
            "season": season,
            "underperformers": [format_candidate(row, True) for _, row in underperformers.iterrows()],
            "overperformers": [format_candidate(row, False) for _, row in overperformers.iterrows()],
            "explanation": "Underperformers have scored fewer goals than their xG suggests - they may 'regress to the mean' and score more. Overperformers have outscored their xG and may slow down."
        }
        
    except ImportError:
        return {"available": False, "message": "Understat service not available"}
    except Exception as e:
        print(f"Regression candidates error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# OWNERSHIP ANALYTICS ENDPOINTS
# =============================================================================

@app.get("/api/ownership/all")
async def get_all_ownership():
    """
    Get ownership data for all players.
    
    Returns ownership percentage, estimated captain %, and effective ownership.
    """
    try:
        from ownership_tracker import OwnershipTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = OwnershipTracker(players_df, teams_df)
        df = tracker.get_all_ownership()
        
        # Sort by ownership
        df = df.sort_values('ownership_pct', ascending=False)
        
        return {
            "total_players": len(df),
            "players": df.to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Ownership error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ownership/template")
async def get_template_players(min_ownership: float = 20.0):
    """
    Get highly-owned 'template' players.
    
    These are must-have players that most managers own.
    """
    try:
        from ownership_tracker import OwnershipTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = OwnershipTracker(players_df, teams_df)
        template = tracker.get_template_players(min_ownership)
        
        return {
            "threshold": min_ownership,
            "count": len(template),
            "template_players": template.to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Template players error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ownership/differentials")
async def get_differentials(max_ownership: float = 5.0):
    """
    Get low-ownership differential players.
    
    These are potential punts that could help climb ranks.
    Filtered to players with decent form.
    """
    try:
        from ownership_tracker import OwnershipTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = OwnershipTracker(players_df, teams_df)
        differentials = tracker.get_differentials(max_ownership)
        
        return {
            "threshold": max_ownership,
            "count": len(differentials),
            "differentials": differentials.head(30).to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Differentials error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ownership/captains")
async def get_captain_rankings():
    """
    Get player rankings by estimated captaincy percentage.
    
    Shows who managers are likely captaining.
    """
    try:
        from ownership_tracker import OwnershipTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = OwnershipTracker(players_df, teams_df)
        captains = tracker.get_captaincy_rankings()
        
        return {
            "top_captain_picks": captains.to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Captain rankings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ownership/by-position")
async def get_ownership_by_position():
    """
    Get ownership breakdown by position.
    
    Shows most owned and top differentials at each position.
    """
    try:
        from ownership_tracker import OwnershipTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = OwnershipTracker(players_df, teams_df)
        by_position = tracker.get_ownership_by_position()
        
        return by_position
        
    except Exception as e:
        print(f"Ownership by position error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SquadCompareRequest(BaseModel):
    """Request to compare squad to template."""
    squad_ids: List[int]


@app.post("/api/ownership/compare-squad")
async def compare_squad_to_template(request: SquadCompareRequest):
    """
    Compare a user's squad to the template.
    
    Shows template coverage, missing must-haves, and differentials owned.
    """
    try:
        from ownership_tracker import OwnershipTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = OwnershipTracker(players_df, teams_df)
        comparison = tracker.compare_squad_to_template(request.squad_ids)
        
        return comparison
        
    except Exception as e:
        print(f"Squad comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ownership/summary")
async def get_ownership_summary():
    """
    Get a quick ownership summary.
    
    Returns top template players, differentials, and captain picks.
    """
    try:
        from ownership_tracker import get_ownership_summary
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        summary = get_ownership_summary(players_df, teams_df)
        
        return summary
        
    except Exception as e:
        print(f"Ownership summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# EV DISTRIBUTION CALCULATOR ENDPOINTS
# =============================================================================

@app.get("/api/ev/all")
async def get_all_ev_distributions():
    """
    Get EV distributions for all players.
    
    Returns expected points, floor, ceiling, and risk metrics.
    """
    try:
        from ev_calculator import EVCalculator
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        calc = EVCalculator(players_df, teams_df)
        df = calc.get_all_distributions()
        
        # Sort by expected points
        df = df.sort_values('expected_points', ascending=False)
        
        return {
            "total_players": len(df),
            "players": df.to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"EV distributions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ev/high-ceiling")
async def get_high_ceiling_players(min_ceiling: float = 8.0, position: Optional[int] = None):
    """
    Get players with high ceiling (explosive potential).
    
    Args:
        min_ceiling: Minimum ceiling threshold (default 8.0)
        position: Filter by position (1=GK, 2=DEF, 3=MID, 4=FWD)
    """
    try:
        from ev_calculator import EVCalculator
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        calc = EVCalculator(players_df, teams_df)
        high_ceiling = calc.get_high_ceiling_players(min_ceiling, position)
        
        return {
            "threshold": min_ceiling,
            "position_filter": position,
            "count": len(high_ceiling),
            "high_ceiling_players": high_ceiling.head(30).to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"High ceiling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ev/safe-players")
async def get_safe_players(max_risk: float = 0.5, min_expected: float = 3.0):
    """
    Get low-risk players with consistent returns.
    
    Args:
        max_risk: Maximum risk score (default 0.5)
        min_expected: Minimum expected points (default 3.0)
    """
    try:
        from ev_calculator import EVCalculator
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        calc = EVCalculator(players_df, teams_df)
        safe = calc.get_safe_players(max_risk, min_expected)
        
        return {
            "max_risk": max_risk,
            "min_expected": min_expected,
            "count": len(safe),
            "safe_players": safe.head(30).to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Safe players error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SquadEVRequest(BaseModel):
    """Request for squad EV calculation."""
    squad_ids: List[int]
    starting_xi: Optional[List[int]] = None


@app.post("/api/ev/squad")
async def calculate_squad_ev(request: SquadEVRequest):
    """
    Calculate aggregate EV distribution for a squad.
    
    Returns expected points, floor, ceiling, and player contributions.
    """
    try:
        from ev_calculator import EVCalculator
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        calc = EVCalculator(players_df, teams_df)
        result = calc.calculate_squad_ev(request.squad_ids, request.starting_xi)
        
        return result
        
    except Exception as e:
        print(f"Squad EV error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ComparePlayersRequest(BaseModel):
    """Request to compare players."""
    player_ids: List[int]


@app.post("/api/ev/compare")
async def compare_players_ev(request: ComparePlayersRequest):
    """
    Compare EV distributions of multiple players.
    
    Useful for transfer decisions - shows expected, ceiling, floor, value.
    """
    try:
        from ev_calculator import EVCalculator
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        calc = EVCalculator(players_df, teams_df)
        result = calc.compare_players(request.player_ids)
        
        return result
        
    except Exception as e:
        print(f"Compare players error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ev/simulate")
async def simulate_gameweek(request: SquadEVRequest, n_simulations: int = 1000):
    """
    Monte Carlo simulation of gameweek outcomes.
    
    Returns probability distribution of total points.
    """
    try:
        from ev_calculator import EVCalculator
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        calc = EVCalculator(players_df, teams_df)
        result = calc.simulate_gameweek(request.squad_ids, min(n_simulations, 5000))
        
        return result
        
    except Exception as e:
        print(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ev/summary")
async def get_ev_summary():
    """
    Get a quick EV summary.
    
    Returns high-ceiling, safe, and high-upside players.
    """
    try:
        from ev_calculator import get_ev_summary
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        summary = get_ev_summary(players_df, teams_df)
        
        return summary
        
    except Exception as e:
        print(f"EV summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# LIVE GAMEWEEK TRACKER ENDPOINTS
# =============================================================================

@app.get("/api/live/prices")
async def get_live_prices():
    """
    Get current player prices and price change predictions.
    
    Shows transfer activity and predicted price movements.
    """
    try:
        from live_tracker import LiveTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = LiveTracker(players_df, teams_df)
        df = tracker.get_live_prices()
        
        return {
            "total_players": len(df),
            "players": df.head(100).to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Live prices error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/live/price-risers")
async def get_price_risers(min_net_transfers: int = 20000):
    """Get players likely to rise in price."""
    try:
        from live_tracker import LiveTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = LiveTracker(players_df, teams_df)
        risers = tracker.get_price_risers(min_net_transfers)
        
        return {
            "threshold": min_net_transfers,
            "count": len(risers),
            "risers": risers.to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Price risers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/live/price-fallers")
async def get_price_fallers(min_net_out: int = 20000):
    """Get players likely to fall in price."""
    try:
        from live_tracker import LiveTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = LiveTracker(players_df, teams_df)
        fallers = tracker.get_price_fallers(min_net_out)
        
        return {
            "threshold": min_net_out,
            "count": len(fallers),
            "fallers": fallers.to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Price fallers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/live/transfers")
async def get_top_transfers():
    """Get most transferred in and out players this GW."""
    try:
        from live_tracker import LiveTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = LiveTracker(players_df, teams_df)
        
        return {
            "transfers_in": tracker.get_top_transfers_in(20).to_dict(orient='records'),
            "transfers_out": tracker.get_top_transfers_out(20).to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Transfers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/live/form")
async def get_form_analysis():
    """
    Analyze current form versus historical averages.
    
    Identifies players in hot/cold streaks.
    """
    try:
        from live_tracker import LiveTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = LiveTracker(players_df, teams_df)
        form = tracker.get_form_analysis()
        
        return {
            "total_players": len(form),
            "players": form.head(50).to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Form analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/live/hot-streaks")
async def get_hot_streaks(min_form: float = 6.0):
    """Get players on hot form streaks."""
    try:
        from live_tracker import LiveTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = LiveTracker(players_df, teams_df)
        hot = tracker.get_hot_streaks(min_form)
        
        return {
            "min_form": min_form,
            "count": len(hot),
            "hot_players": hot.to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Hot streaks error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/live/breakouts")
async def get_breakout_candidates():
    """
    Identify potential breakout candidates.
    
    Players with high recent form but low ownership.
    """
    try:
        from live_tracker import LiveTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = LiveTracker(players_df, teams_df)
        breakouts = tracker.get_breakout_candidates()
        
        return {
            "count": len(breakouts),
            "breakout_candidates": breakouts.to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Breakouts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/live/value-plays")
async def get_value_plays():
    """Get high-value plays (form/price ratio)."""
    try:
        from live_tracker import LiveTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = LiveTracker(players_df, teams_df)
        values = tracker.get_value_plays()
        
        return {
            "count": len(values),
            "value_plays": values.head(30).to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Value plays error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/live/summary")
async def get_gameweek_summary():
    """
    Get comprehensive summary of current gameweek.
    
    Includes top scorers, transfers, form, and price changes.
    """
    try:
        from live_tracker import get_live_summary
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Get current gameweek
        events = static_data.get("events", [])
        current_gw = get_active_gameweek(events)
        
        summary = get_live_summary(players_df, teams_df, current_gw)
        
        return summary
        
    except Exception as e:
        print(f"GW summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/live/captains")
async def get_captain_analysis():
    """Analyze captain options based on recent returns."""
    try:
        from live_tracker import LiveTracker
        
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        tracker = LiveTracker(players_df, teams_df)
        captains = tracker.get_captain_analysis()
        
        return {
            "top_captains": captains.head(20).to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Captain analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
