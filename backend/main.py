from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from fpl_service import FPLService
from optimizer import FPLOptimizer

app = FastAPI(title="FPL Optimizer API")

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

class OptimizationRequest(BaseModel):
    budget: float
    gameweeks: int = 1
    strategy: str = "standard"  # standard, safe, differential, my_squad, transfers
    excluded_players: List[int] = []
    manager_id: Optional[int] = None
    free_transfers: int = 1

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
    """Generates the optimal lineup based on constraints."""
    try:
        data = fpl_service.get_latest_data()
        
        current_team = None
        if request.manager_id:
             try:
                current_team = fpl_service.get_manager_team(request.manager_id)
                if request.strategy == "standard": 
                     # If user selected "standard" (Wildcard) but provided an ID, maybe they just want the budget?
                     # For now, let's keep it simple: ID + "Use My Squad" = Best XI. 
                     # ID + "Standard" = Wildcard (ignore squad, just pure optimization)
                     pass
             except Exception:
                 pass # Ignore if fetch fails for now

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

@app.get("/health")
def health_check():
    return {"status": "ok"}
