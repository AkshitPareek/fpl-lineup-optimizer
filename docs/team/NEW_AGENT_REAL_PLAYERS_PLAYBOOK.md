# New Agent Playbook: ML Path With Real Players (No Fallback)

Date: 2026-03-05
Branch: `dev/ml-phase2`

This playbook is for running the **ML prediction path** (`MLPredictor`) against real FPL players.
Do not use rule-based fallback endpoints (`/api/predictions` with `PointPredictor`) for this validation.

## Scope You Should Validate
- ML model artifact loading (`models/current` contract)
- Real-player feature extraction from live FPL bootstrap + fixtures + history cache
- Batch ML predictions with confidence
- Explicit fallback detection (`confidence == 30` from batch fallback branch)

## 1. Preconditions

### 1.1 Activate env
```bash
cd /home/akshit/fpl-lineup-optimizer
source venv/bin/activate
```

### 1.2 Required packages
```bash
cd backend
pip install -r requirements.txt
```

## 2. ML Artifact Contract (Required)
`MLPredictor` expects this structure:

```text
backend/models/current/
  metadata.json
  scaler.pkl
  xgb/model.pkl
  xgb/feature_columns.csv
  nn/model.keras
```

### Check quickly
```bash
cd /home/akshit/fpl-lineup-optimizer/backend
ls -la models/current models/current/xgb models/current/nn
```

If missing, you must provision artifacts before continuing.

## 3. Run ML Path Directly (Real Players)
Run this from `backend/`:

```bash
python - <<'PY'
import numpy as np
import pandas as pd

from fpl_service import FPLService
from historical_data_service import HistoricalDataService
from ml_predictor import MLPredictor

def get_active_gw(events):
    current_gw = next((e for e in events if e.get("is_current")), None)
    next_gw = next((e for e in events if e.get("is_next")), None)
    if current_gw and current_gw.get("finished") and next_gw:
        return next_gw["id"]
    if current_gw:
        return current_gw["id"]
    if next_gw:
        return next_gw["id"]
    return 1

fpl = FPLService()
data = fpl.get_latest_data()
static_data = data["static"]
fixtures = data["fixtures"]

players_df = pd.DataFrame(static_data["elements"])
teams_df = pd.DataFrame(static_data["teams"])

events = static_data.get("events", [])
target_gw = get_active_gw(events)

# Build history dataframe from cache (real-player context)
history_service = HistoricalDataService()
history_service.fetch_all_player_history(static_data["elements"])

history_records = []
for _, p in players_df.iterrows():
    pid = str(p["id"])
    for r in history_service._history_cache.get(pid, {}).get("history", []):
        rec = dict(r)
        rec["player_id"] = p["id"]
        history_records.append(rec)

history_df = pd.DataFrame(history_records)

ctx = {
    "players": players_df,
    "teams": teams_df,
    "fixtures": fixtures,
    "history": history_df,
}

predictor = MLPredictor(model_version="current", models_dir="models")

# Predict top-priced 200 players as smoke set
smoke_players = (
    players_df.sort_values("now_cost", ascending=False)
    .head(200)["id"]
    .tolist()
)

pred_df = predictor.predict_batch(smoke_players, target_gw, context_data=ctx)

fallback_count = int((pred_df["confidence"] == 30.0).sum())
print("target_gw:", target_gw)
print("predictions:", len(pred_df))
print("fallback_count:", fallback_count)
print("top_predictions:")
print(pred_df.sort_values("predicted_points", ascending=False).head(10).to_string(index=False))

# Hard fail if too much fallback behavior
ratio = fallback_count / max(len(pred_df), 1)
if ratio > 0.05:
    raise SystemExit(f"FAIL: fallback ratio too high ({ratio:.1%})")

print("PASS: ML path exercised with low fallback ratio")
PY
```

## 4. Manager-Specific Real Squad Check
Use a real manager ID:

```bash
python - <<'PY'
import pandas as pd
from fpl_service import FPLService
from historical_data_service import HistoricalDataService
from ml_predictor import MLPredictor

MANAGER_ID = 1234567  # replace

fpl = FPLService()
data = fpl.get_latest_data()
static_data = data["static"]
fixtures = data["fixtures"]
players_df = pd.DataFrame(static_data["elements"])
teams_df = pd.DataFrame(static_data["teams"])

manager = fpl.get_manager_team(MANAGER_ID)
player_ids = [p["element"] for p in manager.get("picks", [])]

events = static_data.get("events", [])
current = next((e for e in events if e.get("is_current")), None)
nextgw = next((e for e in events if e.get("is_next")), None)
if current and current.get("finished") and nextgw:
    target_gw = nextgw["id"]
elif current:
    target_gw = current["id"]
elif nextgw:
    target_gw = nextgw["id"]
else:
    target_gw = 1

h = HistoricalDataService()
h.fetch_all_player_history(static_data["elements"])
records = []
for _, p in players_df.iterrows():
    pid = str(p["id"])
    for r in h._history_cache.get(pid, {}).get("history", []):
        rr = dict(r)
        rr["player_id"] = p["id"]
        records.append(rr)
history_df = pd.DataFrame(records)

ctx = {"players": players_df, "teams": teams_df, "fixtures": fixtures, "history": history_df}
pred = MLPredictor(model_version="current", models_dir="models")
out = pred.predict_batch(player_ids, target_gw, context_data=ctx)

joined = out.merge(players_df[["id", "web_name", "now_cost"]], left_on="player_id", right_on="id", how="left")
print(joined.sort_values("predicted_points", ascending=False)[["web_name", "predicted_points", "confidence", "now_cost"]].to_string(index=False))
PY
```

## 5. What Not To Use For This Test
Do not use these to claim ML-path success:
- `GET /api/predictions` (rule-based `PointPredictor`)
- Any endpoint that only uses `ep_next`/form and not `MLPredictor`

## 6. Mandatory Evidence To Record
- Model info from `MLPredictor.get_model_info()`
- Target GW and player sample size
- Fallback ratio (`confidence == 30.0`)
- Top 10 predicted players with confidences
- Any missing-artifact or shape-mismatch errors

## 7. Gate Reminder
Non-optional gates must still pass before completion:
```bash
cd /home/akshit/fpl-lineup-optimizer
./backend/scripts/verify_task_completion.sh P2-T1
./backend/scripts/verify_task_completion.sh P2-T2
./backend/scripts/verify_task_completion.sh P2-T3
./backend/scripts/verify_task_completion.sh P2-T5
./backend/scripts/verify_task_completion.sh P2-T6
```
