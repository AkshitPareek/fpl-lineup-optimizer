# REAL Artifacts Manager Test Runbook (2026-03-05)

## Purpose
Validate that `backend/models/current` points to `real_2026_03_05`, that ML inference is used (not rule fallback), and that manager `9777842` multi-period planning works with `FT=5`, `bank=1.9`, `horizon=5`.

API behavior note:
- `force_next_gameweek` defaults to `true` on `/api/optimize/multi-period`.
- With that default, if GW29 is already finished, the optimizer starts at GW30 (next playable round) instead of re-optimizing a completed GW.

## Prerequisites / Artifact Contract
```bash
cd /home/akshit/fpl-lineup-optimizer
source venv/bin/activate
cd backend
```

Required contract for `MLPredictor(model_version="current", models_dir="models")`:
```text
backend/models/current/
  metadata.json
  scaler.pkl
  xgb/model.pkl
  xgb/feature_columns.csv
  nn/model.keras
```

## Verify `current -> real_2026_03_05`
```bash
cd /home/akshit/fpl-lineup-optimizer/backend
readlink models/current
ls -la models/current models/current/xgb models/current/nn
for f in \
  models/current/metadata.json \
  models/current/scaler.pkl \
  models/current/xgb/model.pkl \
  models/current/xgb/feature_columns.csv \
  models/current/nn/model.keras; do
  test -f "$f" && echo "OK $f" || { echo "MISSING $f"; exit 1; }
done
/home/akshit/fpl-lineup-optimizer/venv/bin/python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path('models/current/metadata.json').read_text())
print('metadata.version:', m.get('version'))
PY
```

Expected:
```text
readlink models/current -> real_2026_03_05
metadata.version: real_2026_03_05
```

## Fallback Smoke Command + Threshold
```bash
cd /home/akshit/fpl-lineup-optimizer/backend
/home/akshit/fpl-lineup-optimizer/venv/bin/python - <<'PY'
import pandas as pd
from fpl_service import FPLService
from historical_data_service import HistoricalDataService
from ml_predictor import MLPredictor

def active_gw(events):
    cur = next((e for e in events if e.get('is_current')), None)
    nxt = next((e for e in events if e.get('is_next')), None)
    if cur and cur.get('finished') and nxt: return nxt['id']
    if cur: return cur['id']
    if nxt: return nxt['id']
    return 1

fpl = FPLService()
data = fpl.get_latest_data()
static = data['static']
fixtures = data['fixtures']
players = pd.DataFrame(static['elements'])
teams = pd.DataFrame(static['teams'])
gw = active_gw(static.get('events', []))

h = HistoricalDataService()
h.fetch_all_player_history(static['elements'])
rows = []
for _, p in players.iterrows():
    for r in h._history_cache.get(str(p['id']), {}).get('history', []):
        rr = dict(r); rr['player_id'] = p['id']; rows.append(rr)
history = pd.DataFrame(rows)
ctx = {'players': players, 'teams': teams, 'fixtures': fixtures, 'history': history}

pred = MLPredictor(model_version='current', models_dir='models')
smoke_ids = players.sort_values('now_cost', ascending=False).head(200)['id'].tolist()
out = pred.predict_batch(smoke_ids, gw, context_data=ctx)

fallback_count = int((out['confidence'] == 30.0).sum())
ratio = fallback_count / max(len(out), 1)
print('target_gw:', gw)
print('predictions:', len(out))
print('fallback_count:', fallback_count)
print('fallback_ratio:', ratio)
if ratio > 0.05:
    raise SystemExit(f'FAIL fallback_ratio={ratio:.4f} > 0.05')
print('PASS fallback_ratio <= 0.05')
PY
```
Threshold: pass if `fallback_ratio <= 0.05`.

## ML-Path Manager Test (9777842, FT=5, bank=1.9, horizon=5)
```bash
cd /home/akshit/fpl-lineup-optimizer/backend
/home/akshit/fpl-lineup-optimizer/venv/bin/python - <<'PY'
import pandas as pd
from fpl_service import FPLService
from historical_data_service import HistoricalDataService
from ml_predictor import MLPredictor
from advanced_optimizer import MultiPeriodFPLOptimizer

MANAGER_ID = 9777842
FREE_TRANSFERS = 5
BANK = 1.9
HORIZON = 5

def active_gw(events):
    cur = next((e for e in events if e.get('is_current')), None)
    nxt = next((e for e in events if e.get('is_next')), None)
    if cur and cur.get('finished') and nxt: return nxt['id']
    if cur: return cur['id']
    if nxt: return nxt['id']
    return 1

fpl = FPLService()
data = fpl.get_latest_data()
static = data['static']
fixtures = data['fixtures']
players = pd.DataFrame(static['elements'])
teams = pd.DataFrame(static['teams'])
manager = fpl.get_manager_team(MANAGER_ID)
current_squad_ids = [p['element'] for p in manager.get('picks', [])]

gw = active_gw(static.get('events', []))
horizon_gws = list(range(gw, gw + HORIZON))

h = HistoricalDataService()
h.fetch_all_player_history(static['elements'])
rows = []
for _, p in players.iterrows():
    for r in h._history_cache.get(str(p['id']), {}).get('history', []):
        rr = dict(r); rr['player_id'] = p['id']; rows.append(rr)
history = pd.DataFrame(rows)
ctx = {'players': players, 'teams': teams, 'fixtures': fixtures, 'history': history}

ml = MLPredictor(model_version='current', models_dir='models')
all_ids = players['id'].tolist()
preds = players[['id', 'web_name', 'element_type', 'team', 'now_cost']].copy()
for tg in horizon_gws:
    b = ml.predict_batch(all_ids, tg, context_data=ctx)
    b = b.rename(columns={'player_id': 'id', 'predicted_points': f'xp_gw{tg}'})
    preds = preds.merge(b[['id', f'xp_gw{tg}']], on='id', how='left')

xp_cols = [f'xp_gw{x}' for x in horizon_gws]
preds[xp_cols] = preds[xp_cols].fillna(0.0)
preds['total_xp'] = preds[xp_cols].sum(axis=1)

squad_value = float(players[players['id'].isin(current_squad_ids)]['now_cost'].sum()) / 10.0
budget = squad_value + BANK

opt = MultiPeriodFPLOptimizer(
    players_df=players,
    teams_df=teams,
    fixtures=fixtures,
    current_gameweek=gw,
    predictions_df=preds[['id', 'web_name', 'element_type', 'team', 'now_cost', 'total_xp'] + xp_cols],
)
sol = opt.optimize_multi_period(
    budget=budget,
    gameweeks=HORIZON,
    current_squad_ids=current_squad_ids,
    banked_transfers=FREE_TRANSFERS,
)

print('status:', sol.status)
print('model_version:', ml.get_model_info().get('version'))
print('manager_id:', MANAGER_ID)
print('target_gw:', gw)
print('bank:', BANK)
print('free_transfers:', FREE_TRANSFERS)
print('horizon:', HORIZON)
print('transfer_summary_len:', len(sol.transfer_summary))
PY
```

## API Request Example (`/api/optimize/multi-period`)
Use this to exercise the same scenario via HTTP (with default-next-GW behavior made explicit):

```bash
curl -sS -X POST "http://localhost:8000/api/optimize/multi-period" \
  -H "Content-Type: application/json" \
  -d '{
    "manager_id": 9777842,
    "banked_transfers": 5,
    "budget": 1.9,
    "use_ml_predictions": true,
    "ml_blend_weight": 0.6,
    "force_next_gameweek": true
  }'

## Chip suggestion endpoint (`/api/chip-suggestions`)

```bash
curl -sS -X POST "http://localhost:8000/api/chip-suggestions" \
  -H "Content-Type: application/json" \
  -d '{
    "manager_id": 9777842,
    "chips_used": ["wildcard"],
    "force_next_gameweek": true
  }'
```

Key output:
- `chip_advice.summary` lists next DGW/BGW and remaining chips.
- `chip_advice.chip_recommendations` ranks chips by `estimated_gain`.
- `chip_advice.captain_recommendations` repeats the captain analysis per GW.
```

Expected targeting behavior:
- If GW29 is finished at request time, `force_next_gameweek=true` should target GW30.
- If the current GW is still active, it targets that active GW.

## Expected Output Notes
- `fallback_ratio` should be `0.0` on the successful smoke run.
- `model_version` should be `real_2026_03_05`.
- With `ml_blend_weight=0.6`, expected points are blended (60% ML, 40% PointPredictor). Pure PointPredictor uses only heuristic projections and is less responsive to form signals captured by the model.

## Troubleshooting
1. XGBoost `early_stopping` incompatibility
- Symptom: `TypeError` around `early_stopping_rounds` in `XGBRegressor.fit(...)`.
- Fix: use an XGBoost version compatible with the training call, or switch to callback-style early stopping.

2. `metadata.json` serialization fails on NumPy types
- Symptom: `Object of type float32/int64 is not JSON serializable`.
- Fix: cast NumPy scalars/arrays to native Python before `json.dump`.

3. `--test-gw` cutoff issue
- Symptom: empty/tiny validation/test split when cutoff exceeds available rounds.
- Fix: set `--test-gw` at or below observed max `target_gw` in training data.

## Final Checklist
```text
[ ] venv active, cwd is backend/
[ ] models/current points to real_2026_03_05
[ ] artifact files exist (metadata/scaler/xgb/nn)
[ ] metadata.version == real_2026_03_05
[ ] smoke command ran and fallback_ratio <= 0.05
[ ] manager command ran for 9777842, FT=5, bank=1.9, horizon=5
[ ] manager command output shows model_version real_2026_03_05
[ ] smoke fallback_ratio note recorded as 0.0 when successful
```
