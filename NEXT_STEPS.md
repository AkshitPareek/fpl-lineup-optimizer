# 🎯 What's Next? - Action Plan

> **Current Status:** EXP-031 Complete (73% Spearman), EXP-030 Champion Deployed ✅

---

## ✅ Recently Completed

### EXP-031: Historical Data Training
```
Data Collected:    106,042 samples (4 seasons)
Final Dataset:     52,974 samples (after dedup)
Test RMSE:         1.4629
Spearman:          0.7263 (73% correlation)
vs EXP-030:        +279% better Spearman!
```

**✅ Achievement:** EXP-031 ranks players 4x better than champion model (EXP-030)

### Parallel Agent System
```
4 agents, 4 seasons
Collection time:   2.5 minutes (3.2x speedup)
Storage used:      17 MB
```

---

## 🚀 Immediate Actions (Next 24 Hours)

### 1. **Deploy EXP-031 for Ranking** ⭐
Use EXP-031's 73% Spearman correlation for player ranking:

```python
# Example: Rank players by predicted points
import pickle
import numpy as np

with open('models/exp031_clean/model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']

# Rank players
players = load_player_data()  # Your player loading function
ranked = sorted(players, key=lambda p: model.predict(scaler.transform(p['features']))[0], reverse=True)
```

### 2. **Compare EXP-030 vs EXP-031**
Run both models and compare recommendations:

```bash
python run_team_prediction_v2.py --team-id 9777842 --compare-models
```

---

## 📊 This Week

### 3. **Hybrid Model Ensemble**
Combine EXP-030 (exact) + EXP-031 (ranking):

```python
# Weighted ensemble
exp030_weight = 0.4
exp031_weight = 0.6

final_score = (exp030_weight * exp030_pred + 
               exp031_weight * exp031_pred)
```

**Hypothesis:** Best of both worlds - accurate points + good ranking

### 4. **A/B Test Setup**
Test EXP-031 in production:
- Week 1: EXP-030 baseline
- Week 2: EXP-031 ranking
- Compare actual FPL points

---

## 🧪 Research - Find EXP-032 (Next Champion)

### 5. **Feature Expansion with Data**
Now that we have 52k samples, try more features:

```bash
# Add fixture difficulty
train_exp031_clean.py --add-features fdr,team_strength

# Add position-specific models
train_exp031_clean.py --position-specific
```

**Expected:** With 4,800 samples/feature ratio, should avoid overfitting

### 6. **Deep Learning Experiment**
Try LSTM for time-series prediction:

```python
# LSTM for player form trends
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, input_shape=(5, 10)),  # 5 GW history, 10 features
    Dense(32, activation='relu'),
    Dense(1)
])
```

### 7. **Collect 2024-25 Data**
Add current season to training set:

```bash
python agent_orchestrator.py --seasons 2024-25
python aggregate_historical_data.py
python train_exp031_clean.py
```

---

## 🎨 User Experience

### 8. **Web Dashboard Update**
Add EXP-031 to Streamlit UI:

```python
# dashboard.py
import streamlit as st

model_choice = st.radio(
    "Select Model",
    ["EXP-030 (Champion)", "EXP-031 (Historical)"]
)

if model_choice == "EXP-031 (Historical)":
    # Use EXP-031 for ranking
    predictions = exp031_predict(team_id)
    st.info("Using historical data model with 73% ranking accuracy")
```

### 9. **Model Comparison Report**
Auto-generate weekly comparison:
```
EXP-030 vs EXP-031 Comparison - GW 31
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXP-030 Predictions:    [list]
EXP-031 Rankings:       [list]
Actual Points:          [list]
EXP-030 Error:          RMSE X.XX
EXP-031 Error:          RMSE X.XX
Winner:                 EXP-0XX
```

---

## 🔧 Technical Debt

### 10. **Model Versioning**
Set up MLflow for both models:
```python
import mlflow

# Log EXP-031
mlflow.log_param("model", "EXP-031")
mlflow.log_metric("rmse", 1.4629)
mlflow.log_metric("spearman", 0.7263)
mlflow.log_param("samples", 52974)
mlflow.sklearn.log_model(model, "exp031")
```

### 11. **Data Pipeline Automation**
Auto-collect new season data:
```bash
# cron job - weekly data collection
0 0 * * 1 cd /path && python agent_orchestrator.py --seasons current
```

---

## 🎯 Decision Matrix

| Priority | Task | Effort | Impact | Recommendation |
|----------|------|--------|--------|----------------|
| 🔴 High | Deploy EXP-031 ranking | 1 hour | Critical | **DO NOW** |
| 🔴 High | Hybrid ensemble test | 2 hours | High | This week |
| 🟡 Med | Collect 2024-25 data | 30 min | High | This week |
| 🟡 Med | Web dashboard update | 3 hours | Medium | Next week |
| 🟢 Low | Find EXP-032 | Ongoing | High | Background |

---

## 💡 My Recommendation

**Today:**
1. ✅ Deploy EXP-031 for player ranking (use 73% Spearman)
2. ✅ Run A/B comparison between EXP-030 and EXP-031

**This Week:**
1. 🎯 Build hybrid ensemble (best of both models)
2. 🎯 Collect 2024-25 season data
3. 🎯 Update dashboard with model selector

**Next Week:**
1. 🔬 Experiment with feature expansion (now we have data!)
2. 🔬 Try deep learning models
3. 📊 Analyze model performance in live play

---

## 🎉 Success Metrics

**Current:**
- ✅ EXP-030: RMSE 0.8284, Spearman 0.19
- ✅ EXP-031: RMSE 1.46, Spearman 0.73 ⭐
- ✅ 52k historical samples collected
- ✅ Parallel agent system deployed

**Target (4 weeks):**
- 🎯 Hybrid model beats both individually
- 🎯 +5-10 FPL points from better ranking
- 🎯 100k+ samples with 2024-25 data
- 🎯 EXP-032 candidate identified

---

## 🤔 Questions to Answer

1. **Does EXP-031's ranking translate to better FPL performance?**
   - A/B test for 4 GWs
   - Compare total points with EXP-030 vs EXP-031 picks

2. **Can hybrid model beat both?**
   - Combine EXP-030 exact + EXP-031 ranking
   - Weighted ensemble approach

3. **What features matter most with 50k+ samples?**
   - Run feature importance analysis
   - Try all 47 features now

---

## 📚 Documentation Status

| Document | Status | Location |
|----------|--------|----------|
| EXP-031 Results | ✅ Updated | `/EXP031_RESULTS.md` |
| Feature Expansion | ✅ Updated | `/FEATURE_EXPANSION_SUMMARY.md` |
| Experiment Report | ✅ Created | `/research/03-experiments/2026-03-13-exp-031/` |
| Results Summary | ✅ Created | `/research/04-results/EXP031_HISTORICAL_TRAINING.md` |

---

**What's your priority?**
- A) Deploy EXP-031 for ranking (recommended)
- B) Build hybrid model (most ambitious)
- C) Collect more data (foundational)
- D) Something else?
