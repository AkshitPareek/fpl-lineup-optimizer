# EXP-032 Research Goals

**Date:** 2026-03-13  
**Status:** 🔄 ACTIVE  
**Parallel Agents:** 4 (Production, Data, Research Alpha, Research Beta)

---

## Mission Statement

**Find a model that beats EXP-031** (Spearman 0.7263, RMSE 1.4629) through systematic parallel research while maintaining production reliability.

---

## Current State (EXP-031)

| Metric | Value | Notes |
|--------|-------|-------|
| Spearman | 0.7263 | 73% ranking correlation |
| RMSE | 1.4629 | Prediction accuracy |
| Data | 52,974 samples | 4 seasons (2020-2024) |
| Features | 11 | Clean, no leakage |
| Top Feature | form_3gw | 78% importance |

**EXP-031 Achievement:** 4x better Spearman than EXP-030 (0.19 → 0.73)

---

## Research Targets

### Primary Goal: EXP-032

| Metric | Target | Current | Required Improvement |
|--------|--------|---------|---------------------|
| Spearman | **≥ 0.75** | 0.7263 | +3.3% |
| RMSE | **≤ 1.40** | 1.4629 | -4.3% |

**Success Criteria:**
- Either Spearman ≥ 0.75 OR RMSE ≤ 1.40
- Statistical significance (p < 0.05)
- No data leakage
- Inference < 100ms

### Stretch Goal: EXP-033

| Metric | Target |
|--------|--------|
| Spearman | ≥ 0.78 |
| RMSE | ≤ 1.35 |

### Ambitious Goal: EXP-034+

| Metric | Target |
|--------|--------|
| Spearman | ≥ 0.80 |
| RMSE | ≤ 1.30 |

---

## Hypothesis Bank

### Category 1: Feature Engineering (Agent 3 Focus)

#### H1: Fixture Difficulty Ratings (FDR)
**Hypothesis:** Adding official FPL fixture difficulty ratings will improve predictions by 2-3%.

**Implementation:**
- Fetch FDR from FPL API
- Add `fdr_next` feature (1-5 scale)
- Add `opponent_attack_strength` and `opponent_defense_strength`

**Expected Impact:** Spearman +0.015 to +0.025

---

#### H2: Team Strength Features
**Hypothesis:** Team-level attack/defense ratings improve predictions beyond individual player form.

**Implementation:**
- Add `team_attack_6gw` (team goals last 6 games)
- Add `team_defense_6gw` (team conceded last 6 games)
- Add `team_form_trend` (improving/declining)

**Expected Impact:** Spearman +0.01 to +0.02

---

#### H3: Ownership Trends
**Hypothesis:** Changes in player ownership indicate insider knowledge or form changes.

**Implementation:**
- Add `ownership_change_1gw` (week-over-week change)
- Add `ownership_percentile` (vs other players)
- Add `transfers_velocity` (rate of change)

**Expected Impact:** Spearman +0.005 to +0.015

---

#### H4: Feature Interactions
**Hypothesis:** Interaction effects (form × fixture difficulty) capture context better.

**Implementation:**
- `form_3gw × fdr_next` (form matters more in easy fixtures)
- `value × form_3gw` (cheap form players are gems)
- `was_home × team_strength` (home advantage varies by team)

**Expected Impact:** Spearman +0.01 to +0.02

---

### Category 2: Deep Learning (Agent 4 Focus)

#### H5: Multi-Layer Perceptron (MLP)
**Hypothesis:** Neural networks can capture non-linear patterns in player performance.

**Architectures to Try:**
```python
# Small network
MLP(hidden_layer_sizes=(32,), alpha=0.001)

# Medium network
MLP(hidden_layer_sizes=(64, 32), alpha=0.001)

# Large network
MLP(hidden_layer_sizes=(128, 64, 32), alpha=0.0001)
```

**Expected Impact:** Spearman +0.00 to +0.03 (uncertain)

---

#### H6: LSTM for Time Series
**Hypothesis:** Sequential models capture temporal patterns in player form.

**Implementation:**
```python
# 5-game sequences
LSTM(input_shape=(5, n_features), units=64)
```

**Challenge:** Requires sequence data preparation

**Expected Impact:** Spearman +0.01 to +0.04 (if implemented well)

---

### Category 3: Ensemble Strategies

#### H7: Position-Specific Ensembles
**Hypothesis:** Different models work better for different positions.

**Implementation:**
- Train separate models for GK, DEF, MID, FWD
- Combine with position-specific weights
- Features may vary by position

**Expected Impact:** Spearman +0.01 to +0.02

---

#### H8: Stacking Ensemble
**Hypothesis:** A meta-learner can optimally combine base model predictions.

**Implementation:**
```python
# Base models
base_models = [Ridge(), GradientBoosting(), RandomForest()]

# Meta-learner
meta_model = Ridge()

# Stacking
StackingRegressor(estimators=base_models, final_estimator=meta_model)
```

**Expected Impact:** Spearman +0.005 to +0.015

---

### Category 4: Data Strategies

#### H9: Add 2024-25 Season (Agent 2)
**Hypothesis:** More recent data improves predictions for current form.

**Target:** 15,000+ new samples

**Expected Impact:** Spearman +0.01 to +0.03

---

#### H10: Time-Weighted Training
**Hypothesis:** Recent seasons should be weighted more heavily.

**Implementation:**
```python
# Weight by recency
weights = [0.5, 0.7, 0.9, 1.0]  # 2020-21 to 2023-24
```

**Expected Impact:** Spearman +0.005 to +0.015

---

## Parallel Agent Strategy

### Agent 1: Production Testing
**Task:** Deploy EXP-031 and confirm 73% Spearman in production

**Deliverables:**
- Production comparison report
- Team selection comparison (EXP-031 vs EXP-030)
- Recommendation: Deploy or wait for EXP-032

**Timeline:** 4 gameweeks (~4 weeks)

---

### Agent 2: Data Collection
**Task:** Fetch 2024-25 season data

**Target:**
- 15,000+ new samples
- Update master dataset
- Retrain EXP-031 with expanded data

**Timeline:** ~2 hours

---

### Agent 3: Ralph Loop (Feature Expansion)
**Task:** Try feature engineering hypotheses (H1-H4)

**Strategy:**
- 20 iterations max
- Focus on FDR, team strength, interactions
- Target: Spearman ≥ 0.75

**Timeline:** ~6 hours (20 × 18 min avg)

---

### Agent 4: Ralph Loop (Deep Learning)
**Task:** Try neural network architectures (H5-H6)

**Strategy:**
- 20 iterations max
- Focus on MLP variations
- Target: Spearman ≥ 0.75

**Timeline:** ~8 hours (20 × 24 min avg)

---

## Evaluation Protocol

### Weekly Review
Every 7 days, review all agent outputs:

1. **Production Agent (Agent 1)**
   - Did EXP-031 maintain 73% Spearman?
   - Are picks better than EXP-030?
   - Decision: Deploy / Wait / Hybrid

2. **Data Agent (Agent 2)**
   - How many new samples collected?
   - Data quality checks passed?
   - Retrain with new data?

3. **Research Agents (3 & 4)**
   - Any new champions discovered?
   - Which hypotheses showed promise?
   - Adjust search strategy?

### Success Criteria for EXP-032

```python
def is_new_champion(candidate_metrics):
    """Determine if candidate beats EXP-031."""
    
    # Primary: Spearman improvement > 3%
    if candidate_metrics['spearman'] > 0.75:
        return True
    
    # Secondary: RMSE improvement > 4%
    if candidate_metrics['rmse'] < 1.40:
        return True
    
    # Combined: Both metrics improve
    spearman_gain = (candidate_metrics['spearman'] - 0.7263) / 0.7263
    rmse_gain = (1.4629 - candidate_metrics['rmse']) / 1.4629
    
    if spearman_gain > 0.01 and rmse_gain > 0.02:
        return True
    
    return False
```

---

## Documentation Requirements

Every experiment MUST document:

1. **Hypothesis** - What was tested and why
2. **Implementation** - How it was implemented
3. **Results** - Metrics compared to baseline
4. **Analysis** - Why it succeeded or failed
5. **Next Steps** - What to try next

### Publication-Ready Checklist

- [ ] All experiments documented in `research/03-experiments/`
- [ ] Results summarized in `research/04-results/`
- [ ] Code committed to git with clear messages
- [ ] Models saved with metadata
- [ ] Reproducible scripts provided
- [ ] Comparison with baselines (EXP-030, EXP-031)

---

## Risk Management

### Risk: No Improvement Found
**Mitigation:** 
- Broad search space (4 categories)
- Parallel exploration
- Learn from failures

### Risk: Overfitting
**Mitigation:**
- Clean features only (no leakage)
- Time-series split
- Cross-validation

### Risk: Computational Cost
**Mitigation:**
- 30-minute timeout per experiment
- Parallel agents
- Early stopping

---

## Timeline

| Week | Activities |
|------|-----------|
| 1 | Launch all 4 agents, Agent 2 completes data collection |
| 2 | Agent 1 production results, Agents 3-4 continue search |
| 3 | Mid-point review, adjust strategies if needed |
| 4 | Final results, document findings, publish |

---

## Expected Outcomes

### Optimistic Scenario
- EXP-032 discovered with Spearman ≥ 0.76
- 2024-25 data adds 15k+ samples
- EXP-031 validated in production
- **Publication:** "From 19% to 76%: Scaling FPL Predictions with Parallel Research"

### Realistic Scenario
- No EXP-032 found, but valuable insights gained
- 2024-25 data collected
- EXP-031 confirmed as production champion
- **Publication:** "The Limits of FPL Prediction: A Systematic Study"

### Pessimistic Scenario
- Agents fail due to technical issues
- Retry with fixed infrastructure
- Document challenges and solutions

---

## Resources

### Compute
- 4 CPU cores (1 per agent)
- 8GB RAM sufficient
- ~10 hours total compute time

### Storage
- Models: ~50MB per experiment
- Data: ~20MB for 2024-25 season
- Logs: ~5MB

### External APIs
- FPL API (free, rate-limited)
- GitHub (for historical data)

---

## Contact & Collaboration

- **Lead Researcher:** Kimi Code Agent
- **GitHub:** /AkshitPareek/fpl-lineup-optimizer
- **Branch:** ml-backend

---

*Goal: Find EXP-032. Never stop improving.*
