# FPL Lineup Optimizer - Task Tracker

> **Active Research:** EXP-032 Parallel Agent Search  
> **Last Updated:** 2026-03-13  
> **Current Champion:** EXP-031 (Spearman 0.7263)

---

## 📋 Active Tasks

### 🔴 CRITICAL - Do First

#### TASK-001: Collect 2024-25 Season Data via FPL API
**Status:** 🔄 IN PROGRESS  
**Priority:** CRITICAL  
**Assigned:** Agent 2 (Enhanced)  
**Due:** 2026-03-14

**Description:**
Current season data (2024-25) is not on GitHub yet (vaastav/FPL) because season is in progress. Need to collect via FPL API directly.

**Acceptance Criteria:**
- [ ] Collect all gameweeks up to current
- [ ] Collect player history data
- [ ] Collect fixture data with FDR ratings
- [ ] Aggregate into trainable format
- [ ] Add to master dataset (target: 15k+ new samples)

**Implementation Notes:**
```python
# Use existing FPLService as base
# Extend to collect historical gameweek data
# URL pattern: /api/element-summary/{player_id}/
```

**Related Files:**
- `backend/fpl_service.py` - Base API client
- `agents/collect_2024_25_data.py` - Current (GitHub-based, needs FPL API)

---

#### TASK-002: Build Real-Time Dashboard
**Status:** 📋 NOT STARTED  
**Priority:** CRITICAL  
**Assigned:** TBD  
**Due:** 2026-03-15

**Description:**
Create a visual dashboard that auto-updates to show:
- Current model performance
- Team recommendations
- Model comparisons (EXP-030 vs EXP-031)
- Live FPL data

**Requirements:**
1. **Auto-refresh:** Updates every hour or on demand
2. **Model Comparison Panel:** Side-by-side EXP-030 vs EXP-031
3. **Team Builder:** Interactive team selection with predictions
4. **Performance Metrics:** RMSE, Spearman over time
5. **Gameweek Planning:** Upcoming fixture difficulty

**Tech Stack:**
- Streamlit (primary)
- Plotly (charts)
- Pandas (data)
- APScheduler (auto-refresh)

**Pages:**
1. **Overview:** Key metrics, champion model info
2. **Model Lab:** Compare models, run predictions
3. **Team Builder:** Build optimal team, transfers
4. **Research:** Experiment results, agent status
5. **Settings:** API keys, preferences

**Acceptance Criteria:**
- [ ] Streamlit app runs without errors
- [ ] All 5 pages functional
- [ ] Auto-refresh works
- [ ] Mobile-responsive
- [ ] Deployed to Streamlit Cloud or local

---

### 🟡 HIGH PRIORITY

#### TASK-003: Implement Real FDR Features
**Status:** 📋 NOT STARTED  
**Priority:** HIGH  
**Assigned:** Ralph Loop Agent 3  
**Due:** 2026-03-16

**Description:**
The Ralph Loop showed that FDR (Fixture Difficulty Rating) features don't exist in current dataset. Need to:
1. Extract FDR from FPL API
2. Add to feature engineering pipeline
3. Retrain EXP-031 with FDR
4. Re-run Ralph Loop

**Acceptance Criteria:**
- [ ] FDR extraction from FPL API
- [ ] Feature engineering module updated
- [ ] Model retrained with FDR
- [ ] Spearman improvement measured

---

#### TASK-004: Document Ralph Loop Results
**Status:** 📋 NOT STARTED  
**Priority:** HIGH  
**Assigned:** Research Lead  
**Due:** 2026-03-14

**Description:**
40 experiments completed but not properly documented. Need comprehensive analysis of:
- What was tested
- What worked/didn't work
- Why EXP-031 is hard to beat
- Next hypothesis priorities

**Acceptance Criteria:**
- [ ] Analysis of all 40 experiments
- [ ] Publication-ready summary
- [ ] Updated research goals based on findings

---

### 🟢 MEDIUM PRIORITY

#### TASK-005: Position-Specific Models
**Status:** 📋 NOT STARTED  
**Priority:** MEDIUM  
**Assigned:** Ralph Loop (Future)  
**Due:** 2026-03-20

**Description:**
Train separate models for GK/DEF/MID/FWD. Different features may matter for different positions.

---

#### TASK-006: LSTM Time-Series Model
**Status:** 📋 NOT STARTED  
**Priority:** MEDIUM  
**Assigned:** Ralph Loop Agent 4 (Enhanced)  
**Due:** 2026-03-22

**Description:**
Implement proper LSTM with sequence data (5-game history per player).

**Acceptance Criteria:**
- [ ] Sequence data preparation
- [ ] LSTM architecture
- [ ] Training pipeline
- [ ] Evaluation vs EXP-031

---

## ✅ Completed Tasks

### Recently Completed

#### ✅ TASK-COMPLETED: EXP-031 Historical Training
**Completed:** 2026-03-13  
**Result:** 52,974 samples, Spearman 0.7263

#### ✅ TASK-COMPLETED: Parallel Agent System
**Completed:** 2026-03-13  
**Result:** 4 agents working, 40 experiments completed

#### ✅ TASK-COMPLETED: Ralph Loop Skill
**Completed:** 2026-03-13  
**Result:** Continuous improvement framework ready

---

## 📊 Task Statistics

| Priority | Count | Completed | In Progress | Not Started |
|----------|-------|-----------|-------------|-------------|
| 🔴 Critical | 2 | 0 | 1 | 1 |
| 🟡 High | 2 | 0 | 0 | 2 |
| 🟢 Medium | 2 | 0 | 0 | 2 |
| **Total** | **6** | **0** | **1** | **5** |

---

## 🎯 Research Milestones

### Milestone 1: Data Foundation ⏰ 2026-03-14
- [ ] 2024-25 data collected (TASK-001)
- [ ] Dataset > 60k samples

### Milestone 2: Dashboard Launch ⏰ 2026-03-15
- [ ] Dashboard deployed (TASK-002)
- [ ] Model comparison live

### Milestone 3: Feature Expansion ⏰ 2026-03-18
- [ ] FDR features implemented (TASK-003)
- [ ] EXP-032 candidate identified

### Milestone 4: Publication ⏰ 2026-03-25
- [ ] All experiments documented
- [ ] Paper/blog post ready

---

## 🐛 Known Issues

| Issue | Severity | Status | Notes |
|-------|----------|--------|-------|
| 2024-25 data not on GitHub | Medium | Workaround | Use FPL API instead |
| Ralph Loop needs real features | Medium | In Progress | FDR extraction pending |
| Dashboard missing | High | Not Started | Critical for monitoring |

---

## 💡 Ideas Backlog

### Future Experiments
- [ ] Weather data integration
- [ ] Betting odds as features
- [ ] Social media sentiment
- [ ] Player injury prediction
- [ ] Team chemistry networks

### Dashboard Features
- [ ] Push notifications for transfers
- [ ] Captain pick optimizer
- [ ] Chip usage advisor
- [ ] Head-to-head comparison
- [ ] Expected points timeline

### Infrastructure
- [ ] MLflow integration
- [ ] Automated retraining
- [ ] A/B testing framework
- [ ] Model versioning

---

## 📝 Task Template

When creating new tasks, use this format:

```markdown
#### TASK-XXX: Task Name
**Status:** 📋 NOT STARTED / 🔄 IN PROGRESS / ✅ COMPLETED  
**Priority:** 🔴 CRITICAL / 🟡 HIGH / 🟢 MEDIUM / ⚪ LOW  
**Assigned:** Name/Agent  
**Due:** YYYY-MM-DD

**Description:**
What needs to be done

**Acceptance Criteria:**
- [ ] Criterion 1
- [ ] Criterion 2

**Implementation Notes:**
Technical details

**Related Files:**
- `path/to/file.py`

**Dependencies:**
- TASK-XXX (must complete first)
```

---

## 🔄 Update Schedule

- **Daily:** Update task status
- **Weekly:** Review priorities
- **Milestone:** Major updates

---

*Keep this file updated! It's our source of truth.*
