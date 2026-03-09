# Knowledge Graph Index

> AutoFPL Research Knowledge Graph  
> Purpose: Navigate concepts, relationships, and discoveries  
> Last Updated: 2026-03-09

---

## 🗺️ Graph Overview

This knowledge graph captures:
- **Concepts:** Key ideas, algorithms, metrics
- **Relationships:** How concepts connect
- **Experiments:** What we've tried
- **Insights:** What we've learned

---

## 📑 Quick Navigation

### By Category

| Category | Description | Link |
|----------|-------------|------|
| **Core Concepts** | Fundamental ideas | [concepts.md](./concepts.md) |
| **Relationships** | How concepts connect | [relationships.md](./relationships.md) |
| **Domain** | FPL-specific concepts | [concepts.md#domain](./concepts.md#domain-concepts) |
| **Methodology** | Research methods | [concepts.md#methodology](./concepts.md#methodology-concepts) |
| **Models** | ML architectures | [concepts.md#models](./concepts.md#model-concepts) |
| **Metrics** | Evaluation measures | [concepts.md#metrics](./concepts.md#evaluation-metrics) |

### By Concept Type

#### 🧠 Algorithms & Models
- [Gradient Boosting](./concepts.md#gradient-boosting)
- [LSTM](./concepts.md#lstm)
- [Attention Mechanism](./concepts.md#attention-mechanism)
- [Ensemble Methods](./concepts.md#ensemble-methods)

#### 📊 Evaluation & Metrics
- [RMSE](./concepts.md#rmse)
- [Spearman Correlation](./concepts.md#spearman-correlation)
- [Top-k Accuracy](./concepts.md#top-k-accuracy)
- [A/B Testing](./concepts.md#ab-testing)

#### 🔬 Methodology
- [Validation Framework](./concepts.md#validation-framework)
- [Cross-Validation](./concepts.md#cross-validation)
- [Statistical Significance](./concepts.md#statistical-significance)
- [Effect Size](./concepts.md#effect-size)

#### ⚽ FPL Domain
- [Expected Points (xP)](./concepts.md#expected-points)
- [Gameweek](./concepts.md#gameweek)
- [Fixture Difficulty](./concepts.md#fixture-difficulty)
- [Form](./concepts.md#player-form)

---

## 🕸️ Relationship Map

```
┌─────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE GRAPH STRUCTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │   Domain     │◄───────►│   Methodology │                    │
│  │   (FPL)      │         │  (Research)   │                    │
│  └──────┬───────┘         └──────┬───────┘                    │
│         │                        │                             │
│         │    ┌──────────────┐    │                             │
│         └───►│   Models     │◄───┘                             │
│              │  (ML Algos)  │                                  │
│              └──────┬───────┘                                  │
│                     │                                          │
│              ┌──────┴───────┐                                  │
│              │   Metrics    │                                  │
│              │ (Evaluation) │                                  │
│              └──────────────┘                                  │
│                                                                 │
│  All connected through:                                         │
│  • Experiments (what we've tried)                               │
│  • Results (what we found)                                      │
│  • Insights (what we learned)                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Search by Use Case

### "I want to improve model performance"
→ Start with: [Model Concepts](./concepts.md#model-concepts)  
→ Related: [Ensemble Methods](./concepts.md#ensemble-methods), [Feature Engineering](./concepts.md#feature-engineering)

### "I need to validate an improvement"
→ Start with: [Validation Framework](./concepts.md#validation-framework)  
→ Related: [A/B Testing](./concepts.md#ab-testing), [Statistical Significance](./concepts.md#statistical-significance)

### "I want to understand FPL prediction"
→ Start with: [Domain Concepts](./concepts.md#domain-concepts)  
→ Related: [Expected Points](./concepts.md#expected-points), [Form](./concepts.md#player-form)

### "I need to choose an evaluation metric"
→ Start with: [Evaluation Metrics](./concepts.md#evaluation-metrics)  
→ Related: [RMSE](./concepts.md#rmse), [Rank Correlation](./concepts.md#rank-correlation)

---

## 📈 Concept Evolution

Track how concepts develop over time:

| Date | Concept | Status | Related Experiment |
|------|---------|--------|-------------------|
| 2026-03-09 | Validation Framework | ✅ Validated | EXP-001 |
| 2026-03-09 | A/B Testing | ✅ Validated | EXP-001 |
| 2026-03-09 | Baseline Models | ✅ Established | EXP-001 |
| 2026-03-?? | LSTM Attention | 🟡 Planned | EXP-004 |
| 2026-03-?? | Position Models | 🟡 Planned | EXP-003 |

---

## 🔗 Cross-References

### Experiments → Concepts

| Experiment | Primary Concept | Secondary Concepts |
|------------|-----------------|-------------------|
| [EXP-001](../03-experiments/2026-03-09-baseline-establishment/) | Baseline Establishment | Validation Framework, A/B Testing |
| EXP-004 (planned) | LSTM Attention | Recurrent Networks, Attention Mechanism |
| EXP-003 (planned) | Position-Specific Models | Model Specialization, Domain Knowledge |

### Concepts → Experiments

| Concept | Validated By | Planned In |
|---------|--------------|------------|
| Validation Framework | EXP-001 | All future |
| A/B Testing | EXP-001 | All future |
| LSTM | - | EXP-004 |
| Ensemble | - | EXP-005 |

---

## 📝 Adding to the Knowledge Graph

### When to Add

Add a new concept when:
- Introducing new algorithm/method
- Discovering important pattern
- Formalizing learned insight
- Connecting disparate ideas

### How to Add

1. **Add to concepts.md:**
   ```markdown
   ## New Concept Name
   
   **Type:** [Algorithm/Metric/Domain/Methodology]
   **Status:** [Proposed/Validated/Deprecated]
   **Introduced:** [Experiment ID or Date]
   
   ### Definition
   [Clear definition]
   
   ### Related
   - [Link to related concept]
   - [Link to experiment]
   ```

2. **Update relationships.md:**
   ```markdown
   ## New Concept Relationships
   
   - **is-a:** [Parent concept]
   - **uses:** [Dependency concepts]
   - **improves:** [What this improves]
   - **tested-by:** [Experiment ID]
   ```

3. **Update this index:**
   - Add to navigation tables
   - Update evolution log

---

## 🎯 Key Insights (So Far)

### Validated Insights

1. **Validation is Critical** ✅
   - Source: EXP-001
   - Finding: Rigorous validation prevents false positives
   - Impact: All future experiments use validation framework

2. **Baseline Diversity Matters** ✅
   - Source: EXP-001
   - Finding: Multiple model types (XGB, LGBM, RF) have similar performance
   - Impact: Ensemble potential confirmed

### Proposed Insights (Pending Validation)

3. **Attention Improves Form Modeling** 🟡
   - Hypothesis: Attention mechanisms better capture recent form
   - To Test: EXP-004

4. **Position-Specific Models Help** 🟡
   - Hypothesis: Different positions need different architectures
   - To Test: EXP-003

---

## 🚧 Work in Progress

### Concepts Being Developed

- [ ] Uncertainty Quantification
- [ ] Multi-Task Learning (points + minutes)
- [ ] Transfer Learning Across Seasons
- [ ] Online Learning / Adaptation

### Relationships Being Explored

- [ ] Feature Importance vs Position
- [ ] Model Complexity vs Overfitting
- [ ] Historical Form vs Future Performance

---

## 📚 Full Index

### All Concepts (Alphabetical)

| Concept | Type | Status | Link |
|---------|------|--------|------|
| A/B Testing | Methodology | ✅ Validated | [link](./concepts.md#ab-testing) |
| Attention Mechanism | Algorithm | 🟡 Planned | [link](./concepts.md#attention-mechanism) |
| Baseline | Methodology | ✅ Validated | [link](./concepts.md#baseline) |
| Captain | Domain | ✅ Documented | [link](./concepts.md#captain) |
| Cohen's d | Methodology | ✅ Validated | [link](./concepts.md#effect-size) |
| Cross-Validation | Methodology | ✅ Validated | [link](./concepts.md#cross-validation) |
| Effect Size | Methodology | ✅ Validated | [link](./concepts.md#effect-size) |
| Ensemble Methods | Algorithm | ✅ Documented | [link](./concepts.md#ensemble-methods) |
| Expected Points | Domain | ✅ Documented | [link](./concepts.md#expected-points) |
| Feature Engineering | Methodology | 🟡 Planned | [link](./concepts.md#feature-engineering) |
| Fixture Difficulty | Domain | ✅ Documented | [link](./concepts.md#fixture-difficulty) |
| Form | Domain | ✅ Documented | [link](./concepts.md#player-form) |
| Gameweek | Domain | ✅ Documented | [link](./concepts.md#gameweek) |
| Gradient Boosting | Algorithm | ✅ Validated | [link](./concepts.md#gradient-boosting) |
| LSTM | Algorithm | 🟡 Planned | [link](./concepts.md#lstm) |
| MAE | Metric | ✅ Validated | [link](./concepts.md#mae) |
| RMSE | Metric | ✅ Validated | [link](./concepts.md#rmse) |
| Spearman Correlation | Metric | ✅ Validated | [link](./concepts.md#spearman-correlation) |
| Statistical Significance | Methodology | ✅ Validated | [link](./concepts.md#statistical-significance) |
| Top-k Accuracy | Metric | ✅ Validated | [link](./concepts.md#top-k-accuracy) |
| Validation Framework | Methodology | ✅ Validated | [link](./concepts.md#validation-framework) |
| XGBoost | Algorithm | ✅ Validated | [link](./concepts.md#gradient-boosting) |

---

**Maintained by:** Research Team  
**Update Frequency:** After each experiment  
**Last Review:** 2026-03-09
