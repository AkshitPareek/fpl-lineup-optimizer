# AutoFPL: Autonomous Optimization for Fantasy Premier League Prediction

> **Research Status:** 🟡 Active Development  
> **Current Phase:** Validation Framework & Baseline Establishment  
> **Target Venue:** NeurIPS/KDD/ICML Workshop or Journal of Sports Analytics  
> **Last Updated:** 2026-03-09

---

## 🎯 Research Objective

Develop and validate a systematic methodology for improving machine learning models in high-noise, limited-data prediction domains, using Fantasy Premier League (FPL) as a case study. 

**Core Question:** *How do we rigorously validate model improvements when ground truth is stochastic and data is scarce?*

---

## 📊 Quick Status Dashboard

| Component | Status | Documentation | Last Update |
|-----------|--------|---------------|-------------|
| Validation Framework | ✅ Complete | [Validation Framework](./02-methodology/validation-framework.md) | 2026-03-09 |
| Autonomous Research Harness | ✅ Complete | [Harness](./02-methodology/autonomous-research-harness.md) | 2026-03-09 |
| Data Pipeline | ✅ Stable | [Data Validation](./02-methodology/data-pipeline.md) | 2026-03-09 |
| Baseline Models | ✅ Trained | [Baseline](./03-experiments/2026-03-09-baseline-establishment/) | 2026-03-09 |
| Feature Engineering | 🟡 Planned | [Features](./02-methodology/feature-engineering.md) | - |
| Model Improvements | 🟡 Queue | [Improvements](./02-methodology/model-improvements.md) | - |
| A/B Testing | ✅ Ready | [A/B Testing](./02-methodology/ab-testing-framework.md) | 2026-03-09 |
| Ensemble Methods | 🟡 Planned | [Ensembles](./02-methodology/ensemble-strategies.md) | - |

**Overall Progress:** 40% → Target: 100% by 2026-06-01

**Overall Progress:** 35% → Target: 100% by 2026-06-01

---

## 🗺️ Repository Map (Knowledge Graph)

### For First-Time Visitors

1. **Start Here:** [Research Overview](./01-background/problem-statement.md)
2. **Current Work:** [Validation Framework](./02-methodology/validation-framework.md) 
3. **How to Contribute:** [Agent Guidelines](./CONTRIBUTING.md)
4. **Experiment Template:** [Experiment Template](./03-experiments/TEMPLATE.md)

### Quick Navigation

```
research/
├── 📖 START HERE
│   └── README.md (you are here)
│
├── 01-background/          # Why this research matters
│   ├── problem-statement.md
│   ├── related-work.md
│   └── fpl-domain.md
│
├── 02-methodology/         # How we approach the problem
│   ├── validation-framework.md    ⭐ CURRENT FOCUS
│   ├── data-pipeline.md
│   ├── experimental-design.md
│   ├── feature-engineering.md
│   ├── model-improvements.md      📋 QUEUE
│   └── ensemble-strategies.md
│
├── 03-experiments/         # What we've tried
│   ├── TEMPLATE.md               📋 Use this for new experiments
│   ├── 2026-03-09-baseline-establishment/  ⭐ DONE
│   └── [future-experiments]/
│
├── 04-results/             # What we found
│   ├── benchmarks.md
│   ├── ab-tests.md
│   └── performance-tracking.md
│
├── 05-paper/               # Publication artifacts
│   ├── outline.md
│   └── sections/
│
├── 06-artifacts/           # Models, data, configs
│   ├── models/
│   ├── datasets/
│   └── configs/
│
└── knowledge-graph/        # Concept map
    ├── index.md
    ├── concepts.md
    └── relationships.md
```

---

## 🔬 Current Research Phase

### Phase 1: Foundation & Validation (🟡 IN PROGRESS)

**Goal:** Establish rigorous validation infrastructure before any model improvements.

**Why this matters:** In high-noise domains like FPL, we need to distinguish real improvements from random fluctuations. This framework is our contribution to the methodology.

**Current Tasks:**
- [x] Build 5-layer validation framework (unit → data → model → regression → benchmark)
- [x] Establish baseline metrics for all models
- [x] Create A/B testing infrastructure with statistical significance
- [ ] Document baseline results
- [ ] Create experiment tracking system

**Key Deliverable:** [Validation Framework Documentation](./02-methodology/validation-framework.md)

---

## 🧪 Active Experiments

| ID | Name | Status | Owner | Hypothesis | Key Finding |
|----|------|--------|-------|------------|-------------|
| EXP-001 | Baseline Establishment | ✅ Complete | Initial | Establish reference metrics | [Results](./03-experiments/2026-03-09-baseline-establishment/) |
| EXP-002 | LightGBM vs XGBoost | ✅ Complete | Autonomous | LightGBM better than XGBoost | [No sig. improvement](./03-experiments/2026-03-09-exp-001-lightgbm-vs-xgboost/) |
| EXP-003 | Feature Engineering v1 | 🟡 Planned | TBD | Polynomial features improve RMSE | - |
| EXP-003 | Position-Specific Models | 🟡 Planned | TBD | Separate models per position help | - |
| EXP-004 | Attention LSTM | 🟡 Planned | TBD | Attention mechanism captures form | - |
| EXP-005 | Optimized Ensembles | 🟡 Planned | TBD | Learned weights beat uniform | - |

---

## 📋 Improvement Queue

Priority-ranked improvements to test (from [model-improvements.md](./02-methodology/model-improvements.md)):

### High Priority (Test First)
1. **Enable LSTM Attention** - Already implemented, just need to enable
2. **Position-Specific Models** - Train separate models per position
3. **Hyperparameter Optimization** - Increase Optuna trials (50→200)
4. **Log-Transformed Targets** - Reduce impact of outliers

### Medium Priority
5. **Polynomial Features** - Capture non-linear relationships
6. **Advanced Ensemble Weighting** - Optimize meta-learner weights
7. **Temporal Features** - Add rest days, momentum
8. **Feature Selection** - Remove redundant features

### Low Priority (Research Phase 2)
9. **Transformer Architecture** - Self-attention over sequences
10. **Uncertainty Quantification** - MC Dropout, ensembles
11. **Multi-Task Learning** - Predict points + minutes simultaneously
12. **Neural Architecture Search** - Auto-discover architectures

---

## 🎓 Research Questions

### Primary RQ
**RQ1:** How can we rigorously validate model improvements in high-noise, limited-data prediction domains?

### Secondary RQs
**RQ2:** What types of model improvements provide the most reliable gains for FPL prediction?

**RQ3:** How should validation frameworks be designed for stochastic prediction targets?

**RQ4:** Can systematic A/B testing outperform ad-hoc model improvement in sports analytics?

---

## 📚 Key Concepts

### Domain Concepts
- **FPL (Fantasy Premier League):** Fantasy football game where players assemble squads
- **Gameweek:** Weekly round of matches
- **Expected Points (xP):** Predicted points for a player in upcoming gameweek
- **Captain:** Player whose points are doubled

### Methodology Concepts
- **Validation Framework:** 5-layer testing system (data → model → regression → benchmark)
- **A/B Testing:** Statistical comparison of two model variants
- **Baseline:** Reference model for comparison
- **Effect Size:** Cohen's d measuring practical significance

### Evaluation Metrics
- **RMSE:** Root Mean Squared Error (primary)
- **MAE:** Mean Absolute Error
- **Spearman ρ:** Rank correlation (crucial for FPL)
- **Top-k Accuracy:** % of actual top players in predicted top-k

See [knowledge-graph/concepts.md](./knowledge-graph/concepts.md) for full glossary.

---

## 🛠️ For Contributors (AI Agents)

### Before You Start
1. Read [CONTRIBUTING.md](./CONTRIBUTING.md)
2. Check [Experiment Template](./03-experiments/TEMPLATE.md)
3. Review [Current Status](#-quick-status-dashboard)
4. Understand [Validation Requirements](#-validation-requirements)

### Documentation Rules (⚠️ CRITICAL)

**Every experiment MUST include:**
- [ ] Hypothesis (what you expect to happen)
- [ ] Methodology (exact changes made)
- [ ] Results (quantitative outcomes)
- [ ] Statistical significance (p-values, confidence intervals)
- [ ] Interpretation (what it means)
- [ ] Next steps (what to try next)

**Never commit without:**
- [ ] Running full validation suite
- [ ] Updating experiment log
- [ ] Cross-referencing related work
- [ ] Adding to knowledge graph

### Validation Requirements

**ALL improvements must pass:**
1. ✅ Data validation (no NaN, consistent shapes)
2. ✅ Model validation (reasonable predictions, no crashes)
3. ✅ A/B test vs baseline (statistical significance p < 0.05)
4. ✅ Regression test (no degradation on existing metrics)
5. ✅ Documentation review (follows template)

**Command to run:**
```bash
python backend/scripts/validation_runner.py --all
```

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@misc{autofpl2026,
  title={AutoFPL: Systematic Validation and Improvement of Machine Learning Models for Fantasy Sports Prediction},
  author={[Authors TBD]},
  year={2026},
  howpublished={\url{https://github.com/[repo]/research}},
  note={Work in progress}
}
```

---

## 🔗 External Resources

### Datasets
- [FPL Historical Data](https://github.com/vaastav/Fantasy-Premier-League)
- [Understat](https://understat.com/) (xG data)
- [FBref](https://fbref.com/) (advanced stats)

### Related Research
- [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) (inspiration)
- [AutoGluon](https://auto.gluon.ai/) (AutoML baseline)
- [MLflow](https://mlflow.org/) (experiment tracking)

### FPL Community
- [FPL Algorithm](https://fpl.readthedocs.io/)
- [Fantasy Football Scout](https://www.fantasyfootballscout.co.uk/)

---

## 📞 Research Team

| Role | Entity | Contact | Responsibility |
|------|--------|---------|----------------|
| Principal Investigator | User | [GitHub] | Research direction, domain expertise |
| Research Assistant | AI Agent | Kimi Code CLI | Implementation, documentation |
| Reviewer | AI Agent | Future agents | Peer review, validation |

---

## 📝 Changelog

### 2026-03-09
- **Milestone:** Validation framework complete
- **Added:** 5-layer testing system
- **Added:** A/B testing CLI
- **Added:** Regression testing suite
- **Status:** Ready for first experiments

### 2026-03-08
- **Milestone:** Project initialization
- **Added:** Research folder structure
- **Added:** Knowledge graph foundation

---

## 🚦 Decision Log

| Date | Decision | Rationale | Status |
|------|----------|-----------|--------|
| 2026-03-09 | Use validation framework as entry point | Need rigor before improvements | ✅ Adopted |
| 2026-03-09 | Target top-tier ML venues | Novel methodology contribution | ✅ Adopted |
| 2026-03-09 | Document everything research-grade | Reproducibility & publication | ✅ Adopted |
| 2026-03-09 | Defer full autoresearch | Too risky for current timeline | ✅ Deferred |

See [knowledge-graph/decisions.md](./knowledge-graph/decisions.md) for full log.

---

## 🎯 Next Actions

### Immediate (This Week)
- [x] Build autonomous research harness ✅
- [x] Create improvement tracking system ✅
- [ ] Run first autonomous experiments (try harness)
- [ ] Document baseline establishment results

### Short-term (Next 2 Weeks)
- [ ] Run 10 autonomous experiments
- [ ] EXP-002: Feature Engineering v1
- [ ] EXP-003: Position-Specific Models
- [ ] EXP-004: Attention LSTM

### Medium-term (Next Month)
- [ ] Complete improvement queue
- [ ] Analyze which strategies work best
- [ ] Write methodology section
- [ ] Prepare workshop submission

## 🚀 Quick Start: Run Autonomous Experiments

The research harness can now run experiments autonomously:

```bash
# Navigate to project
cd /home/akshit/fpl-lineup-optimizer

# Run 10 experiments autonomously
python backend/scripts/autoresearch_harness.py run --auto --max-runs 10

# Monitor progress (in another terminal)
python backend/scripts/improvement_tracker.py dashboard

# Check status anytime
python backend/scripts/autoresearch_harness.py status

# View final report
python backend/scripts/improvement_tracker.py analyze
```

**The harness will:**
1. Select best strategy automatically
2. Run full validation on each experiment
3. Perform A/B testing with statistical rigor
4. Track if we're actually improving
5. Stop if we hit a plateau or keep failing
6. Document everything automatically

**See:** [Autonomous Research Harness](./02-methodology/autonomous-research-harness.md) for details.

---

**🤖 Agent Note:** If you're reading this for the first time, start with [CONTRIBUTING.md](./CONTRIBUTING.md), then check the [experiment template](./03-experiments/TEMPLATE.md). Welcome to the project!

**📧 Human Note:** This is a living document. Update it as the research evolves.
