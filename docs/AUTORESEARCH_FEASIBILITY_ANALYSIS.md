# AutoResearch for FPL: Feasibility Analysis & Research Paper Proposal

## Executive Summary

**Verdict:** ✅ **Worth implementing, with adaptations.**

The autoresearch methodology can be applied to FPL model optimization, but requires significant modifications from Karpathy's LLM-focused approach. The constrained data regime and noisy targets of FPL prediction create unique challenges that could make for an interesting research contribution.

---

## 1. Understanding AutoResearch (Karpathy's Approach)

### Core Concept
- **Single-file modification**: Agent only edits `train.py`
- **Fixed time budget**: 5-minute experiments for fair comparison
- **Clear metric**: validation bits per byte (lower = better)
- **Autonomous iteration**: ~100 experiments overnight
- **Human oversight**: Program.md guides the agent

### Key Design Decisions
| Aspect | Karpathy's LLM Setup | FPL Adaptation Needed |
|--------|---------------------|----------------------|
| Data | ~10B tokens (infinite) | ~50K samples (finite) |
| Target | Next token prediction | Points prediction (noisy) |
| Metric | val_bpb (deterministic) | RMSE/MAE (stochastic) |
| Budget | 5 min wall-clock | Variable (seconds to minutes) |
| Generalization | Cross-domain | Time-series specific |

---

## 2. Challenges for FPL AutoResearch

### Challenge 1: Data Scarcity ⭐ Critical
**Problem:** FPL has limited historical data
- ~38 gameweeks × 500 players = ~19K samples per season
- Only 3-5 seasons of reliable data = ~100K samples total
- LLMs have billions of tokens; we have thousands of samples

**Impact:**
- High risk of overfitting
- Limited training runs before data exhaustion
- Can't use "more compute" to overcome data limits

**Mitigation:**
- Use time-series cross-validation
- Fix validation set across experiments
- Add regularization constraints to search space
- Use data augmentation (synthetic players, bootstrapping)

### Challenge 2: Noisy Targets ⭐ Critical
**Problem:** Football is inherently stochastic
- A player can score 15 points one week, 2 the next
- Same inputs → wildly different outputs
- Model can't learn what it can't predict

**Impact:**
- High variance in experiment results
- Difficult to distinguish signal from noise
- "Improvements" may be luck

**Mitigation:**
- Run multiple seeds per experiment
- Use confidence intervals, not point estimates
- Require statistical significance (p < 0.05)
- Track metrics over multiple gameweeks

### Challenge 3: Distribution Shift ⭐ High
**Problem:** Football evolves
- Team tactics change
- Player roles change
- New metrics become available
- Season-to-season non-stationarity

**Impact:**
- Model that worked last season may fail this season
- Historical "improvements" don't guarantee future performance

**Mitigation:**
- Use rolling validation (always test on most recent)
- Weight recent data more heavily
- Monitor for concept drift

### Challenge 4: Evaluation Cost
**Problem:** How do we know if FPL predictions are "better"?
- Can't just check RMSE on test set
- Real test is actual FPL performance
- Need backtesting framework (which we have!)

**Mitigation:**
- Use comprehensive metrics (RMSE, rank correlation, top-k, backtest points)
- Multi-objective optimization
- Simulate full FPL season for each candidate

---

## 3. Adapted AutoResearch Design for FPL

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FPL AUTORESEARCH                         │
├─────────────────────────────────────────────────────────────┤
│  Human Layer                                                │
│  ├── research_program.md    # Research goals & constraints  │
│  ├── experiment_history.md  # Log of all experiments        │
│  └── review.py              # Human review interface        │
├─────────────────────────────────────────────────────────────┤
│  Agent Layer                                                │
│  ├── experiment_proposer.py # Suggest next experiments      │
│  ├── code_modifier.py       # Modify model/training code    │
│  ├── evaluator.py           # Run experiments & evaluate    │
│  └── learner.py             # Learn from experiment history │
├─────────────────────────────────────────────────────────────┤
│  Experiment Layer                                           │
│  ├── model_variants/        # Generated model code          │
│  ├── training_logs/         # Training curves               │
│  └── evaluation_results/    # Metrics & comparisons         │
├─────────────────────────────────────────────────────────────┤
│  Validation Layer (Your Framework)                          │
│  ├── Data validation        # Test data integrity           │
│  ├── Model validation       # Test model behavior           │
│  ├── A/B testing            # Statistical comparison        │
│  └── Backtesting            # Real FPL simulation           │
└─────────────────────────────────────────────────────────────┘
```

### Modified Design Decisions

| Aspect | Karpathy (LLM) | FPL Adaptation |
|--------|---------------|----------------|
| **What to modify** | train.py | Multiple files (model.py, features.py, ensemble.py) |
| **Experiment budget** | Fixed 5 min | Variable based on model complexity |
| **Success metric** | Single (val_bpb) | Multi-objective (RMSE, rank_corr, backtest_pts) |
| **Validation** | Fixed held-out | Time-series CV (respects temporal order) |
| **Iterations** | ~100/night | ~20-50/night (slower training) |
| **Search space** | Architecture, optimizer | Features, architecture, ensemble, preprocessing |

### Key Files

```
autoresearch/
├── research_program.md       # Your research goals & constraints
├── experiment_history.jsonl  # Log of all experiments
├── candidate_generator.py    # Generate model variants
├── experiment_runner.py      # Train & evaluate models
├── objective.py              # Multi-objective scoring
├── orchestrator.py           # Main loop
└── utils/
    ├── model_patcher.py      # Safe code modification
    ├── statistical_tests.py  # Significance testing
    └── visualization.py      # Plot experiment history
```

---

## 4. Research Paper Potential: ⭐⭐⭐⭐ HIGH

### Novelty Assessment

**What's new:**
1. **First application** of autonomous ML research to sports prediction
2. **Domain-specific challenges**: Noisy targets, distribution shift, limited data
3. **Multi-objective optimization**: Balance prediction accuracy vs roster construction
4. **Human-AI collaboration**: How to guide agents in high-stakes domains

**Contribution types:**
- ✅ Methodology: Adapted autoresearch for tabular/time-series data
- ✅ Empirical: Comprehensive comparison of AI-discovered vs human-designed models
- ✅ Analysis: What works (and doesn't) in automated sports prediction

### Target Venues

| Venue | Fit | Likelihood |
|-------|-----|------------|
| **KDD (Applied Data Science)** | High | Medium |
| **NeurIPS (Workshop)** | High | High |
| **ICML (Workshop)** | High | High |
| **AAAI (Sports Analytics)** | Very High | Medium |
| **Journal of Sports Analytics** | Very High | High |
| **MIT Sloan Sports Conf** | Very High | Medium |
| **Kaggle Blog/Research** | High | High |
| **arXiv + Blog post** | - | Very High |

### Paper Structure Proposal

```
Title: "AutoFPL: Autonomous Research for Fantasy Sports Prediction"

Abstract: 
- Adapt autoresearch to sports prediction domain
- Handle unique challenges: noise, distribution shift, limited data
- Achieve X% improvement over human-designed baseline
- Insights on what strategies AI discovers

1. Introduction
   - FPL as a prediction problem
   - Challenges: noise, non-stationarity, limited data
   - Autoresearch as potential solution

2. Related Work
   - AutoML & Neural Architecture Search
   - Sports prediction literature
   - Automated research systems

3. Method: AutoFPL Framework
   3.1 Search Space Design
       - Feature engineering strategies
       - Model architectures
       - Ensemble methods
   3.2 Multi-Objective Optimization
       - Prediction accuracy (RMSE)
       - Rank correlation (Spearman)
       - Portfolio optimization (backtest points)
   3.3 Statistical Validation
       - Significance testing between candidates
       - Controlling for multiple comparisons
   3.4 Safety & Constraints
       - Data leakage prevention
       - Overfitting detection
       - Human oversight mechanisms

4. Experimental Setup
   4.1 Dataset: FPL 2019-2024
   4.2 Baseline: Human-designed XGBoost ensemble
   4.3 Compute: Single GPU, overnight runs
   4.4 Evaluation: Walk-forward validation

5. Results
   5.1 Discovered Improvements
       - Feature engineering strategies
       - Architecture modifications
       - Ensemble weights
   5.2 Comparison to Human Baseline
   5.3 Ablation Studies
   5.4 Transfer to Real FPL (2024-25 season)

6. Analysis: What Did AI Discover?
   - Unexpected feature interactions
   - Novel ensemble strategies
   - Position-specific architectures
   - Limitations & failures

7. Discussion
   - Generalizability to other sports
   - Computational cost vs benefit
   - Future of automated sports analytics

8. Conclusion
```

### Key Research Questions

1. **RQ1:** Can autonomous research outperform human-designed models in noisy, low-data regimes?

2. **RQ2:** What types of improvements does AI discover vs humans?
   - Humans: Domain knowledge, interpretable features
   - AI: Complex interactions, non-obvious regularization

3. **RQ3:** How to design search spaces for high-stakes, explainable domains?
   - Medical diagnosis, sports betting, financial forecasting

4. **RQ4:** How to validate improvements when ground truth is stochastic?

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Finalize validation framework (✅ Done!)
- [ ] Establish strong baseline
- [ ] Design experiment tracking system
- [ ] Create safe code modification framework

### Phase 2: Core System (Week 3-4)
- [ ] Implement candidate generator
- [ ] Build multi-objective evaluator
- [ ] Create experiment orchestrator
- [ ] Add statistical significance testing

### Phase 3: Search Space (Week 5-6)
- [ ] Define feature engineering search space
- [ ] Define model architecture search space
- [ ] Define ensemble strategy search space
- [ ] Implement safety constraints

### Phase 4: Experiments (Week 7-10)
- [ ] Run initial overnight experiments
- [ ] Analyze discovered improvements
- [ ] Iterate on search space
- [ ] Run comprehensive study (100+ experiments)

### Phase 5: Analysis & Paper (Week 11-14)
- [ ] Statistical analysis of results
- [ ] Ablation studies
- [ ] Real FPL season validation
- [ ] Write paper

---

## 6. Risk Assessment

### High Risk ⚠️
- **No meaningful improvements discovered**
  - Mitigation: Start with known improvements to validate system
  
- **Overfitting to validation set**
  - Mitigation: Strict train/val/test splits, final test only at end

- **Computational cost exceeds benefit**
  - Mitigation: Track cost per experiment, compare to manual effort

### Medium Risk ⚠️
- **Code modification introduces bugs**
  - Mitigation: Extensive unit tests, sandboxing, rollback capability

- **Paper rejected due to "just AutoML" critique**
  - Mitigation: Emphasize domain challenges, thorough analysis

### Low Risk ✅
- **System doesn't work at all**
  - We have strong baseline framework to build on

---

## 7. Recommendations

### ✅ DO Implement If:
1. You're genuinely interested in the research question
2. You have time for 10-14 week commitment
3. You're comfortable with uncertainty in results
4. You want to contribute to AutoML + Sports Analytics literature

### ❌ DON'T Implement If:
1. You just want a better FPL model (manual tuning may be faster)
2. You need guaranteed results for a deadline
3. You're not interested in publishing

### 🎯 Hybrid Approach (Recommended)
Instead of full autoresearch, implement **"Guided AutoResearch"**:

```
Human: Define candidate improvements based on domain knowledge
        ↓
System: Automatically test each candidate with proper validation
        ↓
System: Report statistical significance & effect sizes
        ↓
Human: Select best candidates for next round
        ↓
Repeat
```

This is more practical and still paper-worthy:
- **Title:** "Human-Guided Autonomous Optimization for Fantasy Sports Prediction"
- **Focus:** Best of both worlds (human intuition + automated validation)

---

## 8. Next Steps

If you want to proceed:

1. **Confirm scope**: Full autoresearch vs guided approach?

2. **Set up experiment tracking**:
   ```bash
   # Create experiment database
   mkdir autoresearch_experiments
   # Design schema for tracking experiments
   ```

3. **Define search space** (start small):
   - Feature engineering: polynomial, interaction, ratio features
   - Models: XGBoost, LightGBM, small NN
   - Ensembles: weighted average, stacking

4. **Run pilot study** (10 experiments):
   - Test the infrastructure
   - Validate that improvements can be found
   - Estimate compute requirements

5. **Scale up** if pilot succeeds

---

## Conclusion

**AutoResearch for FPL is feasible and potentially impactful**, but requires significant adaptation from Karpathy's LLM-focused approach. The unique challenges (noise, distribution shift, limited data) are actually strengths for a research paper—they differentiate your work from generic AutoML.

**The hybrid "guided autoresearch" approach offers the best risk/reward ratio**: leverage human domain knowledge for candidate generation while using automation for rigorous validation and statistical testing.

Would you like me to:
1. **Implement the guided autoresearch system**?
2. **Design the full experiment tracking infrastructure**?
3. **Create the research paper outline and start writing**?
4. **Build a pilot study** to validate feasibility?
