# Methodology

> How we approach the FPL prediction research problem

---

## 📚 Methodology Documents

| Document | Status | Description |
|----------|--------|-------------|
| [Validation Framework](./validation-framework.md) | ✅ Complete | 5-layer testing system |
| [Autonomous Research Harness](./autonomous-research-harness.md) | ✅ Complete | Self-driving experiment runner |
| [Experimental Design](./experimental-design.md) | 🟡 Planned | How we design experiments |
| [Data Pipeline](./data-pipeline.md) | 🟡 Planned | Data processing methodology |
| [Feature Engineering](./feature-engineering.md) | 🟡 Planned | Feature strategies |
| [Model Improvements](./model-improvements.md) | 🟡 Planned | Planned improvements |
| [Ensemble Strategies](./ensemble-strategies.md) | 🟡 Planned | Ensemble methods |
| [A/B Testing Framework](./ab-testing-framework.md) | ✅ Complete | Statistical testing |

---

## 🎯 Core Methodological Principles

### 1. Statistical Rigor First

Every claim must be statistically validated:
- P-values < 0.05
- Effect sizes reported
- Confidence intervals provided
- Multiple comparisons controlled

### 2. Automation

Research should be reproducible without human intervention:
- Automated experiment running
- Auto-documentation
- State management
- Progress tracking

### 3. Incremental Progress

Small, testable improvements:
- Each experiment isolated
- Fast feedback loops
- Build on validated findings

---

## 🔬 Key Systems

### Validation Framework

```
Layer 1: Unit Tests ──────────────► Component correctness
Layer 2: Data Validation ─────────► Data quality
Layer 3: Model Validation ────────► Model behavior
Layer 4: Regression Testing ──────► No degradation
Layer 5: Benchmark Testing ───────► Statistical comparison
```

**File:** [validation-framework.md](./validation-framework.md)

### Autonomous Research Harness

```
Strategy Library ───► Experiment Runner ───► Improvement Tracker
       │                     │                      │
       ▼                     ▼                      ▼
  10 strategies     4-layer validation      Trajectory analysis
       │                     │                      │
       └─────────────────────┴──────────────────────┘
                          │
                          ▼
                   Research State
```

**File:** [autonomous-research-harness.md](./autonomous-research-harness.md)

**Usage:**
```bash
# Run 10 experiments autonomously
python backend/scripts/autoresearch_harness.py run --auto --max-runs 10

# Check status
python backend/scripts/improvement_tracker.py dashboard
```

---

## 📊 Evaluation Methodology

### Primary Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| RMSE | Overall accuracy | < 2.3 |
| MAE | Interpretable error | < 1.8 |
| Spearman ρ | Ranking quality | > 0.35 |

### Secondary Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| Top-5 Accuracy | Captain picks | > 40% |
| Top-10 Accuracy | Squad selection | > 45% |
| Within 2pt | Practical accuracy | > 60% |

---

## 🧪 Experimental Standards

### Every Experiment Must Have

1. **Falsifiable Hypothesis**
   - Clear prediction
   - Testable outcome

2. **Control Group**
   - Baseline comparison
   - Same conditions

3. **Statistical Testing**
   - Significance test
   - Effect size
   - Confidence interval

4. **Documentation**
   - Full methodology
   - Results
   - Interpretation

---

## 🔄 Research Workflow

```
1. HYPOTHESIS
   └─ Formulate testable prediction
   
2. DESIGN
   └─ Plan experiment
   
3. IMPLEMENTATION
   └─ Code the strategy
   
4. VALIDATION
   └─ Run full test suite
   
5. EVALUATION
   └─ A/B test vs baseline
   
6. DOCUMENTATION
   └─ Record everything
   
7. DECISION
   └─ Accept or reject
```

---

## 🎯 Research Questions Addressed

### RQ1: Validation in High-Noise Domains
**How do we validate improvements when ground truth is stochastic?**

**Approach:**
- Statistical significance testing
- Effect size thresholds
- Multiple metrics
- Trajectory analysis

### RQ2: Autonomous Experimentation
**How much can research be automated?**

**Approach:**
- Pre-defined strategy library
- Automated A/B testing
- State management
- Stopping criteria

### RQ3: Improvement Detection
**How do we know if we're actually improving?**

**Approach:**
- 4-layer validation
- Trajectory tracking
- Trend analysis
- Convergence detection

---

## 📈 Success Criteria

### For Individual Experiments

- [ ] P-value < 0.05
- [ ] Effect size > 0.2
- [ ] Relative improvement > 1%
- [ ] All validation layers pass

### For Research Program

- [ ] Improvement rate > 30%
- [ ] Total improvement > 5%
- [ ] Reproducible results
- [ ] Publication-quality documentation

---

## 🛠️ Tools & Infrastructure

### Validation
```bash
# Full validation suite
python backend/scripts/validation_runner.py --all

# A/B testing
python backend/scripts/ab_testing_cli.py compare \
    --model-a baseline --model-b treatment
```

### Autonomous Research
```bash
# Run experiments
python backend/scripts/autoresearch_harness.py run --auto

# Track improvements
python backend/scripts/improvement_tracker.py dashboard

# Analyze results
python backend/scripts/improvement_tracker.py analyze
```

---

## 📚 Next Steps

1. **Review [Validation Framework](./validation-framework.md)** - Understand testing
2. **Review [Autonomous Harness](./autonomous-research-harness.md)** - Understand automation
3. **Run First Experiment** - Try the system
4. **Document Results** - Follow experiment template

---

**All methodology follows the [Research Rules](../RESEARCH_RULES.md).**
