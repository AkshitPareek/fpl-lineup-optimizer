# Experiments

> Documentation of all research experiments

---

## 📋 Experiment Registry

### Completed Experiments

| ID | Name | Status | Date | Result | RMSE |
|----|------|--------|------|--------|------|
| EXP-001 | Baseline Establishment | ✅ Complete | 2026-03-09 | Baseline set | 2.29 |

### Planned Experiments

| ID | Name | Status | Hypothesis | Expected |
|----|------|--------|------------|----------|
| EXP-002 | Feature Engineering v1 | 🟡 Planned | Polynomial features improve RMSE | -0.05 to -0.15 |
| EXP-003 | Position-Specific Models | 🟡 Planned | Separate models per position help | -0.05 to -0.15 |
| EXP-004 | Attention LSTM | 🟡 Planned | Attention improves form modeling | -0.10 to -0.20 |
| EXP-005 | Optimized Ensemble | 🟡 Planned | Learned weights beat uniform | -0.05 to -0.10 |

---

## 🚀 Running Experiments

### Method 1: Autonomous Harness (Recommended)

The harness selects strategies and runs experiments automatically:

```bash
# Run 10 experiments autonomously
python backend/scripts/autoresearch_harness.py run --auto --max-runs 10

# Check progress
python backend/scripts/improvement_tracker.py dashboard
```

**The harness handles:**
- Strategy selection
- Validation
- A/B testing
- Documentation
- Improvement tracking

### Method 2: Manual Experiment

For specific strategies or custom experiments:

```bash
# Create experiment directory
mkdir research/03-experiments/$(date +%Y-%m-%d)-my-experiment

# Copy template
cp research/03-experiments/TEMPLATE.md \
   research/03-experiments/$(date +%Y-%m-%d)-my-experiment/README.md

# Edit README with hypothesis, methodology
# Run experiment
# Fill in results
# Update knowledge graph
```

---

## 📊 How We Know If We're Improving

### 4-Layer Validation

Every experiment must pass:

| Layer | Test | Threshold | Purpose |
|-------|------|-----------|---------|
| 1 | P-value | < 0.05 | Statistically significant |
| 2 | Effect size | > 0.2 | Practically meaningful |
| 3 | Relative improvement | > 1% | Worth the change |
| 4 | Trend analysis | Improving | Not just lucky |

### Improvement Metrics

Track these to know if research is working:

```bash
python backend/scripts/improvement_tracker.py analyze
```

**Key metrics:**
- **Improvement Rate:** % of experiments that improve (target: > 30%)
- **Total Improvement:** Cumulative RMSE reduction (target: > 5%)
- **Recent Trend:** Direction of last 5 runs (target: "improving")
- **Effect Size:** Average Cohen's d (target: > 0.3)

### Visual Confirmation

```bash
python backend/scripts/improvement_tracker.py visualize
```

Generates:
- `improvement_trajectory.png` - RMSE over time
- `detailed_analysis.png` - Effect sizes, p-values, trends

---

## 🛑 Stopping Criteria

Research automatically stops when:

1. **Max runs reached** (e.g., 10 experiments)
2. **Too many failures** (3 consecutive)
3. **Plateau detected** (5 runs without improvement)
4. **Converged** (CV < 0.01 in recent runs)
5. **Low improvement rate** (< 10% after 10 runs)

---

## 📝 Documentation Requirements

Every experiment must include:

1. **Hypothesis** - What we expect and why
2. **Methodology** - Exact changes made
3. **Results** - Quantitative outcomes
4. **Statistical Analysis** - P-values, effect sizes
5. **Interpretation** - What it means
6. **Next Steps** - What to try next

See [TEMPLATE.md](./TEMPLATE.md) for full format.

---

## 📈 Experiment Statistics

### Current Progress

```
Total Experiments: 1
Successful Improvements: 0 (N/A for baseline)
Failed Experiments: 0
Improvement Rate: N/A

Best RMSE: 2.29 (Ensemble, EXP-001)
Total Improvement: 0% (baseline)
```

### Strategy Success Rates

| Strategy | Tried | Successes | Rate |
|----------|-------|-----------|------|
| Baseline | 1 | N/A | N/A |
| Others | 0 | 0 | - |

---

## 🔗 Links

- [Template](./TEMPLATE.md) - Create new experiments
- [Autonomous Harness](../02-methodology/autonomous-research-harness.md) - Run experiments
- [Validation Framework](../02-methodology/validation-framework.md) - Validation standards
- [Knowledge Graph](../knowledge-graph/) - Concepts and relationships

---

**Remember:** If it's not documented, it didn't happen.
