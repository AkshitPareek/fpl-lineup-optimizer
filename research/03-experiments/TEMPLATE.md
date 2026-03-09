# Experiment Template

> Copy this file to `research/03-experiments/YYYY-MM-DD-experiment-name/README.md`  
> Fill out all sections completely  
> Remove instructional comments (lines starting with >)

---

# EXP-XXX: [Descriptive Experiment Title]

> Use format: EXP-XXX where XXX is next number  
> Example: EXP-002: LSTM with Attention Mechanism

**Status:** 🟡 In Progress / ✅ Complete / ❌ Failed / 🔄 Needs Revision  
**Date Started:** YYYY-MM-DD  
**Date Completed:** YYYY-MM-DD (if complete)  
**Owner:** [Agent Name / GitHub Handle]  
**Branch/Commit:** [Git branch or commit hash]  
**Related Issues:** #[issue-number]  

---

## 1. Hypothesis

### 1.1 Primary Hypothesis

> State the main hypothesis clearly. Must be falsifiable.

**H1:** [Clear statement of what you expect to happen]

**Rationale:** [Why do you expect this? Link to theory/literature]

### 1.2 Secondary Hypotheses (if applicable)

**H2:** [Secondary prediction]

**H3:** [Tertiary prediction]

### 1.3 Success Criteria

> Define what constitutes "success" before running experiment

- [ ] Criterion 1: [e.g., RMSE reduction > 0.1]
- [ ] Criterion 2: [e.g., Statistical significance p < 0.05]
- [ ] Criterion 3: [e.g., No regression on other metrics]

---

## 2. Background & Motivation

### 2.1 Problem Context

> What problem are we trying to solve?

[Explain the context - why this experiment matters]

### 2.2 Related Work

> Link to previous experiments, papers, concepts

- [Previous Experiment EXP-XXX](../YYYY-MM-DD-previous-exp/) - [What we learned]
- [Concept: Knowledge Graph](../../knowledge-graph/concepts.md#concept-name)
- [External Paper](link) - [Key finding that motivates this]

### 2.3 Theoretical Justification

> Why should this work from a theoretical perspective?

[Explain the theory behind the approach]

---

## 3. Methodology

### 3.1 Experimental Design

> High-level approach

**Type:** [A/B Test / Ablation Study / Architecture Search / etc.]  
**Control:** [Baseline model to compare against]  
**Treatment:** [Your modified model]  
**Sample Size:** [Number of training samples]  
**Random Seed:** [Seed used for reproducibility]

### 3.2 Changes Made

> Exact, detailed description of what was changed

#### Code Changes
```python
# Before (baseline):
[Show relevant baseline code]

# After (treatment):
[Show your modified code]
```

#### Configuration Changes
```yaml
# config.yaml changes
hyperparameters:
  learning_rate: 0.001  # Changed from 0.01
  batch_size: 256       # Unchanged
  new_parameter: 0.5    # Added
```

#### Feature Changes (if applicable)
- Added: [List new features]
- Modified: [List modified features]
- Removed: [List removed features]

### 3.3 Implementation Details

> Technical details needed for reproduction

**Dependencies:**
- Package versions: [e.g., torch==2.1.0]
- New requirements: [If any added]

**Training Setup:**
- Hardware: [GPU model, RAM]
- Training time: [Duration]
- Number of runs: [For statistical significance]

**Data Pipeline:**
- Preprocessing: [Any changes]
- Augmentation: [If any]
- Splits: [train/val/test sizes]

### 3.4 Evaluation Metrics

> How will success be measured?

| Metric | Description | Target | Importance |
|--------|-------------|--------|------------|
| RMSE | Root Mean Squared Error | < 2.3 | Primary |
| MAE | Mean Absolute Error | < 1.8 | Secondary |
| Spearman ρ | Rank correlation | > 0.35 | Secondary |
| Top-10 Acc | % actual top-10 in predicted | > 45% | Secondary |
| Inference Time | ms per prediction | < 50ms | Constraint |

---

## 4. Results

### 4.1 Quantitative Results

> Present all numerical results

#### Primary Metrics

| Model | RMSE | MAE | R² | Spearman | Inference (ms) |
|-------|------|-----|-----|----------|----------------|
| Baseline | X.XXXX | X.XXXX | X.XX | X.XXXX | XX.X |
| Treatment | X.XXXX | X.XXXX | X.XX | X.XXXX | XX.X |
| **Δ (Improvement)** | **-X.XX** | **-X.XX** | **+X.XX** | **+X.XXX** | **+X.X** |

#### Statistical Significance

| Comparison | Metric | Diff | 95% CI | p-value | Significant? |
|------------|--------|------|--------|---------|--------------|
| Baseline vs Treatment | RMSE | -0.XXX | [-X.XX, -X.XX] | 0.XXXX | ✅ Yes / ❌ No |
| Baseline vs Treatment | MAE | -0.XXX | [-X.XX, -X.XX] | 0.XXXX | ✅ Yes / ❌ No |

**Effect Size (Cohen's d):** X.XX ([negligible/small/medium/large])

#### Secondary Metrics

| Metric | Baseline | Treatment | Improvement |
|--------|----------|-----------|-------------|
| Top-5 Accuracy | XX.X% | XX.X% | +X.X% |
| Top-10 Accuracy | XX.X% | XX.X% | +X.X% |
| Within 1pt | XX.X% | XX.X% | +X.X% |
| Within 2pt | XX.X% | XX.X% | +X.X% |
| Captain Accuracy | XX.X% | XX.X% | +X.X% |

### 4.2 Visualizations

> Include plots, learning curves, feature importance

**Figure 1: Learning Curves**
```
[Insert plot or describe what it shows]
Path: artifacts/learning_curves.png
```

**Figure 2: Prediction Scatter Plot**
```
[Actual vs Predicted scatter]
Path: artifacts/prediction_scatter.png
```

**Figure 3: Feature Importance (if applicable)**
```
[Top features]
Path: artifacts/feature_importance.png
```

### 4.3 Ablation Studies (if applicable)

> Break down contribution of each component

| Component | RMSE | Contribution |
|-----------|------|--------------|
| Full Model | X.XXX | - |
| Without Component A | X.XXX | +0.XX |
| Without Component B | X.XXX | +0.XX |
| Baseline Only | X.XXX | +0.XX |

---

## 5. Analysis & Interpretation

### 5.1 Key Findings

> Summarize what was discovered

1. **Finding 1:** [Most important result]
   - Evidence: [What data supports this]
   - Implication: [What this means]

2. **Finding 2:** [Second finding]
   - Evidence: [Supporting data]
   - Implication: [What this means]

### 5.2 Comparison to Hypothesis

> Did results match expectations?

**H1:** [Restate hypothesis]  
**Result:** [What actually happened]  
**Conclusion:** ✅ Supported / ❌ Not Supported / 🟡 Partially Supported  
**Explanation:** [Why did it match/not match?]

### 5.3 Unexpected Results

> Document any surprises

- **Unexpected 1:** [What happened]
  - Possible explanation: [Why it might have happened]
  - Action: [How to investigate further]

### 5.4 Limitations

> Be honest about limitations

1. **Limitation 1:** [e.g., Small sample size for position X]
   - Impact: [How this affects conclusions]
   - Mitigation: [What we did about it]

2. **Limitation 2:** [e.g., Hyperparameters not fully tuned]
   - Impact: [Effect on results]
   - Mitigation: [Future work]

### 5.5 Threats to Validity

| Threat | Severity | Mitigation |
|--------|----------|------------|
| Overfitting | Medium | Used held-out test set |
| Data Leakage | Low | Strict temporal split |
| Multiple Comparisons | Medium | Bonferroni correction |
| Selection Bias | Low | Random train/val split |

---

## 6. Artifacts

### 6.1 Code

**Modified Files:**
- `backend/ml/model_file.py` - [What changed]
- `backend/ml/training.py` - [What changed]

**New Files:**
- `backend/ml/new_module.py` - [Purpose]

**Commit:** [Git commit hash with link]

### 6.2 Models

| Model | Path | Description | Size |
|-------|------|-------------|------|
| Baseline | `models/baseline/model.pkl` | Original model | X MB |
| Treatment | `models/experiment_name/model.pkl` | Improved model | X MB |

### 6.3 Data

**Datasets Used:**
- Training: `datasets/fpl_points_v1/train_X.npy` (n=X, features=Y)
- Validation: `datasets/fpl_points_v1/validation_X.npy` (n=X)
- Test: `datasets/fpl_points_v1/test_X.npy` (n=X)

**Generated Data:**
- Predictions: `artifacts/predictions.csv`
- Metrics: `artifacts/metrics.json`

### 6.4 Configuration

**Config File:** `artifacts/config.yaml`
```yaml
[Full configuration used]
```

### 6.5 Logs

**Training Log:** `artifacts/training.log`  
**Validation Log:** `artifacts/validation.log`

---

## 7. Discussion

### 7.1 Relationship to Prior Work

> How does this compare to previous experiments?

- **EXP-XXX:** [Similar experiment] - [How results compare]
- **EXP-YYY:** [Different approach] - [Why this is better/worse]

### 7.2 Theoretical Implications

> What does this tell us about FPL prediction?

[Discuss theoretical insights]

### 7.3 Practical Implications

> How should this change our approach?

1. **For Model Development:** [Guidance]
2. **For FPL Strategy:** [How to use this]
3. **For Future Research:** [What to explore]

---

## 8. Conclusion

### 8.1 Summary

> One-paragraph summary

[Concise summary of what was done and found]

### 8.2 Success Assessment

**Hypothesis:** [Supported / Not Supported / Partially Supported]  
**Criteria Met:** X / Y  
**Overall:** ✅ Success / ❌ Failure / 🟡 Partial Success

### 8.3 Recommendations

1. **Adopt/Reject:** [Should this be part of main codebase?]
2. **Further Tuning:** [What parameters to optimize next]
3. **Integration:** [How to integrate with existing system]

---

## 9. Next Steps

### 9.1 Immediate Follow-ups

- [ ] **Experiment EXP-XXX+1:** [Specific next experiment]
  - Rationale: [Why this is next]
  - Expected Timeline: [How long]

### 9.2 Future Directions

- [Longer-term research direction 1]
- [Longer-term research direction 2]

### 9.3 Questions Raised

1. [Question that needs investigation]
2. [Another open question]

---

## 10. References

### 10.1 Internal References

- [Previous Experiment](../YYYY-MM-DD-previous-exp/)
- [Methodology Doc](../../02-methodology/methodology-name.md)
- [Concept](../../knowledge-graph/concepts.md#concept-name)

### 10.2 External References

- [Paper Title](link) - [Key contribution]
- [Blog Post](link) - [Relevant insight]
- [Documentation](link) - [Technical reference]

---

## 11. Appendix

### A.1 Detailed Hyperparameters

```python
[Full hyperparameter configuration]
```

### A.2 Raw Results

```json
[Complete metrics output]
```

### A.3 Additional Plots

[More figures if needed]

### A.4 Reproduction Instructions

```bash
# Step-by-step commands to reproduce
1. git checkout [commit]
2. python backend/scripts/prepare_data.py
3. python backend/scripts/train.py --config artifacts/config.yaml
4. python backend/scripts/evaluate.py --model artifacts/model.pkl
```

---

## 12. Metadata

**Tags:** #lstm #attention #neural-network #feature-engineering #ensemble  
**Concepts:** [Links to knowledge graph concepts]  
**Status History:**
- 2026-03-09: Created
- 2026-03-09: Started implementation
- 2026-03-10: Completed

**Reviewers:** [Names of reviewers]  
**Review Status:** ✅ Approved / 🟡 Pending / ❌ Needs Revision  

---

> **END OF TEMPLATE**
> 
> Remember:
> 1. Remove all lines starting with ">" (these are instructions)
> 2. Fill out EVERY section
> 3. Be specific with numbers and evidence
> 4. Link to related work
> 5. Document limitations honestly
> 6. Keep the "END OF TEMPLATE" marker above
