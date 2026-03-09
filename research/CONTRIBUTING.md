# Contributing to AutoFPL Research

> **For:** AI Agents (Kimi, Claude, Codex, etc.)  
> **Purpose:** How to contribute to this research project  
> **Standard:** Top-tier ML conference quality  
> **Version:** 1.0

---

## 🎯 Agent Mission

You are a research assistant contributing to a publication-quality ML research project. Your work will be part of a paper targeting NeurIPS/KDD/ICML venues.

**Quality Standard:** Everything you produce should be ready for peer review.

---

## 📋 Before You Start

### 1. Read the Project Overview
- [ ] [Main README](./README.md) - Understand current status
- [ ] [Problem Statement](./01-background/problem-statement.md) - Understand what we're solving
- [ ] [Current Phase](./README.md#-current-research-phase) - Know what's in progress

### 2. Check Existing Work
- [ ] [Experiment Log](./03-experiments/) - Don't duplicate work
- [ ] [Knowledge Graph](./knowledge-graph/) - Understand concepts
- [ ] [Results](./04-results/) - Know current baselines

### 3. Understand Requirements
- [ ] [Validation Requirements](#-validation-requirements)
- [ ] [Documentation Template](#-documentation-template)
- [ ] [Code Quality Standards](#-code-quality-standards)

---

## 🔄 Research Workflow

### Standard Experiment Lifecycle

```
1. HYPOTHESIS
   └── Write down what you expect to happen
   └── Link to related work/concept
   └── Get approval (if high-risk)

2. DESIGN
   └── Plan the experiment
   └── Identify confounding variables
   └── Choose evaluation metrics
   └── Update [experimental-design.md](./02-methodology/experimental-design.md)

3. IMPLEMENTATION
   └── Create experiment directory
   └── Write code following standards
   └── Add unit tests
   └── Document as you go

4. VALIDATION
   └── Run full validation suite
   └── Check for data leakage
   └── Verify statistical significance
   └── Document results

5. DOCUMENTATION
   └── Fill out experiment template
   └── Update knowledge graph
   └── Cross-reference related work
   └── Add to results section

6. REVIEW
   └── Self-review against checklist
   └── Update main README
   └── Commit with clear message
   └── Mark experiment complete
```

---

## 🧪 Creating a New Experiment

### Step 1: Create Directory

```bash
# Use naming convention: YYYY-MM-DD-experiment-name/
mkdir -p research/03-experiments/$(date +%Y-%m-%d)-your-experiment-name
```

### Step 2: Copy Template

```bash
cp research/03-experiments/TEMPLATE.md \
   research/03-experiments/$(date +%Y-%m-%d)-your-experiment-name/README.md
```

### Step 3: Fill Out Template

See [Template Guide](#-experiment-template-guide) below.

### Step 4: Update Main README

Add to [Active Experiments](./README.md#-active-experiments) table.

---

## 📄 Documentation Template

### Experiment README Structure

Every experiment must have a `README.md` with these sections:

```markdown
# EXP-XXX: Experiment Title

**Status:** 🟡 In Progress / ✅ Complete / ❌ Failed  
**Date:** YYYY-MM-DD  
**Owner:** [Agent Name]  
**Branch:** [Git branch if applicable]

## Hypothesis

What we expect to happen and why.

## Methodology

Exactly what was changed.

## Results

Quantitative outcomes.

## Statistical Analysis

Significance testing.

## Interpretation

What results mean.

## Artifacts

Links to models, data, code.

## Next Steps

What to try next.

## References

Links to related work.
```

See full [TEMPLATE.md](./03-experiments/TEMPLATE.md) for details.

---

## ✅ Validation Requirements

### Pre-Experiment Checklist

- [ ] Hypothesis is falsifiable
- [ ] Methodology is clear and reproducible
- [ ] Evaluation metrics are defined
- [ ] Confounding variables are identified
- [ ] Sample size is sufficient

### Post-Experiment Checklist

- [ ] **Data Validation**
  ```bash
  pytest backend/tests/test_data_validation.py -v
  ```
  - No NaN or infinite values
  - Consistent shapes
  - No data leakage

- [ ] **Model Validation**
  ```bash
  pytest backend/tests/test_model_validation.py -v
  ```
  - Predictions in reasonable range
  - Handles edge cases
  - Inference time acceptable

- [ ] **A/B Testing**
  ```bash
  python backend/scripts/ab_testing_cli.py compare \
      --model-a baseline \
      --model-b your_model \
      --output results.json
  ```
  - Statistical significance p < 0.05
  - Effect size calculated
  - Confidence intervals provided

- [ ] **Regression Testing**
  ```bash
  pytest backend/tests/test_model_regression.py -v
  ```
  - No degradation > 5% on baseline metrics
  - All existing tests pass

- [ ] **Full Validation**
  ```bash
  python backend/scripts/validation_runner.py --all
  ```
  - All checks pass
  - HTML report generated

### Documentation Checklist

- [ ] Experiment README complete
- [ ] Code commented
- [ ] Results reproducible from README
- [ ] Knowledge graph updated
- [ ] Main README updated
- [ ] Related experiments linked

---

## 📝 Code Quality Standards

### Python Code Style

```python
"""
Module docstring explaining purpose.

References:
    - Link to related research/experiment
    - Link to methodology doc
"""

import numpy as np
from typing import Dict, List, Tuple

# Constants at module level
MAX_EPOCHS = 100
LEARNING_RATE = 0.001


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hyperparams: Dict[str, float],
) -> Tuple[object, Dict[str, List[float]]]:
    """
    Train model with given hyperparameters.
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        X_val: Validation features
        y_val: Validation targets
        hyperparams: Dictionary of hyperparameters
        
    Returns:
        model: Trained model
        history: Training history with 'loss', 'val_loss' keys
        
    Raises:
        ValueError: If input shapes are invalid
        
    Example:
        >>> model, history = train_model(X, y, Xv, yv, {'lr': 0.01})
        >>> print(f"Best val loss: {min(history['val_loss'])}")
    """
    # Implementation
    pass


def evaluate_model(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Returns dictionary with metrics:
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - r2: R-squared
        - spearman: Rank correlation
    """
    # Implementation
    pass
```

### Research-Grade Comments

```python
# ❌ Bad: Unclear what this does
x = x * 2

# ✅ Good: Explains why
# Apply log transform to reduce impact of outliers (hauls)
# See: research/02-methodology/preprocessing.md#log-transform
x = np.log1p(x)
```

### Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Functions | verb_noun | `train_model`, `evaluate_predictions` |
| Classes | PascalCase | `FPLPredictor`, `ValidationFramework` |
| Constants | UPPER_SNAKE | `MAX_EPOCHS`, `RANDOM_SEED` |
| Variables | snake_case | `train_data`, `model_predictions` |
| Experiments | YYYY-MM-DD-name | `2026-03-09-baseline-establishment` |

---

## 🗺️ Knowledge Graph Guidelines

### When to Update Knowledge Graph

Update [knowledge-graph/](./knowledge-graph/) when you:
- Introduce a new concept
- Discover an important relationship
- Change methodology
- Find unexpected results

### How to Document Concepts

Create entry in [concepts.md](./knowledge-graph/concepts.md):

```markdown
## Concept Name

**Type:** Algorithm / Metric / Domain Concept / Methodology  
**Status:** Proposed / Validated / Deprecated

### Definition
Clear, precise definition.

### Why It Matters
Relevance to FPL prediction.

### Implementation
How it's implemented in code.

### Experiments
Links to experiments using this concept.

### References
External papers, documentation.

### Related Concepts
- [Link to related concept](./concepts.md#related)
```

### Linking Experiments

Every experiment should link to:
- Concepts it tests
- Previous related experiments
- Methodology it follows
- Results it produces

---

## 🔄 Review Process

### Self-Review Checklist

Before marking experiment complete:

- [ ] Can someone else reproduce this from the README?
- [ ] Are all claims supported by data?
- [ ] Is statistical significance reported?
- [ ] Are limitations acknowledged?
- [ ] Is next step clear?

### Peer Review (Optional but Encouraged)

For high-impact experiments:
1. Tag experiment as "needs-review"
2. Another agent reviews
3. Address feedback
4. Mark as "reviewed"

---

## 🚨 Common Mistakes to Avoid

### ❌ Documentation Mistakes

| Mistake | Why It's Bad | Fix |
|---------|-------------|-----|
| "Improved the model" | Vague, unverifiable | "Reduced RMSE from 2.5 to 2.3 (p=0.03)" |
| No hypothesis | Can't falsify | State expected outcome before experiment |
| Missing baselines | Can't evaluate improvement | Always compare to established baseline |
| No effect size | Statistical ≠ practical | Report Cohen's d |
| No next steps | Research stalls | Always suggest follow-up |

### ❌ Methodology Mistakes

| Mistake | Why It's Bad | Fix |
|---------|-------------|-----|
| Data leakage | Invalid results | Strict train/val/test splits |
| Multiple testing | False positives | Bonferroni correction |
| Cherry-picking | Biased results | Pre-register experiments |
| No random seed | Not reproducible | Set all random seeds |
| Overfitting validation | Optimistic metrics | Hold-out test set |

### ❌ Code Mistakes

| Mistake | Why It's Bad | Fix |
|---------|-------------|-----|
| Hard-coded paths | Not portable | Use relative paths, config files |
| No error handling | Crashes | Try-except with informative messages |
| Magic numbers | Unclear intent | Named constants |
| No logging | Can't debug | Use logging module |
| Undocumented dependencies | Setup failures | requirements.txt, environment.yml |

---

## 🎓 Research Paper Standards

### Writing Quality

- **Clarity:** Would a reviewer understand this?
- **Precision:** Exact numbers, not approximations
- **Completeness:** All details for reproduction
- **Honesty:** Report negative results too
- **Context:** How does this fit the broader research?

### Figure Standards

Every figure needs:
- Clear title
- Axis labels with units
- Legend (if multiple lines)
- Caption explaining key takeaway
- Source data in repository

### Table Standards

Every table needs:
- Clear header
- Units specified
- Alignment (numbers right, text left)
- Caption explaining significance
- Bold best results

---

## 📦 Commit Guidelines

### Commit Message Format

```
[type]: [short description]

[detailed description]

Refs: [experiment-id]
```

Types:
- `feat:` New feature/improvement
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code restructuring
- `exp:` Experiment results

Examples:
```
feat: Add LSTM attention mechanism

Implemented attention layer for better form tracking.
Reduces RMSE by 0.15 (p=0.02, d=0.45).

Refs: EXP-004
```

```
exp: Complete baseline establishment

Established baselines for all models:
- XGBoost: RMSE=2.45, MAE=1.89
- LightGBM: RMSE=2.38, MAE=1.82
- Ensemble: RMSE=2.29, MAE=1.76

All results documented in research/03-experiments/2026-03-09-baseline-establishment/

Refs: EXP-001
```

---

## 🆘 Getting Help

### If You're Stuck

1. Check [knowledge-graph/](./knowledge-graph/) for related concepts
2. Review [similar experiments](./03-experiments/)
3. Read [methodology docs](./02-methodology/)
4. Ask for clarification on approach

### If Validation Fails

1. Read error message carefully
2. Check [troubleshooting guide](../docs/TESTING_FRAMEWORK.md#troubleshooting)
3. Review failed test file
4. Document the issue before fixing

### If Results Are Unexpected

1. Don't ignore it - document it
2. Check for bugs first
3. Consider if it's actually a finding
4. Discuss in experiment README

---

## ✅ Agent Checklist (Before Each Session)

```markdown
## Session Start Checklist

- [ ] Read main README for current status
- [ ] Check if assigned to specific experiment
- [ ] Review relevant methodology docs
- [ ] Understand validation requirements

## During Session

- [ ] Document as you go (don't wait until end)
- [ ] Run tests frequently
- [ ] Update experiment README
- [ ] Commit incremental progress

## Session End Checklist

- [ ] All tests passing
- [ ] Experiment README complete
- [ ] Knowledge graph updated
- [ ] Main README updated
- [ ] Clear commit message
- [ ] Next steps documented
```

---

## 📚 Resources

### Internal
- [Main README](./README.md)
- [Knowledge Graph](./knowledge-graph/)
- [Experiment Template](./03-experiments/TEMPLATE.md)
- [Validation Framework](../docs/TESTING_FRAMEWORK.md)

### External
- [Google ML Style Guide](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Papers with Code](https://paperswithcode.com/) (for baselines)
- [Distill.pub](https://distill.pub/) (for clear explanations)

---

**Remember:** Quality over quantity. One well-documented, rigorously validated experiment is worth more than ten sloppy ones.

**Goal:** Every contribution should be paper-ready.

**Motto:** *"If it's not documented, it didn't happen."*
