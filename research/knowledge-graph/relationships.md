# Knowledge Graph Relationships

> Maps connections between concepts, experiments, and findings  
> Last Updated: 2026-03-09

---

## Relationship Types

| Type | Symbol | Meaning | Example |
|------|--------|---------|---------|
| **is-a** | → | Subtype relationship | XGBoost → Gradient Boosting |
| **uses** | ⟹ | Depends on | Ensemble uses XGBoost |
| **improves** | ↑ | Increases performance | Attention improves LSTM |
| **tested-by** | 🧪 | Validated through | Validation Framework tested-by EXP-001 |
| **produces** | ⇒ | Generates | Model produces Predictions |
| **measures** | 📏 | Evaluates | RMSE measures Accuracy |
| **related-to** | ↔ | Associated with | Form related-to Recent Performance |

---

## Concept Hierarchy

### Models

```
Machine Learning Model
├── Tree-Based Model
│   ├── Gradient Boosting
│   │   ├── XGBoost [tested-by: EXP-001]
│   │   ├── LightGBM [tested-by: EXP-001]
│   │   └── Gradient Boosting (sklearn) [tested-by: EXP-001]
│   ├── Random Forest [tested-by: EXP-001]
│   └── Decision Tree
├── Linear Model
│   ├── Ridge [tested-by: EXP-001]
│   ├── Lasso
│   └── Linear Regression
├── Neural Network
│   ├── Recurrent Network
│   │   └── LSTM [tested-by: EXP-004 (planned)]
│   ├── Attention Mechanism [tested-by: EXP-004 (planned)]
│   │   └── Self-Attention
│   └── Feedforward Network
└── Ensemble Method
    ├── Weighted Average [tested-by: EXP-001]
    ├── Stacking
    └── Blending
```

### Evaluation Metrics

```
Evaluation Metric
├── Error Metric
│   ├── RMSE [measures: Overall Accuracy]
│   ├── MAE [measures: Average Error]
│   └── MSE
├── Correlation Metric
│   ├── Spearman ρ [measures: Rank Correlation]
│   ├── Pearson r
│   └── Kendall τ
├── Ranking Metric
│   ├── Top-k Accuracy [measures: Captain Selection]
│   ├── NDCG
│   └── Precision@k
└── Business Metric
    └── Backtest Points [measures: Actual FPL Performance]
```

### Methodology

```
Research Methodology
├── Validation
│   ├── Data Validation [tested-by: EXP-001]
│   ├── Model Validation [tested-by: EXP-001]
│   ├── Regression Testing [tested-by: EXP-001]
│   └── A/B Testing [tested-by: EXP-001]
├── Statistical Testing
│   ├── Hypothesis Testing
│   │   ├── t-test
│   │   └── Wilcoxon Test
│   ├── Effect Size
│   │   └── Cohen's d
│   └── Multiple Comparisons
│       └── Bonferroni Correction
└── Experimental Design
    ├── Cross-Validation
    ├── Train/Test Split
    └── Bootstrap Sampling
```

---

## Experiment Relationships

### EXP-001: Baseline Establishment

```
EXP-001
├── tests → [XGBoost, LightGBM, Random Forest, Ensemble]
├── validates → [Validation Framework, A/B Testing, Data Pipeline]
├── produces → [Baseline Metrics]
├── enables → [EXP-002, EXP-003, EXP-004, EXP-005]
└── related-to → [Methodology Validation]
```

### EXP-002: Feature Engineering (Planned)

```
EXP-002
├── builds-on → [EXP-001]
├── tests → [Polynomial Features, Interaction Features, Ratio Features]
├── uses → [Validation Framework]
├── compares-to → [EXP-001 Baseline]
└── enables → [Better Model Performance]
```

### EXP-004: Attention LSTM (Planned)

```
EXP-004
├── builds-on → [EXP-001]
├── tests → [LSTM with Attention]
├── compares-to → [XGBoost Baseline, LightGBM Baseline]
├── hypothesis → [Attention improves form modeling]
└── related-to → [Sequential Modeling, Form Prediction]
```

---

## Domain-Methodology Connections

### FPL Domain → ML Concepts

```
Fantasy Premier League
├── Player
│   ├── has-feature → [Form, Price, Position]
│   ├── produces → [Points (target)]
│   └── used-in → [Squad Selection]
├── Gameweek
│   ├── contains → [Fixtures]
│   └── produces → [Points]
├── Fixture
│   ├── has → [Fixture Difficulty]
│   └── influences → [Expected Points]
└── Squad
    ├── requires → [15 Players]
    ├── optimizes → [Expected Points]
    └── constraint → [Budget, Position Limits]
```

### Problem → Solution Mapping

```
High Noise in Targets
├── causes → [Difficulty Validating Improvements]
├── addressed-by → [Statistical Significance Testing]
├── requires → [Larger Effect Sizes]
└── measured-by → [Confidence Intervals]

Limited Data
├── causes → [Overfitting Risk]
├── addressed-by → [Regularization, Cross-Validation]
├── requires → [Conservative Improvements]
└── measured-by → [Test Set Performance]

Need for Ranking
├── requires → [Spearman Correlation]
├── optimizes → [Top-k Accuracy]
└── measured-by → [Captain Accuracy]
```

---

## Dependencies

### Infrastructure Dependencies

```
Validation Framework
├── depends-on → [Data Pipeline]
├── depends-on → [Model Pipeline]
├── depends-on → [Statistical Testing]
├── enables → [All Future Experiments]
└── tested-by → [EXP-001]

A/B Testing CLI
├── depends-on → [Model Benchmark]
├── depends-on → [Baseline Metrics]
├── uses → [Statistical Tests]
└── enables → [Model Comparison]
```

### Model Dependencies

```
Ensemble
├── uses → [XGBoost]
├── uses → [LightGBM]
├── uses → [Random Forest]
├── improves → [Single Model Performance]
└── tested-by → [EXP-001]

LSTM with Attention
├── extends → [LSTM]
├── adds → [Attention Mechanism]
├── improves → [Sequential Modeling]
└── tested-by → [EXP-004 (planned)]
```

---

## Improvement Chains

### Potential Improvement Pathways

```
Current State (EXP-001)
│
├── Path 1: Feature Engineering
│   ├── EXP-002a: Polynomial Features
│   ├── EXP-002b: Interaction Features
│   ├── EXP-002c: Ratio Features
│   └── EXP-002d: Temporal Features
│
├── Path 2: Architecture
│   ├── EXP-003: Position-Specific Models
│   ├── EXP-004: Attention LSTM
│   └── EXP-009: Transformer
│
├── Path 3: Ensembling
│   ├── EXP-005: Optimized Weights
│   ├── EXP-006: Stacking
│   └── EXP-007: Blending
│
└── Path 4: Training
    ├── EXP-008: Hyperparameter Optimization
    ├── EXP-010: Multi-Task Learning
    └── EXP-011: Transfer Learning
```

---

## Validation Coverage

### What Validates What

```
Data Validation
├── validates → [Data Quality]
├── checks → [No NaN, No Inf, Consistent Shapes]
├── prevents → [Training Crashes]
└── required-for → [All Experiments]

Model Validation
├── validates → [Model Behavior]
├── checks → [Reasonable Predictions, Edge Cases]
├── prevents → [Bad Models in Production]
└── required-for → [Model Deployment]

Regression Testing
├── validates → [No Degradation]
├── checks → [Performance vs Baseline]
├── prevents → [Broken Improvements]
└── required-for → [Commit/Merge]

A/B Testing
├── validates → [Statistical Significance]
├── checks → [p-values, Effect Sizes]
├── prevents → [False Positives]
└── required-for → [Claiming Improvement]
```

---

## Knowledge Accumulation

### What We've Learned

```
EXP-001 Findings
├── Ensemble > Single Models (numerically)
├── Tree Models Cluster Closely
├── Rank Correlation Critical
├── High Noise Limits R²
└── Validation Framework Works

→ Implies: Focus on ensembles and ranking metrics
→ Implies: Need larger samples for significance
→ Implies: Attention to feature engineering
```

### Open Questions

```
Unvalidated Hypotheses
├── Does Attention Help? → [Test: EXP-004]
├── Do Position Models Help? → [Test: EXP-003]
├── Which Features Matter? → [Test: EXP-002]
├── Can We Automate? → [Future Research]
└── What's Theoretical Limit? → [Future Analysis]
```

---

## Cross-References

### Documents Referencing Each Other

| Document | References | Referenced By |
|----------|-----------|---------------|
| [Problem Statement](../01-background/problem-statement.md) | - | Validation Framework, EXP-001 |
| [Validation Framework](../02-methodology/validation-framework.md) | Problem Statement | EXP-001, EXP-002+ |
| [EXP-001](../03-experiments/2026-03-09-baseline-establishment/) | Validation Framework, Concepts | Future Experiments |
| [Concepts](concepts.md) | - | All Documents |
| [Index](index.md) | All | - |

---

## Adding New Relationships

When documenting a new experiment or concept, add relationships:

```markdown
## New Experiment: EXP-00X

```
EXP-00X
├── builds-on → [Previous Experiment]
├── tests → [Concept]
├── uses → [Infrastructure]
├── produces → [Results]
└── enables → [Future Work]
```
```

Update this file to maintain the knowledge graph.

---

**Navigation:**
- [← Back to Index](./index.md)
- [↑ Concepts](./concepts.md)
- [↓ Experiments](../03-experiments/)
