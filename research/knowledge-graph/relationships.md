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
├── enables → [EXP-002, EXP-003, EXP-004, EXP-005, EXP-006]
└── related-to → [Methodology Validation]
```

### EXP-002: LightGBM vs XGBoost

```
EXP-002
├── builds-on → [EXP-001]
├── tests → [LightGBM, Gradient Boosting Algorithms]
├── compares-to → [XGBoost Baseline]
├── produces → [RMSE: 0.8526, +0.48% improvement]
├── finding → [LightGBM ≈ XGBoost (no significant difference)]
└── enables → [EXP-003, EXP-005]
```

### EXP-003: Ensemble XGBoost + LightGBM

```
EXP-003
├── builds-on → [EXP-001, EXP-002]
├── tests → [Simple Averaging Ensemble]
├── uses → [XGBoost, LightGBM]
├── compares-to → [XGBoost Baseline, LightGBM]
├── produces → [RMSE: 0.8537, +0.36% improvement]
├── finding → [Simple ensemble doesn't help (models too correlated)]
└── enables → [EXP-005 (Weighted Ensemble)]
```

### EXP-004: Polynomial Features

```
EXP-004
├── builds-on → [EXP-001]
├── tests → [Polynomial Features (30 → 465 features)]
├── compares-to → [XGBoost Baseline]
├── produces → [RMSE: 0.8541, +0.31% improvement]
├── finding → [More features ≠ better (curse of dimensionality)]
└── enables → [EXP-008 (Feature Selection)]
```

### EXP-005: Feature Selection (Failed)

```
EXP-005
├── builds-on → [EXP-001]
├── tests → [RFE with XGBoost]
├── compares-to → [XGBoost Baseline]
├── status → [Failed - NameError]
├── blocking → [Fix: Correct variable reference in run_autoresearch.py]
└── enables → [Retry after fix]
```

### EXP-006: Log Transform

```
EXP-006
├── builds-on → [EXP-001]
├── tests → [Log1p Transform for Targets]
├── compares-to → [XGBoost Baseline]
├── produces → [RMSE: 0.8692, -1.45% (WORSE)]
├── finding → [Log transform hurts FPL prediction]
└── implies → [FPL points not log-normal distributed]
```

### EXP-004: Attention LSTM (Planned)

```
EXP-007 (Planned)
├── builds-on → [EXP-001]
├── tests → [Position-Specific Models]
├── compares-to → [XGBoost Baseline]
├── hypothesis → [Position models capture different point distributions]
├── expected → [2-5% improvement]
└── related-to → [Domain Knowledge, Position Features]
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

### Actual vs Potential Pathways

```
Current State (EXP-001: RMSE 0.8568)
│
├── Path 1: Algorithm Choice → ❌ Dead End
│   ├── EXP-002: LightGBM → 0.8526 (+0.48%, not significant)
│   └── Finding: Algorithm choice less important than features/data
│
├── Path 2: Simple Ensemble → ❌ Dead End
│   ├── EXP-003: Average Ensemble → 0.8537 (+0.36%, not significant)
│   └── Finding: Simple averaging ineffective (models correlated)
│
├── Path 3: Feature Engineering → ❌ Dead End (Generic)
│   ├── EXP-004: Polynomial Features → 0.8541 (+0.31%, not significant)
│   ├── EXP-006: Log Transform → 0.8692 (-1.45%, WORSE)
│   └── Finding: Generic feature engineering doesn't help
│
└── Promising Future Paths
    ├── Path A: Position-Specific Models (EXP-007) → Expected 2-5%
    ├── Path B: Hyperparameter Optimization (EXP-008) → Expected 1-3%
    ├── Path C: Weighted/Learned Ensemble (EXP-009) → Expected 1-2%
    └── Path D: Domain-Specific Features → Expected 1-3%
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

EXP-002-006 Findings (Autonomous Session 1)
├── LightGBM ≈ XGBoost (no significant difference)
├── Simple ensembles don't help (models correlated)
├── Polynomial features ineffective (curse of dimensionality)
├── Log transform hurts performance
├── Baseline XGBoost is well-optimized
└── Need different strategies for improvement

→ Implies: Algorithm choice less important than data/features
→ Implies: Simple ensemble strategies exhausted
→ Implies: Need position-specific models and hyperopt
→ Implies: Focus on domain knowledge, not generic ML
```

### Open Questions

```
Validated Hypotheses (Autonomous Session 1)
├── LightGBM > XGBoost? → [NO: Same performance, EXP-002]
├── Simple Ensemble Helps? → [NO: +0.36%, not significant, EXP-003]
├── Polynomial Features Help? → [NO: +0.31%, not significant, EXP-004]
├── Log Transform Helps? → [NO: -1.45%, hurts performance, EXP-006]

Remaining Unvalidated Hypotheses
├── Do Position Models Help? → [Test: EXP-007]
├── Does Hyperopt Help? → [Test: EXP-008]
├── Does Weighted Ensemble Help? → [Test: EXP-009]
├── Does Attention Help? → [Test: EXP-010 (LSTM)]
├── Which Features Matter? → [Test: EXP-011 (Feature Importance)]
├── Can We Automate? → [YES: This session proved it!]
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
