# Research Concepts

> Comprehensive glossary of concepts in AutoFPL research  
> Organized by category  
> Updated: 2026-03-09

---

## Domain Concepts (FPL-Specific)

### Expected Points (xP)

**Type:** Domain Concept  
**Status:** ✅ Well-Established  
**Also Known As:** xPts, Predicted Points

#### Definition
The predicted number of points a player will score in an upcoming gameweek. This is the primary prediction target of our models.

#### Why It Matters
- Core output of our prediction system
- Used for squad selection, captain picks, transfers
- Differs from bookmaker odds (which are optimized for betting, not fantasy)

#### Calculation
```
xP = f(player_features, fixture_features, team_features)
```

Where features include:
- Recent form (rolling averages)
- Fixture difficulty
- Team strength
- Player role/position
- Injury/availability

#### Related Concepts
- [Gameweek](#gameweek)
- [Form](#player-form)
- [Fixture Difficulty](#fixture-difficulty)

#### References
- FPL Official: [How are bonus points calculated](https://www.premierleague.com/news/2174909)
- Community: [Expected Goals Philosophy](https://fbref.com/en/expected-goals-model-explained/)

---

### Gameweek

**Type:** Domain Concept  
**Status:** ✅ Well-Established  
**Abbreviation:** GW

#### Definition
A weekly round of Premier League matches. FPL season consists of 38 gameweeks (typically August to May).

#### Why It Matters
- Primary time unit for predictions
- Squad changes (transfers) made between gameweeks
- Points calculated per gameweek

#### Structure
- Each GW: 10 matches (typically)
- Deadline: 90 minutes before first match
- Double GW: Team plays twice (rare)
- Blank GW: Team doesn't play (rare)

#### Related Concepts
- [Expected Points](#expected-points)
- [Fixture Difficulty](#fixture-difficulty)

---

### Fixture Difficulty Rating (FDR)

**Type:** Domain Concept  
**Status:** ✅ Well-Established  
**Range:** 2 (easiest) to 5 (hardest)

#### Definition
Official Premier League rating of match difficulty based on team strengths. Used to assess matchup quality.

#### Calculation
FDR considers:
- Home/away advantage
- Attacking vs defensive strength
- Historical performance

#### Why It Matters
- Strong predictor of player performance
- Used in feature engineering
- Key input to fixture analysis

#### Example
```
Manchester City (Home) vs Luton Town = FDR 2 (Easy)
Luton Town (Away) vs Manchester City = FDR 5 (Hard)
```

#### Related Concepts
- [Expected Points](#expected-points)
- [Form](#player-form)

---

### Player Form

**Type:** Domain Concept  
**Status:** ✅ Well-Established

#### Definition
Recent performance trend of a player, typically measured over last 3-6 gameweeks.

#### Calculation Methods
1. **Simple Average:** Mean points over last N games
2. **Weighted Average:** Recent games weighted more heavily
3. **Exponential Moving Average:** Smooth decay weighting

```python
# Exponential weighted form
weights = exp(linspace(-1, 0, n_games))
form = sum(points * weights) / sum(weights)
```

#### Why It Matters
- Strongest predictor of future performance
- Captures momentum, confidence, tactical role
- Different calculation methods produce different signals

#### Limitations
- Small sample size (high variance)
- Ignores fixture difficulty
- Doesn't capture role changes

#### Related Concepts
- [Expected Points](#expected-points)
- [Gameweek](#gameweek)

---

### Captain

**Type:** Domain Concept  
**Status:** ✅ Well-Established

#### Definition
One player per gameweek whose points are doubled. The most important decision in FPL.

#### Why It Matters
- Single biggest impact on weekly score
- Captain choice worth ~20-25% of total points
- Makes top-k accuracy crucial (need to identify the very best player)

#### Captain Metrics
- **Captain Accuracy:** % of weeks where predicted captain = actual top scorer
- **Vice-Captain:** Backup captain if captain doesn't play

#### Related Concepts
- [Top-k Accuracy](#top-k-accuracy)
- [Expected Points](#expected-points)

---

## Methodology Concepts

### Validation Framework

**Type:** Methodology  
**Status:** ✅ Implemented & Validated  
**Introduced:** EXP-001 (2026-03-09)

#### Definition
A 5-layer testing system to ensure model improvements are real and not artifacts.

#### The 5 Layers

| Layer | Purpose | Tests |
|-------|---------|-------|
| **1. Unit Tests** | Component functionality | API endpoints, data pipelines |
| **2. Data Validation** | Data quality | No NaN, consistent shapes, no leakage |
| **3. Model Validation** | Model behavior | Reasonable predictions, edge cases |
| **4. Regression Tests** | Prevent degradation | Performance vs baseline |
| **5. Benchmark Tests** | Compare models | A/B testing, statistical significance |

#### Why It Matters
- Distinguishes real improvements from noise
- Prevents overfitting to validation set
- Enables rigorous research

#### Implementation
```bash
python backend/scripts/validation_runner.py --all
```

#### References
- [Testing Framework Docs](../../docs/TESTING_FRAMEWORK.md)
- Experiment: [EXP-001](../03-experiments/2026-03-09-baseline-establishment/)

---

### A/B Testing

**Type:** Methodology  
**Status:** ✅ Implemented & Validated  
**Also Known As:** Controlled Experiment

#### Definition
Statistical comparison of two model variants (A: control/baseline, B: treatment) to determine if B is significantly better.

#### Key Elements
1. **Null Hypothesis:** No difference between A and B
2. **Alternative Hypothesis:** B is better than A
3. **Significance Level:** α = 0.05 (5% false positive rate)
4. **Effect Size:** Cohen's d (practical significance)

#### Statistical Tests Used
- **Paired t-test:** Compare error distributions
- **Wilcoxon signed-rank:** Non-parametric alternative
- **Bootstrap CI:** Confidence intervals for differences

#### Output
```
Model A MAE: 2.456
Model B MAE: 2.345
Difference: -0.111
95% CI: [-0.198, -0.024]
p-value: 0.012
Significant: ✅ Yes
Effect Size: 0.42 (medium)
```

#### Implementation
```bash
python backend/scripts/ab_testing_cli.py compare \
    --model-a baseline \
    --model-b treatment
```

#### References
- [A/B Testing Guide](../../docs/TESTING_FRAMEWORK.md#ab-testing-models)
- [Statistical Significance](#statistical-significance)

---

### Statistical Significance

**Type:** Methodology  
**Status:** ✅ Well-Established

#### Definition
The probability that an observed difference did not occur by random chance.

#### Key Concepts

**p-value:**
- Probability of seeing results this extreme if null hypothesis were true
- p < 0.05: Typically considered "significant"
- p < 0.01: "Highly significant"
- p ≥ 0.05: "Not significant"

**Confidence Interval (CI):**
- Range where true effect likely lies
- 95% CI: We're 95% confident true effect is in this range
- If CI includes 0: Not significant

#### Multiple Comparisons Problem
- Testing many hypotheses → false positives
- **Solution:** Bonferroni correction (α / n_tests)

#### Why It Matters
- Separates real improvements from luck
- Required for publication-quality research

---

### Effect Size

**Type:** Methodology  
**Status:** ✅ Well-Established  
**Common Measure:** Cohen's d

#### Definition
Magnitude of difference, independent of sample size.

#### Cohen's d
```
d = (mean_A - mean_B) / pooled_std_dev
```

#### Interpretation
| d | Effect Size | Meaning |
|---|-------------|---------|
| < 0.2 | Negligible | Trivial difference |
| 0.2 - 0.5 | Small | Noticeable but small |
| 0.5 - 0.8 | Medium | Important difference |
| > 0.8 | Large | Substantial difference |

#### Why It Matters
- Statistical significance ≠ Practical significance
- Small p-value can occur with tiny effect in large sample
- Effect size tells you if improvement matters

---

### Cross-Validation

**Type:** Methodology  
**Status:** ✅ Well-Established

#### Definition
Technique to assess model generalization by training on subsets and testing on held-out data.

#### Types

**K-Fold CV:**
- Split data into K folds
- Train on K-1 folds, test on remaining
- Average performance across all folds

**Time-Series CV:**
- Respects temporal order
- Train on past, test on future
- Critical for FPL (prevents data leakage)

#### Why It Matters
- Better estimate of true performance
- Reduces variance in evaluation
- Required for small datasets

---

### Baseline

**Type:** Methodology  
**Status:** ✅ Established (EXP-001)

#### Definition
Reference model against which improvements are measured.

#### Our Baselines
| Model | RMSE | MAE | Status |
|-------|------|-----|--------|
| XGBoost | 2.45 | 1.89 | ✅ Primary |
| LightGBM | 2.38 | 1.82 | ✅ Primary |
| Random Forest | 2.52 | 1.95 | ✅ Secondary |
| Ridge | 2.61 | 2.02 | ✅ Simple |
| Ensemble | 2.29 | 1.76 | ✅ Best |

#### Why It Matters
- Can't claim improvement without comparison
- Sets expectations for what's achievable
- Validates that new methods are actually better

---

## Model Concepts

### Gradient Boosting

**Type:** Algorithm  
**Status:** ✅ Validated  
**Variants:** XGBoost, LightGBM, CatBoost

#### Definition
Ensemble method that builds trees sequentially, with each tree correcting errors of previous trees.

#### Key Characteristics
- **Strengths:** High accuracy, handles mixed data types, feature importance
- **Weaknesses:** Can overfit, sensitive to hyperparameters
- **Best For:** Tabular data, feature-rich problems

#### Hyperparameters
- `n_estimators`: Number of trees
- `max_depth`: Tree depth
- `learning_rate`: Shrinkage factor
- `subsample`: Row sampling
- `colsample_bytree`: Feature sampling

#### Our Implementation
```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    # ... see configs
)
```

#### Performance
- XGBoost: RMSE = 2.45 (baseline)
- LightGBM: RMSE = 2.38 (slightly better)

---

### LSTM (Long Short-Term Memory)

**Type:** Algorithm  
**Status:** 🟡 Implemented, Testing Planned  
**Experiment:** EXP-004

#### Definition
Recurrent neural network architecture designed to capture long-term dependencies in sequential data.

#### Key Characteristics
- **Strengths:** Captures temporal patterns, memory of past events
- **Weaknesses:** Slow training, needs sequences, can be unstable
- **Best For:** Time series, sequential data

#### Architecture
```
Input → LSTM Layer(s) → Dense → Output
       ↓
   Hidden State (memory)
```

#### Our Implementation
```python
from ml.lstm_model import FPLLSTM

model = FPLLSTM(
    input_dim=121,
    hidden_dim=64,
    num_layers=2,
    dropout=0.2,
)
```

#### Planned Experiments
- EXP-004: Standard LSTM
- EXP-004b: LSTM with Attention

---

### Attention Mechanism

**Type:** Algorithm Component  
**Status:** 🟡 Implemented, Testing Planned  
**Experiment:** EXP-004

#### Definition
Mechanism that allows model to focus on relevant parts of input when making predictions.

#### Key Idea
- Assign weights to different time steps
- Higher weight = more important
- Learn which historical gameweeks matter most

#### Why for FPL?
- Recent form matters more than old form
- But some patterns need longer history
- Attention learns optimal weighting

#### Our Implementation
```python
from ml.lstm_model import FPLLSTMWithAttention

model = FPLLSTMWithAttention(
    input_dim=121,
    hidden_dim=64,
    use_attention=True,
)
```

#### Hypothesis
Attention will improve form modeling by learning optimal lookback window.

---

### Ensemble Methods

**Type:** Algorithm  
**Status:** ✅ Implemented  
**Variants:** Averaging, Weighted, Stacking

#### Definition
Combine multiple models to improve performance beyond any single model.

#### Types

**Simple Average:**
```
ensemble_pred = mean(model_1_pred, model_2_pred, ...)
```

**Weighted Average:**
```
ensemble_pred = sum(weight_i * model_i_pred)
```

**Stacking:**
```
meta_model.fit([model_1_pred, model_2_pred, ...], y_true)
```

#### Why It Works
- Reduces variance
- Combines different biases
- More robust to outliers

#### Our Implementation
```python
from ml.ensemble import EnsemblePredictor

ensemble = EnsemblePredictor(
    base_models=['xgboost', 'lightgbm', 'random_forest']
)
```

#### Performance
- Ensemble: RMSE = 2.29 (best so far)
- Improvement: ~7% over best single model

---

## Evaluation Metrics

### RMSE (Root Mean Squared Error)

**Type:** Metric  
**Status:** ✅ Primary Metric  
**Formula:** sqrt(mean((y_pred - y_true)²))

#### Properties
- Penalizes large errors more than small ones
- Same units as target (points)
- Sensitive to outliers

#### Interpretation
RMSE = 2.5 means typical prediction is off by ~2.5 points.

#### Why Primary?
- Standard in regression
- Punishes big misses (important for FPL)
- Well-understood by community

---

### MAE (Mean Absolute Error)

**Type:** Metric  
**Status:** ✅ Secondary Metric  
**Formula:** mean(|y_pred - y_true|)

#### Properties
- Linear penalty (vs quadratic for RMSE)
- More robust to outliers
- Easier to interpret

#### Interpretation
MAE = 2.0 means average prediction is off by 2.0 points.

---

### Spearman Correlation

**Type:** Metric  
**Status:** ✅ Critical for FPL  
**Range:** -1 to 1

#### Definition
Rank correlation between predicted and actual values.

#### Why Critical?
FPL is about **ranking** players, not exact point prediction:
- We need to identify the best captain
- We need to rank players for transfers
- Relative ordering matters more than absolute values

#### Interpretation
- ρ = 1.0: Perfect ranking
- ρ = 0.0: No correlation (random)
- ρ = 0.3: Weak but useful
- ρ = 0.5: Strong
- ρ = 0.7: Very strong

#### Target
Spearman ρ > 0.35 (validated baseline)

---

### Top-k Accuracy

**Type:** Metric  
**Status:** ✅ Important for FPL

#### Definition
Percentage of actual top-k players that appear in predicted top-k.

#### Variants
- **Top-5:** Captain selection
- **Top-10:** Core squad players
- **Top-20:** Differential picks

#### Example
```
Actual Top-5: [Salah, Haaland, Saka, Son, Palmer]
Predicted Top-5: [Salah, Haaland, Saka, Foden, Rashford]
Overlap: 3/5 = 60% Top-5 Accuracy
```

#### Why It Matters
- Directly measures captain pick quality
- More actionable than RMSE
- Domain-specific relevance

---

## Feature Engineering Concepts

### Feature Engineering

**Type:** Methodology  
**Status:** 🟡 Planned  
**Experiment:** EXP-002

#### Definition
Creating new features from raw data to improve model performance.

#### Types

**Polynomial Features:**
```
x², x³ (capture non-linear relationships)
```

**Interaction Features:**
```
x1 * x2 (combined effect of two features)
```

**Ratio Features:**
```
x1 / x2 (relative measures)
```

**Temporal Features:**
```
Days since last match, rest days, momentum
```

#### Planned Experiments
- EXP-002a: Polynomial features
- EXP-002b: Interaction features
- EXP-002c: Ratio features
- EXP-002d: Temporal features

---

## Adding New Concepts

When adding a new concept, use this template:

```markdown
### Concept Name

**Type:** [Algorithm/Metric/Domain/Methodology]  
**Status:** [Proposed/Implemented/Validated/Deprecated]  
**Introduced:** [Experiment ID or Date]

#### Definition
[Clear, precise definition]

#### Why It Matters
[Relevance to FPL prediction]

#### Implementation
[How it's implemented]

#### Related Concepts
- [Link to related concept]

#### References
- [External paper/resource]
```

---

**See Also:**
- [Knowledge Graph Index](./index.md)
- [Relationships](./relationships.md)
- [Experiments](../03-experiments/)
