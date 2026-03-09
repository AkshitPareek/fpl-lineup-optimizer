# Problem Statement: Rigorous Validation for Stochastic Prediction

> **Research Question:** How do we rigorously validate machine learning model improvements in high-noise, limited-data prediction domains?

---

## 1. The Challenge

### 1.1 Domain Context: Fantasy Premier League

Fantasy Premier League (FPL) is a game where 10+ million players assemble virtual football squads and score points based on real-world player performance. Success requires predicting how many points each player will score in upcoming gameweeks.

**Key Characteristics:**
- **Prediction Target:** Player points (continuous, bounded roughly [-5, 20])
- **Time Horizon:** Weekly predictions (38 gameweeks per season)
- **Data Volume:** ~500 players × 38 weeks = ~19K predictions per season
- **Available History:** 3-5 reliable seasons = ~100K training samples
- **Noise Level:** Very high (football is inherently stochastic)

### 1.2 The Core Problem

When improving FPL prediction models, we face a fundamental challenge:

> **How do we know if a model improvement is real or just noise?**

Consider this scenario:
- Model A (baseline): RMSE = 2.50
- Model B (new): RMSE = 2.45
- Difference: 0.05 points (2% improvement)

**Is Model B actually better?** Or did we just get lucky with the test set?

### 1.3 Why This Is Hard

**Factor 1: High Noise**
- Football results are highly variable
- Same player, similar situation → can score 15 points or 2 points
- Model cannot predict what it cannot know (injuries, red cards, luck)

**Factor 2: Limited Data**
- Only ~100K samples across all seasons
- Can't just "get more data"
- Risk of overfitting is high

**Factor 3: Distribution Shift**
- Football evolves (tactics, rules, player roles)
- 2022 patterns may not apply to 2024
- Need to validate across time

**Factor 4: Multiple Comparisons**
- Try 20 improvements → expect 1 false positive (p < 0.05)
- Easy to fool yourself into thinking something works
- Need correction for multiple testing

### 1.4 Consequences of Getting It Wrong

If we accept spurious improvements:
- **Wasted compute:** Training models that don't actually help
- **Overfitting:** Models that fail in production
- **Wrong decisions:** Poor FPL choices based on bad predictions
- **Research validity:** Unreproducible results

---

## 2. Current State of Practice

### 2.1 Typical Approach (Insufficient)

Most FPL prediction projects follow this pattern:

```
1. Train model A (baseline)
2. Train model B (improvement)
3. Compare RMSE on test set
4. If RMSE_B < RMSE_A → declare victory
```

**Why This Fails:**
- No statistical testing (could be random)
- Single metric (misses important aspects)
- No control for overfitting
- No documentation of attempts

### 2.2 Academic ML Approach (Impractical)

Academic papers typically:
- Use large datasets (millions of samples)
- Have fixed benchmarks
- Can afford extensive hyperparameter tuning
- Focus on aggregate metrics

**Why This Doesn't Transfer:**
- FPL data is tiny by ML standards
- No standard benchmark dataset
- Domain-specific metrics matter (ranking > exact prediction)
- Need production-ready reliability

### 2.3 The Gap

We need a validation methodology that is:
- **Rigorous:** Statistical significance testing
- **Comprehensive:** Multiple metrics, multiple validations
- **Practical:** Works with limited data
- **Reproducible:** Full documentation, version control
- **Automated:** Can run overnight without human oversight

---

## 3. Research Objectives

### 3.1 Primary Objective

Develop and validate a systematic methodology for rigorously evaluating machine learning model improvements in high-noise, limited-data domains, using FPL as a case study.

### 3.2 Specific Goals

**Goal 1: Validation Framework**
Create a 5-layer validation system:
1. Data integrity checks
2. Model behavior validation
3. Statistical A/B testing
4. Regression testing
5. Benchmarking

**Goal 2: Statistical Rigor**
- Implement proper significance testing
- Control for multiple comparisons
- Report effect sizes, not just p-values
- Use confidence intervals

**Goal 3: Domain-Specific Metrics**
- Beyond RMSE: ranking metrics, top-k accuracy
- Captain pick accuracy
- Full season backtesting

**Goal 4: Automation**
- One-command validation
- Automated experiment tracking
- HTML report generation

### 3.3 Success Criteria

We will consider this research successful if:

1. **Methodology Validation:**
   - Framework catches at least 90% of spurious improvements (simulation)
   - Framework correctly identifies validated improvements

2. **Practical Application:**
   - Successfully validates 5+ model improvements
   - Achieves measurable FPL performance gains
   - Runs autonomously overnight

3. **Research Contribution:**
   - Novel methodology for small-data validation
   - Open-source implementation
   - Publication in top-tier venue

---

## 4. Research Questions

### RQ1: Primary
**How can we design a validation framework that reliably distinguishes true model improvements from noise in high-variance prediction tasks?**

**Sub-questions:**
- What statistical tests are appropriate for small samples?
- How do we control the false discovery rate?
- What sample size is needed for reliable comparison?

### RQ2: Metrics
**What evaluation metrics best capture FPL prediction quality?**

**Sub-questions:**
- Is RMSE sufficient, or do we need ranking metrics?
- How important is top-k accuracy vs overall RMSE?
- Can we optimize for captain picks specifically?

### RQ3: Automation
**To what extent can model improvement validation be automated?**

**Sub-questions:**
- What decisions require human judgment?
- Can agents propose and test improvements autonomously?
- How do we prevent automation bias?

### RQ4: Generalization
**Does this methodology transfer to other sports/domains?**

**Sub-questions:**
- Other fantasy sports (NFL, NBA)?
- Other high-noise prediction tasks?
- What are domain-specific requirements?

---

## 5. Approach Overview

### 5.1 Philosophy

> **"Trust but verify"** - Every improvement claim must be statistically validated.

### 5.2 Methodology

**Phase 1: Foundation** (Current)
- Build validation framework
- Establish baselines
- Create experiment infrastructure

**Phase 2: Systematic Improvement** (Next)
- Propose improvements based on literature/domain knowledge
- Test each with full validation
- Document results rigorously

**Phase 3: Analysis** (Future)
- Analyze what types of improvements work
- Characterize the validation framework's effectiveness
- Generalize findings

### 5.3 Key Principles

1. **Statistical Rigor First:**
   - No claim without significance test
   - Report confidence intervals
   - Control for multiple comparisons

2. **Comprehensive Evaluation:**
   - Multiple metrics (RMSE, MAE, rank correlation)
   - Multiple validations (data, model, regression)
   - Domain-specific metrics (top-k, captain accuracy)

3. **Full Documentation:**
   - Every experiment documented
   - Reproducible from README
   - Knowledge graph for discoverability

4. **Incremental Progress:**
   - Small, testable improvements
   - Fast feedback loops
   - Build on validated findings

---

## 6. Significance

### 6.1 Academic Contribution

**Novelty:**
- First systematic study of validation for sports prediction
- Methodology for high-noise, small-data domains
- Automated A/B testing for ML models

**Impact:**
- Guidelines for sports analytics research
- Validation framework reusable across domains
- Statistical best practices for small data

### 6.2 Practical Impact

**For FPL Players:**
- Better predictions → better decisions
- More reliable models
- Transparent validation

**For ML Practitioners:**
- Reusable validation framework
- Best practices for noisy domains
- Automated experimentation tools

**For Researchers:**
- Methodology paper
- Open-source tools
- Community standards

---

## 7. Related Work

### 7.1 Sports Prediction

**Existing FPL Research:**
- Matthews (2012): Early FPL prediction using linear models
- Various Kaggle competitions: Focus on accuracy, not validation
- Recent deep learning approaches: Limited statistical rigor

**Gap:** None systematically address validation methodology.

### 7.2 AutoML & Architecture Search

**AutoML:**
- Hyperparameter optimization (e.g., Hyperopt, Optuna)
- Neural Architecture Search (NAS)
- Focus: Finding best model, not validating improvements

**AutoResearch:**
- Karpathy (2026): AutoResearch for LLMs
- Focus: Autonomous experimentation
- Gap: Not adapted for small data/high noise

### 7.3 Statistical Methodology

**Multiple Comparisons:**
- Bonferroni correction
- False Discovery Rate (FDR)
- Benjamini-Hochberg procedure

**Small Sample Statistics:**
- Bootstrap methods
- Permutation tests
- Bayesian alternatives

**Our Contribution:**
Adapt and combine these for FPL-specific challenges.

---

## 8. Conclusion

### 8.1 Summary

This research addresses a fundamental challenge in applied machine learning: **how to validate improvements when ground truth is noisy and data is scarce.**

We propose:
1. A 5-layer validation framework
2. Statistical rigor appropriate for small data
3. Domain-specific evaluation metrics
4. Full automation and documentation

### 8.2 Expected Outcomes

**Immediate:**
- Validated improvements to FPL prediction
- Open-source validation tools
- Research paper

**Long-term:**
- Methodology applicable to other domains
- Community standards for sports analytics
- Foundation for autonomous sports ML research

---

**Next:** Read about our [Validation Framework](../02-methodology/validation-framework.md) implementation.
