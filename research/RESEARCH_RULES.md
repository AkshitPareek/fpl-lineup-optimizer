# AutoFPL Research Rules

> **The "Constitution" for AutoFPL Research**  
> **Version:** 1.0  
> **Last Updated:** 2026-03-09  
> **Applies To:** All contributors (human and AI agents)

---

## 🎯 Preamble

This document establishes the non-negotiable standards for the AutoFPL research project. Our goal is to produce publication-quality research targeting top-tier ML venues (NeurIPS, KDD, ICML).

**Core Principle:** *"If it's not documented, it didn't happen."*

---

## 📜 The Rules

### Rule 1: Everything Must Be Documented

**Statement:** Every experiment, result, decision, and insight must be documented in the research folder.

**Requirements:**
- [ ] Every experiment has a README following the [template](./03-experiments/TEMPLATE.md)
- [ ] Every concept is defined in the [knowledge graph](./knowledge-graph/concepts.md)
- [ ] Every decision is recorded in the decision log
- [ ] Every result is reproducible from documentation alone

**Consequences of Violation:**
- Work cannot be merged
- Results are not trusted
- Cannot be cited in paper

**Verification:**
```bash
# Before claiming completion, ask:
1. Can someone reproduce this from README alone?
2. Is the knowledge graph updated?
3. Are all sections of template filled?
```

---

### Rule 2: All Improvements Must Be Statistically Validated

**Statement:** No model improvement can be claimed without statistical significance testing (p < 0.05).

**Requirements:**
- [ ] A/B test against baseline
- [ ] Report p-values
- [ ] Report effect size (Cohen's d)
- [ ] Report confidence intervals
- [ ] Correction for multiple comparisons if applicable

**Prohibited:**
- Claiming improvement based on single metric point estimate
- Ignoring p-values
- Cherry-picking best results

**Verification:**
```bash
python backend/scripts/ab_testing_cli.py compare \
    --model-a baseline \
    --model-b your_model

# Must show: p < 0.05 and meaningful effect size
```

---

### Rule 3: Full Validation Suite Must Pass

**Statement:** All 5 layers of validation must pass before any work is considered complete.

**The 5 Layers:**
1. ✅ Unit tests pass
2. ✅ Data validation passes
3. ✅ Model validation passes
4. ✅ Regression tests pass
5. ✅ Benchmark/A/B testing passes

**Command:**
```bash
python backend/scripts/validation_runner.py --all
# Must show: "All validations passed!"
```

**No Exceptions:**
- Even "small" changes require full validation
- Even "obvious" improvements require testing
- Even "documentation only" changes require checks

---

### Rule 4: Baselines Must Not Regress

**Statement:** Any new model must not degrade existing baseline performance by > 5%.

**Metrics to Monitor:**
- RMSE: Must not increase by > 5%
- MAE: Must not increase by > 5%
- R²: Must not decrease significantly
- Inference time: Must not increase by > 50%

**If Regression Detected:**
1. Investigate cause
2. Fix or document justification
3. If justified, update baseline with explanation

**Command:**
```bash
pytest backend/tests/test_model_regression.py -v
```

---

### Rule 5: Experiments Must Be Reproducible

**Statement:** Any experiment must be reproducible by another agent from documentation alone.

**Requirements:**
- [ ] Exact code versions (commit hash)
- [ ] Exact data versions
- [ ] Exact hyperparameters
- [ ] Random seeds specified
- [ ] Environment documented

**Test:**
```bash
# Can someone else run:
git checkout [commit-hash]
python backend/scripts/reproduce_experiment.py --exp EXP-XXX
# And get same results?
```

---

### Rule 6: Knowledge Graph Must Stay Current

**Statement:** The knowledge graph is the single source of truth for concepts and relationships.

**Requirements:**
- [ ] New concepts added to [concepts.md](./knowledge-graph/concepts.md)
- [ ] Relationships updated in [relationships.md](./knowledge-graph/relationships.md)
- [ ] Index updated in [index.md](./knowledge-graph/index.md)
- [ ] Experiments linked to concepts

**When to Update:**
- After every new experiment
- After discovering new concept
- After changing methodology

---

### Rule 7: Negative Results Must Be Documented

**Statement:** Failed experiments are as valuable as successful ones and must be documented.

**Requirements:**
- [ ] Failed experiments documented fully
- [ ] Hypothesis and why it failed explained
- [ ] Lessons learned recorded
- [ ] Prevents others from repeating mistake

**Template for Failed Experiments:**
```markdown
## Status: ❌ Failed

### What Was Tried
[Description]

### Why It Failed
[Analysis]

### Lessons Learned
[Insights]

### What to Try Instead
[Future direction]
```

---

### Rule 8: Code Must Be Production Quality

**Statement:** All code must meet production standards, even "research code."

**Requirements:**
- [ ] Type hints on all functions
- [ ] Docstrings with Args/Returns/Raises
- [ ] Unit tests for new functions
- [ ] Error handling
- [ ] No hard-coded paths
- [ ] Logging, not print statements

**Style Guide:**
- PEP 8 compliance
- Max line length: 100
- Meaningful variable names
- Comments explain "why," not "what"

---

### Rule 9: Main README Must Stay Current

**Statement:** The main README is the project's face and must accurately reflect current status.

**Requirements:**
- [ ] Status dashboard updated
- [ ] Experiment table current
- [ ] Progress percentage accurate
- [ ] Next actions clear

**Update After:**
- Every completed experiment
- Every status change
- Weekly at minimum

---

### Rule 10: Paper Quality Writing Standard

**Statement:** All documentation must be written to publication standards.

**Requirements:**
- [ ] Clear, precise language
- [ ] No undefined jargon
- [ ] All claims supported by evidence
- [ ] Proper citations
- [ ] Figures with captions
- [ ] Tables formatted correctly

**Quality Check:**
```markdown
Would this paragraph be acceptable in a NeurIPS paper?
If not, rewrite it.
```

---

## 🔍 Enforcement

### Pre-Commit Checklist

Before any commit, verify:

```markdown
## Commit Checklist

- [ ] Rule 1: Documentation complete?
- [ ] Rule 2: Statistical validation done?
- [ ] Rule 3: All validation layers pass?
- [ ] Rule 4: No regressions?
- [ ] Rule 5: Reproducible?
- [ ] Rule 6: Knowledge graph updated?
- [ ] Rule 7: Negative results documented (if applicable)?
- [ ] Rule 8: Code quality high?
- [ ] Rule 9: README updated?
- [ ] Rule 10: Writing quality high?
```

### Review Process

**Self-Review:**
- Author verifies all rules
- Runs validation suite
- Updates documentation

**Peer Review (for major changes):**
- Another agent reviews
- Checks rule compliance
- Verifies reproducibility
- Approves or requests changes

### Consequences

| Violation | Consequence |
|-----------|-------------|
| Missing documentation | Revert and redo |
| No statistical validation | Results rejected |
| Validation fails | Fix before merge |
| Regression detected | Investigate and fix |
| Not reproducible | Document better |
| Poor code quality | Refactor required |
| Outdated README | Update immediately |

---

## 📝 Amendments

These rules can be amended by:
1. Proposing change in new experiment
2. Justifying with evidence
3. Getting approval
4. Documenting the change

**Current Version:** 1.0  
**Amendment History:** None yet

---

## 🤝 Agreement

By contributing to this project, you agree to follow these rules.

**Signature Block:**

| Role | Entity | Date | Agreement |
|------|--------|------|-----------|
| Principal Investigator | User | 2026-03-09 | ✅ Agreed |
| Research Assistant | Kimi Code CLI | 2026-03-09 | ✅ Agreed |

---

**Remember:** These rules exist to ensure research quality. They're not obstacles—they're guardrails to keep us on the path to publication-quality work.

**When in doubt:** Document more, test more, validate more. Always err on the side of rigor.
