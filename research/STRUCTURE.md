# Research Folder Structure

Complete guide to the AutoFPL research documentation structure.

```
research/
│
├── README.md                           ⭐ START HERE - Project overview
├── CONTRIBUTING.md                     📋 How to contribute (for agents)
├── STRUCTURE.md                        📁 This file - Structure guide
├── VALIDATION_QUICKSTART.md            🚀 Quick start guide
│
├── 01-background/                      📖 Why this research matters
│   ├── README.md
│   ├── problem-statement.md            ⭐ Core research question
│   ├── related-work.md                 Literature review
│   └── fpl-domain.md                   FPL-specific background
│
├── 02-methodology/                     🔬 How we approach the problem
│   ├── README.md
│   ├── validation-framework.md         ⭐ 5-layer validation (DONE)
│   ├── data-pipeline.md                Data processing
│   ├── experimental-design.md          How we design experiments
│   ├── feature-engineering.md          Feature strategies
│   ├── model-improvements.md           Planned improvements
│   ├── ensemble-strategies.md          Ensemble methods
│   └── ab-testing-framework.md         Statistical testing
│
├── 03-experiments/                     🧪 What we've tried
│   ├── README.md
│   ├── TEMPLATE.md                     📋 Template for new experiments
│   └── 2026-03-09-baseline-establishment/  ⭐ EXP-001 (DONE)
│       └── README.md
│
├── 04-results/                         📊 What we found
│   ├── README.md
│   ├── benchmarks.md
│   ├── ab-tests.md
│   └── performance-tracking.md
│
├── 05-paper/                           📝 Publication artifacts
│   ├── README.md
│   ├── outline.md
│   └── sections/
│       ├── abstract.md
│       ├── introduction.md
│       ├── methodology.md
│       ├── experiments.md
│       ├── results.md
│       └── conclusion.md
│
├── 06-artifacts/                       💾 Models, data, configs
│   ├── models/
│   ├── datasets/
│   └── configs/
│
└── knowledge-graph/                    🕸️ Concept navigation
    ├── README.md
    ├── index.md                        ⭐ Knowledge graph entry
    ├── concepts.md                     Concept definitions
    └── relationships.md                How concepts connect
```

## Quick Navigation

### For First-Time Visitors
1. Read [README.md](./README.md)
2. Read [Problem Statement](./01-background/problem-statement.md)
3. Check [Current Status](./README.md#-quick-status-dashboard)
4. Read [Contributing Guide](./CONTRIBUTING.md)

### For Running Experiments
1. Check [Experiment Template](./03-experiments/TEMPLATE.md)
2. Review [Validation Framework](./02-methodology/validation-framework.md)
3. Look at [Example Experiment](./03-experiments/2026-03-09-baseline-establishment/)
4. Update [Knowledge Graph](./knowledge-graph/)

### For Understanding Concepts
1. Browse [Knowledge Graph Index](./knowledge-graph/index.md)
2. Read [Concepts](./knowledge-graph/concepts.md)
3. Explore [Relationships](./knowledge-graph/relationships.md)

## File Naming Conventions

### Experiments
```
YYYY-MM-DD-experiment-name/
└── README.md
```

Examples:
- `2026-03-09-baseline-establishment/`
- `2026-03-15-lstm-attention-v1/`
- `2026-03-20-feature-engineering-v2/`

### Documentation
```
lowercase-with-hyphens.md
```

Examples:
- `validation-framework.md`
- `model-improvements.md`
- `experimental-design.md`

## Status Icons

| Icon | Meaning |
|------|---------|
| ⭐ | Critical/Start here |
| 📋 | Template/Guide |
| ✅ | Complete |
| 🟡 | In Progress |
| 🔄 | Needs Revision |
| ❌ | Failed/Deprecated |

## Maintenance

**Last Updated:** 2026-03-09  
**Maintained by:** AutoFPL Research Team  
**Update Frequency:** After each experiment
