# FPL Dashboard - Comprehensive Design Document

> **Purpose:** Real-time visualization and control center for FPL model development and team optimization

---

## Overview

### Vision
A single-pane-of-glass dashboard that provides:
- Real-time model performance monitoring
- Interactive team building and optimization
- Research experiment tracking
- Live FPL data integration

### Target Users
1. **Researchers** - Track experiments, compare models
2. **FPL Managers** - Build optimal teams, make transfers
3. **Stakeholders** - View high-level metrics and ROI

---

## Architecture

### Tech Stack
```
Frontend:     Streamlit (Python-based, rapid development)
Visualization: Plotly (interactive charts)
Data:         Pandas + Parquet (efficient storage)
Auto-refresh: APScheduler (background updates)
Deployment:   Streamlit Cloud / Local / Docker
```

---

## Page Structure

### 1. 🏠 Overview (Home)
- Champion Model Card (EXP-031 stats)
- Live FPL Status (Current GW, deadline)
- Quick Stats Row
- Recent Activity Feed

### 2. 🔬 Model Lab
- Model Comparison Table
- Performance Over Time Chart
- Feature Importance
- Prediction Playground

### 3. 🏃 Team Builder
- Current Team View
- Transfer Suggestions
- Optimization Engine
- Player Explorer

### 4. 🔍 Research
- Agent Status Board
- Experiment Results Table
- Ralph Loop Progress

### 5. 📊 Analytics
- Dataset Statistics
- Points Distribution
- Form Analysis

### 6. ⚙️ Settings
- Team ID configuration
- Refresh rate
- Theme settings

---

## Implementation Phases

### Phase 1: Core Dashboard (MVP) - 1 day
- [ ] Overview page
- [ ] Model Lab basic comparison
- [ ] Manual refresh only

### Phase 2: Live Integration - 1 day
- [ ] FPL API integration
- [ ] Auto-refresh
- [ ] Real team loading

### Phase 3: Research Integration - 1 day
- [ ] Agent status monitoring
- [ ] Experiment results

### Phase 4: Polish & Deploy - 1 day
- [ ] Charts and visualizations
- [ ] Mobile responsiveness
- [ ] Deploy to Streamlit Cloud

---

*See full implementation in dashboard/app.py*
