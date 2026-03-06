# FPL Lineup Optimizer Architecture and Design

## 1. System Overview
- **Goal:** Provide advanced Fantasy Premier League lineup recommendations (single-gameweek, multi-period planning, backtests, and analytics) while keeping the UI fast and compatible with rule-based fallbacks. All optimization, predictions, and scientific logic live in `backend/`; `frontend/` consumes the REST surface provided by `backend/main.py`.
- **Layers:** Data ingestion → Prediction services (rule-based + ML) → Optimization engines (single, multi-period, robust) → API endpoints → Frontend and tooling consumers.

## 2. Data Acquisition & Context
- `backend/fpl_service.py` fetches `bootstrap-static`, fixtures, managers' teams/chips, and caches each endpoint for ~1 hour to avoid rate limits.
- `backend/historical_data_service.py` keeps a local cache under `backend/data/cache/all_player_history.json` of every player's element-summary, enabling matchup reconstruction for backtests, ML feature extraction, and analytics (see the multi-period ML path in `docs/team/NEW_AGENT_REAL_PLAYERS_PLAYBOOK.md`).
- Fixtures and team metadata power `backend/fixture_analyzer.py`, which exposes average FDR, DGW/BGW detection, fixture strings, and tickers used across optimizers, chip advisors, and the frontend ticker components.
- Optional enrichments tie in `understat_service.py` and `minutes_predictor.py` during training and live ML inference (see `backend/ml_training.py` and `backend/ml_predictor.py`).

## 3. Prediction Stack
- **Rule-based:** `backend/point_predictor.py` builds expected points using xG/xA proxies, ICT index, CBIT/CBIRT bonuses, fixture strength, clean-sheet models, and Understat-derived adjustments. It powers the default `PointPredictor` used everywhere if ML is disabled.
- **ML-based:** `backend/ml_predictor.py` lazily loads artifacts from `backend/models/current/` (metadata, scaler, XGBoost, and Keras model). It mirrors training feature engineering, blends XGB + NN outputs, and surfaces confidence scores plus baseline fallbacks. Batch predictions feed into multi-period planning or ad-hoc ML smoke tests.
- **ML integration:** `backend/main.py` conditionally builds ML context (players, fixtures, history) via `HistoricalDataService`, runs `MLPredictor.predict_batch`, and blends per-GW values with rule-based predictions using `ml_blend_weight` from `MultiPeriodRequest`. Predictions with `confidence == 30` flag fallback cases (see doc warning in `docs/team/NEW_AGENT_REAL_PLAYERS_PLAYBOOK.md`).

## 4. Optimization Engines
- **Single-gameweek (`backend/optimizer.py`):** Sets up a pulp LP with decision variables for 15-player squads and 11 starters, enforces budget, positional, and per-team caps, and optimizes a weighted objective (0.9 × starting expected points + 0.1 × bench). Strategies such as `safe`, `differential`, `transfers`, and `my_squad` tweak expected points boost logic and transfer penalties.
- **Multi-period rolling horizon (`backend/advanced_optimizer.py`):** Plans over 3–8 GWs with transfer banking (max 5 FTs), squad continuity, captain/vice choices, and integrated chip logic (Wildcard, Free Hit, Triple Captain, Bench Boost). Objective blends starting lineup points, bench value, and transfer hits. `MultiPeriodFPLOptimizer` consumes predictions (rule-only or blended) and surface metrics like `transfer_summary`, `gameweek_plans`, and `chip_recommendations`.
- **Robust optimizer (`backend/robust_optimizer.py`):** Applies box uncertainty (σ_j derived from predicted XP and minutes) to maximize worst-case points via γ-protection. It mirrors the single GW structure but penalizes uncertain players and is selectable through `robust` and `uncertainty_budget` flags.
- **Transfer & chip explainers:** `backend/transfer_explainer.py` walks through proposed transfers per GW, while `backend/chip_advisor.py` + `FixtureAnalyzer` focus on chip timing heuristics. Their outputs feed both API responses and multi-period explanations.

## 5. Analytics & Supporting Modules
- **EV / Ownership / Live analytics:** `backend/ev_calculator.py`, `backend/ownership_tracker.py`, and `backend/live_*` modules compute expected value buckets, ownership differentials, live price movers, and hot streaks; `backend/main.py` exposes them under `/api/ev/*`, `/api/ownership/*`, and `/api/live/*` for dashboard components in `frontend/src/components/analytics`.
- **Fixture analyzer:** shared between chip advisor, optimizer, and analytics to keep fixture-aware reasoning consistent across UI cards and backend decisions.
- **Backtest engine (`backend/backtest_engine.py`):** Replays past GWs using reconstructed player states, re-runs `MultiPeriodFPLOptimizer`, and collects actual vs predicted outcomes for regression: this engine drives `/api/backtest` and `/api/backtest/stream`.

## 6. API & Control Surface (`backend/main.py`)
- **Core endpoints:** `/api/data`, `/api/manager/{id}`, `/api/optimize`, `/api/optimize/multi-period`, `/api/optimize/compare`, `/api/optimize/robust`, `/api/predictions`, `/api/chip-suggestions`, `/api/backtest`, `/api/dream-team`, plus analytics/ownership/EV/live helper routes (see `backend/main.py` for full list of 40+ handlers).
- **Multi-period flow:** Fetches static data via `FPLService`, computes `current_gw`, optionally loads current squad/budget, optionally runs ML blending, instantiates `MultiPeriodFPLOptimizer`, explains transfers, and enriches results with EV/ownership analytics.
- **Chip & recommendation flows:** `/api/chip-suggestions` builds recommendations using the real manager squad (if provided) and `ChipAdvisor`; `/api/chip-recommendations` supports frontend controls.
- **Backtest & streaming:** `/api/backtest` orchestrates `BacktestEngine.run_backtest`, while `/api/backtest/stream` sends SSE updates to the UI.

## 7. Frontend Consumption (`frontend/`)
- `frontend/src/App.jsx` is the orchestrator: it defaults to multi-period mode, toggles between single/multi/backtest/dream views, hits `/api/optimize/compare`, `/api/chip-recommendations`, `/api/manager-chips`, `/api/backtest/stream`, and renders results via `Pitch` and analytics components.
- Analytics dashboards (`frontend/src/components/analytics/*`) consume EV/ownership/live endpoints. Chip state and manager IDs flow from UI controls into API payloads.
- **ML gap:** There is currently no UI to toggle `use_ml_predictions` or surface ML confidence scores; the frontend still relies on `/api/optimize/compare` and rule-based defaults. A future UI change must add new controls/endpoints (or extend existing ones) if ML predictions should affect user-facing recommendations.

## 8. ML Training & Artifact Lifecycle
- **Training pipeline (`backend/ml_training.py`):** Pulls bootstrap/static data, caches history, extracts per-player/GW features (form, XG/A, minutes, fixtures), trains XGBoost + Keras models with time-series splits, and serializes scalers, metadata, and ensemble weights into `backend/models/<version>` plus `metadata.json`.
- **Model contract:** `backend/MLPredictor` expects `models/current/` to contain `metadata.json`, `scaler.pkl`, `xgb/model.pkl` + `feature_columns.csv`, and `nn/model.keras` (see `docs/plans/ML-INTEGRATION-SPEC.md` for the full contract and `docs/plans/ML-MODELS-DESIGN.md` for feature/training details).
- **Datasets:** `datasets/fpl_points_v1/` contains train/validation/test splits (`*_X.npy`, `*_y.npy`, raw CSV exports) used for offline experimentation and the regression tests powering analytics.

## 9. Operations & Observability
- **Caching & resilience:** `FPLService` reduces API pressure; `HistoricalDataService` saves a weeks-long cache and can reconstruct past GWs for backtests/ML inference. Both print logs for failure detection.
- **Logging:** FastAPI’s startup events log origins; multi-period requests log ML blend weights, `force_next_gameweek` overrides, and fallback detection; any exception dumps stack traces for debugging.
- **Backtests & test harnesses:** `backend/reproduce_issue.py`, `test_smart_bench.py`, and `backend/tests/` help validate the optimizer, while `stateful SSE` streaming ensures progress is visible on the frontend.

## 10. References & Next Steps
- **Docs:** `docs/plans/INTEGRATION-DESIGN.md`, `docs/plans/ML-INTEGRATION-SPEC.md`, and `docs/plans/ML-MODELS-DESIGN.md` capture deeper decisions about optional ML usage, artifact contracts, and training architecture.
- **Runbook:** `docs/team/NEW_AGENT_REAL_PLAYERS_PLAYBOOK.md` shows how to run the ML path manually; follow that when verifying new artifacts.
- **Next additions:** surface ML controls on the frontend, expand chip suggestions to cover new chips, and add a dedicated `/api/optimize/ml-path` endpoint once artifacts stabilize.
