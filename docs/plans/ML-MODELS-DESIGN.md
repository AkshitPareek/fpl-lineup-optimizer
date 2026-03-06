# ML Models Design for FPL Lineup Optimizer

**Document Version**: 1.0  
**Date**: March 4, 2026  
**Team**: FPL ML Integration  
**Target Season**: 2025/26  

---

## 1. Feature Engineering

### 1.1 Complete Feature Matrix

#### A. Player Static Features
- `player_id`: FPL element ID
- `position`: 1=GK, 2=DEF, 3=MID, 4=FWD
- `now_cost`: Current price (tenths of £m)
- `team`: Team ID
- `age`: Player age (from external data if available)
- `height`, `weight`: Physical attributes (optional)

#### B. Form Features (Last 6 GWs)
**Formula: Weighted average with exponential decay**

```python
def weighted_rolling_average(values, weights=None):
    if weights is None:
        weights = np.exp(np.linspace(-1, 0, len(values)))  # Exponential decay
    weights = weights / weights.sum()
    return np.dot(values[-len(weights):], weights[-len(values):])
```

Features:
- `form_points`: Weighted avg points (last 6 GWs)
- `form_minutes`: Weighted avg minutes
- `form_xg`: Weighted avg xG (or Understat xG if available)
- `form_xa`: Weighted avg xA (or Understat xA if available)
- `form_ict`: Weighted avg ICT index
- `form_creativity`: Weighted avg creativity
- `form_threat`: Weighted avg threat
- `goals_per_90`: Total goals / (total minutes/90) if minutes > 270
- `assists_per_90`: Total assists / (total minutes/90) if minutes > 270

#### C. Fixture Features
For each upcoming GW (typically 5 GW horizon):

```python
def compute_fixture_score(team_id, gw, team_strengths, fixture_map):
    """Score fixture difficulty on 0-1 scale (1=easiest)"""
    fixtures = fixture_map.get((team_id, gw), [])
    if not fixtures:
        return 0.5  # Neutral default
    
    total_score = 0
    for f in fixtures:
        opponent = f['opponent']
        is_home = f['is_home']
        
        # Team strength differential
        if is_home:
            attack_strength = team_strengths['attack_home'][team_id]
            defense_strength = team_strengths['defense_home'][opponent]
        else:
            attack_strength = team_strengths['attack_away'][team_id]
            defense_strength = team_strengths['defense_away'][opponent]
        
        # Expected goal environment
        xg_env = (attack_strength / 1000) * (1000 - defense_strength) / 1000
        total_score += xg_env
    
    return total_score / len(fixtures)
```

Features:
- `fixture_score_gw1` to `fixture_score_gw5`: 0-1 scale (higher = better fixture)
- `double_gw_flag`: Binary indicator if 2+ fixtures that GW
- `blank_gw_flag`: Binary indicator if no fixture

#### D. Team Context Features
- `team_attack_strength`: Team's normalized attack rating (home/away blended)
- `team_defense_strength`: Team's normalized defense rating
- `team_clean_sheet_rate`: Historical CS% for team (last 10 GWs)
- `team_avg_points`: Team's average points per match (last 10 GWs)

#### E. Advanced xG/xA Features (if Understat available)
- `understat_matched`: Binary flag
- `understat_xG_per_90`: Understat xG per 90 minutes
- `understat_xA_per_90`: Understat xA per 90 minutes
- `understat_npxG_per_90`: Non-penalty xG per 90
- `understat_shots_per_90`: Shots per 90
- `understat_key_passes_per_90`: Key passes per 90

#### F. Minutes/Rotation Features
- `recent_4_minutes_avg`: Average minutes last 4 GWs
- `start_probability`: Probability of starting (from MinutesPredictor)
- `rotation_risk`: Rotation risk score (from MinutesPredictor)
- `nailedness_score`: 0-100 nailedness score
- `injury_status`: Encoded: 0=available, 1=doubtful, 2=injured
- `availability_factor`: chance_of_playing_next_round / 100

#### G. Market & Popularity Features (if available)
- `ownership_percent`: Selected by % of top 100k managers
- ` Rising/Falling`: Trend in ownership (last 3 GWs delta)
- `price_change_odds`: Probability of price rise/fall (from external API)

#### H. Interaction Features
```python
# Polynomial interactions
features['form_points_x_fixture'] = features['form_points'] * features['fixture_score_gw1']
features['xg_per_90_x_team_attack'] = features['understat_xG_per_90'] * features['team_attack_strength']
features['minutes_x_status'] = features['recent_4_minutes_avg'] * features['availability_factor']

# Position dummies (one-hot encoded)
for pos in [1, 2, 3, 4]:
    features[f'pos_{pos}'] = (player['element_type'] == pos).astype(int)
```

### 1.2 Feature Extraction Pipeline

**Data Source**:
1. Bootstrap-static (current player/team/fixture data)
2. HistoricalDataService (per-GW history)
3. UnderstatService (xG/xA when available)
4. OwnershipTracker (market data if available)

**Processing Steps**:
```python
def extract_features(player_id: int, target_gw: int, 
                     history_service, static_data, 
                     use_understat=True) -> pd.DataFrame:
    """
    Extract complete feature set for a player in a target gameweek.
    
    Returns: DataFrame with one row per player, target column 'actual_points'
    """
    
    # 1. Get player's historical data up to target_gw-1
    player_history = history_service.get_player_history(player_id)
    past_gws = [gw for gw in player_history if gw < target_gw]
    
    # 2. Build features dictionary
    features = {'player_id': player_id, 'target_gw': target_gw}
    
    # 3. Static features (from current player record)
    player_static = get_player_static(player_id, static_data)
    features.update({
        'position': player_static['element_type'],
        'now_cost': player_static['now_cost'],
        'team': player_static['team'],
        'injury_status': encode_status(player_static['status']),
        'availability_factor': player_static.get('chance_of_playing_next_round', 100) / 100
    })
    
    # 4. Form features (last 6 GWs weighted)
    recent_history = past_gws[-6:] if len(past_gws) >= 6 else past_gws
    if recent_history:
        features.update(compute_weighted_form_features(recent_history))
    
    # 5. Season aggregates (if enough minutes)
    season_history = past_gws[-38:]  # Full season (or available)
    total_minutes = sum(h['minutes'] for h in season_history)
    if total_minutes > 270:  # ~3 full games
        features['goals_per_90'] = sum(h['goals_scored'] for h in season_history) / (total_minutes / 90)
        features['assists_per_90'] = sum(h['assists'] for h in season_history) / (total_minutes / 90)
    else:
        # Position-based baselines
        pos_baselines = {1: 0.0, 2: 0.03, 3: 0.08, 4: 0.35}
        features['goals_per_90'] = pos_baselines[features['position']]
        pos_assist_baselines = {1: 0.0, 2: 0.04, 3: 0.08, 4: 0.12}
        features['assists_per_90'] = pos_assist_baselines[features['position']]
    
    # 6. Fixture features for target GW
    fixture_info = get_fixtures_for_player(player_id, target_gw)
    features['fixture_score_gw1'] = compute_fixture_score(player_id, target_gw, team_strengths, fixture_map)
    features['double_gw_flag'] = len(fixture_info) > 1
    features['blank_gw_flag'] = len(fixture_info) == 0
    
    # 7. Team context
    team_id = player_static['team']
    team_stats = compute_team_stats(team_id, past_gws[-10:])
    features.update(team_stats)
    
    # 8. Understat features (if available and matched)
    if use_understat:
        understat_data = understat_service.get_player_data(player_id)
        if understat_data:
            features.update(understat_data)
            features['understat_matched'] = 1
        else:
            features['understat_matched'] = 0
    
    # 9. Minutes/Rotation features
    minutes_pred = minutes_predictor.get_predictions(player_id)
    features.update(minutes_pred)
    
    # 10. Target variable: actual points in target_gw
    features['actual_points'] = history_service.get_actual_score(player_id, target_gw)
    
    return features
```

### 1.3 Feature Scaling & Normalization

**Approach**: Per-feature standardization (z-score) using training set statistics

```python
from sklearn.preprocessing import StandardScaler

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save scaler for inference
import joblib
joblib.dump(scaler, 'models/scaler.pkl')
```

**Non-scaled features**:
- Categorical: `position` (one-hot), `team` (one-hot or embedding)
- Binary: `double_gw_flag`, `blank_gw_flag`, `understat_matched`
- IDs: `player_id` (stored separately, not used in training)

### 1.4 Handling Missing Data & Outliers

**Missing Data**:
- Fill with position-specific median or 0 (for "no data" cases)
- For Understat: binary indicator `understat_matched` + fill values with 0 or median
```python
# Example: xG per 90 missing → use position baseline
if pd.isna(xg):
    xg = position_baseline[position]
```

**Outliers**:
- Cap extreme values at 99th percentile during training
- Apply winsorization: `np.clip(value, lower, upper)`
- For points target: cap at 20 (max possible in extreme circumstances)

---

## 2. XGBoost Model

### 2.1 Hyperparameter Configuration

**Base Model**: `xgb.XGBRegressor`

```python
DEFAULT_XGB_PARAMS = {
    'n_estimators': 500,           # Number of boosting rounds
    'max_depth': 6,                # Maximum tree depth (7-8 might overfit)
    'learning_rate': 0.03,         # Low LR for better generalization
    'subsample': 0.8,              # Row subsample ratio
    'colsample_bytree': 0.7,       # Feature subsample ratio
    'colsample_bylevel': 0.7,      # Feature subsample per level
    'min_child_weight': 5,         # Minimum sum of instance weight in child
    'gamma': 0.1,                  # Minimum loss reduction for split
    'reg_alpha': 1.0,              # L1 regularization
    'reg_lambda': 2.0,             # L2 regularization
    'random_state': 42,
    'n_jobs': -1,                  # Use all cores
    'tree_method': 'hist',         # Fast histogram-based algorithm
    'eval_metric': ['mae', 'rmse'] # Track metrics
}
```

**Why these values**:
- `max_depth=6`: Deep enough for interactions but not overfit
- `learning_rate=0.03`: With 500 trees, provides smooth convergence
- Regularization (`reg_alpha`, `reg_lambda`): Prevents overfitting on noisy FPL data
- `subsample=0.8`: Bagging reduces variance

### 2.2 Training Procedure

```python
def train_xgb(X_train, y_train, X_val, y_val, params=None):
    """
    Train XGBoost with early stopping.
    """
    if params is None:
        params = DEFAULT_XGB_PARAMS.copy()
    
    model = xgb.XGBRegressor(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
        early_stopping_rounds=50  # Stop if val MAE doesn't improve
    )
    
    return model
```

**Cross-Validation Strategy**: Time Series Split (respect temporal ordering)

```python
from sklearn.model_selection import TimeSeriesSplit

# For hyperparameter tuning
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # Train model...
```

### 2.3 Feature Importance Analysis

**Methods**:
1. **Gain**: Average gain of splits where feature is used
2. **Cover**: Average coverage (samples) of splits where feature is used
3. **Frequency**: Number of times feature is used in splits

```python
# After training
importance = model.feature_importances_
feature_names = X_train.columns

# Plot top 20 features
import matplotlib.pyplot as plt
xgb.plot_importance(model, max_num_features=20)
plt.show()

# Get detailed importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)
```

**Expected Top Features**:
- `form_points`: Recent form is dominant predictor
- `fixture_score_gw1`: Single-gameweek fixture matters most
- `understat_xG_per_90`: Underlying quality (if available)
- `recent_4_minutes_avg`: Minutes consistency
- `team_attack_strength`: Team context
- `position` dummies: Position-specific baselines

### 2.4 Expected Performance Metrics

**Baseline**: Rule-based PointPredictor achieves R² ≈ 0.05-0.08 on single GW

**Target XGBoost Performance**:
- **MAE**: 1.8 - 2.2 points per gameweek
- **RMSE**: 2.4 - 2.8 points
- **R²**: 0.12 - 0.18 (significant improvement over baseline)
- **Median Absolute Error**: ~1.5 points

**Why achievable**:
- Historical FPL point distributions are noisy (variance high)
- Even perfect model limited by inherent randomness
- But XGBoost can capture non-linear interactions better than linear rules

### 2.5 Model Serialization

```python
import joblib
import xgboost as xgb

# Save model
joblib.dump(model, 'models/xgb_2025_v1.pkl')

# Also save in XGB's native format (more efficient)
model.save_model('models/xgb_2025_v1.json')  # or .ubj for binary

# Load in production
model = joblib.load('models/xgb_2025_v1.pkl')
# or
model = xgb.XGBRegressor()
model.load_model('models/xgb_2025_v1.json')
```

---

## 3. Neural Network Model

### 3.1 Architecture Design

**Framework**: TensorFlow/Keras (easier deployment, good ecosystem)

**Architecture**:
```python
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_nn_model(input_dim, hidden_layers=[128, 64, 32], 
                   dropout_rate=0.3, l2_reg=1e-4):
    """
    Feedforward Neural Network for point prediction.
    
    Args:
        input_dim: Number of features
        hidden_layers: List of units per layer
        dropout_rate: Dropout for regularization (0.2-0.5)
        l2_reg: L2 regularization coefficient
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Normalization layer (learn mean/std from data)
    x = layers.Normalization()(inputs)
    
    # Hidden layers with BatchNorm, Dropout, ReLU
    for units in hidden_layers:
        x = layers.Dense(
            units,
            activation='linear',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Output layer (single value, no activation for regression)
    outputs = layers.Dense(1, activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
```

**Example architecture**:
```
Input (64 features)
    ↓
Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(64) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(32) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Dense(1)  # Output: expected points
```

### 3.2 Optimizer & Loss Function

```python
def compile_model(model, learning_rate=0.001):
    """
    Compile NN with appropriate loss and optimizer.
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999
    )
    
    model.compile(
        optimizer=optimizer,
        loss='huber',  # Huber loss less sensitive to outliers than MSE
        metrics=['mae', 'mse']
    )
    
    return model
```

**Loss choice rationale**:
- **Huber loss**: δ=1.0 default. Smooth transition between MSE (small errors) and MAE (large errors). Robust to occasional huge residuals (e.g., unexpected hauls).
- Alternative: `rmse` (root MSE) directly penalizes large errors
- Alternative: `mae` (median prediction, but less gradient signal)

### 3.3 Training Configuration

```python
# Training callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_mae',
        patience=30,          # Stop if no improvement 30 epochs
        restore_best_weights=True,
        mode='min'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_mae',
        factor=0.5,           # Halve LR on plateau
        patience=10,
        min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='models/nn_checkpoint.keras',
        save_best_only=True,
        monitor='val_mae'
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir='logs/nn_training'
    )
]

# Training
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,              # Max epochs (early stopping will end earlier)
    batch_size=256,          # Large batch for stable gradients
    callbacks=callbacks,
    verbose=1
)
```

**Hyperparameters**:
- `batch_size=256`: Good balance of gradient stability and speed
- `epochs=200` with early stopping: Typically stops around 50-80 epochs
- `learning_rate=0.001`: Standard Adam default, reduces on plateau
- `dropout=0.3`: Prevents overfitting on ~10k-50k samples

### 3.4 Expected Performance

NN should match or slightly exceed XGBoost (R² 0.14-0.20) but with:
- **Pros**: Can learn complex interactions, better calibrated uncertainties with MC Dropout
- **Cons**: Slower training, more hyperparameter tuning, less interpretable

### 3.5 Model Serialization

```python
# Save entire model (architecture + weights + optimizer state)
model.save('models/nn_2025_v1.keras')  # Keras native format

# Or just weights + config (smaller)
model.save_weights('models/nn_weights.h5')
with open('models/nn_architecture.json', 'w') as f:
    f.write(model.to_json())

# Load in production
model = tf.keras.models.load_model('models/nn_2025_v1.keras')
```

---

## 4. Ensemble Strategy

### 4.1 Weighted Average Ensemble

**Simple weighted average** (weights tuned on validation):

```python
def ensemble_predict(xgb_pred, nn_pred, xgb_weight=0.6, nn_weight=0.4):
    """
    Combine predictions from XGBoost and Neural Network.
    
    Args:
        xgb_pred: XGBoost point prediction
        nn_pred: Neural Network point prediction
        xgb_weight: Weight for XGB (0-1)
        nn_weight: Weight for NN (0-1, should = 1-xgb_weight)
    
    Returns:
        Ensemble prediction
    """
    return xgb_weight * xgb_pred + nn_weight * nn_pred
```

**Dynamic weights** (per-player confidence):

```python
def dynamic_ensemble(xgb_pred, nn_pred, xgb_std, nn_std):
    """
    Inverse variance weighting: more confident model gets higher weight.
    
    Args:
        xgb_std: XGB's uncertainty estimate (e.g., from quantile regression or bootstrap)
        nn_std: NN's uncertainty (e.g., MC Dropout variance)
    
    Returns:
        Weighted prediction + combined uncertainty
    """
    # Inverse variance weights (add epsilon to avoid div by zero)
    eps = 1e-6
    xgb_var = xgb_std**2 + eps
    nn_var = nn_std**2 + eps
    
    w_xgb = (1 / xgb_var) / (1/xgb_var + 1/nn_var)
    w_nn = (1 / nn_var) / (1/xgb_var + 1/nn_var)
    
    ensemble_pred = w_xgb * xgb_pred + w_nn * nn_pred
    ensemble_std = np.sqrt(1 / (1/xgb_var + 1/nn_var))
    
    return ensemble_pred, ensemble_std
```

### 4.2 Confidence Score Calculation

**Lower-level uncertainty** (predictive standard deviation):

```python
def compute_confidence_score(prediction, uncertainty, player_id=None):
    """
    Compute a 0-100 confidence score.
    
    Formula: confidence = max(0, 100 - uncertainty_normalized * scale)
    
    Args:
        prediction: Predicted points
        uncertainty: Standard deviation estimate
        player_id: Optional, for player-specific scaling
    
    Returns:
        confidence_score: 0-100
    """
    # Uncertainty normalized by prediction magnitude
    # E.g., 5 pt prediction with 2 pt uncertainty = 40% CV
    cv = uncertainty / max(prediction, 0.1)  # Coef of variation
    
    # Convert to 0-100 scale (lower CV = higher confidence)
    # CV=0 -> 100, CV=1 -> ~50, CV=2+ -> near 0
    confidence = 100 * np.exp(-cv * 2)  # Exponential decay
    
    # Clamp
    return max(0, min(100, confidence))
```

**High-level confidence factors** (multiplicative):
```python
confidence_factors = {
    'form_consistency': compute_form_std_dev(player_history),  # Recent form variance
    'minutes_nailedness': start_probability,                   # Likely to play
    'fixture_clarity': fixture_score_std,                      # Fixture difficulty variance
    'data_sparsity': minutes / 1000,                           # Less data = lower confidence
}

# Combine multiplicatively
base_confidence = 100
for factor_name, factor_value in confidence_factors.items():
    base_confidence *= factor_value  # Each factor 0-1

confidence = max(0, min(100, base_confidence))
```

### 4.3 Fallback Logic

When one model fails (returns NaN or huge outlier):

```python
def ensemble_with_fallback(xgb_pred, nn_pred, xgb_ok=True, nn_ok=True,
                          fallback='rule_based', rule_pred=None):
    """
    Ensemble with robust fallback.
    
    Args:
        xgb_ok, nn_ok: Boolean indicators if model succeeded
        fallback: 'rule_based' or 'other_model'
        rule_pred: Prediction from rule-based PointPredictor
    
    Returns:
        Final prediction (non-NaN)
    """
    if xgb_ok and nn_ok:
        # Normal ensemble
        return ensemble_predict(xgb_pred, nn_pred)
    elif xgb_ok:
        return xgb_pred
    elif nn_ok:
        return nn_pred
    elif fallback == 'rule_based' and rule_pred is not None:
        return rule_pred
    else:
        # Ultimate fallback: position baseline
        position_baseline = {1: 1.5, 2: 2.5, 3: 4.0, 4: 5.0}
        return position_baseline.get(position, 3.0)
```

**Integration with existing system**:
- Fallback to `PointPredictor` if both ML models fail
- Log fallback occurrences for monitoring

---

## 5. Training Pipeline

### 5.1 Script Structure: `backend/ml_training.py`

**Main functions**:
```python
# Data pipeline
def load_data():              # Load all historical data + cache
def prepare_features():      # Build feature matrix with target
def create_splits():         # Train/val/test split (time-aware)
def scale_features():        # Fit scaler on train, transform all

# Model training
def train_xgb():             # Train XGBoost with CV
def train_nn():              # Train Neural Network
def tune_hyperparams():      # Grid/Random/Bayesian search

# Evaluation & serialization
def evaluate():              # Compute metrics on test set
def save_models():           # Serialize models + scaler + metadata
def generate_report():       # HTML/PDF report with plots

# Main orchestration
if __name__ == "__main__":
    main()
```

### 5.2 Training Data Split Strategy

**Time-series split** (respect temporal ordering):

```
Data: Gameweeks 1-38 for multiple seasons (2023-24, 2024-25, 2025-26 if available)

Split by chronology:
- Train: Seasons Y-2, Y-1 (GW 1-30 of season Y)  (~70%)
- Val: Season Y (GW 31-35)                        (~15%)
- Test: Season Y (GW 36-38)                       (~15%)

Alternative: Expanding window CV
Window 1: Train GW1-20, Val GW21-25
Window 2: Train GW1-25, Val GW26-30
...
```

**Implementation**:
```python
def time_based_split(df, test_cutoff_gw, val_cutoff_gw):
    """
    Split by gameweek number.
    
    Args:
        df: DataFrame with 'target_gw' column
        test_cutoff_gw: GW >= this goes to test
        val_cutoff_gw: GW >= this and < test goes to val
    """
    train_mask = df['target_gw'] < val_cutoff_gw
    val_mask = (df['target_gw'] >= val_cutoff_gw) & (df['target_gw'] < test_cutoff_gw)
    test_mask = df['target_gw'] >= test_cutoff_gw
    
    return df[train_mask], df[val_mask], df[test_mask]
```

### 5.3 Cross-Validation Approach

**For hyperparameter tuning**: Use expanding window time series CV

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train model with specific hyperparams
    model = train_xgb(X_train, y_train, X_val, y_val)
    
    # Collect validation scores
    scores.append(model.best_score)
```

**Note**: No k-fold CV (shuffling breaks temporal structure). Use `TimeSeriesSplit`.

### 5.4 Hyperparameter Tuning

**Recommended**: Bayesian optimization (`scikit-optimize`) over grid/random

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

param_space = {
    'n_estimators': Integer(200, 1000),
    'max_depth': Integer(4, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'min_child_weight': Integer(1, 10),
    'gamma': Real(0.0, 1.0),
    'reg_alpha': Real(0.0, 10.0),
    'reg_lambda': Real(0.0, 10.0)
}

opt = BayesSearchCV(
    estimator=xgb.XGBRegressor(tree_method='hist', n_jobs=-1),
    search_spaces=param_space,
    n_iter=50,               # 50 configurations
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

opt.fit(X_train, y_train)
best_params = opt.best_params_
```

**For NN**: Hyperopt or manual search (fewer hyperparameters to tune)

### 5.5 Model Versioning & Storage

**Directory structure**:
```
models/
├── version_2025_v1/
│   ├── xgb/
│   │   ├── model.pkl
│   │   ├── params.json
│   │   └── feature_columns.csv
│   ├── nn/
│   │   ├── model.keras
│   │   ├── params.json
│   │   └── feature_columns.csv
│   ├── scaler.pkl
│   ├── metadata.json          # Training summary, metrics, data version
│   └── report.html
└── current -> version_2025_v1/  # Symlink to production version
```

**Metadata tracking**:
```python
metadata = {
    'version': '2025_v1',
    'training_date': '2025-03-04',
    'data_source': 'FPL API + Understat',
    'train_seasons': ['2023-24', '2024-25'],
    'train_samples': 150000,
    'features': feature_columns,
    'xgb_params': best_xgb_params,
    'nn_params': best_nn_params,
    'metrics': {
        'xgb': {'mae': 1.95, 'rmse': 2.45, 'r2': 0.16},
        'nn': {'mae': 1.92, 'rmse': 2.42, 'r2': 0.17},
        'ensemble': {'mae': 1.89, 'rmse': 2.38, 'r2': 0.18}
    },
    'feature_importance_top20': importance_df.head(20).to_dict(),
    'training_time_seconds': 3600,
    'git_commit': 'abc123def'
}

import json
with open('models/version_2025_v1/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

## 6. Inference Pipeline

### 6.1 Loading Models in Production

```python
# backend/ml_predictor.py

import joblib
import xgboost as xgb
import tensorflow as tf
import numpy as np
import pandas as pd

class MLPredictor:
    def __init__(self, model_version='current'):
        """
        Load ML models and scaler.
        """
        model_path = f'models/{model_version}'
        
        # Load XGBoost
        self.xgb_model = joblib.load(f'{model_path}/xgb/model.pkl')
        
        # Load Neural Network
        self.nn_model = tf.keras.models.load_model(f'{model_path}/nn/model.keras')
        
        # Load scaler
        self.scaler = joblib.load(f'{model_path}/scaler.pkl')
        
        # Load feature columns (to ensure consistent ordering)
        self.feature_columns = pd.read_csv(f'{model_path}/xgb/feature_columns.csv')['feature'].tolist()
        
        # Load ensemble weights from metadata
        with open(f'{model_path}/metadata.json') as f:
            metadata = json.load(f)
        self.xgb_weight = metadata['ensemble_weights']['xgb']
        self.nn_weight = metadata['ensemble_weights']['nn']
        
        print(f"Loaded ML models version {model_version}")
    
    def preprocess(self, features_df):
        """
        Prepare features for model input.
        
        Args:
            features_df: Raw features (from feature extraction)
        
        Returns:
            X: Scaled feature matrix (same order as training)
        """
        # Ensure all required columns present, fill missing with 0
        X = features_df.reindex(columns=self.feature_columns, fill_value=0)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict_single(self, features, return_components=False):
        """
        Predict expected points for a single player.
        
        Args:
            features: Dict or DataFrame row with all features
            return_components: If True, return breakdown
        
        Returns:
            prediction: Final ensemble expected points
            confidence: 0-100 confidence score
            breakdown: Dict with XGB, NN, ensemble, confidence
        """
        # Convert to DataFrame with one row
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Preprocess
        X_scaled = self.preprocess(features)
        
        # XGBoost prediction
        xgb_pred = self.xgb_model.predict(X_scaled)[0]
        
        # NN prediction
        nn_pred = self.nn_model.predict(X_scaled, verbose=0)[0][0]
        
        # Ensemble
        ensemble_pred = self.xgb_weight * xgb_pred + self.nn_weight * nn_pred
        
        # Confidence score (simple version)
        # More sophisticated: use uncertainty estimates
        confidence = self._compute_confidence(xgb_pred, nn_pred, features)
        
        if return_components:
            breakdown = {
                'xgb': float(xgb_pred),
                'nn': float(nn_pred),
                'ensemble': float(ensemble_pred),
                'confidence': float(confidence),
                'weights': {'xgb': self.xgb_weight, 'nn': self.nn_weight}
            }
            return float(ensemble_pred), float(confidence), breakdown
        
        return float(ensemble_pred), float(confidence)
    
    def predict_batch(self, players_df, target_gw):
        """
        Batch prediction for all players.
        
        Args:
            players_df: DataFrame with player data + features
            target_gw: Target gameweek
        
        Returns:
            DataFrame with predictions and confidence
        """
        # Feature extraction already done externally
        
        predictions = []
        for _, player in players_df.iterrows():
            pred, conf = self.predict_single(player)
            predictions.append({
                'player_id': player['player_id'],
                'predicted_points': pred,
                'confidence': conf
            })
        
        return pd.DataFrame(predictions)
    
    def _compute_confidence(self, xgb_pred, nn_pred, features):
        """
        Compute confidence score as agreement between models + data quality.
        """
        # Model agreement
        diff = abs(xgb_pred - nn_pred)
        max_pred = max(xgb_pred, nn_pred, 1.0)
        agreement = 1.0 - min(diff / max_pred, 1.0)
        
        # Data quality factors
        availability = features.get('availability_factor', 1.0)
        minutes_consistency = features.get('start_probability', 0.5)
        
        # Combine
        confidence = agreement * 0.6 + availability * 0.2 + minutes_consistency * 0.2
        return min(100, max(0, confidence * 100))
```

### 6.2 Batch Prediction Function

```python
def predict_for_upcoming_gw(gameweek, use_ml=True, fallback_to_rule=True):
    """
    Full pipeline: fetch data, extract features, predict.
    
    Args:
        gameweek: Target GW
        use_ml: If False, use rule-based only
        fallback_to_rule: If ML fails, fall back to PointPredictor
    
    Returns:
        predictions_df: DataFrame with predictions for all players
    """
    # 1. Load current FPL data
    fpl_service = FPLService()
    static_data = fpl_service.get_latest_data()
    players_df = pd.DataFrame(static_data['elements'])
    teams_df = pd.DataFrame(static_data['teams'])
    fixtures = static_data['fixtures']
    
    # 2. Ensure history cached
    history_service = HistoricalDataService()
    history_service.fetch_all_player_history(static_data['elements'])
    
    # 3. Extract features for all players for target GW
    all_features = []
    for _, player in players_df.iterrows():
        features = extract_features(
            player_id=player['id'],
            target_gw=gameweek,
            history_service=history_service,
            static_data=static_data,
            use_understat=True
        )
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    # 4. Predict
    if use_ml:
        try:
            ml_predictor = MLPredictor(model_version='current')
            predictions = ml_predictor.predict_batch(features_df, gameweek)
            return predictions
        except Exception as e:
            if fallback_to_rule:
                print(f"ML prediction failed: {e}. Falling back to rule-based.")
                return predict_rule_based(players_df, teams_df, fixtures, gameweek)
            else:
                raise
    
    # Rule-based fallback
    return predict_rule_based(players_df, teams_df, fixtures, gameweek)
```

### 6.3 Confidence Threshold Logic

**For optimizer integration**:

```python
def get_predictions_with_threshold(predictions_df, confidence_threshold=50):
    """
    Filter predictions by confidence.
    
    Players below threshold get reduced weight or excluded from differential picks.
    """
    high_conf_mask = predictions_df['confidence'] >= confidence_threshold
    high_conf_players = predictions_df[high_conf_mask]
    
    # Apply confidence-weighted scaling to optimization objective
    # In optimizer, use: predicted_points * (confidence / 100) as adjusted value
    predictions_df['adjusted_points'] = (
        predictions_df['predicted_points'] * 
        (predictions_df['confidence'] / 100)
    )
    
    return predictions_df
```

**Optimizer integration**:
```python
# In MultiPeriodFPLOptimizer, replace 'ep_next' with ML predictions:
predictions = ml_predictor.predict_batch(...)
for _, pred in predictions.iterrows():
    player_id = pred['player_id']
    optimizer.player_data[player_id]['ep_next'] = pred['predicted_points']
    optimizer.player_data[player_id]['ml_confidence'] = pred['confidence']
```

### 6.4 Cache Invalidation Strategy

**Cache levels**:
1. **Feature cache**: Store extracted features for each (player_id, gw) pair
2. **Prediction cache**: Store final predictions + confidence
3. **Model cache**: Load models once at service startup

**Invalidation triggers**:
- New gameweek starts → Clear all caches (data stale)
- Model update deployed → Reload model (check version hash)
- Historical data change (rare) → Invalidate feature cache for affected players

**Implementation**:
```python
from functools import lru_cache

# Feature cache per GW
@lru_cache(maxsize=10000)
def get_cached_features(player_id, gw):
    return extract_features_raw(player_id, gw)

# Prediction cache (expires after 1 hour)
import time
prediction_cache = {}

def get_cached_prediction(player_id, gw, max_age_seconds=3600):
    key = (player_id, gw)
    if key in prediction_cache:
        pred, timestamp = prediction_cache[key]
        if time.time() - timestamp < max_age_seconds:
            return pred
        del prediction_cache[key]  # Expired
    return None

def predict_with_cache(player_id, gw):
    cached = get_cached_prediction(player_id, gw)
    if cached is not None:
        return cached
    
    # Compute fresh prediction
    features = get_cached_features(player_id, gw)
    pred, conf = ml_predictor.predict_single(features)
    prediction_cache[(player_id, gw)] = ((pred, conf), time.time())
    return pred, conf
```

**Cache Warming**: Pre-compute predictions for all players after data refresh.

---

## 7. Monitoring

### 7.1 Training Metrics to Track

**Per-epoch (NN)**:
- Train loss (Huber)
- Validation loss
- Train MAE, Val MAE
- Learning rate

**Per-model (final)**:
```python
metrics = {
    'mae': mean_absolute_error(y_test, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    'r2': r2_score(y_test, y_pred),
    'median_ae': median_absolute_error(y_test, y_pred),
    'mape': mean_absolute_percentage_error(y_test, y_pred),  # Can be unstable
    'within_2pts': (np.abs(y_pred - y_test) <= 2).mean(),
    'within_4pts': (np.abs(y_pred - y_test) <= 4).mean(),
    'overestimation_rate': (y_pred > y_test).mean(),  # Should be ~0.5
    'mean_residual': (y_pred - y_test).mean()  # Should be ~0
}
```

**By position**:
```python
for pos in [1,2,3,4]:
    mask = X_test['position'] == pos
    pos_mae = mean_absolute_error(y_test[mask], y_pred[mask])
    pos_r2 = r2_score(y_test[mask], y_pred[mask])
    print(f"Position {pos}: MAE={pos_mae:.3f}, R²={pos_r2:.3f}")
```

### 7.2 Model Drift Detection

**Drift metrics** (compare train vs prod distributions):

```python
def detect_drift(train_features, prod_features, threshold=0.1):
    """
    Detect feature drift using Population Stability Index (PSI).
    
    PSI = Σ( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )
    
    PSI < 0.1: No significant drift
    0.1 <= PSI < 0.2: Moderate drift
    PSI >= 0.2: Significant drift
    """
    psi_scores = {}
    for col in train_features.columns:
        # Bin continuous features
        train_hist, bins = np.histogram(train_features[col], bins=10)
        prod_hist, _ = np.histogram(prod_features[col], bins=bins)
        
        train_pct = train_hist / len(train_features) + 1e-6
        prod_pct = prod_hist / len(prod_features) + 1e-6
        
        psi = np.sum((prod_pct - train_pct) * np.log(prod_pct / train_pct))
        psi_scores[col] = psi
    
    # Overall drift score
    overall_psi = np.mean(list(psi_scores.values()))
    
    return {
        'overall_psi': overall_psi,
        'feature_psi': psi_scores,
        'drift_detected': overall_psi >= threshold
    }
```

**Model performance drift**:
- Track rolling MAE on latest 100-200 actual outcomes (if available)
- Alert if MAE increases >20% from baseline

### 7.3 Performance Degradation Alerts

**Monitoring dashboard** (to be implemented or use existing logging):

```python
class ModelMonitor:
    def __init__(self):
        self.predictions_log = []  # In production, use database or logging service
        
    def log_prediction(self, player_id, gw, pred, actual, confidence):
        self.predictions_log.append({
            'timestamp': datetime.now(),
            'player_id': player_id,
            'gw': gw,
            'predicted': pred,
            'actual': actual,
            'confidence': confidence,
            'error': abs(pred - actual)
        })
        
    def compute_rolling_metrics(self, window_size=200):
        """Compute MAE, R² over last N predictions."""
        recent = self.predictions_log[-window_size:]
        if len(recent) < 50:
            return None
        
        df = pd.DataFrame(recent)
        mae = df['error'].mean()
        r2 = r2_score(df['actual'], df['predicted'])
        
        return {'mae': mae, 'r2': r2, 'samples': len(df)}
    
    def check_alerts(self, baseline_mae=1.95):
        """Trigger alerts if performance degrades."""
        metrics = self.compute_rolling_metrics()
        if metrics and metrics['mae'] > baseline_mae * 1.2:
            send_alert(
                f"Model performance degraded. Current MAE: {metrics['mae']:.2f}, "
                f"Baseline: {baseline_mae:.2f}"
            )
```

**Alerting triggers**:
- Rolling MAE > 1.2 × baseline for > 3 consecutive GWs
- Feature drift PSI > 0.2 for > 3 consecutive GWs
- Model inference failure rate > 5%
- Confidence distribution shift (mean confidence drops > 15%)

---

## 8. Dependencies

**Add to `backend/requirements.txt`**:

```txt
# ML Core
xgboost>=2.0.0
tensorflow>=2.13.0  # or pytorch>=2.0.0 if preferred
scikit-learn>=1.2.0

# Hyperparameter Tuning
scikit-optimize>=0.9.0  # Bayesian optimization for XGBoost
optuna>=3.0.0           # Alternative hyperparameter optimizer (optional)

# Serialization & Versioning
joblib>=1.2.0

# Monitoring & Logging
wandb>=0.15.0  # Weights & Biases (optional but recommended)
mlflow>=2.0.0  # Alternative model tracking

# Data Processing (already in requirements)
pandas>=1.5.0
numpy>=1.24.0

# Visualization (for training analysis)
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.13.0  # Interactive plots (optional)

# Utilities
pyyaml>=6.0      # Config files
tqdm>=4.64.0     # Progress bars (already in requirements)
```

**Install**:
```bash
pip install xgboost tensorflow scikit-learn scikit-optimize joblib wandb mlflow matplotlib seaborn plotly pyyaml
```

---

## 9. Integration Notes

### 9.1 Replacing PointPredictor

**Option A**: Parallel deployment (A/B test)
```python
# In backend/point_predictor.py or ml_predictor.py wrapper

class HybridPredictor:
    def __init__(self, use_ml=None):
        self.ml_predictor = MLPredictor() if use_ml else None
        self.rule_predictor = PointPredictor(...)
    
    def predict(self, player, gw):
        # Use ML if available and player has enough data
        if (self.ml_predictor and 
            player['minutes'] > 60 and 
            player['understat_matched']):
            ml_pred, conf = self.ml_predictor.predict_single(player)
            if conf > 60:
                return ml_pred
        
        # Fallback to rule-based
        return self.rule_predictor.predict_gameweek(player, gw)
```

**Option B**: Full replacement (after A/B testing proves ML superior)

### 9.2 Backtesting Integration

Update `backend/backtest_engine.py` to use ML predictions:

```python
# In run_backtest(), replace:
# old: optimizer = MultiPeriodFPLOptimizer(...)
#      (uses PointPredictor internally)

# new:
ml_preds = ml_predictor.predict_batch(players_df, gw)
for _, pred in ml_preds.iterrows():
    optimizer.player_data[pred['player_id']]['ep_next'] = pred['predicted_points']
```

### 9.3 Model Retraining Schedule

- **Weekly**: After each GW, update feature cache with new data, but do NOT retrain
- **Monthly**: Retrain on all available data (last 2 seasons + current season so far)
- **End of season**: Full retrain with latest complete season data

**Retraining script**: `backend/retrain_models.py` (to be created)

---

## 10. Expected Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| Data leakage (using future info) | Strict time split, no peeking ahead |
| Sparse data for new players | Use position baselines, hierarchical modeling |
| Outliers (players with 20+ pts) | Winsorize target at 99th percentile during training |
| Understat mismatch (only ~60% matched) | Use `understat_matched` indicator as feature |
| Model instability across seasons | Use ensemble, regularize heavily, rolling retrain |
| Inference latency (real-time) | Pre-compute all predictions, cache results |

---

**End of ML-Models-Design Document**
