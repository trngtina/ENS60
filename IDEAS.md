# ENS Challenge 60 - Stock Auction Volume Prediction
## Complete Implementation Plan & Techniques Guide

---

## Challenge Overview

**Goal**: Predict the natural logarithm of closing auction volume (as a fraction of total daily volume) for 900 stocks across ~350 test days.

**Metric**: RMSE (Root Mean Squared Error) on log-transformed auction volume.

**Data Structure**:
- **Training**: ~800 days × 900 stocks = ~720,000 samples
- **Test**: ~350 days × 900 stocks = ~315,000 samples (strictly future)
- **Features**: 126 columns (61 `abs_retn0-60`, 61 `rel_voln0-60`, `pid`, `day`, `LS`, `NLV`)
- **Target**: `log(auction_volume / total_daily_volume)`

**Benchmark Score**: 0.4742 RMSE
**Target Score**: 0.30-0.35 RMSE (winning level)

---

## Project Structure

```
ENS60/
├── main.ipynb                 # Main notebook - full pipeline
├── IDEAS.md                   # This file - all techniques documented
├── requirements.txt           # Dependencies
├── configs/
│   └── config.yaml            # Hyperparameters, paths
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # Memory-optimized loading
│   ├── preprocessing.py       # NaN handling, scaling
│   ├── feature_engineering.py # 30+ engineered features
│   ├── validation.py          # TimeSeriesSplit, metrics
│   ├── models.py              # All model definitions
│   ├── ensemble.py            # Stacking, blending, OOF
│   ├── mlflow_utils.py        # Experiment tracking
│   └── visualization.py       # Plots, error analysis
└── data/
    └── (existing files)
```

---

## Implementation Phases

### Phase 1: Foundation (Target: ~0.45 RMSE - Beat Benchmark)
- [x] Data loading with memory optimization
- [x] Basic preprocessing (NaN handling)
- [x] 10 core aggregation features
- [x] Two-stage Linear + LightGBM
- [x] TimeSeriesSplit validation
- [x] MLflow tracking setup

### Phase 2: Competitive (Target: ~0.35-0.40 RMSE)
- [ ] Full feature engineering (30+ features)
- [ ] Hyperparameter tuning with Optuna
- [ ] CatBoost addition
- [ ] 3-model ensemble (Linear + LightGBM + CatBoost)
- [ ] Per-stock correction for high-error stocks

### Phase 3: Winning Level (Target: ~0.30-0.35 RMSE)
- [ ] RNN Encoder-Decoder with attention
- [ ] Stock embeddings
- [ ] SHAP-based feature selection
- [ ] Advanced stacking with Ridge meta-model
- [ ] Transformer architecture (optional)

---

## Technique Details

### 1. Data Preprocessing

#### 1.1 Memory-Optimized Loading
```python
# Optimize dtypes for memory efficiency
dtype_dict = {
    'pid': 'int16',
    'day': 'int16',
    'LS': 'float32',
    'NLV': 'float32',
    **{f'abs_ret{i}': 'float32' for i in range(61)},
    **{f'rel_vol{i}': 'float32' for i in range(61)}
}
```

#### 1.2 Missing Value Handling
- **Step 1**: Create NaN count features BEFORE imputation
- **Step 2**: Interpolate along axis=1 (across periods)
- **Step 3**: Fill remaining NaN with 0

```python
df['return_nan_count'] = df[return_cols].isnull().sum(axis=1)
df['volume_nan_count'] = df[volume_cols].isnull().sum(axis=1)
df[return_cols] = df[return_cols].interpolate(axis=1, limit_direction='both')
df[volume_cols] = df[volume_cols].interpolate(axis=1, limit_direction='both')
df = df.fillna(0)
```

---

### 2. Feature Engineering

#### 2.1 Statistical Aggregations (Core Features)

**Returns (abs_retn0-60)**:
| Feature | Description |
|---------|-------------|
| `min_ret` | Minimum return across periods |
| `max_ret` | Maximum return across periods |
| `std_ret` | Standard deviation of returns |
| `median_ret` | Median return |
| `sum_ret` | Sum of returns |
| `mean_ret` | Mean return |
| `range_ret` | max_ret - min_ret |

**Volumes (rel_voln0-60)**:
| Feature | Description |
|---------|-------------|
| `min_vol` | Minimum volume fraction |
| `max_vol` | Maximum volume fraction |
| `std_vol` | Standard deviation of volumes |
| `median_vol` | Median volume |
| `mean_vol` | Mean volume |
| `skew_vol` | Skewness of volume distribution |
| `kurt_vol` | Kurtosis of volume distribution |

#### 2.2 Group-by Day Features (Market-Wide Regime)
```python
# Aggregate statistics per day (across all 900 stocks)
day_stats = df.groupby('day').agg({
    'sum_ret': ['mean', 'std', 'median'],
    'median_vol': ['mean', 'std', 'median'],
    'target': ['mean', 'std', 'median']  # Only on train
})
```

#### 2.3 Temporal Features (Per-Stock)
```python
# Lag features
for lag in [1, 2, 3, 7, 14]:
    df[f'target_lag_{lag}'] = df.groupby('pid')['target'].shift(lag)

# Rolling statistics
for window in [7, 14, 30]:
    df[f'target_rolling_mean_{window}'] = df.groupby('pid')['target'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
```

#### 2.4 Domain-Inspired Features
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `eod_vol_concentration` | sum(rel_vol51-60) | End-of-day volume predicts auction |
| `vol_price_interaction` | std_ret × median_vol | VWAP algorithm amplification |
| `early_return` | mean(abs_ret0-19) | Morning trading pattern |
| `late_return` | mean(abs_ret41-60) | Afternoon trading pattern |
| `return_momentum` | late_return - early_return | Intraday momentum |
| `vol_skew` | skew(rel_vol) | Volume concentration shape |
| `vol_peak_location` | argmax(rel_vol) | When volume peaks |

#### 2.5 NLV Interaction Features
```python
df['NLV_x_median_vol'] = df['NLV'] * df['median_vol']
df['NLV_x_day'] = df['NLV'] * df['day']
df['NLV_per_pid'] = df.groupby('pid')['NLV'].transform('mean')
df['NLV_deviation'] = df['NLV'] - df['NLV_per_pid']
```

---

### 3. Models

#### 3.1 Baseline: Linear Regression on NLV
- **Purpose**: Establish baseline, NLV explains 62% of variance
- **Expected RMSE**: ~1.33

#### 3.2 Two-Stage Benchmark (CFM Official)
```python
# Stage 1: Linear regression
linear_model.fit(X_train_numerical, y_train)
linear_preds = linear_model.predict(X_train_numerical)

# Stage 2: LightGBM on residuals
residuals = y_train - linear_preds
lgb_model.fit(X_train_with_pid, residuals, categorical_feature=['pid'])

# Final: base + residual correction
final_pred = linear_preds_test + lgb_model.predict(X_test_with_pid)
```

#### 3.3 Optimized LightGBM
```python
params = {
    'num_leaves': 45,
    'max_depth': 7,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'min_child_samples': 211,
    'subsample': 0.52,
    'colsample_bytree': 0.79,
    'reg_alpha': 2,
    'reg_lambda': 20
}
```

#### 3.4 CatBoost
```python
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=10,
    cat_features=['pid'],
    early_stopping_rounds=50
)
```

#### 3.5 Per-Stock Ridge
```python
for pid in unique_pids:
    if len(train_pid) > 500:
        model = Ridge(alpha=1.0)
        model.fit(train_pid[features], train_pid['target'])
```

#### 3.6 RNN Encoder-Decoder (Winning Architecture)
```python
class AuctionVolumeRNN(nn.Module):
    def __init__(self):
        self.pid_embedding = nn.Embedding(900, 32)
        self.encoder = nn.LSTM(122, 128, num_layers=2, bidirectional=True)
        self.attention = nn.MultiheadAttention(256, num_heads=4)
        self.fc = nn.Sequential(
            nn.Linear(256 + 32 + 4, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
```

---

### 4. Validation Strategy

#### 4.1 TimeSeriesSplit (MANDATORY)
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

#### 4.2 Custom Temporal Split
```python
train_days = df['day'] <= 600
val_days = (df['day'] > 600) & (df['day'] <= 700)
test_days = df['day'] > 700
```

#### 4.3 Purged K-Fold (Advanced)
- Gap between train and validation to prevent leakage
- Embargo period after validation

---

### 5. Ensemble Strategy

#### 5.1 Out-of-Fold Predictions
```python
oof_predictions = np.zeros((len(train_df), n_models))
for fold, (train_idx, val_idx) in enumerate(tscv.split(train_df)):
    for i, model in enumerate(models):
        model.fit(X_train[train_idx], y_train[train_idx])
        oof_predictions[val_idx, i] = model.predict(X_train[val_idx])
```

#### 5.2 Stacking with Ridge Meta-Learner
```python
meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_predictions, y_train)
final_pred = meta_model.predict(test_predictions)
```

#### 5.3 Weighted Average (Fallback)
```python
weights = {'rnn': 0.4, 'lightgbm': 0.3, 'catboost': 0.2, 'ridge': 0.1}
final_pred = sum(w * preds[name] for name, w in weights.items())
```

---

### 6. Hyperparameter Tuning

#### 6.1 Optuna for LightGBM
```python
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 50)
    }
    # ... cross-validation with TimeSeriesSplit
    return mean_rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

---

### 7. Error Analysis

#### 7.1 Per-Stock RMSE
```python
test_df['error'] = np.abs(test_df['target'] - test_df['pred'])
worst_pids = test_df.groupby('pid')['error'].mean().nlargest(20)
```

#### 7.2 Learning Curves
- Diagnose bias vs variance
- High bias: both curves plateau high → need more complex model
- High variance: large gap → need regularization

#### 7.3 Feature Importance
- SHAP values for interpretability
- Permutation importance for stability

---

### 8. MLflow Tracking

```python
import mlflow

with mlflow.start_run(run_name="lgb_v1"):
    mlflow.log_params(params)
    mlflow.log_metric("rmse_val", rmse)
    mlflow.log_metric("mae_val", mae)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("feature_importance.png")
```

---

## Critical Mistakes to Avoid

1. ❌ **Using random cross-validation** → Always use TimeSeriesSplit
2. ❌ **Ignoring NLV** → It's the most predictive feature (62% R²)
3. ❌ **One-hot encoding `pid`** → Use native categorical support
4. ❌ **Training without `pid` feature** → Per-stock or embedding required
5. ❌ **Not handling NaN carefully** → Both impute AND create NaN count feature
6. ❌ **Skipping feature engineering** → Raw features alone won't beat benchmark

---

## Success Formula

```
Winning Score = (Strong Features) × (Right Model) × (Proper Validation) × (Ensemble)
              = (Temporal + Domain + Aggregated) × (RNN + Tree Ensemble) × (TimeSeriesSplit) × (Stacking)
```

---

## References

- Challenge Data ENS #60: https://challengedata.ens.fr/challenges/60
- Winner: Franck Zibi (2021) - RNN + LightGBM + CatBoost + Stacking
- LightGBM: https://lightgbm.readthedocs.io/
- CatBoost: https://catboost.ai/
- SHAP: https://github.com/slundberg/shap
- Optuna: https://optuna.org/