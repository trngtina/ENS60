# ENS Challenge 60 - ML Agent Guide

## Challenge Overview

**Goal**: Predict the natural logarithm of closing auction volume (as a fraction of total daily volume) for 900 stocks across ~350 test days.

**Metric**: RMSE (Root Mean Squared Error)  
**Benchmark**: 0.4742 RMSE  
**Target**: < 0.40 RMSE (competitive), < 0.35 RMSE (winning)

---

## Critical Must-Haves

### 1. Data Alignment ⚠️ CRITICAL
The target file has IDs starting at 1070752, NOT 0. The merge must be done by **position**, not by ID.

```python
# WRONG - causes complete data misalignment
df['ID'] = range(len(df))  # Creates 0, 1, 2...
df = df.merge(y_train[['ID', target_col]], on='ID')  # FAILS!

# CORRECT - positional assignment
df[target_col] = y_train[target_col].values
```

**File**: `utils/data_loader.py` → `merge_target()` function

### 2. NLV is the Most Important Feature
NLV (Normalized Log Volume) explains **62% of variance** (R² = 0.62). 

**Expected NLV-only baseline**:
- RMSE: ~0.29
- R²: ~0.62

If NLV baseline shows RMSE > 0.5, there's a data alignment issue!

### 3. Exclude Non-Feature Columns
These columns must NEVER be used as features:
- `day` - temporal identifier for splitting only
- `ID` - row identifier from target file
- `index` - pandas index

**File**: `utils/feature_engineering.py` → `get_feature_list()` function

### 4. Time Series Cross-Validation
**NEVER use random K-Fold** - it causes temporal leakage!

```python
# WRONG
KFold(n_splits=5, shuffle=True)  # Mixes past/future

# CORRECT
TimeSeriesSplit(n_splits=5)  # Always train on past, validate on future
```

**File**: `utils/validation.py` → `TimeSeriesValidator` class

### 5. Categorical Feature Handling
`pid` (stock ID) should be used as a **categorical feature**, not numeric.

```python
# LightGBM native categorical support
lgb_model.fit(X, y, categorical_feature=['pid'])
```

---

## Project Structure

```
ENS60/
├── AGENT.md                 # This file - agent guidelines
├── IDEAS.md                 # All techniques and ideas
├── ENS-Challenge-60-ML-Techniques-Guide.md  # Comprehensive techniques reference
├── main.ipynb               # Main pipeline notebook
├── configs/
│   └── config.yaml          # Hyperparameters and paths
├── data/
│   ├── input_training.csv.gz    # Training features (126 cols)
│   ├── output_training_*.csv    # Training targets
│   └── input_test.csv.gz        # Test features
├── outputs/
│   └── submission_*.csv     # Submission files
└── utils/
    ├── data_loader.py       # Data loading and merging
    ├── preprocessing.py     # NaN handling, scaling
    ├── feature_engineering.py  # Feature creation
    ├── validation.py        # Time series CV
    ├── models.py            # Model definitions
    ├── ensemble.py          # Stacking, blending
    ├── mlflow_utils.py      # Experiment tracking
    └── visualization.py     # Plots
```

---

## Data Structure

### Training Input (126 columns)
- `pid`: Stock ID (0-899)
- `day`: Day index (0-799)
- `abs_retn0-60`: 61 intraday return columns
- `rel_voln0-60`: 61 intraday volume fraction columns
- `LS`: Unknown feature
- `NLV`: Normalized Log Volume (**most predictive**)

### Target
- `log(auction_volume / total_daily_volume)`

---

## Model Pipeline

### Phase 1: Baseline (Beat 0.4742)
1. NLV-only linear regression → ~0.29 RMSE
2. Two-stage: Linear + LightGBM on residuals → ~0.35-0.40 RMSE
3. LightGBM with all features → ~0.35-0.40 RMSE

### Phase 2: Competitive (< 0.40)
1. Feature engineering (aggregations, domain features)
2. Hyperparameter tuning (Optuna)
3. Per-stock correction for high-error stocks

### Phase 3: Winning (< 0.35)
1. RNN Encoder-Decoder for temporal patterns
2. Stock embeddings
3. Ensemble stacking (Linear + LightGBM + CatBoost + RNN)

---

## Key Files Reference

| File | Purpose | Critical Functions |
|------|---------|-------------------|
| `utils/data_loader.py` | Load and merge data | `merge_target()` - MUST use positional assignment |
| `utils/feature_engineering.py` | Create features | `get_feature_list()` - MUST exclude ID, day, index |
| `utils/validation.py` | Cross-validation | `TimeSeriesValidator` - MUST use temporal splits |
| `utils/models.py` | Model definitions | `TwoStageModel` - CFM benchmark approach |
| `configs/config.yaml` | All hyperparameters | LightGBM params, feature settings |

---

## Debugging Checklist

### If NLV baseline RMSE > 0.5:
- [ ] Check `merge_target()` uses positional assignment
- [ ] Verify `len(X_train) == len(y_train)`
- [ ] Check target column name matches config

### If feature importance shows ID or day:
- [ ] Check `get_feature_list()` excludes these columns
- [ ] Verify `exclude_cols` set is correct

### If CV scores are unrealistic:
- [ ] Ensure using `TimeSeriesSplit`, not `KFold`
- [ ] Check no future data leakage in features

### If LightGBM categorical error:
- [ ] Verify `pid` is in `categorical_cols` list
- [ ] Check column exists in feature DataFrame

---

## Performance Targets

| Model | Expected RMSE | R² |
|-------|--------------|-----|
| NLV-only baseline | ~0.29 | ~0.62 |
| Two-stage (Linear + LGB) | ~0.35-0.40 | ~0.70-0.75 |
| LightGBM full features | ~0.35-0.40 | ~0.70-0.75 |
| Ensemble (winning) | ~0.30-0.35 | ~0.80+ |

---

## References

- **Techniques Guide**: `ENS-Challenge-60-ML-Techniques-Guide.md`
- **Ideas Document**: `IDEAS.md`
- **Challenge PDF**: `ChallengeData.pdf`
- **Winner's Approach**: Franck Zibi (2021) - RNN + LightGBM + CatBoost + Stacking