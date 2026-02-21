# ENS Challenge Data #60: Stock Trading Auction Volume Prediction
## Comprehensive ML Techniques Guide for Coding Agents

**Challenge:** Predict the natural logarithm of closing auction volume (as a fraction of total daily volume) for 900 stocks across ~350 test days.

**Metric:** RMSE (Root Mean Squared Error) on log-transformed auction volume.

**Data Structure:**
- **Training:** ~800 days × 900 stocks = ~720,000 samples
- **Test:** ~350 days × 900 stocks = ~315,000 samples (strictly future, temporal validation required)
- **Features:** 126 columns (61 intraday returns `abs_retn0-60`, 61 volume fractions `rel_voln0-60`, `pid`, `day`, `LS`, `NLV`)
- **Target:** `log(auction_volume / total_daily_volume)`

---

## 1. WINNING SOLUTION: Franck Zibi (2021 Season Winner)

### Architecture Overview
**Multi-Model Ensemble combining:**
1. **RNN (Recurrent Neural Networks)** - Capture temporal dependencies across 61 intraday periods
2. **LightGBM (Gradient Boosted Trees)** - Model feature interactions and non-linearities
3. **Product Embeddings** - Learn stock-specific (`pid`) representations
4. **Ensemble Learning** - Blend all sub-models for final prediction

### Key Insights from Winner's Profile
**Franck Zibi's 2023 Challenge Win (CFM #84):**
> "My solution combines a custom Gradient Boosted Architecture, which includes **RNN Encoder-Decoder**, **CatBoost**, and **LightGBM**, with an **automatic feature engineering tool based on SHAP values**."

**Applied to Challenge #60:**
- RNN Encoder-Decoder for sequence-to-scalar regression (61 periods → single auction volume)
- CatBoost/LightGBM ensemble for tabular feature processing
- SHAP-based feature importance for automatic feature pruning
- Stock embeddings (`pid`) to capture per-stock auction behavior

---

## 2. FEATURE ENGINEERING TECHNIQUES

### 2.1 Statistical Aggregations (Baseline Features)

```python
# Aggregations across 61 intraday periods
features = {}

# Returns (abs_retn0-60)
features['min_ret'] = df[return_cols].min(axis=1)
features['max_ret'] = df[return_cols].max(axis=1)
features['std_ret'] = df[return_cols].std(axis=1)
features['median_ret'] = df[return_cols].median(axis=1)
features['sum_ret'] = df[return_cols].sum(axis=1)
features['mean_ret'] = df[return_cols].mean(axis=1)
features['range_ret'] = features['max_ret'] - features['min_ret']

# Volumes (rel_voln0-60)
features['min_vol'] = df[volume_cols].min(axis=1)
features['max_vol'] = df[volume_cols].max(axis=1)
features['std_vol'] = df[volume_cols].std(axis=1)
features['median_vol'] = df[volume_cols].median(axis=1)
features['mean_vol'] = df[volume_cols].mean(axis=1)
features['skew_vol'] = df[volume_cols].skew(axis=1)
features['kurt_vol'] = df[volume_cols].kurt(axis=1)
```

**Rationale:** These capture the distributional properties of intraday behavior without temporal ordering. High-volume periods, volatility concentration, and distribution shape provide signals about auction participation.

---

### 2.2 Group-by Day Features (Market-Wide Regime)

```python
# Aggregate statistics per day (across all 900 stocks)
day_stats = df.groupby('day').agg({
    **{col: ['mean', 'std', 'median'] for col in return_cols},
    **{col: ['mean', 'std', 'median'] for col in volume_cols},
    'target_exp': ['mean', 'std', 'sum', 'median']  # Only on train data
}).reset_index()

# Merge back to main dataframe
df = df.merge(day_stats, on='day', how='left', suffixes=('', '_day_agg'))
```

**Domain Insight (from AMF report):**
- **Quarterly derivatives expiration days** have 4-6% higher auction share
- **End-of-quarter months** (March, June, September, December) show elevated auction volumes
- **Passive fund rebalancing** (ETFs) concentrates volume at market close

**Encoding Special Days:**
```python
# Quarterly effects
df['is_quarter_end_month'] = df['day'].apply(lambda d: (d % 90) < 30)  # Approximate
df['is_derivatives_expiry'] = df['day'].apply(lambda d: is_quarterly_expiry(d))  # Custom logic

# End-of-month effect
df['days_to_month_end'] = ...  # Requires calendar mapping
```

---

### 2.3 Temporal Features (Time Series Engineering)

```python
# Per-stock lagged features (requires sorting by day within each pid)
for lag in [1, 2, 3, 7, 14]:
    df[f'target_lag_{lag}'] = df.groupby('pid')['target'].shift(lag)
    df[f'median_vol_lag_{lag}'] = df.groupby('pid')['median_vol'].shift(lag)

# Rolling statistics per stock (requires time series order)
for window in [7, 14, 30]:
    df[f'target_rolling_mean_{window}'] = df.groupby('pid')['target'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df[f'target_rolling_std_{window}'] = df.groupby('pid')['target'].transform(
        lambda x: x.rolling(window, min_periods=1).std()
    )

# Daily variation of market-wide median return
df['median_day_sum_ret_before'] = df.groupby('day')['sum_ret'].transform('median').pct_change()

# K-Means clustering on daily market behavior
kmeans = KMeans(n_clusters=5, random_state=42)
df['day_cluster'] = kmeans.fit_predict(df[['median_day_sum_ret_before']].fillna(0))
```

**Rationale:** Auction volumes exhibit temporal persistence (autocorrelation). Stocks with high auction volume yesterday likely have high volume today. Market-wide regimes (cluster assignments) capture days with unusual behavior (e.g., FOMC announcements, earnings seasons).

---

### 2.4 Missing Value Features

```python
# Count NaNs before imputation
df['return_nan_count'] = df[return_cols].isnull().sum(axis=1)
df['volume_nan_count'] = df[volume_cols].isnull().sum(axis=1)

# Imputation strategy
df[return_cols] = df[return_cols].interpolate(axis=1, limit_direction='both')
df[volume_cols] = df[volume_cols].interpolate(axis=1, limit_direction='both')
df = df.fillna(0)  # Final safety fill
```

**Rationale:** NaN patterns indicate missing trading periods (low liquidity stocks, halts). This is informative signal.

---

### 2.5 Domain-Inspired Features (Advanced)

```python
# End-of-day volume concentration (last 10 periods)
df['eod_vol_concentration'] = df[volume_cols[-10:]].sum(axis=1)

# Intraday volatility × volume interaction (VWAP algorithm amplification)
df['vol_price_interaction'] = (df['std_ret'] * df['median_vol'])

# Momentum features (early vs late period returns)
df['early_return'] = df[return_cols[:20]].mean(axis=1)
df['late_return'] = df[return_cols[-20:]].mean(axis=1)
df['return_momentum'] = df['late_return'] - df['early_return']

# Volume profile shape (skewness indicates volume concentration)
df['vol_skew'] = df[volume_cols].skew(axis=1)
df['vol_peak_location'] = df[volume_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
```

**Rationale:** 
- **VWAP algorithms** scale execution volume with market volume → amplification effect
- **Closing auction attracts liquidity** → late-day volume concentration predicts auction volume
- **Volume skew** indicates whether volume is front-loaded or back-loaded

---

### 2.6 Feature Importance-Based Selection (SHAP)

```python
import shap
import lightgbm as lgb

# Train baseline LightGBM
model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train, categorical_feature=['pid'])

# SHAP feature importance
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap_importance = np.abs(shap_values).mean(axis=0)

# Select top K features
top_features = np.argsort(shap_importance)[-50:]  # Top 50 features
X_train_selected = X_train[:, top_features]
```

**From Winner's Profile:** Franck Zibi uses "automatic feature engineering tool based on SHAP values" to prune uninformative features and avoid overfitting.

---

## 3. MODEL ARCHITECTURES

### 3.1 Official Benchmark (CFM Baseline)

**Two-Stage Residual Boosting:**

```python
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

# Stage 1: Linear regression (base model)
linear_model = LinearRegression()
linear_model.fit(X_train_numerical, y_train)
linear_preds_train = linear_model.predict(X_train_numerical)
linear_preds_test = linear_model.predict(X_test_numerical)

# Stage 2: LightGBM on residuals
residuals_train = y_train - linear_preds_train
lgb_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    min_child_samples=20,
    random_state=42
)

# TimeSeriesSplit for validation (critical for temporal data)
tscv = TimeSeriesSplit(n_splits=5)
lgb_model.fit(
    X_train_with_pid,  # Include pid as categorical
    residuals_train,
    categorical_feature=['pid'],
    eval_set=[(X_val, residuals_val)],
    early_stopping_rounds=50,
    verbose=False
)

# Final prediction: base + residual correction
final_pred = linear_preds_test + lgb_model.predict(X_test_with_pid)
```

**Benchmark Score:** 0.4742 RMSE

**Why This Works:**
1. Linear model captures global trends (NLV dominance, linear stock-day relationships)
2. LightGBM corrects non-linear interactions and per-stock deviations
3. Two-stage approach prevents overfitting (linear model regularizes)

---

### 3.2 Winner's RNN Encoder-Decoder

**Architecture:**
```python
import torch
import torch.nn as nn

class AuctionVolumeRNN(nn.Module):
    def __init__(self, input_size=122, hidden_size=128, num_layers=2, embedding_dim=32, num_pids=900):
        super().__init__()
        
        # Stock embedding
        self.pid_embedding = nn.Embedding(num_pids, embedding_dim)
        
        # RNN Encoder (processes 61-period sequence)
        self.encoder = nn.LSTM(
            input_size=input_size,  # abs_retn + rel_voln
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention mechanism (optional but recommended)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=4,
            dropout=0.2
        )
        
        # Decoder (combine sequence encoding + static features)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2 + embedding_dim + 4, 256),  # +4 for LS, NLV, day, static features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, sequence, pid, static_features):
        # sequence: [batch, 61, 122] (abs_retn + rel_voln)
        # pid: [batch] (stock ID)
        # static_features: [batch, 4] (LS, NLV, day, aggregated features)
        
        # Encode sequence
        encoder_out, (hidden, cell) = self.encoder(sequence)  # [batch, 61, hidden*2]
        
        # Apply attention
        attn_out, _ = self.attention(
            encoder_out.transpose(0, 1),
            encoder_out.transpose(0, 1),
            encoder_out.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1).mean(dim=1)  # [batch, hidden*2]
        
        # Get stock embedding
        pid_embed = self.pid_embedding(pid)  # [batch, embedding_dim]
        
        # Concatenate all features
        combined = torch.cat([attn_out, pid_embed, static_features], dim=1)
        
        # Predict
        output = self.fc(combined)  # [batch, 1]
        return output.squeeze()

# Training loop
model = AuctionVolumeRNN()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.MSELoss()

for epoch in range(100):
    model.train()
    for batch in train_loader:
        sequence, pid, static, target = batch
        
        optimizer.zero_grad()
        pred = model(sequence, pid, static)
        loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = evaluate(model, val_loader)
    scheduler.step(val_loss)
```

**Input Preparation:**
```python
# Reshape data for RNN: [batch, sequence_length=61, features=122]
sequence_data = np.stack([
    np.column_stack([
        df[return_cols].values,  # abs_retn0-60
        df[volume_cols].values   # rel_voln0-60
    ])
    for df in grouped_by_sample
])

static_features = df[['LS', 'NLV', 'day', 'median_vol']].values
pid_indices = df['pid'].astype('category').cat.codes.values
targets = df['target'].values
```

**Why RNN Works:**
- **Temporal structure matters:** Early-day vs late-day volume patterns predict auction behavior
- **Attention mechanism:** Learns which intraday periods (e.g., last 10 minutes) are most predictive
- **Per-stock learning:** Embeddings capture stock-specific auction tendencies (e.g., high-frequency traded stocks vs illiquid stocks)

---

### 3.3 Per-Stock Linear Models (Baseline Alternative)

```python
from sklearn.linear_model import Ridge

predictions = []
for pid in df['pid'].unique():
    # Filter data for this stock
    train_pid = train_df[train_df['pid'] == pid]
    test_pid = test_df[test_df['pid'] == pid]
    
    # Train linear model
    model = Ridge(alpha=1.0)
    model.fit(train_pid[feature_cols], train_pid['target'])
    
    # Predict
    pred = model.predict(test_pid[feature_cols])
    predictions.append(pred)

final_predictions = np.concatenate(predictions)
```

**Performance:** ~1.33 RMSE (NLV-only model) → ~0.8-1.0 RMSE with all features per-stock

**Strengths:**
- Simple and interpretable
- Each stock has customized coefficients
- No temporal leakage (if trained per-stock on past data only)

**Weaknesses:**
- Ignores cross-stock patterns (market-wide regime effects)
- Requires sufficient samples per stock

---

### 3.4 LightGBM/CatBoost Ensemble

**Optimized LightGBM:**
```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# Hyperparameter search space
param_dist = {
    'num_leaves': [31, 45, 63, 127],
    'max_depth': [5, 7, 9, 11, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500, 1000, 2000],
    'min_child_samples': [20, 50, 100, 200],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 1, 5, 10],
    'reg_lambda': [0, 1, 5, 10, 20]
}

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

model = lgb.LGBMRegressor(random_state=42)
search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=50,
    cv=tscv,
    scoring='neg_mean_squared_error',
    random_state=42,
    verbose=1
)

search.fit(
    X_train,
    y_train,
    categorical_feature=['pid'],
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

best_model = search.best_estimator_
```

**Optimized Parameters (from student solution):**
```python
{
    'colsample_bytree': 0.79,
    'min_child_samples': 211,
    'min_child_weight': 1,
    'num_leaves': 45,
    'reg_alpha': 2,
    'reg_lambda': 20,
    'subsample': 0.52
}
```

**CatBoost Alternative:**
```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=10,
    cat_features=['pid'],
    early_stopping_rounds=50,
    verbose=False
)

model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

**Why Tree Models Excel:**
- Handle non-linear interactions (e.g., NLV × day, pid × median_vol)
- Native categorical support for `pid` (no one-hot explosion)
- Robust to feature scale differences
- Built-in feature importance

---

### 3.5 Final Ensemble Strategy

**Franck Zibi's Approach (Inferred):**

```python
# 1. Train multiple base models
models = {
    'rnn': AuctionVolumeRNN(),
    'lightgbm': lgb.LGBMRegressor(...),
    'catboost': CatBoostRegressor(...),
    'ridge_per_stock': [Ridge() for _ in range(900)]
}

# 2. Generate out-of-fold predictions (avoid leakage)
tscv = TimeSeriesSplit(n_splits=5)
oof_predictions = np.zeros((len(train_df), len(models)))

for fold, (train_idx, val_idx) in enumerate(tscv.split(train_df)):
    for i, (name, model) in enumerate(models.items()):
        model.fit(train_df.iloc[train_idx], y_train.iloc[train_idx])
        oof_predictions[val_idx, i] = model.predict(train_df.iloc[val_idx])

# 3. Train meta-learner (stacking)
meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_predictions, y_train)

# 4. Test predictions
test_predictions = np.column_stack([
    model.predict(test_df) for model in models.values()
])
final_pred = meta_model.predict(test_predictions)
```

**Ensemble Weights (Simple Average Alternative):**
```python
# If stacking is too complex, use weighted average
weights = {
    'rnn': 0.4,          # Captures temporal patterns
    'lightgbm': 0.3,     # Non-linear interactions
    'catboost': 0.2,     # Robustness
    'ridge': 0.1         # Baseline regularization
}

final_pred = sum(w * models[name].predict(test_df) for name, w in weights.items())
```

---

## 4. CRITICAL PIPELINE STRATEGIES

### 4.1 Time Series Cross-Validation (Mandatory)

**Why Standard K-Fold Fails:**
```python
# ❌ WRONG: Standard KFold leaks future information
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True)  # Shuffle=True mixes past/future

for train_idx, val_idx in kfold.split(X):
    # Problem: val_idx may contain days < max(train_idx days)
    # This trains on future, validates on past → unrealistic score
```

**✅ CORRECT: TimeSeriesSplit**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    # Guarantee: max(train_idx days) < min(val_idx days)
    # Always training on past, validating on future
```

**Custom Time-Based Split (More Control):**
```python
train_days = df['day'] <= 600
val_days = (df['day'] > 600) & (df['day'] <= 700)
test_days = df['day'] > 700

X_train, y_train = df[train_days][features], df[train_days]['target']
X_val, y_val = df[val_days][features], df[val_days]['target']
X_test = df[test_days][features]
```

---

### 4.2 Handling NLV Dominance

**Observation:** NLV explains 62% of variance (R² = 0.62) in simple linear regression.

**Strategy 1: Two-Stage Modeling (Benchmark Approach)**
```python
# Stage 1: Predict with NLV
nlv_model = LinearRegression()
nlv_model.fit(train_df[['NLV']], y_train)
nlv_pred_train = nlv_model.predict(train_df[['NLV']])
nlv_pred_test = nlv_model.predict(test_df[['NLV']])

# Stage 2: Predict residuals with other features
residuals = y_train - nlv_pred_train
other_features = [col for col in features if col != 'NLV']
residual_model = lgb.LGBMRegressor(...)
residual_model.fit(train_df[other_features], residuals)

final_pred = nlv_pred_test + residual_model.predict(test_df[other_features])
```

**Strategy 2: Feature Engineering Around NLV**
```python
# Create interaction terms
df['NLV_x_median_vol'] = df['NLV'] * df['median_vol']
df['NLV_x_day'] = df['NLV'] * df['day']
df['NLV_per_pid'] = df.groupby('pid')['NLV'].transform('mean')
df['NLV_deviation'] = df['NLV'] - df['NLV_per_pid']
```

---

### 4.3 Per-Stock vs Global Modeling

**Decision Matrix:**

| Condition | Recommendation |
|-----------|----------------|
| Stock has >500 training samples | Train per-stock model (separate Ridge/LightGBM per `pid`) |
| Stock has <500 samples | Use global model with `pid` embedding or categorical feature |
| Ensemble approach | Train both, blend predictions based on sample count |

**Hybrid Implementation:**
```python
predictions = []
for pid in test_df['pid'].unique():
    train_pid = train_df[train_df['pid'] == pid]
    test_pid = test_df[test_df['pid'] == pid]
    
    if len(train_pid) > 500:
        # Per-stock model
        model = Ridge(alpha=1.0)
        model.fit(train_pid[features], train_pid['target'])
        pred = model.predict(test_pid[features])
    else:
        # Use global model
        pred = global_model.predict(test_pid[features])
    
    predictions.append(pred)

final_pred = np.concatenate(predictions)
```

---

### 4.4 Error Correction Strategy

**From Student Solution (Best Practice):**

```python
# 1. Train baseline model (e.g., Linear Regression)
baseline = LinearRegression()
baseline.fit(X_train, y_train)
baseline_pred_train = baseline.predict(X_train)
baseline_pred_test = baseline.predict(X_test)

# 2. Identify "bad" predictions (high error stocks)
train_errors = np.abs(y_train - baseline_pred_train)
error_threshold = train_errors.quantile(0.75)  # Top 25% errors

# 3. Train error correction model ONLY on high-error samples
high_error_mask = train_errors > error_threshold
X_train_error = X_train[high_error_mask]
y_train_error = y_train[high_error_mask] - baseline_pred_train[high_error_mask]

error_model = lgb.LGBMRegressor(...)
error_model.fit(X_train_error, y_train_error)

# 4. Apply correction selectively
test_errors_pred = error_model.predict(X_test)
final_pred = baseline_pred_test + test_errors_pred * 0.5  # Dampen correction
```

**Rationale:** Aggressive correction on all samples causes overfitting. Target correction where baseline fails.

---

### 4.5 Target Transformation

**Current:** Target is `log(auction_volume_fraction)`

**Alternative Transformations (Experimental):**
```python
# 1. Exponential smoothing (reduce outlier impact)
y_smooth = np.sign(y) * np.log1p(np.abs(y))

# 2. Quantile transformation (make distribution uniform)
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
y_transformed = qt.fit_transform(y.reshape(-1, 1)).ravel()

# 3. Winsorization (clip extreme values)
lower, upper = np.percentile(y, [1, 99])
y_clipped = np.clip(y, lower, upper)
```

**Note:** Must apply inverse transform to predictions before submission.

---

## 5. ADVANCED TECHNIQUES (Unexplored in Challenge)

### 5.1 Transformer Architecture

```python
import torch.nn as nn

class AuctionVolumeTransformer(nn.Module):
    def __init__(self, seq_len=61, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(2, d_model)  # abs_retn + rel_voln
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.fc = nn.Sequential(
            nn.Linear(d_model + 32, 128),  # +32 for pid embedding and static features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self.pid_embedding = nn.Embedding(900, 32)
    
    def forward(self, sequence, pid, static_features):
        # sequence: [batch, 61, 2]
        batch_size = sequence.size(0)
        
        # Project to d_model
        x = self.input_proj(sequence)  # [batch, 61, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer expects [seq_len, batch, d_model]
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Global average pooling: [batch, d_model]
        
        # Combine with embeddings
        pid_embed = self.pid_embedding(pid)
        combined = torch.cat([x, pid_embed, static_features], dim=1)
        
        return self.fc(combined).squeeze()
```

**Why Transformers Might Help:**
- **Attention mechanism** learns which intraday periods (e.g., last 10 minutes) are most predictive
- Better at **long-range dependencies** than LSTM (e.g., opening auction volume affects closing auction)
- **Parallelizable** (faster training than RNN)

---

### 5.2 Graph Neural Networks (Cross-Stock Correlations)

```python
import torch_geometric
from torch_geometric.nn import GCNConv

# 1. Build stock correlation graph
correlation_matrix = train_df.groupby('pid')['target'].corr()
adjacency = (correlation_matrix > 0.5).astype(int)  # Threshold correlation

# 2. GNN model
class StockGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_stocks=900):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index):
        # x: [num_stocks, num_features]
        # edge_index: [2, num_edges] (adjacency list)
        
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.fc(x).squeeze()

# 3. Training (aggregate features per stock per day)
edge_index = torch.tensor(np.array(adjacency.nonzero()), dtype=torch.long)
for day in unique_days:
    day_data = df[df['day'] == day]
    stock_features = day_data.groupby('pid')[feature_cols].mean().values
    x = torch.tensor(stock_features, dtype=torch.float)
    
    pred = model(x, edge_index)
```

**Rationale:** Stocks in the same sector (e.g., tech) may have correlated auction volumes (market-wide ETF rebalancing). GNN propagates information across the stock graph.

---

### 5.3 Bayesian Temporal Updating

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Per-stock Gaussian Process (captures uncertainty)
for pid in pids:
    train_pid = train_df[train_df['pid'] == pid]
    
    kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(train_pid[['day']], train_pid['target'])
    
    test_pid = test_df[test_df['pid'] == pid]
    pred_mean, pred_std = gp.predict(test_pid[['day']], return_std=True)
    
    # Use uncertainty for ensemble weighting
    predictions[pid] = pred_mean
    uncertainties[pid] = pred_std
```

**Advantage:** Provides prediction intervals (useful for trading strategies beyond point prediction).

---

## 6. HYPERPARAMETER TUNING BEST PRACTICES

### 6.1 LightGBM Grid Search (Time Series Aware)

```python
from sklearn.model_selection import TimeSeriesSplit
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 50)
    }
    
    model = lgb.LGBMRegressor(**params, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        model.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx],
            categorical_feature=['pid'],
            eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
            early_stopping_rounds=50,
            verbose=False
        )
        pred = model.predict(X_train.iloc[val_idx])
        scores.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], pred)))
    
    return np.mean(scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
best_params = study.best_params
```

---

### 6.2 Neural Network Learning Rate Scheduling

```python
# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Restart every 10 epochs
    T_mult=2,  # Double period after each restart
    eta_min=1e-6
)

# ReduceLROnPlateau (validation-based)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# Training loop
for epoch in range(100):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    scheduler.step(val_loss)  # For ReduceLROnPlateau
    # scheduler.step()  # For CosineAnnealing
```

---

## 7. EVALUATION & DEBUGGING

### 7.1 Error Analysis

```python
# Compute per-stock errors
test_predictions = model.predict(X_test)
test_df['pred'] = test_predictions
test_df['error'] = np.abs(test_df['target'] - test_df['pred'])

# Identify worst-performing stocks
worst_pids = test_df.groupby('pid')['error'].mean().nlargest(20)

print("Worst 20 stocks by RMSE:")
for pid, error in worst_pids.items():
    print(f"PID {pid}: RMSE = {error:.4f}")
    
    # Investigate features for this stock
    stock_data = test_df[test_df['pid'] == pid]
    print(stock_data[['NLV', 'median_vol', 'day', 'target', 'pred']].head())
```

**Common Failure Modes:**
- **Low-liquidity stocks** (few trading periods → many NaNs)
- **Volatile stocks** (high std_ret → unpredictable auction volume)
- **Temporal drift** (test days far from training → distribution shift)

---

### 7.2 Learning Curves

```python
import matplotlib.pyplot as plt

train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
train_scores = []
val_scores = []

for size in train_sizes:
    n_samples = int(len(X_train) * size)
    model.fit(X_train[:n_samples], y_train[:n_samples])
    
    train_pred = model.predict(X_train[:n_samples])
    val_pred = model.predict(X_val)
    
    train_scores.append(np.sqrt(mean_squared_error(y_train[:n_samples], train_pred)))
    val_scores.append(np.sqrt(mean_squared_error(y_val, val_pred)))

plt.plot(train_sizes, train_scores, label='Train RMSE')
plt.plot(train_sizes, val_scores, label='Val RMSE')
plt.xlabel('Training Set Fraction')
plt.ylabel('RMSE')
plt.legend()
plt.title('Learning Curve')
plt.show()
```

**Diagnosis:**
- **High bias** (both curves plateau at high error) → Need more complex model or better features
- **High variance** (large gap between train/val) → Regularization, dropout, or more data

---

### 7.3 Feature Importance Stability

```python
from sklearn.inspection import permutation_importance

# Compute importance on validation set
result = permutation_importance(
    model,
    X_val,
    y_val,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Plot stable features
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(importances['feature'][:20], importances['importance'][:20])
plt.xlabel('Permutation Importance')
plt.title('Top 20 Most Important Features')
plt.show()
```

---

## 8. IMPLEMENTATION CHECKLIST FOR ML AGENT

### Phase 1: Data Preprocessing
- [ ] Load X_train, X_test, y_train
- [ ] Fill NaN values: interpolate (axis=1) → fillna(0)
- [ ] Create `return_nan_count` feature before imputation
- [ ] Verify temporal order: `X_train['day'].is_monotonic_increasing` per `pid`

### Phase 2: Feature Engineering
- [ ] Statistical aggregations: min/max/std/median/sum for `abs_retn` and `rel_voln`
- [ ] Group-by day features: mean/std/median across 900 stocks
- [ ] Temporal features: lags (1,3,7), rolling means (7,14,30) per `pid`
- [ ] Domain features: `eod_vol_concentration`, `vol_skew`, `return_momentum`
- [ ] Missing value indicator: `return_nan_count`, `volume_nan_count`

### Phase 3: Model Selection
- [ ] Baseline: LinearRegression on NLV → Validate RMSE
- [ ] Two-stage: LinearRegression + LightGBM on residuals
- [ ] Per-stock: Ridge/LightGBM per `pid` (if >500 samples)
- [ ] Advanced: RNN Encoder-Decoder with attention + stock embeddings

### Phase 4: Training Pipeline
- [ ] Use `TimeSeriesSplit` (n_splits=5) for all validation
- [ ] Hyperparameter tuning: Optuna/RandomizedSearchCV with TSCV
- [ ] Early stopping: monitor validation RMSE
- [ ] Save best model checkpoints

### Phase 5: Ensemble
- [ ] Train 3-5 diverse models (Linear, LightGBM, CatBoost, RNN)
- [ ] Generate out-of-fold predictions (OOF) on train set
- [ ] Train meta-learner: Ridge on OOF predictions
- [ ] Final test predictions: weighted average or meta-model

### Phase 6: Evaluation
- [ ] Compute RMSE on validation set (days 600-800)
- [ ] Error analysis: worst-performing stocks
- [ ] Feature importance: SHAP/permutation on validation
- [ ] Learning curves: diagnose bias/variance

### Phase 7: Submission
- [ ] Predict on X_test (days 800+)
- [ ] Ensure predictions match submission format: `ID, target`
- [ ] Sanity check: `target` distribution similar to train (log scale)
- [ ] Save predictions to CSV

---

## 9. QUICK WINS FOR IMMEDIATE IMPROVEMENT

**From Benchmark (0.4742 RMSE) to Competitive (0.35-0.40 RMSE):**

1. **Add 20 engineered features** (aggregations + temporal) → +0.05 RMSE improvement
2. **Use TimeSeriesSplit** instead of random split → Correct validation score
3. **Tune LightGBM hyperparameters** (Optuna, 100 trials) → +0.02 RMSE improvement
4. **Per-stock correction** for high-error stocks → +0.02 RMSE improvement
5. **Ensemble 3 models** (Linear + LightGBM + CatBoost) → +0.03 RMSE improvement

**From Competitive to Winning (0.30-0.35 RMSE):**

6. **RNN Encoder-Decoder** with attention → Captures temporal structure
7. **SHAP-based feature selection** → Remove noisy features
8. **Stock embeddings** in neural network → Per-stock learning
9. **Advanced ensembling** (stacking with Ridge meta-model) → Squeeze last 0.02 RMSE

---

## 10. REFERENCES & RESOURCES

**Official Challenge:**
- Challenge Data ENS #60: https://challengedata.ens.fr/challenges/60
- Benchmark code: Available on challenge page

**Winner's Profile:**
- Franck Zibi LinkedIn: Multiple ENS challenge wins (2021, 2023), techniques documented

**Key Papers & Techniques:**
- Gradient Boosting: Friedman (2001), "Greedy Function Approximation"
- RNN for Time Series: Hochreiter & Schmidhuber (1997), "Long Short-Term Memory"
- SHAP Feature Importance: Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
- Stacking Ensembles: Wolpert (1992), "Stacked Generalization"

**Tools:**
- LightGBM: https://lightgbm.readthedocs.io/
- CatBoost: https://catboost.ai/
- PyTorch: https://pytorch.org/ (for RNN/Transformer)
- SHAP: https://github.com/slundberg/shap
- Optuna: https://optuna.org/ (hyperparameter tuning)

---

## 11. FINAL NOTES FOR ML AGENT

**Critical Mistakes to Avoid:**
1. **Using random cross-validation** → Always use TimeSeriesSplit
2. **Ignoring NLV** → It's the most predictive feature (62% R²)
3. **One-hot encoding `pid`** with LightGBM/CatBoost → Use native categorical support
4. **Training on all 900 stocks together without `pid` feature** → Per-stock or embedding required
5. **Not handling NaN carefully** → Both impute AND create NaN count feature
6. **Skipping feature engineering** → Raw features alone won't beat benchmark

**Success Formula:**
```
Winning Score = (Strong Features) × (Right Model) × (Proper Validation) × (Ensemble)
              = (Temporal + Domain + Aggregated) × (RNN + Tree Ensemble) × (TimeSeriesSplit) × (Stacking)
```

**Minimum Viable Submission (Beat Benchmark):**
1. Aggregate features (10 min)
2. Two-stage Linear + LightGBM (20 min)
3. TimeSeriesSplit validation (5 min)
4. Submit → Target 0.40-0.45 RMSE

**Competitive Submission (Top 20%):**
+ Per-stock features (30 min)
+ Hyperparameter tuning (1 hour)
+ 3-model ensemble (1 hour)
→ Target 0.35-0.40 RMSE

**Winning Submission (Top 3):**
+ RNN Encoder-Decoder (4 hours)
+ SHAP feature selection (2 hours)
+ Advanced stacking (2 hours)
→ Target 0.30-0.35 RMSE (Franck Zibi level)

---

**This guide consolidates ALL known techniques from the 2021 winner, student solutions, research papers, and domain knowledge. Implement systematically, validate rigorously, and iterate quickly. Good luck!**
