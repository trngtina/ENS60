import json
from pathlib import Path
from dataclasses import dataclass
import sys

import numpy as np
import pandas as pd
import mlflow
import lightgbm as lgb
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure local package imports work when script is run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import (
    load_config,
    load_data,
    merge_target,
    create_submission,
    get_column_names,
)
from utils.preprocessing import preprocess_data
from utils.feature_engineering import create_features, get_feature_list
from utils.validation import create_temporal_split, compute_metrics


@dataclass
class RunConfig:
    lgb_trials: int = 40
    cat_trials: int = 25
    lgb_tune_sample_step: int = 3
    cat_tune_sample_step: int = 4
    rnn_train_sample_step: int = 2
    rnn_epochs: int = 8
    rnn_batch_size: int = 1024
    rnn_lr: float = 1e-3
    rnn_weight_decay: float = 1e-5
    seed: int = 42


class ZibiSeqModel(nn.Module):
    def __init__(self, num_pids: int, static_dim: int, emb_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.pid_emb = nn.Embedding(num_pids, emb_dim)
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + emb_dim + static_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, seq_x, pid_x, static_x):
        out, _ = self.gru(seq_x)
        scores = self.attn(out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (out * weights).sum(dim=1)
        emb = self.pid_emb(pid_x)
        z = torch.cat([ctx, emb, static_x], dim=1)
        return self.head(z).squeeze(1)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def optimize_blend_weights(y_true: np.ndarray, preds: np.ndarray) -> np.ndarray:
    n = preds.shape[1]

    def objective(w):
        y = preds @ w
        return rmse(y_true, y)

    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(objective, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
    if not res.success:
        return x0
    return res.x


def train_rnn_with_pid_embedding(
    seq_train: np.ndarray,
    pid_train: np.ndarray,
    static_train: np.ndarray,
    y_train: np.ndarray,
    seq_val: np.ndarray,
    pid_val: np.ndarray,
    static_val: np.ndarray,
    y_val: np.ndarray,
    cfg: RunConfig,
) -> tuple[ZibiSeqModel, np.ndarray, float]:
    device = torch.device("cpu")
    model = ZibiSeqModel(num_pids=900, static_dim=static_train.shape[1]).to(device)

    train_ds = TensorDataset(
        torch.tensor(seq_train, dtype=torch.float32),
        torch.tensor(pid_train, dtype=torch.long),
        torch.tensor(static_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_seq_t = torch.tensor(seq_val, dtype=torch.float32, device=device)
    val_pid_t = torch.tensor(pid_val, dtype=torch.long, device=device)
    val_static_t = torch.tensor(static_val, dtype=torch.float32, device=device)

    train_loader = DataLoader(train_ds, batch_size=cfg.rnn_batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.rnn_lr, weight_decay=cfg.rnn_weight_decay)
    crit = nn.SmoothL1Loss()

    best_rmse = float("inf")
    best_state = None
    patience = 2
    no_improve = 0

    for epoch in range(cfg.rnn_epochs):
        model.train()
        for b_seq, b_pid, b_static, b_y in train_loader:
            b_seq = b_seq.to(device)
            b_pid = b_pid.to(device)
            b_static = b_static.to(device)
            b_y = b_y.to(device)

            opt.zero_grad()
            p = model(b_seq, b_pid, b_static)
            loss = crit(p, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_seq_t, val_pid_t, val_static_t).cpu().numpy()
        val_rmse = rmse(y_val, val_pred)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_pred = model(val_seq_t, val_pid_t, val_static_t).cpu().numpy()
    return model, final_pred, best_rmse


def main():
    run_cfg = RunConfig()
    np.random.seed(run_cfg.seed)
    torch.manual_seed(run_cfg.seed)

    cfg = load_config("configs/config.yaml")
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("ENS60_Auction_Volume")

    print("Loading data...")
    X_train_raw, y_train, X_test_raw = load_data(config=cfg, load_test=True, verbose=False)
    X_train_proc, _, X_test_proc = preprocess_data(X_train_raw, y_train, X_test_raw, config=cfg, verbose=False)
    train_df = merge_target(X_train_proc, y_train, cfg, align_mode="auto", strict=True)

    print("Building engineered features...")
    train_df = create_features(
        train_df,
        config=cfg,
        include_aggregations=True,
        include_domain=True,
        include_non_leaky_temporal=True,
        include_nlv=True,
        include_day=True,
        include_temporal=False,
        verbose=False,
    )
    X_test_feat = create_features(
        X_test_proc,
        config=cfg,
        include_aggregations=True,
        include_domain=True,
        include_non_leaky_temporal=True,
        temporal_history_df=train_df,
        include_nlv=True,
        include_day=True,
        include_temporal=False,
        verbose=False,
    )

    day_col = cfg["features"]["day_col"]
    target_col = cfg["features"]["target_col"]
    train_end_day = int(train_df[day_col].max() * 0.85)
    tr_df, va_df, _ = create_temporal_split(train_df, train_end_day=train_end_day, config=cfg)

    feat, cat_cols = get_feature_list(train_df, cfg, exclude_raw=False, exclude_target=True, include_id=True)
    if day_col not in feat:
        feat.append(day_col)
    cat_cols = [c for c in cat_cols if c in feat]

    X_tr = tr_df[feat]
    y_tr = tr_df[target_col]
    X_va = va_df[feat]
    y_va = va_df[target_col]

    # -------------------- LightGBM Optuna --------------------
    print("Tuning LightGBM...")
    tune_idx_lgb = np.arange(len(X_tr))[:: run_cfg.lgb_tune_sample_step]
    X_lgb_tune = X_tr.iloc[tune_idx_lgb]
    y_lgb_tune = y_tr.iloc[tune_idx_lgb]

    def lgb_objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 350),
            "subsample": trial.suggest_float("subsample", 0.5, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 40.0),
            "random_state": run_cfg.seed,
            "verbose": -1,
        }
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_lgb_tune,
            y_lgb_tune,
            eval_set=[(X_va, y_va)],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        pred = m.predict(X_va)
        return rmse(y_va.values, pred)

    lgb_study = optuna.create_study(direction="minimize")
    lgb_study.optimize(lgb_objective, n_trials=run_cfg.lgb_trials, show_progress_bar=False)
    best_lgb_params = lgb_study.best_params.copy()
    best_lgb_params.update(
        {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "random_state": run_cfg.seed,
            "verbose": -1,
        }
    )

    with mlflow.start_run(run_name="zibi_lgb_optuna_best"):
        mlflow.log_params({"model": "lightgbm", **best_lgb_params})
        mlflow.log_metric("optuna_best_val_rmse", float(lgb_study.best_value))

    lgb_model = lgb.LGBMRegressor(**best_lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    lgb_val_pred = lgb_model.predict(X_va)
    lgb_val_rmse = rmse(y_va.values, lgb_val_pred)

    # -------------------- CatBoost Optuna --------------------
    print("Tuning CatBoost...")
    tune_idx_cat = np.arange(len(X_tr))[:: run_cfg.cat_tune_sample_step]
    X_cat_tune = X_tr.iloc[tune_idx_cat]
    y_cat_tune = y_tr.iloc[tune_idx_cat]

    def cat_objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "depth": trial.suggest_int("depth", 5, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": run_cfg.seed,
            "verbose": False,
        }
        m = CatBoostRegressor(**params)
        m.fit(
            X_cat_tune,
            y_cat_tune,
            eval_set=(X_va, y_va),
            cat_features=cat_cols,
            use_best_model=True,
        )
        pred = m.predict(X_va)
        return rmse(y_va.values, pred)

    cat_study = optuna.create_study(direction="minimize")
    cat_study.optimize(cat_objective, n_trials=run_cfg.cat_trials, show_progress_bar=False)
    best_cat_params = cat_study.best_params.copy()
    best_cat_params.update(
        {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": run_cfg.seed,
            "verbose": False,
        }
    )

    with mlflow.start_run(run_name="zibi_cat_optuna_best"):
        mlflow.log_params({"model": "catboost", **best_cat_params})
        mlflow.log_metric("optuna_best_val_rmse", float(cat_study.best_value))

    cat_model = CatBoostRegressor(**best_cat_params)
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_cols, use_best_model=True)
    cat_val_pred = cat_model.predict(X_va)
    cat_val_rmse = rmse(y_va.values, cat_val_pred)

    # -------------------- Sequence Model (RNN + pid embedding) --------------------
    print("Training sequence model...")
    ret_cols, vol_cols = get_column_names(cfg)

    seq_all = np.stack(
        [X_train_proc[ret_cols].values.astype(np.float32), X_train_proc[vol_cols].values.astype(np.float32)],
        axis=2,
    )

    # Align sequence to engineered train_df indices
    seq_df = pd.DataFrame(index=train_df.index)
    tr_idx = tr_df.index.values
    va_idx = va_df.index.values

    static_cols = ["NLV", "LS", day_col]
    static_train = tr_df[static_cols].copy()
    static_val = va_df[static_cols].copy()
    static_train[day_col] = static_train[day_col] / train_df[day_col].max()
    static_val[day_col] = static_val[day_col] / train_df[day_col].max()

    scaler = StandardScaler()
    static_train_np = scaler.fit_transform(static_train.values).astype(np.float32)
    static_val_np = scaler.transform(static_val.values).astype(np.float32)

    tr_sample_idx = tr_idx[:: run_cfg.rnn_train_sample_step]
    seq_tr = seq_all[tr_sample_idx]
    pid_tr = train_df.loc[tr_sample_idx, "pid"].values.astype(np.int64)
    y_tr_rnn = train_df.loc[tr_sample_idx, target_col].values.astype(np.float32)
    static_tr = static_train_np[:: run_cfg.rnn_train_sample_step]

    seq_va = seq_all[va_idx]
    pid_va = train_df.loc[va_idx, "pid"].values.astype(np.int64)
    y_va_rnn = train_df.loc[va_idx, target_col].values.astype(np.float32)
    static_va = static_val_np

    rnn_model, rnn_val_pred, rnn_val_rmse = train_rnn_with_pid_embedding(
        seq_train=seq_tr,
        pid_train=pid_tr,
        static_train=static_tr,
        y_train=y_tr_rnn,
        seq_val=seq_va,
        pid_val=pid_va,
        static_val=static_va,
        y_val=y_va_rnn,
        cfg=run_cfg,
    )

    with mlflow.start_run(run_name="zibi_rnn_pid_embedding"):
        mlflow.log_params(
            {
                "model": "rnn_pid_embedding",
                "epochs": run_cfg.rnn_epochs,
                "batch_size": run_cfg.rnn_batch_size,
                "lr": run_cfg.rnn_lr,
                "weight_decay": run_cfg.rnn_weight_decay,
                "train_sample_step": run_cfg.rnn_train_sample_step,
            }
        )
        mlflow.log_metric("val_rmse", float(rnn_val_rmse))

    # -------------------- Blend --------------------
    print("Optimizing blend weights...")
    pred_matrix = np.column_stack([lgb_val_pred, cat_val_pred, rnn_val_pred])
    weights = optimize_blend_weights(y_va.values, pred_matrix)
    blend_val_pred = pred_matrix @ weights
    blend_val_rmse = rmse(y_va.values, blend_val_pred)

    with mlflow.start_run(run_name="zibi_blend_final"):
        mlflow.log_params(
            {
                "models": "lgb,cat,rnn",
                "w_lgb": float(weights[0]),
                "w_cat": float(weights[1]),
                "w_rnn": float(weights[2]),
            }
        )
        mlflow.log_metric("val_rmse_blend", float(blend_val_rmse))
        mlflow.log_metric("val_rmse_lgb", float(lgb_val_rmse))
        mlflow.log_metric("val_rmse_cat", float(cat_val_rmse))
        mlflow.log_metric("val_rmse_rnn", float(rnn_val_rmse))

    # -------------------- Final fit for submission --------------------
    print("Training final models on full train and creating submission...")
    X_full = train_df[feat]
    y_full = train_df[target_col]
    X_test_full = X_test_feat[feat]

    lgb_final = lgb.LGBMRegressor(**best_lgb_params)
    lgb_final.fit(X_full, y_full, categorical_feature=cat_cols)
    lgb_test_pred = lgb_final.predict(X_test_full)

    cat_final = CatBoostRegressor(**best_cat_params)
    cat_final.fit(X_full, y_full, cat_features=cat_cols)
    cat_test_pred = cat_final.predict(X_test_full)

    # Sequence test prediction
    seq_test = np.stack(
        [X_test_proc[ret_cols].values.astype(np.float32), X_test_proc[vol_cols].values.astype(np.float32)],
        axis=2,
    )
    static_test = X_test_feat[static_cols].copy()
    static_test[day_col] = static_test[day_col] / train_df[day_col].max()
    static_test_np = scaler.transform(static_test.values).astype(np.float32)
    pid_test = X_test_feat["pid"].values.astype(np.int64)

    with torch.no_grad():
        rnn_test_pred = (
            rnn_model(
                torch.tensor(seq_test, dtype=torch.float32),
                torch.tensor(pid_test, dtype=torch.long),
                torch.tensor(static_test_np, dtype=torch.float32),
            )
            .cpu()
            .numpy()
        )

    test_pred_matrix = np.column_stack([lgb_test_pred, cat_test_pred, rnn_test_pred])
    test_pred = test_pred_matrix @ weights

    submission = create_submission(
        predictions=test_pred,
        X_test=X_test_feat,
        config=cfg,
        output_path="outputs/submission_zibi_ensemble.csv",
        validate_against_example=True,
    )

    summary = {
        "val_rmse": {
            "lgb": float(lgb_val_rmse),
            "cat": float(cat_val_rmse),
            "rnn": float(rnn_val_rmse),
            "blend": float(blend_val_rmse),
        },
        "blend_weights": {"lgb": float(weights[0]), "cat": float(weights[1]), "rnn": float(weights[2])},
        "submission": {
            "rows": int(len(submission)),
            "id_first": int(submission["ID"].iloc[0]),
            "id_last": int(submission["ID"].iloc[-1]),
            "path": "outputs/submission_zibi_ensemble.csv",
        },
    }

    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open("outputs/zibi_ensemble_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
