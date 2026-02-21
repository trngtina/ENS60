"""
Validation Utilities for ENS Challenge 60
Time series cross-validation and metrics computation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any, List, Generator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
import warnings

warnings.filterwarnings('ignore')

from .data_loader import load_config


class TimeSeriesValidator:
    """
    Time series cross-validation for stock data.
    
    Ensures that validation data always comes AFTER training data
    to prevent temporal leakage.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the validator.
        
        Args:
            n_splits: Number of cross-validation splits
            gap: Number of days to skip between train and validation
            config: Configuration dictionary
        """
        self.n_splits = n_splits
        self.gap = gap
        self.config = config if config is not None else load_config()
        self.day_col = self.config['features']['day_col']
    
    def split(
        self,
        df: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/validation indices for time series cross-validation.
        
        Args:
            df: DataFrame with day column
            y: Target series (not used, for sklearn compatibility)
            
        Yields:
            Tuple of (train_indices, validation_indices)
        """
        # Get unique days sorted
        unique_days = sorted(df[self.day_col].unique())
        n_days = len(unique_days)
        
        # Calculate split points
        fold_size = n_days // (self.n_splits + 1)
        
        for fold in range(self.n_splits):
            # Training days: from start to current fold
            train_end_idx = (fold + 1) * fold_size
            train_days = unique_days[:train_end_idx]
            
            # Validation days: after gap
            val_start_idx = train_end_idx + self.gap
            val_end_idx = val_start_idx + fold_size
            
            if val_end_idx > n_days:
                val_end_idx = n_days
            
            val_days = unique_days[val_start_idx:val_end_idx]
            
            if len(val_days) == 0:
                continue
            
            # Get indices
            train_mask = df[self.day_col].isin(train_days)
            val_mask = df[self.day_col].isin(val_days)
            
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            
            yield train_idx, val_idx
    
    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


def create_temporal_split(
    df: pd.DataFrame,
    train_end_day: int,
    val_end_day: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Create a simple temporal train/validation/test split.
    
    Args:
        df: DataFrame with day column
        train_end_day: Last day to include in training
        val_end_day: Last day to include in validation (if None, all remaining is validation)
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_df, val_df, test_df) where test_df is None if val_end_day is None
    """
    if config is None:
        config = load_config()
    
    day_col = config['features']['day_col']
    
    train_df = df[df[day_col] <= train_end_day].copy()
    
    if val_end_day is not None:
        val_df = df[(df[day_col] > train_end_day) & (df[day_col] <= val_end_day)].copy()
        test_df = df[df[day_col] > val_end_day].copy()
        return train_df, val_df, test_df
    else:
        val_df = df[df[day_col] > train_end_day].copy()
        return train_df, val_df, None


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        metrics: List of metrics to compute (default: rmse, mae, r2)
        
    Returns:
        Dictionary of metric names to values
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'r2']
    
    results = {}
    
    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(y_true, y_pred)
    
    if 'r2' in metrics:
        results['r2'] = r2_score(y_true, y_pred)
    
    if 'mape' in metrics:
        # Mean Absolute Percentage Error (avoid division by zero)
        mask = y_true != 0
        if mask.sum() > 0:
            results['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            results['mape'] = np.nan
    
    if 'mse' in metrics:
        results['mse'] = mean_squared_error(y_true, y_pred)
    
    return results


def compute_per_stock_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Compute metrics per stock.
    
    Args:
        df: DataFrame with predictions and targets
        y_true_col: Column name for true values
        y_pred_col: Column name for predictions
        config: Configuration dictionary
        
    Returns:
        DataFrame with per-stock metrics
    """
    if config is None:
        config = load_config()
    
    id_col = config['features']['id_col']
    
    def stock_metrics(group):
        y_true = group[y_true_col].values
        y_pred = group[y_pred_col].values
        
        return pd.Series({
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
            'n_samples': len(y_true),
            'mean_target': y_true.mean(),
            'std_target': y_true.std(),
        })
    
    stock_metrics_df = df.groupby(id_col).apply(stock_metrics).reset_index()
    
    return stock_metrics_df


def compute_per_day_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Compute metrics per day.
    
    Args:
        df: DataFrame with predictions and targets
        y_true_col: Column name for true values
        y_pred_col: Column name for predictions
        config: Configuration dictionary
        
    Returns:
        DataFrame with per-day metrics
    """
    if config is None:
        config = load_config()
    
    day_col = config['features']['day_col']
    
    def day_metrics(group):
        y_true = group[y_true_col].values
        y_pred = group[y_pred_col].values
        
        return pd.Series({
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
            'n_samples': len(y_true),
        })
    
    day_metrics_df = df.groupby(day_col).apply(day_metrics).reset_index()
    
    return day_metrics_df


def cross_validate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    validator: Optional[TimeSeriesValidator] = None,
    n_splits: int = 5,
    split_df: Optional[pd.DataFrame] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform time series cross-validation.
    
    Args:
        model: Model with fit/predict interface
        X: Feature DataFrame
        y: Target Series
        validator: TimeSeriesValidator instance (if None, creates one)
        n_splits: Number of splits (if validator is None)
        split_df: DataFrame used only for generating temporal splits (e.g., day column only)
        fit_kwargs: Additional keyword arguments passed to model.fit per fold
        config: Configuration dictionary
        verbose: Whether to print progress
        
    Returns:
        Dictionary with CV results
    """
    if config is None:
        config = load_config()
    
    if validator is None:
        validator = TimeSeriesValidator(n_splits=n_splits, config=config)
    
    if split_df is not None and len(split_df) != len(X):
        raise ValueError(f"split_df length ({len(split_df)}) must match X length ({len(X)})")
    
    fold_metrics = []
    oof_predictions = np.zeros(len(X))
    oof_mask = np.zeros(len(X), dtype=bool)
    split_source = split_df if split_df is not None else X
    fit_kwargs = fit_kwargs or {}
    
    for fold, (train_idx, val_idx) in enumerate(validator.split(split_source)):
        if verbose:
            print(f"Fold {fold + 1}/{validator.get_n_splits()}")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Clone model to ensure fold isolation
        fold_model = clone(model)
        
        # Train model
        fold_model.fit(X_train, y_train, **fit_kwargs)
        
        # Predict
        y_pred = fold_model.predict(X_val)
        
        # Store OOF predictions
        oof_predictions[val_idx] = y_pred
        oof_mask[val_idx] = True
        
        # Compute metrics
        metrics = compute_metrics(y_val.values, y_pred)
        metrics['fold'] = fold + 1
        metrics['train_size'] = len(train_idx)
        metrics['val_size'] = len(val_idx)
        fold_metrics.append(metrics)
        
        if verbose:
            print(f"  RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    
    # Aggregate results
    fold_df = pd.DataFrame(fold_metrics)
    
    results = {
        'fold_metrics': fold_df,
        'mean_rmse': fold_df['rmse'].mean(),
        'std_rmse': fold_df['rmse'].std(),
        'mean_mae': fold_df['mae'].mean(),
        'std_mae': fold_df['mae'].std(),
        'mean_r2': fold_df['r2'].mean(),
        'std_r2': fold_df['r2'].std(),
        'oof_predictions': oof_predictions,
        'oof_mask': oof_mask,
    }
    
    if verbose:
        print(f"\nCV Results:")
        print(f"  RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
        print(f"  MAE:  {results['mean_mae']:.4f} ± {results['std_mae']:.4f}")
        print(f"  R²:   {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
    
    return results


def get_worst_predictions(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    n: int = 20,
    by: str = 'stock',
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Get worst predictions by stock or day.
    
    Args:
        df: DataFrame with predictions and targets
        y_true_col: Column name for true values
        y_pred_col: Column name for predictions
        n: Number of worst items to return
        by: 'stock' or 'day'
        config: Configuration dictionary
        
    Returns:
        DataFrame with worst predictions
    """
    if config is None:
        config = load_config()
    
    if by == 'stock':
        metrics_df = compute_per_stock_metrics(df, y_true_col, y_pred_col, config)
        return metrics_df.nlargest(n, 'rmse')
    elif by == 'day':
        metrics_df = compute_per_day_metrics(df, y_true_col, y_pred_col, config)
        return metrics_df.nlargest(n, 'rmse')
    else:
        raise ValueError(f"Unknown 'by' value: {by}")


def print_cv_summary(results: Dict[str, Any]):
    """
    Print a summary of cross-validation results.
    
    Args:
        results: Dictionary from cross_validate function
    """
    print("\n" + "="*50)
    print("CROSS-VALIDATION SUMMARY")
    print("="*50)
    print(f"RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
    print(f"MAE:  {results['mean_mae']:.4f} ± {results['std_mae']:.4f}")
    print(f"R²:   {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
    print("\nPer-Fold Results:")
    print(results['fold_metrics'].to_string(index=False))
    print("="*50 + "\n")
