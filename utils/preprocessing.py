"""
Preprocessing Utilities for ENS Challenge 60
Missing value handling, scaling, and data transformation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings

warnings.filterwarnings('ignore')

from .data_loader import load_config, get_column_names


def handle_missing_values(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    strategy: str = "interpolate",
    create_nan_features: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Strategy options:
    - "interpolate": Interpolate across periods (axis=1), then fill with 0
    - "zero": Fill all NaN with 0
    - "mean": Fill with column mean
    - "median": Fill with column median
    - "forward": Forward fill within each stock
    
    Args:
        df: DataFrame with missing values
        config: Configuration dictionary
        strategy: Missing value handling strategy
        create_nan_features: Whether to create NaN count features
        verbose: Whether to print information
        
    Returns:
        DataFrame with missing values handled
    """
    if config is None:
        config = load_config()
    
    return_cols, volume_cols = get_column_names(config)
    df = df.copy()
    
    # Count missing values before
    if verbose:
        missing_before = df[return_cols + volume_cols].isnull().sum().sum()
        print(f"Missing values before: {missing_before:,}")
    
    # Create NaN count features BEFORE imputation
    if create_nan_features:
        df['return_nan_count'] = df[return_cols].isnull().sum(axis=1).astype('int8')
        df['volume_nan_count'] = df[volume_cols].isnull().sum(axis=1).astype('int8')
        df['total_nan_count'] = (df['return_nan_count'] + df['volume_nan_count']).astype('int8')
        
        if verbose:
            print(f"Created NaN count features: return_nan_count, volume_nan_count, total_nan_count")
    
    # Apply imputation strategy
    if strategy == "interpolate":
        # Interpolate across periods (axis=1)
        df[return_cols] = df[return_cols].interpolate(axis=1, limit_direction='both')
        df[volume_cols] = df[volume_cols].interpolate(axis=1, limit_direction='both')
        # Fill any remaining NaN with 0
        df[return_cols] = df[return_cols].fillna(0)
        df[volume_cols] = df[volume_cols].fillna(0)
        
    elif strategy == "zero":
        df[return_cols] = df[return_cols].fillna(0)
        df[volume_cols] = df[volume_cols].fillna(0)
        
    elif strategy == "mean":
        for col in return_cols + volume_cols:
            df[col] = df[col].fillna(df[col].mean())
            
    elif strategy == "median":
        for col in return_cols + volume_cols:
            df[col] = df[col].fillna(df[col].median())
            
    elif strategy == "forward":
        # Forward fill within each stock
        id_col = config['features']['id_col']
        df = df.sort_values([id_col, config['features']['day_col']])
        for col in return_cols + volume_cols:
            df[col] = df.groupby(id_col)[col].ffill().bfill()
        df[return_cols + volume_cols] = df[return_cols + volume_cols].fillna(0)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Fill any remaining NaN in extra columns
    for col in config['features']['extra_cols']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Count missing values after
    if verbose:
        missing_after = df[return_cols + volume_cols].isnull().sum().sum()
        print(f"Missing values after: {missing_after:,}")
    
    return df


def preprocess_data(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Full preprocessing pipeline for training and test data.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features (optional)
        config: Configuration dictionary
        verbose: Whether to print information
        
    Returns:
        Tuple of (processed_train, y_train, processed_test)
    """
    if config is None:
        config = load_config()
    
    preprocess_config = config['preprocessing']
    
    if verbose:
        print("\n" + "="*50)
        print("PREPROCESSING PIPELINE")
        print("="*50)
    
    # Handle missing values in training data
    if verbose:
        print("\nProcessing training data...")
    X_train_processed = handle_missing_values(
        X_train,
        config=config,
        strategy=preprocess_config['nan_strategy'],
        create_nan_features=preprocess_config['create_nan_features'],
        verbose=verbose
    )
    
    # Handle missing values in test data
    X_test_processed = None
    if X_test is not None:
        if verbose:
            print("\nProcessing test data...")
        X_test_processed = handle_missing_values(
            X_test,
            config=config,
            strategy=preprocess_config['nan_strategy'],
            create_nan_features=preprocess_config['create_nan_features'],
            verbose=verbose
        )
    
    if verbose:
        print("\nPreprocessing complete!")
        print("="*50 + "\n")
    
    return X_train_processed, y_train, X_test_processed


def scale_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    scaler_type: str = "standard",
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Any]:
    """
    Scale numerical features.
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
        feature_cols: Columns to scale (if None, scales all numeric)
        scaler_type: Type of scaler ("standard", "minmax", "robust")
        config: Configuration dictionary
        
    Returns:
        Tuple of (scaled_train, scaled_test, scaler)
    """
    if config is None:
        config = load_config()
    
    # Select scaler
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Get feature columns to scale
    if feature_cols is None:
        return_cols, volume_cols = get_column_names(config)
        feature_cols = return_cols + volume_cols + config['features']['extra_cols']
    
    # Filter to existing columns
    feature_cols = [col for col in feature_cols if col in X_train.columns]
    
    # Fit and transform training data
    X_train_scaled = X_train.copy()
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    
    # Transform test data
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = X_test.copy()
        X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])
    
    return X_train_scaled, X_test_scaled, scaler


def clip_outliers(
    df: pd.DataFrame,
    columns: List[str],
    lower_percentile: float = 1,
    upper_percentile: float = 99
) -> pd.DataFrame:
    """
    Clip outliers in specified columns.
    
    Args:
        df: DataFrame
        columns: Columns to clip
        lower_percentile: Lower percentile for clipping
        upper_percentile: Upper percentile for clipping
        
    Returns:
        DataFrame with clipped values
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            lower = np.percentile(df[col].dropna(), lower_percentile)
            upper = np.percentile(df[col].dropna(), upper_percentile)
            df[col] = df[col].clip(lower, upper)
    
    return df


def remove_constant_features(
    df: pd.DataFrame,
    threshold: float = 0.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with zero or near-zero variance.
    
    Args:
        df: DataFrame
        threshold: Variance threshold (features with variance <= threshold are removed)
        
    Returns:
        Tuple of (filtered DataFrame, list of removed columns)
    """
    variances = df.var()
    constant_cols = variances[variances <= threshold].index.tolist()
    
    df_filtered = df.drop(columns=constant_cols)
    
    return df_filtered, constant_cols


def get_preprocessing_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get summary of preprocessing changes.
    
    Args:
        df_before: DataFrame before preprocessing
        df_after: DataFrame after preprocessing
        config: Configuration dictionary
        
    Returns:
        Dictionary with preprocessing summary
    """
    if config is None:
        config = load_config()
    
    return_cols, volume_cols = get_column_names(config)
    
    summary = {
        'rows_before': len(df_before),
        'rows_after': len(df_after),
        'cols_before': len(df_before.columns),
        'cols_after': len(df_after.columns),
        'missing_before': df_before[return_cols + volume_cols].isnull().sum().sum(),
        'missing_after': df_after[return_cols + volume_cols].isnull().sum().sum() if return_cols[0] in df_after.columns else 0,
        'new_features': [col for col in df_after.columns if col not in df_before.columns],
    }
    
    return summary