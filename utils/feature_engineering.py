"""
Feature Engineering Utilities for ENS Challenge 60
Statistical aggregations, temporal features, and domain-specific features
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

from .data_loader import load_config, get_column_names


def create_aggregation_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create statistical aggregation features from returns and volumes.
    
    Features created:
    - Returns: min, max, std, median, sum, mean, range
    - Volumes: min, max, std, median, mean, skew, kurt
    
    Args:
        df: DataFrame with return and volume columns
        config: Configuration dictionary
        verbose: Whether to print information
        
    Returns:
        DataFrame with aggregation features added
    """
    if config is None:
        config = load_config()
    
    return_cols, volume_cols = get_column_names(config)
    df = df.copy()
    
    if verbose:
        print("Creating aggregation features...")
    
    # Return aggregations
    agg_config = config['feature_engineering']['aggregations']
    
    if 'min' in agg_config['returns']:
        df['min_ret'] = df[return_cols].min(axis=1)
    if 'max' in agg_config['returns']:
        df['max_ret'] = df[return_cols].max(axis=1)
    if 'std' in agg_config['returns']:
        df['std_ret'] = df[return_cols].std(axis=1)
    if 'median' in agg_config['returns']:
        df['median_ret'] = df[return_cols].median(axis=1)
    if 'sum' in agg_config['returns']:
        df['sum_ret'] = df[return_cols].sum(axis=1)
    if 'mean' in agg_config['returns']:
        df['mean_ret'] = df[return_cols].mean(axis=1)
    
    # Range feature
    if 'min' in agg_config['returns'] and 'max' in agg_config['returns']:
        df['range_ret'] = df['max_ret'] - df['min_ret']
    
    # Volume aggregations
    if 'min' in agg_config['volumes']:
        df['min_vol'] = df[volume_cols].min(axis=1)
    if 'max' in agg_config['volumes']:
        df['max_vol'] = df[volume_cols].max(axis=1)
    if 'std' in agg_config['volumes']:
        df['std_vol'] = df[volume_cols].std(axis=1)
    if 'median' in agg_config['volumes']:
        df['median_vol'] = df[volume_cols].median(axis=1)
    if 'mean' in agg_config['volumes']:
        df['mean_vol'] = df[volume_cols].mean(axis=1)
    if 'skew' in agg_config['volumes']:
        df['skew_vol'] = df[volume_cols].skew(axis=1)
    if 'kurt' in agg_config['volumes']:
        df['kurt_vol'] = df[volume_cols].kurt(axis=1)
    
    if verbose:
        new_cols = [col for col in df.columns if col.endswith('_ret') or col.endswith('_vol')]
        print(f"  Created {len(new_cols)} aggregation features")
    
    return df


def create_domain_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create domain-specific features based on financial knowledge.
    
    Features:
    - End-of-day volume concentration
    - Volume-price interaction
    - Return momentum (late vs early)
    - Volume skew and peak location
    
    Args:
        df: DataFrame with return and volume columns
        config: Configuration dictionary
        verbose: Whether to print information
        
    Returns:
        DataFrame with domain features added
    """
    if config is None:
        config = load_config()
    
    return_cols, volume_cols = get_column_names(config)
    domain_config = config['feature_engineering']['domain_features']
    df = df.copy()
    
    if verbose:
        print("Creating domain-specific features...")
    
    n_features_before = len(df.columns)
    
    # End-of-day volume concentration (last 10 periods)
    if domain_config.get('eod_concentration', True):
        eod_cols = volume_cols[-10:]  # Last 10 periods
        df['eod_vol_concentration'] = df[eod_cols].sum(axis=1)
        
        # Also create start-of-day concentration
        sod_cols = volume_cols[:10]  # First 10 periods
        df['sod_vol_concentration'] = df[sod_cols].sum(axis=1)
        
        # Mid-day concentration
        mid_cols = volume_cols[25:36]  # Middle periods
        df['mid_vol_concentration'] = df[mid_cols].sum(axis=1)
    
    # Volume-price interaction (volatility × volume)
    if domain_config.get('vol_price_interaction', True):
        if 'std_ret' in df.columns and 'median_vol' in df.columns:
            df['vol_price_interaction'] = df['std_ret'] * df['median_vol']
        else:
            df['vol_price_interaction'] = df[return_cols].std(axis=1) * df[volume_cols].median(axis=1)
    
    # Return momentum (late periods vs early periods)
    if domain_config.get('return_momentum', True):
        early_ret_cols = return_cols[:20]  # First 20 periods
        late_ret_cols = return_cols[-20:]  # Last 20 periods
        
        df['early_return'] = df[early_ret_cols].mean(axis=1)
        df['late_return'] = df[late_ret_cols].mean(axis=1)
        df['return_momentum'] = df['late_return'] - df['early_return']
    
    # Volume skew (already computed in aggregations, but add peak location)
    if domain_config.get('vol_skew', True):
        # Find which period has the maximum volume
        df['vol_peak_period'] = df[volume_cols].values.argmax(axis=1)
        
        # Normalize to 0-1 range
        df['vol_peak_location'] = df['vol_peak_period'] / (len(volume_cols) - 1)
    
    if verbose:
        n_features_after = len(df.columns)
        print(f"  Created {n_features_after - n_features_before} domain features")
    
    return df


def create_nlv_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create NLV interaction features.
    
    NLV is the most predictive feature (62% R²), so we create
    interaction terms to capture non-linear relationships.
    
    Args:
        df: DataFrame with NLV column
        config: Configuration dictionary
        verbose: Whether to print information
        
    Returns:
        DataFrame with NLV features added
    """
    if config is None:
        config = load_config()
    
    df = df.copy()
    id_col = config['features']['id_col']
    day_col = config['features']['day_col']
    
    if verbose:
        print("Creating NLV interaction features...")
    
    if 'NLV' not in df.columns:
        if verbose:
            print("  NLV column not found, skipping...")
        return df
    
    # NLV × median_vol interaction
    if 'median_vol' in df.columns:
        df['NLV_x_median_vol'] = df['NLV'] * df['median_vol']
    
    # NLV × std_ret interaction
    if 'std_ret' in df.columns:
        df['NLV_x_std_ret'] = df['NLV'] * df['std_ret']
    
    # Per-stock NLV statistics
    df['NLV_per_pid_mean'] = df.groupby(id_col)['NLV'].transform('mean')
    df['NLV_per_pid_std'] = df.groupby(id_col)['NLV'].transform('std')
    df['NLV_deviation'] = df['NLV'] - df['NLV_per_pid_mean']
    df['NLV_zscore'] = df['NLV_deviation'] / (df['NLV_per_pid_std'] + 1e-8)
    
    # NLV rank within day
    df['NLV_rank_in_day'] = df.groupby(day_col)['NLV'].rank(pct=True)
    
    if verbose:
        print("  Created 7 NLV interaction features")
    
    return df


def create_temporal_features(
    df: pd.DataFrame,
    target_col: str = 'target',
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create temporal/lag features for time series modeling.
    
    WARNING: Only use on training data, and be careful about leakage!
    
    Args:
        df: DataFrame sorted by (pid, day)
        target_col: Name of target column
        config: Configuration dictionary
        verbose: Whether to print information
        
    Returns:
        DataFrame with temporal features added
    """
    if config is None:
        config = load_config()
    
    temporal_config = config['feature_engineering']
    id_col = config['features']['id_col']
    day_col = config['features']['day_col']
    
    df = df.copy()
    
    # Sort by stock and day
    df = df.sort_values([id_col, day_col])
    
    if verbose:
        print("Creating temporal features...")
    
    n_features_before = len(df.columns)
    
    # Lag features
    if temporal_config['lag_features']['enabled']:
        lags = temporal_config['lag_features']['lags']
        
        for lag in lags:
            if target_col in df.columns:
                df[f'target_lag_{lag}'] = df.groupby(id_col)[target_col].shift(lag)
            
            # Also create lags for key aggregated features
            if 'median_vol' in df.columns:
                df[f'median_vol_lag_{lag}'] = df.groupby(id_col)['median_vol'].shift(lag)
            if 'std_ret' in df.columns:
                df[f'std_ret_lag_{lag}'] = df.groupby(id_col)['std_ret'].shift(lag)
    
    # Rolling features
    if temporal_config['rolling_features']['enabled']:
        windows = temporal_config['rolling_features']['windows']
        
        for window in windows:
            if target_col in df.columns:
                df[f'target_rolling_mean_{window}'] = df.groupby(id_col)[target_col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
                df[f'target_rolling_std_{window}'] = df.groupby(id_col)[target_col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).std()
                )
    
    if verbose:
        n_features_after = len(df.columns)
        print(f"  Created {n_features_after - n_features_before} temporal features")
    
    return df


def create_non_leaky_temporal_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    history_df: Optional[pd.DataFrame] = None,
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create inference-safe temporal features using only past information.
    
    Features are generated per stock (`pid`) with shift(1) semantics to avoid
    using current-row information from the same series window.
    
    Args:
        df: Current dataset (train or test)
        config: Configuration dictionary
        history_df: Optional past dataframe used as history context (e.g., train when building test)
        lags: Lag steps to create
        windows: Rolling window sizes for lagged rolling means
        verbose: Whether to print information
    
    Returns:
        DataFrame with non-leaky temporal features added
    """
    if config is None:
        config = load_config()
    
    id_col = config['features']['id_col']
    day_col = config['features']['day_col']
    lags = lags or [1, 3, 7]
    windows = windows or [3, 7, 14]
    
    df_out = df.copy()
    n_features_before = len(df_out.columns)
    
    # Candidate columns available at inference time
    candidate_cols = [col for col in ['NLV', 'median_vol', 'std_ret'] if col in df_out.columns]
    if not candidate_cols:
        if verbose:
            print("No candidate columns found for non-leaky temporal features; skipping")
        return df_out
    
    if verbose:
        print("Creating non-leaky temporal features...")
    
    current_sorted = df_out.sort_values([id_col, day_col]).copy()
    current_sorted['__original_index'] = current_sorted.index
    current_sorted['__is_current'] = True
    
    if history_df is not None:
        # Keep only compatible history columns needed for temporal construction
        history_cols = [id_col, day_col] + [col for col in candidate_cols if col in history_df.columns]
        history_part = history_df[history_cols].copy()
        history_part = history_part.sort_values([id_col, day_col])
        history_part['__is_current'] = False
        full = pd.concat([history_part, current_sorted], axis=0, ignore_index=False, sort=False)
    else:
        full = current_sorted.copy()
    
    full = full.sort_values([id_col, day_col]).copy()
    
    created_cols = []
    for base_col in candidate_cols:
        # Lag features
        for lag in lags:
            feat_col = f'{base_col}_lag_{lag}'
            full[feat_col] = full.groupby(id_col)[base_col].shift(lag)
            created_cols.append(feat_col)
        
        # Rolling mean features from lagged series
        lagged_col = full.groupby(id_col)[base_col].shift(1)
        for window in windows:
            feat_col = f'{base_col}_roll_mean_{window}'
            full[feat_col] = lagged_col.groupby(full[id_col]).transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            created_cols.append(feat_col)
    
    # Map back current rows in original order
    current_only = full[full['__is_current'] == True].copy()
    current_only = current_only.sort_values('__original_index')
    
    for col in created_cols:
        df_out[col] = current_only[col].values
    
    # Fill NaNs from insufficient history with 0
    if created_cols:
        df_out[created_cols] = df_out[created_cols].fillna(0)
    
    if verbose:
        n_features_after = len(df_out.columns)
        print(f"  Created {n_features_after - n_features_before} non-leaky temporal features")
    
    return df_out


def create_day_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create day-level aggregate features (market-wide regime).
    
    Args:
        df: DataFrame with day column
        config: Configuration dictionary
        verbose: Whether to print information
        
    Returns:
        DataFrame with day features added
    """
    if config is None:
        config = load_config()
    
    day_col = config['features']['day_col']
    df = df.copy()
    
    if verbose:
        print("Creating day-level features...")
    
    n_features_before = len(df.columns)
    
    # Day-level aggregations
    if 'sum_ret' in df.columns:
        df['day_mean_sum_ret'] = df.groupby(day_col)['sum_ret'].transform('mean')
        df['day_std_sum_ret'] = df.groupby(day_col)['sum_ret'].transform('std')
    
    if 'median_vol' in df.columns:
        df['day_mean_median_vol'] = df.groupby(day_col)['median_vol'].transform('mean')
        df['day_std_median_vol'] = df.groupby(day_col)['median_vol'].transform('std')
    
    if 'std_ret' in df.columns:
        df['day_mean_std_ret'] = df.groupby(day_col)['std_ret'].transform('mean')
        # Market volatility indicator
        df['market_volatility'] = df['day_mean_std_ret']
    
    # Stock count per day (should be ~900, but might vary)
    df['stocks_per_day'] = df.groupby(day_col)[config['features']['id_col']].transform('count')
    
    if verbose:
        n_features_after = len(df.columns)
        print(f"  Created {n_features_after - n_features_before} day-level features")
    
    return df


def create_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    include_aggregations: bool = True,
    include_domain: bool = True,
    include_nlv: bool = True,
    include_day: bool = True,
    include_non_leaky_temporal: bool = False,
    temporal_history_df: Optional[pd.DataFrame] = None,
    include_temporal: bool = False,
    target_col: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        include_aggregations: Include statistical aggregations
        include_domain: Include domain-specific features
        include_nlv: Include NLV interaction features
        include_day: Include day-level features
        include_non_leaky_temporal: Include inference-safe lag/rolling features
        temporal_history_df: Optional history dataframe used when creating non-leaky temporal features
        include_temporal: Include temporal/lag features (careful with leakage!)
        target_col: Target column name (required for temporal features)
        verbose: Whether to print information
        
    Returns:
        DataFrame with all engineered features
    """
    if config is None:
        config = load_config()
    
    if verbose:
        print("\n" + "="*50)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*50)
        print(f"Initial features: {len(df.columns)}")
    
    # Statistical aggregations
    if include_aggregations:
        df = create_aggregation_features(df, config, verbose)
    
    # Domain-specific features
    if include_domain:
        df = create_domain_features(df, config, verbose)
    
    # Inference-safe temporal features (does not use target)
    if include_non_leaky_temporal:
        df = create_non_leaky_temporal_features(
            df=df,
            config=config,
            history_df=temporal_history_df,
            verbose=verbose
        )
    
    # NLV interaction features
    if include_nlv:
        df = create_nlv_features(df, config, verbose)
    
    # Day-level features
    if include_day:
        df = create_day_features(df, config, verbose)
    
    # Temporal features (use with caution!)
    if include_temporal:
        if target_col is None:
            target_col = config['features']['target_col']
        df = create_temporal_features(df, target_col, config, verbose)
    
    if verbose:
        print(f"\nFinal features: {len(df.columns)}")
        print("="*50 + "\n")
    
    return df


def get_feature_list(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    exclude_raw: bool = False,
    exclude_target: bool = True,
    include_id: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Get list of feature columns for modeling.
    
    Args:
        df: DataFrame with features
        config: Configuration dictionary
        exclude_raw: Whether to exclude raw return/volume columns
        exclude_target: Whether to exclude target column
        include_id: Whether to include stock ID (pid) as a categorical feature
        
    Returns:
        Tuple of (feature_columns, categorical_columns)
    """
    if config is None:
        config = load_config()
    
    return_cols, volume_cols = get_column_names(config)
    id_col = config['features']['id_col']
    day_col = config['features']['day_col']
    target_col = config['features']['target_col']
    
    # Columns to always exclude (not features)
    exclude_cols = {
        day_col,           # Day is for temporal splitting, not prediction
        'ID',              # Row identifier from target file
        'index',           # Pandas index if present
    }
    
    # Start with all columns
    feature_cols = list(df.columns)
    
    # Remove excluded columns
    feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    # Handle stock ID (pid) column - should be categorical, not numeric
    categorical_cols = []
    if include_id:
        if id_col in feature_cols:
            categorical_cols.append(id_col)
            # Keep pid in feature_cols for LightGBM (it handles categoricals)
    else:
        # Remove pid entirely if not using as categorical
        feature_cols = [col for col in feature_cols if col != id_col]
    
    # Remove target
    if exclude_target and target_col in feature_cols:
        feature_cols.remove(target_col)
    
    # Remove raw columns if requested
    if exclude_raw:
        feature_cols = [col for col in feature_cols if col not in return_cols + volume_cols]
    
    return feature_cols, categorical_cols


def get_engineered_feature_names() -> List[str]:
    """
    Get list of all possible engineered feature names.
    
    Returns:
        List of engineered feature names
    """
    return [
        # Aggregation features
        'min_ret', 'max_ret', 'std_ret', 'median_ret', 'sum_ret', 'mean_ret', 'range_ret',
        'min_vol', 'max_vol', 'std_vol', 'median_vol', 'mean_vol', 'skew_vol', 'kurt_vol',
        
        # Domain features
        'eod_vol_concentration', 'sod_vol_concentration', 'mid_vol_concentration',
        'vol_price_interaction', 'early_return', 'late_return', 'return_momentum',
        'vol_peak_period', 'vol_peak_location',
        
        # NLV features
        'NLV_x_median_vol', 'NLV_x_std_ret', 'NLV_per_pid_mean', 'NLV_per_pid_std',
        'NLV_deviation', 'NLV_zscore', 'NLV_rank_in_day',
        
        # Day features
        'day_mean_sum_ret', 'day_std_sum_ret', 'day_mean_median_vol', 'day_std_median_vol',
        'day_mean_std_ret', 'market_volatility', 'stocks_per_day',
        
        # NaN features
        'return_nan_count', 'volume_nan_count', 'total_nan_count',
    ]
