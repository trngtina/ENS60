"""
Data Loading Utilities for ENS Challenge 60
Memory-optimized loading and configuration management
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Literal
import warnings

warnings.filterwarnings('ignore')


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_column_names(config: Dict[str, Any]) -> Tuple[list, list]:
    """
    Generate column names for returns and volumes based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (return_cols, volume_cols)
    """
    n_periods = config['features']['n_periods']
    ret_prefix = config['features']['return_cols_prefix']
    vol_prefix = config['features']['volume_cols_prefix']
    
    return_cols = [f"{ret_prefix}{i}" for i in range(n_periods)]
    volume_cols = [f"{vol_prefix}{i}" for i in range(n_periods)]
    
    return return_cols, volume_cols


def get_optimized_dtypes(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate optimized dtype dictionary for memory-efficient loading.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping column names to dtypes
    """
    return_cols, volume_cols = get_column_names(config)
    
    dtype_dict = {
        config['features']['id_col']: 'int16',
        config['features']['day_col']: 'int16',
    }
    
    # Add extra columns
    for col in config['features']['extra_cols']:
        dtype_dict[col] = 'float32'
    
    # Add return and volume columns
    for col in return_cols + volume_cols:
        dtype_dict[col] = 'float32'
    
    return dtype_dict


def load_data(
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "configs/config.yaml",
    load_test: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load training and test data with memory optimization.
    
    Args:
        config: Configuration dictionary (if None, loads from config_path)
        config_path: Path to configuration file
        load_test: Whether to load test data
        verbose: Whether to print loading information
        
    Returns:
        Tuple of (X_train, y_train, X_test) where X_test is None if load_test=False
    """
    if config is None:
        config = load_config(config_path)
    
    dtype_dict = get_optimized_dtypes(config)
    
    # Load training input
    if verbose:
        print("Loading training input data...")
    X_train = pd.read_csv(
        config['data']['train_input'],
        dtype=dtype_dict,
        compression='gzip'
    )
    
    if verbose:
        print(f"  Shape: {X_train.shape}")
        print(f"  Memory: {X_train.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    # Load training output (target)
    if verbose:
        print("Loading training output data...")
    y_train = pd.read_csv(config['data']['train_output'])
    
    # Rename target column if needed
    if 'TARGET' in y_train.columns:
        y_train = y_train.rename(columns={'TARGET': config['features']['target_col']})
    
    if verbose:
        print(f"  Shape: {y_train.shape}")
    
    # Load test data
    X_test = None
    if load_test:
        if verbose:
            print("Loading test input data...")
        X_test = pd.read_csv(
            config['data']['test_input'],
            dtype=dtype_dict,
            compression='gzip'
        )
        if verbose:
            print(f"  Shape: {X_test.shape}")
            print(f"  Memory: {X_test.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    return X_train, y_train, X_test


def merge_target(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    align_mode: Literal["auto", "id", "position"] = "auto",
    strict: bool = True
) -> pd.DataFrame:
    """
    Merge target values with training features using strict alignment checks.
    
    Alignment modes:
    - "auto": Use ID-based alignment when both frames have "ID", else positional.
    - "id": Force ID-based alignment (preserves X_train row order).
    - "position": Force positional assignment.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training target DataFrame
        config: Configuration dictionary
        align_mode: Target alignment strategy ("auto", "id", or "position")
        strict: Whether to raise errors on alignment inconsistencies
        
    Returns:
        Merged DataFrame with target column
    """
    if config is None:
        config = load_config()
    
    target_col = config['features']['target_col']
    row_id_col = 'ID'
    
    if align_mode not in {"auto", "id", "position"}:
        raise ValueError(f"Unknown align_mode: {align_mode}")
    
    if target_col not in y_train.columns:
        raise ValueError(f"Target column '{target_col}' not found in y_train")
    
    has_input_id = row_id_col in X_train.columns
    has_target_id = row_id_col in y_train.columns
    
    mode = align_mode
    if mode == "auto":
        mode = "id" if has_input_id and has_target_id else "position"
    
    if mode == "id":
        if not (has_input_id and has_target_id):
            msg = "ID-based alignment requested but 'ID' column is missing in X_train or y_train"
            if strict:
                raise ValueError(msg)
            warnings.warn(f"{msg}. Falling back to positional alignment.", RuntimeWarning)
            mode = "position"
        else:
            if strict:
                if X_train[row_id_col].isna().any() or y_train[row_id_col].isna().any():
                    raise ValueError("Null values found in ID column during strict ID alignment")
                if X_train[row_id_col].duplicated().any():
                    raise ValueError("Duplicate IDs found in X_train during strict ID alignment")
                if y_train[row_id_col].duplicated().any():
                    raise ValueError("Duplicate IDs found in y_train during strict ID alignment")
            
            y_target = y_train[[row_id_col, target_col]]
            df = X_train.merge(
                y_target,
                on=row_id_col,
                how='left',
                sort=False
            )
            
            if strict:
                missing_target = df[target_col].isna().sum()
                if missing_target > 0:
                    raise ValueError(
                        f"Strict ID alignment failed: {missing_target} rows in X_train have no matching target"
                    )
                
                missing_in_target = len(pd.Index(X_train[row_id_col]).difference(pd.Index(y_train[row_id_col])))
                extra_in_target = len(pd.Index(y_train[row_id_col]).difference(pd.Index(X_train[row_id_col])))
                if missing_in_target > 0 or extra_in_target > 0:
                    raise ValueError(
                        f"Strict ID alignment mismatch: {missing_in_target} IDs missing in y_train, "
                        f"{extra_in_target} extra IDs in y_train"
                    )
            return df
    
    # Positional alignment
    n_x = len(X_train)
    n_y = len(y_train)
    if strict and n_x != n_y:
        raise ValueError(f"Length mismatch in strict positional alignment: X_train={n_x}, y_train={n_y}")
    
    n_rows = min(n_x, n_y)
    if n_x != n_y and not strict:
        warnings.warn(
            f"Non-strict positional alignment with length mismatch: using first {n_rows} rows",
            RuntimeWarning
        )
    
    df = X_train.iloc[:n_rows].copy()
    df[target_col] = y_train[target_col].iloc[:n_rows].values
    return df


def get_feature_columns(
    config: Optional[Dict[str, Any]] = None,
    include_raw: bool = True,
    include_engineered: bool = True
) -> Dict[str, list]:
    """
    Get lists of feature columns by category.
    
    Args:
        config: Configuration dictionary
        include_raw: Include raw feature columns
        include_engineered: Include engineered feature columns
        
    Returns:
        Dictionary with feature column lists by category
    """
    if config is None:
        config = load_config()
    
    return_cols, volume_cols = get_column_names(config)
    
    feature_dict = {
        'return_cols': return_cols,
        'volume_cols': volume_cols,
        'id_col': config['features']['id_col'],
        'day_col': config['features']['day_col'],
        'extra_cols': config['features']['extra_cols'],
        'target_col': config['features']['target_col'],
    }
    
    # All raw features (excluding target)
    if include_raw:
        feature_dict['raw_features'] = (
            return_cols + 
            volume_cols + 
            config['features']['extra_cols']
        )
    
    return feature_dict


def load_submission_example(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Load submission example file.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Submission example DataFrame
    """
    if config is None:
        config = load_config()
    
    return pd.read_csv(config['data']['submission_example'])


def create_submission(
    predictions: np.ndarray,
    X_test: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    validate_against_example: bool = False
) -> pd.DataFrame:
    """
    Create submission file from predictions using test row IDs.
    
    Args:
        predictions: Array of predictions
        X_test: Test DataFrame (for ID creation)
        config: Configuration dictionary
        output_path: Path to save submission (optional)
        validate_against_example: Whether to validate shape/ID range against example file
        
    Returns:
        Submission DataFrame
    """
    if config is None:
        config = load_config()
    
    pred_arr = np.asarray(predictions).reshape(-1)
    if len(pred_arr) != len(X_test):
        raise ValueError(
            f"Predictions length ({len(pred_arr)}) does not match X_test length ({len(X_test)})"
        )
    
    if 'ID' not in X_test.columns:
        raise ValueError("X_test must contain an 'ID' column to build submission")
    
    test_ids = X_test['ID']
    if test_ids.isna().any():
        raise ValueError("X_test contains null IDs")
    if test_ids.duplicated().any():
        raise ValueError("X_test contains duplicate IDs")
    
    # Create submission DataFrame with true test IDs
    submission = pd.DataFrame({
        'ID': test_ids.values,
        config['features']['target_col']: pred_arr
    })
    
    if validate_against_example:
        example_path = config['data'].get('submission_example')
        if example_path and Path(example_path).exists():
            example = pd.read_csv(example_path, usecols=['ID'])
            if len(example) != len(submission):
                raise ValueError(
                    f"Submission length ({len(submission)}) does not match example length ({len(example)})"
                )
            if submission['ID'].iloc[0] != example['ID'].iloc[0] or submission['ID'].iloc[-1] != example['ID'].iloc[-1]:
                raise ValueError(
                    "Submission ID range does not match example file ID range"
                )
    
    # Save if path provided
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
    
    return submission


def get_data_info(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get summary information about the dataset.
    
    Args:
        df: DataFrame to analyze
        config: Configuration dictionary
        
    Returns:
        Dictionary with data information
    """
    if config is None:
        config = load_config()
    
    return_cols, volume_cols = get_column_names(config)
    id_col = config['features']['id_col']
    day_col = config['features']['day_col']
    
    info = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'n_stocks': df[id_col].nunique() if id_col in df.columns else None,
        'n_days': df[day_col].nunique() if day_col in df.columns else None,
        'day_range': (df[day_col].min(), df[day_col].max()) if day_col in df.columns else None,
        'memory_mb': df.memory_usage(deep=True).sum() / 1e6,
        'missing_returns': df[return_cols].isnull().sum().sum() if return_cols[0] in df.columns else None,
        'missing_volumes': df[volume_cols].isnull().sum().sum() if volume_cols[0] in df.columns else None,
    }
    
    return info


def print_data_info(df: pd.DataFrame, name: str = "Dataset", config: Optional[Dict[str, Any]] = None):
    """
    Print summary information about the dataset.
    
    Args:
        df: DataFrame to analyze
        name: Name of the dataset for display
        config: Configuration dictionary
    """
    info = get_data_info(df, config)
    
    print(f"\n{'='*50}")
    print(f"{name} Information")
    print(f"{'='*50}")
    print(f"Samples:        {info['n_samples']:,}")
    print(f"Features:       {info['n_features']}")
    print(f"Stocks:         {info['n_stocks']:,}" if info['n_stocks'] else "Stocks: N/A")
    print(f"Days:           {info['n_days']:,}" if info['n_days'] else "Days: N/A")
    print(f"Day Range:      {info['day_range']}" if info['day_range'] else "Day Range: N/A")
    print(f"Memory:         {info['memory_mb']:.2f} MB")
    print(f"Missing Returns: {info['missing_returns']:,}" if info['missing_returns'] is not None else "")
    print(f"Missing Volumes: {info['missing_volumes']:,}" if info['missing_volumes'] is not None else "")
    print(f"{'='*50}\n")
