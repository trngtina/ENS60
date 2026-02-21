"""
MLflow Utilities for ENS Challenge 60
Experiment tracking and model logging
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Union
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not installed. Experiment tracking will be disabled.")

from .data_loader import load_config


def setup_mlflow(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Set up MLflow experiment tracking.
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI
        config: Configuration dictionary
        
    Returns:
        True if setup successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Skipping setup.")
        return False
    
    if config is None:
        config = load_config()
    
    mlflow_config = config.get('mlflow', {})
    
    if not mlflow_config.get('enabled', True):
        print("MLflow disabled in config.")
        return False
    
    # Set tracking URI
    uri = tracking_uri or mlflow_config.get('tracking_uri', 'mlruns')
    mlflow.set_tracking_uri(uri)
    
    # Set experiment
    exp_name = experiment_name or mlflow_config.get('experiment_name', 'ENS60_Auction_Volume')
    mlflow.set_experiment(exp_name)
    
    print(f"MLflow setup complete:")
    print(f"  Tracking URI: {uri}")
    print(f"  Experiment: {exp_name}")
    
    return True


def log_experiment(
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model: Optional[Any] = None,
    artifacts: Optional[Dict[str, str]] = None,
    tags: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Log an experiment run to MLflow.
    
    Args:
        run_name: Name of the run
        params: Dictionary of parameters
        metrics: Dictionary of metrics
        model: Model to log (optional)
        artifacts: Dictionary of artifact_name -> file_path (optional)
        tags: Dictionary of tags (optional)
        config: Configuration dictionary
        
    Returns:
        Run ID if successful, None otherwise
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Logging to console instead.")
        print(f"\nRun: {run_name}")
        print(f"Params: {params}")
        print(f"Metrics: {metrics}")
        return None
    
    if config is None:
        config = load_config()
    
    mlflow_config = config.get('mlflow', {})
    
    if not mlflow_config.get('enabled', True):
        return None
    
    try:
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            for key, value in params.items():
                # MLflow has limits on parameter value length
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)[:250]
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    mlflow.log_metric(key, value)
            
            # Log tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Log model
            if model is not None and mlflow_config.get('log_models', True):
                try:
                    mlflow.sklearn.log_model(model, "model")
                except Exception as e:
                    print(f"Warning: Could not log model: {e}")
            
            # Log artifacts
            if artifacts and mlflow_config.get('log_artifacts', True):
                for name, path in artifacts.items():
                    if Path(path).exists():
                        mlflow.log_artifact(path, name)
            
            return run.info.run_id
    
    except Exception as e:
        print(f"Error logging to MLflow: {e}")
        return None


def log_cv_results(
    run_name: str,
    cv_results: Dict[str, Any],
    model_name: str,
    params: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Log cross-validation results to MLflow.
    
    Args:
        run_name: Name of the run
        cv_results: Dictionary from cross_validate function
        model_name: Name of the model
        params: Model parameters
        config: Configuration dictionary
        
    Returns:
        Run ID if successful, None otherwise
    """
    metrics = {
        'cv_rmse_mean': cv_results['mean_rmse'],
        'cv_rmse_std': cv_results['std_rmse'],
        'cv_mae_mean': cv_results['mean_mae'],
        'cv_mae_std': cv_results['std_mae'],
        'cv_r2_mean': cv_results['mean_r2'],
        'cv_r2_std': cv_results['std_r2'],
    }
    
    # Add per-fold metrics
    fold_df = cv_results.get('fold_metrics')
    if fold_df is not None:
        for i, row in fold_df.iterrows():
            metrics[f'fold_{int(row["fold"])}_rmse'] = row['rmse']
    
    all_params = {'model_name': model_name, **params}
    
    tags = {
        'model_type': model_name,
        'cv_folds': str(len(fold_df)) if fold_df is not None else 'unknown',
    }
    
    return log_experiment(
        run_name=run_name,
        params=all_params,
        metrics=metrics,
        tags=tags,
        config=config
    )


def log_feature_importance(
    feature_importance: pd.DataFrame,
    run_id: Optional[str] = None,
    artifact_name: str = "feature_importance"
) -> None:
    """
    Log feature importance to MLflow.
    
    Args:
        feature_importance: DataFrame with feature importance
        run_id: MLflow run ID (if None, logs to active run)
        artifact_name: Name for the artifact
    """
    if not MLFLOW_AVAILABLE:
        return
    
    # Save to temporary file
    temp_path = f"/tmp/{artifact_name}.csv"
    feature_importance.to_csv(temp_path, index=False)
    
    try:
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(temp_path, artifact_name)
        else:
            mlflow.log_artifact(temp_path, artifact_name)
    except Exception as e:
        print(f"Warning: Could not log feature importance: {e}")


def get_best_run(
    experiment_name: Optional[str] = None,
    metric: str = "cv_rmse_mean",
    ascending: bool = True,
    config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Get the best run from an experiment.
    
    Args:
        experiment_name: Name of the experiment
        metric: Metric to sort by
        ascending: Whether lower is better
        config: Configuration dictionary
        
    Returns:
        Dictionary with best run info, or None
    """
    if not MLFLOW_AVAILABLE:
        return None
    
    if config is None:
        config = load_config()
    
    exp_name = experiment_name or config.get('mlflow', {}).get('experiment_name', 'ENS60_Auction_Volume')
    
    try:
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if len(runs) == 0:
            return None
        
        best_run = runs.iloc[0]
        
        return {
            'run_id': best_run['run_id'],
            'run_name': best_run.get('tags.mlflow.runName', 'unknown'),
            'metrics': {col.replace('metrics.', ''): best_run[col] 
                       for col in runs.columns if col.startswith('metrics.')},
            'params': {col.replace('params.', ''): best_run[col] 
                      for col in runs.columns if col.startswith('params.')},
        }
    
    except Exception as e:
        print(f"Error getting best run: {e}")
        return None


def compare_runs(
    experiment_name: Optional[str] = None,
    metric: str = "cv_rmse_mean",
    n_runs: int = 10,
    config: Optional[Dict[str, Any]] = None
) -> Optional[pd.DataFrame]:
    """
    Compare top runs from an experiment.
    
    Args:
        experiment_name: Name of the experiment
        metric: Metric to sort by
        n_runs: Number of runs to return
        config: Configuration dictionary
        
    Returns:
        DataFrame with run comparison, or None
    """
    if not MLFLOW_AVAILABLE:
        return None
    
    if config is None:
        config = load_config()
    
    exp_name = experiment_name or config.get('mlflow', {}).get('experiment_name', 'ENS60_Auction_Volume')
    
    try:
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"],
            max_results=n_runs
        )
        
        if len(runs) == 0:
            return None
        
        # Select relevant columns
        metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
        param_cols = [col for col in runs.columns if col.startswith('params.')]
        
        result = runs[['run_id', 'tags.mlflow.runName'] + metric_cols[:5] + param_cols[:5]].copy()
        result.columns = [col.replace('metrics.', '').replace('params.', '').replace('tags.mlflow.', '') 
                         for col in result.columns]
        
        return result
    
    except Exception as e:
        print(f"Error comparing runs: {e}")
        return None


class ExperimentTracker:
    """
    Simple experiment tracker that works with or without MLflow.
    """
    
    def __init__(
        self,
        experiment_name: str = "ENS60_Auction_Volume",
        use_mlflow: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            use_mlflow: Whether to use MLflow
            config: Configuration dictionary
        """
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.config = config or load_config()
        
        self.runs = []
        
        if self.use_mlflow:
            setup_mlflow(experiment_name, config=self.config)
    
    def log_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        model: Optional[Any] = None
    ) -> Optional[str]:
        """
        Log a run.
        
        Args:
            run_name: Name of the run
            params: Parameters
            metrics: Metrics
            model: Model to log
            
        Returns:
            Run ID if using MLflow, None otherwise
        """
        # Store locally
        self.runs.append({
            'run_name': run_name,
            'params': params,
            'metrics': metrics,
        })
        
        # Log to MLflow
        if self.use_mlflow:
            return log_experiment(
                run_name=run_name,
                params=params,
                metrics=metrics,
                model=model,
                config=self.config
            )
        
        return None
    
    def get_best_run(self, metric: str = "rmse", ascending: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the best run.
        
        Args:
            metric: Metric to sort by
            ascending: Whether lower is better
            
        Returns:
            Best run dictionary
        """
        if not self.runs:
            return None
        
        sorted_runs = sorted(
            self.runs,
            key=lambda x: x['metrics'].get(metric, float('inf')),
            reverse=not ascending
        )
        
        return sorted_runs[0]
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of all runs.
        
        Returns:
            DataFrame with run summary
        """
        if not self.runs:
            return pd.DataFrame()
        
        rows = []
        for run in self.runs:
            row = {'run_name': run['run_name']}
            row.update(run['metrics'])
            rows.append(row)
        
        return pd.DataFrame(rows)