"""
Ensemble Utilities for ENS Challenge 60
Stacking, blending, and out-of-fold predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any, List, Union
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin, clone
import warnings

warnings.filterwarnings('ignore')

from .data_loader import load_config
from .validation import TimeSeriesValidator, compute_metrics


def create_oof_predictions(
    models: List[BaseEstimator],
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    validator: Optional[TimeSeriesValidator] = None,
    n_splits: int = 5,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, List[List[BaseEstimator]]]:
    """
    Create out-of-fold predictions for multiple models.
    
    Args:
        models: List of model instances
        X: Feature DataFrame
        y: Target Series/array
        validator: TimeSeriesValidator instance
        n_splits: Number of CV splits
        config: Configuration dictionary
        verbose: Whether to print progress
        
    Returns:
        Tuple of (oof_predictions array [n_samples, n_models], trained_models)
    """
    if config is None:
        config = load_config()
    
    if validator is None:
        validator = TimeSeriesValidator(n_splits=n_splits, config=config)
    
    if isinstance(y, pd.Series):
        y = y.values
    
    n_samples = len(X)
    n_models = len(models)
    
    # Initialize OOF predictions
    oof_predictions = np.zeros((n_samples, n_models))
    oof_mask = np.zeros(n_samples, dtype=bool)
    
    # Store trained models for each fold
    trained_models = [[] for _ in range(n_models)]
    
    for fold, (train_idx, val_idx) in enumerate(validator.split(X)):
        if verbose:
            print(f"\nFold {fold + 1}/{validator.get_n_splits()}")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        for i, model in enumerate(models):
            if verbose:
                print(f"  Training model {i + 1}/{n_models}...")
            
            # Clone model to avoid fitting the same instance
            model_clone = clone(model)
            
            # Train model
            if hasattr(model_clone, 'fit'):
                # Check if model accepts validation data
                try:
                    model_clone.fit(X_train, y_train, X_val=X_val, y_val=y_val)
                except TypeError:
                    model_clone.fit(X_train, y_train)
            
            # Predict on validation
            val_preds = model_clone.predict(X_val)
            oof_predictions[val_idx, i] = val_preds
            
            # Store trained model
            trained_models[i].append(model_clone)
            
            # Compute metrics
            if verbose:
                rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))
                print(f"    RMSE: {rmse:.4f}")
        
        oof_mask[val_idx] = True
    
    return oof_predictions, trained_models


def stack_models(
    oof_predictions: np.ndarray,
    y: Union[pd.Series, np.ndarray],
    meta_model: Optional[BaseEstimator] = None,
    meta_alpha: float = 1.0
) -> BaseEstimator:
    """
    Train a meta-model on out-of-fold predictions.
    
    Args:
        oof_predictions: OOF predictions array [n_samples, n_models]
        y: Target values
        meta_model: Meta-model instance (default: Ridge)
        meta_alpha: Ridge regularization parameter
        
    Returns:
        Trained meta-model
    """
    if isinstance(y, pd.Series):
        y = y.values
    
    if meta_model is None:
        meta_model = Ridge(alpha=meta_alpha)
    
    # Filter out samples without OOF predictions
    valid_mask = ~np.isnan(oof_predictions).any(axis=1)
    
    meta_model.fit(oof_predictions[valid_mask], y[valid_mask])
    
    return meta_model


def predict_with_ensemble(
    trained_models: List[List[BaseEstimator]],
    X: pd.DataFrame,
    meta_model: Optional[BaseEstimator] = None,
    weights: Optional[List[float]] = None,
    method: str = 'stacking'
) -> np.ndarray:
    """
    Make predictions using ensemble of trained models.
    
    Args:
        trained_models: List of lists of trained models (from create_oof_predictions)
        X: Features to predict on
        meta_model: Trained meta-model (for stacking)
        weights: Model weights (for weighted average)
        method: 'stacking' or 'weighted_average'
        
    Returns:
        Ensemble predictions
    """
    n_models = len(trained_models)
    
    # Get predictions from each model (average across folds)
    model_predictions = np.zeros((len(X), n_models))
    
    for i, fold_models in enumerate(trained_models):
        fold_preds = np.zeros((len(X), len(fold_models)))
        
        for j, model in enumerate(fold_models):
            fold_preds[:, j] = model.predict(X)
        
        # Average predictions across folds
        model_predictions[:, i] = fold_preds.mean(axis=1)
    
    if method == 'stacking':
        if meta_model is None:
            raise ValueError("meta_model required for stacking")
        return meta_model.predict(model_predictions)
    
    elif method == 'weighted_average':
        if weights is None:
            weights = [1.0 / n_models] * n_models
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        return (model_predictions * weights).sum(axis=1)
    
    else:
        raise ValueError(f"Unknown method: {method}")


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble with time series cross-validation.
    """
    
    def __init__(
        self,
        base_models: List[BaseEstimator],
        meta_model: Optional[BaseEstimator] = None,
        meta_alpha: float = 1.0,
        n_splits: int = 5,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of base model instances
            meta_model: Meta-model instance (default: Ridge)
            meta_alpha: Ridge regularization parameter
            n_splits: Number of CV splits for OOF predictions
            config: Configuration dictionary
        """
        self.base_models = base_models
        self.meta_model = meta_model or Ridge(alpha=meta_alpha)
        self.meta_alpha = meta_alpha
        self.n_splits = n_splits
        self.config = config
        
        self.trained_models_ = None
        self.meta_model_ = None
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self
        """
        # Create OOF predictions
        oof_predictions, self.trained_models_ = create_oof_predictions(
            models=self.base_models,
            X=X,
            y=y,
            n_splits=self.n_splits,
            config=self.config,
            verbose=False
        )
        
        # Train meta-model
        self.meta_model_ = stack_models(
            oof_predictions=oof_predictions,
            y=y,
            meta_model=clone(self.meta_model),
            meta_alpha=self.meta_alpha
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        return predict_with_ensemble(
            trained_models=self.trained_models_,
            X=X,
            meta_model=self.meta_model_,
            method='stacking'
        )


class WeightedAverageEnsemble(BaseEstimator, RegressorMixin):
    """
    Weighted average ensemble.
    """
    
    def __init__(
        self,
        base_models: List[BaseEstimator],
        weights: Optional[List[float]] = None,
        optimize_weights: bool = False,
        n_splits: int = 5,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize weighted average ensemble.
        
        Args:
            base_models: List of base model instances
            weights: Model weights (if None, uses equal weights)
            optimize_weights: Whether to optimize weights on validation
            n_splits: Number of CV splits
            config: Configuration dictionary
        """
        self.base_models = base_models
        self.weights = weights
        self.optimize_weights = optimize_weights
        self.n_splits = n_splits
        self.config = config
        
        self.trained_models_ = None
        self.weights_ = None
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'WeightedAverageEnsemble':
        """
        Fit the ensemble.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        # Create OOF predictions
        oof_predictions, self.trained_models_ = create_oof_predictions(
            models=self.base_models,
            X=X,
            y=y,
            n_splits=self.n_splits,
            config=self.config,
            verbose=False
        )
        
        if self.optimize_weights:
            # Optimize weights using OOF predictions
            self.weights_ = self._optimize_weights(oof_predictions, y)
        else:
            self.weights_ = self.weights or [1.0 / len(self.base_models)] * len(self.base_models)
        
        return self
    
    def _optimize_weights(self, oof_predictions: np.ndarray, y: np.ndarray) -> List[float]:
        """
        Optimize ensemble weights using OOF predictions.
        
        Uses simple grid search over weight combinations.
        """
        from itertools import product
        
        n_models = oof_predictions.shape[1]
        valid_mask = ~np.isnan(oof_predictions).any(axis=1)
        
        oof_valid = oof_predictions[valid_mask]
        y_valid = y[valid_mask]
        
        best_rmse = float('inf')
        best_weights = [1.0 / n_models] * n_models
        
        # Grid search over weights
        weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        for weights in product(weight_options, repeat=n_models):
            if sum(weights) == 0:
                continue
            
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            preds = (oof_valid * weights).sum(axis=1)
            rmse = np.sqrt(np.mean((y_valid - preds) ** 2))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = weights.tolist()
        
        return best_weights
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        return predict_with_ensemble(
            trained_models=self.trained_models_,
            X=X,
            weights=self.weights_,
            method='weighted_average'
        )


def blend_predictions(
    predictions: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Blend predictions from multiple models.
    
    Args:
        predictions: Dictionary of model_name -> predictions
        weights: Dictionary of model_name -> weight
        
    Returns:
        Blended predictions
    """
    if weights is None:
        weights = {name: 1.0 / len(predictions) for name in predictions}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {name: w / total_weight for name, w in weights.items()}
    
    # Blend
    blended = np.zeros_like(list(predictions.values())[0])
    for name, preds in predictions.items():
        blended += weights.get(name, 0) * preds
    
    return blended


def evaluate_ensemble(
    oof_predictions: np.ndarray,
    y: Union[pd.Series, np.ndarray],
    method: str = 'stacking',
    meta_alpha: float = 1.0,
    weights: Optional[List[float]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate ensemble performance using OOF predictions.
    
    Args:
        oof_predictions: OOF predictions array
        y: Target values
        method: 'stacking' or 'weighted_average'
        meta_alpha: Ridge alpha for stacking
        weights: Weights for weighted average
        verbose: Whether to print results
        
    Returns:
        Dictionary with evaluation results
    """
    if isinstance(y, pd.Series):
        y = y.values
    
    valid_mask = ~np.isnan(oof_predictions).any(axis=1)
    oof_valid = oof_predictions[valid_mask]
    y_valid = y[valid_mask]
    
    n_models = oof_predictions.shape[1]
    
    # Individual model performance
    individual_metrics = []
    for i in range(n_models):
        metrics = compute_metrics(y_valid, oof_valid[:, i])
        metrics['model_idx'] = i
        individual_metrics.append(metrics)
    
    # Ensemble performance
    if method == 'stacking':
        meta_model = Ridge(alpha=meta_alpha)
        meta_model.fit(oof_valid, y_valid)
        ensemble_preds = meta_model.predict(oof_valid)
    else:
        if weights is None:
            weights = [1.0 / n_models] * n_models
        weights = np.array(weights)
        weights = weights / weights.sum()
        ensemble_preds = (oof_valid * weights).sum(axis=1)
    
    ensemble_metrics = compute_metrics(y_valid, ensemble_preds)
    
    results = {
        'individual_metrics': pd.DataFrame(individual_metrics),
        'ensemble_metrics': ensemble_metrics,
        'method': method,
    }
    
    if verbose:
        print("\n" + "="*50)
        print("ENSEMBLE EVALUATION")
        print("="*50)
        print("\nIndividual Model Performance:")
        print(results['individual_metrics'].to_string(index=False))
        print(f"\nEnsemble ({method}) Performance:")
        print(f"  RMSE: {ensemble_metrics['rmse']:.4f}")
        print(f"  MAE:  {ensemble_metrics['mae']:.4f}")
        print(f"  R²:   {ensemble_metrics['r2']:.4f}")
        print("="*50)
    
    return results