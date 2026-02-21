"""
Model Definitions for ENS Challenge 60
Linear models, LightGBM, and two-stage models
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any, List, Union
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.base import BaseEstimator, RegressorMixin
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

from .data_loader import load_config, get_column_names


class TwoStageModel(BaseEstimator, RegressorMixin):
    """
    Two-stage model: Linear regression + LightGBM on residuals.
    
    This is the official CFM benchmark approach:
    1. Stage 1: Linear regression on numerical features
    2. Stage 2: LightGBM on residuals with categorical features
    """
    
    def __init__(
        self,
        linear_alpha: float = 1.0,
        lgb_params: Optional[Dict[str, Any]] = None,
        categorical_features: Optional[List[str]] = None,
        early_stopping_rounds: int = 50,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the two-stage model.
        
        Args:
            linear_alpha: Ridge regularization parameter
            lgb_params: LightGBM parameters
            categorical_features: List of categorical feature names
            early_stopping_rounds: Early stopping rounds for LightGBM
            config: Configuration dictionary
        """
        self.linear_alpha = linear_alpha
        self.lgb_params = lgb_params
        self.categorical_features = categorical_features
        self.early_stopping_rounds = early_stopping_rounds
        self.config = config if config is not None else load_config()
        
        # Initialize models
        self.linear_model = Ridge(alpha=linear_alpha)
        self.lgb_model = None
        
        # Default LightGBM parameters
        if self.lgb_params is None:
            self.lgb_params = self.config['models']['lightgbm']['params'].copy()
    
    def _get_numerical_features(self, X: pd.DataFrame) -> List[str]:
        """Get list of numerical feature columns."""
        cat_cols = self.categorical_features or []
        return [col for col in X.columns if col not in cat_cols]
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> 'TwoStageModel':
        """
        Fit the two-stage model.
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features (optional, for early stopping)
            y_val: Validation target (optional)
            
        Returns:
            self
        """
        # Convert to numpy if needed
        if isinstance(y, pd.Series):
            y = y.values
        
        # Get numerical features
        numerical_cols = self._get_numerical_features(X)
        
        # Stage 1: Linear regression on numerical features
        X_numerical = X[numerical_cols].values
        self.linear_model.fit(X_numerical, y)
        linear_preds = self.linear_model.predict(X_numerical)
        
        # Compute residuals
        residuals = y - linear_preds
        
        # Stage 2: LightGBM on residuals
        self.lgb_model = lgb.LGBMRegressor(**self.lgb_params)
        
        # Prepare categorical features
        cat_features = self.categorical_features or []
        cat_features = [col for col in cat_features if col in X.columns]
        
        # Fit LightGBM
        if X_val is not None and y_val is not None:
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            
            # Compute validation residuals
            X_val_numerical = X_val[numerical_cols].values
            linear_preds_val = self.linear_model.predict(X_val_numerical)
            residuals_val = y_val - linear_preds_val
            
            self.lgb_model.fit(
                X,
                residuals,
                eval_set=[(X_val, residuals_val)],
                categorical_feature=cat_features if cat_features else 'auto',
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            )
        else:
            self.lgb_model.fit(
                X,
                residuals,
                categorical_feature=cat_features if cat_features else 'auto'
            )
        
        # Store feature names
        self.numerical_cols_ = numerical_cols
        self.feature_names_ = list(X.columns)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        # Stage 1: Linear predictions
        X_numerical = X[self.numerical_cols_].values
        linear_preds = self.linear_model.predict(X_numerical)
        
        # Stage 2: Residual predictions
        residual_preds = self.lgb_model.predict(X)
        
        # Combine
        return linear_preds + residual_preds
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from LightGBM model.
        
        Returns:
            DataFrame with feature importance
        """
        if self.lgb_model is None:
            raise ValueError("Model not fitted yet")
        
        importance = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': self.lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


def train_linear(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    alpha: float = 1.0,
    model_type: str = 'ridge'
) -> Tuple[Any, np.ndarray]:
    """
    Train a linear model.
    
    Args:
        X_train: Training features
        y_train: Training target
        alpha: Regularization parameter
        model_type: 'ridge', 'linear', or 'elasticnet'
        
    Returns:
        Tuple of (model, predictions)
    """
    if model_type == 'ridge':
        model = Ridge(alpha=alpha)
    elif model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'elasticnet':
        model = ElasticNet(alpha=alpha)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    
    return model, predictions


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    params: Optional[Dict[str, Any]] = None,
    categorical_features: Optional[List[str]] = None,
    early_stopping_rounds: int = 50,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[lgb.LGBMRegressor, np.ndarray]:
    """
    Train a LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        params: LightGBM parameters
        categorical_features: List of categorical feature names
        early_stopping_rounds: Early stopping rounds
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, predictions)
    """
    if config is None:
        config = load_config()
    
    if params is None:
        params = config['models']['lightgbm']['params'].copy()
    
    model = lgb.LGBMRegressor(**params)
    
    # Prepare categorical features
    cat_features = categorical_features or []
    cat_features = [col for col in cat_features if col in X_train.columns]
    
    # Fit model
    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_features if cat_features else 'auto',
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )
    else:
        model.fit(
            X_train,
            y_train,
            categorical_feature=cat_features if cat_features else 'auto'
        )
    
    predictions = model.predict(X_train)
    
    return model, predictions


class SimpleLinearModel(BaseEstimator, RegressorMixin):
    """
    Simple linear model wrapper for sklearn compatibility.
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)


class LightGBMModel(BaseEstimator, RegressorMixin):
    """
    LightGBM model wrapper for sklearn compatibility.
    """
    
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        categorical_features: Optional[List[str]] = None,
        early_stopping_rounds: int = 50
    ):
        self.params = params or {}
        self.categorical_features = categorical_features
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
    
    def fit(self, X, y, X_val=None, y_val=None):
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 45,
            'max_depth': 7,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'random_state': 42,
            'verbose': -1
        }
        default_params.update(self.params)
        
        self.model = lgb.LGBMRegressor(**default_params)
        
        cat_features = self.categorical_features or []
        cat_features = [col for col in cat_features if col in X.columns]
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                categorical_feature=cat_features if cat_features else 'auto',
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            )
        else:
            self.model.fit(
                X, y,
                categorical_feature=cat_features if cat_features else 'auto'
            )
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_


def get_model(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseEstimator:
    """
    Get a model by name.
    
    Args:
        model_name: Name of the model ('linear', 'lightgbm', 'two_stage')
        config: Configuration dictionary
        **kwargs: Additional model parameters
        
    Returns:
        Model instance
    """
    if config is None:
        config = load_config()
    
    if model_name == 'linear':
        alpha = kwargs.get('alpha', config['models']['linear']['alpha'])
        return SimpleLinearModel(alpha=alpha)
    
    elif model_name == 'lightgbm':
        params = kwargs.get('params', config['models']['lightgbm']['params'])
        cat_features = kwargs.get('categorical_features', ['pid'])
        early_stopping = kwargs.get('early_stopping_rounds', 
                                    config['models']['lightgbm']['early_stopping_rounds'])
        return LightGBMModel(
            params=params,
            categorical_features=cat_features,
            early_stopping_rounds=early_stopping
        )
    
    elif model_name == 'two_stage':
        linear_alpha = kwargs.get('linear_alpha', config['models']['linear']['alpha'])
        lgb_params = kwargs.get('lgb_params', config['models']['lightgbm']['params'])
        cat_features = kwargs.get('categorical_features', ['pid'])
        early_stopping = kwargs.get('early_stopping_rounds',
                                    config['models']['lightgbm']['early_stopping_rounds'])
        return TwoStageModel(
            linear_alpha=linear_alpha,
            lgb_params=lgb_params,
            categorical_features=cat_features,
            early_stopping_rounds=early_stopping,
            config=config
        )
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_and_evaluate(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_val: pd.DataFrame,
    y_val: Union[pd.Series, np.ndarray],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a model and evaluate on validation set.
    
    Args:
        model: Model instance
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        verbose: Whether to print results
        
    Returns:
        Dictionary with results
    """
    from .validation import compute_metrics
    
    # Train
    if hasattr(model, 'fit') and 'X_val' in model.fit.__code__.co_varnames:
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train)
    
    # Predict
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Compute metrics
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(y_val, pd.Series):
        y_val = y_val.values
    
    train_metrics = compute_metrics(y_train, train_preds)
    val_metrics = compute_metrics(y_val, val_preds)
    
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_preds': train_preds,
        'val_preds': val_preds,
        'model': model
    }
    
    if verbose:
        print(f"Train RMSE: {train_metrics['rmse']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"Train R²: {train_metrics['r2']:.4f}, Val R²: {val_metrics['r2']:.4f}")
    
    return results