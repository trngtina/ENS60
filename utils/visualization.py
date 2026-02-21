"""
Visualization Utilities for ENS Challenge 60
Plots for feature importance, residuals, and predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Union
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from .data_loader import load_config


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    figsize: tuple = (10, 8),
    title: str = "Feature Importance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get top N features
    top_features = importance_df.nlargest(top_n, 'importance')
    
    # Plot
    sns.barplot(
        data=top_features,
        y='feature',
        x='importance',
        ax=ax,
        palette='viridis'
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: tuple = (14, 5),
    title: str = "Residual Analysis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot residual analysis.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('Residuals', fontsize=11)
    axes[0].set_title('Residuals vs Predicted', fontsize=12)
    
    # 2. Residual Distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Residual Distribution', fontsize=12)
    
    # Add statistics
    stats_text = f'Mean: {residuals.mean():.4f}\nStd: {residuals.std():.4f}'
    axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot', fontsize=12)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: tuple = (10, 8),
    title: str = "Predictions vs Actual",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Compute metrics
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    
    # Add metrics text
    metrics_text = f'RMSE: {rmse:.4f}\nR²: {r2:.4f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_cv_results(
    cv_results: Dict[str, Any],
    figsize: tuple = (12, 5),
    title: str = "Cross-Validation Results",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cross-validation results.
    
    Args:
        cv_results: Dictionary from cross_validate function
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fold_df = cv_results['fold_metrics']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. RMSE per fold
    axes[0].bar(fold_df['fold'], fold_df['rmse'], color='steelblue', edgecolor='black')
    axes[0].axhline(y=cv_results['mean_rmse'], color='r', linestyle='--', 
                    linewidth=2, label=f'Mean: {cv_results["mean_rmse"]:.4f}')
    axes[0].fill_between(
        [0.5, len(fold_df) + 0.5],
        cv_results['mean_rmse'] - cv_results['std_rmse'],
        cv_results['mean_rmse'] + cv_results['std_rmse'],
        alpha=0.2, color='red', label=f'±1 Std: {cv_results["std_rmse"]:.4f}'
    )
    axes[0].set_xlabel('Fold', fontsize=11)
    axes[0].set_ylabel('RMSE', fontsize=11)
    axes[0].set_title('RMSE per Fold', fontsize=12)
    axes[0].legend()
    
    # 2. R² per fold
    axes[1].bar(fold_df['fold'], fold_df['r2'], color='forestgreen', edgecolor='black')
    axes[1].axhline(y=cv_results['mean_r2'], color='r', linestyle='--',
                    linewidth=2, label=f'Mean: {cv_results["mean_r2"]:.4f}')
    axes[1].set_xlabel('Fold', fontsize=11)
    axes[1].set_ylabel('R²', fontsize=11)
    axes[1].set_title('R² per Fold', fontsize=12)
    axes[1].legend()
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_per_stock_metrics(
    stock_metrics: pd.DataFrame,
    metric: str = 'rmse',
    top_n: int = 20,
    figsize: tuple = (12, 6),
    title: str = "Per-Stock Performance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot per-stock metrics.
    
    Args:
        stock_metrics: DataFrame with per-stock metrics
        metric: Metric to plot
        top_n: Number of worst stocks to show
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Distribution of metric across stocks
    axes[0].hist(stock_metrics[metric], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=stock_metrics[metric].mean(), color='r', linestyle='--',
                    linewidth=2, label=f'Mean: {stock_metrics[metric].mean():.4f}')
    axes[0].set_xlabel(metric.upper(), fontsize=11)
    axes[0].set_ylabel('Number of Stocks', fontsize=11)
    axes[0].set_title(f'{metric.upper()} Distribution', fontsize=12)
    axes[0].legend()
    
    # 2. Worst performing stocks
    worst_stocks = stock_metrics.nlargest(top_n, metric)
    axes[1].barh(range(len(worst_stocks)), worst_stocks[metric].values, color='coral')
    axes[1].set_yticks(range(len(worst_stocks)))
    axes[1].set_yticklabels([f"Stock {int(pid)}" for pid in worst_stocks['pid'].values])
    axes[1].set_xlabel(metric.upper(), fontsize=11)
    axes[1].set_title(f'Top {top_n} Worst Stocks', fontsize=12)
    axes[1].invert_yaxis()
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_temporal_performance(
    day_metrics: pd.DataFrame,
    metric: str = 'rmse',
    figsize: tuple = (14, 5),
    title: str = "Temporal Performance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot performance over time.
    
    Args:
        day_metrics: DataFrame with per-day metrics
        metric: Metric to plot
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot metric over time
    ax.plot(day_metrics['day'], day_metrics[metric], linewidth=1, alpha=0.7)
    
    # Add rolling average
    window = min(20, len(day_metrics) // 5)
    if window > 1:
        rolling_avg = day_metrics[metric].rolling(window=window, center=True).mean()
        ax.plot(day_metrics['day'], rolling_avg, linewidth=2, color='red',
                label=f'{window}-day Rolling Average')
    
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_target_distribution(
    y_train: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    figsize: tuple = (10, 5),
    title: str = "Target Distribution",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot target distribution.
    
    Args:
        y_train: Training target values
        y_pred: Predicted values (optional)
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training distribution
    ax.hist(y_train, bins=50, alpha=0.7, label='Actual', edgecolor='black')
    
    # Plot prediction distribution if provided
    if y_pred is not None:
        ax.hist(y_pred, bins=50, alpha=0.5, label='Predicted', edgecolor='black')
    
    ax.set_xlabel('Target Value (log auction volume fraction)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    # Add statistics
    stats_text = f'Mean: {y_train.mean():.4f}\nStd: {y_train.std():.4f}\nMin: {y_train.min():.4f}\nMax: {y_train.max():.4f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: tuple = (12, 10),
    title: str = "Feature Correlation Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot correlation matrix.
    
    Args:
        df: DataFrame with features
        columns: Columns to include (if None, uses all numeric)
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Limit to reasonable number of features
    if len(columns) > 30:
        columns = columns[:30]
    
    corr_matrix = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_learning_curve(
    train_sizes: List[float],
    train_scores: List[float],
    val_scores: List[float],
    figsize: tuple = (10, 6),
    title: str = "Learning Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot learning curve.
    
    Args:
        train_sizes: Training set sizes (fractions)
        train_scores: Training RMSE scores
        val_scores: Validation RMSE scores
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(train_sizes, train_scores, 'o-', label='Training RMSE', linewidth=2, markersize=8)
    ax.plot(train_sizes, val_scores, 'o-', label='Validation RMSE', linewidth=2, markersize=8)
    
    ax.set_xlabel('Training Set Size (fraction)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_summary_dashboard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cv_results: Optional[Dict[str, Any]] = None,
    feature_importance: Optional[pd.DataFrame] = None,
    figsize: tuple = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a summary dashboard with multiple plots.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        cv_results: Cross-validation results (optional)
        feature_importance: Feature importance DataFrame (optional)
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    n_rows = 2
    n_cols = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # 1. Predictions vs Actual
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ax.set_title(f'Predictions vs Actual (RMSE: {rmse:.4f})', fontsize=12)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    
    # 2. Residual Distribution
    ax = axes[0, 1]
    residuals = y_true - y_pred
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_title('Residual Distribution', fontsize=12)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    
    # 3. CV Results or Target Distribution
    ax = axes[1, 0]
    if cv_results is not None:
        fold_df = cv_results['fold_metrics']
        ax.bar(fold_df['fold'], fold_df['rmse'], color='steelblue', edgecolor='black')
        ax.axhline(y=cv_results['mean_rmse'], color='r', linestyle='--', linewidth=2)
        ax.set_title(f'CV RMSE (Mean: {cv_results["mean_rmse"]:.4f})', fontsize=12)
        ax.set_xlabel('Fold')
        ax.set_ylabel('RMSE')
    else:
        ax.hist(y_true, bins=50, alpha=0.7, label='Actual', edgecolor='black')
        ax.hist(y_pred, bins=50, alpha=0.5, label='Predicted', edgecolor='black')
        ax.set_title('Target Distribution', fontsize=12)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    # 4. Feature Importance or Residuals vs Predicted
    ax = axes[1, 1]
    if feature_importance is not None:
        top_features = feature_importance.nlargest(15, 'importance')
        ax.barh(range(len(top_features)), top_features['importance'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_title('Top 15 Features', fontsize=12)
        ax.set_xlabel('Importance')
        ax.invert_yaxis()
    else:
        ax.scatter(y_pred, residuals, alpha=0.3, s=10)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_title('Residuals vs Predicted', fontsize=12)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
    
    plt.suptitle('Model Performance Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig