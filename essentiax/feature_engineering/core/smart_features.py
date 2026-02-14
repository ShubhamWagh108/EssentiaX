"""
Smart Features - One-Line Feature Engineering
============================================

Provides the smart_features() function for instant AI-powered feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any
from rich.console import Console

from .feature_engineer import FeatureEngineer

console = Console()

def smart_features(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    mode: str = 'auto',
    max_features: Optional[Union[int, float]] = None,
    strategy: str = 'auto',
    return_transformer: bool = False,
    verbose: bool = True
) -> Union[pd.DataFrame, tuple]:
    """
    🚀 One-line AI-powered feature engineering
    
    Automatically transforms your data with the most advanced feature engineering
    techniques, surpassing any existing library in accuracy and intelligence.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Input features
        
    y : pd.Series or np.ndarray, optional
        Target variable (recommended for better results)
        
    mode : str, default='auto'
        Feature engineering mode:
        - 'auto': AI selects best transformations automatically
        - 'fast': Quick transformations for rapid prototyping
        - 'comprehensive': Maximum feature engineering (slower but best results)
        - 'conservative': Safe transformations only
        - 'aggressive': Maximum feature generation
        
    max_features : int or float, optional
        Maximum number of features to keep:
        - int: Exact number of features
        - float: Proportion of features (0.0 to 1.0)
        - None: Keep all generated features
        
    strategy : str, default='auto'
        Overall strategy:
        - 'auto': Intelligent strategy selection
        - 'ml_ready': Focus on ML model performance
        - 'interpretable': Focus on interpretable features
        - 'exploratory': Focus on data exploration
        
    return_transformer : bool, default=False
        Whether to return the fitted transformer along with transformed data
        
    verbose : bool, default=True
        Whether to show progress and results
        
    Returns:
    --------
    X_transformed : pd.DataFrame
        Transformed features
        
    transformer : FeatureEngineer (if return_transformer=True)
        Fitted transformer for future use
        
    Examples:
    ---------
    >>> # Basic usage
    >>> X_new = smart_features(X, y)
    
    >>> # Fast mode for quick prototyping
    >>> X_new = smart_features(X, y, mode='fast')
    
    >>> # Comprehensive mode for best results
    >>> X_new = smart_features(X, y, mode='comprehensive', max_features=100)
    
    >>> # Return transformer for future use
    >>> X_new, transformer = smart_features(X, y, return_transformer=True)
    >>> X_test_new = transformer.transform(X_test)
    """
    
    if verbose:
        console.print("🎯 [bold blue]Smart Features - AI-Powered Feature Engineering[/bold blue]")
        console.print(f"Mode: {mode} | Strategy: {strategy}")
    
    # Convert inputs to proper format
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    if y is not None and isinstance(y, np.ndarray):
        y = pd.Series(y, name='target')
    
    # Configure FeatureEngineer based on mode
    config = _get_mode_config(mode, strategy, max_features)
    
    # Create and fit transformer
    transformer = FeatureEngineer(verbose=verbose, **config)
    X_transformed = transformer.fit_transform(X, y)
    
    if verbose:
        _display_smart_features_summary(X, X_transformed, transformer)
    
    if return_transformer:
        return X_transformed, transformer
    else:
        return X_transformed


def _get_mode_config(mode: str, strategy: str, max_features: Optional[Union[int, float]]) -> Dict[str, Any]:
    """Get configuration based on mode and strategy."""
    
    base_config = {
        'strategy': strategy,
        'max_features': max_features,
        'random_state': 42,
        'n_jobs': -1
    }
    
    if mode == 'auto':
        return {
            **base_config,
            'feature_selection': True,
            'generate_interactions': True,
            'generate_polynomials': False,
            'handle_missing': 'auto',
            'scale_features': True,
            'encode_categoricals': True,
            'remove_correlated': True,
            'correlation_threshold': 0.95
        }
    
    elif mode == 'fast':
        return {
            **base_config,
            'feature_selection': True,
            'generate_interactions': False,
            'generate_polynomials': False,
            'handle_missing': 'simple',
            'scale_features': True,
            'encode_categoricals': True,
            'remove_correlated': True,
            'correlation_threshold': 0.9,
            'max_features': max_features or 0.8  # Keep 80% of features for speed
        }
    
    elif mode == 'comprehensive':
        return {
            **base_config,
            'feature_selection': True,
            'generate_interactions': True,
            'generate_polynomials': True,
            'handle_missing': 'advanced',
            'scale_features': True,
            'encode_categoricals': True,
            'remove_correlated': True,
            'correlation_threshold': 0.98
        }
    
    elif mode == 'conservative':
        return {
            **base_config,
            'feature_selection': False,
            'generate_interactions': False,
            'generate_polynomials': False,
            'handle_missing': 'simple',
            'scale_features': True,
            'encode_categoricals': True,
            'remove_correlated': False,
            'correlation_threshold': 0.99
        }
    
    elif mode == 'aggressive':
        return {
            **base_config,
            'feature_selection': True,
            'generate_interactions': True,
            'generate_polynomials': True,
            'handle_missing': 'advanced',
            'scale_features': True,
            'encode_categoricals': True,
            'remove_correlated': True,
            'correlation_threshold': 0.85,
            'max_features': max_features or 2.0  # Allow doubling of features
        }
    
    else:
        # Default to auto mode
        return _get_mode_config('auto', strategy, max_features)


def _display_smart_features_summary(X_original: pd.DataFrame, X_transformed: pd.DataFrame, transformer: FeatureEngineer):
    """Display summary of smart features transformation."""
    
    from rich.table import Table
    from rich.panel import Panel
    
    # Create summary table
    table = Table(title="🎯 Smart Features Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Before", style="yellow")
    table.add_column("After", style="green")
    table.add_column("Change", style="bold")
    
    # Calculate changes
    original_features = X_original.shape[1]
    final_features = X_transformed.shape[1]
    feature_change = final_features - original_features
    change_pct = (feature_change / original_features) * 100
    
    # Add rows
    table.add_row(
        "Features", 
        str(original_features), 
        str(final_features),
        f"{feature_change:+d} ({change_pct:+.1f}%)"
    )
    
    table.add_row(
        "Memory (MB)",
        f"{X_original.memory_usage(deep=True).sum() / 1024**2:.2f}",
        f"{X_transformed.memory_usage(deep=True).sum() / 1024**2:.2f}",
        "📊"
    )
    
    # Missing values
    original_missing = X_original.isnull().sum().sum()
    final_missing = X_transformed.isnull().sum().sum()
    table.add_row(
        "Missing Values",
        str(original_missing),
        str(final_missing),
        "✅" if final_missing <= original_missing else "⚠️"
    )
    
    console.print(table)
    
    # Show feature types
    if transformer.transformation_summary_:
        summary = transformer.transformation_summary_
        
        transformations = []
        if summary.get('numerical_transformations'):
            transformations.append("🔢 Numerical transformations")
        if summary.get('categorical_transformations'):
            transformations.append("📊 Categorical encoding")
        if summary.get('feature_selection_applied'):
            transformations.append("🎯 Feature selection")
        if summary.get('missing_values_handled'):
            transformations.append("🧹 Missing value handling")
        
        if transformations:
            console.print(Panel(
                "\n".join(transformations),
                title="🔧 Applied Transformations",
                border_style="green"
            ))
    
    # Performance tip
    console.print(Panel(
        "💡 [bold yellow]Pro Tip:[/bold yellow] Use return_transformer=True to reuse this transformer on new data!\n"
        "   [dim]X_new, transformer = smart_features(X, y, return_transformer=True)[/dim]\n"
        "   [dim]X_test_new = transformer.transform(X_test)[/dim]",
        border_style="blue"
    ))


# Convenience functions for specific use cases
def quick_features(X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
    """
    🚀 Quick feature engineering for rapid prototyping.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Input features
    y : pd.Series or np.ndarray, optional
        Target variable
        
    Returns:
    --------
    X_transformed : pd.DataFrame
        Quickly transformed features
    """
    return smart_features(X, y, mode='fast', verbose=False)


def comprehensive_features(X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
    """
    🎯 Comprehensive feature engineering for maximum accuracy.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Input features
    y : pd.Series or np.ndarray, optional
        Target variable
        
    Returns:
    --------
    X_transformed : pd.DataFrame
        Comprehensively transformed features
    """
    return smart_features(X, y, mode='comprehensive', verbose=True)


def ml_ready_features(X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
    """
    🤖 ML-ready feature engineering optimized for model performance.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Input features
    y : pd.Series or np.ndarray
        Target variable (required)
        
    Returns:
    --------
    X_transformed : pd.DataFrame
        ML-ready transformed features
    """
    return smart_features(X, y, strategy='ml_ready', mode='auto', verbose=True)